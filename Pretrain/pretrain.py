"""
EEG Data Pretraining Script
This script implements masked language modeling pretraining for EEG signals using RoFormer architecture.
The approach treats discretized EEG data as sequences of tokens for self-supervised learning.
"""

import os
import torch
import json
import numpy as np
from transformers import (
    PreTrainedTokenizerFast, 
    AutoTokenizer, 
    DataCollatorForLanguageModeling,
    RoFormerForMaskedLM, 
    RoFormerConfig
)
from streaming import StreamingDataset
from torch.utils.data import DataLoader
from composer.models.huggingface import HuggingFaceModel
from composer.metrics import MaskedAccuracy, LanguageCrossEntropy
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import LinearWithWarmupScheduler
from composer import Trainer
from composer.algorithms import GradientClipping
from composer.utils import reproducibility
from clearml_logger import ClearMLLogger

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================
# Set GPU visibility and ClearML configuration
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Specify GPU devices
os.environ['CLEARML_WEB_HOST'] = ""
os.environ['CLEARML_API_HOST'] = ""
os.environ['CLEARML_FILES_HOST'] = ""
os.environ['CLEARML_API_ACCESS_KEY'] = ""
os.environ['CLEARML_API_SECRET_KEY'] = ""

# ============================================================================
# GLOBAL CONSTANTS
# ============================================================================
EFFECTIVE_VOCAB_SIZE = 2000  # Vocabulary size for EEG token discretization
SEED = 42                    # Random seed for reproducibility

# ============================================================================
# REPRODUCIBILITY SETUP
# ============================================================================
# Configure deterministic behavior for reproducible training
reproducibility.configure_deterministic_mode()
reproducibility.seed_all(SEED)

# ============================================================================
# DATA PREPROCESSING FUNCTIONS
# ============================================================================
def bin_data(data, effective_vocab_size=EFFECTIVE_VOCAB_SIZE):
    """
    Discretize EEG data into tokens using binning strategy.
    
    Converts continuous EEG signals into discrete token IDs by:
    1. Normalizing data to [0,1] range
    2. Scaling to vocabulary size range  
    3. Rounding to integer token IDs
    
    Args:
        data (numpy.ndarray): EEG data array of length 2000 with fp16 format
        effective_vocab_size (int): Target vocabulary size for discretization
    
    Returns:
        numpy.ndarray: Discretized token ID array with same length as input
    """
    # Validate input data format and dimensions
    assert isinstance(data, np.ndarray), "Input must be numpy array"
    assert data.dtype == np.float16, "Input must be float16 format"
    assert len(data) == 2000, "Input must have length 2000"
    
    # Normalize to [0,1] range using min-max scaling
    min_val = data.min()
    max_val = data.max()
    normalized = (data - min_val) / (max_val - min_val)
    
    # Scale to vocabulary range and convert to integer tokens
    scaled = normalized * (effective_vocab_size - 1)
    input_ids = np.round(scaled).astype(np.int32)
    
    return input_ids

# ============================================================================
# DATASET CLASS
# ============================================================================
class EEGData(StreamingDataset):
    """
    Streaming dataset class for EEG data with automatic tokenization.
    
    Extends StreamingDataset to provide on-the-fly EEG data discretization
    for masked language modeling pretraining.
    """
    
    def __init__(self, local_path, batch_size, shuffle, split=None, 
                 allow_unsafe_types=True, batching_method="random"):
        """
        Initialize EEG streaming dataset.
        
        Args:
            local_path (str): Path to streaming dataset
            batch_size (int): Batch size for data loading
            shuffle (bool): Whether to shuffle the data
            split (str): Dataset split identifier
            allow_unsafe_types (bool): Allow unsafe data types in streaming
            batching_method (str): Method for batching data
        """
        super().__init__(
            local=local_path, 
            shuffle=shuffle, 
            split=split,
            allow_unsafe_types=allow_unsafe_types,
            batching_method=batching_method, 
            batch_size=batch_size
        )

    def __getitem__(self, index):
        """
        Get single sample with automatic EEG tokenization.
        
        Args:
            index (int): Sample index
            
        Returns:
            dict: Sample with tokenized EEG data as 'input_ids'
        """
        # Get raw sample from parent class
        obj = super().__getitem__(index)
        
        # Convert original EEG data to tokens and store as input_ids
        obj["input_ids"] = np.array(bin_data(obj.pop("original_data"))).astype(np.int16)
        
        return obj

# ============================================================================
# DATA LOADING SETUP
# ============================================================================
# Initialize streaming dataset
local_path = ""
eegds = EEGData(local_path=local_path, batch_size=4, shuffle=True)

# ============================================================================
# MASKED LANGUAGE MODELING SETUP
# ============================================================================
def collate_fn_factory(tokenizer, mlm_probability=0.15):
    """
    Factory function to create collate function for masked language modeling.
    
    Args:
        tokenizer: Tokenizer for handling special tokens
        mlm_probability (float): Probability of masking tokens for MLM
        
    Returns:
        function: Collate function for DataLoader
    """
    # Initialize data collator for masked language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,  # Enable masked language modeling
        mlm_probability=mlm_probability,
    )
    
    def masking_collate_fn(batch):
        """
        Collate function that applies masking to batch of EEG sequences.
        
        Args:
            batch (list): List of samples with input_ids
            
        Returns:
            dict: Batch with masked input_ids and labels
        """
        # Concatenate input_ids from all samples in batch
        input_ids = np.concatenate([b["input_ids"][None] for b in batch])
        
        # Apply masking using data collator
        mask = data_collator(input_ids)
        input_ids_all = mask["input_ids"]
        labels_all = mask["labels"]
        
        return {"input_ids": input_ids_all, "labels": labels_all}
    
    return masking_collate_fn

# ============================================================================
# TOKENIZER AND DATA LOADER SETUP
# ============================================================================
# Load pre-trained tokenizer for EEG data
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "/root/project/EEGPT/pretrain/model_archive/rank_tokenizer"
)

# Create collate function with 15% masking probability
collate_fn = collate_fn_factory(tokenizer, mlm_probability=0.15)

# Initialize data loader for training
train_loader = DataLoader(
    eegds,
    batch_size=4,
    collate_fn=collate_fn,
    shuffle=False,  # Streaming dataset handles shuffling
    drop_last=False
)

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================
# Load model configuration and initialize RoFormer for masked LM
config = RoFormerConfig.from_pretrained(
    "/root/project/EEGPT/pretrain/model_archive/rank_model_Feb_2_12"
)
model = RoFormerForMaskedLM(config=config)

# ============================================================================
# TRAINING SETUP
# ============================================================================
# Define evaluation metrics for pretraining
metrics = [
    LanguageCrossEntropy(ignore_index=-100),  # Cross-entropy loss for MLM
    MaskedAccuracy(ignore_index=-100),        # Accuracy on masked tokens
]

# Wrap model with Composer HuggingFace interface
composer_model = HuggingFaceModel(
    model,
    tokenizer=tokenizer,
    metrics=metrics,
    use_logits=True,
)

# Configure optimizer with decoupled weight decay
optimizer = DecoupledAdamW(
    model.parameters(),
    lr=5.0e-4,           # Learning rate
    betas=(0.9, 0.98),   # Adam beta parameters
    eps=1.0e-06,         # Epsilon for numerical stability
    weight_decay=1.0e-5  # Weight decay for regularization
)

# Configure learning rate scheduler with warmup
scheduler = LinearWithWarmupScheduler(
    t_warmup='0.01dur',  # Warmup for 1% of training
    alpha_f=0.02         # Final learning rate multiplier
)

# ============================================================================
# LOGGING AND MONITORING SETUP
# ============================================================================
# Initialize ClearML logger for experiment tracking
clogger = ClearMLLogger(
    project_name='test', 
    task_name='pretrain_bin_model', 
    log_interval=50
)

# Set run name for experiment identification
run_name = 'my_autoresume_training_run'

# ============================================================================
# TRAINER CONFIGURATION AND EXECUTION
# ============================================================================
# Change to output directory
os.chdir("/root/project/EEGPT/pretrain/model_result")

# Configure gradient clipping to prevent gradient explosion
gc = GradientClipping(clipping_type='norm', clipping_threshold=2.0)

# Initialize Composer trainer with all configurations
trainer = Trainer(
    model=composer_model,
    train_dataloader=train_loader,
    max_duration='100ep',              # Train for 100 epochs
    optimizers=optimizer,
    schedulers=[scheduler],
    device='gpu',                      # Use GPU for training
    precision='amp_fp16',              # Mixed precision training
    eval_interval="1ep",               # Evaluate every epoch
    seed=SEED,
    
    # Checkpoint configuration
    save_num_checkpoints_to_keep=50,   # Keep last 50 checkpoints
    save_folder='./hf_tiny_output',    # Checkpoint save directory
    save_filename='composer-hf-ba{batch}.pt',  # Checkpoint filename pattern
    save_latest_filename="latest",     # Latest checkpoint name
    save_overwrite=True,               # Overwrite existing checkpoints
    save_interval="1000ba",            # Save every 1000 batches
    
    # Training configuration
    auto_log_hparams=True,             # Automatically log hyperparameters
    algorithms=[gc],                   # Apply gradient clipping
    loggers=[clogger],                 # Use ClearML logger
    autoresume=True,                   # Enable automatic resuming
    run_name=run_name
)

# ============================================================================
# START TRAINING
# ============================================================================
print("Starting EEG pretraining with masked language modeling...")
trainer.fit()
print("Training completed successfully!")