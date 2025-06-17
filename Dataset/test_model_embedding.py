"""
EEG Data Embedding Generation Module
This module provides functionality to generate embeddings from EEG data using a pre-trained Roformer model.
Includes data preprocessing, binning, and parallel processing capabilities.
"""

import torch
import time
import numpy as np
from joblib import Parallel, delayed
from transformers import RoformerModel
from pytorch_lightning import LightningModule

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# GLOBAL CONSTANTS
# ============================================================================
EFFECTIVE_VOCAB_SIZE = 2000  # Target vocabulary size for tokenization

# ============================================================================
# DATA PREPROCESSING FUNCTIONS
# ============================================================================
def bin_data(data, effective_vocab_size=EFFECTIVE_VOCAB_SIZE):
    """
    Bin and discretize multi-channel EEG data (vectorized version).
    
    This function converts continuous EEG signals into discrete tokens by:
    1. Normalizing each channel to [0,1] range
    2. Scaling to vocabulary size range
    3. Rounding to integer token IDs
    
    Args:
        data (numpy.ndarray): EEG data with shape (channels, time_points)
                             Expected format: fp16 numpy array
        effective_vocab_size (int): Target vocabulary size for discretization
    
    Returns:
        numpy.ndarray: Discretized token ID array with same shape as input
                      Values range from 0 to (effective_vocab_size-1)
    """
    # Calculate min and max for each channel, keeping dimensions for broadcasting
    mins = data.min(axis=1, keepdims=True)
    maxs = data.max(axis=1, keepdims=True)
    
    # Normalize to [0,1] range (add epsilon to avoid division by zero)
    epsilon = np.finfo(data.dtype).eps  # Get smallest positive value for data type
    normalized = (data - mins) / (maxs - mins + epsilon)
    
    # Scale to [0, effective_vocab_size-1] range and convert to integers
    scaled = normalized * (effective_vocab_size - 1)
    input_ids = np.round(scaled).astype(np.int32)
    
    return input_ids

# ============================================================================
# MODEL WRAPPER CLASS
# ============================================================================
class RoformerModelWrapper(LightningModule):
    """
    PyTorch Lightning wrapper for Roformer model with custom token handling.
    
    This wrapper adds special CLS and SEP tokens to input sequences and
    provides checkpoint loading functionality for pre-trained models.
    """
    
    def __init__(self):
        """Initialize the Roformer model wrapper"""
        super().__init__()
        self.model = RoformerModel.from_pretrained('roformer-base')

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        """
        Load model from PyTorch Lightning checkpoint.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
            **kwargs: Additional arguments for model initialization
            
        Returns:
            RoformerModelWrapper: Loaded model instance
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = cls(**kwargs)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        return model
    
    def forward(self, x, **kwargs):
        """
        Forward pass with automatic CLS and SEP token addition.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length)
            **kwargs: Additional arguments passed to the underlying model
            
        Returns:
            Model output with added special tokens
        """
        # Add CLS token (ID: 2003) at the beginning of each sequence
        cls_token = torch.tensor([[2003]]).to(x.device).expand(x.size(0), -1)
        
        # Add SEP token (ID: 2004) at the end of each sequence
        sep_token = torch.tensor([[2004]]).to(x.device).expand(x.size(0), -1)
        
        # Concatenate: [CLS] + sequence[:-2] + [SEP]
        # Remove last 2 tokens to make room for CLS and SEP
        x = torch.cat([cls_token, x[:, :-2], sep_token], dim=1)
        
        return self.model(x, **kwargs)

# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================
def process_single_sample(model, input_ids):
    """
    Process inference for a single sample with mixed precision.
    
    Args:
        model: The model to use for inference
        input_ids (torch.Tensor): Input token IDs
        
    Returns:
        Model output with hidden states
    """
    with torch.no_grad(), torch.cuda.amp.autocast():
        # Enable hidden states output for embedding extraction
        output = model(input_ids, output_hidden_states=True)
    return output

def process_samples_parallel(model, input_samples, n_jobs=4):
    """
    Process multiple samples in parallel for faster inference.
    
    Args:
        model: The model to use for inference
        input_samples (list): List of input samples to process
        n_jobs (int): Number of parallel jobs to run
        
    Returns:
        list: List of model outputs for all input samples
    """
    return Parallel(n_jobs=n_jobs)(
        delayed(process_single_sample)(model, input_sample)
        for input_sample in input_samples
    )

def get_cls_embedding(model, input_data):
    """
    Generate CLS token embeddings using the model for multi-channel EEG data.
    
    This function processes EEG data through the model and extracts the CLS token
    embedding from the final hidden layer, which serves as a representation
    of the entire sequence.
    
    Args:
        model (RoformerModelWrapper): Loaded model instance
        input_data (torch.Tensor or numpy.ndarray): Input EEG data with shape 
                                                   (channels, sequence_length)
    
    Returns:
        torch.Tensor: CLS token embeddings for each channel with shape 
                     (channels, hidden_size)
    """
    # Ensure input data is a tensor and on the correct device
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.tensor(input_data)
    input_data = input_data.to(device)
    
    # Process all samples with mixed precision for efficiency
    with torch.no_grad(), torch.cuda.amp.autocast():
        output = model(input_data, output_hidden_states=True)
    
    # Extract CLS token embeddings from the final hidden layer
    # Shape: (channels, hidden_size) - first token of each sequence
    cls_embeddings = output.hidden_states[-1][:, 0, :]
    
    return cls_embeddings

# ============================================================================
# MAIN TESTING SCRIPT
# ============================================================================
if __name__ == "__main__":
    
    # ===== MODEL LOADING =====
    checkpoint_path = ""  # Path to model checkpoint
    model = RoformerModelWrapper.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    print("Model loaded successfully and moved to GPU")

    # ===== TEST DATA GENERATION =====
    # Generate random test data with shape (23 channels, 1000 time points)
    # Token IDs should be within vocabulary range
    input_data = torch.randint(0, 1433, (23, 1000)).to(device)
    print(f"Test input data shape: {input_data.shape}")

    # ===== PERFORMANCE TESTING =====
    print("\nRunning embedding generation performance test...")
    
    for i in range(10):
        start_time = time.time()
        
        # Generate CLS token embeddings
        cls_embedding = get_cls_embedding(model, input_data)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Log results
        print(f"Iteration {i+1}/10:")
        print(f"  CLS Embedding Shape: {cls_embedding.shape}")
        print(f"  Processing Time: {processing_time:.2f} seconds")
        
        # Optional: Print first few values for debugging
        # print(f"  Sample values: {cls_embedding[0, :5]}")

    print("\nPerformance testing completed!")

