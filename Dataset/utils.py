"""
EEG Dataset Utilities Module
This module provides dataset classes and utility functions for loading and processing
EEG data from various sources including streaming datasets and LMDB databases.
"""

import torch
import numpy as np
import pickle
import lmdb
import random
from torch.utils.data import Dataset, DataLoader, Subset
from streaming import StreamingDataset
from transformers import AutoTokenizer, AutoModel

# ============================================================================
# GLOBAL CONSTANTS
# ============================================================================
BATCH_SIZE = 2  # Default batch size for data loading

# ============================================================================
# STREAMING DATASET CLASSES
# ============================================================================
class EEGData(StreamingDataset):
    """
    Streaming dataset class for EEG data with preprocessing and augmentation.
    
    This class extends StreamingDataset to provide EEG-specific preprocessing
    including standardization and optional noise augmentation.
    """
    
    def __init__(self, local_path, batch_size, shuffle=False, split=None,
                 allow_unsafe_types=True, batching_method="random", 
                 add_noise=False):
        """
        Initialize EEG streaming dataset.
        
        Args:
            local_path (str): Path to the local dataset
            batch_size (int): Batch size for data loading
            shuffle (bool): Whether to shuffle the data
            split (str): Dataset split identifier
            allow_unsafe_types (bool): Allow unsafe data types in streaming
            batching_method (str): Method for batching data
            add_noise (bool): Whether to add augmentation noise
        """
        super().__init__(local=local_path, shuffle=shuffle, split=split,
                         allow_unsafe_types=allow_unsafe_types,
                         batching_method=batching_method, batch_size=batch_size)
        self.add_noise = add_noise

    def __getitem__(self, index):
        """
        Get a single sample with preprocessing and optional augmentation.
        
        Args:
            index (int): Sample index
            
        Returns:
            dict: Processed sample with channel embeddings and labels
        """
        # Get raw sample from parent class
        obj = super().__getitem__(index)
        embeddings = np.copy(obj["channel_embeddings"]).astype(np.float32)
        
        # Channel-wise standardization (zero mean, unit variance)
        embeddings = (embeddings - embeddings.mean(axis=1, keepdims=True)) / \
                    (embeddings.std(axis=1, keepdims=True) + 1e-8)

        # Optional noise augmentation for data diversity
        if self.add_noise and np.random.rand() < 0.5:
            data_std = np.std(embeddings)
            noise = np.random.normal(0, data_std * 0.1, embeddings.shape)
            # Clip to prevent extreme values
            embeddings = np.clip(embeddings + noise, 
                               a_min=np.percentile(embeddings, 0.1),
                               a_max=np.percentile(embeddings, 99.9))

        # Convert to PyTorch tensors
        embeddings = torch.from_numpy(embeddings)
        return {
            "channel_embeddings": embeddings,  
            "labels": torch.tensor(obj["labels"]).to(torch.long)
        }

# ============================================================================
# DATA LOADER FUNCTIONS
# ============================================================================
def get_dl(train_path="", val_path="", batch_size=BATCH_SIZE):
    """
    Create data loaders for training and validation with embeddings.
    
    Args:
        train_path (str): Path to training dataset
        val_path (str): Path to validation dataset
        batch_size (int): Batch size for data loading
        
    Returns:
        tuple: (train_loader, val_loader) DataLoader instances
    """
    # Create datasets
    train_ds = EEGData(local_path=train_path, batch_size=BATCH_SIZE)
    val_ds = EEGData(local_path=val_path, batch_size=BATCH_SIZE)

    def collate_fn(batch):
        """Collate function to batch samples together"""
        inputs = torch.stack([torch.as_tensor(item["channel_embeddings"].copy()) for item in batch])
        labels = torch.tensor([item["labels"] for item in batch])
        return {"channel_embeddings": inputs, "labels": labels}

    # Create data loaders with multi-processing
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=16,  # Parallel data loading
        drop_last=True,  # Drop incomplete batches
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=16,
        drop_last=True,
        collate_fn=collate_fn
    )
    return train_loader, val_loader

def get_ds(train_path="", val_path="", batch_size=BATCH_SIZE):
    """
    Create raw datasets without data loaders.
    
    Args:
        train_path (str): Path to training dataset
        val_path (str): Path to validation dataset
        batch_size (int): Batch size for dataset configuration
        
    Returns:
        tuple: (train_ds, val_ds) Dataset instances
    """
    train_ds = EEGData(local_path=train_path, batch_size=BATCH_SIZE)
    val_ds = EEGData(local_path=val_path, batch_size=BATCH_SIZE)
    return train_ds, val_ds

# ============================================================================
# ORIGINAL DATA FUNCTIONS (WITHOUT EMBEDDINGS)
# ============================================================================
def get_original_tuab_ds(train_path="", val_path="", batch_size=BATCH_SIZE):
    """
    Create datasets for original TUAB data without embeddings.
    
    Args:
        train_path (str): Path to training dataset
        val_path (str): Path to validation dataset
        batch_size (int): Batch size for dataset configuration
        
    Returns:
        tuple: (train_ds, val_ds) StreamingDataset instances
    """
    train_ds = StreamingDataset(local=train_path, batch_size=BATCH_SIZE)
    val_ds = StreamingDataset(local=val_path, batch_size=BATCH_SIZE)
    return train_ds, val_ds

def get_original_tuab_dl(train_path="", val_path="", batch_size=BATCH_SIZE):
    """
    Create data loader iterators for original TUAB data.
    
    Args:
        train_path (str): Path to training dataset
        val_path (str): Path to validation dataset
        batch_size (int): Batch size for data loading
        
    Returns:
        tuple: (train_iter, val_iter) DataLoader iterators
    """
    train_ds, val_ds = get_original_tuab_ds(train_path, val_path, batch_size)

    def collate_fn(batch):
        """Collate function for original EEG data"""
        inputs = torch.stack([torch.as_tensor(item["original_data"].copy()) for item in batch])
        labels = torch.tensor([item["labels"] for item in batch])
        return {"original_data": inputs, "labels": labels}

    # Create data loaders (single-threaded for streaming compatibility)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=0,  # Single-threaded for streaming datasets
        drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn
    )
    return iter(train_loader), iter(val_loader)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def to_tensor(array):
    """
    Convert numpy array to PyTorch float tensor.
    
    Args:
        array (numpy.ndarray): Input array
        
    Returns:
        torch.Tensor: Float tensor
    """
    return torch.from_numpy(array).float()

# ============================================================================
# PRETRAINING DATASET CLASS
# ============================================================================
class PretrainingDataset(Dataset):
    """
    Dataset class for pretraining data stored in LMDB format.
    
    This class provides access to large-scale pretraining datasets
    stored efficiently in LMDB (Lightning Memory-Mapped Database).
    """
    
    def __init__(self, dataset_dir):
        """
        Initialize pretraining dataset from LMDB.
        
        Args:
            dataset_dir (str): Path to LMDB dataset directory
        """
        super(PretrainingDataset, self).__init__()
        
        # Open LMDB database in read-only mode
        self.db = lmdb.open(dataset_dir, readonly=True, lock=False, 
                           readahead=True, meminit=False)
        
        # Load dataset keys for indexing
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.keys)

    def __getitem__(self, idx):
        """
        Retrieve a single patch from the dataset.
        
        Args:
            idx (int): Sample index
            
        Returns:
            torch.Tensor: EEG patch data as float tensor
        """
        key = self.keys[idx]
        
        # Retrieve and deserialize patch data
        with self.db.begin(write=False) as txn:
            patch = pickle.loads(txn.get(key.encode()))
        
        return to_tensor(patch)

# ============================================================================
# DATASET FACTORY FUNCTIONS
# ============================================================================
def get_pretraining_ds(dataset_dir="/root/project/EEGPT/dataset/pretrain/TUEG_for_pretrain_full"):
    """
    Create pretraining dataset instance.
    
    Args:
        dataset_dir (str): Path to pretraining dataset directory
        
    Returns:
        PretrainingDataset: Dataset instance for pretraining
    """
    return PretrainingDataset(dataset_dir)

def get_bci2a_ds(path=""):
    """
    Create BCI Competition IV Dataset 2a streaming dataset.
    
    Args:
        path (str): Path to BCI2a dataset
        
    Returns:
        StreamingDataset: BCI2a dataset instance
    """
    return StreamingDataset(local=path, batch_size=BATCH_SIZE)

# ============================================================================
# REPRODUCIBILITY UTILITIES
# ============================================================================
def seed_everything(seed=42):
    """
    Set random seeds for reproducible results across all libraries.
    
    This function ensures deterministic behavior by setting seeds for:
    - Python random module
    - NumPy random number generator
    - PyTorch CPU random number generator
    - PyTorch CUDA random number generators
    - CuDNN deterministic mode
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True  # Ensure deterministic algorithms

# ============================================================================
# MAIN TESTING SCRIPT
# ============================================================================
if __name__ == "__main__":
    """
    Test script to verify dataset loading functionality.
    """
    print("Testing dataset loading functionality...")
    
    # Load and test TUAB datasets
    print("Loading TUAB datasets...")
    train_ds, val_ds = get_original_tuab_ds()
    
    print(f"Dataset sizes - Train: {len(train_ds)}, Validation: {len(val_ds)}")
    print("✓ Dataset loading completed successfully!")
