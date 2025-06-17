"""
Downstream Task Dataset Utilities Module
This module provides dataset classes and utility functions for loading and processing
EEG data for downstream classification tasks including TUAB and BCI2a datasets.
"""

import torch
import numpy as np
import pickle
import lmdb
from torch.utils.data import Dataset, DataLoader, Subset
from streaming import StreamingDataset
from transformers import AutoTokenizer, AutoModel
import random

# ============================================================================
# GLOBAL CONSTANTS
# ============================================================================
BATCH_SIZE = 256  # Default batch size for downstream tasks

# ============================================================================
# STREAMING DATASET CLASSES
# ============================================================================
class EEGData(StreamingDataset):
    """
    Streaming dataset class for EEG data with preprocessing and augmentation.
    
    Extends StreamingDataset to provide EEG-specific preprocessing including
    channel-wise standardization and optional noise augmentation for robustness.
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
            add_noise (bool): Whether to add augmentation noise for robustness
        """
        super().__init__(local=local_path, shuffle=shuffle, split=split,
                         allow_unsafe_types=allow_unsafe_types,
                         batching_method=batching_method, batch_size=batch_size)
        self.add_noise = add_noise

    def __getitem__(self, index):
        """
        Get a single sample with channel-wise standardization and optional augmentation.
        
        Args:
            index (int): Sample index
            
        Returns:
            dict: Processed sample with channel embeddings and labels
        """
        # Get raw sample from parent class
        obj = super().__getitem__(index)
        embeddings = np.copy(obj["channel_embeddings"]).astype(np.float32)
        
        # Channel-wise standardization: zero mean and unit variance per channel
        embeddings = (embeddings - embeddings.mean(axis=1, keepdims=True)) / \
                    (embeddings.std(axis=1, keepdims=True) + 1e-8)

        # Optional noise augmentation for data robustness (50% probability)
        if self.add_noise and np.random.rand() < 0.5:
            data_std = np.std(embeddings)
            noise = np.random.normal(0, data_std * 0.1, embeddings.shape)
            # Clip to prevent extreme outliers
            embeddings = np.clip(embeddings + noise, 
                               a_min=np.percentile(embeddings, 0.1),
                               a_max=np.percentile(embeddings, 99.9))

        # Convert to PyTorch tensors
        embeddings = torch.from_numpy(embeddings)
        # Note: Channel averaging could be applied here if needed
        # embeddings = embeddings.mean(dim=0)
        
        return {
            "channel_embeddings": embeddings,  
            "labels": torch.tensor(obj["labels"]).to(torch.long)
        }

# ============================================================================
# TUAB DATASET FUNCTIONS
# ============================================================================
def get_dl(train_path="", val_path="", batch_size=BATCH_SIZE):
    """
    Create data loaders for TUAB dataset with embeddings.
    
    Args:
        train_path (str): Path to training dataset
        val_path (str): Path to validation dataset
        batch_size (int): Batch size for data loading
        
    Returns:
        tuple: (train_loader, val_loader) DataLoader instances
    """
    # Create TUAB datasets with embeddings
    train_ds = EEGData(local_path=train_path, batch_size=BATCH_SIZE)
    val_ds = EEGData(local_path=val_path, batch_size=BATCH_SIZE)

    def collate_fn(batch):
        """Collate function to batch channel embeddings and labels"""
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
    Create raw TUAB datasets without data loaders.
    
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
# ORIGINAL TUAB DATA FUNCTIONS (WITHOUT EMBEDDINGS)
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
        """Collate function for original EEG data without embeddings"""
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
# EMBEDDING DATASET FUNCTIONS
# ============================================================================
def get_embedding_ds(train_path="", val_path=""):
    """
    Create datasets for data with pre-computed embeddings.
    
    Args:
        train_path (str): Path to training dataset with embeddings
        val_path (str): Path to validation dataset with embeddings
        
    Returns:
        tuple: (train_ds, val_ds) StreamingDataset instances
    """
    train_ds = StreamingDataset(local=train_path, batch_size=BATCH_SIZE)
    val_ds = StreamingDataset(local=val_path, batch_size=BATCH_SIZE)
    return train_ds, val_ds

def get_embedding_dl(train_path="", val_path="", batch_size=BATCH_SIZE):
    """
    Create data loaders for TUAB dataset with pre-computed embeddings.
    
    This function creates high-performance data loaders optimized for training
    with both original EEG data and corresponding embeddings.
    
    Args:
        train_path (str): Path to training dataset with embeddings
        val_path (str): Path to validation dataset with embeddings
        batch_size (int): Batch size for data loading
        
    Returns:
        tuple: (train_loader, val_loader) DataLoader instances
    """
    train_subset, val_subset = get_embedding_ds(train_path, val_path)

    def collate_fn(batch):
        """Collate function for data with both original and embedding features"""
        # Explicitly copy NumPy arrays to ensure tensor writability
        original_data = torch.stack([torch.as_tensor(item["original_data"].copy()) for item in batch])
        embedding_data = torch.stack([torch.as_tensor(item["embedding_data"].copy()) for item in batch])
        labels = torch.tensor([item["labels"] for item in batch])
        return {
            "original_data": original_data, 
            "embedding_data": embedding_data, 
            "labels": labels
        }

    # Create high-performance data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        num_workers=40,      # High parallelism for performance
        drop_last=True,
        shuffle=False,       # Streaming datasets handle shuffling
        prefetch_factor=200, # Large prefetch for smooth data flow
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        num_workers=40,
        drop_last=True,
        shuffle=False,
        prefetch_factor=200,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

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
# BCI2A DATASET FUNCTIONS
# ============================================================================
def get_bci2a_embedding_ds(path="/root/project/EEGPT/dataset/downstream/BCI2a/cabramod_processed_bci2a_embedding"):
    """
    Create BCI Competition IV Dataset 2a with embeddings.
    
    Args:
        path (str): Path to BCI2a dataset with embeddings
        
    Returns:
        StreamingDataset: BCI2a dataset instance
    """
    return StreamingDataset(local=path, batch_size=BATCH_SIZE)

def get_embedding_bci2a_ds_cross_subject(path="/root/project/EEGPT/dataset/downstream/BCI2a/cabramod_processed_bci2a_embedding",
                                        train_subset_ratio=5/9,
                                        val_subset_ratio=2/9,
                                        test_subset_ratio=2/9):
    """
    Create cross-subject splits for BCI2a dataset.
    
    Splits the dataset by subject ratio: subjects 1-5 for training, 
    subjects 6-7 for validation, subjects 8-9 for testing.
    
    Args:
        path (str): Path to BCI2a dataset
        train_subset_ratio (float): Ratio for training set (default: 5/9)
        val_subset_ratio (float): Ratio for validation set (default: 2/9)
        test_subset_ratio (float): Ratio for test set (default: 2/9)
        
    Returns:
        tuple: (train_ds, val_ds, test_ds) Dataset splits
    """
    full_ds = get_bci2a_embedding_ds(path)
    
    # Calculate split sizes based on total dataset size
    train_size = int(len(full_ds) * train_subset_ratio)
    val_size = int(len(full_ds) * val_subset_ratio)
    test_size = int(len(full_ds) * test_subset_ratio)
    
    # Create sequential splits (preserves subject grouping)
    train_ds = full_ds[:train_size]
    val_ds = full_ds[train_size:train_size + val_size]
    test_ds = full_ds[train_size + val_size:]

    return train_ds, val_ds, test_ds

def get_embedding_bci2a_ds_cross_subject_dl(path="/root/project/EEGPT/dataset/downstream/BCI2a/cabramod_processed_bci2a_embedding",
                                           train_subset_ratio=5/9,
                                           val_subset_ratio=2/9,
                                           test_subset_ratio=2/9):
    """
    Create cross-subject data loaders for BCI2a dataset.
    
    Args:
        path (str): Path to BCI2a dataset
        train_subset_ratio (float): Ratio for training set
        val_subset_ratio (float): Ratio for validation set
        test_subset_ratio (float): Ratio for test set
        
    Returns:
        tuple: (train_iter, val_iter, test_iter) DataLoader iterators
    """
    train_ds, val_ds, test_ds = get_embedding_bci2a_ds_cross_subject(
        path, train_subset_ratio, val_subset_ratio, test_subset_ratio
    )

    def collate_fn(batch):
        """Collate function for BCI2a data with embeddings"""
        original_data = torch.stack([torch.as_tensor(item["original_data"].copy()) for item in batch])
        embedding_data = torch.stack([torch.as_tensor(item["embedding_data"].copy()) for item in batch])
        labels = torch.stack([torch.as_tensor(item["labels"].copy()) for item in batch])
        return {
            "original_data": original_data, 
            "embedding_data": embedding_data, 
            "labels": labels
        }

    # Create high-performance data loaders for BCI2a
    train_dl = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        num_workers=40,
        prefetch_factor=200, 
        drop_last=True, 
        shuffle=True,  # Shuffle training data
        collate_fn=collate_fn
    )
    val_dl = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        num_workers=40,
        prefetch_factor=200, 
        drop_last=True, 
        shuffle=False,  # No shuffling for validation
        collate_fn=collate_fn
    )
    test_dl = DataLoader(
        test_ds, 
        batch_size=BATCH_SIZE, 
        num_workers=40,
        prefetch_factor=200, 
        drop_last=True, 
        shuffle=False,  # No shuffling for testing
        collate_fn=collate_fn
    )

    return iter(train_dl), iter(val_dl), iter(test_dl)

# ============================================================================
# REPRODUCIBILITY UTILITIES
# ============================================================================
def seed_everything(seed=42):
    """
    Set random seeds for reproducible results across all libraries.
    
    Ensures deterministic behavior by setting seeds for Python random,
    NumPy, PyTorch CPU/CUDA, and enabling deterministic CuDNN algorithms.
    
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
    Test script to verify BCI2a dataset loading functionality.
    """
    print("Testing BCI2a cross-subject dataset loading...")
    
    ds_path = ""  # Set your dataset path here
    
    # Test cross-subject data loading
    train_dl, val_dl, test_dl = get_embedding_bci2a_ds_cross_subject_dl(
        ds_path, 
        train_subset_ratio=5/9, 
        val_subset_ratio=2/9, 
        test_subset_ratio=2/9
    )
    
    # Test first batch
    for batch in train_dl:
        print(f"Original data shape: {batch['original_data'].shape}")
        print(f"Embedding data shape: {batch['embedding_data'].shape}")
        print(f"Labels: {batch['labels']}")
        break
    
    print("✓ BCI2a dataset testing completed successfully!")
    
