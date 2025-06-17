"""
BCI Competition IV Dataset 2a Embedding Generator
Script to generate embeddings for BCI2a dataset using a pre-trained Roformer model.
Creates streaming datasets with original EEG data, model embeddings, and labels.
"""

import torch
import numpy as np
import time
import os
import random
import lmdb
import pickle
from torch.utils.data import Dataset, DataLoader
from streaming import MDSWriter
from Dataset.test_model_embedding import get_cls_embedding, RoformerModelWrapper, bin_data

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def to_tensor(array):
    """Convert numpy array to PyTorch tensor with float dtype"""
    return torch.from_numpy(array).float()

# ============================================================================
# CUSTOM DATASET CLASS
# ============================================================================
class CustomDataset(Dataset):
    """
    Custom Dataset class for loading BCI2a data from LMDB database.
    
    Args:
        data_dir (str): Path to LMDB database directory
        mode (str): Dataset mode ('train', 'val', 'test')
    """
    
    def __init__(self, data_dir, mode='train'):
        super(CustomDataset, self).__init__()
        
        # Open LMDB database in read-only mode
        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        
        # Load dataset keys for specified mode
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.keys)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (normalized_data, label) where data is scaled by 1/100
        """
        key = self.keys[idx]
        
        # Retrieve sample from LMDB
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        
        # Process data: reshape to (22, -1) and normalize
        data = pair['sample'].reshape(22, -1)
        label = pair['label']
        
        return data / 100, label  # Normalize data by dividing by 100

    def collate(self, batch):
        """
        Collate function for DataLoader to batch samples.
        
        Args:
            batch (list): List of (data, label) tuples
            
        Returns:
            tuple: Batched tensors (x_data, y_label)
        """
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label).long()

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ============================================================================
# DATASET FORMAT DOCUMENTATION
# ============================================================================
"""
Dataset Format Specifications:

Streaming Dataset Format (Output):
{
    "original_data": [22, 1000],     # Original EEG data (22 channels, 1000 time points)
    "embedding_data": [22, 512],     # Model-generated embeddings from distilled model
    "labels": [1]                    # Classification labels (4 classes for BCI2a)
}

Original Dataset Format (Input):
- Full BCI2a dataset from LMDB
- Format after loading:
{
    "original_data": [22, 1000],     # Raw EEG data (normalized)
    "labels": [1]                    # Labels only (no embeddings)
}

Note: BCI2a has 4 classes for motor imagery tasks:
- Class 0: Left hand
- Class 1: Right hand  
- Class 2: Both feet
- Class 3: Tongue
"""

# ============================================================================
# MAIN PROCESSING SCRIPT
# ============================================================================
if __name__ == "__main__":
    
    # ===== MODEL SETUP =====
    checkpoint_path = ""  # Path to pre-trained Roformer model checkpoint
    model = RoformerModelWrapper.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    print(f"Model loaded and moved to {device}")

    # ===== DATA LOADING =====
    original_ds = CustomDataset(data_dir='')  # Path to BCI2a LMDB dataset
    print(f"Loaded BCI2a dataset with {len(original_ds)} samples")

    # ===== OUTPUT PATH =====
    full_BCI2a_embedding_path = ""  # Output path for BCI2a dataset with embeddings

    # ===== MDS WRITER CONFIGURATION =====
    columns = {
        "original_data": "ndarray:float64",   # Original EEG data (full precision)
        "embedding_data": "ndarray:float32",  # Embeddings (reduced precision)
        "labels": "int64",                    # Labels (standard integer)
    }
    size_limit = 1 << 31  # 2GB per MDS file limit

    # ===== DATASET PROCESSING =====
    print("\nProcessing BCI2a dataset with embedding generation...")
    
    with MDSWriter(
        out=full_BCI2a_embedding_path,
        columns=columns,
        size_limit=size_limit
    ) as writer:
        
        for idx, original_data in enumerate(original_ds):
            start_time = time.time()
            
            # Extract EEG data and label
            eeg_data, label = original_data
            
            # Debug information for first few samples
            if idx < 5:
                print(f"Sample {idx} - EEG data shape: {eeg_data.shape}, Label: {label}")
            
            # Generate embedding using pre-trained model
            embedding = get_cls_embedding(model, bin_data(eeg_data)).cpu().numpy()
            
            # Write sample to streaming dataset
            writer.write({
                "original_data": eeg_data,
                "embedding_data": embedding,
                "labels": [label],  # Wrap label in list for consistency
            })
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Log progress for every 100 samples or first 10 samples
            if idx % 100 == 0 or idx < 10:
                print(f"Processed sample {idx}/{len(original_ds)} - Time: {processing_time:.2f}s")

    # ===== COMPLETION SUMMARY =====
    print("\n" + "="*60)
    print("BCI2A EMBEDDING GENERATION COMPLETE")
    print("="*60)
    print(f"Dataset saved to: {full_BCI2a_embedding_path}")
    print(f"Total samples processed: {len(original_ds)}")
    print("Dataset ready for downstream classification tasks!")


