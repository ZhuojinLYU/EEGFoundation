"""
TUAB Dataset Embedding Generator
Script to generate embeddings for TUAB dataset using a pre-trained Roformer model.
Creates streaming datasets with original EEG data, model embeddings, and labels.
"""

import torch
import numpy as np
import time
from utils import get_original_tuab_ds
from streaming import MDSWriter
from Dataset.test_model_embedding import get_cls_embedding, RoformerModelWrapper, bin_data

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ============================================================================
# DATASET FORMAT DOCUMENTATION
# ============================================================================
"""
Dataset Format Specifications:

Streaming Dataset Format:
{
    "original_data": [23, 2000],    # Original EEG data (channels x time_points)
    "embedding_data": [23, 512],    # Model-generated embeddings
    "labels": [1]                   # Classification labels
}

Original Dataset (before processing):
- train_ds: 228,796 samples
- val_ds: 59,603 samples

Original Format:
{
    "original_data": [23, 2000],    # Raw EEG data
    "labels": [1]                   # Labels only (no embeddings)
}
"""

# ============================================================================
# MAIN PROCESSING SCRIPT
# ============================================================================
if __name__ == "__main__":
    
    # ===== MODEL SETUP =====
    checkpoint_path = ""  # Path to pre-trained model checkpoint
    model = RoformerModelWrapper.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    print(f"Model loaded and moved to {device}")

    # ===== DATA LOADING =====
    train_ds, val_ds = get_original_tuab_ds()
    print(f"Loaded datasets - Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # ===== OUTPUT PATHS =====
    train_ds_embedding_path = ""  # Output path for training dataset with embeddings
    val_ds_embedding_path = ""    # Output path for validation dataset with embeddings
    
    # Optional: Demo mode with limited samples
    # train_demo_len = 1000
    # val_demo_len = 200

    # ===== MDS WRITER CONFIGURATION =====
    columns = {
        "original_data": "ndarray:float16",   # Original EEG data (reduced precision)
        "embedding_data": "ndarray:float32",  # Embeddings (full precision)
        "labels": "int8",                     # Labels (minimal space)
    }
    size_limit = 1 << 31  # 2GB per MDS file limit

    # ===== VALIDATION DATASET PROCESSING =====
    print("\nProcessing validation dataset...")
    with MDSWriter(
        out=val_ds_embedding_path,
        columns=columns,
        size_limit=size_limit
    ) as writer:
        
        for idx, val_data in enumerate(val_ds):
            start_time = time.time()
            
            # Generate embedding for current sample
            original_data = val_data["original_data"]
            embedding = get_cls_embedding(model, bin_data(original_data)).cpu().numpy()
            
            # Write sample to streaming dataset
            writer.write({
                "original_data": original_data,
                "embedding_data": embedding,
                "labels": val_data["labels"],
            })
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Log progress
            if idx % 1000 == 0 or idx < 10:
                print(f"Val sample {idx}/{len(val_ds)} - Processing time: {processing_time:.2f}s")

    print("Validation dataset processing completed!")

    # ===== TRAINING DATASET PROCESSING =====
    print("\nProcessing training dataset...")
    with MDSWriter(
        out=train_ds_embedding_path,
        columns=columns,
        size_limit=size_limit
    ) as writer:
        
        for idx, train_data in enumerate(train_ds):
            start_time = time.time()
            
            # Generate embedding for current sample
            original_data = train_data["original_data"]
            embedding = get_cls_embedding(model, bin_data(original_data)).cpu().numpy()
            
            # Write sample to streaming dataset
            writer.write({
                "original_data": original_data,
                "embedding_data": embedding,
                "labels": train_data["labels"],
            })
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Log progress
            if idx % 1000 == 0 or idx < 10:
                print(f"Train sample {idx}/{len(train_ds)} - Processing time: {processing_time:.2f}s")

    print("Training dataset processing completed!")
    
    # ===== COMPLETION SUMMARY =====
    print("\n" + "="*60)
    print("EMBEDDING GENERATION COMPLETE")
    print("="*60)
    print(f"Training dataset saved to: {train_ds_embedding_path}")
    print(f"Validation dataset saved to: {val_ds_embedding_path}")
    print(f"Total samples processed: {len(train_ds) + len(val_ds)}")


