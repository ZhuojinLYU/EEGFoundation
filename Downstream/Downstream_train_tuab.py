"""
EEG Classification Training Script using PAttn Model
Training script for TUAB dataset with embedding features
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import numpy as np
from torchmetrics import (
    Accuracy, 
    F1Score, 
    CohenKappa,
    Precision,
    Recall,
    ConfusionMatrix
)

from utils import seed_everything, get_embedding_dl
from PAttn import PAttnClassifier
from config import Config

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
EPOCHS = 200
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
NUM_CLASSES = 2
Scale_factor = 0.01
PATCH_SIZE = 150

# ============================================================================
# INITIALIZATION
# ============================================================================
config = Config(patch_size=PATCH_SIZE)
seed_everything(seed=config.seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# LOGGING SETUP
# ============================================================================
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f""
os.makedirs(log_dir, exist_ok=True)

train_log = os.path.join(log_dir, "training.log")
metrics_log = os.path.join(log_dir, "metrics.csv")

# Initialize CSV file with headers
with open(metrics_log, 'w') as f:
    f.write("epoch,phase,loss,accuracy,balanced_acc,weighted_f1,macro_f1,cohen_kappa\n")

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================
model = PAttnClassifier(
    configs=config,
    device=device,
).to(device)

# ============================================================================
# LOSS FUNCTION AND OPTIMIZER
# ============================================================================
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)

# ============================================================================
# METRICS FUNCTIONS
# ============================================================================
def init_metrics():
    """Initialize evaluation metrics for training and validation"""
    return {
        'accuracy': Accuracy(task='multiclass', num_classes=NUM_CLASSES).to(device),
        'balanced_acc': Recall(task='multiclass', num_classes=NUM_CLASSES, average='macro').to(device),
        'weighted_f1': F1Score(task='multiclass', num_classes=NUM_CLASSES, average='weighted').to(device),
        'macro_f1': F1Score(task='multiclass', num_classes=NUM_CLASSES, average='macro').to(device),
        'cohen_kappa': CohenKappa(task='multiclass', num_classes=NUM_CLASSES).to(device),
        'confusion_matrix': ConfusionMatrix(task='multiclass', num_classes=NUM_CLASSES).to(device)
    }

# ============================================================================
# LOGGING UTILITY FUNCTIONS
# ============================================================================
def log_message(message, print_msg=True):
    """Write message to log file and optionally print to console"""
    with open(train_log, 'a') as f:
        f.write(message + "\n")
    if print_msg:
        print(message)

def save_metrics_to_csv(epoch, phase, loss, metrics_dict):
    """Save epoch metrics to CSV file for later analysis"""
    with open(metrics_log, 'a') as f:
        f.write(f"{epoch},{phase},{loss:.4f},")
        f.write(f"{metrics_dict['accuracy']:.4f},")
        f.write(f"{metrics_dict['balanced_acc']:.4f},")
        f.write(f"{metrics_dict['weighted_f1']:.4f},")
        f.write(f"{metrics_dict['macro_f1']:.4f},")
        f.write(f"{metrics_dict['cohen_kappa']:.4f}\n")

def log_metrics(epoch, phase, metrics_dict, loss, cm=None):
    """Comprehensive logging of metrics to both file and console"""
    log_str = f"\n{phase} Epoch {epoch}:\n"
    log_str += f"Loss: {loss:.4f}\n"
    
    for name, value in metrics_dict.items():
        if name != 'confusion_matrix':
            log_str += f"{name}: {value:.4f}\n"
    
    # Save confusion matrix every 10 epochs
    if cm is not None and epoch % 10 == 0:
        cm_file = os.path.join(log_dir, f'confusion_matrix_{phase}_epoch{epoch}.txt')
        np.savetxt(cm_file, cm.cpu().numpy(), fmt='%d')
        log_str += f"Confusion Matrix saved to {cm_file}\n"
    
    log_str += "-"*50
    log_message(log_str)
    save_metrics_to_csv(epoch, phase, loss, metrics_dict)

# ============================================================================
# TRAINING SETUP
# ============================================================================
# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))

# Log initial training information
log_message(f"Starting training at {timestamp}\n")
log_message(f"Device: {device}")
log_message(f"Model architecture:\n{model}\n")

# Initialize tracking variables
best_val_acc = 0.0
train_metrics = init_metrics()
val_metrics = init_metrics()

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
for epoch in range(EPOCHS):
    # ===== DATA LOADING =====
    train_dl, val_dl = get_embedding_dl()
    log_message(f"\nEpoch {epoch+1}/{EPOCHS} - Data loaded")
    
    # ===== TRAINING PHASE =====
    model.train()
    epoch_loss = 0.0
    total_samples = 0
    
    for batch_idx, batch in enumerate(train_dl):
        # Prepare batch data
        inputs = batch["original_data"].to(device).float()
        labels = batch["labels"].to(device).long()
        embedding = batch["embedding_data"].to(device).float() * Scale_factor
        
        # Forward pass
        outputs, _ = model(inputs, embedding)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update training metrics
        preds = outputs.argmax(dim=1)
        for metric in train_metrics.values():
            metric.update(preds, labels)
        
        # Accumulate loss
        epoch_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        
        # Log batch progress
        if batch_idx % 100 == 0:
            log_message(f"Epoch {epoch} Batch {batch_idx} - Loss: {loss.item():.4f}", print_msg=False)
    
    # ===== TRAINING METRICS COMPUTATION =====
    train_loss = epoch_loss / total_samples
    train_results = {name: metric.compute() for name, metric in train_metrics.items() 
                    if not isinstance(metric, ConfusionMatrix)}
    cm_train = train_metrics['confusion_matrix'].compute()
    
    # Log training results
    log_metrics(epoch, "train", train_results, train_loss, cm_train)
    
    # TensorBoard logging for training
    writer.add_scalars('Loss', {'train': train_loss}, epoch)
    writer.add_scalars('Accuracy', {'train': train_results['accuracy']}, epoch)
    
    # Reset training metrics for next epoch
    for metric in train_metrics.values():
        metric.reset()
    
    # ===== VALIDATION PHASE =====
    model.eval()
    val_loss = 0.0
    val_samples = 0
    
    with torch.no_grad():
        for batch in val_dl:
            # Prepare validation batch data
            inputs = batch["original_data"].to(device).float()
            labels = batch["labels"].to(device).long()
            embedding = batch["embedding_data"].to(device).float() * Scale_factor

            # Forward pass (no gradients)
            outputs, _ = model(inputs, embedding)
            val_loss += criterion(outputs, labels) * inputs.size(0)
            
            # Update validation metrics
            preds = outputs.argmax(dim=1)
            for metric in val_metrics.values():
                metric.update(preds, labels)
            
            val_samples += inputs.size(0)
    
    # ===== VALIDATION METRICS COMPUTATION =====
    val_loss /= val_samples
    val_results = {name: metric.compute() for name, metric in val_metrics.items() 
                  if not isinstance(metric, ConfusionMatrix)}
    cm_val = val_metrics['confusion_matrix'].compute()
    
    # Log validation results
    log_metrics(epoch, "val", val_results, val_loss, cm_val)
    
    # TensorBoard logging for validation
    writer.add_scalars('Loss', {'val': val_loss}, epoch)
    writer.add_scalars('Accuracy', {'val': val_results['accuracy']}, epoch)
    
    # ===== MODEL CHECKPOINTING =====
    # Save best model based on validation accuracy
    if val_results['accuracy'] > best_val_acc:
        best_val_acc = val_results['accuracy']
        best_model_path = os.path.join(log_dir, f'best_model_{timestamp}.pth')
        torch.save(model.state_dict(), best_model_path)
        log_message(f"New best model saved with val acc: {best_val_acc:.4f}")
    
    # Save periodic checkpoints
    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        checkpoint_path = os.path.join(log_dir, f'model_epoch_{epoch}_{timestamp}.pth')
        torch.save(model.state_dict(), checkpoint_path)
    
    # Reset validation metrics for next epoch
    for metric in val_metrics.values():
        metric.reset()

# ============================================================================
# TRAINING COMPLETION
# ============================================================================
# Save final model
final_model_path = os.path.join(log_dir, f'model_final_{timestamp}.pth')
torch.save(model.state_dict(), final_model_path)

# Close TensorBoard writer
writer.close()

# Final logging
log_message(f"\nTraining completed at {datetime.now().strftime('%Y%m%d_%H%M%S')}")
log_message(f"Best validation accuracy: {best_val_acc:.4f}")
log_message(f"Final model saved to {final_model_path}")
log_message(f"Best model saved to {best_model_path}")
log_message(f"Logs saved to {log_dir}")