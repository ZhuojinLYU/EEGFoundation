"""
Configuration class for EEG classification models and training parameters.
"""

class Config:
    """
    Configuration class for PAttn model architecture and training hyperparameters.
    
    Contains all necessary parameters for EEG classification tasks including
    model dimensions, training settings, and preprocessing options.
    """
    
    def __init__(self, 
                 # Model architecture parameters
                 seq_len=2000,              # EEG sequence length (time points)
                 patch_size=200,            # Patch size for attention mechanism
                 stride=100,                # Stride for patch extraction
                 d_model=512,               # Model hidden dimension
                 num_classes=2,             # Number of output classes
                 num_channel=23,            # Number of EEG channels
                 
                 # Model configuration options
                 rms_norm=True,             # Use RMS normalization
                 cc_attention=False,        # Use cross-channel attention
                 use_label_smoothing=False, # Apply label smoothing
                 activation='SiLU',         # Activation function
                 
                 # Training hyperparameters
                 EPOCHS=200,                # Number of training epochs
                 BATCH_SIZE=256,            # Training batch size
                 LEARNING_RATE=1e-4,        # Learning rate (uppercase)
                 learning_rate=1e-4,        # Learning rate (lowercase - legacy)
                 NUM_CLASSES=2,             # Number of classes (uppercase - legacy)
                 seed=42):                  # Random seed for reproducibility
        
        # Model architecture
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_channel = num_channel
        
        # Model options
        self.rms_norm = rms_norm
        self.cc_attention = cc_attention
        self.use_label_smoothing = use_label_smoothing
        self.activation = activation
        
        # Training parameters
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.LEARNING_RATE = LEARNING_RATE
        self.learning_rate = learning_rate  # Legacy compatibility
        self.NUM_CLASSES = NUM_CLASSES      # Legacy compatibility
        self.seed = seed