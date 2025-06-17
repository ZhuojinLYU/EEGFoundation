"""
Pre-trained Roformer Model Factory
This module provides functionality to create and configure Roformer models 
for EEG data processing with customizable architecture parameters.
"""

from transformers import RoFormerConfig, RoFormerForMaskedLM

# ============================================================================
# MODEL FACTORY FUNCTION
# ============================================================================
def get_roformer_model(num_hidden_layers=12,
                      num_attention_heads=8,
                      intermediate_size=2048,
                      max_position_embeddings=2000,
                      vocab_size=2006,
                      hidden_size=512):
    """
    Create and configure a Roformer model for EEG data processing.
    
    This function creates a student model with specified architecture parameters
    optimized for EEG signal processing and masked language modeling tasks.
    
    Args:
        num_hidden_layers (int): Number of transformer layers in the model.
                                Default: 12 (standard for base models)
        num_attention_heads (int): Number of attention heads per layer.
                                  Default: 8 (must divide hidden_size evenly)
        intermediate_size (int): Size of the feed-forward network intermediate layer.
                               Default: 2048 (typically 4x hidden_size)
        max_position_embeddings (int): Maximum sequence length the model can handle.
                                     Default: 2000 (suitable for EEG time series)
        vocab_size (int): Size of the vocabulary for tokenized EEG data.
                         Default: 2006 (includes special tokens: 2000 + CLS + SEP + others)
        hidden_size (int): Dimensionality of the model's hidden states.
                          Default: 512 (balance between capacity and efficiency)
    
    Returns:
        RoFormerForMaskedLM: Configured Roformer model ready for training or fine-tuning
        
    Notes:
        - The model is configured specifically for EEG data characteristics
        - Vocabulary size includes special tokens (CLS: 2003, SEP: 2004, etc.)
        - Position embeddings are set to handle typical EEG sequence lengths
        - Architecture is optimized for time-series pattern recognition
    """
    
    # Define student model configuration with EEG-specific parameters
    model_config = RoFormerConfig(
        num_hidden_layers=num_hidden_layers,          # Transformer depth
        num_attention_heads=num_attention_heads,      # Multi-head attention configuration
        intermediate_size=intermediate_size,          # Feed-forward network size
        max_position_embeddings=max_position_embeddings,  # Maximum sequence length
        vocab_size=vocab_size,                        # Tokenized EEG vocabulary size
        hidden_size=hidden_size,                      # Model dimensionality
    )

    # Create student model with masked language modeling head
    model = RoFormerForMaskedLM(config=model_config)

    return model

# ============================================================================
# MAIN TESTING SCRIPT
# ============================================================================
if __name__ == "__main__":
    """
    Test script to verify model creation and display architecture information.
    """

    roformer_model = get_roformer_model()
    print("roformer_model loaded")
    print(roformer_model)