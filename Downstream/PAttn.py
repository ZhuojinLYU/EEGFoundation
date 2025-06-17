import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from attentions import MultiHeadAttention
from config import Config

class PAttnClassifier(nn.Module):
    """PyTorch module for EEG classification using patch-based attention mechanism."""

    def __init__(self, configs, device):
        """
        Initialize the PAttnClassifier model.

        Args:
            configs: Configuration object containing model parameters
            device: Device to run the model on (e.g., 'cuda' or 'cpu')
        """
        super(PAttnClassifier, self).__init__()
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size 
        self.stride = configs.stride 
        
        self.d_model = configs.d_model
        self.num_classes = configs.num_classes  # Number of output classes
        self.num_channel = configs.num_channel  # Number of EEG channels

        # Patch calculation (consistent with original PAttn)
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 2
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        
        # Feature extraction layers (maintaining original PAttn structure)
        self.in_layer = nn.Linear(self.patch_size, self.d_model)

        # Configuration flags
        self.cc_attention = configs.cc_attention
        self.use_rms_norm = configs.rms_norm
        self.use_label_smoothing = configs.use_label_smoothing

        # Attention mechanism
        self.basic_attn = MultiHeadAttention(d_model=self.d_model)

        # Embedding projection layer
        self.projection_embedding = nn.Linear(512, 512)  # TUAB (512,128) DEAP (512,512)

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.d_model * self.patch_num * self.num_channel, self.d_model),
            nn.Dropout(0.5),
            nn.SiLU(),
            nn.Dropout(0.8),  # dropout 0.2 : 72.5%    0.5 : 73.3%  0.8: 74.5%  
            nn.Linear(self.d_model, self.num_classes),
        )

    def norm(self, x, dim=0, means=None, stdev=None):
        """
        Normalize input tensor (same as original PAttn).

        Args:
            x: Input tensor
            dim: Dimension to normalize along
            means: Precomputed means (if available)
            stdev: Precomputed standard deviation (if available)

        Returns:
            Normalized tensor and optionally means and stdev
        """
        if means is not None:  
            return x * stdev + means
        else: 
            means = x.mean(dim, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=dim, keepdim=True, unbiased=False)+ 1e-5).detach() 
            x /= stdev
            return x, means, stdev 
    
    def rms_norm(self, x, dim=0, scale=None):
        """
        RMS normalization implementation.

        Args:
            x: Input tensor
            dim: Dimension to normalize along
            scale: Precomputed scale factor (if available)

        Returns:
            Normalized tensor and optionally scale factor
        """
        if scale is not None:
            return x * scale
        else:
            rms = torch.sqrt(torch.mean(x.pow(2), dim=dim, keepdim=True) + 1e-5).detach()
            x = x / rms
            return x, rms

    def forward(self, x, embedding, return_attn_x=False, return_dropout_x=False):
        """
        Forward pass of the model.

        Args:
            x: Input EEG data
            embedding: Additional embedding features
            return_attn_x: Whether to return attention outputs
            return_dropout_x: Whether to return dropout outputs

        Returns:
            Model outputs and optionally attention or dropout outputs
        """
        B, C = x.size(0), x.size(1)
        x = x.float()
        
        # Input normalization
        if self.use_rms_norm:
            x, _ = self.rms_norm(x, dim=2)  
        else:
            x, _, _ = self.norm(x, dim=2)  
        
        # Patch processing
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = self.in_layer(x)

        # Process embedding
        embedding = self.projection_embedding(embedding.unsqueeze(2))
        x = x + embedding
        
        # Attention mechanism
        if self.cc_attention:
            x, attn = self.basic_attn(x)
            if return_attn_x:
                attn_x = x
            x = rearrange(x, 'b c m l -> b (c m l)')
        else:
            x = rearrange(x, 'b c m l -> (b c) m l')
            x, attn = self.basic_attn(x, x, x)
            if return_attn_x:
                attn_x = x
            x = rearrange(x, '(b c) m l -> b (c m l)', b=B, c=C)
        
        # Classification head
        x = self.classification_head[0](x)  # Linear
        x = self.classification_head[1](x)  # Dropout(0.5)
        x = self.classification_head[2](x)  # SiLU()
        dropout_output = self.classification_head[3](x)  # Target Dropout layer
        x = self.classification_head[4](dropout_output)  # Linear
        
        if return_dropout_x:
            return x, attn, dropout_output
        else:
            return x, attn


if __name__ == "__main__":
    print("hello")
    downstream_clf = PAttnClassifier()