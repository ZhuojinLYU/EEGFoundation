"""
Multi-Head Attention Module for EEG Signal Processing
Implements scaled dot-product attention and multi-head attention mechanisms
optimized for time-series EEG data classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
    
class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    
    Computes attention weights using the scaled dot-product formula:
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
    """

    def __init__(self, temperature, attn_dropout=0.1):
        """
        Initialize scaled dot-product attention.
        
        Args:
            temperature (float): Scaling factor (typically sqrt(d_k))
            attn_dropout (float): Dropout rate for attention weights
        """
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        """
        Compute scaled dot-product attention.
        
        Args:
            q (torch.Tensor): Query tensor
            k (torch.Tensor): Key tensor  
            v (torch.Tensor): Value tensor
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            tuple: (output, attention_weights)
        """
        # Compute attention scores: Q * K^T / temperature
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # Apply mask if provided (set masked positions to large negative value)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # Apply softmax and dropout to attention weights
        attn = self.dropout(F.softmax(attn, dim=-1))
        
        # Apply attention weights to values
        output = torch.matmul(attn, v)
        
        return output, attn
        
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module for capturing different types of dependencies.
    
    Runs multiple attention heads in parallel and concatenates their outputs,
    allowing the model to focus on different representation subspaces.
    """
    
    def __init__(self, d_model=-1, n_head=8, d_k=-1, d_v=-1, dropout=0.1):
        """
        Initialize multi-head attention module.
        
        Args:
            d_model (int): Model dimension (hidden size)
            n_head (int): Number of attention heads
            d_k (int): Key dimension per head (auto-computed if -1)
            d_v (int): Value dimension per head (auto-computed if -1)  
            dropout (float): Dropout rate for output
        """
        super().__init__()
        self.n_head = n_head
        
        # Auto-compute head dimensions if not specified
        d_k = d_model // n_head 
        d_v = d_k
        self.d_k = d_k
        self.d_v = d_v

        # Linear projections for queries, keys, and values
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)  # Query projection
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)  # Key projection
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)  # Value projection
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)    # Output projection

        # Scaled dot-product attention with temperature scaling
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # Regularization and normalization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        """
        Forward pass through multi-head attention.
        
        Args:
            q (torch.Tensor): Query tensor [batch_size, seq_len, d_model]
            k (torch.Tensor): Key tensor [batch_size, seq_len, d_model]
            v (torch.Tensor): Value tensor [batch_size, seq_len, d_model]
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            tuple: (output, attention_weights)
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Store residual connection
        residual = q

        # Project inputs to multiple heads: [batch, seq_len, n_head, d_k/d_v]
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention computation: [batch, n_head, seq_len, d_k/d_v]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Expand mask for multi-head broadcasting if provided
        if mask is not None:
            mask = mask.unsqueeze(1)   # Add head dimension

        # Apply scaled dot-product attention
        q, attn = self.attention(q, k, v, mask=mask)

        # Reshape and concatenate heads: [batch, seq_len, n_head * d_v]
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        
        # Apply output projection and dropout
        q = self.dropout(self.fc(q))
        
        # Add residual connection
        q += residual

        # Apply layer normalization
        q = self.layer_norm(q)
        
        return q, attn