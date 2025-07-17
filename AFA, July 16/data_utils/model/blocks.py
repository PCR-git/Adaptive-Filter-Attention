import numpy as np
import torch
import torch.nn as nn

from model import ModReLU
from model import init_complexlinear, init_complex_matrix
from model import ComplexLinearLayer, AttentionLayer, ComplexAttentionLayer
from model import GatedSequenceNorm, GatedLayerNorm, GatedComplexSequenceNorm, GatedComplexLayerNorm
from model import FullPrecisionAttentionBlock_Nlayer

##########################################################################################
##########################################################################################

class TransformerBlock(nn.Module):
    """
    Custom transformer block using vanilla attention, with residual connections,
    layer normalization, and a feedforward network.
    """

    def __init__(self, input_dim, hidden_dim, args):
        """
        Initializes a single transformer block.

        Args:
            input_dim (int): Dimensionality of input and output.
            hidden_dim (int): Dimensionality of the hidden feedforward layer.
            args: Additional parameters passed to the AttentionLayer.
        """
        super(TransformerBlock, self).__init__()

        # Self-attention layer
        self.attn = AttentionLayer(input_dim, input_dim, args)

        # Layer norms before attention and MLP
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        # Feedforward network: Linear -> ReLU -> Linear
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        """
        Forward pass through the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            out (torch.Tensor): Output tensor of the same shape as input.
            attn (torch.Tensor): Attention weights.
        """

        # Layer norm before attention (pre-norm)
        x_norm = self.norm1(x)

        # Self-attention with residual connection
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out  # Residual connection

        # Layer norm before feedforward network
        x_norm = self.norm2(x)

        # Feedforward network with residual connection
        ffn_out = self.ffn(x_norm)
        out = x + ffn_out  # Residual connection

        return out, attn_weights
    
##########################################################################################
##########################################################################################

class ComplexTransformerBlock(nn.Module):
    """
    Complex-valued Transformer block with:
    - Complex self-attention
    - Two-layer complex feedforward network
    - Pre-layer normalization
    - Residual connections
    """

    def __init__(self, input_dim, hidden_dim, args):
        """
        Args:
            input_dim (int): Dimensionality of input and output.
            hidden_dim (int): Hidden size for the feedforward network.
            args: Additional arguments for attention or normalization.
        """
        super(ComplexTransformerBlock, self).__init__()

        # === Attention block ===
        self.attn = ComplexAttentionLayer(input_dim, hidden_dim, args)
        
        # === Normalization ===
        if args.norm_type == 'LayerNorm':
            self.norm1 = ComplexLayerNorm(input_dim, args)
            self.norm2 = ComplexLayerNorm(input_dim, args)
        elif args.norm_type == 'SequenceNorm':
            self.norm1 = ComplexSequenceNorm(input_dim, args)
            self.norm2 = ComplexSequenceNorm(input_dim, args)
        else:
            pass

        # === Feedforward block ===
        self.ffn = nn.Sequential(
            ComplexLinearLayer(input_dim, hidden_dim),
            ModReLU(),
            ComplexLinearLayer(hidden_dim, input_dim)
        )

        # === Initialization for FFN ===
        W1, b1 = init_complex_matrix(input_dim, hidden_dim, bias=True)
        W2, b2 = init_complex_matrix(hidden_dim, input_dim, bias=True)
        init_complexlinear(self.ffn[0], W1, b1)
        init_complexlinear(self.ffn[2], W2, b2)
        
        self.args = args

    def forward(self, x):
        """
        Forward pass for the complex transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            out (torch.Tensor): Output tensor of same shape as input.
            attn (torch.Tensor): Complex attention weights.
        """

        # === Self-attention block ===
        residual_attn = x
        
        if self.args.norm_type != None:
            x = self.norm1(x)
        
        attn_out, attn_weights = self.attn(x, x, x)
        x = residual_attn + attn_out  # First residual connection

        # === Feedforward block ===
        residual_ffn = x
        
        if self.args.norm_type != None:
            x = self.norm2(x)
        
        ffn_out = self.ffn(x)
        out = residual_ffn + ffn_out  # Second residual connection

        return out, attn_weights

##########################################################################################
##########################################################################################

class ComplexRealTransformerBlock(nn.Module):
    """
    Transformer block that uses complex-valued attention and real values for all other layers:
    - Complex self-attention that passes real values in and out
    - Real-valued feedforward network
    - Real valued pre-layer normalization
    - Residual connections
    """

    def __init__(self, input_dim, hidden_dim, args):
        """
        Initializes a single complex transformer block (with real inputs/outputs):

        Args:
            input_dim (int): Dimensionality of input and output.
            hidden_dim (int): Dimensionality of the hidden feedforward layer.
            args: Additional parameters passed to the AttentionLayer.
        """
        super(ComplexRealTransformerBlock, self).__init__()

        # Self-attention layer
        self.attn = ComplexAttentionLayer(input_dim, input_dim, args)

        # Layer norms before attention and MLP
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        # Feedforward network: Linear -> ReLU -> Linear
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        """
        Forward pass through the complex transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            out (torch.Tensor): Output tensor of the same shape as input.
            attn (torch.Tensor): Attention weights.
        """

        # Layer norm before attention (pre-norm)
        x_norm = self.norm1(x)

        # Self-attention with residual connection
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out  # Residual connection

        # Layer norm before feedforward network
        x_norm = self.norm2(x)

        # Feedforward network with residual connection
        ffn_out = self.ffn(x_norm)
        out = x + ffn_out  # Residual connection

        return out, attn_weights
    
##########################################################################################
##########################################################################################    
    
class AFATransformerBlock(nn.Module):
    """
    AFA Transformer Block
    """

    def __init__(self, input_dim, query_key_dim, value_dim, num_layers, args):
        """
        Initializes a single AFA transformer block (with real inputs/outputs):

        Args:
            input_dim (int): Dimensionality of input and output.
            hidden_dim (int): Dimensionality of the hidden feedforward layer.
            args: Additional parameters passed to the AttentionLayer.
        """
        super(AFATransformerBlock, self).__init__()

        # Self-attention layer
        self.attn = FullPrecisionAttentionBlock_Nlayer(input_dim, query_key_dim, value_dim, num_layers, args)

        # Layer norms before attention and MLP
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        # Feedforward network: Linear -> ReLU -> Linear
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, value_dim),
            nn.ReLU(),
            nn.Linear(value_dim, input_dim)
        )

    def forward(self, x, t_v):
        """
        Forward pass through the AFA transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            out (torch.Tensor): Output tensor of the same shape as input.
            attn (torch.Tensor): Attention weights.
        """

        # Layer norm before attention (pre-norm)
        x_norm = self.norm1(x)
#         x_norm = x

        # Self-attention with residual connection
        est, attn_out, attn_weight, X_ij_hat, lambda_h = self.attn(x_norm, x_norm, x_norm,  t_v)
        x = x + attn_out  # Residual connection

        # Layer norm before feedforward network
        x_norm = self.norm2(x)

        # Feedforward network with residual connection
        ffn_out = self.ffn(x_norm)
        out = x + ffn_out  # Residual connection

        return est, out, attn_weight, X_ij_hat, lambda_h
    
    