import numpy as np
import torch
import torch.nn as nn

# from model import ModReLU
from model import resolve_multihead_dims
from model import init_complexlinear, init_complex_matrix
from model import ComplexLinearLayer, MultiHeadAttentionLayer
# from model import MultiHeadComplexAttentionLayer
from model import ComplexRMSNorm, MultiheadIsotropicAFA

##########################################################################################
##########################################################################################

class TransformerBlock(nn.Module):
    """
    Custom transformer block using vanilla attention, with residual connections,
    layer normalization, and a feedforward network.
    """

    def __init__(self, input_dim, qkv_dim, hidden_dim, num_heads, args, Norm=nn.RMSNorm):
        """
        Initializes a single transformer block.

        Args:
            input_dim (int): Dimensionality of input and output.
            hidden_dim (int): Dimensionality of the hidden feedforward layer.
            args: Additional parameters passed to the AttentionLayer.
        """
        super(TransformerBlock, self).__init__()

        # Self-attention layer
        if args.complex_transformer == 1:
            self.attn = ComplexMultiHeadAttentionLayer(input_dim, qkv_dim, num_heads, args)
        else:
            self.attn = MultiHeadAttentionLayer(input_dim, qkv_dim, num_heads, args)
        
        # Layer norms before attention and MLP
        if Norm == None:
            self.norm1 = self.norm2 = None
        else:
            self.norm1 = Norm(input_dim)
            self.norm2 = Norm(input_dim)

        # Feedforward network: Linear -> ReLU -> Linear
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
#         self.g_param = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        """
        Forward pass through the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            out (torch.Tensor): Output tensor of the same shape as input.
            attn (torch.Tensor): Attention weights.
        """
        
        # === Self-attention block ===
        
        # Layer norm before attention (pre-norm)
        if self.norm1 == None:
            x_norm = x
        else:
            x_norm = self.norm1(x)

        # Self-attention with residual connection
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out  # Residual connection
        
#         eta = torch.sigmoid(self.g_param)
#         x = (1 - eta) * x + eta * attn_out

        # Layer norm before feedforward network
        if self.norm2 == None:
            x_norm = x
        else:
            x_norm = self.norm2(x)

        # Feedforward network with residual connection
        ffn_out = self.ffn(x_norm)
        out = x + ffn_out  # Residual connection

        return out, attn_weights

##########################################################################################
##########################################################################################

# class ComplexTransformerBlock(nn.Module):
#     """
#     Custom transformer block using vanilla attention, with residual connections,
#     layer normalization, and a feedforward network.
#     """

#     def __init__(self, input_dim, qkv_dim, hidden_dim, num_heads, args, Norm=ComplexRMSNorm):
#         """
#         Initializes a single transformer block.

#         Args:
#             input_dim (int): Dimensionality of input and output.
#             hidden_dim (int): Dimensionality of the hidden feedforward layer.
#             args: Additional parameters passed to the AttentionLayer.
#         """
#         super(ComplexTransformerBlock, self).__init__()

#         # Self-attention layer
#         self.attn = ComplexMultiHeadAttentionLayer(input_dim, qkv_dim, num_heads, args)
        
#         # Layer norms before attention and MLP
#         if Norm == None:
#             self.norm1 = self.norm2 = None
#         else:
#             self.norm1 = Norm(input_dim)
#             self.norm2 = Norm(input_dim)

#         # Feedforward network: Linear -> ReLU -> Linear
#         self.ffn = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, input_dim)
#         )
        
#         self.g_param = nn.Parameter(torch.zeros(input_dim))

#     def forward(self, x):
#         """
#         Forward pass through the transformer block.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

#         Returns:
#             out (torch.Tensor): Output tensor of the same shape as input.
#             attn (torch.Tensor): Attention weights.
#         """
        
#         # === Self-attention block ===
        
#         # Layer norm before attention (pre-norm)
#         if self.norm1 == None:
#             x_norm = x
#         else:
#             x_norm = self.norm1(x)

#         # Self-attention with residual connection
#         attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm)
#         x = x + attn_out  # Residual connection
        
# #         eta = torch.sigmoid(self.g_param)
# #         x = (1 - eta) * x + eta * attn_out

#         # Layer norm before feedforward network
#         if self.norm2 == None:
#             x_norm = x
#         else:
#             x_norm = self.norm2(x)

#         # Feedforward network with residual connection
#         ffn_out = self.ffn(x_norm)
#         out = x + ffn_out  # Residual connection

#         return out, attn_weights
    
##########################################################################################
##########################################################################################    

class AFATransformerBlock(nn.Module):
    """
    Multihead Simplified AFA Transformer Block
    """

    def __init__(self, args, n_heads, input_dim, query_key_dim=None, value_dim=None, query_key_dim_total=None, value_dim_total=None, hidden_dim=None, Norm=nn.RMSNorm):
        """
        Initializes a single Multihead Simplified AFA transformer block (with real inputs/outputs):

        Args:
            input_dim (int): Dimensionality of input and output.
            hidden_dim (int): Dimensionality of the hidden feedforward layer.
            args: Additional parameters passed to the AttentionLayer.
        """
        super(AFATransformerBlock, self).__init__()
        
        if query_key_dim==None or value_dim==None or query_key_dim_total==None or value_dim_total==None:
            # Set query_key and value dims, depending on whether user provided total dims, or head dims
            query_key_dim, value_dim, query_key_dim_total, value_dim_total = resolve_multihead_dims(n_heads, query_key_dim, value_dim, query_key_dim_total, value_dim_total)

        # Self-attention layer
        self.attn = MultiheadIsotropicAFA(args, n_heads, input_dim, query_key_dim, value_dim, query_key_dim_total, value_dim_total)

        # Layer norms before attention and MLP
        if Norm == None:
            self.norm1 = self.norm2 = None
        else:
            self.norm1 = Norm(input_dim)
            self.norm2 = Norm(input_dim)

        # Feedforward network: Linear -> ReLU -> Linear
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.args = args

    def forward(self, x, t_measure=None):
        """
        Forward pass through the AFA transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, seq_len, input_dim)

        Returns:
            out (torch.Tensor): Output tensor of the same shape as input.
            attn (torch.Tensor): Attention weights.
        """
        
#         x_mag = torch.sqrt(torch.sum(x**2,axis=-1,keepdims=True))

        # Norm before attention (pre-norm)
        if self.norm1 == None or self.args.use_complex_input_norm == 1:
            x_norm = x
        else:
            x_norm = self.norm1(x)

        # Self-attention with residual connection
        attn_out, output_dict = self.attn(x_norm, x_norm, x_norm, t_measure)

        if self.args.use_inner_residual_and_norm == 0:
            x = x + attn_out
            x_norm = self.norm2(x)
        else:
            x = output_dict['est_latent']
            x_norm = attn_out
            
        ffn_out = self.ffn(x_norm) # Feedforward network with residual connection
        out = x + ffn_out  # Residual connection

        return out, output_dict
    
