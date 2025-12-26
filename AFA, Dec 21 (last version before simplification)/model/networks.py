import numpy as np
import torch
import torch.nn as nn

from model import init_complexlinear, init_complex_matrix
from model import ComplexLinearLayer, MultiHeadAttentionLayer
from model import RoPE
from model import TransformerBlock
from model import MultiheadIsotropicAFA, AFATransformerBlock
from model import resolve_multihead_dims

##########################################################################################
##########################################################################################

class Attention_1layer(nn.Module):
    """
    Neural network with a single attention layer
    """

    # Initialize the network and specify input/output dimensions:
    def __init__(self, input_dim, hidden_dim, num_heads, args):
        super(Attention_1layer, self).__init__()

        self.a1 = MultiHeadAttentionLayer(input_dim, hidden_dim, num_heads, args)
        
     # Build the network:
    def forward(self, inputs):

        out, attn = self.a1(inputs, inputs, inputs)
        
        return out, attn
    
##########################################################################################
##########################################################################################

# class ComplexAttention_1layer(nn.Module):
#     """
#     Neural network with a single attention layer
#     """

#     # Initialize the network and specify input/output dimensions:
#     def __init__(self, input_dim, hidden_dim, num_heads, args):
#         super(ComplexAttention_1layer, self).__init__()

#         self.a1 = MultiHeadComplexAttentionLayer(input_dim, hidden_dim, num_heads, args)
        
#      # Build the network:
#     def forward(self, inputs):

#         out, attn = self.a1(inputs, inputs, inputs)
        
#         return out, attn
    
##########################################################################################
##########################################################################################

class AFA_1layer(nn.Module):
    """
    Neural network with a single precision attention block
    """

    # Initialize the network and specify input/output dimensions:
    def __init__(self, args):
        super(AFA_1layer, self).__init__()

#         self.a1 = FullPrecisionAttentionBlock(args.d_e, args)
        self.layers = nn.ModuleList([FullPrecisionAttentionBlock(args.d_e, args.d_k, args.d_v, args)])

     # Build the network:
    def forward(self, inputs, t_measure):
        
        layer = self.layers[0]
        est, out, Q_ij, Z_ij_hat_all, lambda_h = layer(inputs, inputs, inputs, t_measure)

        return est, out, Q_ij, Z_ij_hat_all, lambda_h
    
##########################################################################################
##########################################################################################

class MultiheadIsotropicAFA_1layer(nn.Module):
    """
    Neural network with a single multihead simplified precision attention block
    """

    # Initialize the network and specify input/output dimensions:
    def __init__(self, args, n_heads, input_dim, query_key_dim=None, value_dim=None, query_key_dim_total=None, value_dim_total=None):
        super(MultiheadIsotropicAFA_1layer, self).__init__()

        self.layers = nn.ModuleList([MultiheadIsotropicAFA(args, n_heads, input_dim, query_key_dim, value_dim, query_key_dim_total, value_dim_total)])

     # Build the network:
    def forward(self, inputs, t_measure=None):
        
        layer = self.layers[0]
        out, output_dict = layer(inputs, inputs, inputs, t_measure)

        return out, output_dict
    
##########################################################################################
##########################################################################################

class SimpleAttention_Net(torch.nn.Module):
    """
    Neural network using vanilla attention
    """

    # Initialize the network and specify input/output dimensions:
    def __init__(self, input_dim, hidden_dim, num_heads, args):
        super(SimpleAttention_Net, self).__init__()

        self.a1 = MultiHeadAttentionLayer(input_dim, hidden_dim, num_heads, args)
        self.a2 = MultiHeadAttentionLayer(input_dim, hidden_dim, num_heads, args)
        
        self.l1 = nn.Linear(input_dim, input_dim)
        self.l2 = nn.Linear(input_dim, input_dim)
        self.l3 = nn.Linear(input_dim, input_dim)
        
        self.ReLU = nn.ReLU()
        
     # Build the network:
    def forward(self, inputs):
        
        inputs = self.l1(inputs)
        inputs = self.ReLU(inputs)

        out1, attn = self.a1(inputs, inputs, inputs)
        
        out1 = out1 + inputs

        out1 = self.l2(out1)
        out1 = self.ReLU(out1)
        
        out2, attn = self.a2(out1, out1, out1)
        
        out2 = out2 + out1
        out2 = self.l3(out2)
        
        return out2, attn
    
##########################################################################################
##########################################################################################

# class SimpleComplexAttention_Net(torch.nn.Module):
#     """
#     Neural network using complex attention, but passing only real values in and out.
#     The rest of the network uses real values.
#     """

#     # Initialize the network and specify input/output dimensions:
#     def __init__(self, input_dim, hidden_dim, num_heads, args):
#         super(SimpleComplexRealAttention_Net, self).__init__()

#         self.a1 = MultiHeadComplexAttentionLayer(input_dim, hidden_dim, num_heads, args)
#         self.a2 = MultiHeadComplexAttentionLayer(input_dim, hidden_dim, num_heads, args)
        
#         self.l1 = nn.Linear(input_dim, input_dim)
#         self.l2 = nn.Linear(input_dim, input_dim)
#         self.l3 = nn.Linear(input_dim, input_dim)
        
#         self.ReLU = nn.ReLU()
        
#      # Build the network:
#     def forward(self, inputs):
        
#         inputs = self.l1(inputs)
#         inputs = self.ReLU(inputs)

#         out1, attn = self.a1(inputs, inputs, inputs)
        
#         out1 = out1 + inputs

#         out1 = self.l2(out1)
#         out1 = self.ReLU(out1)
        
#         out2, attn = self.a2(out1, out1, out1)
        
#         out2 = out2 + out1
#         out2 = self.l3(out2)
        
#         return out2, attn
    
##########################################################################################
##########################################################################################

class TransformerNetwork(nn.Module):
    """
    A network composed of multiple stacked Transformer blocks.

    Each block consists of attention, feedforward layers, and residual connections.
    """

    def __init__(self, input_dim, qkv_dim, hidden_dim, num_heads, args, num_blocks=3, Norm=nn.RMSNorm):
        """
        Args:
            input_dim (int): Input and output dimensionality for transformer blocks.
            hidden_dim (int): Hidden dimensionality in the feedforward layers.
            args: Additional arguments passed to each TransformerBlock and AttentionLayer.
            num_blocks (int): Number of stacked Transformer blocks.
        """
        super().__init__()
        
        self.args = args

        # Initial linear layer
        self.input_proj = nn.Linear(input_dim, input_dim)

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(input_dim, qkv_dim, hidden_dim, num_heads, args, Norm)
            for _ in range(num_blocks)
        ])

        # Optional final LayerNorm
        if Norm == None:
            self.final_norm = None
        else:
            self.final_norm = Norm(input_dim)

        self.output_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        """
        Forward pass through the Transformer network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            out (torch.Tensor): Final output (possibly projected).
            attn_list (list): List of attention weights from each block.
        """
        attn_list = []

        x = self.input_proj(x)

        for block in self.blocks:
            x, attn = block(x)
            attn_list.append(attn)

        # Apply final normalization
        if self.final_norm == None:
            pass
        else:
            x = self.final_norm(x)

        x = self.output_proj(x)

        return x, attn_list

##########################################################################################
##########################################################################################

# class ComplexTransformerNetwork(nn.Module):
#     """
#     A network composed of a stack of complex-real Transformer blocks, i.e. using complex-valued attention internally,
#     but operating on real-valued inputs and outputs. Optionally ends with a projection layer.

#     Attributes:
#         blocks (nn.ModuleList): A list of ComplexRealTransformerBlock modules applied sequentially.
#         final_norm: Normalization applied after the stack of blocks.
#         use_output_layer (bool): Whether to apply a final linear projection.
#         output_layer (nn.Linear): Optional linear projection layer if use_output_layer is True.
#     """

#     def __init__(self, input_dim, qkv_dim, hidden_dim, num_heads, args, num_blocks=2, Norm=nn.RMSNorm):
#         """
#         Initializes the transformer network.

#         Args:
#             input_dim (int): Dimensionality of the input and output vectors.
#             hidden_dim (int): Internal dimension used by the attention blocks.
#             args (Namespace): Additional model hyperparameters (e.g., device, config flags).
#             num_blocks (int): Number of stacked Transformer blocks.
#         """
#         super().__init__()
        
#         # Initial linear layer
#         self.input_proj = nn.Linear(input_dim, input_dim)

#         # Stack of Transformer blocks, each using complex-valued attention
#         self.blocks = nn.ModuleList([
#             ComplexRealTransformerBlock(input_dim, qkv_dim, hidden_dim, num_heads, args)
#             for _ in range(num_blocks)
#         ])

#         # Final normalization applied to the output of the last block
#         self.final_norm = Norm(input_dim)

#         # Output linear projection
#         self.output_proj = nn.Linear(input_dim, input_dim)

#     def forward(self, x):
#         """
#         Forward pass through the Transformer network.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

#         Returns:
#             x (torch.Tensor): Output tensor, optionally projected to output_dim.
#             attn_list (list): List of attention weight tensors from each block.
#         """
#         attn_list = []
        
#         x = self.input_proj(x)

#         # Sequentially apply each Transformer block
#         for block in self.blocks:
#             x, attn = block(x)
#             attn_list.append(attn)

#         # Apply final normalization
#         x = self.final_norm(x)

#         # Output projection layer
#         x = self.output_proj(x)

#         return x, attn_list

##########################################################################################
##########################################################################################

class AFATransformerNetwork(nn.Module):
    """
    Adaptive Filter Attention (AFA) Transformer Network.

    Implements a sequence-to-sequence model composed of stacked AFA Transformer blocks,
    which leverage continuous-time SDE dynamics for positional encoding and uncertainty-aware filtering.

    Attributes:
        args (Namespace): Configuration and hyperparameters for the model.
        input_proj (nn.Linear): Initial linear projection of the input features.
        blocks (nn.ModuleList): A stack of AFATransformerBlock modules applied sequentially.
        final_norm (nn.Module | None): Normalization layer (e.g., RMSNorm) applied after the stack of blocks.
        output_proj (nn.Linear): Final linear projection of the processed sequence to the output dimension.
    """

    def __init__(self, args, num_blocks=2, n_heads=1, input_dim=None, query_key_dim=None, value_dim=None, query_key_dim_total=None, value_dim_total=None, hidden_dim=None, Norm=nn.RMSNorm):
        """
        Initializes the AFA Transformer network architecture.

        Args:
            args (Namespace): Global model configuration and hyperparameters.
            num_blocks (int): The number of stacked AFATransformerBlock layers. Defaults to 2.
            n_heads (int): The number of attention heads in the Multi-Head Attention layer. Defaults to 1.
            input_dim (int): Dimensionality of the raw input features (and the residual path). Must be provided.
            query_key_dim (int | None): Dimensionality of query/key vectors per head. Mutually exclusive with query_key_dim_total.
            value_dim (int | None): Dimensionality of value vectors per head. Mutually exclusive with value_dim_total.
            query_key_dim_total (int | None): Total dimensionality of Q/K space across all heads. Mutually exclusive with query_key_dim.
            value_dim_total (int | None): Total dimensionality of V space across all heads. Mutually exclusive with value_dim.
            hidden_dim (int | None): Internal expansion dimension for the Feed-Forward Network (FFN).
            Norm (Type[nn.Module]): Normalization class (e.g., nn.RMSNorm, nn.LayerNorm). Can be None.
        """
        
        super().__init__()
        
        self.args = args
        
        # Set query_key and value dims, depending on whether user provided total dims, or head dims
        query_key_dim, value_dim, query_key_dim_total, value_dim_total = resolve_multihead_dims(n_heads, query_key_dim, value_dim, query_key_dim_total, value_dim_total)
        
        # Initial linear layer
        self.input_proj = nn.Linear(input_dim, input_dim)

        # Stack of Transformer blocks, each using complex-valued attention
        self.blocks = nn.ModuleList([      
            AFATransformerBlock(args, n_heads, input_dim, query_key_dim, value_dim, query_key_dim_total, value_dim_total, hidden_dim, Norm)
            for _ in range(num_blocks)
        ])

        # Final normalization applied to the output of the last block
        if Norm == None:
            self.final_norm = None
        else:
            self.final_norm = Norm(input_dim)

        # Output linear projection
        self.output_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x, t_measure=None):
        """
        Forward pass through the AFA Transformer network stack.

        Args:
            x (torch.Tensor): Input sequence tensor, shape (batch_size, seq_len, input_dim).
            t_measure (torch.Tensor): Time of measurement for each token, shape (batch_size, seq_len) or (seq_len,). 
                                      Used to compute time lag (Delta t) for SDE propagation.

        Returns:
            out (torch.Tensor): The final output tensor after processing and output projection,
                                shape (batch_size, seq_len, input_dim).
            output_dict (dict): Dictionary containing outputs from the last AFA block (e.g., final attention weights, norms, etc.).
        """
        
        x = self.input_proj(x) # Input projection

        # Sequentially apply each Transformer block
        for block in self.blocks:
            x, output_dict = block(x, t_measure)

        # Apply final normalization
        if self.final_norm == None:
            pass
        else:
            x = self.final_norm(x)

        out = self.output_proj(x) # Output projection

        return out, output_dict

##########################################################################################
##########################################################################################
