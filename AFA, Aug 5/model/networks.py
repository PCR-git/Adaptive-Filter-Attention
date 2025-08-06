import numpy as np
import torch
import torch.nn as nn

from model import ModReLU
from model import init_complexlinear, init_complex_matrix
from model import ComplexLinearLayer, AttentionLayer, ComplexAttentionLayer
from model import GatedSequenceNorm, GatedLayerNorm, GatedComplexSequenceNorm, GatedComplexLayerNorm
from model import TransformerBlock, ComplexTransformerBlock, ComplexRealTransformerBlock
from model import FullPrecisionAttentionBlock, FullPrecisionAttentionBlock_Nlayer
from model import RealPositionalEncoding, ComplexPositionalEncoding, LearnedRealPositionalEncoding, LearnedComplexPositionalEncoding
from model import AFATransformerBlock
from model import SimplifiedPrecisionAttentionBlock, MultiheadSimplifiedPrecisionAttentionBlock

##########################################################################################
##########################################################################################

class Attention_1layer(nn.Module):
    """
    Neural network with a single attention layer
    """

    # Initialize the network and specify input/output dimensions:
    def __init__(self, input_dim, hidden_dim, args):
        super(Attention_1layer, self).__init__()

        self.a1 = AttentionLayer(input_dim, hidden_dim, args)
        
     # Build the network:
    def forward(self, inputs):

        out, attn = self.a1(inputs, inputs, inputs)
        
        return out, attn
    
##########################################################################################
##########################################################################################

class ComplexAttention_1layer(nn.Module):
    """
    Neural network with a single attention layer
    """

    # Initialize the network and specify input/output dimensions:
    def __init__(self, input_dim, hidden_dim, args):
        super(ComplexAttention_1layer, self).__init__()

        self.a1 = ComplexAttentionLayer(input_dim, hidden_dim, args)
        
     # Build the network:
    def forward(self, inputs):

        out, attn = self.a1(inputs, inputs, inputs)
        
        return out, attn
    
##########################################################################################
##########################################################################################

class SimpleAttention_Net(torch.nn.Module):
    """
    Neural network using vanilla attention
    """

    # Initialize the network and specify input/output dimensions:
    def __init__(self, input_dim, hidden_dim, args):
        super(SimpleAttention_Net, self).__init__()

        self.a1 = AttentionLayer(input_dim, hidden_dim, args)
        self.a2 = AttentionLayer(input_dim, hidden_dim, args)
        
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
 
class SimpleComplexAttention_Net(torch.nn.Module):
    """
    Neural network using complex-valued attention and complex values throughout.
    """

    # Initialize the network and specify input/output dimensions:
    def __init__(self, input_dim, hidden_dim, args):
        super(SimpleComplexAttention_Net, self).__init__()

        self.a1 = ComplexAttentionLayer(input_dim, hidden_dim, args)
        self.a2 = ComplexAttentionLayer(input_dim, hidden_dim, args)
        
        self.l1 = ComplexLinearLayer(input_dim, input_dim)
        self.l2 = ComplexLinearLayer(input_dim, input_dim)
        self.l3 = ComplexLinearLayer(input_dim, input_dim)
        
        W1, b1 = init_complex_matrix(args.d_e, args.d_v, bias=True)
        W2, b2 = init_complex_matrix(args.d_e, args.d_v, bias=True)
        W3, b3 = init_complex_matrix(args.d_e, args.d_v, bias=True)
        init_complexlinear(self.l1, W1, b1)
        init_complexlinear(self.l2, W1, b2)
        init_complexlinear(self.l3, W1, b3)
        
#         self.ReLU = nn.ReLU()
        self.ModReLU = ModReLU()
        
     # Build the network:
    def forward(self, inputs):
        
        inputs = self.l1(inputs)
#         inputs = self.ReLU(inputs)
        inputs = self.ModReLU(inputs)

        out1, attn = self.a1(inputs, inputs, inputs)
        
        out1 = out1 + inputs

        out1 = self.l2(out1)
#         out1 = self.ReLU(out1)
        out1 = self.ModReLU(out1)
        
        out2, attn = self.a2(out1, out1, out1)
        
        out2 = out2 + out1
        out2 = self.l3(out2)
        
        return out2, attn
    
##########################################################################################
##########################################################################################

class SimpleComplexRealAttention_Net(torch.nn.Module):
    """
    Neural network using complex attention, but passing only real values in and out.
    The rest of the network uses real values.
    """

    # Initialize the network and specify input/output dimensions:
    def __init__(self, input_dim, hidden_dim, args):
        super(SimpleComplexRealAttention_Net, self).__init__()

        self.a1 = ComplexAttentionLayer(input_dim, hidden_dim, args)
        self.a2 = ComplexAttentionLayer(input_dim, hidden_dim, args)
        
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

class TransformerNetwork(nn.Module):
    """
    A network composed of multiple stacked Transformer blocks.

    Each block consists of attention, feedforward layers, and residual connections.
    """

    def __init__(self, input_dim, hidden_dim, args, num_blocks=2, Norm=nn.LayerNorm):
        """
        Args:
            input_dim (int): Input and output dimensionality for transformer blocks.
            hidden_dim (int): Hidden dimensionality in the feedforward layers.
            args: Additional arguments passed to each TransformerBlock and AttentionLayer.
            num_blocks (int): Number of stacked Transformer blocks.
        """
        super().__init__()

        # Initial linear layer
        self.input_proj = nn.Linear(input_dim, input_dim)

        # Positional encoding
        self.pos_encoder = RealPositionalEncoding(input_dim)

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(input_dim, hidden_dim, args)
            for _ in range(num_blocks)
        ])

        # Optional final LayerNorm
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
        x = self.pos_encoder(x)

        for block in self.blocks:
            x, attn = block(x)
            attn_list.append(attn)

        x = self.final_norm(x)

        x = self.output_proj(x)

        return x, attn_list

##########################################################################################
##########################################################################################
    
class ComplexTransformerNetwork(nn.Module):
    """
    A network composed of a stack of complex Transformer blocks, optionally ending with a projection layer.
    """

    def __init__(self, input_dim, hidden_dim, args, num_blocks=2, Norm=GatedComplexLayerNorm):
        """
        Args:
            input_dim (int): Input/output dimensionality.
            hidden_dim (int): Hidden dimensionality in feedforward layers.
            args: Arguments for the attention and normalization modules.
            num_blocks (int): Number of stacked Transformer blocks.
        """
        super().__init__()

        # Initial complex linear projection
        self.input_proj = ComplexLinearLayer(input_dim, input_dim)

        # Complex positional encoding
        self.pos_encoder = ComplexPositionalEncoding(seq_len=5000, embed_dim=input_dim)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            ComplexTransformerBlock(input_dim, hidden_dim, args)
            for _ in range(num_blocks)
        ])

        # Final norm layer
        self.final_norm = Norm(input_dim)

        # Output projection layer
        self.output_proj = ComplexLinearLayer(input_dim, input_dim)  # Or other target dim

    def forward(self, x):
        """
        Forward pass through the transformer stack.

        Args:
            x (torch.Tensor): Complex input of shape (batch, seq_len, input_dim)

        Returns:
            out (torch.Tensor): Final output tensor.
            attn_list (list): List of attention maps from each block.
        """
        attn_list = []

        # Reshape to (B, 2, L, D) if needed
        if x.ndim == 3:
            x = x.unsqueeze(1).repeat(1, 2, 1, 1)  # (B, 2, L, D), real + imaginary

        x = self.input_proj(x)
        x = self.pos_encoder(x)

        for block in self.blocks:
            x, attn = block(x)
            attn_list.append(attn)

        x = self.final_norm(x)
        x = self.output_proj(x)

        return x, attn_list
    
##########################################################################################
##########################################################################################

class ComplexRealTransformerNetwork(nn.Module):
    """
    A network composed of a stack of complex-real Transformer blocks, i.e. using complex-valued attention internally,
    but operating on real-valued inputs and outputs. Optionally ends with a projection layer.

    Attributes:
        blocks (nn.ModuleList): A list of ComplexRealTransformerBlock modules applied sequentially.
        final_norm (nn.LayerNorm): Layer normalization applied after the stack of blocks.
        use_output_layer (bool): Whether to apply a final linear projection.
        output_layer (nn.Linear): Optional linear projection layer if use_output_layer is True.
    """

    def __init__(self, input_dim, hidden_dim, args, num_blocks=2, Norm=nn.LayerNorm):
        """
        Initializes the transformer network.

        Args:
            input_dim (int): Dimensionality of the input and output vectors.
            hidden_dim (int): Internal dimension used by the attention blocks.
            args (Namespace): Additional model hyperparameters (e.g., device, config flags).
            num_blocks (int): Number of stacked Transformer blocks.
        """
        super().__init__()
        
        # Initial linear layer
        self.input_proj = nn.Linear(input_dim, input_dim)

        # Positional encoding
        self.pos_encoder = RealPositionalEncoding(input_dim)

        # Stack of Transformer blocks, each using complex-valued attention
        self.blocks = nn.ModuleList([
            ComplexRealTransformerBlock(input_dim, hidden_dim, args)
            for _ in range(num_blocks)
        ])

        # Final normalization applied to the output of the last block
        self.final_norm = Norm(input_dim)

        # Output linear projection
        self.output_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        """
        Forward pass through the Transformer network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            x (torch.Tensor): Output tensor, optionally projected to output_dim.
            attn_list (list): List of attention weight tensors from each block.
        """
        attn_list = []
        
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        # Sequentially apply each Transformer block
        for block in self.blocks:
            x, attn = block(x)
            attn_list.append(attn)

        # Apply final normalization
        x = self.final_norm(x)

        # Output projection layer
        x = self.output_proj(x)

        return x, attn_list
    
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
    def forward(self, inputs, t_v):
        
        layer = self.layers[0]
        est, out, Q_ij, Z_ij_hat_all, lambda_h = layer(inputs, inputs, inputs, t_v)

        return est, out, Q_ij, Z_ij_hat_all, lambda_h
    
##########################################################################################
##########################################################################################

class AFA_Nlayer(nn.Module):
    """
    Neural network with an N-layer AFA block
    """

    # Initialize the network and specify input/output dimensions:
    def __init__(self, args):
        super(AFA_Nlayer, self).__init__()

        self.layers = nn.ModuleList([FullPrecisionAttentionBlock_Nlayer(args.d_e, args.d_k, args.d_v, args.num_layers, args)])

     # Build the network:
    def forward(self, inputs, t_v):
        
        layer = self.layers[0]
        est, out, Q_ij, Z_ij_hat_all, lambda_h = layer(inputs, inputs, inputs, t_v)

        return est, out, Q_ij, Z_ij_hat_all, lambda_h
    
##########################################################################################
##########################################################################################

# class PrecisionNet(nn.Module):
#     """
#     Neural network, alternating precision attention and fully-connected blocks
#     """

#     # Initialize the network and specify input/output dimensions:
#     def __init__(self, args):
#         super(PrecisionNet, self).__init__()

#         self.fc1 = nn.Linear(args.d_e, args.d_e)
#         self.fc2 = nn.Linear(args.d_e, args.d_e)

#         # Xavier initialization
#         init.xavier_normal_(self.fc1.weight)
#         init.zeros_(self.fc1.bias)
#         init.xavier_normal_(self.fc2.weight)
#         init.zeros_(self.fc2.bias)

#         self.a1 = FullPrecisionAttentionBlock(args.d_e, args)
#         self.a2 = FullPrecisionAttentionBlock(args.d_e, args)

#         self.ReLU = nn.ReLU()
# #         self.GELU = nn.GELU()

#      # Build the network:
#     def forward(self, inputs, t_v):

#         x1 = self.fc1(inputs.squeeze(-1)).unsqueeze(-1)
#         x2 = self.GELU(x1) + x1
#         _, x3, _, _ = self.a1(x2, x2, x2, t_v)
#         x4 = self.fc2(x3.squeeze(-1)).unsqueeze(-1)
#         x5 = self.GELU(x4) + x4
#         est, out, Q_ij, Z_ij_hat_all, lambda_h = self.a2(x5, x5, x5, t_v)

#         return est, out, Q_ij, Z_ij_hat_all, lambda_h
    
##########################################################################################
##########################################################################################

class AFATransformerNetwork(nn.Module):
    """
    AFA Transformer Network
    
    Attributes:
        blocks (nn.ModuleList): A list of ComplexRealTransformerBlock modules applied sequentially.
        final_norm (nn.LayerNorm): Layer normalization applied after the stack of blocks.
        use_output_layer (bool): Whether to apply a final linear projection.
        output_layer (nn.Linear): Optional linear projection layer if use_output_layer is True.
    """

    def __init__(self, input_dim, query_key_dim, value_dim, args, num_inner_layers=2, num_blocks=2, Norm=nn.LayerNorm):
        """
        Initializes the transformer network.

        Args:
            input_dim (int): Dimensionality of the input and output vectors.
            hidden_dim (int): Internal dimension used by the attention blocks.
            args (Namespace): Additional model hyperparameters (e.g., device, config flags).
            num_blocks (int): Number of stacked Transformer blocks.
        """
        super().__init__()
        
        # Initial linear layer
        self.input_proj = nn.Linear(input_dim, input_dim)

        # Positional encoding
        self.pos_encoder = RealPositionalEncoding(input_dim)

        # Stack of Transformer blocks, each using complex-valued attention
        self.blocks = nn.ModuleList([      
            AFATransformerBlock(input_dim, query_key_dim, value_dim, num_inner_layers, args, Norm=Norm)
            for _ in range(num_blocks)
        ])

        # Final normalization applied to the output of the last block
        if Norm == None:
            self.final_norm = None
        else:
            self.final_norm = Norm(input_dim)

        # Output linear projection
        self.output_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x, t_v):
        """
        Forward pass through the Transformer network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            x (torch.Tensor): Output tensor, optionally projected to output_dim.
            attn_list (list): List of attention weight tensors from each block.
        """
#         attn_list = []
        
        x = self.input_proj(x) # Input projection
        x = self.pos_encoder(x) # Add pos embed

        # Sequentially apply each Transformer block
        for block in self.blocks:
            est, x, attn_weight, Z_ij_hat, lambda_h = block(x, t_v)
#             attn_list.append(attn_weight)

        # Apply final normalization
        if self.final_norm == None:
            pass
        else:
            x = self.final_norm(x)

        out = self.output_proj(x) # Output projection

        return est, out, attn_weight, Z_ij_hat, lambda_h
    
##########################################################################################
##########################################################################################

class SimplifiedAFA_1layer(nn.Module):
    """
    Neural network with a single simplified precision attention block
    """

    # Initialize the network and specify input/output dimensions:
    def __init__(self, args):
        super(SimplifiedAFA_1layer, self).__init__()

        self.layers = nn.ModuleList([SimplifiedPrecisionAttentionBlock(args.d_e, args.d_k, args.d_v, args)])

     # Build the network:
    def forward(self, inputs, t_v):
        
        layer = self.layers[0]
        est, out, Q_ij, Z_ij_hat_all, lambda_h = layer(inputs, inputs, inputs, t_v)

        return est, out, Q_ij, Z_ij_hat_all, lambda_h
    
    
##########################################################################################
##########################################################################################

class MultiheadSimplifiedAFA_1layer(nn.Module):
    """
    Neural network with a single multihead simplified precision attention block
    """

    # Initialize the network and specify input/output dimensions:
    def __init__(self, args, seq_len, n_heads, input_dim, query_key_dim=None, value_dim=None, query_key_dim_total=None, value_dim_total=None):
        super(MultiheadSimplifiedAFA_1layer, self).__init__()

        self.layers = nn.ModuleList([MultiheadSimplifiedPrecisionAttentionBlock(args, seq_len, n_heads, input_dim, query_key_dim, value_dim, query_key_dim_total, value_dim_total)])

     # Build the network:
    def forward(self, inputs, t_v):
        
        layer = self.layers[0]
        est, out, Q_ij, Z_ij_hat_all, lambda_h = layer(inputs, inputs, inputs, t_v)

        return est, out, Q_ij, Z_ij_hat_all, lambda_h
    
    