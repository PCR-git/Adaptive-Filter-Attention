
import numpy as np
import torch

from utils import complex_exp_v2, batched_complex_hadamard, batched_complex_hadamard_full

##########################################################################################
##########################################################################################

def _split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Splits the input tensor into multiple heads and prepares for attention.
    (batch_size, 2, seq_len, embed_dim) -> (batch_size * num_heads, 2, seq_len, head_dim)
    """
    seq_len = x.size(2)
    # Reshape to (batch_size, 2, seq_len, num_heads, head_dim)
    x = x.view(batch_size, 2, seq_len, self.num_heads, self.head_dim)
    # Permute to (batch_size, num_heads, seq_len, head_dim)
    x = x.permute(0, 3, 1, 2, 4)
    # Reshape to (batch_size * num_heads, 2, seq_len, head_dim) for batched attention
    return x.reshape(batch_size * self.num_heads, 2, seq_len, self.head_dim)

##########################################################################################
##########################################################################################

def _combine_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Combines multiple attention heads back into a single tensor.
    (batch_size * num_heads, 2, seq_len, head_dim) -> (batch_size, 2, seq_len, embed_dim)
    """
    seq_len = x.size(2)
    # Reshape to (batch_size, num_heads, 2, seq_len, head_dim)
    x = x.view(batch_size, self.num_heads, 2, seq_len, self.head_dim)
    # Permute to (batch_size, 2, seq_len, num_heads, head_dim)
    x = x.permute(0, 2, 3, 1, 4)
    # # Reshape to (batch_size, seq_len, embed_dim)
    return x.reshape(batch_size, 2, seq_len, self.d_e)

##########################################################################################
##########################################################################################



