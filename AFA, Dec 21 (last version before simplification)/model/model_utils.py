import numpy as np
import torch
import torch.nn.functional as F

##########################################################################################
##########################################################################################

def compute_lambda(lambda_real_input, lambda_imag_input, args, scale_c=10, scale_r=1, scale_i=1, epsilon=1e-5):
    """
    Compute complex conjugate eigenvalue pairs for each attention head.

    Args:
        lambda_real_in: Tensor of shape [n_heads, 1]
        lambda_imag_in: Tensor of shape [n_heads, d_v_head // 2]
        args: must contain dt
    Returns:
        lambda_h: [2, d_v_head, n_heads, 1] (real/imag, eigenvalue, head, 1)
    """
    
    device = lambda_real_input.device
    
#     scale_r = scale_c / (2*args.tf) # Normalize by time interval
#     scale_i = scale_c * 2*np.pi / (2*args.tf) # Normalize by time interval
    scale_r = scale_c / (2*args.seq_len) # Normalize by sequence length
    scale_i = scale_c * 2*np.pi / (2*args.seq_len) # Normalize by sequence length

    n_heads, d_half = lambda_imag_input.shape
    d_v_head = 2 * d_half

    # Compute bounded decay rate for real part
#     if args.allow_unstable_eigenvals == 0:
#         lambda_real = (torch.sigmoid(lambda_real_input) - 1.0) * scale_r - epsilon # Bounded above and below
#     else:
#         lambda_real = (torch.sigmoid(lambda_real_input) - 0.9) * scale_r # Allow positive values
#         print('Allowing unstable eigenvalues (real(lambda) > 0)')
#     lambda_real = -F.softplus(-lambda_real_input) * scale_r - epsilon # Bounded above; Unbounded below
    lambda_real = (torch.sigmoid(lambda_real_input) - 1.0) * scale_r - epsilon # Bounded above and below

    if args.lambda_real_zero == 1:
        lambda_real = lambda_real*0    
    
    # Expand real parts to match imag
    lambda_real_expanded = lambda_real.expand(d_half, -1)  # [d_half, n_heads]
#     if args.isotropic_afa == 1:
#         lambda_real_expanded = lambda_real.expand(d_half, -1)  # [d_half, n_heads]
#     else:
#         lambda_real_expanded = lambda_real.permute(1,0)
        
    # Create interleaved real/imag conjugate pairs
    real_parts = lambda_real_expanded.repeat_interleave(2, dim=0)                # [d_v_head, n_heads]
    imag_parts = torch.stack([lambda_imag_input, -lambda_imag_input], dim=2) * scale_i   # [d_half, 2, n_heads]
    imag_parts = imag_parts.reshape(n_heads, d_v_head).T                          # interleave imag/-imag

    # Stack real and imag into shape [2, d_v_head, n_heads, 1]
    lambda_h = torch.stack([real_parts, imag_parts], dim=0).unsqueeze(1).unsqueeze(-1)  # [2, 1, d_v_head, n_heads, 1]

    return lambda_h

##########################################################################################
##########################################################################################

def resolve_multihead_dims(n_heads, query_key_dim=None, value_dim=None, query_key_dim_total=None, value_dim_total=None):
    """
    Resolve per-head and total key/value dimensions for multihead attention.

    You must supply either:
        - query_key_dim (per-head) or query_key_dim_total (total), but not both, and
        - value_dim (per-head) or value_dim_total (total), but not both.

    Returns:
        dict with keys:
            query_key_dim, query_key_dim_total, value_dim, value_dim_total
    """
    # Validate head count
    if n_heads <= 0 or not isinstance(n_heads, int):
        raise ValueError(f"n_heads must be a positive integer, got {n_heads}")

    # Query/key dims
    if (query_key_dim is not None) and (query_key_dim_total is not None):
        raise ValueError("Specify either query_key_dim or query_key_dim_total, not both.")
    if (query_key_dim is None) and (query_key_dim_total is None):
        raise ValueError("Must specify one of query_key_dim or query_key_dim_total.")

    if query_key_dim is not None:
        query_key_dim_total = n_heads * query_key_dim
    else:
        if query_key_dim_total % n_heads != 0:
            raise ValueError(f"query_key_dim_total={query_key_dim_total} is not divisible by n_heads={n_heads}")
        query_key_dim = query_key_dim_total // n_heads

    # Value dims
    if (value_dim is not None) and (value_dim_total is not None):
        raise ValueError("Specify either value_dim or value_dim_total, not both.")
    if (value_dim is None) and (value_dim_total is None):
        raise ValueError("Must specify one of value_dim or value_dim_total.")

    if value_dim is not None:
        value_dim_total = n_heads * value_dim
    else:
        if value_dim_total % n_heads != 0:
            raise ValueError(f"value_dim_total={value_dim_total} is not divisible by n_heads={n_heads}")
        value_dim = value_dim_total // n_heads

    return query_key_dim, value_dim, query_key_dim_total, value_dim_total
