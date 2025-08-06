import numpy as np
import torch
import torch.nn.functional as F

from utils import complex_exp_v2, batched_complex_hadamard_full, batched_complex_matmul_full

##########################################################################################
##########################################################################################

# def compute_lambda_h_v0(lambda1):
#     """
#     Construct lambda_h on the fly from lambda1
    
#     In this function, we relax the constraint that the eigenvalues come in complex conjugate pairs,
#     and only require that they be nonpositive (to ensure stable dynamics)
#     """
    
#     lambda1_0 = -torch.abs(lambda1[0])
#     lambda1_1 = lambda1[1]
#     lambda_h = torch.stack((lambda1_0, lambda1_1))

#     return lambda_h.unsqueeze(1)

##########################################################################################
##########################################################################################

# def compute_lambda_h_v1(lambda1):
#     """
#     Construct lambda_h on the fly from lambda1
    
#     (lambda1 is a torch parameter tensor of length d_e/2, where d_e is the embedding dimension)
#     lambda_h is the full set of eigenvalues, which are nonpositive and come in complex-conjugate pairs
#     (the latter constraint forces the full state transition matrix to be real, provided W_v and W_p are unitary
#      and W_v = W_p^* ie a complex conjugate transpose pair)
#     """
    
#     lambda1_0 = -torch.abs(lambda1[0])
#     lambda1_1 = lambda1[1]
#     lambda2_0 = -torch.abs(lambda1[0])
#     lambda2_1 = -lambda1[1]
#     lambda1 = torch.stack((lambda1_0, lambda1_1))
#     lambda2 = torch.stack((lambda2_0, lambda2_1))
#     # lambda_h = torch.cat((lambda1, lambda2), dim=1).unsqueeze(1)
#     B, N, D = lambda1.shape
#     lambda_h = torch.stack((lambda1, lambda2), dim=2)  # shape: (B, N, 2, D)
#     lambda_h = lambda_h.view(B, 2*N, D).unsqueeze(1)

#     return lambda_h

##########################################################################################
##########################################################################################

def compute_lambda_h(lambder,args,scale_c=10,odd_embed_size=0, min_dt_decay=10e-3):
    """
    Construct lambda_h on the fly from lambda1
    
    (lambda1 is a torch parameter tensor of length d_e/2, where d_e is the embedding dimension)
    lambda_h is the full set of eigenvalues, which are nonpositive and come in complex-conjugate pairs
    (the latter constraint forces the full state transition matrix to be real, provided W_v and W_p are unitary
     and W_v = W_p^* ie a complex conjugate transpose pair)
    """
    
#     mag = torch.abs(lambder[0])
#     mag = scale * torch.abs(lambder[0])
#     mag = scale * lambder[0]**2

#     mag = scale * torch.sigmoid(lambder[0])
#     mag = scale * lambder[0]**2
#     mag = scale * F.softplus(lambder[0])

    scale_r = scale_c / (2*args.tf) # Normalize by time interval
    scale_i = scale_c * 2*np.pi / (2*args.tf) # Normalize by time interval
#     scale_r = scale_c / 2
#     scale_i = scale_c * np.pi

    # Ensure the eigenvalues have negative real parts
    lambda_r = -torch.abs(lambder[0]) * scale_r
#     lambda1_0 = (1 - torch.abs(lambder[0])) * scale_r
#     lambda2_0 = (1 - torch.abs(lambder[0])) * scale_r

    ########################################
    # Set lower bound to outlaw extremely fast modes
    # This is probably not necessary. Aimed at preventing a pathological case just in case.
    min_lambda_real_bound = np.log(min_dt_decay) / args.dt # Modes can decay no faster than to 0.1% (10e−3 ) of their initial value within one dt

    # Apply soft lower bound using F.softplus
    # This will smoothly approach min_lambda_real_bound if unclamped_lambda_real is much lower,
    # and approach unclamped_lambda_real if it's above min_lambda_real_bound (up to 0).
    lambda_r = min_lambda_real_bound + F.softplus(lambda_r - min_lambda_real_bound)
    ########################################
    
    lambda1_i =  lambder[1] * scale_i
    lambda2_i = -lambder[1] * scale_i
    lambda1 = torch.stack((lambda_r, lambda1_i))
    lambda2 = torch.stack((lambda_r, lambda2_i))
    # lambda_h = torch.cat((lambda1, lambda2), dim=1).unsqueeze(1)
    B, N, D = lambder.shape
    lambda_h = torch.stack((lambda1, lambda2), dim=2)  # shape: (B, N, 2, D)
#     lambda_h = lambda_h.view(B, 2*N, D).unsqueeze(1)
    lambda_h = lambda_h.view(2, 2*N, 1).unsqueeze(1)
    
    # If using odd embedding size, chop off last entry
    if odd_embed_size == 1:
        lambda_h = lambda_h[:,:,0:-1,:]

    return lambda_h

##########################################################################################
##########################################################################################

def compute_lambda_shared(lambda1_real,lambda1_imag,args,scale_c=10,odd_embed_size=0, min_dt_decay=1e-3):
    """

    """
#     scale_r = scale_c / (2*args.tf) # Normalize by time interval
#     scale_i = scale_c * 2*np.pi / (2*args.tf) # Normalize by time interval
    scale_r = 1
    scale_i = 1

    lambda_real = -torch.abs(lambda1_real) * scale_r
    min_lambda_real_bound = np.log(min_dt_decay) / args.dt # Modes can decay no faster than to 0.1% (10e−3 ) of their initial value within one dt
    lambda_real = min_lambda_real_bound + F.softplus(lambda_real - min_lambda_real_bound)

    lambda1_i =  lambda1_imag * scale_i
    lambda2_i = -lambda1_imag * scale_i

    lambda_r = torch.ones_like(lambda1_i) * lambda_real

    lambda1 = torch.stack((lambda_r, lambda1_i))
    lambda2 = torch.stack((lambda_r, lambda2_i))

    N = lambda1_imag.size()[0]

    lambda_h = torch.stack((lambda1, lambda2), dim=2)
    lambda_h = lambda_h.view(2, 2*N, 1).unsqueeze(1)

    # If using odd embedding size, chop off last entry
    if odd_embed_size == 1:
        lambda_h = lambda_h[:,:,0:-1,:]
        
    return lambda_h

##########################################################################################
##########################################################################################

def compute_lambda_shared_multihead(lambda_real_input, lambda_imag_input, args, scale_r=1, scale_i=1, min_dt_decay=1e-3):
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

    n_heads, d_half = lambda_imag_input.shape
    d_v_head = 2 * d_half

    # Compute bounded decay rate for real part
    min_lambda_real_bound = torch.tensor(np.log(min_dt_decay) / args.dt, device=device)
    lambda_real = -torch.abs(lambda_real_input) * scale_r
    lambda_real = min_lambda_real_bound + F.softplus(lambda_real - min_lambda_real_bound)  # [n_heads]

    # Expand real parts to match imag
    lambda_real_expanded = lambda_real.expand(d_half, -1)  # [n_heads, d_half]

    lambda_real_expanded.size()
    # Create interleaved real/imag conjugate pairs
    real_parts = lambda_real_expanded.repeat_interleave(2, dim=0)                # [d_v_head, n_heads]
    imag_parts = torch.stack([lambda_imag_input, -lambda_imag_input], dim=2) * scale_i   # [d_half, 2, n_heads]
    imag_parts = imag_parts.reshape(n_heads, d_v_head).T                          # interleave imag/-imag

    # Stack real and imag into shape [2, d_v_head, n_heads, 1]
    lambda_h = torch.stack([real_parts, imag_parts], dim=0).unsqueeze(1).unsqueeze(-1)  # [2, 1, d_v_head, n_heads, 1]

    return lambda_h

##########################################################################################
##########################################################################################

def predict_multiple_steps(lambda_h, est_eigenbasis, W_p, t_forward):
    """
    Given a sequence of time steps t_forward, make multiple future predictions 
    """

    mag_f, phase_f = complex_exp_v2(lambda_h*t_forward.unsqueeze(0).unsqueeze(2).unsqueeze(3))  # Forward
    mat_exp_f = mag_f*phase_f
    preds_p = batched_complex_hadamard_full(mat_exp_f.unsqueeze(0), est_eigenbasis[:,:,-1,:,:].unsqueeze(2))
    preds = batched_complex_matmul_full(W_p.unsqueeze(0), preds_p)
    
    return preds

##########################################################################################
##########################################################################################

# def get_complex_weights(module, layer_name):
#     """
#     Extracts and stacks the real and imaginary weights from a ComplexLinear layer.

#     Parameters:
#         module (nn.Module): The module containing the layers.
#         name (str): Name of the ComplexLinear layer (e.g., 'W_p').

#     Returns:
#         Complex weight tensor of shape (2, out_dim, in_dim).
#     """
    
#     layer = getattr(module, layer_name)

#     W = torch.stack([layer.real.weight, layer.imag.weight], dim=0)

#     return W

def get_complex_weights(module, layer_name):
    """
    Extracts and stacks the real and imaginary weights from a ComplexLinearLayer
    or ComplexLinearHermitianLayer.

    Parameters:
        module (nn.Module): The module containing the layers.
        layer_name (str): Name of the ComplexLinear layer (e.g., 'W_p').

    Returns:
        Complex weight tensor of shape (2, out_dim, in_dim).
    """

    layer = getattr(module, layer_name)

    try:
        # Regular ComplexLinearLayer
        real_weight = layer.real.weight
        imag_weight = layer.imag.weight
    except:
        # Unwrap source layer weights
        real_weight = layer.source.real.weight.T
        imag_weight = -layer.source.imag.weight.T

    W = torch.stack([real_weight, imag_weight], dim=0)

    return W

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