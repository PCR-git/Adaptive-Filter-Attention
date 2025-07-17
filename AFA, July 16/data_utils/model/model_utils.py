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

    scale_r = scale_c / (2*args.tf)
    scale_i = scale_c * 2*np.pi / (2*args.tf)
#     scale_i = np.pi

    # Ensure the eigenvalues have negative real parts
    lambda1_r = -torch.abs(lambder[0]) * scale_r
    lambda2_r = -torch.abs(lambder[0]) * scale_r
#     lambda1_0 = (1 - torch.abs(lambder[0])) * scale_r
#     lambda2_0 = (1 - torch.abs(lambder[0])) * scale_r

    ########################################
    # Set lower bound to outlaw extremely fast modes
    # This is probably not necessary. Aimed at preventing a pathological case just in case.
    min_lambda_real_bound = np.log(min_dt_decay) / args.dt # Modes can decay no faster than to 0.1% (10e−3 ) of their initial value within one dt

    # Apply soft lower bound using F.softplus
    # This will smoothly approach min_lambda_real_bound if unclamped_lambda_real is much lower,
    # and approach unclamped_lambda_real if it's above min_lambda_real_bound (up to 0).
    lambda1_r = min_lambda_real_bound + F.softplus(lambda1_r - min_lambda_real_bound)
    lambda2_r = min_lambda_real_bound + F.softplus(lambda2_r - min_lambda_real_bound)
    ########################################
    
    lambda1_i =  lambder[1] * scale_i
    lambda2_i = -lambder[1] * scale_i
    lambda1 = torch.stack((lambda1_r, lambda1_i))
    lambda2 = torch.stack((lambda2_r, lambda2_i))
    # lambda_h = torch.cat((lambda1, lambda2), dim=1).unsqueeze(1)
    B, N, D = lambder.shape
    lambda_h = torch.stack((lambda1, lambda2), dim=2)  # shape: (B, N, 2, D)
    lambda_h = lambda_h.view(B, 2*N, D).unsqueeze(1)
    
    # If using odd embedding size, chop off last entry
    if odd_embed_size == 1:
        lambda_h = lambda_h[:,:,0:-1,:]

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

def get_complex_weights(module, layer_name):
    """
    Extracts and stacks the real and imaginary weights from a ComplexLinear layer.

    Parameters:
        module (nn.Module): The module containing the layers.
        name (str): Name of the ComplexLinear layer (e.g., 'W_p').

    Returns:
        Complex weight tensor of shape (2, out_dim, in_dim).
    """
    layer = getattr(module, layer_name)

    W = torch.stack([layer.real.weight, layer.imag.weight], dim=0)

    return W
