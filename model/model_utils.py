import numpy as np
import torch

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

def compute_lambda_h(lambder,args,scale_c=20,odd_embed_size=0):
    """
    Construct lambda_h on the fly from lambda1
    
    (lambda1 is a torch parameter tensor of length d_e/2, where d_e is the embedding dimension)
    lambda_h is the full set of eigenvalues, which are nonpositive and come in complex-conjugate pairs
    (the latter constraint forces the full state transition matrix to be real, provided W_v and W_p are unitary
     and W_v = W_p^* ie a complex conjugate transpose pair)
    """
    
    scale = scale_c/(2*args.tf)
#     mag = torch.abs(lambder[0])
#     mag = scale * torch.abs(lambder[0])
#     mag = scale * lambder[0]**2

#     mag = scale * torch.sigmoid(lambder[0])
    mag = scale * torch.abs(lambder[0])
#     mag = scale * lambder[0]**2

    lambda1_0 = -mag
    lambda2_0 = -mag
    lambda1_1 = lambder[1]
    lambda2_1 = -lambder[1]
#     lambda1_1 = (2*np.pi/(2*args.tf)) * lambder[1]
#     lambda2_1 = -(2*np.pi/(2*args.tf)) * lambder[1]
    lambda1 = torch.stack((lambda1_0, lambda1_1))
    lambda2 = torch.stack((lambda2_0, lambda2_1))
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
