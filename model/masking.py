import numpy as np
import torch

##########################################################################################
##########################################################################################

def init_weight_masks(module, args):
    """
    Create masks for parameter matrices (used for testing)
    """
    module.eigen_mask = torch.zeros(2,1,module.d_v,1).to(args.device)
    module.eigen_mask[:,:,0:2,:] = 1

    module.noise_mask = torch.zeros(1,module.d_v,1).to(args.device)
    module.noise_mask[:,0:2,:] = 1

    module.weight_mask_k = torch.zeros_like(module.W_k).to(args.device)
    module.weight_mask_k[:,:,0:2,0:2] = 1

    module.weight_mask_v = torch.zeros_like(module.W_v).to(args.device)
    module.weight_mask_v[:,:,0:2,0:2] = 1

    module.weight_mask_p = torch.zeros_like(module.W_p).to(args.device)
    module.weight_mask_p[:,:,0:2,0:2] = 1
    
##########################################################################################
##########################################################################################

def apply_weight_masks(module, args):
    """
    Mask out all eigenvalues & weights above a certain dimension
    """
    
    if args.weight_mask == 1:
        # Mask out eigenvalues:
        lambda_h = module.lambda_h*module.eigen_mask
        
        # Mask out noise covariance eigenvals
        lambda_Omega = module.lambda_Omega * module.noise_mask
        lambda_Omega0 = module.lambda_Omega0 * module.noise_mask
        lambda_Gamma = module.lambda_Gamma * module.noise_mask

        # Mask weight values
        W_q = module.W_q * module.weight_mask_k
        W_k = module.W_k * module.weight_mask_k
        W_v = module.W_v * module.weight_mask_v
        W_p = module.W_p * module.weight_mask_p
        W_r = module.W_r * module.weight_mask_v
        W_e = module.W_e * module.weight_mask_p

        # Mask biases
        W_q_b = module.W_q_b * module.weight_mask_k[:,:,:,0].unsqueeze(0).unsqueeze(-1)
        W_k_b = module.W_k_b * module.weight_mask_k[:,:,:,0].unsqueeze(0).unsqueeze(-1)
        W_v_b = module.W_v_b * module.weight_mask_v[:,:,:,0].unsqueeze(0).unsqueeze(-1)
        W_p_b = module.W_p_b * module.weight_mask_p[:,:,:,0].unsqueeze(0).unsqueeze(-1)
        W_r_b = module.W_r_b * module.weight_mask_v[:,:,:,0].unsqueeze(0).unsqueeze(-1)
        W_e_b = module.W_e_b * module.weight_mask_p[:,:,:,0].unsqueeze(0).unsqueeze(-1)
    else:
        lambda_h = module.lambda_h
        
        lambda_Omega = module.lambda_Omega
        lambda_Omega0 = module.lambda_Omega0
        lambda_Gamma = module.lambda_Gamma

        W_q = module.W_q
        W_k = module.W_k
        W_v = module.W_v
        W_p = module.W_p
        W_r = module.W_r
        W_e = module.W_e

        W_q_b = module.W_q_b
        W_k_b = module.W_k_b
        W_v_b = module.W_v_b
        W_p_b = module.W_p_b
        W_r_b = module.W_r_b
        W_e_b = module.W_e_b
        
    return lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, W_q, W_k, W_v, W_p, W_r, W_e, W_q_b, W_k_b, W_v_b, W_p_b, W_r_b, W_e_b

