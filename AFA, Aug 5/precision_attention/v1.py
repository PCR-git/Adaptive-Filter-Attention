import numpy as np
import torch
from utils import complex_exp, complex_hadamard

##########################################################################################
##########################################################################################

def compute_residuals(X,lambda_h,t_v,args):
    """
    Computes the residuals between the actual and estimated states over time.

    Parameters:
        X (torch.Tensor): The input tensor of shape [vec_size, Npts, d1, d2].
        lambda_h (torch.Tensor): The diagonal matrix of eigenvalues (from the state transition matrix)
        t_v (torch.Tensor): Time differences vector.
        args: Arguments containing system parameters

    Returns:
        R_qk_ij (torch.Tensor): Residuals of shape [vec_size, Npts, Npts, d1, d2].
    """

    R_qk_ij = torch.zeros(X.size()[0], X.size()[1], X.size()[1], X.size()[2], X.size()[3]).to(args.device)

    # Compute the forward and backward matrix exponentials
    mat_exp_f = complex_exp(lambda_h*t_v.unsqueeze(0).unsqueeze(2).unsqueeze(3)) # Forward
    mat_exp_b = complex_exp(-lambda_h*t_v.unsqueeze(0).unsqueeze(2).unsqueeze(3)) # Backward

    # Construct the kernel by flipping the forward exponential and concatenating
    K_exp = torch.concatenate((mat_exp_f.flip(dims=[1])[:,:-1,:,:], mat_exp_b),dim=1) # Matrix exponential kernel

    # Compute the residuals iteratively
    for i in range(args.seq_len):
        mat_exp = K_exp[:,args.seq_len-1-i:2*args.seq_len-1-i,:,:]

        X_ij_hat = complex_hadamard(mat_exp,X)
        r_qk_ij = X[:,i,:,:].unsqueeze(1) - X_ij_hat
        R_qk_ij[:,:,i,:,:] = r_qk_ij

    return R_qk_ij

##########################################################################################
##########################################################################################

def compute_kernel_v1(lambda_h, t_v, args):
    """
    Computes the exponential kernel for a given decay matrix and array of time steps.

    Parameters:
      lambda_h (torch.Tensor): The decay matrix for state propagation.
      t_v (torch.Tensor): Time differences vector.

    Returns:
      K_exp (torch.Tensor): The computed kernel of shape [vec_size, 2*Npts-1, d1, d2].
    """
    # Matrix exponentials
    mat_exp_f = complex_exp(lambda_h*t_v.unsqueeze(0).unsqueeze(2).unsqueeze(3))  # Forward
#     mat_exp_b = complex_exp(-lambda_h*t_v.unsqueeze(0).unsqueeze(2).unsqueeze(3))  # Backward
    mat_exp_b = torch.stack((torch.ones(args.seq_len,args.d_v,1), torch.zeros(args.seq_len,args.d_v,1))).to(lambda_h.device) # IF CAUSAL
    
    # Full kernel
    K_exp = torch.concatenate((mat_exp_f.flip(dims=[1])[:,:-1,:,:], mat_exp_b), dim=1)  # Matrix exponential kernel

    return K_exp

##########################################################################################
##########################################################################################

def compute_estimates_and_residuals_vectorized(X_q, X_k, X_v, K_exp, args):
    """
    Computes the estimated states and residuals between the actual and estimated states.

    Parameters:
      X (torch.Tensor): Input tensor of shape [vec_size, Npts, d1, d2].
      K_exp (torch.Tensor): Precomputed matrix exponential kernel.
      args: Arguments containing `Npts` and `device`.

    Returns:
      X_ij_hat_all (torch.Tensor): Estimated states.
      R_qk_ij (torch.Tensor): Residuals between estimated and actual states.
    """
    
    # Generate index ranges for (i, j) computation
    i_values = torch.arange(args.seq_len, device=args.device)
    start_indices = args.seq_len - 1 - i_values
    indices = start_indices.unsqueeze(1) + torch.arange(args.seq_len, device=args.device).unsqueeze(0)

    # Extract kernel slices for all pairs (i, j)
    mat_exp_slices = K_exp[:, indices, :, :]

    # Compute all estimated states at once
    X_ij_hat_k = complex_hadamard(mat_exp_slices, X_k.unsqueeze(1))
    X_ij_hat_v = complex_hadamard(mat_exp_slices, X_v.unsqueeze(1))

    # Compute residuals
    R_qk_ij = X_q.unsqueeze(2) - X_ij_hat_k

    return X_ij_hat_v, R_qk_ij

##########################################################################################
##########################################################################################

def get_time_diffs(t_v_all, args):
    """
    Get the time differences between all pairs of inputs,
    for use in compute_precision.
    """
    
    # Generate index ranges for (i, j) computation
    i_values = torch.arange(args.seq_len, device=args.device)
    start_indices = args.seq_len - 1 - i_values
    indices = start_indices.unsqueeze(1) + torch.arange(args.seq_len, device=args.device).unsqueeze(0)
    
    if len(t_v_all.size()) > 1:
        t_v = t_v_all[0]
    else:
        t_v = t_v_all

    if args.t_equal == 1: # If equal time intevals
        tji_v = torch.cat((-t_v.flip(0)[:-1], t_v))
        t_ji = tji_v[indices].unsqueeze(0)
    else: # If unequal time intervals
        tji_v_all = torch.concatenate((-t_v_all.flip(1)[:, :-1], t_v_all),axis=1)
        t_ji = tji_v_all[:, indices]
        
    t_ji = t_ji*(t_ji<=0)

    return t_ji, t_v

##########################################################################################
##########################################################################################

def compute_neg_kernel(lambda_h, t_v):
    """
    Compute a kernel of matrix exponential values.
    Speeds up computation in compute_backwards_mat_exp by allowing us to use a sliding window
    """
    
    # Matrix exponentials
    mat_exp2 = torch.exp(-2*lambda_h[0]*t_v.unsqueeze(-1).unsqueeze(-1)).unsqueeze(0)

#     ones = (torch.ones(1,args.seq_len,args.d_v,1)).to(lambda_h.device) # IF CAUSAL
    ones = torch.ones_like(mat_exp2).to(lambda_h.device) # IF CAUSAL

    K_exp2 = torch.concatenate((mat_exp2.flip(dims=[1])[:,:-1,:,:], ones), dim=1)  # Matrix exponential kernel

    return K_exp2

##########################################################################################
##########################################################################################

def clamp_exponent_arg(exp_arg,max_lim=10):
    """
    Clamp the argument of the exponent to prevent infinite values
    """
    
#     return max_lim - torch.log(1+torch.exp(max_lim - exp_arg))
    return torch.clamp(exp_arg, min=0, max=max_lim)

