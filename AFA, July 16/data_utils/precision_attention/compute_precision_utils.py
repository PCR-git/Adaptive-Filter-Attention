
import numpy as np
import torch
from utils import complex_exp, complex_exp_v2, batched_complex_hadamard, batched_complex_hadamard_full

##########################################################################################
##########################################################################################

def compute_exp_kernel(lambda_h, t_v):
    """
    Computes the exponential kernel for a given decay matrix and array of time steps.

    Parameters:
      lambda_h (torch.Tensor): The decay matrix for state propagation.''
      t_v (torch.Tensor): Time differences vector.

    Returns:
      K_exp (torch.Tensor): The computed kernel of shape [vec_size, 2*Npts-1, d1, d2].
    """
    # Matrix exponentials
    
#     mag_f, phase_f = complex_exp_v2(lambda_h*t_v.unsqueeze(0).unsqueeze(2).unsqueeze(3))  # Forward
#     mat_exp_f = mag_f*phase_f
#     mag_f2 = (mag_f**2).unsqueeze(0)

    exp_f = complex_exp(lambda_h*t_v.unsqueeze(0).unsqueeze(2).unsqueeze(3))  # Forward
#     exp_b = complex_exp(-lambda_h*t_v.unsqueeze(0).unsqueeze(2).unsqueeze(3))  # Forward

    mag_f2 = torch.sum(exp_f**2, dim=0, keepdims=True)

    exp_b = torch.stack((torch.ones_like(exp_f[0]), torch.zeros_like(exp_f[1]))).to(lambda_h.device) # IF CAUSAL
    mag_b = torch.ones_like(mag_f2).to(lambda_h.device)
#     mat_exp_b = torch.stack((torch.ones(args.seq_len,args.d_v,1), torch.zeros(args.seq_len,args.d_v,1))).to(lambda_h.device) # IF CAUSAL
#     mag_b = torch.ones(1, args.seq_len,args.d_v,1).to(lambda_h.device)

    # Full kernel
    K_exp = torch.concatenate((exp_f.flip(dims=[1])[:,:-1,:,:], exp_b), dim=1)  # Matrix exponential kernel
    K_exp2 = torch.concatenate((mag_f2.flip(dims=[1])[:,:-1,:,:], mag_b), dim=1)  # Matrix exponential kernel

    return K_exp, K_exp2

# def compute_kernel(lambda_h, t_v):
#     """
#     Computes the exponential kernel for a given decay matrix and array of time steps.

#     Parameters:
#       lambda_h (torch.Tensor): The decay matrix for state propagation.''
#       t_v (torch.Tensor): Time differences vector.

#     Returns:
#       K_exp (torch.Tensor): The computed kernel of shape [vec_size, 2*Npts-1, d1, d2].
#     """
#     # Matrix exponentials
#     mag_f, phase_f = complex_exp_v2(lambda_h*t_v.unsqueeze(0).unsqueeze(2).unsqueeze(3))  # Forward
#     mat_exp_f = mag_f*phase_f
    
#     mag_f2 = (mag_f**2).unsqueeze(0)
# #     mag_b, phase_b = complex_exp_v2(-lambda_h*t_v.unsqueeze(0).unsqueeze(2).unsqueeze(3))  # Forward
# #     mag_b2 = (mag_b**2).unsqueeze(0)

#     mat_exp_1 = torch.stack((torch.ones_like(mat_exp_f[0]), torch.zeros_like(mat_exp_f[1]))).to(lambda_h.device) # IF CAUSAL
#     mag_1 = torch.ones_like(mag_f2).to(lambda_h.device)
# #     mat_exp_b = torch.stack((torch.ones(args.seq_len,args.d_v,1), torch.zeros(args.seq_len,args.d_v,1))).to(lambda_h.device) # IF CAUSAL
# #     mag_b = torch.ones(1, args.seq_len,args.d_v,1).to(lambda_h.device)

#     # Full kernel
#     K_exp = torch.concatenate((mat_exp_f.flip(dims=[1])[:,:-1,:,:], mat_exp_1), dim=1)  # Matrix exponential kernel
#     K_expb2 = torch.concatenate((mag_f2.flip(dims=[1])[:,:-1,:,:], mag_1), dim=1)  # Matrix exponential kernel

#     return K_exp, K_expb2

##########################################################################################
##########################################################################################

def batched_compute_estimates_and_residuals_vectorized(X_q, X_k, X_v, K_exp, args):
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
    X_ij_hat_k = batched_complex_hadamard(mat_exp_slices, X_k.unsqueeze(2))
    X_ij_hat_v = batched_complex_hadamard(mat_exp_slices, X_v.unsqueeze(2))

    # Compute residuals
    R_qk_ij = X_q.unsqueeze(3) - X_ij_hat_k

    return X_ij_hat_v, R_qk_ij

##########################################################################################
##########################################################################################

def compute_estimates_and_residuals_irregular_times(lambda_h, t_v_all, X_q, X_k, X_v, args):
    """
    Computes the estimated states and residuals between the actual and estimated states.
    This version works when measurements comes at irregular time intervals.
    However, it is more expensive, because you have to do the computation at all pairs of time diferences,
    rather than using a sliding window.
    """
    
    t_diff_mat = t_v_all[:, :, None] - t_v_all[:, None, :] # Matrix of time differences

    # Compute matrix exponentials
    mat_exp_slices = batched_complex_exp(lambda_h.unsqueeze(1).unsqueeze(0) * t_diff_mat.unsqueeze(1).unsqueeze(-1).unsqueeze(-1))

    # Compute all estimated states at once
    X_ij_hat_k = batched_complex_hadamard_full(mat_exp_slices, X_k.unsqueeze(2))
    X_ij_hat_v = batched_complex_hadamard_full(mat_exp_slices, X_v.unsqueeze(2))

    R_qk_ij = X_q.unsqueeze(3) - X_ij_hat_k

    return X_ij_hat_v, R_qk_ij

##########################################################################################
##########################################################################################

def compute_backwards_mat_exp(lambda_h, K_exp2, t_v_all, args):
    """
    Compute the backwards matrix exponential, for use in compute_precision
    """

    if len(t_v_all.size()) > 1: # If multiple arrays of time steps
        t_v = t_v_all[0]
    else: # If only a single array of time steps
        t_v = t_v_all
        t_v_all = t_v_all.unsqueeze(0)

    # Generate indices of time differences
    i_values = torch.arange(args.seq_len, device=args.device)
    start_indices = args.seq_len - 1 - i_values
    indices = start_indices.unsqueeze(1) + torch.arange(args.seq_len, device=args.device).unsqueeze(0)

    if args.t_equal == 0 or args.tanh == 1: # If equal time intervals
        tji_v_all = torch.concatenate((-t_v_all.flip(1)[:, :-1], t_v_all*0),axis=1)
        t_ji = tji_v_all[:, indices] # Get all pairs of time differences
        t_ji = t_ji*(t_ji<=0) # Mask out positive times

        mat_exp_b2 = torch.exp(-2*lambda_h[0].unsqueeze(0).unsqueeze(0).squeeze(-1) * t_ji.unsqueeze(-1))

    else: # If variable time intervals
#         K_exp2 = compute_neg_kernel(lambda_h, -t_v) # Compute kernel of negative mat exp

        # Extract kernel slices for all pairs (i, j)
        mat_exp_b2 = K_exp2[:, indices, :, 0]
        t_ji = None
        
    return mat_exp_b2.unsqueeze(-1), t_ji

##########################################################################################
##########################################################################################

# def compute_nu(V_ij_dagger_p, R_qk_ij, alpha_nu, beta_nu, epsilon = 1E-5):
#     """
#     Computes nu, the scaling of the precision matrix, accounting for uncertainty of the prior vs the measurements.
#     This allows for adaptive weighting of the model and the measurements (i.e. allows for adjusting for a model that is not exactly correct).
#     nu is in the range [0, 1]. nu = 1 implies high confidence in the measurements, whereas nu = 0 implies no confidence.
#     """
    
#     P_ij_p = 1/(V_ij_dagger_p + epsilon) # Prior for the propagated precision matrix

#     # Mahalanobis distance of residuals given prior
#     mahalanobis_distance_p = P_ij_p * (R_qk_ij[:,0]**2 + R_qk_ij[:,1]**2)
    
#     # Compute nu
#     nu = 1 - torch.exp(-torch.abs(alpha_nu) * mahalanobis_distance_p - torch.abs(beta_nu))
    
#     return nu
# #     return nu + epsilon