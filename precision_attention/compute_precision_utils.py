
import numpy as np
import torch
from utils import complex_exp_v2, batched_complex_hadamard, batched_complex_hadamard_full

##########################################################################################
##########################################################################################

def compute_kernel(lambda_h, t_v):
    """
    Computes the exponential kernel for a given decay matrix and array of time steps.

    Parameters:
      lambda_h (torch.Tensor): The decay matrix for state propagation.''
      t_v (torch.Tensor): Time differences vector.

    Returns:
      K_exp (torch.Tensor): The computed kernel of shape [vec_size, 2*Npts-1, d1, d2].
    """
    # Matrix exponentials
    mag_f, phase_f = complex_exp_v2(lambda_h*t_v.unsqueeze(0).unsqueeze(2).unsqueeze(3))  # Forward
    mat_exp_f = mag_f*phase_f
    mag_f2 = (mag_f**2).unsqueeze(0)

    mat_exp_b = torch.stack((torch.ones_like(mat_exp_f[0]), torch.zeros_like(mat_exp_f[1]))).to(lambda_h.device) # IF CAUSAL
    mag_b = torch.ones_like(mag_f2).to(lambda_h.device)
#     mat_exp_b = torch.stack((torch.ones(args.seq_len,args.d_v,1), torch.zeros(args.seq_len,args.d_v,1))).to(lambda_h.device) # IF CAUSAL
#     mag_b = torch.ones(1, args.seq_len,args.d_v,1).to(lambda_h.device)

    # Full kernel
    K_exp = torch.concatenate((mat_exp_f.flip(dims=[1])[:,:-1,:,:], mat_exp_b), dim=1)  # Matrix exponential kernel
    K_exp2 = torch.concatenate((mag_f2.flip(dims=[1])[:,:-1,:,:], mag_b), dim=1)  # Matrix exponential kernel

    return K_exp, K_exp2

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

def compute_nu(V_ij_dagger_p, R_qk_ij, alpha_nu, beta_nu, epsilon = 1E-5):
    """
    Computes nu, the scaling of the precision matrix, accounting for uncertainty of the prior vs the measurements.
    This allows for adaptive weighting of the model and the measurements (i.e. allows for adjusting for a model that is not exactly correct).
    nu is in the range [0, 1]. nu = 1 implies high confidence in the measurements, whereas nu = 0 implies no confidence.
    """
    
    P_ij_p = 1/(V_ij_dagger_p + epsilon) # Prior for the propagated precision matrix

    # Mahalanobis distance of residuals given prior
    mahalanobis_distance_p = P_ij_p * (R_qk_ij[:,0]**2 + R_qk_ij[:,1]**2)
    
    # Compute nu
    nu = 1 - torch.exp(-torch.abs(alpha_nu) * mahalanobis_distance_p - torch.abs(beta_nu))
    
    return nu
#     return nu + epsilon

##########################################################################################
##########################################################################################

def compute_precision(lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, K_exp2, t_v_all, args, R_qk_ij=None, alpha_nu=None, beta_nu=None, lambda_C=1, epsilon=1E-5):
    """
    Computes the precision matrix for the attention mechanism.
    Parameters:
    lambda_h (torch.Tensor): Diagonal of state transition matrix.
    lambda_Omega (torch.Tensor): Process noise covariance matrix.
    lambda_Omega0 (torch.Tensor): Initial condition of process noise covariance matrix.
    lambda_C (torch.Tensor): Measurement output matrix.
    lambda_Gamma (torch.Tensor): Measurement noise covariance.
    t_ji (torch.Tensor): Time differences.
    args: Arguments containing system parameters.

    Returns:
      P_ij (torch.Tensor): Precision matrix.
    """
    
    # Compute backwards matrix exponential
    mat_exp_b2, _ = compute_backwards_mat_exp(lambda_h, K_exp2, t_v_all, args)

    frac = (lambda_Omega/(2*torch.abs(lambda_h[0]) + epsilon)).unsqueeze(0).unsqueeze(0)

    V_ij_tilde = torch.abs(frac * (1 - mat_exp_b2))

    # Backpropagated covariance matrix
    V_ij = V_ij_tilde + lambda_Omega0.unsqueeze(0).unsqueeze(0)

    # Incorporate measurement model
    V_ij_dagger_p = lambda_C**2 * V_ij + mat_exp_b2 * lambda_Gamma

    Omega_z = (lambda_C**2 * lambda_Omega0 + lambda_Gamma).unsqueeze(0).unsqueeze(0)

    if args.nu_adaptive == 1: # Use adaptive computation for nu
        nu = compute_nu(V_ij_dagger_p, R_qk_ij, alpha_nu, beta_nu)
    else: # Fixed nu
        nu = args.nu

    # Backpropagated covariance matrix with measurement model
    V_ij_dagger = nu*V_ij_dagger_p + (1 - nu) * Omega_z

    # Compute precision matrix
    P_ij = 1/(V_ij_dagger + epsilon)

    return P_ij, nu

##########################################################################################
##########################################################################################

# More general version of compute_precision, accounting for continuous measurements
def compute_precision_tanh(lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, K_exp2, t_v_all, args, R_qk_ij=None, alpha_nu=None, beta_nu=None, lambda_C=1, epsilon = 1E-5):
    """
    Computes the precision matrix P_ij, accounting for continuous measurements.
  
    Parameters:
        lambda_h (torch.Tensor): Diagonal of state transition matrix.
        lambda_Omega (torch.Tensor): Process noise covariance matrix.
        lambda_Omega0 (torch.Tensor): Initial condition of process noise covariance matrix.
        lambda_C (torch.Tensor): Measurement output matrix.
        lambda_Gamma (torch.Tensor): Measurement noise covariance.
        t_ji (torch.Tensor): Time differences.
        args: Arguments containing system parameters.

    Returns:
        P_ij (torch.Tensor): Precision matrix.
    """
    
    # Compute backpropagated matrix exponentials
    mat_exp_b2, t_ji = compute_backwards_mat_exp(lambda_h, K_exp2, t_v_all, args)

    # Compute covariance
    lambda_h_r = lambda_h[0]

    #     # Adjust lambda_Omega to account for non-unitary W_v
    #     if args.adjust == 1:
    #         mag_sq = W_v[0]**2 + W_v[1]**2
    #         lambda_Omega = lambda_Omega * torch.sum(mag_sq,axis=-1,keepdims=True)
    # #         lambda_Omega0 = lambda_Omega0 * torch.sum(mag_sq,axis=-1,keepdims=True)

    c2 = torch.sqrt(lambda_h_r**2 + (lambda_Omega/(lambda_Gamma+epsilon))*lambda_C**2).squeeze(2).unsqueeze(0).unsqueeze(0)
    c3 = ((lambda_Omega0/(lambda_Gamma + epsilon))*lambda_C**2 - lambda_h_r).squeeze(2).unsqueeze(0).unsqueeze(0)
    c1 = torch.log(torch.abs(c2 + c3 + epsilon)/(torch.abs(c2 - c3) + epsilon))/(torch.abs(c2)+epsilon)

    frac = (lambda_Gamma / (lambda_C**2 + epsilon)).squeeze(2).unsqueeze(0).unsqueeze(0)

    V_ij_tilde = frac*(c2*torch.tanh((c2/2)*(t_ji.unsqueeze(-1) - c1)) + lambda_h_r.squeeze(2).unsqueeze(0).unsqueeze(0))

    # Backpropagated covariance
    V_ij_tilde_b = mat_exp_b2*V_ij_tilde.unsqueeze(-1)

    V_ij = V_ij_tilde_b + lambda_Omega0.unsqueeze(0).unsqueeze(0)

    # Incorporate measurement model
    V_ij_dagger_p = lambda_C**2 * V_ij + mat_exp_b2 * lambda_Gamma

    Omega_z = (lambda_C**2 * lambda_Omega0 + lambda_Gamma).unsqueeze(0).unsqueeze(0)

    if args.nu_adaptive == 1: # Use adaptive computation for nu
        nu = compute_nu(V_ij_dagger_p, R_qk_ij, alpha_nu, beta_nu)
    else: # Fixed nu
        nu = args.nu

    # Backpropagated covariance matrix with measurement model
    V_ij_dagger = nu*V_ij_dagger_p + (1 - nu) * Omega_z

    # Compute precision matrix
    # P_ij = 1/V_ij_dagger
    P_ij = 1/(torch.abs(V_ij_dagger) + epsilon)

    return P_ij, nu