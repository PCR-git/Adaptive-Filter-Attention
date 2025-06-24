
import numpy as np
import torch
from precision_attention import compute_backwards_mat_exp, compute_nu

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