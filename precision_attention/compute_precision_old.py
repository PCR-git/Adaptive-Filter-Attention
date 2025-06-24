import numpy as np
import torch
from precision_attention import compute_backwards_mat_exp

##########################################################################################
##########################################################################################

def compute_precision_v1(lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, tji_v, args, lambda_C = 1, epsilon = 1E-5):

    """
    Computes the precision matrix for the attention mechanism.

    Parameters:
      lambda_h (torch.Tensor): Diagonal of state transition matrix.
      lambda_Omega (torch.Tensor): Process noise covariance matrix.
      lambda_Omega0 (torch.Tensor): Initial condition of process noise covariance matrix.
      lambda_C (torch.Tensor): Measurement output matrix.
      lambda_Gamma (torch.Tensor): Measurement noise covariance.
      tji_v (torch.Tensor): Time differences.=
      args: Arguments containing system parameters.

    Returns:
      P_ij (torch.Tensor): Precision matrix.
    """

    # Generate index ranges for (i, j) computation
    i_values = torch.arange(args.seq_len, device=args.device)
    start_indices = args.seq_len - 1 - i_values
    indices = start_indices.unsqueeze(1) + torch.arange(args.seq_len, device=args.device).unsqueeze(0)

    # Extract time intervals
    t_ji = tji_v[indices]

    # Compute backpropagated matrix exponentials
    lambda_reshaped = lambda_h[0].squeeze(2).unsqueeze(0).unsqueeze(0)
    mat_exp_b2 = torch.exp(-2 * lambda_reshaped * t_ji.unsqueeze(-1))

    # Compute covariance
    lambda_h_r = torch.abs(lambda_h[0])
#     lambda_h_r = lambda_h[0]
    frac = (lambda_Omega/(2*lambda_h_r + epsilon)).squeeze(-1).unsqueeze(0).unsqueeze(0)
    
    V_ij_tilde = torch.abs(frac*(1 - mat_exp_b2)).unsqueeze(-1)
    V_ij = args.nu*V_ij_tilde + lambda_Omega0

    # Incorporate measurement model
    #     V_ij_dagger = V_ij + mat_exp_b2.unsqueeze(-1) * lambda_Gamma + epsilon
    V_ij_dagger = lambda_C**2 * V_ij + mat_exp_b2.unsqueeze(-1) * lambda_Gamma + epsilon

    # Compute precision matrix
    P_ij = 1/V_ij_dagger

    return P_ij

##########################################################################################
##########################################################################################

# def compute_precision_v2(lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, t_ji, args, W_v=None, lambda_C = 1, epsilon = 1E-5):
#     """
#     Computes the precision matrix for the attention mechanism.
#     Parameters:
#     lambda_h (torch.Tensor): Diagonal of state transition matrix.
#     lambda_Omega (torch.Tensor): Process noise covariance matrix.
#     lambda_Omega0 (torch.Tensor): Initial condition of process noise covariance matrix.
#     lambda_C (torch.Tensor): Measurement output matrix.
#     lambda_Gamma (torch.Tensor): Measurement noise covariance.
#     t_ji (torch.Tensor): Time differences.
#     args: Arguments containing system parameters.

#     Returns:
#       P_ij (torch.Tensor): Precision matrix.
#     """
    
#     mat_exp_b2 = torch.exp(-2*lambda_h[0].unsqueeze(0).unsqueeze(0).squeeze(-1) * t_ji.unsqueeze(-1))

#     frac = (lambda_Omega/(2*torch.abs(lambda_h[0]) + epsilon)).squeeze(-1).unsqueeze(0).unsqueeze(0)

#     # Covariance
#     V_ij_tilde = torch.abs(frac * (1 - mat_exp_b2)).unsqueeze(-1)

#     # Full covariance matrix
#     V_ij = args.nu*V_ij_tilde + lambda_Omega0.unsqueeze(0)

#     # Incorporate measurement model
#     V_ij_dagger = lambda_C**2 * V_ij + mat_exp_b2.unsqueeze(-1) * lambda_Gamma + epsilon

#     # Compute precision matrix
#     P_ij = 1/V_ij_dagger

#     return P_ij

##########################################################################################
##########################################################################################

# def compute_precision_v3(lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, t_ji, args, R_qk_ij=None, alpha_nu=None, beta_nu=None, lambda_C=1, epsilon=1E-5):
#     """
#     Computes the precision matrix for the attention mechanism.
#     Parameters:
#     lambda_h (torch.Tensor): Diagonal of state transition matrix.
#     lambda_Omega (torch.Tensor): Process noise covariance matrix.
#     lambda_Omega0 (torch.Tensor): Initial condition of process noise covariance matrix.
#     lambda_C (torch.Tensor): Measurement output matrix.
#     lambda_Gamma (torch.Tensor): Measurement noise covariance.
#     t_ji (torch.Tensor): Time differences.
#     args: Arguments containing system parameters.

#     Returns:
#       P_ij (torch.Tensor): Precision matrix.
#     """
    
#     mat_exp_b2 = torch.exp(-2*lambda_h[0].unsqueeze(0).unsqueeze(0).squeeze(-1) * t_ji.unsqueeze(-1))

#     frac = (lambda_Omega/(2*torch.abs(lambda_h[0]) + epsilon)).squeeze(-1).unsqueeze(0).unsqueeze(0)

#     # Covariance
#     V_ij_tilde = torch.abs(frac * (1 - mat_exp_b2)).unsqueeze(-1)

#     # Full covariance matrix
#     V_ij = V_ij_tilde + lambda_Omega0.unsqueeze(0)

#     # Incorporate measurement model
#     V_ij_dagger_p = lambda_C**2 * V_ij + mat_exp_b2.unsqueeze(-1) * lambda_Gamma
    
#     Omega_z = lambda_C**2 * lambda_Omega0 + lambda_Gamma
    
#     if args.nu_adaptive == 1:
#         nu = compute_nu(V_ij_dagger_p, R_qk_ij, alpha_nu, beta_nu)
#     else:
#         nu = args.nu

#     V_ij_dagger = nu*V_ij_dagger_p + (1 - nu) * Omega_z
    
#     # Compute precision matrix
#     P_ij = 1/(V_ij_dagger + epsilon)

#     return P_ij, nu

##########################################################################################
##########################################################################################

# def compute_precision_v4(lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, t_ji, args, R_qk_ij=None, alpha_nu=None, beta_nu=None, lambda_C=1, epsilon=1E-5):
#     """
#     Computes the precision matrix for the attention mechanism.
#     Parameters:
#     lambda_h (torch.Tensor): Diagonal of state transition matrix.
#     lambda_Omega (torch.Tensor): Process noise covariance matrix.
#     lambda_Omega0 (torch.Tensor): Initial condition of process noise covariance matrix.
#     lambda_C (torch.Tensor): Measurement output matrix.
#     lambda_Gamma (torch.Tensor): Measurement noise covariance.
#     t_ji (torch.Tensor): Time differences.
#     args: Arguments containing system parameters.

#     Returns:
#       P_ij (torch.Tensor): Precision matrix.
#     """
    
#     mat_exp_b2 = torch.exp(-2*lambda_h[0].unsqueeze(0).unsqueeze(0).squeeze(-1) * t_ji.unsqueeze(-1))

#     frac = (lambda_C**2 * lambda_Omega/(2*torch.abs(lambda_h[0]) + epsilon)).squeeze(-1).unsqueeze(0).unsqueeze(0)

#     V_ij_dagger_p = torch.abs(1 - mat_exp_b2) * (frac + lambda_Gamma.squeeze(-1).unsqueeze(0).unsqueeze(0))

#     Omega_z = (lambda_C**2 * lambda_Omega0 + lambda_Gamma).squeeze(-1).unsqueeze(0).unsqueeze(0)

#     if args.nu_adaptive == 1:
#         nu = compute_nu(V_ij_dagger_p, R_qk_ij, alpha_nu, beta_nu)
#     else:
#         nu = args.nu
    
#     # Covariance
#     V_ij_dagger = nu*V_ij_dagger_p + Omega_z

#     # Compute precision matrix
#     P_ij = 1/(V_ij_dagger + epsilon).unsqueeze(-1)

#     return P_ij, nu

##########################################################################################
##########################################################################################

# def compute_precision_v5(lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, t_v_all, args, R_qk_ij=None, alpha_nu=None, beta_nu=None, lambda_C=1, epsilon=1E-5):
#     """
#     Computes the precision matrix for the attention mechanism.
#     Parameters:
#     lambda_h (torch.Tensor): Diagonal of state transition matrix.
#     lambda_Omega (torch.Tensor): Process noise covariance matrix.
#     lambda_Omega0 (torch.Tensor): Initial condition of process noise covariance matrix.
#     lambda_C (torch.Tensor): Measurement output matrix.
#     lambda_Gamma (torch.Tensor): Measurement noise covariance.
#     t_ji (torch.Tensor): Time differences.
#     args: Arguments containing system parameters.

#     Returns:
#       P_ij (torch.Tensor): Precision matrix.
#     """
    
#     mat_exp_b2 = compute_backwards_mat_exp(lambda_h, t_v_all, args)

#     frac = (lambda_C**2 * lambda_Omega/(2*torch.abs(lambda_h[0]) + epsilon)).squeeze(-1).unsqueeze(0).unsqueeze(0)

#     V_ij_dagger_p = torch.abs(1 - mat_exp_b2) * (frac + lambda_Gamma.squeeze(-1).unsqueeze(0).unsqueeze(0))

#     Omega_z = (lambda_C**2 * lambda_Omega0 + lambda_Gamma).squeeze(-1).unsqueeze(0).unsqueeze(0)

#     if args.nu_adaptive == 1:
#         nu = compute_nu(V_ij_dagger_p, R_qk_ij, alpha_nu, beta_nu)
#     else:
#         nu = args.nu
    
#     # Covariance
#     V_ij_dagger = nu*V_ij_dagger_p + Omega_z

#     # Compute precision matrix
#     P_ij = 1/(V_ij_dagger + epsilon).unsqueeze(-1)

#     return P_ij, nu

##########################################################################################
##########################################################################################

# def compute_precision_v6(lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, K_exp2, t_v_all, args, R_qk_ij=None, alpha_nu=None, beta_nu=None, lambda_C=1, epsilon=1E-5):
#     """
#     Computes the precision matrix for the attention mechanism.
#     Parameters:
#     lambda_h (torch.Tensor): Diagonal of state transition matrix.
#     lambda_Omega (torch.Tensor): Process noise covariance matrix.
#     lambda_Omega0 (torch.Tensor): Initial condition of process noise covariance matrix.
#     lambda_C (torch.Tensor): Measurement output matrix.
#     lambda_Gamma (torch.Tensor): Measurement noise covariance.
#     t_ji (torch.Tensor): Time differences.
#     args: Arguments containing system parameters.

#     Returns:
#       P_ij (torch.Tensor): Precision matrix.
#     """
    
#     mat_exp_b2, _ = compute_backwards_mat_exp(lambda_h, K_exp2, t_v_all, args)

#     frac = (lambda_C**2 * lambda_Omega/(2*torch.abs(lambda_h[0]) + epsilon)).unsqueeze(0).unsqueeze(0)

#     V_ij_dagger_p = torch.abs(1 - mat_exp_b2) * (frac - lambda_Gamma.unsqueeze(0).unsqueeze(0))

#     Omega_z = (lambda_C**2 * lambda_Omega0 + lambda_Gamma).unsqueeze(0).unsqueeze(0)

#     if args.nu_adaptive == 1:
#         nu = compute_nu(V_ij_dagger_p, R_qk_ij, alpha_nu, beta_nu)
#     else:
#         nu = args.nu

#     # Covariance
#     V_ij_dagger = nu*V_ij_dagger_p + Omega_z

#     # Compute precision matrix
#     P_ij = 1/(V_ij_dagger + epsilon)

#     return P_ij, nu

##########################################################################################
##########################################################################################

# def compute_precision_adaptive(lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, t_v_all, args, R_qk_ij, alpha_nu=None, beta_nu=None, lambda_C=1, epsilon=1E-5):
#     """
#     Computes the precision matrix for the attention mechanism.
#     Parameters:
#     lambda_h (torch.Tensor): Diagonal of state transition matrix.
#     lambda_Omega (torch.Tensor): Process noise covariance matrix.
#     lambda_Omega0 (torch.Tensor): Initial condition of process noise covariance matrix.
#     lambda_C (torch.Tensor): Measurement output matrix.
#     lambda_Gamma (torch.Tensor): Measurement noise covariance.
#     t_ji (torch.Tensor): Time differences.
#     args: Arguments containing system parameters.

#     Returns:
#       P_ij (torch.Tensor): Precision matrix.
#     """
    
# #     exp_arg = clamp_exponent_arg(-2*lambda_h[0].unsqueeze(0).unsqueeze(0).squeeze(-1) * t_ji.unsqueeze(-1))
#     exp_arg = -2*lambda_h[0].unsqueeze(0).unsqueeze(0).squeeze(-1) * t_ji.unsqueeze(-1)
#     mat_exp_b2 = torch.exp(exp_arg)

#     frac = (lambda_C**2 * lambda_Omega/(2*torch.abs(lambda_h[0]) + epsilon)).squeeze(-1).unsqueeze(0).unsqueeze(0)

#     V_ij_dagger_p = torch.abs(1 - mat_exp_b2) * (frac + lambda_Gamma.squeeze(-1).unsqueeze(0).unsqueeze(0))

#     Omega_z = (lambda_C**2 * lambda_Omega0 + lambda_Gamma).squeeze(-1).unsqueeze(0).unsqueeze(0)
    
#     nu = compute_nu(V_ij_dagger_p, R_qk_ij, alpha_nu, beta_nu)

#     # Covariance
#     V_ij_dagger = args.nu*V_ij_dagger_p + Omega_z

#     # Compute precision matrix
#     P_ij = 1/(V_ij_dagger + epsilon).unsqueeze(-1)

#     return P_ij, nu


