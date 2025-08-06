
import numpy as np
import torch
from precision_attention import compute_backwards_mat_exp

##########################################################################################
##########################################################################################

# def compute_precision(lambda_h, lambda_Omega, lambda_Gamma, K_exp2, t_v_all, args, lambda_C=1, epsilon=1E-5):
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
    
#     # Compute backwards matrix exponential
#     mat_exp_b2, _ = compute_backwards_mat_exp(lambda_h, K_exp2, t_v_all, args)

#     frac = (lambda_Omega/(2*torch.abs(lambda_h[0]) + epsilon)).unsqueeze(0).unsqueeze(0)

#     V_ij = torch.abs(frac * (1 - mat_exp_b2))

#     # Incorporate measurement model
#     V_ij_dagger = lambda_C**2 * V_ij + lambda_Gamma * mat_exp_b2

#     # Compute precision matrix
#     P_ij = 1/(V_ij_dagger + epsilon)

#     return P_ij

#     frac1 = (lambda_Omega/(2*torch.abs(lambda_h[0]) + epsilon)).unsqueeze(0).unsqueeze(0)
#     V_ij1 = torch.abs(frac * (1 - mat_exp_b2))
#     V_ij_dagger1 = lambda_C**2 * V_ij1 + lambda_Gamma * mat_exp_b2
#     P_ij1 = 1/(V_ij_dagger1 + epsilon)
#     print(torch.sum(P_ij - P_ij1))

def compute_precision(lambda_h, lambda_Omega, lambda_Gamma, K_exp2, t_v_all, args, lambda_C=1, epsilon=1e-5):
    """
    Computes the precision matrix for the attention mechanism.

    Parameters:
        lambda_h (torch.Tensor): Diagonal of state transition matrix (complex).
        lambda_Omega (torch.Tensor): Process noise covariance (diagonal).
        lambda_Gamma (torch.Tensor): Measurement noise covariance (diagonal).
        K_exp2 (torch.Tensor): Cached exponential terms.
        t_v_all (torch.Tensor): Time differences.
        args: System parameters.
        lambda_C (torch.Tensor or float): Measurement output matrix (or scalar).
        epsilon (float): Small constant for numerical stability.

    Returns:
        P_ij (torch.Tensor): Precision matrix.
    """
    # Compute matrix exponential in backward direction
    mat_exp_b2, _ = compute_backwards_mat_exp(lambda_h, K_exp2, t_v_all, args)

    # Use real part explicitly (guaranteed to be ≤ 0)
    lambda_h_real = lambda_h[0]  # real part of eigenvalues

    # Compute variance
    denom = -2 * lambda_h_real + epsilon
    frac = (lambda_Omega / denom).unsqueeze(0).unsqueeze(0)
    V_ij = frac * (1 - mat_exp_b2)

    # Add measurement noise (assumes λ_C is diagonal or scalar)
    V_ij_dagger = lambda_C**2 * V_ij + lambda_Gamma * mat_exp_b2

    # Compute precision (elementwise inverse)
    P_ij = 1 / (V_ij_dagger + epsilon)

    return V_ij_dagger, P_ij

##########################################################################################
##########################################################################################

def compute_precision_tanh(lambda_h, lambda_Omega, lambda_Gamma, K_exp2, t_v_all, args, lambda_C=1.0, epsilon=1e-5):
    """
    Computes the precision matrix P_ij using a tanh-based approximation,
    incorporating precomputed matrix exponential magnitudes (K_exp2).

    Inputs:
        lambda_h       : [2, d, 1] complex eigenvalues (real/imag parts).
        lambda_Omega   : [1, d, 1] process noise variance (diagonal).
        lambda_Gamma   : [1, d, 1] measurement noise variance (diagonal).
        K_exp2         : [m, d] precomputed |K|^2 = K_exp[0]**2 + K_exp[1]**2.
        t_v_all        : [m] time vector.
        args           : model hyperparameters.
        lambda_C       : scalar or [1, d, 1] measurement output matrix.
        epsilon        : small constant for numerical stability.

    Outputs:
        V_ij_dagger    : [1, m, m, d, 1] modified covariance.
        P_ij           : [1, m, m, d, 1] precision matrix.
    """

    # Compute backward matrix exponential using K_exp2 as kernel
    mat_exp_b2, t_ji = compute_backwards_mat_exp(lambda_h, K_exp2, t_v_all, args)  # [1, m, m, d], [m, m]

    lambda_r = lambda_h[0]         # [d, 1]
    Omega = lambda_Omega.squeeze(0).squeeze(-1)   # [d]
    Gamma = lambda_Gamma.squeeze(0).squeeze(-1)   # [d]
    C = lambda_C.squeeze(0).squeeze(-1) if isinstance(lambda_C, torch.Tensor) else lambda_C  # [d]

    # Precompute constants
    ratio = (Omega / (Gamma + epsilon)) * C**2             # [d]
    c2 = torch.sqrt(lambda_r.squeeze(-1)**2 + ratio + epsilon)   # [d]
    c3 = -lambda_r.squeeze(-1)                             # [d]
    c1 = torch.log((torch.abs(c2 + c3) + epsilon) / (torch.abs(c2 - c3) + epsilon)) / (torch.abs(c2) + epsilon)  # [d]

    # Expand and compute safe tanh kernel
    t_diff = t_ji.unsqueeze(-1) - c1.view(1, 1, -1)                 # [m, m, d]
    x = 0.5 * c2.view(1, 1, -1) * t_diff
    full = c2.view(1, 1, -1) * torch.tanh(x)                        # [m, m, d]
    taylor = 0.5 * c2.view(1, 1, -1)**2 * t_diff                    # [m, m, d]
    use_taylor = (torch.abs(c2) < epsilon).view(1, 1, -1)
    safe_term = torch.where(use_taylor, taylor, full)              # [m, m, d]

    # Combine to get V_ij and add measurement noise
    factor = Gamma / (C**2 + epsilon)                              # [d]
    V_ij = factor.view(1, 1, -1) * (safe_term + lambda_r.squeeze(-1).view(1, 1, -1))  # [m, m, d]
    V_ij_dagger = C**2 * V_ij + Gamma.view(1, 1, -1) * mat_exp_b2[0]                  # [m, m, d]

    # Add batch and singleton dims: [1, m, m, d, 1]
    return V_ij_dagger.unsqueeze(0).unsqueeze(-1), (1 / (V_ij_dagger + epsilon)).unsqueeze(0).unsqueeze(-1)


