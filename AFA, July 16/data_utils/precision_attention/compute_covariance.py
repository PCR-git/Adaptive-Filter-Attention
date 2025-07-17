
import numpy as np
import torch
from precision_attention import compute_backwards_mat_exp

##########################################################################################
##########################################################################################

def compute_covariance_kernel(lambda_h, lambda_Omega, lambda_Gamma, K_exp, t_v, args, lambda_C=1.0, epsilon=1e-5):
    """
    Kernelized computation of the diagonal covariance matrix using precomputed exponentials.
    
    Inputs:
      lambda_h      : [2, d, 1] complex eigenvalues (real/imag parts).
      lambda_Omega  : [1, d, 1] process noise precision.
      lambda_Gamma  : [1, d, 1] measurement noise precision.
      K_exp        : [1, 2m-1, d, 1] matrix exponential kernel.
      args          : args with seq_len, device, dt.
      lambda_C      : scalar or [1, d, 1] output matrix magnitude.
    
    Returns:
      K_cov: [2m-1, d] covariance kernel such that V[i,j,k] = K_cov[k, |i-j|]
    """

    # Reshape
    lambda_real = lambda_h[0].squeeze(-1)   # [d]
    lambda_Omega = lambda_Omega.squeeze(0).squeeze(-1)  # [d]
    lambda_Gamma = lambda_Gamma.squeeze(0).squeeze(-1)  # [d]

#     K_mag_sq = (K_exp[0]**2 + K_exp[1]**2).squeeze(-1)   # shape: [2m-1, d, 1]
    K_mag_sq = (K_exp[0]**2 + K_exp[1]**2).squeeze(-1)[0:args.seq_len]   # shape: [2m-1, d, 1]

#     # Compute inverse precision kernel
#     denom = -2 * lambda_real + epsilon  # [d]
    
# #     direct_term = (1 - K_mag_sq + epsilon) / denom

    ##########################
    # Use Taylor approximation for stability
    
    #     mask = torch.abs(lambda_t) < stability_threshold
    mask = torch.abs(lambda_real) < epsilon
    
#     direct_term = (1-mask) * (1 - K_mag_sq) / (-2 * lambda_real)    
    direct_term = (1 - K_mag_sq + epsilon) / (-2 * lambda_real  + epsilon)
    
    t_flip = t_v.flip(dims=[0]).unsqueeze(-1) # Flip to be consistent with right-to-left ordering of time steps in kernel

#     lambda_t = lambda_real * t_flip

    # The Taylor expansion of (1-e^{-tx})/x (at x = 0) is t - t^2 x /2 = t ( 1 - t x /2)
    # Letting x = - 2 lambda_real, this is t ( 1 + t lambda_real)
#   The Taylor expansion of (1-e^(-tx) + epsilon)/(x + epsilon) is 1 + (t-1) x / epsilon = 1 - 2 (t-1) lambda_real / epsilon
#     taylor_approx = t_flip * (1 + lambda_t)
    taylor_approx = 1 - 2 * lambda_real * (t_flip - 1)  / epsilon

    stable_frac = torch.where(mask, taylor_approx, direct_term) # shape: [m, d]

#     stable_frac = direct_term
    ##########################

    term1 = lambda_C.squeeze(-1)**2 * lambda_Omega * stable_frac # [m, d]
    term2 = lambda_Gamma * K_mag_sq              # [m, d]

    K_cov = term1 + term2    # [m, d]
    
#     K_cov *= 0
#     K_cov += 1
    
    ones = torch.ones_like(K_cov[:-1,:])
    K_cov_full = torch.concatenate((K_cov, ones), dim=0)  # Matrix exponential kernel
    
    return K_cov_full  # [2m-1, d]

##########################################################################################
##########################################################################################

def build_covariance_from_kernel(K_cov, args, epsilon=1E-5):
    """
    Build full covariance tensor V_ij from kernel using custom indexing.

    Inputs:
      K_cov : [2m-1, d]     — covariance kernel.
      args  : args object with seq_len and device.

    Returns:
      V_ij  : [m, m, d]     — full covariance tensor.
    """

    m = args.seq_len

    # Generate kernel indices for all (i, j) pairs: [m, m]
    i_vals = torch.arange(m, device=args.device)
    start_indices = m - 1 - i_vals
    indices = start_indices.unsqueeze(1) + torch.arange(m, device=args.device).unsqueeze(0)  # [m, m]

    # Advanced indexing: K_cov is [d, 2m-1], indices is [m, m]
    # Broadcast K_cov over indices
    V_ij = K_cov[indices, :] + epsilon # [m, m, d]

    return V_ij.unsqueeze(0).unsqueeze(-1) # [m, m, d]

##########################################################################################
##########################################################################################

def build_avg_covariance_from_kernel(K_cov, args, w_v = None, epsilon=1E-5):
    """
    Build full covariance tensor V_ij from kernel using custom indexing.

    Inputs:
      K_cov : [2m-1, d]     — covariance kernel.
      args  : args object with seq_len and device.

    Returns:
      V_ij  : [m, m]     — full covariance tensor.
    """

    m = args.seq_len
    
    if w_v != None:
        K_cov = K_cov * w_v
    
    # Average over embed dim
#     K_cov_avg = torch.sum(K_cov,dim=1) / K_cov.size()[1] # [m, m]
    K_cov_avg = torch.mean(K_cov,dim=1) # [m, m]

    # Generate kernel indices for all (i, j) pairs: [m, m]
    i_vals = torch.arange(m, device=args.device)
    start_indices = m - 1 - i_vals
    indices = start_indices.unsqueeze(1) + torch.arange(m, device=args.device).unsqueeze(0)  # [m, m]

    # Advanced indexing: K_cov_avg is [2m-1], indices is [m, m]
    # Broadcast K_cov over indices
    V_avg_ij = K_cov_avg[indices] + epsilon # [m, m]

    return V_avg_ij.unsqueeze(0).unsqueeze(-1) # [m, m]

##########################################################################################
##########################################################################################

def compute_covariance_kernel_tanh(lambda_h, lambda_Omega, lambda_Gamma, K_exp, t_v, args, lambda_C=1.0, epsilon=1e-5):
    """
    Computes a tanh-based covariance kernel using precomputed exponentials and a hybrid
    solution combining the Riccati and Lyapunov cases for numerical stability.

    Inputs:
        lambda_h     : [2, d, 1] complex eigenvalues (real/imag parts).
        lambda_Omega : [1, d, 1] process noise.
        lambda_Gamma : [1, d, 1] measurement noise.
        K_exp        : [1, 2m-1, d, 1] complex exponentials (real, imag).
        t_v          : [m] time vector.
        args         : contains seq_len, device, dt.
        lambda_C     : scalar or [1, d, 1] measurement gain.
        epsilon      : numerical stability constant.

    Output:
        K_cov_full   : [2m-1, d] full covariance kernel.
    """
    m = args.seq_len

    # Squeeze dimensions to [d]
    lambda_real = lambda_h[0].squeeze(-1)
    Omega = lambda_Omega.squeeze(0).squeeze(-1)
    Gamma = lambda_Gamma.squeeze(0).squeeze(-1)
    C = lambda_C.squeeze(0).squeeze(-1) if isinstance(lambda_C, torch.Tensor) else lambda_C

    # Use only the first m values from the kernel (positive time diffs)
    K_mag_sq = (K_exp[0]**2 + K_exp[1]**2).squeeze(-1)[0:m]  # [m, d]

    # Compute stability-sensitive ratio
    ratio = (Omega / (Gamma + epsilon)) * C**2  # [d]
    use_lyapunov = (torch.abs(ratio) < epsilon) # [d]

    # --- Riccati (tanh) branch ---
    c2 = torch.sqrt(lambda_real**2 + ratio + epsilon)      # [d]
    c3 = -lambda_real
    c1_arg = c3 / (c2 + epsilon)
    c1_arg = torch.clamp(c1_arg, -1.0 + 1e-7, 1.0 - 1e-7)   # Prevent atanh overflow
    c1 = (2 / (c2 + epsilon)) * torch.atanh(c1_arg)         # [d]

    # tanh-based solution: V = factor * (c2 tanh + λ)
    t_diff = t_v.unsqueeze(-1) - c1.view(1, -1)              # [m, d]
    x = 0.5 * c2.view(1, -1) * t_diff
    tanh_term = c2.view(1, -1) * torch.tanh(x)
    factor = Gamma / (C**2 + epsilon)
    V_riccati = factor.view(1, -1) * (tanh_term + lambda_real.view(1, -1))  # [m, d]

    # --- Lyapunov branch (fallback when ratio ≈ 0) ---
    denom = -2 * lambda_real + epsilon
    V_lyapunov = (Omega / denom).view(1, -1) * (1 - K_mag_sq)  # [m, d]

    # Select per-dimension solution
    V_ij = torch.where(use_lyapunov.view(1, -1), V_lyapunov, V_riccati)  # [m, d]

    # Final kernel: V_ij + Gamma * |exp|^2
    K_cov_core = C**2 * V_ij + Gamma.view(1, -1) * K_mag_sq  # [m, d]

    # Extend to [2m-1, d] by padding with ones
    K_cov_full = torch.cat((K_cov_core, torch.ones_like(K_cov_core[:-1, :])), dim=0)

    return K_cov_full


##########################################################################################
##########################################################################################

# def compute_covariance_kernel_stable(lambda_h, lambda_Omega, lambda_Gamma, args, lambda_C=1.0, epsilon=1e-8, stability_threshold=1e-6):
#     """
#     Computes the precision kernel K_P[tau] directly from the given formula,
#     with numerical stability for Re(lambda_k) approaching zero.

#     Formula:
#     K_k^P[tau] = ( lambda_Ck^2 * lambda_Omegak * (1 - e^(2*Re(lambda_k)*dt*tau)) / (-2*Re(lambda_k))
#                  + lambda_Gammak * e^(2*Re(lambda_k)*dt*tau) )^(-1)

#     Parameters:
#       lambda_h (torch.Tensor): [2, d, 1] complex eigenvalues (real/imag parts).
#       lambda_Omega (torch.Tensor): [1, d, 1] process noise magnitude.
#       lambda_Gamma (torch.Tensor): [1, d, 1] measurement noise magnitude.
#       args: Object with attributes:
#             - seq_len (int): Length of the sequence (m).
#             - dt (float): Time interval.
#             - device (torch.device): Device for tensor operations.
#       lambda_C (torch.Tensor or float): [1, d, 1] output matrix magnitude or scalar.
#       epsilon (float): Small constant for numerical stability (added before inverse).
#       stability_threshold (float): Threshold for abs(X_k_tau) to use Taylor approximation.

#     Returns:
#       K_P (torch.Tensor): [d, 2m-1] precision kernel, where K_P[k, lag_idx]
#                           gives the precision for eigen dimension 'k' at 'lag_idx'.
#     """

#     m = args.seq_len
#     # Extract real part of eigenvalues, shape [d]
#     lambda_real = lambda_h[0].squeeze(-1) # shape: [d]

#     # Reshape noise and output matrix magnitudes to [d] for broadcasting
#     lambda_Omega_k = lambda_Omega.squeeze(0).squeeze(-1) # shape: [d]
#     lambda_Gamma_k = lambda_Gamma.squeeze(0).squeeze(-1) # shape: [d]
#     lambda_C_k_sq = (lambda_C.squeeze(0).squeeze(-1))**2 # shape: [d]

#     # Generate absolute time lags (tau values from 0 to (2m-2)*dt)
#     # This represents 'Delta t * tau' from the formula.
#     abs_time_lags = torch.arange(0, 2 * m - 1, device=args.device, dtype=torch.float32) * args.dt # shape: [2m-1]

#     # Calculate X_k_tau = 2 * Re(lambda_k) * Delta t * tau
#     # Shape: [d, 1] * [1, 2m-1] -> [d, 2m-1]
#     X_k_tau = 2 * lambda_real.unsqueeze(-1) * abs_time_lags.unsqueeze(0)

#     # --- Numerically stable calculation of (1 - e^X) / (-X) ---
#     # When X_k_tau is close to zero, use Taylor approximation: -1 - X_k_tau / 2
#     # Otherwise, use direct computation.
    
#     # Create a mask for elements where X_k_tau is very small
#     mask = torch.abs(X_k_tau) < stability_threshold

#     # Compute the direct term and the approximation term
#     direct_term = (1 - torch.exp(X_k_tau)) / (-X_k_tau)
#     taylor_approx = -1 - X_k_tau / 2.0

#     # Combine them using the mask
#     stable_fraction_term = torch.where(mask, taylor_approx, direct_term) # shape: [d, 2m-1]

#     # Calculate the exponential term: e^(2 * Re(lambda_k) * Delta t * tau)
#     # This is simply exp(X_k_tau)
#     exp_term = torch.exp(X_k_tau) # shape: [d, 2m-1]

#     # --- Assemble the terms for the denominator of the precision kernel ---
#     # Term 1: lambda_C_k^2 * lambda_Omega_k * stable_fraction_term
#     # Broadcasting: [d, 1] * [d, 1] * [d, 2m-1] -> [d, 2m-1]
#     term1 = lambda_C_k_sq.unsqueeze(-1) * lambda_Omega_k.unsqueeze(-1) * stable_fraction_term

#     # Term 2: lambda_Gamma_k * exp_term
#     # Broadcasting: [d, 1] * [d, 2m-1] -> [d, 2m-1]
#     term2 = lambda_Gamma_k.unsqueeze(-1) * exp_term

#     # Full denominator for the inverse (this represents the covariance kernel K_cov)
#     K_cov = term1 + term2 # shape: [d, 2m-1]

#     return K_cov
