
import numpy as np
import torch
import torch.nn as nn

# from utils import complex_exp, batched_complex_hadamard

##########################################################################################
##########################################################################################

def get_safe_exp_tot(t_measure, lambda_h, args):

    lambda_real = lambda_h[0]
    lambda_imag = lambda_h[1]
    
    t_exp = t_measure.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).squeeze(0) # [seq_len, 1, 1, 1]
    exp_tot_real = lambda_real * t_exp
    exp_tot_real_safe = torch.clamp(exp_tot_real, min=args.min_exponent, max=args.max_exponent)
    exp_tot_imag = lambda_imag * t_exp
    exp_tot_safe = torch.stack((exp_tot_real_safe, exp_tot_imag))

    return exp_tot_safe

##########################################################################################
##########################################################################################

def compute_exp_kernel_isotropic(omega, t_measure, exp_rel_safe):
    
    t_exp_rot = t_measure.view(t_measure.size()[0], 1, 1)
    omega_t = omega * t_exp_rot
    Phi_tilde_plus = torch.stack((torch.cos(omega_t), torch.sin(omega_t)),axis=-1)
    
    E_rel = torch.exp(exp_rel_safe).unsqueeze(0)

    return Phi_tilde_plus, E_rel

##########################################################################################
##########################################################################################

def compute_covariance_matrix(mu, sigma_squared, eta_squared, gamma_squared, E_rel, args, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Computes the full L x L covariance matrix V_ij directly using the pre-calculated 
    relative decay factor E_rel (e^(-alpha * |Delta T|)) and the analytical solution to the DLE.

    Inputs:
        mu (torch.Tensor): Decay [num_heads].
        lambda_sigma (torch.Tensor): Process noise parameter $\sigma^2$ [num_heads].
        lambda_eta (torch.Tensor): Key measurement noise parameter $\eta^2$ [num_heads].
        lambda_gamma (torch.Tensor): Query anchor noise parameter $\gamma^2$ [num_heads].
        E_rel (torch.Tensor): Relative decay factor $e^{-\alpha |\Delta T_{ij}|}$ [1, L, L, num_heads].
        args (Namespace): Model configuration.
        epsilon (float): Small constant.

    Returns:
        V_ij (torch.Tensor): Full covariance matrix V_{ij} = V( |\Delta T_{ij}| ) [L, L, num_heads].
    """
    
#     V^Z[i,j] = (eta^2 - sigma^2/(2 mu)  E_rel^2[i,j] + sigma^2/(2 mu) + gamma^2

    alpha = (eta_squared - sigma_squared/(-2*mu.squeeze(0))).unsqueeze(0).unsqueeze(1)
    beta = (sigma_squared/(-2*mu.squeeze(0)) + gamma_squared).unsqueeze(0).unsqueeze(1)
    V_ij = alpha * E_rel**2 + beta
    
    return V_ij.squeeze(0) # [L, L, num_heads]

# -----------------------------------------------

def compute_covariance_matrix_safe(mu, Delta_T, exp_rel_safe, sigma_squared, eta_squared, gamma_squared, t_measure, args, epsilon: float = 1e-5) -> torch.Tensor:
    """

    """
    
    # sigma_V^2(|i-j|) = sigma^2 (1 - e^(-2 mu |t_i-t_j|)/(2 mu)) + eta^2 e^{-2 mu |t_i-t_j|} + gamma^2

    numerator = -torch.expm1(2*exp_rel_safe).unsqueeze(0)  # [1, m, m, n_heads]
    
#     mask = torch.abs(mu) < epsilon
#     denom_safe = torch.where(mask, torch.ones_like(mu), -2 * mu_safe).unsqueeze(0)
    
#     direct_term = numerator / denom_safe
#     stable_frac = torch.where(mask, Delta_T, direct_term) # L'Hopital limit as mu -> 0 is simply Delta_T

#     mu = mu.squeeze()
    mask = torch.abs(mu) < epsilon
    denom_safe = torch.where(mask, torch.ones_like(mu), -2 * mu)
    direct_term = numerator / denom_safe
    stable_frac = torch.where(mask, Delta_T, direct_term) # L'Hopital limit as mu -> 0 is simply Delta_T
    
    term1 = sigma_squared.unsqueeze(0).unsqueeze(1) * stable_frac # [m, d]
    term2 = eta_squared.unsqueeze(0).unsqueeze(1) * torch.exp(2*exp_rel_safe)     # [m, d]
    term3 = gamma_squared.unsqueeze(0).unsqueeze(1).unsqueeze(0)

    V_ij = term1 + term2 + term3

    return V_ij.squeeze(0) # [L, L, num_heads]

##########################################################################################
##########################################################################################

# def build_factorized_kernels(Phi_tilde_minus_k, Phi_tilde_minus_v, Q, K, V, args):
#     """
#     Build factorized kernels for use in simplified AFA
#     """
    
#     Q_tilde = batched_complex_hadamard(Phi_tilde_minus_k, Q)
#     K_tilde = batched_complex_hadamard(Phi_tilde_minus_k, K)
    
#     if args.rotate_values == 1:
#         V_tilde = batched_complex_hadamard(Phi_tilde_minus_v, V)
#     else:
#         V_tilde = V
#         print('Not rotating values.')
    
#     return Q_tilde, K_tilde, V_tilde


##########################################################################################
##########################################################################################

def compute_residual_norm_isotropic(Q_tilde, K_tilde, E_rel_k, args):
    """
    Computes the stable squared residual norm |R_qk|^2 using the factorized formula 
    from the Isotropic Adaptive Filter Attention (AFA).

    |R_qk|^2 = |Q_tilde|^2 + E_qk[i,j]^2 * |K_tilde|^2 - 2 * E_qk[i,j] * Re(Q_tilde^* K_tilde)

    Inputs:
        Q: Unrotated query
        Q_tilde (torch.Tensor): Rotated Query vectors (Phi^- * Q). [B, 2, m, d_k, H]
        K_tilde (torch.Tensor): Rotated Key vectors (Phi^- * K). [B, 2, m, d_k, H]
        E_rel_k (torch.Tensor): Decay matrix E_qk[i,j] = e^(alpha*(t_i-t_j)). [m, m, H]

    Returns:
        R_qk_abs_squared: Tensor of shape (B, m, m, H), per-head squared residuals.
    """
    
    # Q/K_tilde shape: [B, 2, m, d, H]. 
    # The magnitude is the same as the unrotated input |Z|^2.

    # 1. Calculate total magnitude squared: |Z_tilde|^2 = sum_d (Re^2 + Im^2)
    # Sum over Real/Imag (dim 1) and Feature (dim 3) dimensions.
    # Resulting shape: [B, m, H]
    
    # We could normalize either Q or Q_tilde
    Q_mag_sq_sum = torch.sum(Q_tilde**2, dim=[-1, -2]) # Normalize rotated Q_tilde

    # --- Term 1: |Z_q|^2 (Broadcast over Keys) ---
    # Shape: [B, m, H] -> [B, m, 1, H]
    T1 = Q_mag_sq_sum.unsqueeze(2)

    # --- Term 2: E_qk^2 * |Z_k|^2 (Broadcast over Queries) ---
    K_mag_sq_sum = torch.sum(K_tilde**2, dim=[-1, -2])

    T2 = (E_rel_k**2) * K_mag_sq_sum.unsqueeze(1) # [B, m, m, H]

    # --- Term 3: -2 * E_rel_k * Re(Q_tilde^H K_tilde) ---
    # Calculate the complex inner product Re[Z_tilde_q^H Z_tilde_k]
    # We use torch.einsum to calculate the dot product across the feature dimension 'd' (dim 3).
    
    Q_tilde_re = Q_tilde[..., 0]
    Q_tilde_im = Q_tilde[..., 1]
    K_tilde_re = K_tilde[..., 0]
    K_tilde_im = K_tilde[..., 1]
    
    Q_c = torch.cat((Q_tilde_re, Q_tilde_im), axis=-1).permute(0,2,1,3).contiguous()
    K_c = torch.cat((K_tilde_re, K_tilde_im), axis=-1).permute(0,2,3,1).contiguous()

    dot_product = torch.matmul(Q_c,K_c).permute(0,2,3,1).contiguous()

    # Combine with E_qk factor
    T3 = 2 * E_rel_k * dot_product # [B, m, m, H]

    if args.use_full_residual_norm:
        # Final residual norm: |R_qk|^2 = T1 + T2 - T3
        R_qk_abs_squared = T1 + T2 - T3
    else:
        R_qk_abs_squared = - T3
        print('Using dot product attention (not using residual norm).')

    return R_qk_abs_squared
