
import numpy as np
import torch
import torch.nn as nn

from utils import complex_exp, complex_conj_transpose, batched_complex_hadamard

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

def compute_exp_kernel_isotropic(lambda_h, t_measure, exp_rel_safe):
    """
    Computes the four factor kernels (Rotation Phi_tilde and Decay E factors) 
    required for the stable, factorized Isotropic Adaptive Filter Attention (AFA).

    Args:
        lambda_h (torch.Tensor): Full Complex decay matrix [2, d_head/2, n_heads, 1].
        t_measure (torch.Tensor): Time vector [seq_len].

    Returns:
        Phi_tilde_plus (torch.Tensor): e^(+i*omega*t) factor [2, seq_len, d_head/2, n_heads].
        Phi_tilde_minus (torch.Tensor): e^(-i*omega*t) factor [2, seq_len, d_head/2, n_heads].
        E_plus (torch.Tensor): e^(mu*t) factor [seq_len, n_heads].
        E_rel (torch.Tensor): e^(mu*(t_i-t_j)) factor [seq_len, seq_len, n_heads].
    """
    
    seq_len = t_measure.shape[0]

    # --- Prepare Inputs ---
    # mu (real part of lambda_h) is a shared scalar across d/H dimensions.
    # lambda_h[0] is real part (mu), lambda_h[1] is imaginary part (omega)
#     mu = torch.mean(lambda_h,axis=2,keepdims=True)[0].unsqueeze(0)
    omega = lambda_h[1]

    # Expand t_measure for kernel calculation
    t_exp_rot = t_measure.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4) # [1, seq_len, 1, 1, 1]

    # --- 1. Rotational Factors (Phi_tilde) ---
    # Create an imaginary-only lambda: [real=0, imag=omega]
    lambda_rot = torch.stack([torch.zeros_like(omega), omega], dim=0) # [2, d_head, n_heads, 1]

    # Phi_tilde_plus: e^(+i*omega*t)
    Phi_tilde_plus = complex_exp(lambda_rot * t_exp_rot).squeeze(-1) # [2, seq_len, d_head, n_heads]

    # Phi_tilde_minus: e^(-i*omega*t)
    Phi_tilde_minus = complex_exp(-lambda_rot * t_exp_rot).squeeze(-1) # [2, seq_len, d_head, n_heads]
    
#     # --- 2. Relative Decay Factor (E_rel) ---
#     # Calculate Time Difference Matrix Delta_T[i, j] = |t_i - t_j|
#     t_measure_i = t_measure.squeeze().unsqueeze(1)  # [m, 1]
#     t_measure_j = t_measure.squeeze().unsqueeze(0)  # [1, m]
#     Delta_T = torch.abs(t_measure_i - t_measure_j)  # [m, m]
#     Delta_T_exp = Delta_T.unsqueeze(-1) # Expand Delta_T: [m, m, 1]
#     mu_exp = mu.squeeze().unsqueeze(0).unsqueeze(1) # Expand mu: [1, 1, n_heads]
#     E_rel = torch.exp(mu_exp * Delta_T_exp).unsqueeze(0)  # [1, m, m, n_heads]
    
    E_rel = torch.exp(exp_rel_safe).unsqueeze(0)

    return Phi_tilde_plus, Phi_tilde_minus, E_rel

# -----------------------------------------------

def compute_exp_kernel_anisotropic(t_measure, lambda_h, args):
    """

    """
    
    exp_tot_safe = get_safe_exp_tot(t_measure, lambda_h, args)
    
    Phi_hat_plus = complex_exp(exp_tot_safe).squeeze(-1) # [2, seq_len, d_head, n_heads]
    Phi_hat_minus = complex_exp(-exp_tot_safe).squeeze(-1) # [2, seq_len, d_head, n_heads]

    return Phi_hat_plus, Phi_hat_minus

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

def compute_covariance_matrix_safe(mu_safe, Delta_T, exp_rel_safe, sigma_squared, eta_squared, gamma_squared, t_measure, args, epsilon: float = 1e-5) -> torch.Tensor:
    """

    """
    
    # sigma_V^2(|i-j|) = sigma^2 (1 - e^(-2 mu |t_i-t_j|)/(2 mu)) + eta^2 e^{-2 mu |t_i-t_j|} + gamma^2

    numerator = -torch.expm1(2*exp_rel_safe).unsqueeze(0)  # [1, m, m, n_heads]
    
    mask = torch.abs(mu_safe) < epsilon
    denom_safe = torch.where(mask, torch.ones_like(mu_safe), -2 * mu_safe).unsqueeze(0)
    
    direct_term = numerator / denom_safe
    stable_frac = torch.where(mask, Delta_T, direct_term) # L'Hopital limit as mu -> 0 is simply Delta_T
    
    term1 = sigma_squared.unsqueeze(0).unsqueeze(1) * stable_frac # [m, d]
    term2 = eta_squared.unsqueeze(0).unsqueeze(1) * torch.exp(2*exp_rel_safe)     # [m, d]
    term3 = gamma_squared.unsqueeze(0).unsqueeze(1).unsqueeze(0)

    V_ij = term1 + term2 + term3

    return V_ij.squeeze(0) # [L, L, num_heads]

##########################################################################################
##########################################################################################

def build_factorized_kernels(Phi_tilde_minus_k, Phi_tilde_minus_v, Q, K, V, args):
    """
    Build factorized kernels for use in simplified AFA
    """
    
    Q_tilde = batched_complex_hadamard(Phi_tilde_minus_k, Q)
    K_tilde = batched_complex_hadamard(Phi_tilde_minus_k, K)
    
    if args.rotate_values == 1:
        V_tilde = batched_complex_hadamard(Phi_tilde_minus_v, V)
    else:
        V_tilde = V
        print('Not rotating values.')
    
    return Q_tilde, K_tilde, V_tilde

##########################################################################################
##########################################################################################

def compute_residual_norm_isotropic(Q, Q_tilde, K_tilde, E_rel_k, args):
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
#     Q_mag_sq_sum = torch.sum(Q**2, dim=[1, 3]) # Normalize the unrotated Q
    Q_mag_sq_sum = torch.sum(Q_tilde**2, dim=[1, 3]) # Normalize rotated Q_tilde

    # --- Term 1: |Z_q|^2 (Broadcast over Keys) ---
    # Shape: [B, m, H] -> [B, m, 1, H]
    T1 = Q_mag_sq_sum.unsqueeze(2) 

    # --- Term 2: E_qk^2 * |Z_k|^2 (Broadcast over Queries) ---
    # E_rel_k squared is broadcast to batch B. [m, m, H] -> [B, m, m, H]
    # K_mag_sq_sum is broadcast to [B, 1, m, H].
    K_mag_sq_sum = torch.sum(K_tilde**2, dim=[1, 3])
    T2 = (E_rel_k**2) * K_mag_sq_sum.unsqueeze(1) # [B, m, m, H]
    
#     print(E_rel_k[0:5,0:5])

    # --- Term 3: -2 * E_rel_k * Re(Q_tilde^H K_tilde) ---
    # Calculate the complex inner product Re[Z_tilde_q^H Z_tilde_k]
    # We use torch.einsum to calculate the dot product across the feature dimension 'd' (dim 3).

    Q_tilde_re = Q_tilde[:, 0] 
    Q_tilde_im = Q_tilde[:, 1]
    K_tilde_re = K_tilde[:, 0]
    K_tilde_im = K_tilde[:, 1]
    
    Q_c = torch.cat((Q_tilde_re, Q_tilde_im), axis=2).permute(0,3,1,2).contiguous()    
    K_c = torch.cat((K_tilde_re, K_tilde_im), axis=2).permute(0,3,2,1).contiguous()
    real_dot_product_qk = torch.matmul(Q_c,K_c).permute(0,2,3,1).contiguous()
    
    # Combine with E_qk factor
    T3 = 2 * E_rel_k * real_dot_product_qk # [B, m, m, H]

    if args.use_full_residual_norm:
        # Final residual norm: |R_qk|^2 = T1 + T2 - T3
        R_qk_abs_squared = T1 + T2 - T3
    else:
        R_qk_abs_squared = - T3
        print('Using dot product attention (not using residual norm).')
        
    seq_len = Q.size()[2]
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=args.device)).unsqueeze(0).unsqueeze(-1)
    
    return R_qk_abs_squared * causal_mask

##########################################################################################
##########################################################################################

def compute_residual_norm_anisotropic(Q, K_hat, Phi_hat_plus, args):
    """

    """
    
    # Term 1:
    Q_mag_sq_sum = torch.sum(Q**2, dim=[1, 3])
    T1 = Q_mag_sq_sum.unsqueeze(2) 
    
    # Term 2:
    
    # Norms of K_hat and Phi_hat_plus
    K_hat_sq = torch.sum(K_hat**2, dim=1)
    Phi_sq = torch.sum(Phi_hat_plus**2, dim=0)

    # Align shapes to [Batch, Head, Seq, Dim]
    K_mat = K_hat_sq.permute(0, 3, 1, 2)
    Phi_mat = Phi_sq.permute(2, 1, 0).unsqueeze(0) 

    # Perform Matmul: [B, H, L_i, D] @ [B, H, D, L_j]
    # We transpose Phi_mat because we want to multiply across D
    T2_raw = torch.matmul(Phi_mat.transpose(-1, -2), K_mat.transpose(-1, -2))

    # Final permute to align with attention scores [B, L_i, L_j, H]
    T2 = T2_raw.permute(0, 2, 3, 1)

    # --- Term 3: -2 * E_rel_k * Re(Q_hat^H K_hat) ---
    # Calculate the complex inner product Re[Z_hat_q^H Z_hat_k]
    # We use torch.einsum to calculate the dot product across the feature dimension 'd' (dim 3).
    
    Phi_hat_plus_star = complex_conj_transpose(Phi_hat_plus)
    
    Q_hat_plus = batched_complex_hadamard(Phi_hat_plus_star, Q)

    Q_hat_re = Q_hat_plus[:, 0] 
    Q_hat_im = Q_hat_plus[:, 1]
    K_hat_re = K_hat[:, 0]
    K_hat_im = K_hat[:, 1]

    Q_c = torch.cat((Q_hat_re, Q_hat_im), axis=2).permute(0,3,1,2).contiguous()
    K_c = torch.cat((K_hat_re, K_hat_im), axis=2).permute(0,3,2,1).contiguous()
    real_dot_product_qk = torch.matmul(Q_c,K_c).permute(0,2,3,1).contiguous()
    
    # Combine with E_qk factor
    T3 = 2 * real_dot_product_qk # [B, m, m, H]

    if args.use_full_residual_norm:
        # Final residual norm: |R_qk|^2 = T1 + T2 - T3
        R_qk_abs_squared = T1 + T2 - T3
    else:
        R_qk_abs_squared = - T3
        print('Using dot product attention (not using residual norm).')
        
#     seq_len = Q.size()[2]
#     causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=args.device)).unsqueeze(0).unsqueeze(-1)
#     R_qk_abs_squared = R_qk_abs_squared * causal_mask

    return R_qk_abs_squared
