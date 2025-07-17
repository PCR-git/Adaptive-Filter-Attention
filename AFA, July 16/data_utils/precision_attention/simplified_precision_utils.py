
import numpy as np
import torch
import torch.nn as nn

from utils import batched_complex_matmul, batched_complex_matmul_full, batched_complex_hadamard, batched_complex_exp

###############################################
###############################################

def build_factorized_kernels(exp_f_k, exp_b_k, exp_f_v, exp_b_v, Q, K, V):
    
    V_k = exp_f_k.unsqueeze(0)
    V_v = exp_f_v.unsqueeze(0)
    U_q = exp_b_k.unsqueeze(0) * Q
    U_k = exp_b_k.unsqueeze(0) * K
    U_v = exp_b_v.unsqueeze(0) * V
    
    return V_k.squeeze(-1), V_v.squeeze(-1), U_q.squeeze(-1), U_k.squeeze(-1), U_v.squeeze(-1)

###############################################
###############################################

# def build_residual_kernels(U_k, U_q):

#     R_k = U_k[:,:,:-1] - U_k[:,:,1:]
#     R_k_cumsum = torch.cumsum(R_k, dim=2)
#     delta_qk = U_k - U_q
    
#     return R_k_cumsum, delta_qk

###############################################
###############################################

def compute_residual_norm(Q, K):
    
    """
    Computes the norm of R = Q - K, that is |R| = R*R.
    Computing this directly by first forming R requires broadcasting Q and K, which requires m^2 d memory.
    In order to compute this with less memory, we expand the product.
    """

    # Separate real and imaginary parts for Q and K
    # Q_real: [batch_size, num_queries, embed_dim]
    Q_real = Q[:, 0]
    Q_imag = Q[:, 1]

    # K_real: [batch_size, num_keys, embed_dim]
    K_real = K[:, 0]
    K_imag = K[:, 1]

    # Compute |Q|^2 for each query vector
    # |Q|^2 = Q_R^2 + Q_I^2 (element-wise square, then sum along embed_dim)
    Q_mag_sq = (torch.square(Q_real) + torch.square(Q_imag)).sum(dim=-1)

    # Compute |K|^2 for each key vector
    # |K|^2 = K_R^2 + K_I^2 (element-wise square, then sum along embed_dim)
    K_mag_sq = (torch.square(K_real) + torch.square(K_imag)).sum(dim=-1)

    # Compute Re(Q^* K) for all query-key pairs
    # Re(A^* B) = A_R^T B_R + A_I^T B_I
    # This involves two matrix multiplications (torch.matmul)
    # We need K_real_T and K_imag_T (transpose of last two dims)
    K_real_T = K_real.transpose(-1, -2) # Shape: [batch_size, embed_dim, num_keys]
    K_imag_T = K_imag.transpose(-1, -2) # Shape: [batch_size, embed_dim, num_keys]

    # term_RR = Q_R @ K_R.T : [batch_size, num_queries, num_keys]
    term_RR = torch.matmul(Q_real, K_real_T)
    # term_II = Q_I @ K_I.T : [batch_size, num_queries, num_keys]
    term_II = torch.matmul(Q_imag, K_imag_T)

    # real_dot_product_QK: [batch_size, num_queries, num_keys]
    real_dot_product_QK = term_RR + term_II

    # Assemble the final |Q - K|^2 matrix using broadcasting
    # Q_mag_sq: [batch_size, num_queries] needs to be [batch_size, num_queries, 1]
    Q_mag_sq_expanded = Q_mag_sq.unsqueeze(-1)
    # K_mag_sq: [batch_size, num_keys] needs to be [batch_size, 1, num_keys]
    K_mag_sq_expanded = K_mag_sq.unsqueeze(-2)

    # Resulting shape: [batch_size, num_queries, num_keys]
    R_squared_magnitude = Q_mag_sq_expanded + K_mag_sq_expanded - 2 * real_dot_product_QK
    
    return R_squared_magnitude

###############################################
###############################################

def compute_weighted_residual_norm(V_k, U_k, U_q): 
    """
    Computes the squared residual norm |R_qk|^2 between query vectors U_q and key vectors U_k,
    weighted by the squared magnitude of value vectors V_k.

    The residual norm is defined as:
        |R_qk|^2 := (|V_k|^2)^T |U_k|^2 
                    + sum_k (|V_k|^2 ⊙ |U_q|^2)
                    - 2 Re[ (|V_k|^2 ⊙ U_q)^H U_k ]

    Inputs:
        V_k: Tensor of shape (B, 2, m, d), complex-valued keys for value vectors (real, imag parts)
        U_k: Tensor of shape (B, 2, m, d), complex-valued key embeddings (real, imag parts)
        U_q: Tensor of shape (B, 2, m, d), complex-valued query embeddings (real, imag parts)

    Returns:
        R_qk_abs: Tensor of shape (B, m, m), squared residual norms between each query-key pair
    """

    # Compute |V_k|^2 = Re^2 + Im^2, sum over complex dim
    V_k_abs = torch.sum(V_k**2, dim=1)     # (B, m, d)

    # Compute |U_k|^2 and |U_q|^2 in same way
    U_k_abs = torch.sum(U_k**2, dim=1)     # (B, m, d)
    U_q_abs = torch.sum(U_q**2, dim=1)     # (B, m, d)

    # Term 1: (|V_k|^2)^T @ |U_k|^2 for each batch
    T1 = torch.matmul(V_k_abs, U_k_abs.permute(0, 2, 1))  # (B, m, m)

    # Term 2: sum_k (|V_k|^2 ⊙ |U_q|^2), sum over feature dim
    T2 = torch.sum(V_k_abs * U_q_abs, dim=2).unsqueeze(-1)  # (B, m, 1), broadcast later

    # Term 3: inner product between weighted U_q and U_k
    # Split into real and imaginary parts
    VU_kq = V_k_abs * U_q  # (B, 2, m, d), elementwise mult per real/imag part

    VU_kq_real = VU_kq[:, 0]  # (B, m, d)
    VU_kq_imag = VU_kq[:, 1]  # (B, m, d)
    U_k_real = U_k[:, 0]      # (B, m, d)
    U_k_imag = U_k[:, 1]      # (B, m, d)

    # Real part of inner product: Re(VU_q^H U_k) = dot(Re parts) + dot(Im parts)
    T3_RR = torch.matmul(VU_kq_real, U_k_real.permute(0, 2, 1))  # (B, m, m)
    T3_II = torch.matmul(VU_kq_imag, U_k_imag.permute(0, 2, 1))  # (B, m, m)
    T3 = T3_RR + T3_II  # (B, m, m)

    # Final residual norm: |R_qk|^2 = T1 + T2 - 2 * T3
    R_qk_abs = T1 + T2 - 2 * T3  # (B, m, m)

    return R_qk_abs


