import numpy as np
import torch
from utils import complex_matmul, complex_hadamard
from precision_attention.v1 import compute_kernel_v1, compute_estimates_and_residuals_vectorized
from precision_attention.compute_precision_old import compute_precision_v1

##########################################################################################
##########################################################################################

def precise_attn(X, lambda_h, lambda_Omega, lambda_Omega0, lambda_C, lambda_Gamma, W_q, W_k, W_v, W_r, W_o, t_v, args):
    """
    Precision-weighted attention block.

    Parameters:
      X (torch.Tensor): Input data.
      lambda_h (torch.Tensor): Diagonal of state transition matrix.
      lambda_Omega (torch.Tensor): Process noise covariance matrix.
      lambda_Omega0 (torch.Tensor): Initial condition of process noise covariance matrix.
      lambda_C (torch.Tensor): Measurement output matrix.
      lambda_Gamma (torch.Tensor): Measurement noise covariance.
      W_q, W_k, W_v, W_r, W_o (torch.Tensor): Query, key, value, residual, and output weight matrices.
      t_v (torch.Tensor): Time differences vector.
      args: Model and system parameters.

    Returns:
      out (torch.Tensor): Output tensor.
      Q_ij (torch.Tensor): Attention weights.
    """

    # A_ij = torch.stack((torch.eye(100,100),torch.eye(100,100)),axis=2).unsqueeze(0).unsqueeze(-1)

    # N = 100
    # i = torch.arange(N).view(1, -1)
    # j = torch.arange(N).view(-1, 1)
    # dist_squared = (1*(i - j))**2  # [100, 100]
    # A_base = 1.0 / (1.0 + dist_squared)  # [100, 100]
    # A_stack = torch.stack((A_base, A_base), dim=2)  # [100, 100, 2]
    # A_ij = A_stack.unsqueeze(0).unsqueeze(-1)  # [1, 100, 100, 2, 1]

    G1 = torch.sigmoid(torch.randn(args.seq_len,args.embed_dim,1))
    G = torch.stack((G1,torch.zeros(args.seq_len,args.embed_dim,1)))
    IG = torch.stack((1 - G1,torch.zeros(args.seq_len,args.embed_dim,1)))

    # Multiply by query, key, and value matrices
    X_q = complex_matmul(W_q,X)
    X_k = complex_matmul(W_k,X)
    X_v = complex_matmul(W_v,X)
    X_r = complex_matmul(W_r,X)
    
    # Compute kernel and residuals
    tji_v = torch.concatenate((-t_v.flip(0)[:-1],t_v)) # Rewrite so that this is passed in as an argument
    K_exp = compute_kernel_v1(lambda_h, t_v, args)
#     K_exp, _ = compute_kernel(lambda_h, t_v)
    X_ij_hat_all, R_qk_ij = compute_estimates_and_residuals_vectorized(X_q, X_k, X_v, K_exp, args)

    # Compute precision
    P_ij = compute_precision_v1(lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, tji_v, args, lambda_C=lambda_C)
#     P_ij = compute_precision_tanh(lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, tji_v, args, W_v=None, lambda_C=lambda_C)

    # Compute attention weights
    denom = (1 + args.nu*torch.sum(P_ij*(R_qk_ij[0]**2 + R_qk_ij[1]**2).unsqueeze(0),axis=3)).unsqueeze(3)
    # A_ij = P_ij
    A_ij = P_ij/denom

    # Normalize attention tensor
    S_ij = torch.sum(A_ij,axis=2).unsqueeze(2)
    Q_ij = A_ij/S_ij

    # Compute Hadamard product and sum to get estimate in diagonalized space
    #   est_v = torch.sum(Q_ij * X_ij_hat_all,axis=1)
    est_v = torch.sum(Q_ij * X_ij_hat_all,axis=2)

    # Residual connection
    # est_r = est_v
    est_r = args.alpha*est_v + (1-args.alpha)*X_r
    # est_r = complex_hadamard(G,est_v) + complex_hadamard(IG,X_r)

    # Multiply by output matrix to get estimate
    est = complex_matmul(W_o,est_r)

    # Get state transition matrix
    mat_exp = K_exp[:, -(args.seq_len+1), :, :]

    # Get prediction in diagonalized space
    pred_v = complex_hadamard(mat_exp, X_v) #

    # Multiply by output matrix to get output prediction
    pred = complex_matmul(W_o,pred_v)

    return est, pred, Q_ij, X_ij_hat_all

##########################################################################################
##########################################################################################

def precise_attn_with_correction(X, lambda_h, lambda_Omega, lambda_Omega0, lambda_C, lambda_Gamma, W_q, W_k, W_v, W_r, W_o, t_v, args):
  """
  Precision-weighted attention block, with correction term.

  Parameters:
      X (torch.Tensor): Input data.
      lambda_h (torch.Tensor): Diagonal of state transition matrix.
      lambda_Omega (torch.Tensor): Process noise covariance matrix.
      lambda_Omega0 (torch.Tensor): Initial condition of process noise covariance matrix.
      lambda_C (torch.Tensor): Measurement output matrix.
      lambda_Gamma (torch.Tensor): Measurement noise covariance.
      W_v, W_o (torch.Tensor): Value and output weight matrices.
      t_v (torch.Tensor): Time differences vector.
      nu (float): Scaling parameter.
      args: Model and system parameters.

  Returns:
      out (torch.Tensor): Output tensor.
      Q_ij (torch.Tensor): Attention weights.
  """

  # Multiply by query, key, and value matrices
  X_q = complex_matmul(W_q,X)
  X_k = complex_matmul(W_k,X)
  X_v = complex_matmul(W_v,X)

  # Compute kernel and residuals
  tji_v = torch.concatenate((-t_v.flip(0)[:-1],t_v)) # Rewrite so that this is passed in as an argument
  K_exp = compute_kernel_v1(lambda_h, t_v, args)
  X_ij_hat_all, R_qk_ij = compute_estimates_and_residuals_vectorized(X_q, X_k, X_v, K_exp)

  # Compute precision
  P_ij = compute_precision_v1(lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, tji_v, args, lambda_C)

  # Compute attention weights
  denom = (1 + nu*torch.sum(P_ij*(R_qk_ij[0]**2 + R_qk_ij[1]**2).unsqueeze(0),axis=3)).unsqueeze(3)
  A_ij = P_ij/denom

  # Compute correction term and add to attention tensor
  dot_prod = torch.sum(P_ij*(R_qk_ij[0]**2 + R_qk_ij[1]**2).unsqueeze(0),axis=3).unsqueeze(4)
  num = dot_prod * P_ij - (P_ij*(R_qk_ij[0] + R_qk_ij[1]**2))**2
  correction = nu*num/denom
  A_ij = A_ij + correction

  # Normalize attention tensor
  S_ij = torch.sum(A_ij,axis=2).unsqueeze(2)
  Q_ij = A_ij/S_ij

  # Compute Hadamard product and sum
  hadamard_prod_sum = torch.sum(Q_ij * X_ij_hat_all,axis=2)

  # Multiply by output matrix
  out = complex_matmul(W_o,hadamard_prod_sum)

  return out, Q_ij

##########################################################################################
##########################################################################################

def precise_attn_full(X, lambda_h, lambda_Omega, lambda_Omega0, lambda_C, lambda_Gamma, W_q, W_k, W_v, W_r, W_o, t_v, args):
  """
  Precision-weighted attention block, using full matrix inversion (not scalable, just used for comparison)

  Parameters:
      X (torch.Tensor): Input data.
      lambda_h (torch.Tensor): Diagonal of state transition matrix.
      lambda_Omega (torch.Tensor): Process noise covariance matrix.
      lambda_Omega0 (torch.Tensor): Initial condition of process noise covariance matrix.
      lambda_C (torch.Tensor): Measurement output matrix.
      lambda_Gamma (torch.Tensor): Measurement noise covariance.
      W_v, W_o (torch.Tensor): Value and output weight matrices.
      t_v (torch.Tensor): Time differences vector.
      nu (float): Scaling parameter.
      args: Model and system parameters.

  Returns:
      out (torch.Tensor): Output tensor.
      Q_ij (torch.Tensor): Attention weights.
  """

  # Multiply by query, key, and value matrices
  X_q = complex_matmul(W_q,X)
  X_k = complex_matmul(W_k,X)
  X_v = complex_matmul(W_v,X)

  # Compute kernel and residuals
  tji_v = torch.concatenate((-t_v.flip(0)[:-1],t_v)) # Rewrite so that this is passed in as an argument
  K_exp = compute_kernel_v1(lambda_h, t_v, args)
  X_ij_hat_all, R_qk_ij = compute_estimates_and_residuals_vectorized(X_q, X_k, X_v, K_exp, args)

  # Compute precision
  P_ij = compute_precision_v1(lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, tji_v, args, lambda_C) # [1, 100, 100, 2, 1]
  P_ij_mat = torch.diag_embed(P_ij.squeeze()) # [100, 2, 2]

  A2_ij = torch.zeros(100,100,d_e,d_e).to(args.device)
  RS_ij = torch.zeros(100,100,d_e,d_e).to(args.device)
  I = torch.eye(d_e,d_e).to(args.device)

  # Compute full inverse sum (wrong because it doesn't account for complex conj tranpose of R_qk_ij)
  for i in range(100):
    for j in range(100):
      r_ij = R_qk_ij[0,i,j]
      r_outer = torch.matmul(r_ij, torch.transpose(r_ij,0,1))
      p_ij = P_ij_mat[i,j]
      outer = torch.matmul(torch.matmul(p_ij, r_outer),p_ij)
      dot_prod = torch.sum(torch.diag(p_ij)*(r_ij.squeeze()**2))
      denom = 1/nu + dot_prod
      rs = outer/denom
      # RS_ij[i,j] = (dot_prod*I - torch.matmul(p_ij, r_outer))/denom
      A2_ij[i,j] = p_ij - rs

  S_ij = torch.sum(A2_ij,axis=2).unsqueeze(2)
  Si_ij = torch.inverse(S_ij)
  Q_ij = torch.matmul(Si_ij, A2_ij)

  # Compute Hadamard product and sum
  hadamard_prod_sum = torch.sum(torch.matmul(Q_ij, X_ij_hat_all),axis=2)

  # Multiply by output matrix
  out = complex_matmul(W_o,hadamard_prod_sum)

  return out, Q_ij

