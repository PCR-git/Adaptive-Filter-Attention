import numpy as np
import torch
import torch.nn as nn

from utils import complex_matmul, complex_hadamard, batched_complex_matmul, batched_complex_hadamard, batched_complex_exp

from precision_attention import compute_kernel
from precision_attention import compute_estimates_and_residuals_vectorized, batched_compute_estimates_and_residuals_vectorized, compute_estimates_and_residuals_irregular_times
from precision_attention import compute_precision_v1, compute_precision, compute_precision_tanh
from model import compute_lambda_h, init_complex_matrix, init_weight_masks, apply_weight_masks

##########################################################################################
##########################################################################################

class PrecisionAttentionBlock(nn.Module):
    def __init__(self, args):
        """
        Initializes the precision-weighted attention block.

        Parameters:
            W_q, W_k, W_v, W_o (torch.Tensor): Learnable weight matrices.
            nu (float): Scaling parameter.
            args: Additional model/system parameters.
        """
        super().__init__()

        assert args.embed_dim % 2 == 0 # embed_dim must be divisible by 2 for lambda1 to make sense

        sqrt_de = torch.sqrt(torch.tensor(args.embed_dim, dtype=torch.float32))
        # sqrt_de = 1
        self.lambda1 = nn.Parameter(torch.randn(2,int(args.embed_dim/2),1)/sqrt_de) # State transition eigenvals

        self.lambda_Omega = nn.Parameter(torch.randn(1,args.embed_dim,1)/sqrt_de) # Process covariance
        self.lambda_Omega0 = nn.Parameter(torch.randn(1,args.embed_dim,1)/sqrt_de) # Initial process covariance
        self.lambda_Gamma = nn.Parameter(torch.randn(1,args.embed_dim,1)/sqrt_de) # Measurement covariance
        self.lambda_C = nn.Parameter(torch.randn(1,args.embed_dim,1)/sqrt_de) # Measurement matrix

        I = torch.stack((torch.eye(args.embed_dim, args.embed_dim),torch.zeros(args.embed_dim, args.embed_dim))).unsqueeze(1)
        self.W_q = nn.Parameter(torch.randn(2, 1, args.embed_dim, args.embed_dim)/sqrt_de) # Query weight matrix
        self.W_k = nn.Parameter(torch.randn(2, 1, args.embed_dim, args.embed_dim)/sqrt_de) # Key weight matrix
        self.W_v = nn.Parameter(torch.randn(2, 1, args.embed_dim, args.embed_dim)/sqrt_de) # Value weight matrix
        self.W_r = nn.Parameter(torch.randn(2, 1, args.embed_dim, args.embed_dim)/sqrt_de) # Residual weight matrix
        self.W_e = nn.Parameter(torch.randn(2, 1, args.embed_dim, args.embed_dim)/sqrt_de) # Estimate weight matrix
        self.W_p = nn.Parameter(torch.randn(2, 1, args.embed_dim, args.embed_dim)/sqrt_de) # Prediction weight matrix
    #     self.W_q = nn.Parameter(I + torch.randn(2, 1, args.embed_dim, args.embed_dim)/sqrt_de) # Query weight matrix
    #     self.W_k = nn.Parameter(I + torch.randn(2, 1, args.embed_dim, args.embed_dim)/sqrt_de) # Key weight matrix
    #     self.W_v = nn.Parameter(I + torch.randn(2, 1, args.embed_dim, args.embed_dim)/sqrt_de) # Value weight matrix
    #     self.W_r = nn.Parameter(I + torch.randn(2, 1, args.embed_dim, args.embed_dim)/sqrt_de) # Residual weight matrix
    #     self.W_e = nn.Parameter(I + torch.randn(2, 1, args.embed_dim, args.embed_dim)/sqrt_de) # Estimate weight matrix
    #     self.W_p = nn.Parameter(I + torch.randn(2, 1, args.embed_dim, args.embed_dim)/sqrt_de) # Prediction weight matrix

        self.args = args
    #     self.nu = 1 # Just set to 1 for now, since this can be absorbed into weight matrices

        # self.G1 = nn.Parameter(torch.randn(args.seq_len,args.embed_dim,1))

    def forward(self, X, t_v):
        """
        Forward pass through the precision-weighted attention block.

        Parameters:
            X (torch.Tensor): Input data.
            lambda_h (torch.Tensor): Diagonal of state transition matrix.
            lambda_Omega (torch.Tensor): Process noise covariance.
            lambda_Omega0 (torch.Tensor): Initial process noise covariance.
            lambda_C (torch.Tensor): Measurement output matrix.
            lambda_Gamma (torch.Tensor): Measurement noise covariance.
            t_v (torch.Tensor): Time differences vector.
            tji_v (torch.Tensor, optional): Time diff vector for precision.
                                            Computed if not provided.

        Returns:
            out (torch.Tensor): Output tensor.
            Q_ij (torch.Tensor): Normalized attention weights.
            X_ij_hat_all (torch.Tensor): Estimated values.
        """
        if tji_v is None:
            tji_v = torch.cat((-t_v.flip(0)[:-1], t_v))

        lambda_h = compute_lambda_h(self.lambda1,args) # Get complex conjugate eigenvals to ensure real state transition matrix

        lambda_Omega = torch.abs(self.lambda_Omega)
        lambda_Omega0 = torch.abs(self.lambda_Omega0)
        lambda_Gamma = torch.abs(self.lambda_Gamma)
        lambda_C = self.lambda_C

        # Project input into Q, K, V
        X_q = complex_matmul(self.W_q, X)
        X_k = complex_matmul(self.W_k, X)
        X_v = complex_matmul(self.W_v, X)
        X_r = complex_matmul(self.W_r, X)

        # G1 = torch.sigmoid(self.G1)
        # G = torch.stack((G1,torch.zeros(args.seq_len,args.embed_dim,1)))
        # IG = torch.stack((1 - G1,torch.zeros(args.seq_len,args.embed_dim,1)))

        # Kernel and estimates
    #     K_exp = compute_kernel_v1(lambda_h, t_v, args)
        K_exp, _ = compute_kernel(lambda_h, t_v)
        X_ij_hat_all, R_qk_ij = compute_estimates_and_residuals_vectorized(X_q, X_k, X_v, K_exp, args)

        # Compute precision matrix
        P_ij = compute_precision_v1(lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, tji_v, self.args, lambda_C)

        # Attention weights (with regularization)
        denom = (1 + self.nu*torch.sum(P_ij * (R_qk_ij[0]**2 + R_qk_ij[1]**2).unsqueeze(0), axis=3)).unsqueeze(3)
        A_ij = P_ij / denom

        # Normalize attention
        S_ij = torch.sum(A_ij, axis=2).unsqueeze(2)
        Q_ij = A_ij / S_ij

        # Compute Hadamard product and sum to get estimate in diagonalized space
        est_v = torch.sum(Q_ij * X_ij_hat_all,axis=2)

        # Residual connection
        # est_e = est_v
        # est_e = est_v + X_v
        est_e = args.alpha*est_v + (1-args.alpha)*X_r
        # est_e = complex_hadamard(G,est_v) + complex_hadamard(IG,X_r)

        # Multiply by output matrix to get estimate
        est = complex_matmul(self.W_e,est_e)

        # Get state transition matrix
        mat_exp = K_exp[:, -(args.seq_len+1), :, :]

        # Get prediction in diagonalized space
        pred_p = complex_hadamard(mat_exp, est_e)

        # Multiply by output matrix to get output prediction
        pred = complex_matmul(self.W_p,pred_p)

        out = args.delta*pred + (1-args.delta)*est

        return est, out, Q_ij, X_ij_hat_all
    

##########################################################################################
##########################################################################################
    
class BatchedPrecisionAttentionBlock_v1(nn.Module):
  def __init__(self, head_dim, args):
    """
    Initializes the batched precision-weighted attention block.

    Parameters:
        W_q, W_k, W_v, W_o (torch.Tensor): Learnable weight matrices (query, key, value, and output).
        nu (float): Scaling parameter.
        args: Additional model/system parameters.
    """
    
    super().__init__()
    
    self.head_dim = head_dim
    
    if args.d_v == None or args.d_k == None:
        self.d_v = head_dim
        self.d_k = head_dim
    else:
        self.d_v = args.d_v
        self.d_k = args.d_k
    
    #####################
    
#     self.complex_identity = torch.stack((torch.eye(args.embed_dim, args.embed_dim),torch.zeros(args.embed_dim, args.embed_dim))).unsqueeze(1).to(args.device)
    self.complex_identity = torch.stack((torch.eye(self.d_v, self.d_v),torch.zeros(self.d_v, self.d_v))).unsqueeze(1).to(args.device)
    
    sqrt_dv = torch.sqrt(torch.tensor(self.d_v))
  
    # Initialize state transition eigenvals
#     self.lambda1 = nn.Parameter(torch.randn(2,int(args.d_v/2),1)) # State transition eigenvals
#     self.lambda1 = nn.Parameter(torch.randn(2,int(args.d_v/2),1)/args.tf) # State transition eigenvals
#     self.lambda1 = nn.Parameter(2.0*torch.randn(2,int(args.d_v/2),1)) # State transition eigenvals

#     lambder_r = torch.rand(int(args.d_v/2))*(2*args.cr_max) - args.cr_max # Real part
#     lambder_i = 2*np.pi*(torch.rand(int(args.d_v/2))*(2*args.ci_max) - args.ci_max) # Imaginary part
    lambda_r = torch.randn(int(self.d_v/2))
    lambda_i = torch.randn(int(self.d_v/2))
    self.lambda1 = nn.Parameter(torch.stack((lambda_r,lambda_i)).unsqueeze(-1)) # Stack and scale by time interval
    self.lambda_h = torch.zeros(2,self.head_dim,1).to(args.device) # Initialize full eigenvalue array
    
    self.lambda_Omega_sqrt = nn.Parameter(torch.randn(1,self.d_v,1)/sqrt_dv) # Process covariance
    self.lambda_Omega0_sqrt = nn.Parameter(torch.randn(1,self.d_v,1)/sqrt_dv) # Initial process covariance
    self.lambda_Gamma_sqrt = nn.Parameter(torch.randn(1,self.d_v,1)/sqrt_dv) # Measurement covariance
#     self.lambda_C = nn.Parameter(torch.randn(1,args.d_v,1)/sqrt_dv) # Measurement matrix
    self.lambda_C = torch.ones(1,self.d_v,1).to(args.device) # Output matrix

#     self.W_q = nn.Parameter(init_complex_matrix(args.embed_dim)) # Query weight matrix
#     self.W_k = nn.Parameter(init_complex_matrix(args.embed_dim)) # Key weight matrix
#     self.W_v = nn.Parameter(init_complex_matrix(args.embed_dim)) # Value weight matrix
#     self.W_r = nn.Parameter(init_complex_matrix(args.embed_dim)) # Residual weight matrix
#     self.W_e = nn.Parameter(init_complex_matrix(args.embed_dim)) # Estimate output weight matrix
#     self.W_p = nn.Parameter(init_complex_matrix(args.embed_dim)) # Prediction output weight matrix

    self.W_q = nn.Parameter(init_complex_matrix(self.d_k, self.head_dim)) # Query weight matrix
    self.W_k = nn.Parameter(init_complex_matrix(self.d_k, self.head_dim)) # Key weight matrix
    self.W_v = nn.Parameter(init_complex_matrix(self.d_v, self.head_dim)) # Value weight matrix
    self.W_p = nn.Parameter(init_complex_matrix(self.head_dim, self.d_v)) # Prediction output weight matrix
    self.W_r = nn.Parameter(init_complex_matrix(self.d_v, self.head_dim)) # Residual weight matrix
    self.W_e = nn.Parameter(init_complex_matrix(self.head_dim, self.d_v)) # Estimate output weight matrix

    self.W_q_b = nn.Parameter(init_complex_matrix(self.d_k, 1).unsqueeze(0)) # Query weight matrix
    self.W_k_b = nn.Parameter(init_complex_matrix(self.d_k, 1).unsqueeze(0)) # Key weight matrix
    self.W_v_b = nn.Parameter(init_complex_matrix(self.d_v, 1).unsqueeze(0)) # Value weight matrix
    self.W_p_b = nn.Parameter(init_complex_matrix(self.head_dim, 1).unsqueeze(0)) # Prediction output weight matrix
    self.W_r_b = nn.Parameter(init_complex_matrix(self.d_v, 1).unsqueeze(0)) # Residual weight matrix
    self.W_e_b = nn.Parameter(init_complex_matrix(self.head_dim, 1).unsqueeze(0)) # Estimate output weight matrix

#     self.W_m = nn.Parameter(create_lower_triangular_parameter(args.seq_len).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) # Optional attention masking weight matrix
    
#     self.W_v += self.complex_identity
#     self.W_q += self.complex_identity
#     self.W_k += self.complex_identity
#     self.W_r += self.complex_identity
#     self.W_p += self.complex_identity
    
    #####################

    self.args = args
#     self.nu = 1 # Measurement weighting in attention; Just set to 1 for now, since this can be absorbed into weight matrices

    # self.G1 = nn.Parameter(torch.randn(args.seq_len,args.embed_dim,1))
    
    self.causal_mask = torch.tril(torch.ones(args.seq_len, args.seq_len)).view(1, args.seq_len, args.seq_len, 1, 1).to(args.device) # Causal attention mask
    
    # Relative weighting of attention (alpha) and residual (beta) connections (useful for diagnosis)
    self.alpha = nn.Parameter(torch.tensor(0.0))
    self.beta = nn.Parameter(torch.tensor(0.0))
    
    # Relative weighting of estimate (delta) and prediction (eta) in output
    self.delta = nn.Parameter(torch.tensor(0.0))
    self.eta = nn.Parameter(torch.tensor(1.0))
    
    self.alpha_nu = nn.Parameter(torch.ones(1,1,1,self.head_dim,1))
    self.beta_nu = nn.Parameter(torch.ones(1,1,1,self.head_dim,1))
    
    ############################################
    
    # Create masks for parameter matrices (used for testing)
    init_weight_masks(self, args)

    ############################################

  def forward(self, X_q, X_k, X_v, t_measure_all):
    """
    Forward pass through the precision-weighted attention block.

    Parameters:
        X (torch.Tensor): Input data.
        lambda_h (torch.Tensor): Diagonal of state transition matrix.
        lambda_Omega (torch.Tensor): Process noise covariance.
        lambda_Omega0 (torch.Tensor): Initial process noise covariance.
        lambda_C (torch.Tensor): Measurement output matrix.
        lambda_Gamma (torch.Tensor): Measurement noise covariance.
        t_measure_all (torch.Tensor): Time differences vector, for each trajectory in batch.

    Returns:
        out (torch.Tensor): Output tensor.
        Q_ij (torch.Tensor): Normalized attention weights.
        X_ij_hat_all (torch.Tensor): Estimated values.
    """

    self.lambda_h = compute_lambda_h(self.lambda1,args) # Get nonpositive complex conjugate eigenvalues
#     lambda_h = compute_lambda_h_v2(self.lambda1)
    
    # Take absolute value of noise parameters to ensure positive definiteness / non-negativeness
#     lambda_Omega = torch.abs(self.lambda_Omega) # Process noise matrix
#     lambda_Omega0 = torch.abs(self.lambda_Omega0) # Initial process noise uncertainty matrix
#     lambda_Gamma = torch.abs(self.lambda_Gamma) # Measurement noise matrix
    self.lambda_Omega = self.lambda_Omega_sqrt**2 # Process noise matrix
    self.lambda_Omega0 = self.lambda_Omega0_sqrt**2 # Initial process noise uncertainty matrix
    self.lambda_Gamma = self.lambda_Gamma_sqrt**2 # Measurement noise matrix
    
#     lambda_C = 1.0*torch.ones(1,args.embed_dim,1).to(args.device) # Output matrix
    
    ############ (Masking; used for testing) ###########
    lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, W_q, W_k, W_v, W_p, W_r, W_e, W_q_b, W_k_b, W_v_b, W_p_b, W_r_b, W_e_b = apply_weight_masks(self, args)
    ####################################################
    
    X_q = X_q.unsqueeze(-1)
    X_k = X_k.unsqueeze(-1)
    X_v = X_v.unsqueeze(-1)
    
    # Project input into Q, K, V
#     X_q = batched_complex_matmul(self.W_q, X)
#     X_k = batched_complex_matmul(self.W_k, X)
#     X_v = batched_complex_matmul(self.W_v, X)
#     X_r = batched_complex_matmul(self.W_r, X)

#     Q = batched_complex_matmul(W_q, X_q)
#     K = batched_complex_matmul(W_k, X_k)
#     V = batched_complex_matmul(W_v, X_v)
    Q = batched_complex_matmul(W_q, X_q) + W_q_b
    K = batched_complex_matmul(W_k, X_k) + W_k_b
    V = batched_complex_matmul(W_v, X_v) + W_v_b

#     X_r = batched_complex_matmul(W_r, X)

    # G1 = torch.sigmoid(self.G1)
    # G = torch.stack((G1,torch.zeros(args.seq_len,args.embed_dim,1)))
    # IG = torch.stack((1 - G1,torch.zeros(args.seq_len,args.embed_dim,1)))

#     # Compute kernel, estimates, and residuals
#     t_measure = t_measure_all[0,:-1]
#     tji_v = torch.cat((-t_measure.flip(0)[:-1], t_measure))
#     K_exp = compute_kernel_v1(lambda_h, t_measure, args)
#     X_ij_hat_all, R_qk_ij = batched_compute_estimates_and_residuals_vectorized(X_q, X_k, X_v, K_exp, args)
#     mat_exp = K_exp[:, -(self.args.seq_len+1), :, :] # Get matrix exponential for next-state prediction
#     # Compute precision matrix
#     P_ij = compute_precision_v1(lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, tji_v, self.args, lambda_C)

#     t_ji, t_measure = get_time_diffs(t_measure_all[:,:-1], args)
    if len(t_measure_all.size()) > 1:
        t_measure = t_measure_all[0,:-1]
    else:
        t_measure = t_measure_all[:,:-1]
    
    # Functionality for possibly unequal time intervals
    if args.t_equal == 1: # If equal time intervals
#         tji_v = torch.cat((-t_measure[:-1].flip(0)[:-1], t_measure[:-1]))
#         K_exp = compute_kernel_v1(lambda_h, t_measure, args)
        K_exp, K_exp2 = compute_kernel(lambda_h, t_measure)
        X_ij_hat_all, R_qk_ij = batched_compute_estimates_and_residuals_vectorized(Q, K, V, K_exp, args)
        mat_exp = K_exp[:, -(args.seq_len+1), :, :] # Get matrix exponential for next-state prediction
    else: # If unequal time intervals
        X_ij_hat_all, R_qk_ij = compute_estimates_and_residuals_irregular_times(lambda_h, t_measure_all[:,:-1], Q, K, V, args)
        mat_exp = batched_complex_exp(lambda_h.squeeze(1).unsqueeze(0) * (t_measure_all[:,-1] - t_measure_all[:,-2]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
        K_exp2 = None
#     if args.tanh == 0:
#         P_ij = compute_precision(lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, t_ji, args, lambda_C=lambda_C)
#     else:
#         P_ij = compute_precision_tanh(lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, t_ji, args, W_v=None, lambda_C=lambda_C)
    if args.tanh == 0:
        P_ij, nu = compute_precision(lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, K_exp2, t_measure_all[:,:-1], args, R_qk_ij=R_qk_ij, alpha_nu=self.alpha_nu, beta_nu=self.beta_nu, lambda_C=self.lambda_C)
#         P_ij = compute_precision(lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, t_ji, args)
#         nu = args.nu
    else:
        P_ij, nu = compute_precision_tanh(lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, K_exp2, t_measure_all[:,:-1], args, R_qk_ij=R_qk_ij, alpha_nu=self.alpha_nu, beta_nu=self.beta_nu, lambda_C=self.lambda_C)
    
    # Compute unnormalized attention matrix
    mahalanobis_distance = P_ij * (R_qk_ij[:,0]**2 + R_qk_ij[:,1]**2)
    denom = (1 + nu*torch.sum(mahalanobis_distance, axis=3, keepdims = True))
    A_ij = P_ij / denom
#     A_ij = P_ij # JUST FOR TESTING
    
#     A_ij = torch.eye(args.seq_len, args.seq_len).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(args.device) # JUST FOR TESTING
#     A_ij = build_nearly_identity(args) # JUST FOR TESTING
 
    A_ij = A_ij * self.causal_mask # Apply causal mask to attention matrix
    X_ij_hat_all = X_ij_hat_all * self.causal_mask # Mask out estimates backward in time (not strictly necessary but useful larter for visualization)
    
#     A_ij = A_ij * torch.sigmoid(self.W_m) # Optionally, apply learnable positional mask
    
    # Normalize attention
    S_ij = torch.sum(A_ij, axis=2, keepdims = True)
    Q_ij = A_ij / S_ij
    
    # Compute Hadamard product and sum to get estimate in diagonalized space
#     est_v = torch.sum(Q_ij * X_ij_hat_all,axis=3)
    est_v = torch.sum(Q_ij.unsqueeze(1) * X_ij_hat_all,axis=3)

    # Add residual connection
    est_eigenbasis = est_v # No residual connection
#     est_e = self.alpha*est_v + self.beta*X_v # JUST FOR TESTING
#     est_e = est_v + self.alpha*(est_v - X_v) # JUST FOR TESTING
#     est_e = est_v + X_r
#     est_e = est_v + batched_complex_matmul(W_r, est_v - X_v)
    
    # Multiply by output matrix to get estimate
#     est = batched_complex_matmul(W_e,est_eigenbasis)
    est = batched_complex_matmul(W_e,est_eigenbasis) + W_e_b
#     est = batched_complex_matmul(W_p,est_eigenbasis)
    
    # Get prediction in diagonalized space
#     pred_p = batched_complex_hadamard(mat_exp, est_e)
#     pred_p = batched_complex_hadamard(lambda_h, est_e)*(args.n*args.dt) + est_e # JUST FOR TESTING
#     pred_p = batched_complex_hadamard(mat_exp, X_v) # JUST FOR TESTING
#     pred_p = batched_complex_hadamard(lambda_h, X_v)*(args.n*args.dt) + X_v # JUST FOR TESTING
    if args.t_equal == 1: # If equal time intervals
        pred_p = batched_complex_hadamard(mat_exp, est_eigenbasis)
    else:
        pred_p = batched_complex_hadamard_full(mat_exp.unsqueeze(2), est_eigenbasis)

    # Multiply by output matrix to get output prediction
#     pred = batched_complex_matmul(self.W_p, pred_p)
#     pred = batched_complex_matmul(W_p, pred_p)
    pred = batched_complex_matmul(W_p, pred_p) + W_p_b
#     pred = batched_complex_matmul(self.W_p, batched_complex_hadamard(lambda_h, X_v))*args.dt + X # JUST FOR TESTING

    # Output is a linear combination of estimate and prediction
    out = args.delta*pred + args.eta*est
#     out = self.delta*pred + self.eta*est
#     out = pred + est

    est = est.squeeze(-1)
    out = out.squeeze(-1)
    X_ij_hat_all = X_ij_hat_all.squeeze(-1)
    
    return est, out, Q_ij, X_ij_hat_all, lambda_h