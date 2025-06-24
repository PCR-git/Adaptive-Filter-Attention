import numpy as np
import torch
import torch.nn as nn

from utils import batched_complex_matmul, batched_complex_hadamard, batched_complex_exp

from precision_attention import compute_kernel
from precision_attention import batched_compute_estimates_and_residuals_vectorized, compute_estimates_and_residuals_irregular_times
from precision_attention import compute_precision, compute_precision_tanh
from model import compute_lambda_h, init_complex_matrix, init_weight_masks, apply_weight_masks

##########################################################################################
##########################################################################################

class BatchedPrecisionAttentionBlock(nn.Module):
    def __init__(self, head_dim, args):
        """
        Initializes the batched precision-weighted attention block.

        Parameters:
            W_q, W_k, W_v, W_o (torch.Tensor): Learnable weight matrices (query, key, value, and output).
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

        ################################################

        self.complex_identity = torch.stack((torch.eye(self.d_v, self.d_v),torch.zeros(self.d_v, self.d_v))).unsqueeze(1).to(args.device)

        sqrt_dv = torch.sqrt(torch.tensor(self.d_v))

        lambda_r = torch.randn(int(self.d_v/2))
        lambda_i = torch.randn(int(self.d_v/2))
        self.lambda1 = nn.Parameter(torch.stack((lambda_r,lambda_i)).unsqueeze(-1)) # Stack and scale by time interval
        self.lambda_h = torch.zeros(2,self.head_dim,1).to(args.device) # Initialize full eigenvalue array

        self.lambda_Omega_sqrt = nn.Parameter(torch.randn(1,self.d_v,1)/sqrt_dv) # Process covariance
        self.lambda_Omega0_sqrt = nn.Parameter(torch.randn(1,self.d_v,1)/sqrt_dv) # Initial process covariance
        self.lambda_Gamma_sqrt = nn.Parameter(torch.randn(1,self.d_v,1)/sqrt_dv) # Measurement covariance
    #     self.lambda_C = nn.Parameter(torch.randn(1,args.d_v,1)/sqrt_dv) # Measurement matrix
        self.lambda_C = torch.ones(1,self.d_v,1).to(args.device) # Output matrix

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

        ################################################

        self.args = args

        # self.G1 = nn.Parameter(torch.randn(args.seq_len,args.embed_dim,1))

        self.causal_mask = torch.tril(torch.ones(args.seq_len, args.seq_len)).view(1, args.seq_len, args.seq_len, 1, 1).to(args.device) # Causal attention mask

        # Relative weighting of attention (alpha) and residual (beta) connections (useful for diagnosis)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

        # Relative weighting of estimate (delta) and prediction (eta) in output
        self.delta = nn.Parameter(torch.tensor(0.0))
        self.eta = nn.Parameter(torch.tensor(1.0))

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

        self.lambda_h = compute_lambda_h(self.lambda1,self.args) # Get nonpositive complex conjugate eigenvalues

        # Take absolute value of noise parameters to ensure positive definiteness / non-negativeness
        self.lambda_Omega = self.lambda_Omega_sqrt**2 # Process noise matrix
        self.lambda_Omega0 = self.lambda_Omega0_sqrt**2 # Initial process noise uncertainty matrix
        self.lambda_Gamma = self.lambda_Gamma_sqrt**2 # Measurement noise matrix

        ############ (Masking; used for testing) ###########
        lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, W_q, W_k, W_v, W_p, W_r, W_e, W_q_b, W_k_b, W_v_b, W_p_b, W_r_b, W_e_b = apply_weight_masks(self, self.args)
        ####################################################

        X_q = X_q.unsqueeze(-1)
        X_k = X_k.unsqueeze(-1)
        X_v = X_v.unsqueeze(-1)

        # Project input into Q, K, V
    #     Q = batched_complex_matmul(W_q, X_q)
    #     K = batched_complex_matmul(W_k, X_k)
    #     V = batched_complex_matmul(W_v, X_v)
        Q = batched_complex_matmul(W_q, X_q) + W_q_b
        K = batched_complex_matmul(W_k, X_k) + W_k_b
        V = batched_complex_matmul(W_v, X_v) + W_v_b

    #     R = batched_complex_matmul(W_r, X_v)

        # G1 = torch.sigmoid(self.G1)
        # G = torch.stack((G1,torch.zeros(self.args.seq_len,self.args.embed_dim,1)))
        # IG = torch.stack((1 - G1,torch.zeros(self.args.seq_len,self.args.embed_dim,1)))

        if len(t_measure_all.size()) > 1:
            t_measure = t_measure_all[0,:-1]
        else:
            t_measure = t_measure_all[:,:-1]

        # Functionality for possibly unequal time intervals
        if self.args.t_equal == 1: # If equal time intervals
            K_exp, K_exp2 = compute_kernel(lambda_h, t_measure)
            X_ij_hat_all, R_qk_ij = batched_compute_estimates_and_residuals_vectorized(Q, K, V, K_exp, self.args)
            mat_exp = K_exp[:, -(self.args.seq_len+1), :, :] # Get matrix exponential for next-state prediction
        else: # If unequal time intervals
            X_ij_hat_all, R_qk_ij = compute_estimates_and_residuals_irregular_times(lambda_h, t_measure_all[:,:-1], Q, K, V, self.args)
            mat_exp = batched_complex_exp(lambda_h.squeeze(1).unsqueeze(0) * (t_measure_all[:,-1] - t_measure_all[:,-2]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
            K_exp2 = None
        
        if self.args.tanh == 0:
            P_ij = compute_precision(lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, K_exp2, t_measure_all[:,:-1], self.args, lambda_C=self.lambda_C)
        else:
            P_ij = compute_precision_tanh(lambda_h, lambda_Omega, lambda_Omega0, lambda_Gamma, K_exp2, t_measure_all[:,:-1], self.args, lambda_C=self.lambda_C)

        # Compute unnormalized attention matrix
        mahalanobis_distance = P_ij * (R_qk_ij[:,0]**2 + R_qk_ij[:,1]**2)
        denom = (1 + self.args.nu*torch.sum(mahalanobis_distance, axis=3, keepdims = True))
        A_ij = P_ij / denom
 
        A_ij = A_ij * self.causal_mask # Apply causal mask to attention matrix
        X_ij_hat_all = X_ij_hat_all * self.causal_mask # Mask out estimates backward in time (not strictly necessary but useful larter for visualization)

        # Normalize attention
        S_ij = torch.sum(A_ij, axis=2, keepdims = True)
        Q_ij = A_ij / S_ij

        # Compute Hadamard product and sum to get estimate in diagonalized space
        est_v = torch.sum(Q_ij.unsqueeze(1) * X_ij_hat_all,axis=3)

        # Add residual connection
        est_eigenbasis = est_v # No residual connection
    #     est_e = self.args.alpha*est_v + self.args.beta*V # JUST FOR TESTING
    #     est_e = est_v + self.alpha*(est_v - V) # JUST FOR TESTING
    #     est_e = est_v + R
    #     est_e = est_v + batched_complex_matmul(W_r, est_v - V)

        # Multiply by output matrix to get estimate
    #     est = batched_complex_matmul(W_e,est_eigenbasis)
        est = batched_complex_matmul(W_e,est_eigenbasis) + W_e_b
    #     est = batched_complex_matmul(W_p,est_eigenbasis)

        # Get prediction in diagonalized space
    #     pred_p = batched_complex_hadamard(mat_exp, est_e)
    #     pred_p = batched_complex_hadamard(lambda_h, est_e)*(self.args.n * self.args.dt) + est_e # JUST FOR TESTING
    #     pred_p = batched_complex_hadamard(mat_exp, V) # JUST FOR TESTING
    #     pred_p = batched_complex_hadamard(lambda_h, V)*(self.args.n * self.args.dt) + V # JUST FOR TESTING
        if self.args.t_equal == 1: # If equal time intervals
            pred_p = batched_complex_hadamard(mat_exp, est_eigenbasis)
        else:
            pred_p = batched_complex_hadamard_full(mat_exp.unsqueeze(2), est_eigenbasis)

        # Multiply by output matrix to get output prediction
    #     pred = batched_complex_matmul(self.W_p, pred_p)
    #     pred = batched_complex_matmul(W_p, pred_p)
        pred = batched_complex_matmul(W_p, pred_p) + W_p_b
    #     pred = batched_complex_matmul(self.W_p, batched_complex_hadamard(lambda_h, X_v))*self.args.dt + X_v # JUST FOR TESTING

        # Output is a linear combination of estimate and prediction
        out = self.args.delta*pred + self.args.eta*est
    #     out = self.delta*pred + self.eta*est
    #     out = pred + est

        est = est.squeeze(-1)
        out = out.squeeze(-1)
        X_ij_hat_all = X_ij_hat_all.squeeze(-1)

        return est, out, Q_ij, X_ij_hat_all, lambda_h