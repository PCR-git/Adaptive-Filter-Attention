import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import batched_complex_matmul, batched_complex_hadamard, batched_complex_exp, batched_complex_matmul_full

from precision_attention import compute_exp_kernel
from precision_attention import batched_compute_estimates_and_residuals_vectorized, compute_estimates_and_residuals_irregular_times
from precision_attention import compute_precision, compute_precision_tanh
from precision_attention import compute_covariance_kernel, compute_covariance_kernel_tanh, build_covariance_from_kernel
from model import compute_lambda_h, init_complexlinear, init_complex_matrix, init_weight_masks, apply_weight_masks, apply_net_weight_masks
from model import ComplexLinearLayer

##########################################################################################
##########################################################################################

class FullPrecisionAttentionBlock(nn.Module):
    def __init__(self, input_dim, query_key_dim, value_dim, args):
        """
        Initializes the batched precision-weighted attention block.

        Parameters:
            W_q, W_k, W_v, W_o (torch.Tensor): Learnable weight matrices (query, key, value, and output).
            args: Additional model/system parameters.
        """

        super().__init__()
            
        # Store dimensions as instance attributes
        self.d_e = input_dim
        self.d_k = query_key_dim
        self.d_v = value_dim
            
        self.W_q = ComplexLinearLayer(self.d_e, self.d_k)
        self.W_k = ComplexLinearLayer(self.d_e, self.d_k)

        self.W_v = ComplexLinearLayer(self.d_e, self.d_v)
        self.W_p = ComplexLinearLayer(self.d_v, self.d_e)
#         self.W_e = ComplexLinearLayer(self.d_e, self.d_v)
#         self.W_r = ComplexLinearLayer(self.d_e, self.d_v) 

        ################################################

        # Complex-valued identity matrix
#         self.complex_identity = torch.stack((torch.eye(self.d_v, self.d_v),torch.zeros(self.d_v, self.d_v))).unsqueeze(1).to(args.device)
        self.register_buffer("complex_identity", torch.stack((torch.eye(self.d_v, self.d_v), \
                                                              torch.zeros(self.d_v, self.d_v))).unsqueeze(1))
        sqrt_dv = torch.sqrt(torch.tensor(self.d_v)) # Scaling constant
        
        ######################
        
        lambda_r = torch.randn(int(self.d_v/2))
        lambda_i = torch.randn(int(self.d_v/2))
        self.lambda1 = nn.Parameter(torch.stack((lambda_r,lambda_i)).unsqueeze(-1)) # Stack and scale by time interval
#         self.lambda_h = torch.zeros(2,self.d_v,1).to(args.device) # Initialize full eigenvalue array
        self.register_buffer("lambda_h", torch.zeros(2,self.d_v,1))

#         self.lambda_Omega = nn.Parameter(torch.randn(1,self.d_v,1)/sqrt_dv) # Process covariance (if using **2)
#         self.lambda_Gamma = nn.Parameter(torch.randn(1,self.d_v,1)/sqrt_dv) # Measurement covariance (if using **2)
        self.lambda_Omega = nn.Parameter(torch.randn(1,self.d_v,1)/args.d_v) # Process covariance (if using abs)
        self.lambda_Gamma = nn.Parameter(torch.randn(1,self.d_v,1)/args.d_v) # Measurement covariance (if using abs)
        
        # If using separate key and value params:
        if args.sep_params == 1:
            
            sqrt_dk = torch.sqrt(torch.tensor(self.d_k))

            lambda_r_k = torch.randn(int(self.d_k/2))
            lambda_i_k = torch.randn(int(self.d_k/2))
            self.lambda1_k = nn.Parameter(torch.stack((lambda_r_k,lambda_i_k)).unsqueeze(-1)) # Stack and scale by time interval
#             self.lambda_h_k = torch.zeros(2,self.d_v,1).to(args.device) # Initialize full eigenvalue array
            self.register_buffer("lambda_h", torch.zeros(2,self.d_v,1))

#             self.lambda_Omega_k = nn.Parameter(torch.randn(1,self.d_k,1)/sqrt_dk) # Process covariance (if using **2)
#             self.lambda_Gamma_k = nn.Parameter(torch.randn(1,self.d_k,1)/sqrt_dk) # Measurement covariance (if using **2)
            self.lambda_Omega_k = nn.Parameter(torch.randn(1,self.d_k,1)/self.d_k) # Process covariance (if using abs)
            self.lambda_Gamma_k = nn.Parameter(torch.randn(1,self.d_k,1)/self.d_k) # Measurement covariance (if using abs)
            
        ######################
        
        self.noise_floor = nn.Parameter(torch.tensor(1.0))
        
        self.tau = nn.Parameter(torch.tensor(1.0))
        self.nu = nn.Parameter(torch.tensor(1.0))
        
    #     self.lambda_C = nn.Parameter(torch.randn(1,args.d_v,1)/sqrt_dv) # Measurement matrix
        self.lambda_C = torch.ones(1,self.d_v,1).to(args.device) # Output matrix

#         self.W_q = nn.Parameter(init_complex_matrix(self.d_k, self.d_v)) # Query weight matrix
#         self.W_k = nn.Parameter(init_complex_matrix(self.d_k, self.d_v)) # Key weight matrix
#         self.W_v = nn.Parameter(init_complex_matrix(self.d_v, self.d_v)) # Value weight matrix
#         self.W_p = nn.Parameter(init_complex_matrix(self.d_v, self.d_v)) # Prediction output weight matrix
#         self.W_r = nn.Parameter(init_complex_matrix(self.d_v, self.d_v)) # Residual weight matrix
#         self.W_e = nn.Parameter(init_complex_matrix(self.d_v, self.d_v)) # Estimate output weight matrix

#         self.W_q_b = nn.Parameter(init_complex_matrix(self.d_k, 1).unsqueeze(0)) # Query weight matrix
#         self.W_k_b = nn.Parameter(init_complex_matrix(self.d_k, 1).unsqueeze(0)) # Key weight matrix
#         self.W_v_b = nn.Parameter(init_complex_matrix(self.d_v, 1).unsqueeze(0)) # Value weight matrix
#         self.W_p_b = nn.Parameter(init_complex_matrix(self.d_v, 1).unsqueeze(0)) # Prediction output weight matrix
#         self.W_r_b = nn.Parameter(init_complex_matrix(self.d_v, 1).unsqueeze(0)) # Residual weight matrix
#         self.W_e_b = nn.Parameter(init_complex_matrix(self.d_v, 1).unsqueeze(0)) # Estimate output weight matrix

        Wqi, bqi = init_complex_matrix(args.d_e, args.d_k, bias=True)
        Wki, bki = init_complex_matrix(args.d_e, args.d_k, bias=True)
        Wvi, bvi = init_complex_matrix(args.d_e, args.d_v, bias=True)
#         Wei, bei = init_complex_matrix(args.d_e, args.d_v, bias=True)
        Wpi, bpi = init_complex_matrix(args.d_e, args.d_v, bias=True)
        init_complexlinear(self.W_q, Wqi, bqi)
        init_complexlinear(self.W_k, Wki, bki)
        init_complexlinear(self.W_v, Wvi, bvi)
#         init_complexlinear(self.W_e, Wei, bei)
        init_complexlinear(self.W_p, Wpi, bpi)

        ################################################

        self.args = args

        # self.G1 = nn.Parameter(torch.randn(args.seq_len,args.embed_dim,1))
#         self.Delta = nn.Parameter(torch.rand(1, 2, args.seq_len, args.d_v, 1))
#         self.delta = nn.Parameter(torch.rand(1))

#         self.causal_mask = torch.tril(torch.ones(args.seq_len, args.seq_len)).view(1, args.seq_len, args.seq_len, 1, 1).to(args.device) # Causal attention mask
        self.register_buffer("causal_mask", torch.tril(torch.ones(args.seq_len, args.seq_len)).view(1, args.seq_len, args.seq_len, 1, 1))

        # Relative weighting of attention (alpha) and residual (beta) connections (useful for diagnosis)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

        # Relative weighting of estimate (delta) and prediction (eta) in output
#         self.eta_param = nn.Parameter(torch.tensor(0.0))
        self.eta_param = nn.Parameter(torch.zeros(1, 1, 1, args.d_v, 1)) # Gradient descent step size

        ############################################

        # Create masks for parameter matrices (used for testing)
        init_weight_masks(self, args)

        ############################################

    def forward(self, Z_q, Z_k, Z_v, t_measure_all, skip=0, shared_proj=0):
        """
        Forward pass through the precision-weighted attention block.

        Parameters:
            X (torch.Tensor): Input data.
            lambda_h (torch.Tensor): Diagonal of state transition matrix.
            lambda_Omega (torch.Tensor): Process noise covariance.
            lambda_C (torch.Tensor): Measurement output matrix.
            lambda_Gamma (torch.Tensor): Measurement noise covariance.
            t_measure_all (torch.Tensor): Time differences vector, for each trajectory in batch.

        Returns:
            out (torch.Tensor): Output tensor.
            Q_ij (torch.Tensor): Normalized attention weights.
            Z_ij_hat_all (torch.Tensor): Estimated values.
        """
        
        # Convert real-valued inputs to complex by appending zero imaginary parts, if necessary
        if Z_q.size()[1] == 1:
            Z_q = torch.cat((Z_q, torch.zeros_like(Z_q)),dim=1)
            Z_k = torch.cat((Z_k, torch.zeros_like(Z_k)),dim=1)
            Z_v = torch.cat((Z_v, torch.zeros_like(Z_v)),dim=1)
        
        ############ (Masking; used for testing) ###########
#         lambda_h, lambda_Omega, lambda_Gamma, W_q, W_k, W_v, W_p, W_r, W_e, W_q_b, W_k_b, W_v_b, W_p_b, W_r_b, W_e_b = apply_weight_masks(self, self.args)
        apply_net_weight_masks(self)
        ####################################################
        
#         self.lambda1.data.clamp_(-10, 10)

        lambda_h = compute_lambda_h(self.lambda1,self.args) # Get nonpositive complex conjugate eigenvalues
#         lambda_h = 0 * compute_lambda_h(self.lambda1,self.args) # JUST FOR TESTING
        
        # Take absolute value of noise parameters to ensure positive definiteness / non-negativeness
#         lambda_Omega = self.lambda_Omega**2 # Process noise matrix
#         lambda_Gamma = self.lambda_Gamma**2 # Measurement noise matrix
#         lambda_Omega = self.lambda_Omega**2 + self.args.epsilon # Process noise matrix
#         lambda_Gamma = self.lambda_Gamma**2 + self.args.epsilon # Measurement noise matrix
#         lambda_Omega = torch.abs(self.lambda_Omega) + self.args.epsilon # Process noise matrix
        lambda_Omega = torch.abs(self.lambda_Omega) # Process noise matrix (can be zero)
        lambda_Gamma = torch.abs(self.lambda_Gamma) + self.args.epsilon # Measurement noise matrix (cannot be zero)

#         X_q = X_q.unsqueeze(-1)
#         X_k = X_k.unsqueeze(-1)
#         X_v = X_v.unsqueeze(-1)

        # Project input into Q, K, V
    #     Q = batched_complex_matmul(W_q, X_q)
    #     K = batched_complex_matmul(W_k, X_k)
    #     V = batched_complex_matmul(W_v, X_v)
#         Q = batched_complex_matmul(W_q, X_q) + W_q_b
#         K = batched_complex_matmul(W_k, X_k) + W_k_b
#         V = batched_complex_matmul(W_v, X_v) + W_v_b

        Q = self.W_q(Z_q).unsqueeze(-1)
        K = self.W_k(Z_k).unsqueeze(-1)
        V = self.W_v(Z_v).unsqueeze(-1)
#         if shared_proj == 0:
#             if skip == 0:
#                 Q = self.W_q(Z_q).unsqueeze(-1)
#                 K = self.W_k(Z_k).unsqueeze(-1)
#                 V = self.W_v(Z_v).unsqueeze(-1)
#             elif skip == 1:
#                 Q = Z_q.unsqueeze(-1)
#                 K = Z_k.unsqueeze(-1)
#                 V = Z_v.unsqueeze(-1)
#             else:
#                 print('Error: set skip param')
#         elif shared_proj == 1:
#             if skip == 0:
#                 V = self.W_v(Z_v).unsqueeze(-1)
#                 V_q = self.W_v(Z_q)
#                 V_k = self.W_v(Z_k)
#                 Q = self.W_q(V_q).unsqueeze(-1)
#                 K = self.W_k(V_k).unsqueeze(-1)
#             elif skip == 1:
#                 V = Z_v.unsqueeze(-1)
#                 Q = self.W_q(Z_q).unsqueeze(-1)
#                 K = self.W_k(Z_k).unsqueeze(-1)
#             else:
#                 print('Error: set skip param')
#         else:
#             print('Error: set shared_proj param')

    #     R = batched_complex_matmul(W_r, X_v)

        # G1 = torch.sigmoid(self.G1)
        # G = torch.stack((G1,torch.zeros(self.args.seq_len,self.args.embed_dim,1)))
        # IG = torch.stack((1 - G1,torch.zeros(self.args.seq_len,self.args.embed_dim,1)))

        if len(t_measure_all.size()) > 1:
            t_measure = t_measure_all[0,:-1]
        else:
            t_measure = t_measure_all[:,:-1]
            
        # Normalize time vector by total time elapsed (optional)
        t_measure /= (t_measure[-1] - t_measure[0]).unsqueeze(0)

        ########################################################
        
        # If using separate params for keys and values
        if self.args.sep_params == 1:
            if self.args.weight_mask == 1: # Apply masks
                self.lambda_h_k = self.lambda_h_k * self.eigen_mask
                self.lambda_Omega_k = self.lambda_Omega_k * self.noise_mask
                self.lambda_Gamma_k = self.lambda_Gamma_k * self.noise_mask
            
            # Prepare params
            lambda_h_k = compute_lambda_h(self.lambda1_k,self.args) # Get nonpositive complex conjugate eigenvalues
#             lambda_Omega_k = self.lambda_Omega_k**2 # Process noise matrix
#             lambda_Gamma_k = self.lambda_Gamma_k**2 # Measurement noise matrix 
            lambda_Omega_k = torch.abs(self.lambda_Omega_k) # Process noise matrix
            lambda_Gamma_k = torch.abs(self.lambda_Gamma_k) # Measurement noise matrix
            
            # Functionality for possibly unequal time intervals
            if self.args.t_equal == 1: # If equal time intervals
                K_exp, K_exp2k, _, _ = compute_exp_kernel(lambda_h, t_measure)
                K_exp_k, K_exp2_k, _, _ = compute_exp_kernel(lambda_h_k, t_measure)
                
                mat_exp = K_exp[:, -(self.args.seq_len+1), :, :] # Get matrix exponential for next-state prediction                  
                Z_ij_hat_all, R_qk_ij = batched_compute_estimates_and_residuals_vectorized(Q, K, V, K_exp_k, self.args)
            else: # If unequal time intervals
                mat_exp = batched_complex_exp(lambda_h.squeeze(1).unsqueeze(0) * (t_measure_all[:,-1] - t_measure_all[:,-2]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                Z_ij_hat_all, R_qk_ij = compute_estimates_and_residuals_irregular_times(lambda_h_k, t_measure_all[:,:-1], Q, K, V, self.args)
                K_exp2 = None

            if self.args.tanh == 0:
                _, P_ij = compute_precision(lambda_h, lambda_Omega, lambda_Gamma, K_exp2, t_measure_all[:,:-1], self.args, lambda_C=self.lambda_C)
                V_ij_k, P_ij_k = compute_precision(lambda_h_k, lambda_Omega_k, lambda_Gamma_k, K_exp2_k, t_measure_all[:,:-1], self.args, lambda_C=self.lambda_C)
            else:
                _, P_ij = compute_precision_tanh(lambda_h, lambda_Omega, lambda_Gamma, K_exp2, t_measure_all[:,:-1], self.args, lambda_C=self.lambda_C)
                V_ij_k, P_ij_k = compute_precision_tanh(lambda_h_k, lambda_Omega_k, lambda_Gamma_k, K_exp2_k, t_measure_all[:,:-1], self.args, lambda_C=self.lambda_C)
            
#             # Compute unnormalized attention matrix
#             mahalanobis_distance = (P_ij_k) * (R_qk_ij[:,0]**2 + R_qk_ij[:,1]**2)
#             denom = (1 + self.args.nu*torch.sum(mahalanobis_distance, axis=3, keepdims = True))
#             A_ij = P_ij / denom
            
#             V_ij_k = torch.clamp(V_ij_k, self.args.epsilon) 

#             print(R_qk_ij.size())
#             print(V_ij_k.size())
        
            mahalanobis_distance = (R_qk_ij[:,0]**2 + R_qk_ij[:,1]**2) / (V_ij_k + lambda_Gamma)

        ########################################################
            
        # If using same params for keys and values
        else:
            # Functionality for possibly unequal time intervals
            if self.args.t_equal == 1: # If equal time intervals
                K_exp, K_exp2, _, _ = compute_exp_kernel(lambda_h, t_measure)
                Z_ij_hat_all, R_qk_ij = batched_compute_estimates_and_residuals_vectorized(Q, K, V, K_exp, self.args)
                mat_exp = K_exp[:, -(self.args.seq_len+1), :, :] # Get matrix exponential for next-state prediction
            else: # If unequal time intervals
                Z_ij_hat_all, R_qk_ij = compute_estimates_and_residuals_irregular_times(lambda_h, t_measure_all[:,:-1], Q, K, V, self.args)
                mat_exp = batched_complex_exp(lambda_h.squeeze(1).unsqueeze(0) * (t_measure_all[:,-1] - t_measure_all[:,-2]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                K_exp2 = None

            if self.args.tanh == 0:
#                 _, P_ij = compute_precision(lambda_h, lambda_Omega, lambda_Gamma, K_exp2, t_measure_all[:,:-1], self.args, lambda_C=self.lambda_C)
                K_cov = compute_covariance_kernel(lambda_h, lambda_Omega, lambda_Gamma, K_exp, t_measure, self.args, lambda_C=self.lambda_C)
#                 K_cov = compute_covariance_kernel_stable(lambda_h, lambda_Omega, lambda_Gamma, self.args, lambda_C=self.lambda_C)
            else:
#                 _, P_ij = compute_precision_tanh(lambda_h, lambda_Omega, lambda_Gamma, K_exp, t_measure_all[:,:-1], self.args, lambda_C=self.lambda_C)

                K_cov = compute_covariance_kernel_tanh(lambda_h, lambda_Omega, lambda_Gamma, K_exp, t_measure, self.args, lambda_C=self.lambda_C)
            
            V_ij =  build_covariance_from_kernel(K_cov, self.args)
            
#             V_ij = torch.clamp(V_ij, self.args.epsilon) 

            # Compute unnormalized attention matrix
            mahalanobis_distance = (R_qk_ij[:,0]**2 + R_qk_ij[:,1]**2) / (V_ij + lambda_Gamma)
        
#         # Version without parameters
#         dist_sum = torch.sum(mahalanobis_distance, axis=3, keepdims = True)
#         base_attn_scores = V_ij * (1 + dist_sum)
#         attention_scores = - torch.log(base_attn_scores)

        # Version with more parameters
        dist_sum = torch.abs(self.nu) * torch.sum(mahalanobis_distance, axis=3, keepdims = True)
        base_attn_scores = V_ij * (self.noise_floor + dist_sum + self.args.epsilon)
#         base_attn_scores = torch.clamp(base_attn_scores, min=1e-30, max=1e30) # Shouldn't need this
        attention_scores = - torch.abs(self.tau) * torch.log(base_attn_scores)
        
#         A_ij = torch.exp(attention_scores) # Full inverse mahalanobis metric
#         A_ij = A_ij * self.causal_mask # Apply causal mask to attention matrix
#         S_ij = torch.sum(A_ij, axis=2, keepdims = True) # Sum
#         Q_ij = A_ij / S_ij # Normalize
        
        attention_scores.masked_fill_(self.causal_mask == 0, float('-inf')) # Set to -infinity where mask is 0
        Q_ij = torch.softmax(attention_scores, dim=2)
        
        Z_ij_hat_all = Z_ij_hat_all * self.causal_mask # Mask out estimates backward in time (not strictly necessary but useful later for visualization)
        
        #########################################################
        
        # Compute Hadamard product and sum to get estimate in diagonalized space
        est_v = torch.sum(Q_ij.unsqueeze(1) * Z_ij_hat_all,axis=3)
        
        ######################################################
#         # JUST FOR TESTING: REGULAR ATTENTION
        
#         attn_mat = batched_complex_matmul_full(Q.squeeze(-1),K.squeeze(-1).permute(0,1,3,2))
#         est_v = batched_complex_matmul_full(attn_mat,V.squeeze(-1)).unsqueeze(-1)
        
#         #########################################################

        # Add residual connection
#         est_latent = est_v # No residual connection

        eta = torch.sigmoid(self.eta_param)
        est_latent = (1 - eta) * V + torch.sigmoid(eta) * est_v
    #     est_e = self.args.alpha*est_v + self.args.beta*V # JUST FOR TESTING
    #     est_e = est_v + self.alpha*(est_v - V) # JUST FOR TESTING
    #     est_e = est_v + R
    #     est_e = est_v + batched_complex_matmul(W_r, est_v - V)
#         est_latent = (1-torch.sigmoid(self.Delta))*V + torch.sigmoid(self.Delta)*est_v
#         print(torch.exp(-self.Delta**2))

        # Multiply by output matrix to get estimate
    #     est = batched_complex_matmul(W_e,est_latent)
#         est = batched_complex_matmul(W_e,est_latent) + W_e_b
#         est = self.W_e(est_latent.squeeze(-1))
    #     est = batched_complex_matmul(W_p,est_latent)

        # Get prediction in diagonalized space
    #     pred_p = batched_complex_hadamard(mat_exp, est_e)
    #     pred_p = batched_complex_hadamard(lambda_h, est_e)*(self.args.n * self.args.dt) + est_e # JUST FOR TESTING
    #     pred_p = batched_complex_hadamard(mat_exp, V) # JUST FOR TESTING
    #     pred_p = batched_complex_hadamard(lambda_h, V)*(self.args.n * self.args.dt) + V # JUST FOR TESTING
        if self.args.t_equal == 1: # If equal time intervals
            pred_p = batched_complex_hadamard(mat_exp, est_latent)
        else:
            pred_p = batched_complex_hadamard_full(mat_exp.unsqueeze(2), est_latent)

        # Multiply by output matrix to get output prediction
    #     pred = batched_complex_matmul(self.W_p, pred_p)
    #     pred = batched_complex_matmul(W_p, pred_p)
#         pred = batched_complex_matmul(W_p, pred_p) + W_p_b

        pred = self.W_p(pred_p.squeeze(-1))
    #     pred = batched_complex_matmul(self.W_p, batched_complex_hadamard(lambda_h, X_v))*self.args.dt + X_v # JUST FOR TESTING

        # Output is a linear combination of estimate and prediction
        out = pred
#         out = self.args.delta*pred + self.args.eta*est
    #     out = self.delta*pred + self.eta*est
    #     out = pred + est

#         est = est.squeeze(-1)
        est_latent = est_latent.squeeze(-1)
        out = out.squeeze(-1)
        Z_ij_hat_all = Z_ij_hat_all.squeeze(-1)

        return est_latent, out, Q_ij, Z_ij_hat_all, lambda_h
    
##########################################################################################
##########################################################################################

    
    