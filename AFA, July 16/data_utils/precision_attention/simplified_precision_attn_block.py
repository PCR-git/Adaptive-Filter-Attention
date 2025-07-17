#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn

from utils import batched_complex_matmul, batched_complex_hadamard, batched_complex_exp

from precision_attention import compute_exp_kernel
from precision_attention import batched_compute_estimates_and_residuals_vectorized, compute_estimates_and_residuals_irregular_times
from precision_attention import compute_precision, compute_precision_tanh, compute_covariance_kernel, build_covariance_from_kernel
from model import compute_lambda_h, init_complexlinear, init_complex_matrix, init_weight_masks, apply_weight_masks, apply_net_weight_masks
from model import ComplexLinear
from utils import batched_complex_matmul_full


# In[ ]:


def build_factorized_kernels(exp_f_k, exp_b_k, exp_f_v, exp_b_v, Q, K, V):
    
    V_k = exp_f_k.unsqueeze(0)
    V_v = exp_f_v.unsqueeze(0)
    U_q = exp_b_k.unsqueeze(0) * Q
    U_k = exp_b_k.unsqueeze(0) * K
    U_v = exp_b_v.unsqueeze(0) * V
    
    return V_k.squeeze(-1), V_v.squeeze(-1), U_q.squeeze(-1), U_k.squeeze(-1), U_v.squeeze(-1)

# def build_residual_kernels(U_k, U_q):

#     R_k = U_k[:,:,:-1] - U_k[:,:,1:]
#     R_k_cumsum = torch.cumsum(R_k, dim=2)
#     delta_qk = U_k - U_q
    
#     return R_k_cumsum, delta_qk

def compute_R_norm(V_k, U_k, U_q):

    V_k_abs = torch.sum(V_k**2,dim=1)
    U_k_abs = torch.sum(U_k**2,dim=1)
    U_q_abs = torch.sum(U_q**2,dim=1)

    T1 = torch.matmul(V_k_abs, U_k_abs.permute(0,2,1))
    T2 = torch.sum(V_k_abs * U_q_abs, dim=2).unsqueeze(-1)
    VU_kq = V_k_abs * U_q
    VU_kq[:,1] *= -1

    T3 = batched_complex_matmul_full(VU_kq, U_k.permute(0,1,3,2))[:,0]
    R_qk_abs = T1 + T2 - 2 * T3
    
    return R_qk_abs


# In[ ]:


class SimplifiedPrecisionAttentionBlock(nn.Module):
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
            
        self.W_q = ComplexLinear(args.d_e, args.d_k)
        self.W_k = ComplexLinear(args.d_e, args.d_k)
        self.W_v = ComplexLinear(args.d_e, args.d_v)
#         self.W_e = ComplexLinear(args.d_e, args.d_v)
        self.W_p = ComplexLinear(args.d_e, args.d_v)
#         self.W_r = ComplexLinear(args.d_e, args.d_v)

        ################################################

        self.complex_identity = torch.stack((torch.eye(self.d_v, self.d_v),torch.zeros(self.d_v, self.d_v))).unsqueeze(1).to(args.device)

        sqrt_dv = torch.sqrt(torch.tensor(self.d_v))
        
        ######################
        
        lambda_r = torch.randn(int(self.d_v/2))
        lambda_i = torch.randn(int(self.d_v/2))
        self.lambda1 = nn.Parameter(torch.stack((lambda_r,lambda_i)).unsqueeze(-1)) # Stack and scale by time interval
        self.lambda_h = torch.zeros(2,self.head_dim,1).to(args.device) # Initialize full eigenvalue array

        self.lambda_Omega_sqrt = nn.Parameter(torch.randn(1,self.d_v,1)/sqrt_dv) # Process covariance
        self.lambda_Gamma_sqrt = nn.Parameter(torch.randn(1,self.d_v,1)/sqrt_dv) # Measurement covariance

        ######################
        
        self.tau = nn.Parameter(torch.tensor(1.0))
        self.nu = nn.Parameter(torch.tensor(1.0))
        
    #     self.lambda_C = nn.Parameter(torch.randn(1,args.d_v,1)/sqrt_dv) # Measurement matrix
        self.lambda_C = torch.ones(1,self.d_v,1).to(args.device) # Output matrix

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
        self.epsilon = 1E-5

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

    def forward(self, Z_q, Z_k, Z_v, t_measure_all):
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
            X_ij_hat_all (torch.Tensor): Estimated values.
        """
        
        ############ (Masking; used for testing) ###########
#         lambda_h, lambda_Omega, lambda_Gamma, W_q, W_k, W_v, W_p, W_r, W_e, W_q_b, W_k_b, W_v_b, W_p_b, W_r_b, W_e_b = apply_weight_masks(self, self.args)
        apply_net_weight_masks(self)
        ####################################################

        #         lambda_h = compute_lambda_h(self.lambda1,self.args) # Get nonpositive complex conjugate eigenvalues
        lambda_h = 0 * compute_lambda_h(self.lambda1,self.args) # JUST FOR TESTING

        # Take absolute value of noise parameters to ensure positive definiteness / non-negativeness
        lambda_Omega = self.lambda_Omega_sqrt**2 + self.epsilon # Process noise matrix
        lambda_Gamma = self.lambda_Gamma_sqrt**2 + self.epsilon # Measurement noise matrix

        # Project input into Q, K, V        
        Q = self.W_q(Z_q).unsqueeze(-1)
        K = self.W_k(Z_k).unsqueeze(-1)
        V = self.W_v(Z_v).unsqueeze(-1)

        if len(t_measure_all.size()) > 1:
            t_measure = t_measure_all[0,:-1]
        else:
            t_measure = t_measure_all[:,:-1]

        ########################################################

        K_exp, K_exp2, exp_f, exp_b = compute_exp_kernel(lambda_h, t_measure)

        mat_exp = K_exp[:, -(self.args.seq_len+1), :, :] # Get matrix exponential for next-state prediction

        K_cov = compute_covariance_kernel(lambda_h, lambda_Omega, lambda_Gamma, K_exp, t_measure, self.args, lambda_C=self.lambda_C)

        V_avg_ij = build_avg_covariance_from_kernel(K_cov, args).squeeze()

        V_k, V_v, U_q, U_k, U_v = build_factorized_kernels(exp_f, exp_b, exp_f, exp_b, Q, K, V)

        R_qk_abs = compute_R_norm(V_k, U_k, U_q)

        base_attn_scores = V_avg_ij + R_qk_abs

        attention_scores = - self.tau**2 * torch.log(base_attn_scores).unsqueeze(-1).unsqueeze(-1)

        attention_scores.masked_fill_(self.causal_mask == 0, float('-inf')) # Set to -infinity where mask is 0
        attention_scores_normalized = torch.softmax(attention_scores, dim=2)

        # Complex-valued attention matrix
        Q_ij = torch.stack((attention_scores_normalized, torch.zeros_like(attention_scores_normalized)),dim=1) # Add zero imaginary part to unnormalized attention

        # Estimate in diagonalized space
        est_inner = batched_complex_matmul_full(Q_ij.squeeze(-1).squeeze(-1), U_v).unsqueeze(-1) # Multiply by Values to get output

        est_v = V_v.unsqueeze(-1) * est_inner

        # Add residual connection
        # est_eigenbasis = est_v # No residual connection
        est_eigenbasis = (1-torch.sigmoid(self.delta))*V + torch.sigmoid(self.delta)*est_v

        # Get prediction in diagonalized space
        pred_p = batched_complex_hadamard(mat_exp, est_eigenbasis)

        # Multiply by output matrix to get output prediction
        pred = self.W_p(pred_p.squeeze(-1))
        #     pred = batched_complex_matmul(self.W_p, batched_complex_hadamard(lambda_h, X_v))*self.args.dt + X_v # JUST FOR TESTING

        # Output is a linear combination of estimate and prediction
        out = pred

        est_eigenbasis = est_eigenbasis.squeeze(-1)
        out = out.squeeze(-1)
        X_ij_hat_all = X_ij_hat_all.squeeze(-1)

        return est_eigenbasis, out, Q_ij, V_v, U_v, lambda_h


# In[2]:


X_ij_hat_all = V_v.unsqueeze(3) * U_v.unsqueeze(2)


# In[ ]:




