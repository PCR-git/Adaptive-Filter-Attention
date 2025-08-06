import numpy as np
import torch
import torch.nn as nn

from utils import batched_complex_matmul, batched_complex_hadamard, batched_complex_exp, batched_complex_matmul_full, batched_complex_hadamard_full

from precision_attention import compute_exp_kernel
from precision_attention import batched_compute_estimates_and_residuals_vectorized, compute_estimates_and_residuals_irregular_times
from precision_attention import compute_precision, compute_precision_tanh, compute_covariance_kernel, compute_covariance_kernel_scalar
from precision_attention import build_covariance_from_kernel, build_covariance_from_kernel_scalar
from precision_attention import build_factorized_kernels, compute_residual_norm, compute_weighted_residual_norm

from model import resolve_multihead_dims
from model import compute_lambda_h, compute_lambda_shared, init_complexlinear, init_complex_matrix, init_weight_masks, apply_weight_masks, apply_net_weight_masks
from model import ComplexLinearLayer, ComplexLinearHermitianLayer

from utils import batched_complex_matmul_multihead
from precision_attention import compute_exp_kernel_multihead, compute_covariance_kernel_scalar_multihead, build_covariance_from_kernel_scalar_multihead, compute_weighted_residual_norm_multihead
from model import compute_lambda_shared_multihead

##########################################################################################
##########################################################################################

def compute_estimate_simplified_multihead(self, K_cov_k, K_cov_v, V_k, V_v, U_q, U_k, U_v, Q, K, V, args):
    """
    
    """
    
    V_avg_ij_v = build_covariance_from_kernel_scalar_multihead(K_cov_v, args)

    R_qk_abs_squared = compute_weighted_residual_norm_multihead(V_k, U_k, U_q)

    if args.sep_params == 1:
        V_avg_ij_k = build_covariance_from_kernel_scalar_multihead(K_cov_k, args)
    #         base_attn_scores = torch.log(V_avg_ij_v) + torch.log(1 + R_qk_abs_squared/V_avg_ij_v)
        base_attn_scores = torch.log(V_avg_ij_v + self.args.epsilon) + torch.log(self.noise_floor**2 + (R_qk_abs_squared + self.args.epsilon)/(V_avg_ij_k + self.args.epsilon) + self.args.epsilon)
        attention_scores = - self.tau**2 * base_attn_scores
    else:
        base_attn_scores = (self.noise_floor**2 + self.args.epsilon) * V_avg_ij_v + R_qk_abs_squared
        attention_scores = - self.tau**2 * torch.log(base_attn_scores)

    attention_scores.masked_fill_(self.causal_mask == 0, float('-inf')) # Set to -infinity where mask is 0
    attention_scores_normalized = torch.softmax(attention_scores, dim=2)

    # Complex-valued attention matrix
    Q_ij = torch.stack((attention_scores_normalized, torch.zeros_like(attention_scores_normalized)),dim=1) # Add zero imaginary part to unnormalized attention

    # Estimate in diagonalized space
    est_inner = batched_complex_matmul_multihead(Q_ij, U_v) # Multiply by Values to get output

    # est_inner.size() # torch.Size([1, 2, 101, 128, 2])
    # V_v.size() # torch.Size([1, 2, 101, 128, 2])
    est_v = batched_complex_hadamard_full(V_v, est_inner)

    return est_v, Q_ij

##########################################################################################
##########################################################################################

class MultiheadSimplifiedPrecisionAttentionBlock(nn.Module):
    def __init__(self, args, seq_len, n_heads, input_dim, query_key_dim=None, value_dim=None, query_key_dim_total=None, value_dim_total=None):
        """

        """

        super().__init__()
        
        # Set query_key and value dims, depending on whether user provided total dims, or head dims
        query_key_dim, value_dim, query_key_dim_total, value_dim_total = resolve_multihead_dims(n_heads, query_key_dim, value_dim, query_key_dim_total=query_key_dim_total, value_dim_total=value_dim_total)

        # Store dimensions as instance attributes
        self.n_heads = n_heads
        self.d_e = input_dim
        self.d_k_head = query_key_dim
        self.d_v_head = value_dim
        self.d_k_total = query_key_dim_total
        self.d_v_total = value_dim_total
        self.seq_len = seq_len

        ################################################

        self.W_q = ComplexLinearLayer(self.d_e, self.d_k_total)
        self.W_k = ComplexLinearLayer(self.d_e, self.d_k_total)
        self.W_v = ComplexLinearLayer(self.d_e, self.d_v_total)
        self.W_p = ComplexLinearLayer(self.d_v_total, self.d_e)

        # if args.lambda_real_zero == 1:
        #     self.W_p = ComplexLinearHermitianLayer(self.W_v)
        # else:
        #     self.W_p = ComplexLinearLayer(self.d_v, self.d_e)

        ################################################

        self.lambda_imag_v = nn.Parameter(torch.randn(self.n_heads, int(self.d_v_head/2)))

        if args.lambda_real_zero == 1:
            self.register_buffer("lambda_real_v", torch.zeros(self.n_heads))
        else:
            self.lambda_real_v = nn.Parameter(torch.randn(self.n_heads))

        self.lambda_omega_sqrt_v = nn.Parameter(torch.randn(self.n_heads))
        self.lambda_gamma_sqrt_v = nn.Parameter(torch.randn(self.n_heads))

        ################################################

        # Define separate dynamic key-query parameters

        if args.sep_params == 1:

            self.lambda_imag_k = nn.Parameter(torch.randn(self.n_heads, int(self.d_v_head/2)))

            if args.lambda_real_zero == 1:
                self.register_buffer("lambda_real_k", torch.zeros(self.n_heads))
            else:
                self.lambda_real_k = nn.Parameter(torch.randn(self.n_heads))

            self.lambda_omega_sqrt_k = nn.Parameter(torch.randn(self.n_heads))
            self.lambda_gamma_sqrt_k = nn.Parameter(torch.randn(self.n_heads))

        ################################################

        self.noise_floor = nn.Parameter(torch.ones(self.n_heads))

        self.tau = nn.Parameter(torch.ones(self.n_heads))
        self.nu = nn.Parameter(torch.ones(self.n_heads))

        ################################################

        Wqi, bqi = init_complex_matrix(self.d_e, self.d_k_total, bias=True)
        Wki, bki = init_complex_matrix(self.d_e, self.d_k_total, bias=True)
        Wvi, bvi = init_complex_matrix(self.d_e, self.d_v_total, bias=True)
        Wpi, bpi = init_complex_matrix(self.d_v_total, self.d_e, bias=True)
        init_complexlinear(self.W_q, Wqi, bqi)
        init_complexlinear(self.W_k, Wki, bki)
        init_complexlinear(self.W_v, Wvi, bvi)
        init_complexlinear(self.W_p, Wpi, bpi)
        #         if args.lambda_real_zero == 0:
        #             init_complexlinear(self.W_p, Wpi, bpi)

        ################################################

        self.args = args

        self.register_buffer("causal_mask", torch.tril(torch.ones(self.seq_len, self.seq_len)).view(1, self.seq_len, self.seq_len, 1))

        self.register_buffer("complex_identity", torch.stack((torch.eye(self.d_v_head, self.d_v_head), \
                                                              torch.zeros(self.d_v_head, self.d_v_head))).unsqueeze(1))

        # Relative weighting of attention (alpha) and residual (beta) connections (useful for diagnosis)
        self.alpha = nn.Parameter(torch.zeros(1, self.n_heads))
        self.beta = nn.Parameter(torch.zeros(1, self.n_heads))

        # Relative weighting of estimate (delta) and prediction (eta) in output
        self.eta_param = nn.Parameter(torch.zeros(1, 1, 1, self.d_v_head, self.n_heads)) # "Gradient descent" step size

        ############################################

        # Create masks for parameter matrices (used for testing)
        init_weight_masks(self, args)

        # self.epsilon = 1e-5

        ############################################

    def forward(self, Z_q, Z_k, Z_v, t_measure_all):
        """

        """
        
        ############ (Masking; used for testing) ###########
        apply_net_weight_masks(self)
        ####################################################

        if Z_q.size()[1] == 1:
            Z_q = torch.cat((Z_q, torch.zeros_like(Z_q)),dim=1)
            Z_k = torch.cat((Z_k, torch.zeros_like(Z_k)),dim=1)
            Z_v = torch.cat((Z_v, torch.zeros_like(Z_v)),dim=1)

        #######################################

        if len(t_measure_all.size()) > 1:
            t_measure = t_measure_all[0,:-1]
        else:
            t_measure = t_measure_all[:,:-1]

        # Normalize time vector by total time elapsed (optional)
        t_measure /= (t_measure[-1] - t_measure[0]).unsqueeze(0)

        #######################################
        
        batch_size = Z_q.size()[0]

        # Project input into Q, K, V        
        Q_proj = self.W_q(Z_q)
        K_proj = self.W_k(Z_k)
        V_proj = self.W_v(Z_v)

        Q = Q_proj.view(batch_size, 2, self.seq_len, self.n_heads, self.d_k_head).transpose(-1, -2).contiguous() 
        K = K_proj.view(batch_size, 2, self.seq_len, self.n_heads, self.d_k_head).transpose(-1, -2).contiguous() 
        V = V_proj.view(batch_size, 2, self.seq_len, self.n_heads, self.d_v_head).transpose(-1, -2).contiguous()

        #######################################

        if self.args.lambda_real_zero == 1:
            lambda_h_v = compute_lambda_shared_multihead(0*self.lambda_real_v, self.lambda_imag_v, self.args)
        else:
            lambda_h_v = compute_lambda_shared_multihead(self.lambda_real_v, self.lambda_imag_v, self.args)

        # Take absolute value of noise parameters to ensure positive definiteness / non-negativeness
        #         lambda_omega = torch.abs(self.lambda_omega_sqrt) # Process noise matrix
        #         lambda_gamma = torch.abs(self.lambda_gamma_sqrt) + self.args.epsilon # Measurement noise matrix
        lambda_omega_v = self.lambda_omega_sqrt_v**2 # Process noise matrix
        lambda_gamma_v = self.lambda_gamma_sqrt_v**2 + self.args.epsilon # Measurement noise matrix

        exp_f_v, exp_b_v = compute_exp_kernel_multihead(lambda_h_v, t_measure)

        mat_exp = exp_f_v[:,0,:,:] # Get matrix exponential for next-state prediction

        K_cov_v = compute_covariance_kernel_scalar_multihead(lambda_h_v, lambda_omega_v, lambda_gamma_v, exp_f_v, t_measure, self.args, epsilon=1e-5)

        if self.args.sep_params == 1:
            if self.args.lambda_real_zero == 1:
                lambda_h_k = compute_lambda_shared_multihead(0*self.lambda_real_k, self.lambda_imag_k, self.args)
            else:
                lambda_h_k = compute_lambda_shared_multihead(self.lambda_real_k, self.lambda_imag_k, self.args)

            lambda_omega_k = self.lambda_omega_sqrt_k**2 # Process noise matrix
            lambda_gamma_k = self.lambda_gamma_sqrt_k**2 + self.args.epsilon # Measurement noise matrix

            exp_f_k, exp_b_k = compute_exp_kernel_multihead(lambda_h_k, t_measure)

            K_cov_k = compute_covariance_kernel_scalar_multihead(lambda_h_k, lambda_omega_k, lambda_gamma_k, exp_f_k, t_measure, self.args, epsilon=1e-5)
        else:
            exp_f_k = exp_f_v
            exp_b_k = exp_b_v
            K_cov_k = None

        V_k, V_v, U_q, U_k, U_v = build_factorized_kernels(exp_f_k, exp_b_k, exp_f_v, exp_b_v, Q, K, V)

        R_qk_abs_squared = compute_weighted_residual_norm_multihead(V_k, U_k, U_q)

        est_v, Q_ij = compute_estimate_simplified_multihead(self, K_cov_k, K_cov_v, V_k, V_v, U_q, U_k, U_v, Q, K, V, self.args)

        # Add residual connection
        eta = torch.sigmoid(self.eta_param)
        est_latent = (1 - eta) * V + torch.sigmoid(eta) * est_v

        # Get prediction in diagonalized space
        pred_p = batched_complex_hadamard(mat_exp, est_latent)

        pred_p_reshape = pred_p.transpose(-1, -2).contiguous().view(batch_size, 2, self.seq_len, self.d_v_total)

        # Multiply by output matrix to get output prediction
        out = self.W_p(pred_p_reshape)

        # Pair-wise estimates (for plotting/testing):
        Z_ij_hat_all_multihead = batched_complex_hadamard_full(V_v.unsqueeze(3), U_v.unsqueeze(2))
        Z = Z_ij_hat_all_multihead.transpose(-2, -1).contiguous() # Flip dimensions to stack heads
        Z_ij_hat_all = Z.view(*Z.size()[:-2], -1) # Merge heads
#         print(V_v.unsqueeze(3).squeeze(-1).size())
#         print(U_v.unsqueeze(2).squeeze(-1).size())
#         print(Z_ij_hat_all.size())
        
        L = lambda_h_v.squeeze(-1).transpose(-2, -1).contiguous() # Flip dimensions to stack heads
        epoch_lambdas = L.view(*L.size()[:-2], -1).unsqueeze(-1) # Merge heads
    
#         return est_latent, out, Q_ij, V_v, U_v, lambda_h
        return est_latent, out, Q_ij, Z_ij_hat_all, epoch_lambdas
