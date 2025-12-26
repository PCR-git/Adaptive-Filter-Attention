import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt

from utils import batched_complex_matmul, batched_complex_hadamard, batched_complex_exp, batched_complex_matmul_full, batched_complex_hadamard_full

from isotropic_afa import build_factorized_kernels, compute_residual_norm_isotropic, compute_residual_norm_anisotropic
from isotropic_afa import compute_exp_kernel_isotropic, compute_exp_kernel_anisotropic, compute_covariance_matrix, compute_covariance_matrix_safe, get_safe_exp_tot

from model import resolve_multihead_dims
from model import apply_weight_masks, initialize_linear_layers, init_complex_matrix, init_complexlinear
from model import ComplexLinearLayer, ComplexLinearHermitianLayer
from model import ComplexRMSNorm
from model import compute_lambda

##########################################################################################
##########################################################################################

def compute_attention_matrix(self, R_qk_abs_squared, V_ij_k, V_ij_v, V_tilde, Delta_T, nu, args):
    
    d_head = V_tilde.size()[3] # Embed dim of each head
    L_seq = V_tilde.size()[2] # Sequence length
    
    if args.additive_bias_type == 0:
        B = 0 # Zero bias
    elif args.additive_bias_type == 1:
        B = - torch.log(V_ij_v + args.epsilon) # DLE Bias
    elif args.additive_bias_type == 2:
        B = - nu * Delta_T # Linear bias
    else:
        pass
        
    if args.multiplicative_bias_type == 0:
        S = d_head # Constant bias
    elif args.multiplicative_bias_type == 1:
        S = nu * d_head * V_ij_k # DLE Bias
    elif args.multiplicative_bias_type == 2:
        S = nu * d_head * Delta_T # Linear bias
    else:
        pass

    if args.use_robust_weight == 1:
        #   P_ij_v/(1 + R_qk_abs_squared * P_ij_k)
        # = exp[ log(P_ij_v) - log(1 + R_qk_abs_squared * P_ij_k)]
        # = exp[-log(V_ij_v) - log(1 + R_qk_abs_squared / V_ij_k)]
        base_attn_scores = B - torch.log(1 + R_qk_abs_squared / S)
    else:
        #   P_ij_v * exp(-R_qk_abs_squared * P_ij_k)
        # = exp[ log(P_ij_v) - R_qk_abs_squared * P_ij_k]
        # = exp[-log(V_ij_v) - R_qk_abs_squared / V_ij_k]
        base_attn_scores = B - R_qk_abs_squared / S

    attention_scores = self.tau**2 * base_attn_scores
    
    attention_scores_masked = attention_scores.masked_fill(self.causal_mask == 0, float('-inf')) # Set to -infinity where mask is 0
    unnormalized_attention = torch.exp(attention_scores_masked) # Measure of total precision; optionally used in gating
    
    A = torch.softmax(attention_scores_masked, dim=2)
   
    # Output the prior, for plotting (not used)
    A_prior = (1/(V_ij_v + args.epsilon)).masked_fill(self.causal_mask == 0, float('0')).squeeze(0)

    return A, A_prior, unnormalized_attention

# ===========================

def compute_estimate(A_hat, V_tilde, Phi_tilde_plus_v, args):
    
    # Since A_hat is real, we don't need to do the full complex matmul
    
    A_hat_permute = A_hat.permute(0,3,1,2)
    V_tilde_permute = V_tilde.permute(0,4,2,3,1)
    sz = V_tilde_permute.size()
    V_tilde_reshape = V_tilde_permute.reshape((sz[0],sz[1],sz[2],sz[3]*sz[4]))
    
    est_inner = torch.matmul(A_hat_permute, V_tilde_reshape)
    est_inner = est_inner.reshape((sz[0],sz[1],sz[2],sz[3], sz[4])).permute(0, 4, 2, 3, 1)

    if args.rotate_values == 1:
        est_v = batched_complex_hadamard_full(Phi_tilde_plus_v.unsqueeze(0), est_inner)
    else:
        est_v = est_inner

    return est_v

##########################################################################################
##########################################################################################

class MultiheadIsotropicAFA(nn.Module):
    def __init__(self, args, n_heads, input_dim, query_key_dim=None, value_dim=None, query_key_dim_total=None, value_dim_total=None):
        """
        Implements the Multi-Head Isotropic Adaptive Filter Attention (AFA) block.

        This module integrates a complex-valued, linear time-invariant Stochastic
        Differential Equation (LTI-SDE) model into the Transformer's attention mechanism.

        The architecture is designed for O(N^2 + Nd) memory complexity by
        enforcing isotropic decay and noise in the diagonalized eigenbasis.

        Key Learned Parameters (per head):
        - Dynamics (lambda_real, lambda_imag): Defines the SDE's decay and rotational frequencies.
        - Noise (sigma, eta, gamma): Defines process noise, measurement noise, and anchor noise.
        - Robustness (nu): Sets the statistical threshold for outlier suppression.
        - Temperature (tau): Scales the final attention logits.

        The core function is to derive the attention matrix from the SDE's propagated
        uncertainty via the Differential Lyapunov Equation (DLE).
        """

        super().__init__()
        
        self.args = args
        
        if query_key_dim==None or value_dim==None or query_key_dim_total==None or value_dim_total==None:
            # Set query_key and value dims, depending on whether user provided total dims, or head dims
            query_key_dim, value_dim, query_key_dim_total, value_dim_total = resolve_multihead_dims(n_heads, query_key_dim, value_dim, query_key_dim_total=query_key_dim_total, value_dim_total=value_dim_total)

        # Store dimensions as instance attributes
        self.n_heads = n_heads
        self.d_e = input_dim
        self.d_k_head = query_key_dim
        self.d_v_head = value_dim
        self.d_k_total = query_key_dim_total
        self.d_v_total = value_dim_total

        ################################################

        # Linear Layers
        self.W_q = nn.Linear(self.args.d_e, self.args.d_k_total*2)
        self.W_k = nn.Linear(self.args.d_e, self.args.d_k_total*2)
        self.W_v = nn.Linear(self.args.d_e, self.args.d_v_total*2)
        self.W_o = nn.Linear(self.args.d_v_total*2, self.args.d_e)
        
        ################################################
        
        # Define lambda (LTI) params

        ##########################
        # Initialize omega
        dim_target = int(self.d_v_head/2)
        if self.args.learn_rotations == 1:
            rand_noise = torch.randn(self.n_heads, dim_target)
            self.lambda_imag_v = nn.Parameter(rand_noise) # Gaussian noise
        else:
#             omega_init = (1.0/self.args.delta_t) / (10000 ** (torch.arange(dim_target).float() / dim_target))
            omega_init = 1.0 / (10000 ** (torch.arange(dim_target).float() / dim_target))
            omega_init_stacked = omega_init.repeat(self.n_heads).view(self.n_heads, dim_target)
            self.lambda_imag_v = omega_init_stacked.to(self.args.device) # Initialize as in RoPE
#             self.lambda_imag_v = nn.Parameter(omega_init_stacked * (1 + 0.5*rand_noise)) # RoPE initialization + multiplicative rand noise
        ##########################
    
        self.lambda_real_v = nn.Parameter(torch.randn(self.n_heads))

        self.sigma_v = nn.Parameter(torch.randn(self.n_heads))
        self.eta_v = nn.Parameter(torch.randn(self.n_heads))
        self.gamma_v = nn.Parameter(torch.randn(self.n_heads))

        # Optionally, define separate dynamic query/key parameters
        if self.args.sep_params == 1:

            rand_noise = torch.randn(self.n_heads, dim_target)
            self.lambda_imag_k = nn.Parameter(rand_noise) # Gaussian noise
            self.lambda_real_k = nn.Parameter(torch.randn(self.n_heads))

            ##########################

            self.sigma_k = nn.Parameter(torch.randn(self.n_heads))
            self.eta_k = nn.Parameter(torch.randn(self.n_heads))
            self.gamma_k = nn.Parameter(torch.randn(self.n_heads))
            
            ##########################
            # Initialize omega
            dim_target = int(self.d_k_head/2)
            if self.args.learn_rotations == 1:
                rand_noise = torch.randn(self.n_heads, dim_target)
                self.lambda_imag_k = nn.Parameter(rand_noise) # Gaussian noise
            else:
#                 omega_init = (1.0/self.args.delta_t) / (10000 ** (torch.arange(0, dim_target, 2).float() / dim_target))
                omega_init = 1.0 / (10000 ** (torch.arange(0, dim_target, 2).float() / dim_target))
                omega_init_stacked = omega_init.repeat(self.n_heads).view(self.n_heads, dim_target)
                self.lambda_imag_k = omega_init_stacked.to(self.args.device)

        self.tau = nn.Parameter(torch.ones(self.n_heads))
        self.nu_sqrt = nn.Parameter(torch.ones(self.n_heads))

        ################################################

        # Initialize linear layers using complex initialization
        W_q_init = init_complex_matrix(self.d_e, self.d_k_total)
        W_k_init = init_complex_matrix(self.d_e, self.d_k_total)
        W_v_init = init_complex_matrix(self.d_e, self.d_v_total)
        W_o_init = init_complex_matrix(self.d_v_total, self.d_e)

        init_complexlinear(self.W_q, W_q_init, layer_type='in')
        init_complexlinear(self.W_k, W_k_init, layer_type='in')
        init_complexlinear(self.W_v, W_v_init, layer_type='in')
        init_complexlinear(self.W_o, W_o_init, layer_type='out')

        ################################################
        
        # Initialize complex-valued normalization layers
        if self.args.use_complex_input_norm == 0:
            pass
        else:
            self.cn_q = ComplexRMSNorm(self.d_k_total, self.n_heads)
            self.cn_k = ComplexRMSNorm(self.d_k_total, self.n_heads)
            self.cn_v = ComplexRMSNorm(self.d_v_total, self.n_heads)

        if self.args.use_inner_residual_and_norm == 0:
            pass
        else:
            self.cn_o = ComplexRMSNorm(self.d_v_total, self.n_heads)
        
        ################################################
        
        causal_mask = torch.tril(torch.ones(self.args.seq_len, self.args.seq_len, device=self.args.device))
        self.register_buffer("causal_mask", causal_mask.view(1, self.args.seq_len, self.args.seq_len, 1))

        # Relative weighting of estimate and prediction in output ("Gradient descent" step size)
        self.P_scale = nn.Parameter(torch.zeros(1, 1, 1, self.d_v_head, self.n_heads))
        self.P_prior_param = nn.Parameter(torch.zeros(1, 1, 1, self.d_v_head, self.n_heads))

        ############################################

    def forward(self, Z_q, Z_k, Z_v, t_measure_all=None):
        """
        Executes the Adaptive Filter Attention (AFA) process to generate the filtered state estimate.

        The function performs:
        1. Projection: Maps real features (Z_q, Z_k, Z_v) to the complex latent eigenbasis.
        2. Kernel Computation: Calculates the time-dependent state transition kernels (Phi_tilde)
           and the covariance kernels (K_cov) using the learned dynamic parameters.
        3. Weight Calculation: Determines the robust attention matrix (A_hat) by scaling
           the calculated precision prior (1/K_cov) based on the Mahalanobis residual distance (R_qk_abs_squared).
        4. Aggregation: Computes the complex, precision-weighted sum of the rotated values (V_tilde)
           and rotates the result back to the forward domain to produce the final filtered estimate (out).

        Args:
            Z_q (Tensor): Input features for Queries (Real, shape B x L x d_e).
            Z_k (Tensor): Input features for Keys (Real).
            Z_v (Tensor): Input features for Values (Real).
            t_measure_all (Tensor, optional): Timestamps for observations. Defaults to regular sampling if None.

        Returns:
            out (Tensor): Final output, projected back to the real domain.
            output_dict (dict): Intermediate tensors including latent estimates and attention matrix.
        """
        
        batch_size = Z_q.size()[0]
        seq_len = Z_q.size()[2]
        
        if t_measure_all == None:
#             t_measure = torch.arange(seq_len).to(self.args.device) * self.args.delta_t
            t_measure = torch.arange(seq_len).to(self.args.device)
        else:
            if len(t_measure_all.size()) > 1:
                t_measure = t_measure_all[0,:-1]
            else:
                t_measure = t_measure_all[:,:-1]

        # Normalize time vector by total time elapsed (optional)
#         t_measure = t_measure / (t_measure[-1] - t_measure[0]).unsqueeze(0)
        
        #######################################
        
        # Apply standard linear layers
        Q_proj = self.W_q(Z_q).view(batch_size, seq_len, 2, self.d_k_total).transpose(1, 2).contiguous()
        K_proj = self.W_k(Z_k).view(batch_size, seq_len, 2, self.d_k_total).transpose(1, 2).contiguous()
        V_proj = self.W_v(Z_v).view(batch_size, seq_len, 2, self.d_v_total).transpose(1, 2).contiguous()
        
        #######################################
        
        # Separate into heads
        Q = Q_proj.view(batch_size, 2, seq_len, self.n_heads, self.d_k_head).transpose(-1, -2).contiguous() 
        K = K_proj.view(batch_size, 2, seq_len, self.n_heads, self.d_k_head).transpose(-1, -2).contiguous() 
        V = V_proj.view(batch_size, 2, seq_len, self.n_heads, self.d_v_head).transpose(-1, -2).contiguous()

        #######################################

        # Optionally, apply complex-valued normalization on inputs (per head)
        if self.args.use_complex_input_norm == 1: # Normalize query, key, and value
            Q_norm = self.cn_q(Q)
            K_norm = self.cn_k(K)
            V_norm = self.cn_v(V)
        elif self.args.use_complex_input_norm == 0: # No normalization
            Q_norm = Q
            K_norm = K
            V_norm = V
        elif self.args.use_complex_input_norm == 2: # Normalize only query and key
            Q_norm = self.cn_q(Q)
            K_norm = self.cn_k(K)
        else:
            print('Eror: args.use_complex_input_norm must be 0, 1, or 2.')
        
        #######################################
        
        nu = F.softplus(self.nu_sqrt) + self.args.epsilon
        
        lambda_v = compute_lambda(self.lambda_real_v, self.lambda_imag_v, self.args)
        
        # Ensure non-negativeness of noise parameters
        sigma_squared_v = F.softplus(self.sigma_v) # Process noise matrix
        eta_squared_v = F.softplus(self.eta_v) + self.args.epsilon # Measurement noise matrix
        gamma_squared_v = F.softplus(self.gamma_v) + self.args.epsilon # Anchor noise
        
        if self.args.zero_process_noise == 1:
            sigma_squared_v = torch.zeros_like(sigma_squared_v).to(sigma_squared_v.device)
        if self.args.zero_key_measurement_noise == 1:
            eta_squared_v = torch.zeros_like(eta_squared_v).to(eta_squared_v.device)
            
#         noise_params = torch.stack((sigma_squared_v, eta_v, gamma_v, nu))
#         print(noise_params)
#         print('sigma^2 = ', sigma_squared_v.detach().cpu().numpy())
#         print('eta^2   = ', eta_squared_v.detach().cpu().numpy())
#         print('gamma^2 = ', gamma_squared_v.detach().cpu().numpy())
#         print('nu      = ', nu.detach().cpu().numpy())
        
        ############ (Weight Masking; used for testing) ###########
        if self.args.weight_mask == 1:
            lambda_v = apply_weight_masks(self, lambda_v, self.args)
        ####################################################
        
        # Get relative time difference
        t_measure_i = t_measure.squeeze().unsqueeze(1)  # [m, 1]
        t_measure_j = t_measure.squeeze().unsqueeze(0)  # [1, m]
        Delta_T = torch.abs(t_measure_i - t_measure_j).unsqueeze(-1)  # [m, m]
        
        # Clamp the exponent to ensure safe values
        mu_v = torch.mean(lambda_v,axis=2)[0].squeeze(-1).view(1, 1, -1)
        exp_rel_v = mu_v * Delta_T
        exp_rel_safe_v = torch.clamp(exp_rel_v, min=self.args.min_exponent, max=self.args.max_exponent)
        
        ####################################################

        Phi_tilde_plus_v, Phi_tilde_minus_v, E_rel_v = compute_exp_kernel_isotropic(lambda_v, t_measure, exp_rel_safe_v)
    
#         V_ij_v = compute_covariance_matrix(mu_v, sigma_squared_v, eta_squared_v, gamma_squared_v, E_rel_v, self.args)
        V_ij_v = compute_covariance_matrix_safe(mu_v, Delta_T, exp_rel_safe_v, sigma_squared_v, eta_squared_v, gamma_squared_v, t_measure, self.args)

        if self.args.sep_params == 1:
            lambda_k = compute_lambda(self.lambda_real_k, self.lambda_imag_k, self.args)

            sigma_squared_k = F.softplus(self.sigma__k) # Process noise
            eta_squared_k = F.softplus(self.eta_k) + self.args.epsilon # Measurement noise
            gamma_squared_k = F.softplus(self.gamma_k) + self.args.epsilon # Anchor noise
            
            if self.args.zero_process_noise == 1:
                sigma_squared_k = torch.zeros_like(sigma_squared_k).to(sigma_squared_k.device)
            if self.args.zero_key_measurement_noise == 1:
                eta_squared_k = torch.zeros_like(eta_squared_k).to(eta_squared_k.device)
            
            #################

            mu_k = torch.mean(lambda_k,axis=2)[0].squeeze(-1).view(1, 1, -1)
            exp_rel_k = mu_k * Delta_T
            exp_rel_safe_k = torch.clamp(exp_rel_k, min=self.args.min_exponent, max=self.args.max_exponent)
            
            #################
            
            Phi_tilde_plus_k, Phi_tilde_minus_k, E_rel_k = compute_exp_kernel_isotropic(lambda_k, t_measure, exp_rel_safe_k)
                           
#             V_ij_k = compute_covariance_matrix(mu_k, sigma_squared_k, eta_squared_k, gamma_squared_k, E_rel_k, self.args)                
            V_ij_k = compute_covariance_matrix_safe(mu_k, Delta_T, exp_rel_safe_k, sigma_squared_k, eta_squared_k, gamma_squared_k, t_measure, self.args)

        else:
            V_ij_k = V_ij_v

            Phi_tilde_plus_k = Phi_tilde_plus_v
            Phi_tilde_minus_k = Phi_tilde_minus_v
            E_rel_k = E_rel_v  

        Q_tilde, K_tilde, V_tilde = build_factorized_kernels(Phi_tilde_minus_k, Phi_tilde_minus_v, Q_norm, K_norm, V_norm, self.args)
        R_qk_abs_squared = compute_residual_norm_isotropic(Q_norm, Q_tilde, K_tilde, E_rel_k, self.args)

        A, A_prior, unnormalized_attention = compute_attention_matrix(self, R_qk_abs_squared, V_ij_k, V_ij_v, V_tilde, Delta_T, nu, self.args)

        A_hat = A * E_rel_v # Scale attention weights by the relative decay

        est_v = compute_estimate(A_hat, V_tilde, Phi_tilde_plus_v, self.args)
        
        # Add Guassian noise to last step (for test-time sampling)
        if self.args.add_gaussian_noise == 1:
            P_tot = torch.sum(unnormalized_attention,-2).unsqueeze(1).unsqueeze(-2) # Total precision
            V_tot = 1/(P_tot)
            V_tot_last = V_tot[:,:,-1].unsqueeze(2)
#             V_tot_last = V_tot_last + eta_squared_v
            V_tot_last = V_tot_last + gamma_squared_v
            std_dev = torch.sqrt(V_tot_last/2) # Divide by 2 to deal with complex numbers
            gaussian_noise = torch.randn_like(std_dev) * std_dev # Sample from normal distrb
            est_v[:, :, :, -1:] = est_v[:, :, :, -1:] + gaussian_noise # Add Gaussian noise to last step
        else:
            pass

        # Add residual connection
        if self.args.use_inner_residual_and_norm == 1:
            if self.args.use_total_precision_gate == 0: # Simple residual
                est_latent = est_v + V # Just add them
                
            elif self.args.use_total_precision_gate == 1: # Precision gate
#                P_tot/ (P_tot + P_prior) = 1 / (1 + P_prior/P_tot)
#                = 1/(1 + e^{-(ln(P_tot) - ln(P_prior)}) = sigmoid[ln(P_tot) - ln(P_prior)]
                P_tot = torch.sum(unnormalized_attention,-2).unsqueeze(1).unsqueeze(-2) # Sum precision
                P_tot_log = torch.log(P_tot)
                P_prior_log = torch.log(self.P_prior_param**2 + self.args.epsilon)
                g = torch.sigmoid(P_tot_log * self.P_scale**2 - P_prior_log) # Mean precision gate
                est_latent = g * est_v + (1-g) * V # Update using convex combination
#                 est_latent = g * est_v + V # No gate on residual connection

            elif self.args.use_total_precision_gate == 2: # Learned gate
                g = torch.sigmoid(self.P_scale) # Learned gate
                est_latent = g * est_v + (1-g) * V # Update using convex combination
#                 est_latent = g * est_v + V # No gate on residual connection
            else:
                print('Error: args.use_total_precision_gate must be 0, 1, or 2.')
        else:
            est_latent = est_v # No residual

        # Use complex normalization on outputs
        if self.args.use_inner_residual_and_norm == 1:
            est_norm = self.cn_o(est_latent)
        else:
            est_norm = est_latent
            
        # Predict next step using same matrix exponential used for estimation
        if self.args.compute_next_step_pred == 1:
#             mat_exp = torch.exp(mu_v*self.args.delta_t) * Phi_tilde_plus_v[:,1]
            mat_exp = torch.exp(mu_v) * Phi_tilde_plus_v[:,1]
            pred_p = batched_complex_hadamard(mat_exp, est_norm)
        else:
            pred_p = est_norm
        
        # Merge heads
        pred_p_reshape = pred_p.transpose(-1, -2).contiguous().view(batch_size, 2, seq_len, self.d_v_total)      
        pred_p_reshape = pred_p_reshape.transpose(1, 2).contiguous().view(batch_size, 1, seq_len, self.d_v_total*2) # Include if using standard linear layers !!
        
        # Multiply by output matrix to get output prediction, and get real part
        out = self.W_o(pred_p_reshape)
#         out = out[:,0].unsqueeze(1)
        
        ###################################

        # Get pulled-forward estimates (for testing/visualization)

        Phi_hat_plus_v = Phi_tilde_plus_v.unsqueeze(0).unsqueeze(3) * E_rel_v.unsqueeze(0).unsqueeze(4)
        Z_ij_hat_all_multihead = batched_complex_hadamard_full(Phi_hat_plus_v, V_tilde.unsqueeze(2))
            
        Z = Z_ij_hat_all_multihead.transpose(-2, -1).contiguous() # Flip dimensions to stack heads
        Z_ij_hat_all = Z.view(*Z.size()[:-2], -1) # Merge heads
        
        Z_ij_hat_causal = Z_ij_hat_all * self.causal_mask
        
        Z_ij_hat_causal = Z_ij_hat_causal.permute(0, 2, 3, 1, 4).contiguous().view(Z_ij_hat_all.size()[0],Z_ij_hat_all.size()[2],Z_ij_hat_all.size()[3],Z_ij_hat_all.size()[-1]*2) # Include if using standard linear layers !!!
        
        x_hat = self.W_o(Z_ij_hat_causal).unsqueeze(-1) # Prediction in original basis

        ###################################
        
        L = lambda_v.squeeze(-1).transpose(-2, -1).contiguous() # Flip dimensions to stack heads
        L = L/self.args.delta_t # Have to divide by dt since we absorbed dt into mu
        epoch_lambdas = L.view(*L.size()[:-2], -1).unsqueeze(-1) # Merge heads
        
#         # Plot attention priors
#         for head in range(A_prior.size()[-1]):
#             A_prior_i = A_prior[:,:,head].detach().cpu().numpy()
#             plt.imshow(A_prior_i)
#             plt.title('Attn Prior, Head: ' + str(head))
#             plt.show()

        est_latent_reshape = est_latent.transpose(-1, -2).contiguous().view(batch_size, 2, seq_len, self.d_v_total)
        est_latent_reshape = est_latent_reshape.transpose(1, 2).contiguous().view(batch_size, 1, seq_len, self.d_v_total*2)
        est_latent = self.W_o(est_latent_reshape)[:,0].unsqueeze(1)
        output_dict = {
            'est_latent': est_latent,
            'attn_mat': A, 
            'x_hat': x_hat, 
            'epoch_lambdas': epoch_lambdas, 
            'unnormalized_attention': unnormalized_attention
        }

        return out, output_dict
