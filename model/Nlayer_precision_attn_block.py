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

class FullPrecisionAttentionBlockShared(nn.Module):
    """
    Precision-weighted attention block with shared query/key projections (W_q, W_k) and dynamics (lambda1).
    Each block has independent noise parameters and an adaptive step-size (delta).
    If num_inner_layers > 1, includes an optional second-order momentum update rule, defaulting to first-order when Z_v_prev is None.
    """
    def __init__(self, input_dim, query_key_dim, value_dim, args, shared_params=None):
        super().__init__()
        self.d_v = value_dim # Value embed dim
        self.d_k = query_key_dim # Key embed dim
        self.input_dim = input_dim # New: input dimension to this block (e.g., from W_v in Nlayer)

        # Eigenvalues of the state transition matrix
        # The lambda_h parameter should be sized according to the value_dim as it relates to the dynamics of V
        self.register_buffer("lambda_h", torch.zeros(2,self.d_v,1))

        # Shared projection matrices for queries and keys across all layers
        self.W_q = shared_params['W_q']
        self.W_k = shared_params['W_k']
        # Shared complex eigenvalue parameter for the underlying linear dynamics
        # Since the eigenvalues should come in complex conjugate pairs, we only need to define half of them.
        # The full eigenvalue vector lambda_h is formed from lambda1. lambda_h is non-positive, while lambda1 is unconstrained.
        self.lambda1 = shared_params['lambda1']

        # Complex identity matrix for various calculations
        self.register_buffer("complex_identity", torch.stack((torch.eye(self.d_v, self.d_v), \
                                                              torch.zeros(self.d_v, self.d_v))).unsqueeze(1))

        # Learnable noise parameters (lambda_Omega, lambda_Gamma), noise floor, and attention scaling factors (tau, nu)
        self.lambda_Omega = nn.Parameter(torch.randn(1,self.d_v,1)/self.d_v) # Process noise
        self.lambda_Gamma = nn.Parameter(torch.randn(1,self.d_v,1)/self.d_v) # Measurement noise
        self.noise_floor = nn.Parameter(torch.tensor(1.0)) # Should be 1
        self.tau = nn.Parameter(torch.tensor(1.0))
        self.nu = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("lambda_C", torch.ones(1,self.d_v,1))
        # Adaptive step-size (eta) and momentum coefficient (gamma) for the iterative latent state update
        self.eta_param = nn.Parameter(torch.zeros(1, 1, 1, self.d_v, 1)) # Gradient descent step size
        self.gamma_param = nn.Parameter(torch.zeros(1, 1, 1, self.d_v, 1)) # Momentum coefficient
        
        # Optional: Separate dynamics and noise parameters specifically for the query/key projection, for increased model capacity
        if args.sep_params == 1:
            lambda_r_k = torch.randn(int(self.d_k/2))
            lambda_i_k = torch.randn(int(self.d_k/2))
            self.lambda1_k = nn.Parameter(torch.stack((lambda_r_k,lambda_i_k)).unsqueeze(-1))
            # lambda_h_k should be related to d_k if it's for key dynamics
            self.lambda_h_k = torch.zeros(2,self.d_k,1).to(args.device)
            self.lambda_Omega_k = nn.Parameter(torch.randn(1,self.d_k,1)/self.d_k)
            self.lambda_Gamma_k = nn.Parameter(torch.randn(1,self.d_k,1)/self.d_k)
        
        self.args = args
        # Causal mask ensures that attention at a given time step only considers past and current time steps
        self.register_buffer("causal_mask", torch.tril(torch.ones(args.seq_len, args.seq_len)).view(1, args.seq_len, args.seq_len, 1, 1))
        init_weight_masks(self, args) # Initializes masks for structured sparsity if enabled
        # weight_masks are used for testing/visualizing on simple systems
        
        self.attn_mat = None
    
    def forward(self, Z_q, Z_k, Z_v, t_measure_all, Z_v_prev=None):
        """
        Forward pass of AFA block.
        Z_q: Query latent state estimates (complex-shaped).
        Z_k: Key latent state estimates (complex-shaped).
        Z_v: Current value latent state estimate (complex-shaped), corresponds to z^(k).
        t_measure_all: Time stamps for measurement points, used to compute kernels.
        Z_v_prev: Previous value latent state estimate (complex-shaped), corresponds to z^(k-1). Optional, used for momentum.
        """

        # Compute stable dynamic eigenvalues (lambda_h)
        lambda_h = compute_lambda_h(self.lambda1,self.args)
        # Compute noise magnitudes (lambda_Omega, lambda_Gamma), ensuring positivity
        lambda_Omega = torch.abs(self.lambda_Omega)
        lambda_Gamma = torch.abs(self.lambda_Gamma) + self.args.epsilon # Measurement noise must be greater than zero

        # Project input latent states into query, key, and value representations
        Q = self.W_q(Z_q).unsqueeze(-1)
        K = self.W_k(Z_k).unsqueeze(-1)
        V = Z_v.unsqueeze(-1) # Current latent state, extended for matrix operations

#         # Normalize by the time interval
#         t_measure_all = t_measure_all / (t_measure_all[:,-1] - t_measure_all[:,0]).unsqueeze(1)

        # Get t_measure
        if len(t_measure_all.size()) > 1:
            t_measure = t_measure_all[0,:-1]
        else:
            t_measure = t_measure_all[:,:-1]

        # Conditional logic for separate or shared parameters for keys
        if self.args.sep_params == 1:
            # Apply masks if structured pruning is enabled for key parameters (used for testing)
            if self.args.weight_mask == 1:
                self.lambda_h_k = self.lambda_h_k * self.eigen_mask
                self.lambda_Omega_k = self.lambda_Omega_k * self.noise_mask
                self.lambda_Gamma_k = self.lambda_Gamma_k * self.noise_mask
            
            # Compute separate dynamics and noise parameters for keys
            lambda_h_k = compute_lambda_h(self.lambda1_k, self.args)
            lambda_Omega_k = torch.abs(self.lambda_Omega_k)
            lambda_Gamma_k = torch.abs(self.lambda_Gamma_k)

            # Compute exponential kernels and state estimates/residuals for keys
            if self.args.t_equal == 1: # For equal time steps
                K_exp, K_exp2, _, _ = compute_exp_kernel(lambda_h, t_measure)
                K_exp_k, K_exp2_k, _, _ = compute_exp_kernel(lambda_h_k, t_measure)
                # State estimates and residuals:
                Z_ij_hat_all, R_qk_ij = batched_compute_estimates_and_residuals_vectorized(Q, K, V, K_exp_k, self.args)
                mat_exp = K_exp[:, -(self.args.seq_len+1), :, :] # Mat exp for next time step
            else: # For irregular time steps
                Z_ij_hat_all, R_qk_ij = compute_estimates_and_residuals_irregular_times(lambda_h_k, t_measure_all[:,:-1], Q, K, V, self.args)
                mat_exp = batched_complex_exp(lambda_h.squeeze(1).unsqueeze(0) * (t_measure_all[:,-1] - t_measure_all[:,-2]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                K_exp2 = None

            # Compute covariance kernels based on dynamics and noise, with optional tanh non-linearity
            if self.args.tanh == 0: # Using soln of Lyapunov equation
                K_cov = compute_covariance_kernel(lambda_h, lambda_Omega, lambda_Gamma, K_exp, t_measure, self.args, lambda_C=self.lambda_C)
                K_cov_k = compute_covariance_kernel(lambda_h_k, lambda_Omega_k, lambda_Gamma_k, K_exp_k, t_measure, self.args, lambda_C=self.lambda_C)
            else: # Using soln of Ricatti equation
                K_cov = compute_covariance_kernel_tanh(lambda_h, lambda_Omega, lambda_Gamma, K_exp, t_measure, self.args, lambda_C=self.lambda_C)
                K_cov_k = compute_covariance_kernel_tanh(lambda_h_k, lambda_Omega_k, lambda_Gamma_k, K_exp_k, t_measure, self.args, lambda_C=self.lambda_C)
            
            # Build marginal variance (diagonal of covariance) from the kernels
            V_ij = build_covariance_from_kernel(K_cov, self.args) # Used in attention scoring for values
            V_ij_k = build_covariance_from_kernel(K_cov_k, self.args) # Used in Mahalanobis distance for keys
            
            # Calculate Mahalanobis distance, which quantifies the "surprise" or discrepancy of a key
            mahalanobis_distance = (R_qk_ij[:,0]**2 + R_qk_ij[:,1]**2) / (V_ij_k + lambda_Gamma)
            
        else: # Shared parameters for queries, keys, and values
            if self.args.t_equal == 1: # Equal time steps
                K_exp, K_exp2, _, _ = compute_exp_kernel(lambda_h, t_measure) # Exponential kernel
                # State estimates and residuals:
                Z_ij_hat_all, R_qk_ij = batched_compute_estimates_and_residuals_vectorized(Q, K, V, K_exp, self.args)
                mat_exp = K_exp[:, -(self.args.seq_len+1), :, :]  # Mat exp for next time step
            else: # Irregular time steps
                Z_ij_hat_all, R_qk_ij = compute_estimates_and_residuals_irregular_times(lambda_h, t_measure_all[:,:-1], Q, K, V, self.args)
                mat_exp = batched_complex_exp(lambda_h.squeeze(1).unsqueeze(0) * (t_measure_all[:,-1] - t_measure_all[:,-2]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                K_exp2 = None

            if self.args.tanh == 0: # Using soln of Lyapunov equation
                K_cov = compute_covariance_kernel(lambda_h, lambda_Omega, lambda_Gamma, K_exp, t_measure, self.args, lambda_C=self.lambda_C)
            else: # Using soln of Ricatti equation
                K_cov = compute_covariance_kernel_tanh(lambda_h, lambda_Omega, lambda_Gamma, K_exp, t_measure, self.args, lambda_C=self.lambda_C)
                
            # Build marginal variance (diagonal of covariance) from the kernels
            V_ij = build_covariance_from_kernel(K_cov, self.args)
            # Calculate Mahalanobis distance, which quantifies the "surprise" or discrepancy of a key
            mahalanobis_distance = (R_qk_ij[:,0]**2 + R_qk_ij[:,1]**2) / (V_ij + lambda_Gamma)
        
        # Calculate precision-weighted attention scores. Higher precision (lower variance) and smaller Mahalanobis distance lead to higher scores.        
#            # Fixed noise floor of 1 (more principled):
#            dist_sum = torch.sum(mahalanobis_distance, axis=3, keepdims = True)
#            base_attn_scores = V_ij * (1 + dist_sum)
#            attention_scores = - torch.log(base_attn_scores)
        # Learnable noise floor and learnable parameters nu and tau for a bit more flexibility:
        dist_sum = torch.abs(self.nu) * torch.sum(mahalanobis_distance, axis=3, keepdims = True)
        base_attn_scores = V_ij * (torch.abs(self.noise_floor) + dist_sum)
        attention_scores = - torch.abs(self.tau) * torch.log(base_attn_scores + self.args.epsilon)
        
        # Apply causal mask: prevents tokens from attending to future tokens
        attention_scores.masked_fill_(self.causal_mask == 0, float('-inf'))
        # Convert attention scores to probabilities using softmax
        Q_ij = torch.softmax(attention_scores, dim=2)
        
        self.attn_mat = torch.mean(Q_ij, axis=3).squeeze()
        
        # Apply causal mask to weighted state estimates
        Z_ij_hat_all = Z_ij_hat_all * self.causal_mask

        # Compute the attention-weighted average of state estimates (z_avg^(k))
        est_v_avg = torch.sum(Q_ij.unsqueeze(1) * Z_ij_hat_all,axis=3)

        # Compute adaptive step-size (eta) and momentum coefficient (gamma)
        eta = torch.sigmoid(self.eta_param)

        # Iterative update rule for the latent state: combines current state, previous state (momentum), and weighted average
        if Z_v_prev is None or self.args.momentum == 0: # First-order update for the very first iteration or if no previous state is given
            est_latent = (1 - eta) * V + eta * est_v_avg
        else: # Second-order momentum update
#                gamma = torch.sigmoid(self.gamma_param) # Purely learned gamma (commented out)
            gamma = torch.sigmoid(self.gamma_param) * (torch.sqrt(eta) - 1)**2 # Scale gamma to ensure it stays in the overdamped regime, to improve training stability
            V_prev = Z_v_prev.unsqueeze(-1) # Ensure Z_v_prev has the correct dimensions
            est_latent = (1 + gamma - eta) * V - gamma * V_prev + eta * est_v_avg
        
        # Squeeze dimensions for downstream operations
        mat_exp_for_pred = mat_exp.squeeze(-1) # Matrix exponential for predicting next state
        est_latent = est_latent.squeeze(-1) # Final estimated latent state for this block
        Z_ij_hat_all = Z_ij_hat_all.squeeze(-1) # Attention-weighted raw estimates

        # Return the refined latent state, attention weights, raw estimates, dynamics, and prediction matrix
        return est_latent, Q_ij, Z_ij_hat_all, lambda_h, mat_exp_for_pred
        
##########################################################################################
##########################################################################################

class FullPrecisionAttentionBlock_Nlayer(nn.Module):
    """
    Multi-layer precision attention block network.
    Handles initial input projection (W_v) and final output prediction (W_p).
    Iteratively refines latent state using FullPrecisionAttentionBlockShared layers.
    """

    def __init__(self, input_dim, query_key_dim, value_dim, num_inner_layers, args):
        super().__init__()
        self.args = args

        # Store dimensions as instance attributes
        self.embed_dim = input_dim
        self.query_key_dim = query_key_dim
        self.value_dim = value_dim
        self.num_inner_layers = num_inner_layers

        # Learnable linear layers for initial input projection and final output prediction
        self.shared_W_q = ComplexLinearLayer(self.embed_dim, self.query_key_dim)
        self.shared_W_k = ComplexLinearLayer(self.embed_dim, self.query_key_dim)
        self.W_v = ComplexLinearLayer(self.embed_dim, self.value_dim)
        self.W_p = ComplexLinearLayer(self.value_dim, self.embed_dim)

        # Initialize the complex-valued linear layers
        init_complexlinear(self.shared_W_q, *init_complex_matrix(self.embed_dim, self.query_key_dim, bias=True))
        init_complexlinear(self.shared_W_k, *init_complex_matrix(self.embed_dim, self.query_key_dim, bias=True))
        init_complexlinear(self.W_v, *init_complex_matrix(self.embed_dim, self.value_dim, bias=True))
        init_complexlinear(self.W_p, *init_complex_matrix(self.value_dim, self.embed_dim, bias=True))

        # Shared complex eigenvalue parameter for the dynamics across all layers in this N-layer block
        lambda_r = torch.randn(int(self.value_dim/2))
        lambda_i = torch.randn(int(self.value_dim/2))
        self.shared_lambda1 = nn.Parameter(torch.stack((lambda_r,lambda_i)).unsqueeze(-1))

        # Bundle shared parameters to be passed to each individual FullPrecisionAttentionBlockShared layer
        self.shared_params_for_inner_blocks = {
            'W_q': self.shared_W_q,
            'W_k': self.shared_W_k,
            'lambda1': self.shared_lambda1,
        }

        # Stack multiple FullPrecisionAttentionBlockShared layers to form the iterative refinement process
        self.layers = nn.ModuleList([
            # Pass input_dim, query_key_dim, value_dim to the inner block
            FullPrecisionAttentionBlockShared(self.embed_dim, self.query_key_dim, self.value_dim, args, shared_params=self.shared_params_for_inner_blocks)
            for _ in range(self.num_inner_layers)
        ])

        # Identity tensor for internal block computations
        self.complex_identity = torch.stack((torch.eye(self.value_dim, self.value_dim),torch.zeros(self.value_dim, self.value_dim))).unsqueeze(1).to(args.device)

        init_weight_masks(self, args) # Initializes masks for structured sparsity if enabled (used for testing)

    def forward(self, Z_q, Z_k, Z_v, t_v):

        # Convert real-valued inputs to complex by appending zero imaginary parts if necessary
        if Z_q.size()[1] == 1:
            Z_q = torch.cat((Z_q, torch.zeros_like(Z_q)),dim=1)
            Z_k = torch.cat((Z_k, torch.zeros_like(Z_k)),dim=1)
            Z_v = torch.cat((Z_v, torch.zeros_like(Z_v)),dim=1)

        # Project inputs into the initial latent state (est_v) for the first iteration
        est_v = self.W_v(Z_v)
        
        if self.num_inner_layers == 1:
            layer = self.layers[0] 
            est_v, final_Q, final_X_hat, final_lambda_h, mat_exp_for_pred = \
                layer(Z_q=Z_q, Z_k=Z_k, Z_v=est_v, t_measure_all=t_v, Z_v_prev=None)
        else:
            # Initialize placeholders for outputs from the *last* attention block
            final_Q, final_X_hat, final_lambda_h, mat_exp_for_pred = None, None, None, None
        
            # Initialize the previous latent state to None for the first attention block, disabling momentum initially
            est_v_prev = None

            # Iterate through each attention block, refining the latent state estimate
            for layer_idx, layer in enumerate(self.layers):
                # Pass current latent state (est_v) as Z_q, Z_k, Z_v.
                # Pass the *previous* latent state (est_v_prev) for the momentum term.
                est_v_next, Q_ij, Z_ij_hat_all, lambda_h, current_mat_exp_for_pred = \
                    layer(Z_q=est_v, Z_k=est_v, Z_v=est_v, t_measure_all=t_v, Z_v_prev=est_v_prev)

                # Update states for the next iteration: current est_v becomes est_v_prev, est_v_next becomes est_v
                est_v_prev = est_v
                est_v = est_v_next

            # Store the outputs of the *last* layer; these will be used for the final prediction
            final_Q, final_X_hat, final_lambda_h, mat_exp_for_pred = \
                Q_ij, Z_ij_hat_all, lambda_h, current_mat_exp_for_pred

        # Ensure that the matrix exponential for prediction was computed (i.e., at least one layer ran)
        if mat_exp_for_pred is None:
            raise ValueError("mat_exp_for_pred was not computed. Check num_inner_layers.")

        # The final refined latent state after all attention layers
        est_latent = est_v

        # Predict the latent state at the next time step using the matrix exponential
        predicted_latent_state = batched_complex_hadamard(mat_exp_for_pred, est_latent)

        # Project the predicted latent state back to the original embedding dimension for the final output
        final_prediction_output = self.W_p(predicted_latent_state.squeeze(-1)).squeeze(-1)

        # Return the final latent state, attention components, and prediction
        return est_v, final_prediction_output, final_Q, final_X_hat, final_lambda_h