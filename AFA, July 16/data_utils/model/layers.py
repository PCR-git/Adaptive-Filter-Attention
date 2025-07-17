import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import math

from utils import batched_complex_matmul, batched_complex_matmul_full

from precision_attention import compute_residual_norm

from model import init_complexlinear, init_complex_matrix

##########################################################################################
##########################################################################################

class AttentionLayer(nn.Module):
    """
    Custom attention layer.
    """
    def __init__(self, input_dim: int, hidden_dim: int, args):
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Projections for Q, K, V, and final output
        self.query_projection = nn.Linear(input_dim, hidden_dim)
        self.key_projection = nn.Linear(input_dim, hidden_dim)
        self.value_projection = nn.Linear(input_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, input_dim)

        # Scaling factor for dot-product attention
        self.scale = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))

        # Causal mask for autoregressive attention
        self.causal_mask = torch.tril(torch.ones(args.seq_len, args.seq_len, device=args.device)).unsqueeze(0)

        self.args = args

    def forward(self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        """
        Forward pass.

        Args:
            query: (batch_size, 1, input_dim)
            keys: (batch_size, seq_len, input_dim)
            values: (batch_size, seq_len, input_dim)

        Returns:
            context_vector: (batch_size, 1, input_dim)
            attention_weights: (batch_size, 1, seq_len)
        """

        # Project Q, K, V
        proj_query = self.query_projection(query)          # (B, 1, H)
        proj_keys = self.key_projection(keys)              # (B, L, H)
        proj_values = self.value_projection(values)        # (B, L, H)

        # Compute dot-product attention scores
        attn_scores = torch.matmul(proj_query, proj_keys.transpose(-2, -1))  # (B, 1, L)
        attn_scores = attn_scores.masked_fill(self.causal_mask == 0, float('-inf'))

        # Scale and normalize scores
        scaled_attn_scores = attn_scores / self.scale
        attn_weights = F.softmax(scaled_attn_scores, dim=-1)            # (B, 1, L)

        # Weighted sum of values
        context_vector = torch.matmul(attn_weights, proj_values)             # (B, 1, H)
        out = self.output_projection(context_vector)                              # (B, 1, input_dim)

        return out, attn_weights

##########################################################################################
##########################################################################################

class ComplexLinearLayer(nn.Module):
    """
    Complex-valued linear layer
    """ 
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.real = nn.Linear(in_features, out_features, bias=bias)
        self.imag = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, inputs):
        # input: (B, 2, in_features, 1)
        real_input = inputs[:, 0]
        imag_input = inputs[:, 1]

        real_out = self.real(real_input) - self.imag(imag_input)
        imag_out = self.real(imag_input) + self.imag(real_input)

        return torch.stack((real_out, imag_out), dim=1)
     
##########################################################################################
##########################################################################################

class ComplexAttentionLayer(nn.Module):
    """
    Complex-valued attention layer
    """
    def __init__(self, input_dim: int, hidden_dim: int, args):
        super(ComplexAttentionLayer, self).__init__()
        
        # Define complex linear layers
        self.W_q = ComplexLinearLayer(input_dim, hidden_dim)
        self.W_k = ComplexLinearLayer(input_dim, hidden_dim)
        self.W_v = ComplexLinearLayer(input_dim, hidden_dim)
        self.W_p = ComplexLinearLayer(input_dim, hidden_dim)
        
        # Initialize complex linear layers
        Wqi, bqi = init_complex_matrix(input_dim, hidden_dim, bias=True)
        Wki, bki = init_complex_matrix(input_dim, hidden_dim, bias=True)
        Wvi, bvi = init_complex_matrix(input_dim, hidden_dim, bias=True)
        Wpi, bpi = init_complex_matrix(input_dim, hidden_dim, bias=True)
        
        init_complexlinear(self.W_q, Wqi, bqi)
        init_complexlinear(self.W_k, Wki, bki)
        init_complexlinear(self.W_v, Wvi, bvi)
        init_complexlinear(self.W_p, Wpi, bpi)
        
        # Scaling factor for attention
        self.scale = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        
        # Causal mask
        self.causal_mask = torch.tril(torch.ones(args.seq_len, args.seq_len, device=args.device)).unsqueeze(0)
        
        self.noise_floor = nn.Parameter(torch.tensor(1.0))
        self.tau = nn.Parameter(torch.tensor(0.0))
        
        self.args = args

    def forward(self, Z_q, Z_k, Z_v):
        """
        Forward pass for complex-valued attention.

        Inputs:
            Z_q, Z_k, Z_v: complex-valued input tensors of shape 
                           (batch_size, 2, seq_len, input_dim, 1)
                           where the second dimension holds [real, imag]

        Returns:
            out: complex-valued output tensor of shape 
                 (batch_size, 2, seq_len, d_v, 1)
        """
        
        using_real = 0
        if len(Z_q.size()) == 3:
            using_real = 1
            Z_q = torch.stack((Z_q,torch.zeros_like(Z_q).to(Z_q.device)),axis=1)
            Z_k = torch.stack((Z_k,torch.zeros_like(Z_k).to(Z_k.device)),axis=1)
            Z_v = torch.stack((Z_v,torch.zeros_like(Z_v).to(Z_v.device)),axis=1)
        
        # Compute query, key, and value matrices
        Q = self.W_q(Z_q)
        K = self.W_k(Z_k)
        V = self.W_v(Z_v)

        # Conjugate-transpose of K
        Kp = K.squeeze(-1).permute(0, 1, 3, 2)  # transpose last two dims
        Kp[:, 1] *= -1  # negate imaginary part

        # Dot product attention
        if self.args.metric_type == 'RealDotProduct':
            # Compute QK (complex matrix product)
            QK = batched_complex_matmul_full(Q.squeeze(-1), Kp)

            # Compute unnormalized attention: norm squared of complex QK
            attn_scores = F.softplus(self.tau) * QK[:,0]
#             attn_scores = torch.clamp(attn_scores, min=-1e30, max=1e30)
            
        if self.args.metric_type == 'MagDotProduct':
            # Compute QK (complex matrix product)
            QK = batched_complex_matmul_full(Q.squeeze(-1), Kp)

            # Compute unnormalized attention: norm squared of complex QK
            attn_scores = F.softplus(self.tau) * torch.sum(torch.abs(QK)**2, dim=1)
#             attn_scores = torch.clamp(attn_scores, min=1e-30, max=1e30)
        
        # Inverse Mahalanobis attention
        elif self.args.metric_type == 'InverseMahalanobis':
            R_squared_magnitude = compute_residual_norm(Q, K)
            attn_scores = - F.softplus(self.tau) * torch.log(self.noise_floor**2 + R_squared_magnitude + self.args.epsilon)
#             attn_scores = torch.clamp(attn_scores, min=1e-30, max=1e30)
        
        attn_scores = attn_scores.masked_fill(self.causal_mask == 0, float('-inf'))

        # Scale and normalize scores
        scaled_attn_scores = attn_scores / self.scale
        attn_weights = F.softmax(scaled_attn_scores, dim=-1)            # (B, 1, L)
        
        attn_weights_complex = torch.stack((attn_weights, torch.zeros_like(attn_weights).to(attn_weights.device)), dim=1)

        # Multiply by V to get output
        est_v = batched_complex_matmul_full(attn_weights_complex, V.squeeze(-1))
        
        # Output projection
        out = self.W_p(est_v)
        
        if using_real == 1:
            out = out[:,0]

        return out, attn_weights
    
##########################################################################################
##########################################################################################

class GatedSequenceNorm(nn.Module):
    """
    Performs temporal normalization of real-valued input, centering,
    normalizing by the standard deviation along the sequence/time dimension,
    then applying a learnable scale and shift.
    Includes gating parameters to allow the model to learn whether and by how much
    to apply mean subtraction and std division.
    """
    
    def __init__(self, input_dim, args):
        super().__init__()
        
        # Gating parameters (so the model can learn whether and by how much to apply
        # subtraction of mean and division by std)
        # These operate per feature dimension
        self.gp1 = nn.Parameter(torch.ones(input_dim)) # Gate for mean subtraction
        self.gp2 = nn.Parameter(torch.ones(input_dim)) # Gate for std scaling (numerator)
        self.gp3 = nn.Parameter(torch.ones(input_dim)) # Gate for std scaling (epsilon/offset)
        
        # Scale and shift params (learnable affine transformation)
        # These operate per feature dimension
        self.alpha = nn.Parameter(torch.ones(input_dim))  # Scale
        self.beta = nn.Parameter(torch.zeros(input_dim)) # Shift
        
        self.args = args

    def forward(self, x):
        # x is expected to be (batch_size, seq_len, input_dim)
        
        # Subtract mean along the sequence/time dimension (dim=-2)
        # The gating parameter gp1 modulates the mean subtraction
        x_centered = x - self.gp1 * x.mean(dim=-2, keepdim=True)

        # Standard deviation along the sequence/time dimension (dim=-2)
        x_std = x_centered.std(dim=-2, keepdim=True)
        
        # Calculate the normalizer using gating parameters
        # F.softplus ensures gp2 and gp3 result in non-negative values for stability
        normalizer = F.softplus(self.gp2) * x_std + F.softplus(self.gp3) + 1e-5 # Added small epsilon for numerical stability

        # Apply the normalization
        x_norm = x_centered / normalizer
        
        # Apply learnable scale and shift
        x_out = self.alpha * x_norm + self.beta
        
        return x_out
    
##########################################################################################
##########################################################################################

class GatedLayerNorm(nn.Module):
    """
    Performs layer normalization of real-valued input, centering,
    normalizing by the standard deviation along the feature/embed dimension,
    then applying a learnable scale and shift.
    Includes gating parameters to allow the model to learn whether and by how much
    to apply mean subtraction and std division.
    """
    
    def __init__(self, input_dim, args):
        super().__init__()
        
        # Gating parameters (so the model can learn whether and by how much to apply
        # subtraction of mean and division by std)
        # These operate per feature dimension
        self.gp1 = nn.Parameter(torch.ones(input_dim)) # Gate for mean subtraction
        self.gp2 = nn.Parameter(torch.ones(input_dim)) # Gate for std scaling (numerator)
        self.gp3 = nn.Parameter(torch.ones(input_dim)) # Gate for std scaling (epsilon/offset)
        
        # Scale and shift params (learnable affine transformation)
        # These operate per feature dimension
        self.alpha = nn.Parameter(torch.ones(input_dim))  # Scale
        self.beta = nn.Parameter(torch.zeros(input_dim)) # Shift
        
        self.args = args

    def forward(self, x):
        # x is expected to be (batch_size, seq_len, input_dim)
        
        # Subtract mean along the feature dimension (dim=-1)
        # The gating parameter gp1 modulates the mean subtraction
        x_centered = x - self.gp1 * x.mean(dim=-1, keepdim=True)

        # Standard deviation along the feature dimension (dim=-1)
        x_std = x_centered.std(dim=-1, keepdim=True)
        
        # Calculate the normalizer using gating parameters
        # F.softplus ensures gp2 and gp3 result in non-negative values for stability
        normalizer = F.softplus(self.gp2) * x_std + F.softplus(self.gp3) + 1e-5 # Added small epsilon for numerical stability

        # Apply the normalization
        x_norm = x_centered / normalizer
        
        # Apply learnable scale and shift
        x_out = self.alpha * x_norm + self.beta
        
        return x_out
    
##########################################################################################
##########################################################################################

class GatedComplexSequenceNorm(nn.Module):
    """
    Performs temporal norm of complex-valued input by rotating to a new basis, centering,
    normalizing by the standard deviation of the magnitude in the new basis along the sequence/time dimension, then rotating back.
    (This is also known as Instance Normalization.)
    """
    
    def __init__(self, input_dim, args):
        super().__init__()
        
        # Linear layers
        self.l1 = ComplexLinearLayer(input_dim, input_dim)
        self.l2 = ComplexLinearLayer(input_dim, input_dim)
        
        # Initialize linear layers
        W1, b1 = init_complex_matrix(input_dim, input_dim, bias=True)
        W2, b2 = init_complex_matrix(input_dim, input_dim, bias=True)
        init_complexlinear(self.l1, W1, b1)
        init_complexlinear(self.l2, W2, b2)
        
        # Gating parameters (so the model can learn whether and by how much to apply substraction of mean and division by std)
        self.gp1 = nn.Parameter(torch.ones(input_dim))
        self.gp2 = nn.Parameter(torch.ones(input_dim))
        self.gp3 = nn.Parameter(torch.ones(input_dim))
        
        # Scale and shift params
        self.alpha_r = nn.Parameter(torch.ones(input_dim)) # Vector of ones
        self.beta_r = nn.Parameter(torch.zeros(input_dim)) # Vector of zeros
        self.alpha_i = nn.Parameter(torch.ones(input_dim)) # Vector of ones
        self.beta_i = nn.Parameter(torch.zeros(input_dim)) # Vector of zeros
        
        self.args = args

    def forward(self, x):

        x_rot = self.l1(x) # Rotate to new basis

        x_rot_centered = x_rot - self.gp1 * x_rot.mean(dim=-2, keepdim=True) # Subtract mean along embed dim

        real, imag = x_rot_centered[:, 0], x_rot_centered[:, 1] # Get real and imaginary parts
        mag = torch.sqrt(real**2 + imag**2) # Magnitude
        
        mag_std = mag.std(dim=-2, keepdim=True) # Standard dev of magnitude along embed dim
        
        normalizer = F.softplus(self.gp2) * mag_std + F.softplus(self.gp3) + 1

        # Apply the scaling to real and imaginary parts based on normalized magnitude
        # This keeps the phase.
        real_norm = real / normalizer # Normalized magnitude * cosine(phase)
        imag_norm = imag / normalizer # Normalized magnitude * sine(phase)
        
        # Scale and shift
        real_out = self.alpha_r * real_norm + self.beta_r
        imag_out = self.alpha_i * imag_norm + self.beta_i
    
        x_norm = torch.stack((real_out, imag_out),dim=1) # Reconstruct full complex vector

        x_out = self.l2(x_norm) # Rotate back

        return x_out
    
##########################################################################################
##########################################################################################

class GatedComplexLayerNorm(nn.Module):
    """
    Performs layer norm of complex-valued input by rotating to a new basis, centering,
    normalizing by the standard deviation of the magnitude in the new basis along the embed dimension, then rotating back.
    """
    
    def __init__(self, input_dim, args):
        super().__init__()
        
        # Linear layers
        self.l1 = ComplexLinearLayer(input_dim, input_dim)
        self.l2 = ComplexLinearLayer(input_dim, input_dim)
        
        # Initialize linear layers
        W1, b1 = init_complex_matrix(input_dim, input_dim, bias=True)
        W2, b2 = init_complex_matrix(input_dim, input_dim, bias=True)
        init_complexlinear(self.l1, W1, b1)
        init_complexlinear(self.l2, W2, b2)
        
        # Gating parameters (so the model can learn whether and by how much to apply substraction of mean and division by std)
        self.gp1 = nn.Parameter(torch.ones(input_dim))
        self.gp2 = nn.Parameter(torch.ones(input_dim))
        self.gp3 = nn.Parameter(torch.ones(input_dim))
        
        # Scale and shift params
        self.alpha_r = nn.Parameter(torch.ones(input_dim)) # Vector of ones
        self.beta_r = nn.Parameter(torch.zeros(input_dim)) # Vector of zeros
        self.alpha_i = nn.Parameter(torch.ones(input_dim)) # Vector of ones
        self.beta_i = nn.Parameter(torch.zeros(input_dim)) # Vector of zeros
        
        self.args = args

    def forward(self, x):

        x_rot = self.l1(x) # Rotate to new basis

        x_rot_centered = x_rot - self.gp1 * x_rot.mean(dim=-1, keepdim=True) # Subtract mean along embed dim

        real, imag = x_rot_centered[:, 0], x_rot_centered[:, 1] # Get real and imaginary parts
        mag = torch.sqrt(real**2 + imag**2) # Magnitude
        
        mag_std = mag.std(dim=-1, keepdim=True) # Standard dev of magnitude along embed dim
        
        normalizer = F.softplus(self.gp2) * mag_std + F.softplus(self.gp3) + 1

        # Apply the scaling to real and imaginary parts based on normalized magnitude
        # This keeps the phase.
        real_norm = real / normalizer # Normalized magnitude * cosine(phase)
        imag_norm = imag / normalizer # Normalized magnitude * sine(phase)
        
        # Scale and shift
        real_out = self.alpha_r * real_norm + self.beta_r
        imag_out = self.alpha_i * imag_norm + self.beta_i
    
        x_norm = torch.stack((real_out, imag_out),dim=1) # Reconstruct full complex vector

        x_out = self.l2(x_norm) # Rotate back

        return x_out
    
##########################################################################################
##########################################################################################

class RealPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for real-valued inputs,
    as introduced in "Attention Is All You Need" (Vaswani et al., 2017).

    Adds deterministic, non-learned position-dependent signals to the input.
    """

    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model (int): Dimensionality of the input embeddings.
            max_len (int): Maximum sequence length supported.
        """
        super().__init__()
        
        # Create zero tensor of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)  # (L, D)

        # Position indices: shape (L, 1)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # Frequency terms for sinusoids: shape (D/2,)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))

        # Apply sin to even dimensions (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd dimensions (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, L, D) to broadcast across batches
        pe = pe.unsqueeze(0)

        # Register as buffer (non-trainable, saved with model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to the input tensor.

        Args:
            x (Tensor): Input of shape (B, L, D)

        Returns:
            Tensor: Positionally encoded input of shape (B, L, D)
        """
        # Add positional encoding up to sequence length x.size(1)
        return x + self.pe[:, :x.size(1)]
    
##########################################################################################
##########################################################################################

class ComplexPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding adapted for complex-valued inputs.
    Generates real and imaginary components separately, maintaining the same
    structure across both even and odd indices.

    This is intended for inputs of shape (batch, 2, seq_len, embed_dim),
    where the second dimension holds the real and imaginary parts respectively.
    """

    def __init__(self, seq_len, embed_dim, device='cpu'):
        super().__init__()

        # Create a position index (seq_len, 1)
        position = torch.arange(seq_len).unsqueeze(1)

        # Compute frequency divisors (embed_dim/2,) using exponential decay
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * 
                             -(np.log(10000.0) / embed_dim))

        # Phase encoding: shape (seq_len, embed_dim/2)
        phase = position * div_term

        # Allocate real and imaginary positional embeddings
        pe_real = torch.zeros(seq_len, embed_dim)
        pe_imag = torch.zeros(seq_len, embed_dim)

        # Apply cosine to real part (both even and odd positions)
        pe_real[:, 0::2] = torch.cos(phase)
        pe_real[:, 1::2] = torch.cos(phase)  # optional: symmetric version

        # Apply sine to imaginary part (both even and odd positions)
        pe_imag[:, 0::2] = torch.sin(phase)
        pe_imag[:, 1::2] = torch.sin(phase)  # optional: symmetric version

        # Stack into shape (2, seq_len, embed_dim), where index 0 is real and 1 is imag
        pe = torch.stack([pe_real, pe_imag], dim=0)

        # Register as a buffer so it's saved in the model but not updated by optimizer
        self.register_buffer("pos_enc", pe.to(device))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input of shape (batch, 2, seq_len, embed_dim)
        
        Returns:
            torch.Tensor: Input with complex sinusoidal positional encoding added.
        """
        # Broadcast positional encoding across batch dimension
        return x + self.pos_enc.unsqueeze(0)  # shape (1, 2, seq_len, embed_dim)

##########################################################################################
##########################################################################################

class LearnedRealPositionalEncoding(nn.Module):
    """
    Learned real-valued positional encoding for complex input tensors shaped (B, 2, L, D).
    The same real encoding is added to both the real and imaginary parts.
    """
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # Create a learnable (L, D) real-valued positional encoding
        self.pe = nn.Parameter(torch.randn(seq_len, embed_dim) * 0.02)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 2, L, D), complex-valued input

        Returns:
            x + positional encoding (broadcasted over batch and complex dims)
        """
        # Shape: (1, 1, L, D) to broadcast over batch and complex dims
        pe_expanded = self.pe.unsqueeze(0).unsqueeze(0)
        return x + pe_expanded

##########################################################################################
##########################################################################################

class LearnedComplexPositionalEncoding(nn.Module):
    """
    Learned complex-valued positional encoding for input of shape (B, 2, L, D).
    Encodes position using separate learned real and imaginary components.
    """

    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # Learnable complex positional encoding: shape (2, L, D)
        self.pe = nn.Parameter(torch.randn(2, seq_len, embed_dim) * 0.02)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 2, L, D) — complex input

        Returns:
            x + learned complex positional encoding (broadcasted)
        """
        # Expand shape to (1, 2, L, D) for broadcasting over batch dim
        return x + self.pe.unsqueeze(0)

##########################################################################################
##########################################################################################

# def initialize_identity(linear_layer):
#         """
#         Initializes a linear layer to approximate an identity mapping.
#         Sets weights to identity (or a scaled identity if dims differ) and biases to zero.
#         """
#         # Ensure bias is zero
#         if linear_layer.bias is not None:
#             nn.init.constant_(linear_layer.bias, 0.0)

#         # Initialize weights
#         if linear_layer.in_features == linear_layer.out_features:
#             nn.init.eye_(linear_layer.weight)
#         else:
#             nn.init.xavier_uniform_(linear_layer.weight)
#             print(f"Warning: Initializing non-square layer {linear_layer} with Xavier Uniform instead of identity.")
            

            