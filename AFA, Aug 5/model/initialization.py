import numpy as np
import torch
import torch.nn as nn

from utils import to_complex

##########################################################################################
##########################################################################################

# def init_complex_matrix(d_1, d_2):
#     """
#     Isotropic initialization of a complex-valued matrix
#     """

#     scale = np.sqrt(2/(d_1 + d_2))
#     mag = scale*torch.randn(d_1, d_2)
#     phase = 2 * np.pi * torch.rand(d_1, d_2) # Phase
#     real = mag * torch.cos(phase) # Real part
#     imag = mag * torch.sin(phase) # Imaginary part
#     W = torch.stack([real, imag]).unsqueeze(1)

#     return W

def init_complex_matrix(d_1, d_2, bias=False):
    """
    Isotropic initialization of a complex-valued matrix and optional bias.
    
    Returns:
        W: torch.Tensor of shape (2, 1, d_1, d_2) for weights
        b: torch.Tensor of shape (2, 1, d_2) for bias (if bias=True)
    """
    scale = np.sqrt(2 / (d_1 + d_2))
    mag = scale * torch.randn(d_1, d_2)
    phase = 2 * np.pi * torch.rand(d_1, d_2)

    real = mag * torch.cos(phase)
    imag = mag * torch.sin(phase)
    W = torch.stack([real, imag]).unsqueeze(1)  # (2, 1, d_1, d_2)

    if bias:
        mag_b = scale * torch.randn(d_2)
        phase_b = 2 * np.pi * torch.rand(d_2)
        real_b = mag_b * torch.cos(phase_b)
        imag_b = mag_b * torch.sin(phase_b)
        b = torch.stack([real_b, imag_b]).unsqueeze(1)  # (2, 1, d_2)
        return W, b

    return W

##########################################################################################
##########################################################################################
        
def init_complexlinear(linear_layer, weight_tensor, bias_tensor=None):
    """
    Initializes a ComplexLinear layer from complex weight (and optional bias) tensors.

    weight_tensor: shape (2, 1, d_in, d_out)
    bias_tensor: shape (2, 1, d_out)
    """
    real_w = weight_tensor[0, 0].T  # (d_out, d_in)
    imag_w = weight_tensor[1, 0].T

    with torch.no_grad():
        linear_layer.real.weight.copy_(real_w)
        linear_layer.imag.weight.copy_(imag_w)

        if bias_tensor is not None:
            linear_layer.real.bias.copy_(bias_tensor[0, 0])
            linear_layer.imag.bias.copy_(bias_tensor[1, 0])

##########################################################################################
##########################################################################################

def build_nearly_identity(args):
    """
    Build a "nearly-identity" attention matrix (for testing)
    """
    
    # Create a distance matrix from the diagonal
    i = torch.arange(args.seq_len, device=args.device)
    j = torch.arange(args.seq_len, device=args.device)
    dist = (i[:, None] - j[None, :]).abs()  # Shape: [Npts, Npts]

    # Exponential decay from diagonal
    decay = torch.exp(-dist.float() / 5)  # You can set length_scale, e.g. 5.0

    # Normalize each row to sum to 1
    Q_ij = decay / decay.sum(dim=1, keepdim=True)

    # Add batch and complex dims (or other dummy dims as needed)
    return Q_ij.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # Shape: [1, Npts, Npts, 1, 1]

##########################################################################################
##########################################################################################

# # NOT USED
# def create_lower_triangular_parameter(size):
#     """
#     Creates a lower triangular parameter matrix.
#     """
    
#     indices = torch.tril_indices(row=size, col=size)
#     data = nn.Parameter(torch.randn(indices[0].size(0)))
#     weight = torch.zeros(size, size, device=data.device)
#     weight[indices[0], indices[1]] = data
#     return weight

##########################################################################################
##########################################################################################

def initialize_to_correct_model(module, D1, S1, Si1, sigma_process, sigma_process_0, sigma_measure, args):
    """
    Initialize to correct model parameter values (for testing)
    """

    with torch.no_grad():

        # Get actual eigenvalues
        lambda_h = torch.randn(2, args.d_v).to(args.device)
        lambda_h[0,0:2] = torch.diag(D1[0].squeeze())
        lambda_h[1,0:2] = torch.diag(D1[1].squeeze())
        lambda_h = lambda_h.unsqueeze(1).unsqueeze(-1)
        module.lambda1[:,0] = torch.nn.Parameter(lambda_h[:, :, 0, :].squeeze(-1))

        module.W_v[:,:,0:2,0:2] = torch.nn.Parameter(Si1.unsqueeze(1))
        module.W_q[:,:,0:2,0:2] = torch.nn.Parameter(Si1.unsqueeze(1))
        module.W_k[:,:,0:2,0:2] = torch.nn.Parameter(Si1.unsqueeze(1))
        module.W_r[:,:,0:2,0:2] = torch.nn.Parameter(Si1.unsqueeze(1))
        module.W_e[:,:,0:2,0:2] = torch.nn.Parameter(S1.unsqueeze(1))
        module.W_p[:,:,0:2,0:2] = torch.nn.Parameter(S1.unsqueeze(1))

        # Process and measurement noise
        lambda_Omega = (np.sqrt(sigma_process)*torch.ones(1,args.d_v,1)).to(args.device)
        lambda_Omega0 = (np.sqrt(sigma_process_0)*torch.ones(1,args.d_v,1)).to(args.device)
        lambda_C = 1.0*torch.ones(1,args.d_v,1).to(args.device)
        lambda_Gamma = (np.sqrt(sigma_measure)*torch.ones(1,args.d_v,1)).to(args.device)

        module.lambda_Omega_sqrt = torch.nn.Parameter(lambda_Omega)
        module.lambda_Omega0_sqrt = torch.nn.Parameter(lambda_Omega0)
        module.lambda_Gamma_sqrt = torch.nn.Parameter(lambda_Gamma)
    #     module.lambda_C = torch.nn.Parameter(lambda_C)

    #     module.alpha = torch.nn.Parameter(torch.tensor(0.5))
    #     module.beta = torch.nn.Parameter(torch.tensor(0.5))

    # DD = torch.stack((torch.diag(lambda_h[0].squeeze()),torch.diag(lambda_h[1].squeeze()))).unsqueeze(1)
    # A_actual = complex_matmul(Pd,complex_matmul(DD,Pu))
    # A_actual = complex_matmul(R1i,complex_matmul(DD,R1))[:,:,0:args.m,0:args.m]

        module.W_q_b = nn.Parameter(torch.zeros(1,2,1,args.d_k)).to(args.device) # Query weight matrix
        module.W_k_b = nn.Parameter(torch.zeros(1,2,1,args.d_k)).to(args.device) # Key weight matrix
        module.W_v_b = nn.Parameter(torch.zeros(1,2,1,args.d_v)).to(args.device) # Value weight matrix
        module.W_p_b = nn.Parameter(torch.zeros(1,2,1,args.head_dim)).to(args.device) # Prediction output weight matrix
        module.W_r_b = nn.Parameter(torch.zeros(1,2,1,args.d_v)).to(args.device) # Residual weight matrix
        module.W_e_b = nn.Parameter(torch.zeros(1,2,1,args.head_dim)).to(args.device) # Estimate output weight matrix

        print('Model initialized to match true dynamics.')
        
##########################################################################################
##########################################################################################
     
# Helper to expand real-valued matrix into (real, imag) weights for ComplexLinearLayer
def set_complex_weight(layer, mat):
    """
    Set weights of a ComplexLinearLayer to real-valued 2x2 matrix.
    Expects mat to be shape (2, 2, 2) where:
        mat[0] = real part, mat[1] = imag part (ignored here).
    """
    assert mat.shape == (2, 2, 2), f"Expected mat shape (2, 2, 2), got {mat.shape}"

    real_mat = mat[0]  # Extract real part (2x2)
    imag_mat = mat[1]  # Extract imag part (2x2)

    layer.real.weight.data.zero_()
    layer.imag.weight.data.zero_()
    layer.real.weight.data[:2, :2] = real_mat
    layer.imag.weight.data[:2, :2] = imag_mat

    if layer.real.bias is not None:
        layer.real.bias.data.zero_()
        layer.imag.bias.data.zero_()
        
##########################################################################################
##########################################################################################     

def initialize_net_to_correct_model(module, D1, S1, Si1, sigma_process, sigma_measure, args, scale_c=10):
    """
    Initialize a complex precision attention block to match a known stable linear model.
    Assumes module uses ComplexLinearLayer for W_q, W_k, W_v, W_p, etc.
    """
    
    with torch.no_grad():
#         scale_r = scale_c / (2 * args.tf)
#         scale_i = scale_c * 2 * np.pi / (2 * args.tf)
#         scale_r = scale_c / 2
#         scale_i = scale_c * np.pi
        scale_r = 1
        scale_i = 1
        
        # Initialize lambda1 (or shared_lambda1)
        if hasattr(module, 'lambda1') and module.lambda1 is not None:
            # This covers `module.lambda1` for individual layers and `model.lambda1` if it's the actual shared param.
            # Assuming D1's shape matches lambda1's expectation for initialization
            lambda1_target = torch.zeros_like(module.lambda1).to(args.device)
            if D1.dim() == 3:
                lambda1_target[0, 0:D1.shape[1], 0] = torch.diag(D1[0].squeeze()) / scale_r
                lambda1_target[1, 0:D1.shape[1], 0] = torch.diag(D1[1].squeeze()) / scale_i
            else:
                lambda1_target[0, :, 0] = D1[0] / scale_r
                lambda1_target[1, :, 0] = D1[1] / scale_i
            module.lambda1.copy_(lambda1_target * args.tf) # MUST ADD args.tf if normalizing time inside model

        if hasattr(module, 'lambda_real_v') and module.lambda_real_v is not None:
            lambda_real_target = D1[0,0,0] / scale_r
            module.lambda_real_v.copy_(lambda_real_target * args.tf)
        if hasattr(module, 'lambda_imag_v') and module.lambda_imag_v is not None:
            lambda_imag_target = D1[1,0,0] / scale_i
            module.lambda_imag_v.copy_(lambda_imag_target * args.tf)
            
        if hasattr(module, 'lambda_real_k') and module.lambda_real_k is not None:
            lambda_real_target = D1[0,0,0] / scale_r
            module.lambda_real_k.copy_(lambda_real_target * args.tf)
        if hasattr(module, 'lambda_imag_k') and module.lambda_imag_k is not None:
            lambda_imag_target = D1[1,0,0] / scale_i
            module.lambda_imag_k.copy_(lambda_imag_target * args.tf)
            
        # Handle 'shared_lambda1' specifically if it's a separate attribute (e.g., on the main Nlayer model)
        if hasattr(module, 'shared_lambda1') and module.shared_lambda1 is not None:
            shared_lambda1_target = torch.zeros_like(module.shared_lambda1).to(args.device)
            if D1.dim() == 3:
                shared_lambda1_target[0, 0:D1.shape[1], 0] = torch.diag(D1[0].squeeze()) / scale_r
                shared_lambda1_target[1, 0:D1.shape[1], 0] = torch.diag(D1[1].squeeze()) / scale_i
            else:
                shared_lambda1_target[0, :, 0] = D1[0] / scale_r
                shared_lambda1_target[1, :, 0] = D1[1] / scale_i
            module.shared_lambda1.copy_(shared_lambda1_target * args.tf) # MUST ADD args.tf if normalizing time inside model

        # Set encoder layers (Si1 -> W_q, W_k, W_v)
        for name in ['W_q', 'W_k', 'shared_W_q', 'shared_W_k']:
            if hasattr(module, name) and getattr(module, name) is not None:
                set_complex_weight(getattr(module, name), Si1)
        
        # W_v remains
        if hasattr(module, 'W_v') and getattr(module, 'W_v') is not None:
            set_complex_weight(getattr(module, 'W_v'), Si1)

        # Set decoder layers (S1 -> W_p)
        try:
            for name in ['W_p']:
                if hasattr(module, name) and getattr(module, name) is not None:
                    set_complex_weight(getattr(module, name), S1)
        except:
            pass

        # Optionally set residual and estimate layers
        for name in ['W_r', 'W_e']:
            if hasattr(module, name) and getattr(module, name) is not None:
                set_complex_weight(getattr(module, name), S1)

        # Zero all biases
        for name in ['W_q', 'W_k', 'W_v', 'W_p', 'W_r', 'W_e', 'shared_W_q', 'shared_W_k']:
            try:
                if hasattr(module, name) and getattr(module, name) is not None:
                    layer_weight = getattr(module, name)
                    if layer_weight.real.bias is not None:
                        layer_weight.real.bias.data.zero_()
                    if layer_weight.imag.bias is not None:
                        layer_weight.imag.bias.data.zero_()
            except:
                pass

        # Initialize noise parameters
        # These are generally specific to FullPrecisionAttentionBlockShared instances
        if hasattr(module, 'lambda_Omega') and module.lambda_Omega is not None:
            lambda_Omega = (sigma_process * torch.ones_like(module.lambda_Omega)).to(args.device)
            module.lambda_Omega.copy_(lambda_Omega)
        if hasattr(module, 'lambda_Gamma') and module.lambda_Gamma is not None:
            lambda_Gamma = (sigma_measure * torch.ones_like(module.lambda_Gamma)).to(args.device)
            module.lambda_Gamma.copy_(lambda_Gamma)
            
        if hasattr(module, 'lambda_omega') and module.lambda_omega is not None:
            lambda_omega = (sigma_process).to(args.device)
            module.lambda_omega.copy_(lambda_omega)
        if hasattr(module, 'lambda_gamma') and module.lambda_gamma is not None:
            lambda_gamma = (sigma_measure).to(args.device)
            module.lambda_gamma.copy_(lambda_gamma)
        
        # Check for lambda_C, ensure it's a tensor/parameter before copying
        if hasattr(module, 'lambda_C') and module.lambda_C is not None:
            if isinstance(module.lambda_C, torch.nn.Parameter):
                module.lambda_C.copy_(torch.ones_like(module.lambda_C).to(args.device))
            elif isinstance(module.lambda_C, torch.Tensor):
                 module.lambda_C = torch.ones_like(module.lambda_C).to(args.device)

        # Handle separate parameters for keys if enabled
        if hasattr(module, 'args') and module.args.sep_params == 1:
            if hasattr(module, 'lambda1_k') and module.lambda1_k is not None:
                # Add specific initialization for lambda1_k here if needed
                pass 
            if hasattr(module, 'lambda_Omega_k') and module.lambda_Omega_k is not None:
                module.lambda_Omega_k.copy_(sigma_process * torch.ones_like(module.lambda_Omega_k).to(args.device))
            if hasattr(module, 'lambda_Gamma_k') and module.lambda_Gamma_k is not None:
                module.lambda_Gamma_k.copy_(sigma_measure * torch.ones_like(module.lambda_Gamma_k).to(args.device))
