import numpy as np
import torch
import torch.nn as nn

##########################################################################################
##########################################################################################

def init_complex_matrix(d_1, d_2):
    """
    Isotropic initialization of a complex-valued matrix
    """

    scale = np.sqrt(2/(d_1 + d_2))
    mag = scale*torch.randn(d_1, d_2)
    phase = 2 * np.pi * torch.rand(d_1, d_2) # Phase
    real = mag * torch.cos(phase) # Real part
    imag = mag * torch.sin(phase) # Imaginary part
    W = torch.stack([real, imag]).unsqueeze(1)

    return W

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
        lambda_h = torch.randn(2, args.d_v).to(args.device)/10
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

        print('Model initialized.')
