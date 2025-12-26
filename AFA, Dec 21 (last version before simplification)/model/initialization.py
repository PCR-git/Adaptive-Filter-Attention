import numpy as np
import torch
import torch.nn as nn

from utils import complex_matmul

##########################################################################################
##########################################################################################
        
def initialize_linear_layers(m):
    if isinstance(m, nn.Linear):
        # Apply Xavier Uniform to the weight matrix
        nn.init.xavier_uniform_(m.weight)
        # Initialize bias to zero (standard practice)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

##########################################################################################
##########################################################################################

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
  
def init_complexlinear(linear_layer, weight_tensor, layer_type):
    """
    Initializes a Complex Linear Layer from complex weight (and optional bias) tensors.

    weight_tensor: shape (2, 1, d_in, d_out)
    bias_tensor: shape (2, 1, d_out)
    """
    real_w = weight_tensor[0, 0].T  # (d_out, d_in)
    imag_w = weight_tensor[1, 0].T
    if layer_type == 'in':
        W = torch.cat((real_w,imag_w),axis=0)
    else:
        W = torch.cat((real_w,imag_w),axis=1)

    with torch.no_grad():
        linear_layer.weight.copy_(W)
        
        if linear_layer.bias is not None:
            nn.init.constant_(linear_layer.bias, 0.0)
            
##########################################################################################
##########################################################################################

def set_complex_weight(layer,Pu,Pd,R1,R1i,mat):
    S_ud = complex_matmul(Pu,complex_matmul(mat,Pd))
    Au = complex_matmul(R1, S_ud)
    Ad = complex_matmul(Au, R1i).squeeze()
    
    layer.real.weight = nn.Parameter(Ad[0])
    layer.imag.weight = nn.Parameter(Ad[1])
    
    if layer.real.bias is not None:
        layer.real.bias.data.zero_()
        layer.imag.bias.data.zero_() 

##########################################################################################
##########################################################################################

def initialize_to_correct_model(module, D1, S1, Si1, sigma_process, sigma_measure, args):
    """
    Initialize to correct model parameter values (for testing)
    """

    with torch.no_grad():

        module.W_q.weight[0:2,0:2].copy_(Si1[0])
        module.W_q.weight[128:130,0:2].copy_(Si1[1])
        module.W_k.weight[0:2,0:2].copy_(Si1[0])
        module.W_k.weight[128:130,0:2].copy_(Si1[1])
        module.W_v.weight[0:2,0:2].copy_(Si1[0])
        module.W_v.weight[128:130,0:2].copy_(Si1[1])
        module.W_o.weight[0:2,0:2].copy_(S1[0])
        module.W_o.weight[0:2,128:130].copy_(-S1[1])
        
    print('Model initialized to match true dynamics.')
