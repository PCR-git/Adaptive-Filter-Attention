import numpy as np
import torch
import torch.nn as nn

from utils import complex_matmul

##########################################################################################
##########################################################################################

# class Batched_Complex_MSE_Loss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, out, target):
#         out_complex = torch.complex(out[:,0], out[:,1])
#         target_complex = torch.complex(target[:,0], target[:,1])
#         return torch.mean(torch.abs(out_complex - target_complex)**2)

##########################################################################################
##########################################################################################

class Complex_MSE_Loss(nn.Module):
    """
    Mean Square Error Loss for complex-valued arrays
    """
    def __init__(self):
        super().__init__()

    def forward(self, out, target):
        real_diff = out[0] - target[0] # Error of real part
        imag_diff = out[1] - target[1] # Error of imaginary part
        squared_error = real_diff**2 + imag_diff**2
        mean_SE = torch.mean(squared_error) # Sum square error of real and imaginary parts and take mean
#         max_SE = torch.max(squared_error) # Sum square error of real and imaginary parts and take mean
        return mean_SE
#         return mean_SE + max_SE

##########################################################################################
##########################################################################################

class Batched_Complex_MSE_Loss(nn.Module):
    """
    Batched Mean Square Error Loss for complex-valued arrays
    """
    def __init__(self):
        super().__init__()

    def forward(self, out, target):
        real_diff = out[:,0] - target[:,0] # Error of real part
        imag_diff = out[:,1] - target[:,1] # Error of imaginary part
        return torch.mean(real_diff**2 + imag_diff**2) # Sum square error of real and imaginary parts and take mean
    
##########################################################################################
##########################################################################################

# class Complex_Trace_Loss(nn.Module):
#     """
#     Penalize trace of the matrix to be near identity
#     """
    
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, M):
        
#         d_e = M.size()[-2]
#         real_trace = torch.trace(M[0].squeeze(0))
#         imag_trace = torch.trace(M[1].squeeze(0))
        
#         return ((real_trace - d_e)**2 + imag_trace**2)/d_e

##########################################################################################
##########################################################################################

def inverse_penalty(model, loss_p, args):
    """
    Penalty to keep the product of W_p and W_v near I
    """
    
    penalty = 0.0
    num_layers = 0
    for name, module in model.named_modules():
        if hasattr(module, 'W_p') and hasattr(module, 'W_v'):
            W_p = getattr(module, 'W_p')
            W_v = getattr(module, 'W_v')
#             prod = complex_matmul(W_p, W_v) # Complex matrix multiply
            prod = complex_matmul(W_v, W_p) # Reverse the order so that the multiplication is in the lower dimension
            layer_penalty = loss_p(prod, module.complex_identity) # Penalty for this layer
            penalty += layer_penalty
            num_layers += 1
    
    if num_layers > 0:
        penalty = penalty/num_layers # Divide by number of layers to get average
        
    return penalty
