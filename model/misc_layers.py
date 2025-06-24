import numpy as np
import torch
import torch.nn as nn

from utils import batched_complex_matmul

##########################################################################################
##########################################################################################

class HadamardLayer(nn.Module):
    """
    Element-wise multiply every input with a set of learnable parameters.
    """
    
    def __init__(self, size_in, size_out, use_bias=False):
        super().__init__()
        self.use_bias = use_bias

        if self.use_bias:
            size_in += 1  # Add extra input for bias term

        self.size_in, self.size_out = size_in, size_out

        self.weight = nn.Parameter(torch.Tensor(size_out, size_in))
        torch.nn.init.kaiming_uniform_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        if self.use_bias:
            bias_term = torch.ones((*x.shape[:-1], 1), device=x.device)
            x = torch.cat((x, bias_term), dim=-1)

        return x * self.weight
    
##########################################################################################
##########################################################################################

class TemporalNorm(nn.Module):
    """
    Normalizes the input tensor over the time axis for each feature dimension.
    """
    
    def __init__(self, eps=1e-5):
        super(TemporalNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        # Mean and standard deviation over time (axis 1) for each feature (axis 2)
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)

        normalized_x = (x - mean) / (std + self.eps) # Normalize the input
        
#         return normalized_x
        return (torch.min(std) + self.eps) * normalized_x # Scale back up to same size as input

##########################################################################################
##########################################################################################

class TemporalWhiteningLayer(nn.Module):
    """
    Performs temporal whitening of complex-valued input by rotating to eigenbasis, centering,
    normalizing by the standard deviation of the magnitude in the eigenbasis, then rotating back.
    """
    
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

        # Complex-valued learnable matrices (real and imaginary parts)
        self.R1 = nn.Parameter(init_complex_matrix(args.m)).to(args.device)
        self.R2 = nn.Parameter(init_complex_matrix(args.m)).to(args.device)

    def forward(self, x):

        z = torch.zeros_like(x)
        xc = torch.stack((x,z),axis=1) # Construct complex variable

        x_rot = batched_complex_matmul(self.R1, xc) # Rotate to eigenbasis

        x_rot_centered = x_rot - x_rot.mean(dim=0, keepdim=True) # Subtract mean

        real, imag = x_rot_centered[:, 0], x_rot_centered[:, 1] # Get real and imaginary parts
        mag = torch.sqrt(real**2 + imag**2) # Magnitude
        mag_std = mag.std(dim=0, keepdim=True) # Std of magnitude

        x_norm = x_rot_centered/mag_std # Normalize

        x_out = batched_complex_matmul(self.R2, x_norm)[:,0,:,:] # Rotate back to original basis and take real part

        return x_out