import torch
import torch.nn as nn
import torch.nn.functional as F

from model import init_complexlinear, init_complex_matrix
from model import ComplexLinearLayer

##########################################################################################
##########################################################################################

class ModReLU(nn.Module):
    def forward(self, x):
        # x is (batch_size, 2, seq_len, dim) where x[:,0] is real, x[:,1] is imag
        magnitude = torch.sqrt(x[:,0]**2 + x[:,1]**2 + 1e-8) # Add epsilon for stability
        relu_magnitude = F.relu(magnitude)

        # Avoid division by zero if magnitude is zero
        ratio = torch.where(magnitude > 1e-8, relu_magnitude / magnitude, torch.zeros_like(magnitude).to(magnitude.device))

        real_out = ratio * x[:,0]
        imag_out = ratio * x[:,1]
        
        return torch.stack((real_out, imag_out), dim=1)
    
##########################################################################################
##########################################################################################

class ModReLU_Rot(nn.Module):
    """
    ModReLU Activation with complex linear layers before and after
    """

    # Initialize the network and specify input/output dimensions:
    def __init__(self, input_dim, hidden_dim, args):
        super(ModReLU_Rot, self).__init__()
        
        self.l1 = ComplexLinearLayer(input_dim, input_dim)
        self.l2 = ComplexLinearLayer(input_dim, input_dim)
        
        W1, b1 = init_complex_matrix(input_dim, input_dim, bias=True)
        W2, b2 = init_complex_matrix(input_dim, input_dim, bias=True)
        init_complexlinear(self.l1, W1, b1)
        init_complexlinear(self.l2, W2, b2)
        
        self.ReLU = nn.ReLU()
        self.ModReLU = ModReLU()
        
     # Build the network:
    def forward(self, x):
        x_rot = self.l1(x) # Rotate to new basis
        
        # x is (batch_size, 2, seq_len, dim) where x[:,0] is real, x[:,1] is imag
        magnitude = torch.sqrt(x_rot[:,0]**2 + x_rot[:,1]**2 + 1e-8) # Add epsilon for stability
        relu_magnitude = F.relu(magnitude)

        # Avoid division by zero if magnitude is zero
        ratio = torch.where(magnitude > 1e-8, relu_magnitude / magnitude, torch.zeros_like(magnitude).to(magnitude.device))

        real_out = ratio * x_rot[:,0]
        imag_out = ratio * x_rot[:,1]
        
        x_norm = torch.stack((real_out, imag_out), dim=1)
        
        x_out = self.l2(x_norm) # Rotate back
        
        return x_out

    