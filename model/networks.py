import numpy as np
import torch
import torch.nn as nn

from model import BatchedPrecisionAttentionBlock
from model import HadamardLayer, TemporalNorm, TemporalWhiteningLayer

##########################################################################################
##########################################################################################

class PrecisionNet_1layer(torch.nn.Module):
    """
    Neural network with a single precision attention block
    """

    # Initialize the network and specify input/output dimensions:
    def __init__(self, args):
        super(PrecisionNet_1layer, self).__init__()

        self.a1 = BatchedPrecisionAttentionBlock(args.head_dim, args)

     # Build the network:
    def forward(self, inputs, t_v):

        est, out, Q_ij, X_ij_hat_all, lambda_h = self.a1(inputs, inputs, inputs, t_v)

        return est, out, Q_ij, X_ij_hat_all, lambda_h
    
##########################################################################################
##########################################################################################

class PrecisionNet(torch.nn.Module):
    """
    Neural network, alternating precision attention and fully-connected blocks
    """

    # Initialize the network and specify input/output dimensions:
    def __init__(self, args):
        super(PrecisionNet, self).__init__()

        self.fc1 = nn.Linear(args.embed_dim, args.embed_dim)
        self.fc2 = nn.Linear(args.embed_dim, args.embed_dim)

        # Xavier initialization
        init.xavier_normal_(self.fc1.weight)
        init.zeros_(self.fc1.bias)
        init.xavier_normal_(self.fc2.weight)
        init.zeros_(self.fc2.bias)

        self.a1 = BatchedPrecisionAttentionBlock(args.head_dim, args)
        self.a2 = BatchedPrecisionAttentionBlock(args.head_dim, args)

        # self.ReLU = nn.ReLU()
        self.GELU = nn.GELU()

     # Build the network:
    def forward(self, inputs, t_v):

        x1 = self.fc1(inputs.squeeze(-1)).unsqueeze(-1)
        x2 = self.GELU(x1) + x1
        _, x3, _, _ = self.a1(x2, x2, x2, t_v)
        x4 = self.fc2(x3.squeeze(-1)).unsqueeze(-1)
        x5 = self.GELU(x4) + x4
        est, out, Q_ij, X_ij_hat_all, lambda_h = self.a2(x5, x5, x5, t_v)

        return est, out, Q_ij, X_ij_hat_all, lambda_h
    
    