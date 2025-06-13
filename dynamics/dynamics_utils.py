import numpy as np
import torch

###############################################################
###############################################################

def construct_mapping(X, m, d_e, args):
    """
    Constructs random matrices to map from dimension of size m to dimension of size d_e
    """

    Pu = torch.randn(2, d_e, m).to(args.device)
    Pd = torch.zeros(2, m, d_e).to(args.device)

    Pu_complex = Pu[0] + 1j * Pu[1]
    Pd_complex = torch.linalg.pinv(Pu_complex)
    Pd[0] = Pd_complex.real
    Pd[1] = Pd_complex.imag

    Pur = Pu[0]
    Pui = Pu[1]

    Pu = Pu.unsqueeze(1)
    Pd = Pd.unsqueeze(1)

    return Pu, Pd