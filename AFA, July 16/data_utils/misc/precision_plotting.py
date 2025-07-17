import numpy as np
import torch
from utils import complex_hadamard

##########################################################################################
##########################################################################################

def f_pij(lambder, lambda_Omega, lambda_Omega0, nu, t_v, eps = 0.01):
    """
    Compute the ijth entry of the precision matrix P
    """
    e = torch.exp(-2*lambder*t_v)
    vij = nu*torch.abs((lambda_Omega/(2*lambder))*(1-e)) + torch.abs(lambda_Omega0) + eps

    return 1/vij

##########################################################################################
##########################################################################################

def f_pij_complex(mat_exp, lambder, lambda_Omega, lambda_Omega0, nu, eps = 0.01):
    """
    Compute the ijth entry of the precision matrix P, when lambder is complex valued
    """
    lambder_ratio = lambda_Omega/(2*lambder[0])
    mat_exp2 = complex_hadamard(mat_exp, mat_exp)
    vij = nu*torch.abs(lambder_ratio*(1-mat_exp2)) + torch.abs(lambda_Omega0[0]) + eps

    pij = torch.zeros_like(vij)
    r_pij = vij[0]**2 + vij[1]**2
    pij[0] = vij[0]/r_pij
    pij[1] = -vij[1]/r_pij

    return pij

##########################################################################################
##########################################################################################

# Visualize an example of the precision (inverse variance) as a function of time

def f_pij_scalar(lambder, lambda_Omega, lambda_Omega0, nu, tj, ti):
  """
  Function to help visualize an example of the precision (inverse variance) as a function of time
  """
  plambder = lambda_Omega/(2*lambder)
  vij = nu*torch.abs(plambder*(1-torch.exp(-2*lambder*(tj-ti)))) + lambda_Omega0
  return 1/vij



