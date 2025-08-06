import numpy as np
import torch

# Construct various special matrices

######################################################
######################################################

def construct_2D_rotation_matrix(theta = np.pi/4):
    """
    Constructs a 2-dimensional rotation matrix
    """
    U1 = torch.tensor(([np.cos(theta),-np.sin(theta)],[0,0]))
    U2 = torch.tensor(([0,0],[np.sin(theta),np.cos(theta)]))
    U = torch.stack((U1,U2))
    
    return U

######################################################
######################################################

def construct_2D_unitary_matrix(rho, sigma, xi, zeta):
    """
    Constructs a 2-dimensional unitary matrix, using four scalar inputs
    """
    Z = torch.zeros(2,2)
    U1 = torch.stack((torch.tensor(([np.cos(rho),-np.sin(rho)],[np.sin(rho),np.cos(rho)])),Z))
    U2 = torch.stack((torch.tensor(([np.cos(sigma),np.sin(sigma)],[-np.sin(sigma),np.cos(sigma)])),Z))
    Um = torch.stack((torch.tensor(([np.cos(xi), 0],[0,np.cos(zeta)])), torch.tensor(([np.sin(xi), 0],[0,np.sin(zeta)]))))

    U = complex_matmul(U1,complex_matmul(Um,U2))
    
    return U

######################################################
######################################################

def construct_special_2D_unitary(alpha, beta):
    """
    Constructs a 2-dimensional unitary matrix U where |U_ij|^2 = 1/2,
    parameterized by two angles theta and phi.
    This ensures that U Lambda U^* is real, if Lambda has complex conjugate eigenvals.
    """

    real_part = torch.tensor([[np.cos(alpha), np.cos(beta)], [np.cos(alpha - np.pi/2), np.cos(beta + np.pi/2)]]/np.sqrt(2))
    imag_part = torch.tensor([[np.sin(alpha), np.sin(beta)], [np.sin(alpha- np.pi/2), np.sin(beta + np.pi/2)]]/np.sqrt(2))
    U = torch.stack((real_part, imag_part), dim=0)
#     U = U*(torch.abs(U)>1E-5)
    
    return U

######################################################
######################################################

def construct_diag_matrix(a, b):
    """
    Construct diagonal matrix of complex-conjugate eigenvals with negative real part
    """
    Lambda1 = torch.tensor(-np.abs(a)*np.eye(2,2))
    Lambda2 = torch.tensor(([b, 0],[0, -b]))
    Lambda = torch.stack((Lambda1,Lambda2))
    
    return Lambda

######################################################
######################################################

# def construct_2D_normal_matrix(alpha_r, alpha_i, beta_r, beta_i, phi):
#     """
#     Construct 2D normal matrix
#     """
#     real = alpha_r*torch.eye(2,2) + torch.tensor(([0, beta_r*np.cos(phi) - beta_i*np.sin(phi)],[beta_r, 0]))
#     imag = alpha_i*torch.eye(2,2) + torch.tensor(([0, beta_r*np.sin(phi) + beta_i*np.cos(phi)],[beta_i, 0]))
#     N = torch.stack((real,imag))
    
#     return N

######################################################
######################################################

def construct_2D_normal_matrix(alpha_r, alpha_i, beta_r, beta_i, phi):
    """
    Construct 2D normal matrix
    """
    N = torch.tensor(([a, b],[-b,a]))
    
    return N

######################################################
######################################################

def orthogonal_matrix_2D(theta):
    """
    Construct a 2D orthogonal matrix.
    """
    c, s = np.cos(theta), np.sin(theta)
    Q= torch.tensor([[c, -s],
                           [s,  c]])
    
    if np.random.uniform(low=0.0, high=1.0) > 0.5:
        Q[:, 0] *= -1  # flip first column to get determinant -1

    Z = torch.zeros(2,2)
    M = torch.stack((Q, Z))
    
    return M