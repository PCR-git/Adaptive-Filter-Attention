import numpy as np
import torch

###############################################
###############################################

def complex_exp(D):
    """
    Computes exponential for complex valued vectors
    """
    return torch.exp(D[0]) * torch.stack((torch.cos(D[1]), torch.sin(D[1])))

###############################################
###############################################

def complex_hadamard(A, B):
    """
    Complex Hadamard product for complex tensors.
    """
    # A, B: (S, 2, N, M)

    A_real, A_imag = A[0], A[1]
    B_real, B_imag = B[0], B[1]

    real_part = A_real * B_real - A_imag * B_imag
    imag_part = A_real * B_imag + A_imag * B_real
    
    return torch.stack([real_part, imag_part], dim=0)

###############################################
###############################################

def complex_matmul(A, B):
    """
    Complex matrix multiplication for tensors representing complex numbers.
    """
    # A, B: (S, 2, N, M)
    
    A_real, A_imag = A[0], A[1]
    B_real, B_imag = B[0], B[1]
    
    real_part = torch.matmul(A_real, B_real) - torch.matmul(A_imag, B_imag)
    imag_part = torch.matmul(A_real, B_imag) + torch.matmul(A_imag, B_real)
    
    return torch.stack([real_part, imag_part], dim=0)

###############################################
###############################################

# Complex conjugate transpose

def complex_conj_transpose(X):
    X_re = X[0]
    X_im = X[1]
    return torch.stack((X_re, -X_im))

def batched_complex_conj_transpose(X):
    """
    Batched complex conjugate transpose
    """
    return torch.stack([X[:,0].transpose(-2,-1), -X[:,1].transpose(-2,-1)], dim=1)

###############################################
###############################################

def batched_complex_hadamard(A, B):
    """
    Batched Hadamard product for complex tensors.
    """
    # A: (S, 2, N, M)
    # B: (B, S, 2, N, M)
    A_real, A_imag = A[0], A[1] # Get real and imaginary parts of first matrix
    B_real, B_imag = B[:, 0], B[:, 1] # Get real and imaginary parts of second matrix

    real_part = A_real * B_real - A_imag * B_imag # Compute real part
    imag_part = A_real * B_imag + A_imag * B_real # Compute imaginary part
    
    return torch.stack([real_part, imag_part], dim=1)

###############################################
###############################################

def batched_complex_matmul(A, B):
    """
    Batched complex matrix multiplication for tensors representing complex numbers.
    """
    # A: (S, 2, N, M)
    # B: (B, S, 2, N, M)
    
    A_real, A_imag = A[0], A[1] # Get real and imaginary parts of first matrix
    B_real, B_imag = B[:, 0], B[:, 1] # Get real and imaginary parts of second matrix

    real_part = torch.matmul(A_real, B_real) - torch.matmul(A_imag, B_imag) # Compute real part
    imag_part = torch.matmul(A_real, B_imag) + torch.matmul(A_imag, B_real) # Compute imaginary part

    return torch.stack([real_part, imag_part], dim=1)

###############################################
###############################################

# These functions are needed to run the variable timestep functionality

def batched_complex_exp(D):
    """
    Batched exponential for complex valued vectors
    """
    return torch.exp(D[:,0]).unsqueeze(1)*torch.stack((torch.cos(D[:,1]), torch.sin(D[:,1])),axis=1)

###############################################
###############################################

def batched_complex_hadamard_full(A, B):
    """
    Batched Hadamard product for complex tensors.
    """
    # A: (B, S, 2, N, M)
    # B: (B, S, 2, N, M)
    A_real, A_imag = A[:, 0], A[:, 1] # Get real and imaginary parts of first matrix
    B_real, B_imag = B[:, 0], B[:, 1] # Get real and imaginary parts of second matrix

    real_part = A_real * B_real - A_imag * B_imag # Compute real part
    imag_part = A_real * B_imag + A_imag * B_real # Compute imaginary part
    
    return torch.stack([real_part, imag_part], dim=1)

###############################################
###############################################

def batched_complex_matmul_full(A, B):
    """
    Batched complex matrix multiplication for tensors representing complex numbers.
    """
    # A: (B, S, 2, N, M)
    # B: (B, S, 2, N, M)
    
    A_real, A_imag = A[:, 0], A[:, 1] # Get real and imaginary parts of first matrix
    B_real, B_imag = B[:, 0], B[:, 1] # Get real and imaginary parts of second matrix

    real_part = torch.matmul(A_real, B_real) - torch.matmul(A_imag, B_imag) # Compute real part
    imag_part = torch.matmul(A_real, B_imag) + torch.matmul(A_imag, B_real) # Compute imaginary part

    return torch.stack([real_part, imag_part], dim=1)

###############################################
###############################################

# def batched_complex_matmul_multihead(A, B):
#     """
#     General batched complex matrix multiplication with head dimension.

#     A: [B, 2, M1, M2, H]
#     B: [B, 2, M2, M3, H]
#     Returns:
#         [B, 2, M1, M3, H]
#     """
#     A_re, A_im = A[:, 0], A[:, 1]  # [B, M1, M2, H]
#     B_re, B_im = B[:, 0], B[:, 1]  # [B, M2, M3, H]

#     real = torch.einsum('b i j h, b j k h -> b i k h', A_re, B_re) \
#          - torch.einsum('b i j h, b j k h -> b i k h', A_im, B_im)
    
#     imag = torch.einsum('b i j h, b j k h -> b i k h', A_re, B_im) \
#          + torch.einsum('b i j h, b j k h -> b i k h', A_im, B_re)

#     return torch.stack([real, imag], dim=1)  # [B, 2, M1, M3, H]
