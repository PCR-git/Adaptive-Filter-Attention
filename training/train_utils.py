import numpy as np
import torch
import torch.nn as nn

from matplotlib import pyplot as plt
import matplotlib.cm as cm
plt.rcParams['figure.figsize'] = [10, 10]
plt.rc('font', size=20)

from model import inverse_penalty

##########################################################################################
##########################################################################################

def single_iter(model, optimizer, loss, loss_p, inputs, target, t_v, args):
    """
    Single iteration of training
    """

    optimizer.zero_grad() # Zero out gradients

    est, out, Q_ij, X_ij_hat_all, lambda_h = model(inputs, t_v) # Forward pass of model

    loss_i = loss(out, target) # Compute loss
    
    # Inverse loss (keep W_p W_v near I)
    penalty = inverse_penalty(model, loss_p, args) # Compute penalty
    loss_i = loss_i + args.penalty_weight * penalty # Add to loss
    
    loss_i.backward() # Backprop

#     nn.utils.clip_grad_norm_(params_list, 1) # Clip grads (optional)

    optimizer.step()

    return loss_i.detach().cpu().numpy(), Q_ij.detach().cpu().numpy(), lambda_h.squeeze().detach().cpu().numpy() # Return loss, Attention mat, and eigenvals

##########################################################################################
##########################################################################################

def single_epoch(model, train_loader, optimizer, loss, loss_p, params_list, args):
    """
    Single epoch of training
    """
    
    epoch_losses = np.zeros(args.num_its)
    epoch_lambdas = np.zeros((args.num_its,2,args.d_v))

    # Iterate through training data
    for it, (train_data, X_true, X_measure, t_measure_full) in enumerate(train_loader):

        # Outputs are one index ahead of inputs
        inputs  = train_data[:, :, :-1]
        outputs = train_data[:, :, 1:]

        # Single iteration of training
        epoch_losses[it], Q_ij, epoch_lambdas[it,:,:] = single_iter(model, optimizer, loss, loss_p, inputs, outputs, t_measure_full, args)

    return epoch_losses, Q_ij, epoch_lambdas

##########################################################################################
##########################################################################################

def hook_fn(grad):
    """
    Hook function to get gradients and plot
    """
    
#     print(grad)
    grad_mean = grad.mean(dim=[0, 1, 2])
#     print(f"Mean gradient per component: {grad_mean}")
    plt.scatter(np.arange(args.embed_dim),grad_mean.detach().cpu().numpy())
    plt.show()
#     print(f"Max gradient per component: {grad.abs().max(dim=[0, 1, 2])}")
    return grad
