import numpy as np
import torch
import torch.nn as nn

from matplotlib import pyplot as plt
import matplotlib.cm as cm
plt.rcParams['figure.figsize'] = [10, 10]
plt.rc('font', size=20)

##########################################################################################
##########################################################################################

def single_iter(model, optimizer, loss, loss_p, inputs, target, args, t_measure_full=None):
    """
    Single iteration of training
    """

    optimizer.zero_grad() # Zero out gradients

    with torch.autograd.set_detect_anomaly(True):
        
        out, output_dict = model(inputs, t_measure_full) # Forward pass of model
        est = output_dict['est_latent']
        attn_mat = output_dict['attn_mat']
#         x_hat = output_dict['x_hat']
        lambda_h = output_dict['epoch_lambdas']
        unnormalized_attention = output_dict['unnormalized_attention']

# #         loss_i = loss(out, target) # Compute loss
#         if args.complex_loss == 0:
#             loss_i = loss(out[:,0], target[:,0])
#         elif args.complex_loss == 1:
#             loss_i = loss(out[:,0], target[:,0]) + loss(out[:,1], target[:,1])
#         else:
#             print('Error, no loss')

        loss_i = loss(out, target)
    
#         print(out.size())
#         print(target.size())

        # Inverse loss (keep W_p W_v near I)
    #     inverse_penalty = inverse_penalty(model, loss_p, args) # Compute penalty
    
#         inverse_penalty = inverse_net_penalty(model, loss_p, args) # Compute penalty
#         loss_i = loss_i + args.inverse_penalty_weight * inverse_penalty # Add to loss

#         # --- Check gradients here ---
#         for name, param in model.named_parameters():
#             if param.grad is not None:
#                 max_abs_grad = param.grad.abs().max().item()
#                 print(f"  {name} Max Abs Grad: {max_abs_grad:.4e}") # Using scientific notation for clarity
#         # --- End gradient check ---
    
        # L1 Penalty on the eigenvals
#         L1_penalty = lambda_L1_penalty(model, args)
#         loss_i = loss_i + args.L1_penalty_weight * L1_penalty

        loss_i.backward() # Backprop
    
#         print(model.layers[0].lambda_real_v.grad)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
    #     nn.utils.clip_grad_norm_(params_list, 1) # Clip grads (optional)

    optimizer.step()
    
    return loss_i.detach().cpu().numpy(), attn_mat.detach().cpu().numpy(), lambda_h.squeeze().detach().cpu().numpy(), unnormalized_attention.detach().cpu().numpy() # Return loss, Attention mat, and eigenvals

##########################################################################################
##########################################################################################

def single_epoch(model, train_loader, optimizer, loss, loss_p, params_list, args):
    """
    Single epoch of training
    """
    
    epoch_losses = np.zeros(args.num_its)
    epoch_lambdas = np.zeros((args.num_its,2,args.d_v))

    # Iterate through training data
    for it, (train_data, _, _, _) in enumerate(train_loader):
#     for it, train_data in enumerate(train_loader):

        # Outputs are one index ahead of inputs
        inputs  = train_data[:, :-1].unsqueeze(1)
        outputs = train_data[:, 1:].unsqueeze(1)

        # Single iteration of training
        epoch_losses[it], attn_mat, epoch_lambdas[it,:,:], unnormalized_attention = single_iter(model, optimizer, loss, loss_p, inputs, outputs, args)
        
    return epoch_losses, attn_mat, epoch_lambdas, unnormalized_attention

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

##########################################################################################
##########################################################################################

def single_iter_attn(model, optimizer, loss, inputs, target, args):
    """
    Single iteration of training
    """

    optimizer.zero_grad() # Zero out gradients

    with torch.autograd.set_detect_anomaly(True):
        
        out, _ = model(inputs)
        
        loss_i = loss(out, target) # Compute loss

        loss_i.backward() # Backprop

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip gradient

    optimizer.step()

    return loss_i.detach().cpu().numpy()

##########################################################################################
##########################################################################################

def single_epoch_attn(model, train_loader, optimizer, loss, params_list, args):
    """
    Single epoch of training
    """
    
    epoch_losses = np.zeros(args.num_its)
    epoch_lambdas = np.zeros((args.num_its,2,args.d_v))

    # Iterate through training data
    for it, (train_data, X_true, X_measure, t_measure_full) in enumerate(train_loader):

        # Outputs are one index ahead of inputs
        inputs  = train_data[:, :-1]
        outputs = train_data[:, 1:]

        # Single iteration of training
        epoch_losses[it] = single_iter_attn(model, optimizer, loss, inputs, outputs, args)
        
    return epoch_losses