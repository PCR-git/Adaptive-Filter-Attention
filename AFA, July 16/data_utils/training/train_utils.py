import numpy as np
import torch
import torch.nn as nn

from matplotlib import pyplot as plt
import matplotlib.cm as cm
plt.rcParams['figure.figsize'] = [10, 10]
plt.rc('font', size=20)

from model import inverse_penalty, inverse_net_penalty, lambda_L1_penalty
from model import apply_net_weight_masks, FullPrecisionAttentionBlock_Nlayer, FullPrecisionAttentionBlockShared

##########################################################################################
##########################################################################################

def single_iter(model, optimizer, loss, loss_p, inputs, target, t_v, args):
    """
    Single iteration of training
    """

    optimizer.zero_grad() # Zero out gradients

    with torch.autograd.set_detect_anomaly(True):
                
        if args.model_type == 'RealInputs':
            inputs = inputs[:,0].unsqueeze(1)
        elif args.model_type == 'ComplexInputs':
            pass
        else:
            print('Error. Specify model type.')
        
        est, out, Q_ij, X_ij_hat_all, lambda_h = model(inputs, t_v) # Forward pass of model

#         loss_i = loss(out, target) # Compute loss
        if args.complex_loss == 0:
            loss_i = loss(out[:,0], target[:,0])
        elif args.complex_loss == 1:
            loss_i = loss(out[:,0], target[:,0]) + loss(out[:,1], target[:,1])
        else:
            print('Error, no loss')

        # Inverse loss (keep W_p W_v near I)
    #     inverse_penalty = inverse_penalty(model, loss_p, args) # Compute penalty
    
        inverse_penalty = inverse_net_penalty(model, loss_p, args) # Compute penalty
        loss_i = loss_i + args.inverse_penalty_weight * inverse_penalty # Add to loss

#         # --- Check gradients here ---
#         for name, param in model.named_parameters():
#             if param.grad is not None:
#                 max_abs_grad = param.grad.abs().max().item()
#                 print(f"  {name} Max Abs Grad: {max_abs_grad:.4e}") # Using scientific notation for clarity
#         # --- End gradient check ---
    
        # L1 Penalty on the eigenvals
        L1_penalty = lambda_L1_penalty(model, args)
        loss_i = loss_i + args.L1_penalty_weight * L1_penalty

        loss_i.backward() # Backprop

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
    #     nn.utils.clip_grad_norm_(params_list, 1) # Clip grads (optional)

    optimizer.step()
    
    # Apply weight masks to prune model (for testing)
    for module in model.modules():
        if isinstance(module, FullPrecisionAttentionBlock_Nlayer) or isinstance(module, FullPrecisionAttentionBlockShared):
            apply_net_weight_masks(module)
    
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
        
        if outputs.size()[1] == 1:
            outputs = outputs.unsqueeze(1)

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

##########################################################################################
##########################################################################################

def single_iter_attn(model, optimizer, loss, inputs, target, args):
    """
    Single iteration of training
    """

    optimizer.zero_grad() # Zero out gradients

    with torch.autograd.set_detect_anomaly(True):
        
#         inputs_reshape = inputs.reshape(args.batch_size,args.seq_len,args.d_e*2)
#         out, _ = model(inputs_reshape)
#         out_reshape = out.reshape(args.batch_size, 2, args.seq_len, args.d_e)
#         loss_i = loss(out_reshape, target) # Compute loss
        
        if args.model_type == 'RealInputs':
            inputs = inputs[:,0]
        elif args.model_type == 'ComplexInputs':
            pass
        
        out, _ = model(inputs)
        
        if args.model_type == 'RealInputs':
            loss_i = loss(out.unsqueeze(1), target[:,0].unsqueeze(1)) # Compute loss
        elif args.model_type == 'ComplexInputs':
#              loss_i = loss(out, target) # Compute loss
            loss_i = loss(out.unsqueeze(-1), target.unsqueeze(-1))

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
        inputs  = train_data[:, :, :-1]
        outputs = train_data[:, :, 1:]

        # Single iteration of training
        epoch_losses[it] = single_iter_attn(model, optimizer, loss, inputs, outputs, args)
        
    return epoch_losses