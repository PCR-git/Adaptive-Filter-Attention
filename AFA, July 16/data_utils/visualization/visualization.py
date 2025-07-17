import numpy as np
import torch
import torch.nn as nn

from matplotlib import pyplot as plt
import matplotlib.cm as cm
plt.rcParams['figure.figsize'] = [10, 10]
plt.rc('font', size=20)

from utils import complex_matmul, batched_complex_matmul
from model import compute_lambda_h, get_complex_weights
from model import ComplexLinearLayer, FullPrecisionAttentionBlockShared, FullPrecisionAttentionBlock_Nlayer, AFATransformerBlock, AFA_Nlayer, AFATransformerNetwork

##########################################################################################
##########################################################################################

def plot_trajectory(X_true,X_measure,est_h,Pd):
    """
    Plot actual, measured, and estimated trajectories with labels for a legend.
    """

    # Actual trajectory
    X_true_plt = X_true.squeeze().detach().cpu().numpy()
    plt.plot(X_true_plt.T[0],X_true_plt.T[1],'black', label='Ground Truth') # Added label

    # Noisy trajectory
    traj = X_measure.detach().cpu().squeeze().numpy()
    plt.plot(traj.T[0], traj.T[1], 'b--', label='Measured') # Added label

    # Predicted trajectory
    est = batched_complex_matmul(Pd,est_h) # Map back to lower dim
    est1_plt = est.squeeze(0)[0].detach().cpu().squeeze().numpy()
    plt.plot(est1_plt.T[0], est1_plt.T[1], 'r--', label='Predicted') # Added label

    plt.grid()

##########################################################################################
##########################################################################################

# def compute_state_transition_matrix_v1(module,Pu,Pd,R1,R1i):
#     with torch.no_grad():
#         lambda_h_model = compute_lambda_h(module.lambda1,args)
#         ############### DELETE #####################
#         mask = torch.zeros(2,1,args.d_e,1).to(args.device)
#         mask[:,:,0:args.m,:] = 1
#         lambda_h_model = lambda_h_model*mask
#         ################################# ############

#         DD = torch.stack((torch.diag(lambda_h_model[0].squeeze()),torch.diag(lambda_h_model[1].squeeze()))).unsqueeze(1)
#         y = complex_matmul(module.W_p, complex_matmul(DD, module.W_v))
        
# #         z = y
#         z = complex_matmul(R1i,complex_matmul(y,R1))
#         A = complex_matmul(Pd,complex_matmul(z,Pu))
#     return A

def compute_state_transition_matrix(model,module,Pu,Pd,R1,R1i,args):
    """
    Compute elements of state transition matrix A
    """
    
    dt = args.dt*args.n # Time step

    with torch.no_grad():
#         W_v = module.W_v
#         W_p = module.W_p
        if hasattr(module, 'W_v'):
            W_v = get_complex_weights(module, 'W_v')
            W_p = get_complex_weights(module, 'W_p')
        else:
            W_v = get_complex_weights(model, 'W_v')
            W_p = get_complex_weights(model, 'W_p')
        
        if args.weight_mask == 1:
            W_v = W_v*module.weight_mask_v
            W_p = W_p*module.weight_mask_v
            module.lambda1 *= module.eigen_mask
            
        # Get eigenvalues
        lambda_h_model = compute_lambda_h(module.lambda1,args)
#         lambda_h_model = compute_lambda_h_v2(module.lambda1)

        DD = torch.stack((torch.diag(lambda_h_model[0].squeeze()),torch.diag(lambda_h_model[1].squeeze()))).unsqueeze(1) # Diagonal matrix
        I = torch.stack((torch.eye(args.d_v, args.d_v), torch.zeros((args.d_v, args.d_v)))).unsqueeze(1).to(args.device) # Identity
        I_out = torch.stack((torch.eye(args.d_e, args.d_e), torch.zeros((args.d_e, args.d_e)))).unsqueeze(1).to(args.device) # Identity
        P = DD*dt + I # State transition (diagonalized)
#         WpPWv = complex_matmul(module.W_p, complex_matmul(P, module.W_v)) - I
#         WpPWv = complex_matmul(W_p, complex_matmul(P, (module.alpha + module.beta)*W_v)) - I # Multiply by W_v and W_p

        A_h = complex_matmul(W_p, complex_matmul(P, W_v)) - I_out # Multiply by W_v and W_p
#         WpPWv = complex_matmul(module.W_p, complex_matmul(P, module.W_v + module.W_r)) - I
        A_h_mask = complex_matmul(R1i, complex_matmul(A_h, R1))/dt # Estimated state transition matrix in higher dimension
        A_est = complex_matmul(Pd,complex_matmul(A_h_mask,Pu)) # Estimated A in lower dimension

    return A_h, A_est

##########################################################################################
##########################################################################################

def plot_state_transition_matrix(A,marker,size=32,color=None):
    """
    Plot entries of the state transition matrix (real values in blue, imaginary in red)
    """
    
    if A.size()[0] == 1:
        A_real = A[0].flatten().detach().cpu().numpy()
        if color == None:
            plt.scatter(range(4),A_real, c='b', marker=marker,s=size)
        else:
            plt.scatter(range(4),A_real, c=color, marker=marker,s=size)
    else:
        A_real = A[0].flatten().detach().cpu().numpy()
        A_imag = A[1].flatten().detach().cpu().numpy()

        if color == None:
            plt.scatter(range(4),A_real, c='b', marker=marker,s=size)
            plt.scatter(range(4),A_imag, c='r', marker=marker,s=size)
        else:
            plt.scatter(range(4),A_real, c=color, marker=marker,s=size)
            plt.scatter(range(4),A_imag, c=color, marker=marker,s=size)

# DD = torch.stack((torch.diag(lambda_h[0].squeeze()),torch.diag(lambda_h[1].squeeze()))).unsqueeze(1)
# A_actual = complex_matmul(Pd,complex_matmul(DD,Pu))

# A_model = compute_state_transition_matrix(model.a1,Pu,Pd,R1,R1i)

# print('A_actual = ')
# print(A_actual)
# print('A_model = ')
# print(A_model)

# plot_state_transition_matrix(A_actual, marker='o')
# plot_state_transition_matrix(A_model, marker='x',size=80)
# plt.grid()
# plt.show()

##########################################################################################
##########################################################################################

def plot_eigenvals(A,eigenvals=None):
    """
    Compute and plot eigenvalues of A
    """
    
    if eigenvals == None:
        A_np = A.detach().cpu().numpy()
        complex_matrix = A_np[0] + 1j * A_np[1]  # shape (2, 2)
        eigenvals = np.linalg.eigvals(complex_matrix).squeeze()
        eig_r = eigenvals.real
        eig_i = eigenvals.imag
    else:
        eigenvals = eigenvals.squeeze().detach().cpu().numpy()
        eig_r = eigenvals[0]
        eig_i = eigenvals[1]

    eig_abs = np.flip(np.sort(np.abs(eig_r.T)))
    plt.plot(eig_abs)

    eig_abs = np.flip(np.sort(np.abs(eig_i.T)))
    plt.plot(eig_abs)
    
    plt.grid()
    plt.show()

##########################################################################################
##########################################################################################

# def visualize_results(model, train_dataset, all_losses, mean_epoch_losses, log_mean_epoch_losses, all_lambdas, R1, R1i, Pu, Pd, A, epoch, args):
#     """
#     Visualize results during training
#     Plots the following:
#         Noisy and true trajectory, and prediction of model
#         State estimates and n_example data points
#         Attention matrix
#         Values of state transition matrix
#     """
    
#     folder = "C://Users//Pracioppo//Desktop//train_imgs//"
#     plt.axis('equal')
    
#     main_attention_block, last_inner_layer_of_main_attention_block = _get_visual_modules(model)

#     module = model.layers[0].layers[-1]

#     with torch.no_grad():
        
#         if hasattr(module, 'W_v'):
#              #         W_p = module.W_p*module.weight_mask
#     #         W_v = module.W_v*module.weight_mask
#     #         W_p = module.W_p
#     #         W_v = module.W_v
#             W_v = get_complex_weights(module, 'W_v')
#             W_p = get_complex_weights(module, 'W_p')
#         elif hasattr(model, 'W_v'):
#             W_v = get_complex_weights(model, 'W_v')
#             W_p = get_complex_weights(model, 'W_p')
#         else:
#             print('Error: W_v and W_p not found')

#         # print(module.lambda1)

#         # Get prediction for random choice of input
#         rand_idx = np.random.choice(args.num_samp)
# #         rand_idx = 0
#         train_data, X_true, X_measure, t_measure = train_dataset.__getitem__(rand_idx)
#         inputs = train_data[:, :-1].unsqueeze(0)
#         # outputs = train_data[:, :-1].unsqueeze(0)
#         if inputs.size()[1] == 1:
#             inputs = torch.stack((inputs, torch.zeros_like(inputs)),dim=1)
            
#         est, out, Q_ij, X_ij_hat_all, epoch_lambdas = model.forward(inputs, inputs, inputs, t_measure.unsqueeze(0))

#         est = est.unsqueeze(-1)
#         out = out.unsqueeze(-1)
#         X_ij_hat_all = X_ij_hat_all.unsqueeze(-1)
        
#         # Set plotting dims
#         x_max = torch.max(X_true[:,0]).detach().cpu().numpy()
#         x_min = torch.min(X_true[:,0]).detach().cpu().numpy()
#         y_max = torch.max(X_true[:,1]).detach().cpu().numpy()
#         y_min = torch.min(X_true[:,1]).detach().cpu().numpy()
#         margin = 2
# #         width = x_max - x_min
# #         height = y_max - y_min
# #         dim = 5*torch.ceil(max(width,height)/5).detach().cpu().numpy()
        
#         #########################################
        
#         # Plot trajectory
# #         fig, ax = plt.subplots(figsize=(dim, dim))
        
#         pred_map = batched_complex_matmul(R1i,out) # Reverse random mapping
#         plot_trajectory(X_true[args.n:].unsqueeze(0),X_measure[1:].unsqueeze(0),pred_map,Pd)
        
#         plt.xlim(x_min-margin, x_max+margin)
#         plt.ylim(y_min-margin, y_max+margin)
        
# #         plt.savefig(folder + 'trajecs//' + str(epoch) + '.png', bbox_inches='tight')
#         plt.title('Trajectory')
#         plt.legend()
#         plt.show()
    
#         #########################################
            
#         # Plot state estimates at n_example data points
# #         fig, ax = plt.subplots(figsize=(dim, dim))

#         plot_trajectory(X_true[:-args.n].unsqueeze(0),0*X_measure[1:].unsqueeze(0),0*pred_map,Pd) # Plot actual trajectory
#         x_hat = batched_complex_matmul(W_p, X_ij_hat_all*module.causal_mask)
#         Xo_h = batched_complex_matmul(R1i,x_hat) # Reverse random mapping
#         Xo = batched_complex_matmul(Pd,Xo_h).detach().cpu()[0,0].squeeze() # State estimates
#     #         Xo = batched_complex_matmul(Pd,x_hat).detach().cpu()[0,0].squeeze()
#         markers = ['o', 'v', 's', 'd', 'P']
#         colors = ['pink', 'red', 'black', 'yellow', 'blue']
#         mi = 0
#         for i in range(args.seq_len):
#             if np.mod(i,int(args.seq_len/args.n_example)) == 0:
#                 xi = Xo[i,i,:]
#                 x_est = Xo[i,0:i+1,:].numpy()
#                 plt.scatter(x_est.T[0],x_est.T[1], s=10, marker=markers[np.mod(mi,len(markers))], color=colors[np.mod(mi,len(colors))])
#                 plt.scatter(xi[0],xi[1], s=100, marker='x', color=colors[np.mod(mi,len(colors))])
#                 mi += 1

#         plt.xlim(x_min-margin, x_max+margin)
#         plt.ylim(y_min-margin, y_max+margin)

#         plt.title('State Estimates')
# #         plt.savefig(folder + 'ests//' + str(epoch) + '.png', bbox_inches='tight')
#         plt.show()

#          #########################################

#         # Plot attention matrix
#         Q_ij_avg = torch.mean(Q_ij.squeeze(0).squeeze(-1),axis=2).detach().cpu().numpy()
#         plt.imshow(Q_ij_avg**0.25) # (Power is just to increase contrast for better visualization)
#         plt.savefig(folder + 'attn//' + str(epoch) + '.png', bbox_inches='tight')
#         plt.title('Attention Matrix')
#         plt.show()

#          #########################################

#         # Plot eigenvalues per epoch

# #         plt.plot(all_lambdas[0:epoch,0,0], 'b')
# #         plt.plot(all_lambdas[0:epoch,1,0], 'r--')
# #         plt.plot(all_lambdas[0:epoch,0,:], 'b')
# #         plt.plot(all_lambdas[0:epoch,1,:], 'r--')

# #         lambdas = all_lambdas[epoch] # Unsorted lambdas
#         # Sort lambdas
#         fig, ax1 = plt.subplots()
        
#         idx = np.argsort(all_lambdas[epoch,0])
#         lambdas = all_lambdas[epoch,:,idx].T # Sorted lambdas

#         ax1.scatter(np.arange(args.d_v), lambdas[0], color='b',marker='o')
#         ax2 = ax1.twinx()
#         ax2.scatter(np.arange(args.d_v), lambdas[1], color ='r',marker='x')
#         ax1.tick_params(axis='y', labelcolor='blue')
#         ax2.tick_params(axis='y', labelcolor='red')
#         plt.grid()
# #         plt.savefig(folder + 'eigs//' + str(epoch) + '.png', bbox_inches='tight')
#         plt.title('Eigenvalues')
#         plt.show()

#          #########################################

# #         # Plot values of state transition matrix
# #         _, A_model = compute_state_transition_matrix(model,module,Pu,Pd,R1,R1i,args)
# #         plot_state_transition_matrix(A_model, marker='x',size=80)
# #         plot_state_transition_matrix(A, marker='o')
# # #         plot_state_transition_matrix(A_model/torch.max(A_model), marker='x',size=80)
# # #         plot_state_transition_matrix(A/torch.max(A), marker='o')
# #         plt.title('State Transition Matrix Comparison')
# #         plt.grid()
# #         plt.show()
        
#         #########################################
        
#         #         # Compute eigenvals of effective A and print
# #         A_model_np = A_model.detach().cpu().numpy()
# #         complex_matrix = A_model_np[0] + 1j * A_model_np[1]  # shape (2, 2)
# #         eigenvals = np.linalg.eigvals(complex_matrix)
# #         print(eigenvals)


# #         # Plot losses
# #         plt.plot(all_losses[0:epoch*args.num_samp])
# #         plt.title('Loss')
# #         plt.grid()
# #         plt.show()
        
# #         # Plot mean losses
# #         plt.plot(mean_epoch_losses[0:epoch])
# #         plt.title('Mean Loss')
# #         plt.grid()
# #         plt.show()

#         #########################################

#         # Plot log mean loss per epoch
#         plt.plot(log_mean_epoch_losses[0:epoch])
#         plt.title('Log Mean Loss')
#         plt.grid()
#         plt.show()

#         #########################################

#         #         print(module.lambda1[:,0,0]) # Print eigenvalues
#         lambda_h = compute_lambda_h(module.lambda1,args)
#         print(lambda_h.squeeze()[:,0]) # Print eigenvalues
#         #         print(torch.max(A)/torch.max(A_model)) # Print relative scales of real and learned state transition matrices
#         # Print relative weighting of attention and residual connections
#         #         print('alpha=', module.alpha)
#         #         print('beta=', module.beta)
#         print(complex_matmul(W_p, W_v).squeeze().detach().cpu()[0][0:args.m,0:args.m]) # Should be near identity
        
        
#         for idx, layer in enumerate(model.layers):
#             if hasattr(layer, 'eta_param'):
#         #         print('Layer:', idx, ': Mean grad step size = ', torch.mean(torch.sigmoid(layer.eta_param)).detach().cpu().numpy())
#                 print('Layer:', idx, ': Min grad step size = ', torch.min(torch.sigmoid(layer.eta_param)).squeeze().detach().cpu().numpy())
#                 print('Layer:', idx, ': Max grad step size = ', torch.max(torch.sigmoid(layer.eta_param)).squeeze().detach().cpu().numpy()) 
# #                 print('Layer:', idx, ': First component grad step size = ', torch.sigmoid(layer.eta_param).squeeze().detach().cpu().numpy()[0])

def visualize_results(model, train_dataset, all_losses, mean_epoch_losses, log_mean_epoch_losses, all_lambdas, R1, R1i, Pu, Pd, A, epoch, args):
    """
    Visualize results during training
    Plots the following:
        Noisy and true trajectory, and prediction of model
        State estimates and n_example data points
        Attention matrix
        Values of state transition matrix
    """
    
    folder = "C://Users//Pracioppo//Desktop//train_imgs//"
    plt.axis('equal')
    
    main_attention_block, last_inner_layer_of_main_attention_block = _get_visual_modules(model)

    # Error checking for the helper function's output
    if main_attention_block is None or last_inner_layer_of_main_attention_block is None:
        print("Error: Could not identify main attention block or its last inner layer for visualization. Skipping visualization.")
        return 

    module = last_inner_layer_of_main_attention_block # 'module' refers to the last inner shared attention block

    with torch.no_grad():
        # Access W_v and W_p from the main_attention_block as these are typically at the Nlayer level
        W_v = get_complex_weights(main_attention_block, 'W_v')
        W_p = get_complex_weights(main_attention_block, 'W_p')

        # Get prediction for random choice of input
        rand_idx = np.random.choice(args.num_samp)
        train_data, X_true, X_measure, t_measure = train_dataset.__getitem__(rand_idx)
        inputs = train_data[:, :-1].unsqueeze(0)
        # outputs = train_data[:, :-1].unsqueeze(0)
        if inputs.size()[1] == 1:
            inputs = torch.stack((inputs, torch.zeros_like(inputs)),dim=1)
            
        est, out, Q_ij, X_ij_hat_all, epoch_lambdas = model.forward(inputs, t_measure.unsqueeze(0))

        est = est.unsqueeze(-1)
        out = out.unsqueeze(-1)
        X_ij_hat_all = X_ij_hat_all.unsqueeze(-1)
        
        # Set plotting dims
        x_max = torch.max(X_true[:,0]).detach().cpu().numpy()
        x_min = torch.min(X_true[:,0]).detach().cpu().numpy()
        y_max = torch.max(X_true[:,1]).detach().cpu().numpy()
        y_min = torch.min(X_true[:,1]).detach().cpu().numpy()
        margin = 2
#           width = x_max - x_min
#           height = y_max - y_min
#           dim = 5*torch.ceil(max(width,height)/5).detach().cpu().numpy()
        
        #########################################
        
        # Plot trajectory
#           fig, ax = plt.subplots(figsize=(dim, dim))
        
        pred_map = batched_complex_matmul(R1i,out) # Reverse random mapping
        plot_trajectory(X_true[args.n:].unsqueeze(0),X_measure[1:].unsqueeze(0),pred_map,Pd)
        
        plt.xlim(x_min-margin, x_max+margin)
        plt.ylim(y_min-margin, y_max+margin)
        
#           plt.savefig(folder + 'trajecs//' + str(epoch) + '.png', bbox_inches='tight')
        plt.title('Trajectory')
        plt.legend()
        plt.show()
    
        #########################################
            
        # Plot state estimates at n_example data points
#           fig, ax = plt.subplots(figsize=(dim, dim))

        plot_trajectory(X_true[:-args.n].unsqueeze(0),0*X_measure[1:].unsqueeze(0),0*pred_map,Pd) # Plot actual trajectory
        x_hat = batched_complex_matmul(W_p, X_ij_hat_all*module.causal_mask)
        Xo_h = batched_complex_matmul(R1i,x_hat) # Reverse random mapping
        Xo = batched_complex_matmul(Pd,Xo_h).detach().cpu()[0,0].squeeze() # State estimates
    #           Xo = batched_complex_matmul(Pd,x_hat).detach().cpu()[0,0].squeeze()
        markers = ['o', 'v', 's', 'd', 'P']
        colors = ['pink', 'red', 'black', 'yellow', 'blue']
        mi = 0
        for i in range(args.seq_len):
            if np.mod(i,int(args.seq_len/args.n_example)) == 0:
                xi = Xo[i,i,:]
                x_est = Xo[i,0:i+1,:].numpy()
                plt.scatter(x_est.T[0],x_est.T[1], s=10, marker=markers[np.mod(mi,len(markers))], color=colors[np.mod(mi,len(colors))])
                plt.scatter(xi[0],xi[1], s=100, marker='x', color=colors[np.mod(mi,len(colors))])
                mi += 1

        plt.xlim(x_min-margin, x_max+margin)
        plt.ylim(y_min-margin, y_max+margin)

        plt.title('State Estimates')
#           plt.savefig(folder + 'ests//' + str(epoch) + '.png', bbox_inches='tight')
        plt.show()

        #########################################

        # Plot attention matrix
        Q_ij_avg = torch.mean(Q_ij.squeeze(0).squeeze(-1),axis=2).detach().cpu().numpy()
        plt.imshow(Q_ij_avg**0.25) # (Power is just to increase contrast for better visualization)
        plt.savefig(folder + 'attn//' + str(epoch) + '.png', bbox_inches='tight')
        plt.title('Attention Matrix')
        plt.show()

        #########################################

        # Plot eigenvalues per epoch

#           plt.plot(all_lambdas[0:epoch,0,0], 'b')
#           plt.plot(all_lambdas[0:epoch,1,0], 'r--')
#           plt.plot(all_lambdas[0:epoch,0,:], 'b')
#           plt.plot(all_lambdas[0:epoch,1,:], 'r--')

#           lambdas = all_lambdas[epoch] # Unsorted lambdas
        # Sort lambdas
        fig, ax1 = plt.subplots()
        
        idx = np.argsort(all_lambdas[epoch,0])
        lambdas = all_lambdas[epoch,:,idx].T # Sorted lambdas

        ax1.scatter(np.arange(args.d_v), lambdas[0], color='b',marker='o')
        ax2 = ax1.twinx()
        ax2.scatter(np.arange(args.d_v), lambdas[1], color ='r',marker='x')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        plt.grid()
#           plt.savefig(folder + 'eigs//' + str(epoch) + '.png', bbox_inches='tight')
        plt.title('Eigenvalues')
        plt.show()

        #########################################

#           # Plot values of state transition matrix
#           _, A_model = compute_state_transition_matrix(model,module,Pu,Pd,R1,R1i,args)
#           plot_state_transition_matrix(A_model, marker='x',size=80)
#           plot_state_transition_matrix(A, marker='o')
# #           plot_state_transition_matrix(A_model/torch.max(A_model), marker='x',size=80)
# #           plot_state_transition_matrix(A/torch.max(A), marker='o')
#           plt.title('State Transition Matrix Comparison')
#           plt.grid()
#           plt.show()
        
        #########################################
        
        #           # Compute eigenvals of effective A and print
#           A_model_np = A_model.detach().cpu().numpy()
#           complex_matrix = A_model_np[0] + 1j * A_model_np[1]  # shape (2, 2)
#           eigenvals = np.linalg.eigvals(complex_matrix)
#           print(eigenvals)


#           # Plot losses
#           plt.plot(all_losses[0:epoch*args.num_samp])
#           plt.title('Loss')
#           plt.grid()
#           plt.show()
        
#           # Plot mean losses
#           plt.plot(mean_epoch_losses[0:epoch])
#           plt.title('Mean Loss')
#           plt.grid()
#           plt.show()

        #########################################

        # Plot log mean loss per epoch
        plt.plot(log_mean_epoch_losses[0:epoch])
        plt.title('Log Mean Loss')
        plt.grid()
        plt.show()

        #########################################

        #           print(module.lambda1[:,0,0]) # Print eigenvalues
        lambda_h = compute_lambda_h(module.lambda1,args)
        print(lambda_h.squeeze()[:,0]) # Print eigenvalues
        #           print(torch.max(A)/torch.max(A_model)) # Print relative scales of real and learned state transition matrices
        # Print relative weighting of attention and residual connections
        #           print('alpha=', module.alpha)
        #           print('beta=', module.beta)
        print(complex_matmul(W_p, W_v).squeeze().detach().cpu()[0][0:args.m,0:args.m]) # Should be near identity
        
        
        for idx, layer in enumerate(main_attention_block.layers):
            if hasattr(layer, 'eta_param'):
        #           print('Layer:', idx, ': Mean grad step size = ', torch.mean(torch.sigmoid(layer.eta_param)).detach().cpu().numpy())
                print('Layer:', idx, ': Min grad step size = ', torch.min(torch.sigmoid(layer.eta_param)).squeeze().detach().cpu().numpy())
                print('Layer:', idx, ': Max grad step size = ', torch.max(torch.sigmoid(layer.eta_param)).squeeze().detach().cpu().numpy()) 
#                   print('Layer:', idx, ': First component grad step size = ', torch.sigmoid(layer.eta_param).squeeze().detach().cpu().numpy()[0])
                    
##########################################################################################
##########################################################################################

def visualize_results_attn(model, train_dataset, all_losses, mean_epoch_losses, log_mean_epoch_losses, all_lambdas, R1, R1i, Pu, Pd, A, epoch, args):
    """
    Visualize results during training
    Plots the following:
        Noisy and true trajectory, and prediction of model
        State estimates and n_example data points
        Attention matrix
        Values of state transition matrix
    """
    
    folder = "C://Users//Pracioppo//Desktop//train_imgs//"
    plt.axis('equal')
    
    with torch.no_grad():

        # print(module.lambda1)

        # Get prediction for random choice of input
        rand_idx = np.random.choice(args.num_samp)
        train_data, X_true, X_measure, t_measure = train_dataset.__getitem__(rand_idx)
        inputs = train_data[:, :-1].unsqueeze(0)
    #         inputs = inputs.reshape(1,100,args.d_e*2)

        if args.model_type == 'RealInputs':
            inputs = inputs[:,0] # Get real part
        elif args.model_type == 'ComplexInputs':
            pass
        else:
            print('Error. Set model_type.')

        out, attn_list = model.forward(inputs)
        
        if args.model_type == 'RealInputs':
            out = torch.stack((out, torch.zeros_like(out)),dim=1)
        elif args.model_type == 'ComplexInputs':
            pass
        else:
            print('Error!. Set model_type.')

        print(out.size())

        # Set plotting dims
        x_max = torch.max(X_true[:,0]).detach().cpu().numpy()
        x_min = torch.min(X_true[:,0]).detach().cpu().numpy()
        y_max = torch.max(X_true[:,1]).detach().cpu().numpy()
        y_min = torch.min(X_true[:,1]).detach().cpu().numpy()
        margin = 2

        #########################################
        
        # Plot trajectory
#         fig, ax = plt.subplots(figsize=(dim, dim))
        
        pred_map = batched_complex_matmul(R1i,out.unsqueeze(-1)) # Reverse random mapping
        plot_trajectory(X_true[args.n:].unsqueeze(0),X_measure[1:].unsqueeze(0),pred_map,Pd)
        
        plt.xlim(x_min-margin, x_max+margin)
        plt.ylim(y_min-margin, y_max+margin)
        
#         plt.savefig(folder + 'trajecs//' + str(epoch) + '.png', bbox_inches='tight')
        plt.show()
    
        #########################################

        # # Plot losses
        # plt.plot(all_losses[0:epoch*args.num_samp])
        # plt.grid()
        # plt.show()
        #       plt.plot(mean_epoch_losses[0:epoch])
        #       plt.grid()
        #       plt.show()
        
        #########################################
        
        # Plot attention matrices
        
        for attn in attn_list:
        
            plt.imshow(attn.squeeze().detach().cpu().numpy())
            plt.show()
        
        #########################################

        # Plot log mean loss per epoch
        plt.plot(log_mean_epoch_losses[0:epoch])
        plt.minorticks_on()
        plt.grid()
        plt.show()
        
##########################################################################################
##########################################################################################        
        
def _get_visual_modules(model: nn.Module):
    """
    Identifies and returns the main FullPrecisionAttentionBlock_Nlayer and its
    last inner FullPrecisionAttentionBlockShared layer from a given model instance.

    This helper function adapts to different top-level model architectures.

    Args:
        model (nn.Module): The top-level model instance (e.g., AFATransformerNetwork,
                           AFATransformerBlock, FullPrecisionAttentionBlock_Nlayer, or AFA_Nlayer).

    Returns:
        tuple[nn.Module | None, nn.Module | None]: A tuple containing:
            - The main FullPrecisionAttentionBlock_Nlayer instance for visualization.
            - The last FullPrecisionAttentionBlockShared instance within that main block.
            Returns (None, None) if the required modules cannot be found.
    """
    main_attention_block = None
    last_inner_layer_of_main_attention_block = None

    # Case 1: Model is directly FullPrecisionAttentionBlock_Nlayer
    if isinstance(model, FullPrecisionAttentionBlock_Nlayer):
        main_attention_block = model
   
    # Case 2: Model is AFATransformerBlock
    # (This block contains 'attn' which is a FullPrecisionAttentionBlock_Nlayer)
    elif isinstance(model, AFATransformerBlock):
        if hasattr(model, 'attn') and isinstance(model.attn, FullPrecisionAttentionBlock_Nlayer):
            main_attention_block = model.attn

    # Case 3: Model is AFATransformerNetwork
    # (This network contains 'blocks', which are AFATransformerBlock instances)
    elif isinstance(model, AFATransformerNetwork):
        # We'll take the attention block from the *first* AFATransformerBlock in the network
        if hasattr(model, 'blocks') and len(model.blocks) > 0:
            first_transformer_block = model.blocks[0]
            if hasattr(first_transformer_block, 'attn') and \
               isinstance(first_transformer_block.attn, FullPrecisionAttentionBlock_Nlayer):
                main_attention_block = first_transformer_block.attn
    
    # Case 4: Model is AFA_Nlayer (based on your original working code, assuming it directly contains Nlayer blocks)
    # (This case is speculative without its definition, but follows the pattern)
    elif isinstance(model, AFA_Nlayer):
        if hasattr(model, 'layers') and len(model.layers) > 0 and \
           isinstance(model.layers[0], FullPrecisionAttentionBlock_Nlayer):
            main_attention_block = model.layers[0]
    
    # If a main_attention_block was found, now try to get its last inner layer
    if main_attention_block is not None:
        if hasattr(main_attention_block, 'layers') and len(main_attention_block.layers) > 0 and \
           isinstance(main_attention_block.layers[-1], FullPrecisionAttentionBlockShared):
            last_inner_layer_of_main_attention_block = main_attention_block.layers[-1]
        else:
            print(f"Warning: Main attention block ({main_attention_block.__class__.__name__}) found, "
                  f"but its last inner layer (FullPrecisionAttentionBlockShared) could not be retrieved. "
                  f"Some visualizations might not work.")
            # If the last inner layer is critical, you might want to return (None, None) here,
            # or raise an error, depending on how robust you need this to be.
            # For now, we'll proceed with None for last_inner_layer_of_main_attention_block
            # but still return main_attention_block if it was found.

    return main_attention_block, last_inner_layer_of_main_attention_block

    