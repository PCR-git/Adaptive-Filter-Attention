import numpy as np
import torch
import torch.nn as nn

from matplotlib import pyplot as plt
import matplotlib.cm as cm
plt.rcParams['figure.figsize'] = [10, 10]
plt.rc('font', size=20)

from utils import complex_matmul, batched_complex_matmul
# from model import get_complex_weights
from model import ComplexLinearLayer, AFATransformerBlock
from model import AFA_1layer, AFATransformerNetwork
from model import MultiheadIsotropicAFA
from model import MultiheadIsotropicAFA_1layer
from model import compute_lambda

##########################################################################################
##########################################################################################

def plot_trajectory(X_true,X_measure,X_est):
    """
    Plot actual, measured, and estimated trajectories with labels for a legend.
    """

    # Actual trajectory
    X_true_plt = X_true.squeeze().detach().cpu().numpy()
    plt.plot(X_true_plt.T[0],X_true_plt.T[1],'black', label='Ground Truth') # Added label

    # Noisy trajectory
    X_noisy = X_measure.detach().cpu().squeeze().numpy()
    plt.plot(X_noisy.T[0], X_noisy.T[1], 'b--', label='Measured') # Added label

    # Predicted trajectory
    X_est_plt = X_est.squeeze(0)[0].detach().cpu().squeeze().numpy()
    plt.plot(X_est_plt.T[0], X_est_plt.T[1], 'r--', label='Predicted') # Added label

    plt.grid()

##########################################################################################
##########################################################################################

def compute_state_matrix(module,Pu,Pd,R1,R1i,args):
    with torch.no_grad():
        lambda_h_model = compute_lambda_shared_multihead(module.lambda_real_v, module.lambda_imag_v, args).squeeze(-1)
        lambda_h_model = lambda_h_model.view(2,1,args.d_v,1)
        ############### DELETE #####################
        mask = torch.zeros(2,1,args.d_v,1).to(args.device)
        mask[:,:,0:2,:] = 1
        lambda_h_model = lambda_h_model*mask
        ################################# ############

        DD = torch.stack((torch.diag(lambda_h_model[0].squeeze()),torch.diag(lambda_h_model[1].squeeze())))

        W_v_r = module.W_v.weight[0:args.d_v, :]
        W_v_i = module.W_v.weight[args.d_v:2*args.d_v, :] 
        W_v = torch.stack((W_v_r, W_v_i), dim=0) # Shape: (2, d_v, d_v)
        W_o_r = module.W_o.weight[:,0:args.d_v] 
        W_o_i = module.W_o.weight[:,args.d_v:2*args.d_v] 
        W_o = torch.stack((W_o_r, W_o_i), dim=0) # Shape: (2, d_v, d_v)
        W_o[1] *= -1
        
#         A = complex_matmul(W_o[:,0:2,0:2], complex_matmul(DD[:,0:2,0:2], W_v[:,0:2,0:2]))
        
        y = complex_matmul(W_o, complex_matmul(DD, W_v))
        z = complex_matmul(R1i,complex_matmul(y,R1))
        A = complex_matmul(Pd,complex_matmul(z,Pu))
    return A[0].unsqueeze(0)

##########################################################################################
##########################################################################################

def plot_state_matrix(A,marker,size=32,color=None):
    """
    Plot entries of the state matrix (real values in blue, imaginary in red)
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

def visualize_results(model, train_dataset, all_losses, mean_epoch_losses, log_mean_epoch_losses, all_lambdas, R1, R1i, Pu, Pd, A, epoch, args):
    """
    Visualize results during training
    Plots the following:
        Noisy and true trajectory, and prediction of model
        State estimates at n_example data points
        Attention matrix
        Values of state matrix
    Prints the following:
        First eigenvalue (which we can compare to our simple 2D example system)
        W_v W_o, which we expect to be about I
        Gradient step sizes in AFA
    Can also save images to folder, for use in producing animations
    """
    
    folder = "C://Users//Pracioppo//Desktop//train_imgs//"
    plt.axis('equal')
    
    main_attention_block, last_inner_layer_of_main_attention_block = _get_visual_modules(model)

    # Error checking for the helper function's output
    if main_attention_block is None:
        print("Error: Could not identify main attention block or its last inner layer for visualization. Skipping visualization.")
        return 

    module = last_inner_layer_of_main_attention_block # 'module' refers to the last inner shared attention block

    with torch.no_grad():

        # Get prediction for random choice of input
        rand_idx = np.random.choice(args.num_samp)
        train_data, X_true, X_measure, t_measure = train_dataset.__getitem__(rand_idx)
        
        inputs = train_data.unsqueeze(0)[:, :-1].unsqueeze(1)

        out, output_dict = model.forward(inputs)
        est = output_dict['est_latent']
        attn_mat = output_dict['attn_mat']
        x_hat = output_dict['x_hat']
        lambda_h = output_dict['epoch_lambdas']
        unnormalized_attention = output_dict['unnormalized_attention']
        
        x_hat = torch.stack((x_hat, torch.zeros_like(x_hat)),dim=1) # Add zero imaginary part

        est = est.unsqueeze(-1)
        out = out.unsqueeze(1).unsqueeze(-1)
        
        # Set plotting dims
        x_max = torch.max(X_true[:,0]).detach().cpu().numpy()
        x_min = torch.min(X_true[:,0]).detach().cpu().numpy()
        y_max = torch.max(X_true[:,1]).detach().cpu().numpy()
        y_min = torch.min(X_true[:,1]).detach().cpu().numpy()
        margin = 2

        #########################################
        
        # Plot trajectory
#           fig, ax = plt.subplots(figsize=(dim, dim))
        
        pred_map = torch.matmul(R1i,out) # Reverse random mapping
        X_est = torch.matmul(Pd,pred_map) # Map back to lower dim
        if args.concat_mag == 1:
            X_est = X_est[:,:,:,0:args.m,:]
        plot_trajectory(X_true[args.n:].unsqueeze(0),X_measure[1:].unsqueeze(0),X_est)
        
        plt.xlim(x_min-margin, x_max+margin)
        plt.ylim(y_min-margin, y_max+margin)
        
#           plt.savefig(folder + 'trajecs//' + str(epoch) + '.png', bbox_inches='tight')
        plt.title('Trajectory')
        plt.legend()
        plt.show()
    
        #########################################
            
        # Plot state estimates at n_example data points
#           fig, ax = plt.subplots(figsize=(dim, dim))

        plot_trajectory(X_true[:-args.n].unsqueeze(0),0*X_measure[1:].unsqueeze(0),0*X_est) # Plot actual trajectory

        Xo_h = torch.matmul(R1i,x_hat) # Reverse random mapping
        Xo = torch.matmul(Pd,Xo_h).detach().cpu()[0,0].squeeze() # State estimates
        markers = ['o', 'v', 's', 'd', 'P']
        colors = ['pink', 'red', 'black', 'yellow', 'blue']
        mi = 0
        for i in range(args.seq_len):
            if np.mod(i,int((args.seq_len)/args.n_example)) == 0 and i > 0:
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
        
#         # Plot row sum of unnormalized attention (measure of total confidence)
 
#         P_tot = np.sum(unnormalized_attention[0].detach().cpu().numpy(),-2)
#         head = 0
#         P_tot_head = P_tot[:,head]
#         plt.plot(P_tot_head)
#         plt.grid()
#         plt.title('Total Confidence (row sum of unnormalized attention)')
#         plt.show()

        #########################################

        # Plot LAST attention matrix in network
        
        # Simplified model:
        for head in range(attn_mat.size()[-1]):
            Q_ij = attn_mat[:,:,:,head]
            Q_ij_avg = Q_ij.squeeze().detach().cpu().numpy() # If using simplified model
            plt.imshow(Q_ij_avg**0.25) # (Power is just to increase contrast for better visualization)
    #         plt.savefig(folder + 'attn//' + str(epoch) + '.png', bbox_inches='tight')
            plt.title('Attention Matrix, Head: ' + str(head))
            plt.show()

        #########################################
        
        # Plot ALL Attention matrices in network
        
        attn_matrix_count = 0
        for module_name, module_plt in model.named_modules():            
            if hasattr(module_plt, 'attn_mat') and module_plt.attn_mat is not None:
      
                Q_matrix = module_plt.attn_mat # Get the stored attention matrix

                # Q_matrix is expected to be (Batch, seq_len, seq_len)
                # Squeeze the batch dimension for plotting if it's a single batch
                Q_ij_viz = Q_matrix.detach().cpu().numpy() # (seq_len, seq_len)

                plt.imshow(Q_ij_viz**0.25)
                plt.title(f'Attention Matrix - Layer {attn_matrix_count}')
                plt.show()
                attn_matrix_count += 1
        
        #########################################        

        # Plot eigenvalues per epoch

#           plt.plot(all_lambdas[0:epoch,0,0], 'b')
#           plt.plot(all_lambdas[0:epoch,1,0], 'r--')
#           plt.plot(all_lambdas[0:epoch,0,:], 'b')
#           plt.plot(all_lambdas[0:epoch,1,:], 'r--')

#           lambdas = all_lambdas[epoch] # Unsorted lambdas
        # Sort lambdas
        fig, ax1 = plt.subplots()
        
        lambdas = all_lambdas[epoch] # Unsorted lambdas
#         lambdas = all_lambdas[epoch].T # Unsorted lambdas
#         idx = np.argsort(all_lambdas[epoch,0])
#         lambdas = all_lambdas[epoch,:,idx].T # Sorted lambdas

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

#         # Plot values of state matrix
#         _, A_model = compute_state_matrix(module,Pu,Pd,R1,R1i,args)
#         plot_state_matrix(A_model, marker='x',size=80)
#         plot_state_matrix(A, marker='o')

#         plt.title('State Matrix Comparison')
#         plt.grid()
#         plt.show()
        
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
        
        one_step_diff = inputs[:,:,1:]- inputs[:,:,:-1]
        baseline_loss = torch.log(torch.mean(one_step_diff**2)).detach().cpu().numpy()
        plt.plot(range(epoch),baseline_loss*np.ones(epoch), 'r--')
        
        plt.minorticks_on()
        plt.grid()
        plt.show()

        #########################################

#         #           print(module.lambda1[:,0,0]) # Print eigenvalues
#         lambda_h = compute_lambda_h(module.lambda1,args)
#         print('0th Eigenvalue:')
#         print(lambda_h.squeeze()[:,0]) # Print eigenvalues
        
#                   print(torch.max(A)/torch.max(A_model)) # Print relative scales of real and learned state matrices
#         Print relative weighting of attention and residual connections
#                   print('alpha=', module.alpha)
#                   print('beta=', module.beta)
#         # Access W_v and W_o from the main_attention_block as these are typically at the Nlayer level
#         W_v = get_complex_weights(main_attention_block, 'W_v')
#         W_o = get_complex_weights(main_attention_block, 'W_o')
#         print('W_o * W_v: (should be near I:')
#         print(complex_matmul(W_o, W_v).squeeze().detach().cpu()[0][0:args.m,0:args.m]) # Should be near identity
        
        #########################################
    
        if hasattr(main_attention_block, 'layers'):
            for idx, layer in enumerate(main_attention_block.layers):
                if hasattr(layer, 'eta_param'):
            #           print('Layer:', idx, ': Mean grad step size = ', torch.mean(torch.sigmoid(layer.eta_param)).detach().cpu().numpy())
                    print('Layer:', idx, ': Min grad step size = ', \
                          torch.min(torch.sigmoid(layer.eta_param)).squeeze().detach().cpu().numpy())
                    print('Layer:', idx, ': Max grad step size = ', \
                          torch.max(torch.sigmoid(layer.eta_param)).squeeze().detach().cpu().numpy()) 
#                       print('Layer:', idx, ': First component grad step size = ', \
#                             torch.sigmoid(layer.eta_param).squeeze().detach().cpu().numpy()[0])
                    
##########################################################################################
##########################################################################################

def visualize_results_attn(model, train_dataset, all_losses, mean_epoch_losses, log_mean_epoch_losses, all_lambdas, R1, R1i, Pu, Pd, A, epoch, args):
    """
    Visualize results during training
    Plots the following:
        Noisy and true trajectory, and prediction of model
        State estimates and n_example data points
        Attention matrix
        Values of state matrix
    """
    
    folder = "C://Users//Pracioppo//Desktop//train_imgs//"
    plt.axis('equal')
    
    with torch.no_grad():

        # print(module.lambda1)

        # Get prediction for random choice of input
        rand_idx = np.random.choice(args.num_samp)
        train_data, X_true, X_measure, t_measure = train_dataset.__getitem__(rand_idx)

        inputs = train_data.unsqueeze(0)[:, :-1]

        out, attn_list = model.forward(inputs)

        out = out.unsqueeze(1)

        # Set plotting dims
        x_max = torch.max(X_true[:,0]).detach().cpu().numpy()
        x_min = torch.min(X_true[:,0]).detach().cpu().numpy()
        y_max = torch.max(X_true[:,1]).detach().cpu().numpy()
        y_min = torch.min(X_true[:,1]).detach().cpu().numpy()
        margin = 2

        #########################################
        
        # Plot trajectory
#         fig, ax = plt.subplots(figsize=(dim, dim))
        
        pred_map = torch.matmul(R1i,out.unsqueeze(-1)) # Reverse random mapping
        est = torch.matmul(Pd,pred_map) # Map back to lower dim
        plot_trajectory(X_true[args.n:].unsqueeze(0),X_measure[1:].unsqueeze(0),est)
        
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
#             print(attn.size())
            for head in range(args.num_heads):
                plt.imshow(attn.squeeze(0)[head].detach().cpu().numpy())
                plt.show()
        
        #########################################

        # Plot log mean loss per epoch
        plt.plot(log_mean_epoch_losses[0:epoch])
        
        one_step_diff = inputs[:,:,1:]- inputs[:,:,:-1]
        baseline_loss = torch.log(torch.mean(one_step_diff**2)).detach().cpu().numpy()
        plt.plot(range(epoch),baseline_loss*np.ones(epoch), 'r--')
        
        plt.minorticks_on()
        plt.grid()
        plt.show()
        
        #########################################
        
#         for idx, layer in enumerate(model.layers):
#             if hasattr(layer, 'eta_param'):
#         #         print('Layer:', idx, ': Mean grad step size = ', torch.mean(torch.sigmoid(layer.eta_param)).detach().cpu().numpy())
#                 print('Layer:', idx, ': Min grad step size = ', torch.min(torch.sigmoid(layer.eta_param)).squeeze().detach().cpu().numpy())
#                 print('Layer:', idx, ': Max grad step size = ', torch.max(torch.sigmoid(layer.eta_param)).squeeze().detach().cpu().numpy())
#                 print('Layer:', idx, ': Min grad step size = ', torch.sigmoid(layer.eta_param).squeeze().detach().cpu().numpy()[0])

#         for idx, layer in enumerate(model.layers):
#             if hasattr(layer, 'noise_floor'):
#                 print('Layer:', idx, ': Noise floor = ', layer.noise_floor.squeeze().detach().cpu().numpy())
        
##########################################################################################
##########################################################################################        
        
def _get_visual_modules(model: nn.Module):
    """
    Identifies and returns the main model and its
    last inner attention layer from a given model instance,
    for use in visualize_results.

    This helper function adapts to different top-level model architectures.

    Args:
        model (nn.Module): The top-level model instance.

    Returns:
        tuple[nn.Module | None, nn.Module | None]: A tuple containing:
            - The main model instance for visualization.
            - The last attention layer instance within that main block.
            Returns (None, None) if the required modules cannot be found.
    """
    
    main_attention_block = None
    last_inner_layer_of_main_attention_block = None
            
   # Model is AFATransformerBlock
    if isinstance(model, AFATransformerBlock):
        if hasattr(model, 'attn'):
            if isinstance(model.attn, MultiheadIsotropicAFA):
                main_attention_block = model.attn
    
    # Model is AFATransformerNetwork
    # (This network contains 'blocks', which are AFATransformerBlock instances)
    elif isinstance(model, AFATransformerNetwork):
        # We'll take the attention block from the *first* AFATransformerBlock in the network
        if hasattr(model, 'blocks') and len(model.blocks) > 0:
            first_transformer_block = model.blocks[0]
            if hasattr(first_transformer_block, 'attn'):
                if isinstance(first_transformer_block.attn, MultiheadIsotropicAFA):
                    main_attention_block = first_transformer_block.attn            
            
    ###################
            
    # If a main_attention_block was found, now try to get its last inner layer
    if main_attention_block is not None:
        if isinstance(main_attention_block, MultiheadIsotropicAFA): 
            last_inner_layer_of_main_attention_block = main_attention_block
        else:
            print(f"Warning: Main attention block ({main_attention_block.__class__.__name__}) found, "
                  f"but its last inner layer (FullPrecisionAttentionBlockShared) could not be retrieved. "
                  f"Some visualizations might not work.")
            
    ###################

    # Model is MultiheadIsotropicAFA_1layer
    if isinstance(model, MultiheadIsotropicAFA_1layer):
        if hasattr(model, 'layers') and len(model.layers) > 0 and \
           isinstance(model.layers[0], MultiheadIsotropicAFA):
            main_attention_block = model.layers[0]
            last_inner_layer_of_main_attention_block = model.layers[0]
            
    return main_attention_block, last_inner_layer_of_main_attention_block

    