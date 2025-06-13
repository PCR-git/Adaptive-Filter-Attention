import numpy as np
import torch

from matplotlib import pyplot as plt
import matplotlib.cm as cm
plt.rcParams['figure.figsize'] = [10, 10]
plt.rc('font', size=20)

from utils import complex_matmul, batched_complex_matmul
from model import compute_lambda_h

##########################################################################################
##########################################################################################

def plot_trajectory(X_true,X_measure,est_h,Pd):
    """
    Plot actual, measured, and estimated trajectories
    """

    # Actual trajectory
    X_true_plt = X_true.squeeze().detach().cpu().numpy()
    plt.plot(X_true_plt.T[0],X_true_plt.T[1],'black')

    # Noisy trajectory
    traj = X_measure.detach().cpu().squeeze().numpy()
    plt.plot(traj.T[0], traj.T[1], 'b--')

    # Predicted trajectory
    est = batched_complex_matmul(Pd,est_h) # Map back to lower dim
    est1_plt = est.squeeze(0)[0].detach().cpu().squeeze().numpy()
    plt.plot(est1_plt.T[0], est1_plt.T[1], 'r--')

    plt.grid()
#     plt.show()


# def compute_state_transition_matrix_v1(module,Pu,Pd,R1,R1i):
#     with torch.no_grad():
#         lambda_h_model = compute_lambda_h(module.lambda1,args)
#         ############### DELETE #####################
#         mask = torch.zeros(2,1,args.embed_dim,1).to(args.device)
#         mask[:,:,0:args.m,:] = 1
#         lambda_h_model = lambda_h_model*mask
#         ################################# ############

#         DD = torch.stack((torch.diag(lambda_h_model[0].squeeze()),torch.diag(lambda_h_model[1].squeeze()))).unsqueeze(1)
#         y = complex_matmul(module.W_p, complex_matmul(DD, module.W_v))
        
# #         z = y
#         z = complex_matmul(R1i,complex_matmul(y,R1))
#         A = complex_matmul(Pd,complex_matmul(z,Pu))
#     return A

##########################################################################################
##########################################################################################

def compute_state_transition_matrix(module,Pu,Pd,R1,R1i,args):
    """
    Compute elements of state transition matrix A
    """
    
    dt = args.dt*args.n # Time step

    with torch.no_grad():
        # Get eigenvalues
        lambda_h_model = compute_lambda_h(module.lambda1,args)
#         lambda_h_model = compute_lambda_h_v2(module.lambda1)
        W_v = module.W_v
        W_p = module.W_p
        
        if args.weight_mask == 1:
            W_v = W_v*module.weight_mask_v
            W_p = W_p*module.weight_mask_v
            lambda_h_model = lambda_h_model*module.eigen_mask

        DD = torch.stack((torch.diag(lambda_h_model[0].squeeze()),torch.diag(lambda_h_model[1].squeeze()))).unsqueeze(1) # Diagonal matrix
        I = torch.stack((torch.eye(args.d_v, args.d_v), torch.zeros((args.d_v, args.d_v)))).unsqueeze(1).to(args.device) # Identity
        I_out = torch.stack((torch.eye(args.embed_dim, args.embed_dim), torch.zeros((args.embed_dim, args.embed_dim)))).unsqueeze(1).to(args.device) # Identity
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

def visualize_results(model, module, train_dataset, all_losses, mean_epoch_losses, log_mean_epoch_losses, all_lambdas, R1, R1i, Pu, Pd, A, epoch, args):
    """
    Visualize results during training
    Plots the following:
        Noisy and true trajectory, and prediction of model
        State estimates and n_example data points
        Attention matrix
        Values of state transition matrix
    """

    with torch.no_grad():

#         W_p = module.W_p*module.weight_mask
#         W_v = module.W_v*module.weight_mask
        W_p = module.W_p
        W_v = module.W_v

        # print(module.lambda1)

        # Get prediction for random choice of input
        rand_idx = np.random.choice(args.num_samp)
        train_data, X_true, X_measure, t_measure = train_dataset.__getitem__(rand_idx)
        inputs = train_data[:, :-1].unsqueeze(0)
        # outputs = train_data[:, :-1].unsqueeze(0)
        est, out, Q_ij, X_ij_hat_all, epoch_lambdas = model.forward(inputs,t_measure.unsqueeze(0))

        est = est.unsqueeze(-1)
        out = out.unsqueeze(-1)
        X_ij_hat_all = X_ij_hat_all.unsqueeze(-1)
        
        # Plot trajectory
        pred_map = batched_complex_matmul(R1i,out)
        plot_trajectory(X_true[args.n:].unsqueeze(0),X_measure[1:].unsqueeze(0),pred_map,Pd)
        plt.show()

        # Plot state estimates at n_example data points
        plot_trajectory(X_true[:-args.n].unsqueeze(0),0*X_measure[1:].unsqueeze(0),0*pred_map,Pd) # Plot actual trajectory
        x_hat = batched_complex_matmul(W_p, X_ij_hat_all*module.causal_mask)
        Xo_h = batched_complex_matmul(R1i,x_hat)
        Xo = batched_complex_matmul(Pd,Xo_h).detach().cpu()[0,0].squeeze() # State estimates
    #         Xo = batched_complex_matmul(Pd,x_hat).detach().cpu()[0,0].squeeze()
        markers = ['o', 'v', 's', 'd', 'P']
        colors = ['pink', 'red', 'black', 'yellow', 'blue']
        mi = 0
        for i in range(args.seq_len):
            if np.mod(i,int(args.seq_len/args.n_example)) == 0:
                xi = Xo[i,i,:]
                x_est = Xo[i,0:i+1,:].numpy()
                plt.scatter(x_est.T[0],x_est.T[1], s=10, marker=markers[np.mod(mi,len(markers))], color=colors[np.mod(mi,len(colors))])
                plt.scatter(xi.T[0],xi.T[1], s=100, marker='x', color=colors[np.mod(mi,len(colors))])
                mi += 1
        plt.show()

        # Plot attention matrix
        Q_ij_avg = torch.mean(Q_ij.squeeze(0).squeeze(-1),axis=2).detach().cpu().numpy()
        plt.imshow(Q_ij_avg**0.25) # (Power is just to increase contrast for better visualization)
        plt.show()

        # # Plot losses
        # plt.plot(all_losses[0:epoch*args.num_samp])
        # plt.grid()
        # plt.show()
        #       plt.plot(mean_epoch_losses[0:epoch])
        #       plt.grid()
        #       plt.show()

        # Plot log mean loss per epoch
        plt.plot(log_mean_epoch_losses[0:epoch])
        plt.grid()
        plt.show()
        
#         # Plot eigenvalues per epoch
# #         plt.plot(all_lambdas[0:epoch,0,0], 'b')
# #         plt.plot(all_lambdas[0:epoch,1,0], 'r--')
#         plt.plot(all_lambdas[0:epoch,0,:], 'b')
#         plt.plot(all_lambdas[0:epoch,1,:], 'r--')
#         plt.grid()
#         plt.show()

        # Plot values of state transition matrix
        _, A_model = compute_state_transition_matrix(module,Pu,Pd,R1,R1i,args)
    #             plot_state_transition_matrix(A, marker='s',color='black')
    #             plot_state_transition_matrix(A_model, marker='x',size=80)
    #             plot_state_transition_matrix(A_actual, marker='o')
        plot_state_transition_matrix(A_model/torch.max(A_model), marker='x',size=80)
        plot_state_transition_matrix(A/torch.max(A), marker='o')
        plt.grid()
        plt.show()

#         print(module.lambda1[:,0,0]) # Print eigenvalues
        lambda_h = compute_lambda_h(module.lambda1,args)
        print(lambda_h.squeeze()[:,0]) # Print eigenvalues
#         print(torch.max(A)/torch.max(A_model)) # Print relative scales of real and learned state transition matrices
        # Print relative weighting of attention and residual connections
#         print('alpha=', module.alpha)
#         print('beta=', module.beta)
        print(complex_matmul(W_p, W_v).squeeze().detach().cpu()[0][0:args.m,0:args.m]) # Should be near identity

#         # Compute eigenvals of effective A and print
#         A_model_np = A_model.detach().cpu().numpy()
#         complex_matrix = A_model_np[0] + 1j * A_model_np[1]  # shape (2, 2)
#         eigenvals = np.linalg.eigvals(complex_matrix)
#         print(eigenvals)