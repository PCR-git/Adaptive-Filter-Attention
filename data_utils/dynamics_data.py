import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

from utils import complex_matmul
from dynamics import stochastic_LTI
from dynamics import get_nth_measurement, get_random_measurements

##########################################################################################
##########################################################################################

def construct_random_mapping(S1, Si1, args):
    """
    Embed data in higher dimension.
    Optionally, map through random orthogonal matrix.
    """
    
    # Construct matrices to map between higher/lower dimensions
    Pu = torch.zeros(2, 1, args.d_e, args.m).to(args.device)
    Pd = torch.zeros(2, 1, args.m, args.d_e).to(args.device)
#     Pu[:,:,0:2,:] = Si1 # Map to higher dim
#     Pd[:,:,:,0:2] = S1 # Map back to lower dim
    Pu[0,:,0:args.m,0:args.m] = torch.eye(args.m,args.m) # Map to higher dim
    Pd[0,:,0:args.m,0:args.m] = torch.eye(args.m,args.m) # Map back to lower dim

    if args.rand_embed == 1:
        # Construct random orthogonal matrix and its inverse
        R1 = torch.zeros(2, 1, args.d_e, args.d_e).to(args.device).to(args.device)
        R1i = torch.zeros(2, 1, args.d_e, args.d_e).to(args.device).to(args.device)
        rand_mat = torch.randn(args.d_e,args.d_e).to(args.device)/torch.sqrt(torch.tensor(args.d_e, dtype=torch.float32))
        Q, _ = torch.linalg.qr(rand_mat) # Q is an orthogonal matrix
        R1[0] = Q # Random complex orthogonal matrix
        R1i[0] = Q.T # For orthogonal matrices, inverse is transpose
    else:
        # Alternatively, instead of a random matrix, just use identity
        R1 = torch.zeros(2, 1, args.d_e, args.d_e).to(args.device).to(args.device)
        R1i = torch.zeros(2, 1, args.d_e, args.d_e).to(args.device).to(args.device)
        R1[0,:,0:args.m,0:args.m] = torch.eye(args.m,args.m)
        R1i[0,:,0:args.m,0:args.m] = torch.eye(args.m,args.m)
    
    return Pu, Pd, R1, R1i

##########################################################################################
##########################################################################################

def construct_data(A, Pu, Pd, R1, R1i, Npts, t_v, sigma_process, sigma_measure, args, x0=None):
    """
    Collect simulated data and map to higher dimension
    """

    if x0 == None:
#         x0 = 10
#         x0 = (torch.randn(args.m)*10).unsqueeze(0).unsqueeze(-1).to(args.device) # Get random initial condition
#         x0 = (torch.randn(args.m)*2).unsqueeze(0).unsqueeze(-1).to(args.device) # Get random initial condition
#         x0 += torch.sign(x0)*10
        rm = 20.0
        rd = 3.0
        theta0 = torch.rand(1) * 2 * np.pi
        r0 = rm + (torch.rand(1) * 2*rd) - rd
        x0 = torch.tensor([r0 * torch.cos(theta0), r0 * torch.sin(theta0)])
        x0 = x0.to(args.device).unsqueeze(0).unsqueeze(-1)

    # Simulate system
    #     X_true, X_measure = stochastic_LTI(A, x0, Npts, args, sigma_process=sigma_process, sigma_process_0=sigma_process_0, sigma_measure=sigma_measure) # Simulate system
    X_true, X_measure_full = stochastic_LTI(A, x0, Npts, args, sigma_process=sigma_process, sigma_measure = sigma_measure) # Simulate system
    
    if args.t_equal == 1: # If equal time intervals
        idxs, t_measure, X_measure = get_nth_measurement(X_measure_full, t_v, Npts, n=args.n)
    else: # If unequal time intervals
        idxs, t_measure, X_measure = get_random_measurements(X_measure_full, t_v, args)
    
    X_measure_c = torch.zeros((2,X_measure.size()[0], X_measure.size()[1], X_measure.size()[2])).to(args.device)
    X_measure_c[0] = X_measure
    X_high = complex_matmul(Pu,X_measure_c) # Map to higher dim

    X_random = complex_matmul(R1,X_high) # Map to random basis
    
#     Xo = torch.matmul(R1i,X_random) # Reverse the mapping

    X_random = X_random.squeeze(-1)
    X_high = X_high.squeeze(-1)
    X_true = X_true.squeeze(-1)
    X_measure = X_measure.squeeze(-1)

    return X_random, X_high, X_true, X_measure, t_measure

##########################################################################################
##########################################################################################

class TrainDataset(Dataset):
    """
    Define a custom Dataset
    """
    
    def __init__(self, Train_Data, X_true_all, X_measure_all, t_measure_all):
        self.Train_Data = Train_Data
        self.X_true_all = X_true_all
        self.X_measure_all = X_measure_all
        self.t_measure_all = t_measure_all

    def __len__(self):
        return self.Train_Data.size(0)

    def __getitem__(self, idx):
        train_data = self.Train_Data[idx]
        X_true = self.X_true_all[idx]
        X_measure = self.X_measure_all[idx]
        t_measure = self.t_measure_all[idx]
        
        return train_data, X_true, X_measure, t_measure
    
##########################################################################################
##########################################################################################

def create_train_loader(A, S1, Si1, Pu, Pd, R1, R1i, t_v, sigma_process, sigma_measure, args):
    """
    Create train loader
    """
    
    Train_Data = torch.zeros(args.num_samp, 2, args.seq_len+1, args.d_e).to(args.device)
    X_true_all = torch.zeros(args.num_samp, args.N_t+args.n+1, 2).to(args.device)
    X_measure_all = torch.zeros(args.num_samp, args.seq_len+1, 2).to(args.device)
    t_measure_all = torch.zeros(args.num_samp, args.seq_len+1).to(args.device)

#     for it in tqdm(range(args.num_samp)):
    for it in range(args.num_samp):
        # Construct data for one iteration using the dynamics simulation
        X_random, X_high, X_true, X_measure, t_measure = construct_data(A, Pu, Pd, R1, R1i, args.N_t + args.n, t_v, sigma_process, sigma_measure, args)

#         train_data = X_high # Use original data
#         train_data = X_random # Use data mapped through random mat mul

        Train_Data[it] = X_random
        X_true_all[it] = X_true
        X_measure_all[it] = X_measure
        t_measure_all[it] = t_measure

    # Create a Dataset and DataLoader
    train_dataset = TrainDataset(Train_Data, X_true_all, X_measure_all, t_measure_all)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
#     # Normalize by the time interval
#     t_measure_all = t_measure_all / (t_measure_all[:,-1] - t_measure_all[:,0]).unsqueeze(1)
    
    return train_loader, train_dataset, X_true_all, X_measure_all, t_measure_all
