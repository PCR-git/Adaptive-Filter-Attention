import numpy as np
import torch

##########################################################################################
##########################################################################################

# Dynamic simulation for linear time invariant system

# def stochastic_LTI(A, x0, N_t, args, sigma_process=1, sigma_process_0=0, sigma_measure=1):
#     """
#     Simulates a linear time-invariant system with zero-mean Gaussian noise using Euler integration.
#     """
#     m = args.m
#     dt = args.dt
#     device = args.device

#     # Precompute noise
#     process_noise = sigma_process * torch.randn((N_t, m, 1), device=device)
#     measurement_noise = sigma_measure * torch.randn((N_t, m, 1), device=device)

#     # Allocate trajectory array
#     X = torch.zeros((N_t, m, 1), device=device)

#     # Initial condition with optional initial process noise
# #     x = x0 + sigma_process_0 * torch.randn((m, 1), device=device)
#     x = x0
#     X[0] = x  # Store initial state as first timestep

#     # Simulate over time (start from t = 1)
#     for t in range(1, N_t):
#         xp = torch.matmul(A, x) # Compute velocity
#         x = x + xp * dt # Euler integration
#         x += process_noise[t] * np.sqrt(dt) # Add process noise
#         X[t] = x # Append current state to array

#     X_measure = X + measurement_noise # Add measuremnt noise
        
#     return X, X_measure

def stochastic_LTI(A, x0, N_t, args, sigma_process=1, sigma_process_0=0, sigma_measure=1):
    """
    Simulates a linear time-invariant system with zero-mean Gaussian noise using Euler integration.
    """
    m = args.m
    dt = args.dt
    device = args.device

    # Precompute noise
    process_noise = sigma_process * torch.randn((N_t+1, m, 1), device=device)
    measurement_noise = sigma_measure * torch.randn((N_t+1, m, 1), device=device)

    # Allocate trajectory array
    X = torch.zeros((N_t+1, m, 1), device=device)

    # Initial condition with optional initial process noise
#     x = x0 + sigma_process_0 * torch.randn((m, 1), device=device)
    x = x0
    X[0] = x  # Store initial state as first timestep

    # Simulate over time (start from t = 1)
    for t in range(1, N_t+1):
        xp = torch.matmul(A, x) # Compute velocity
        x = x + xp * dt # Euler integration
        x += process_noise[t] * np.sqrt(dt) # Add process noise
        X[t] = x # Append current state to array

    X_measure = X + measurement_noise # Add measuremnt noise
        
    return X, X_measure

##########################################################################################
##########################################################################################

class DynamicSim:
    """
    Dynamic simulation class
    Uses simple Euler update
    """
    
    def __init__(self, device):
        self.device = device    # Device
        self.model = None            # Placeholder for a callable model

    def set_model(self, model):
        """
        Set the dynamical system model (must be a callable: model(t, X)).
        """
        self.model = model

    def eq_of_Motion(self, t, X):
        """
        Evaluate the equations of motion. Must return dx/dt.
        """
        if self.model is None:
            raise NotImplementedError("No model set for equation of motion.")
        return self.model(t, X)

    def simulate_ODE(self, x0, tf, t0, dt, sigma_process, sigma_process_0, sigma_measure):
        m = x0.shape[0]
        N_t = int((tf - t0) / dt) + 1  # Number of time steps
        t_v = torch.linspace(t0, tf, N_t, device=self.device)

        # Precompute noise
        process_noise = sigma_process * torch.randn((N_t, m), device=self.device)
        measurement_noise = sigma_measure * torch.randn((N_t, m), device=self.device)

        # Allocate trajectory and velocity arrays
        X = torch.zeros((N_t, m), device=self.device)
        Xp = torch.zeros((N_t, m), device=self.device)

        # Initial condition with optional initial process noise
        x = x0 + sigma_process_0 * torch.randn((m,), device=self.device)
        X[0, :] = x  # Store initial state

        # Simulate over time (start from t = 1)
        for i in range(1, N_t):
            t = t_v[i]
            xp = self.eq_of_Motion(t, x) # Compute velocity
            x = x + xp * dt # Euler update
            x = x + process_noise[i] * torch.sqrt(torch.tensor(dt)) # Add process noise
            X[i, :] = x # Concatenate to trajectory (array of states)
            Xp[i, :] = xp # Array of velocities

        X_measure = X + measurement_noise
        
        return X.unsqueeze(-1), X_measure.unsqueeze(-1), Xp.unsqueeze(-1)  # Return true and noisy trajectory
    
