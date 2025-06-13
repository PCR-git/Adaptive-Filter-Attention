# Adaptive Filter Attention (AFA)

Adaptive Filter Attention (AFA) is a novel attention mechanism designed for filtering and forecasting in stochastic dynamical systems. Unlike standard attention mechanisms,
AFA integrates an explicit, learnable linear time-invariant (LTI) dynamics model into the attention computation, enabling it to filter noise and infer latent state transitions more robustly.

---

## Key Idea

AFA uses a learned LTI dynamics model to propagate latent states before computing similarity. Instead of comparing a query with keys directly, AFA compares a
predicted latent state (from the key, evolved to the query’s time point) to the query. The attention weights are then computed as a function of 
how consistent the transition is under the learned dynamics, effectively filtering out observations that do not align with plausible state evolution.

This structure introduces:
- Time-decayed attention with complex-valued exponential weights,
- Precision-aware weighting using stochastic noise models,
- Joint filtering and forecasting capabilities.

## Repository Structure

adaptive_filter_attention/
│
├── utils/                     # Complex-valued tensor utilities
│
├── dynamics/                 # Tools for simulating dynamical systems for testing
│
├── precision_attention/      # Core of AFA computations, including residuals, estimates, and propagated precision matrices
|                               estimates, and propagated precision matrices
│
├── model/                    # Neural network components
│
├── training/                 # Training utilities and loops
│
└── visualization/            # Tools to visualize attention and predictions
