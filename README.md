[![PyTorch](https://pytorch.org)](https://docs.pytorch.org/docs/stable/index.html?_gl=1*n8m3fr*_up*MQ..*_ga*MTI3MjUwMjk3OS4xNzcyMjU4NTIw*_ga_469Y0W5V62*czE3NzIyNTg1MTkkbzEkZzAkdDE3NzIyNTg1MTkkajYwJGwwJGgw)
# Gradient Entropy Regularization

Gradient Entropy Regularization (GER) is a novel maximum-entropy (MaxEnt) regularization approach for physics-informed neural networks (PINNs) applied to partial differential equations (PDEs) of varying types. GER improves PINN performance across both steady-state and time-dependent PDEs by embedding the predicted solution gradient directly into the residual formulation.

## Methodology

GER is introduced as an additional term subtracted from the standard PINN loss function. Specifically, we perform Monte Carlo sampling of gradient observations across the domain and compute the corresponding mean and covariance, with optional parameters controlling which spatial dimensions are considered. From these statistics, we compute the Gaussian Differential Entropy (GDE) of the gradient field. This entropy term is then subtracted from the loss function with an associated weighting factor, reflecting its maximization objective. Intuitively, this maximum-entropy formulation promotes solution stability and mitigates convergence toward spurious or physically inconsistent approximations.

## Data

The data used in this project consists of sampled points from PDE solution domains. Ground truth values are obtained using analytical solutions for simpler PDEs and computational fluid dynamics (CFD)–sourced datasets for more complex cases (see Yu et al., 2021 and Raissi et al., 2020).
