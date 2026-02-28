###############################################
Gradient Entropy Regularization
###############################################

Gradient Entropy Regularization (GER) is a new maximum-entropy (MaxEnt) approach for regularization in physics-informed neural networks (PINNs) for solving PDEs of varying types. GER enhances the performance of PINNs accross steady-state and time-dependent PDEs through embedding the predicted solution gradient into the residual. 

###############################################
Methodology
###############################################

GER is formulated as a term subtracted from the PINN loss function. Particularly, we perform MC sampling of gradient observations across the domain, then compute both mean and covariance (with optional parameters constorlling which dimensions are considered), form this we compute the Gaussian differential entropy (GDE) of the gradient field. We then subtract this term (with a weight) from the loss function to reflect that it is a maximized quantity. By intuition, the maximum-entropy approach should allow PINNs to attain more stable solutions, and prevent convergence on spurious approximations.

###############################################
Data
###############################################

The data utilized in this project consists of sample points from PDEs. For attaining ground truth values, we utilize analytical solutions (for simpler PDEs) and CFD-sourced data (see Yu et. al, 2021 and Raissi et. al, 2020). 
