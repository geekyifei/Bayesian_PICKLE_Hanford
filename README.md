# Randomized Physics-informed Conditional Karhunen-Loève expansions (rPICKLE) for Large-scale High-dimensional Bayesian Inversion

Implementations of the "randomize-then-optimize" approach for sampling PICKLE posteriors in the Bayesian framework, as described in the paper:

> **Randomized physics-informed machine learning for uncertainty quantification in high-dimensional inverse problems**  
> *Journal of Computational Physics, 2024*  
> [[https://doi.org/10.1016/j.cma.2024.117670](https://doi.org/10.1016/j.cma.2024.117670)](https://doi.org/10.1016/j.jcp.2024.113395)

---

## Overview
This repo contains JAX-based implementations of randomized PINNs (rPINNs), which inject randomness into the PINN loss function to generate posterior samples for inverse problems governed by partial differential equations (PDEs). This method provides a scalable alternative to traditional Bayesian inference techniques for physics-constrained problems.

Key features:
- Efficient and scalable uncertainty quantification in PDE inverse problems with data assimilation
- Outperforms Hamiltonian Monte Carlo (HMC) and Stein Variational Gradient Descent (SVGD) methods, while they suffer from the curse of dimensionality and ill-conditioned posterior covariance structure. Additionally, rPINN is highly parallelizable.
- We propose a weighted-likelihood Bayesian PINN formulation to balance contributions from different terms (e.g., PDE, IC, BC residuals, measurements).

The following inverse PDE problems are included:
- 1D Linear Poisson Equation
- 1D Non-Linear Poisson Equation
- 2D Diffusion Equation with Spatially Varying Coefficient

If you find this code useful for your research, please cite the following paper:
```bibtex
@article{zong2024randomized,
  title={Randomized physics-informed machine learning for uncertainty quantification in high-dimensional inverse problems},
  author={Zong, Yifei and Barajas-Solano, David and Tartakovsky, Alexandre M},
  journal={Journal of Computational Physics},
  volume={519},
  pages={113395},
  year={2024},
  publisher={Elsevier}
}
```

Here is another paper that we have published using the randomized physics-informed conditional Karhunen-Loève expansion (rPICKLE) method for uncertainty quantification in high-dimensional PDE-constrained inverse problems, a real application on the Hanford Site subsurface problem.
```bibtex
@article{zong2024randomized,
  title={Randomized physics-informed machine learning for uncertainty quantification in high-dimensional inverse problems},
  author={Zong, Yifei and Barajas-Solano, David and Tartakovsky, Alexandre M},
  journal={Journal of Computational Physics},
  volume={519},
  pages={113395},
  year={2024},
  publisher={Elsevier}
}
```
