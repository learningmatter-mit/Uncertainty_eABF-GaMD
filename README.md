# Enhanced sampling of robust molecular datasets with uncertainty-based collective variables

Code for performing uncertainty-driven enhanced sampling using the extended system adaptive biasing force (eABF) coupled with Gaussian-accelerated molecular dynamics (GaMD). The neural network (NN) architecture is based on [MACE](https://arxiv.org/abs/2206.07697). The software is based on the paper "[Enhanced sampling of robust molecular datasets with uncertainty-based collective variables]()", and was implemented by Aik Rui Tan and Johannes C. B. Dietschreit. 

The folders contain:
* [eabfgamd](): Classes, methods, and utility functions to implement the uncertainty-driven eABF-GaMD method using the MACE architecture.
  * `ala.py`: Contains classes and functions for handling alanine dipeptide simulations, including topology and energy scaling.
  * `calc.py`: Implements calculators for OpenMM and LAMMPS to perform molecular dynamics simulations and energy calculations.
  * `enhsamp.py`: Provides enhanced sampling methods using uncertainty-driven collective variables for molecular dynamics.
  * `ensemble.py`: Defines an ensemble of neural network models for uncertainty estimation and prediction.
  * `npt.py`: Implements the Berendsen NPT and Stochastic Cell Rescaling dynamics for pressure and temperature control.
  * `nvt.py`: Implements the Langevin dynamics for constant temperature molecular dynamics simulations.
  * `sampling.py`: Contains functions for various sampling strategies based on uncertainty and geometric criteria.
* [scripts](): Wrapper around the classes and methods to run uncertainty-driven eABF-GaMD.
* [configs](): Contains configuration files with parameters for running simulations and experiments.
