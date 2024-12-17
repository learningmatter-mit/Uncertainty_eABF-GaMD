# Enhanced sampling of robust molecular datasets with uncertainty-based collective variables

Code for performing uncertainty-driven enhanced sampling using the extended system adaptive biasing force (eABF) coupled with Gaussian-accelerated molecular dynamics (GaMD). The neural network (NN) architecture is based on [MACE](https://arxiv.org/abs/2206.07697). The software is based on the paper "[Enhanced sampling of robust molecular datasets with uncertainty-based collective variables]()", and was implemented by Aik Rui Tan and Johannes C. B. Dietschreit. 

The folders contain:
* [eabfgamd](): Classes, methods, and utility functions to implement the uncertainty-driven eABF-GaMD method using the MACE architecture.
  * `ala.py`: Provides classes and functions for alanine dipeptide simulations, including topology and energy scaling.
  * `calc.py`: Contains calculators for OpenMM and LAMMPS to perform molecular dynamics simulations and energy calculations.
  * `colvars.py`: Defines collective variables for enhanced sampling, including uncertainty and dihedral angles.
  * `dihedrals.py`: Provides functions to manipulate dihedral angles in molecular structures.
  * `enhsamp.py`: Implements enhanced sampling methods using uncertainty-driven collective variables for molecular dynamics.
  * `ensemble.py`: Manages an ensemble of neural network models for uncertainty estimation and prediction.
  * `mace_calculators.py`: Implements MACE-based calculators for molecular dynamics simulations with biasing techniques.
  * `mdlogger.py`: Provides logging functionality for molecular dynamics simulations.
  * `misc.py`: Contains miscellaneous utility functions for file handling, logging, and data conversion.
  * `npt.py`: Implements the Berendsen NPT and Stochastic Cell Rescaling dynamics for pressure and temperature control.
  * `nvt.py`: Provides Langevin dynamics for constant temperature molecular dynamics simulations.
  * `pmf.py`: Contains functions to compute potential of mean force (PMF) from simulation data.
  * `prediction.py`: Provides functions for making predictions and calculating errors using trained models.
  * `sampling.py`: Offers functions for various sampling strategies based on uncertainty and geometric criteria.
  * `saving.py`: Handles saving and combining datasets, ensuring data integrity and format consistency.
  * `silica.py`: Contains paths and energy scaling functions specific to silica simulations.
  * `train.py`: Provides functions to configure and manage training of neural network models.
  * `umbrella_sampling.py`: Implements umbrella sampling techniques using MACE calculators for enhanced sampling.
  * `uncertainty.py`: Defines classes for different uncertainty estimation methods in molecular simulations.
* [scripts](): Wrapper around the classes and methods to run uncertainty-driven eABF-GaMD.
* [configs](): Contains configuration files with parameters for running simulations and experiments.
