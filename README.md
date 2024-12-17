# Enhanced sampling of robust molecular datasets with uncertainty-based collective variables

Code for performing uncertainty-driven enhanced sampling using the extended system adaptive biasing force (eABF) coupled with Gaussian-accelerated molecular dynamics (GaMD). The neural network (NN) architecture is based on [MACE](https://arxiv.org/abs/2206.07697). The software is based on the paper "[Enhanced sampling of robust molecular datasets with uncertainty-based collective variables]()", and was implemented by Aik Rui Tan and Johannes C. B. Dietschreit. 

The folders contain:
* [eabfgamd](): Classes, methods and utility functions to implement uncertainty-driven eABF-GaMD method using the MACE architecture.
* [scripts](): Wrapper around the classes and methods to run uncertainty-driven eABF-GaMD 
