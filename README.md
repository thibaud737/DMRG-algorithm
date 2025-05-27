# DMRG-algorithm
DMRG algorithm using python for a Master project by Thibaud Lesquereux

## Overview

This project provides a set of tools for simulating and analyzing 1D quantum spin chains using Matrix Product States (MPS) and Matrix Product Operators (MPO).
It includes implementations for various spin models (Heisenberg, Ising, AKLT, and their alternated versions), as well as utilities for data storage, convergence testing, phase diagram calculation, entanglement entropy, and correlation functions.

## Main Features

- **MPS/MPO Classes:** Efficient representation and manipulation of quantum states and operators.
- **Model Implementations:** Heisenberg, Ising, AKLT, and alternated Heisenberg models.
- **Ground State Search:** Variational ground state search using DMRG-like algorithms.
- **Data Storage:** Save and load MPS data for reproducibility and further analysis.
- **Convergence Tests:** Tools to test convergence with respect to bond dimension, system size, and other parameters.
- **Phase Diagrams:** Compute and plot order parameters and critical phase diagrams.
- **Entanglement Entropy:** Calculate and visualize entanglement entropy profiles.
- **Correlation Functions:** Compute and analyze spin correlation functions and correlation lengths.
- **Exact Diagonalization:** Compare DMRG results with exact diagonalization for small systems.

## File Structure

- `main.py`: Main script with example usages and workflow.
- `MPS.py`: Matrix Product State class.
- `MPO.py`: Matrix Product Operator class.
- `Heisenberg_MPO.py`, `Ising_MPO.py`, `AKLT_MPO.py`, `Heisenberg_alternate_MPO.py`: Model-specific MPOs.
- `exact_diag_OBC.py`: Exact diagonalization routines.
- `test_conv.py`: Convergence test utilities.
- `phase_diagram.py`: Phase diagram and order parameter calculations.
- `Data.py`: Data storage and retrieval utilities.

## Usage

1. **Install dependencies:**
   - Python 3.x
   - `numpy`
   - `matplotlib`

2. **Run the main script:**
   ```
   python main.py
   ```
   Uncomment the relevant code blocks in `main.py` to perform the desired calculations (ground state search, entropy, correlations, etc.).

3. **Customize parameters:**  
   Edit the parameters (system size `L`, bond dimension `D`, couplings, etc.) in `main.py` as needed for your study.

## Example
See main.py for an example of how to create an MPS, define an MPO, and perform a variational ground state search. The script also includes examples for data storage, convergence testing, and phase diagram calculations.
## Notes
- Ensure that the `Data.py` module is properly configured to store and retrieve data files.
- The convergence tests and phase diagrams can take significant computational time depending on the system size and bond dimension.
