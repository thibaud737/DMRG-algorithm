# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 14:57:18 2025

@author: 13thi
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
def heisenberg_exact_diag(L, J, Jz, h):
    """
    Perform exact diagonalization of the Heisenberg model with open boundaries and external magnetic field,
    without incorporating magnetization conservation (M_z symmetry) and any others.
    
    Parameters:
        L  : int   - Number of sites
        J  : float - Coupling constant for XY interaction
        Jz : float - Coupling constant for Z interaction
        h  : float - External magnetic field strength
    
    Returns:
        eigvals  : ndarray - Eigenvalues of the Hamiltonian
        eigvecs  : ndarray - Eigenvectors of the Hamiltonian
    """
    dim = 2**L  # Hilbert space dimension
    H = sp.lil_matrix((dim, dim), dtype=np.float64)  # Sparse Hamiltonian

    def spin_op(i, op):
        """Constructs spin operators S^op (op in {x,y,z}) at site i."""
        I = sp.eye(2**i)
        X = sp.csr_matrix([[0, 1], [1, 0]]) / 2
        Y = sp.csr_matrix([[0, -1j], [1j, 0]]) / 2
        Z = sp.csr_matrix([[1, 0], [0, -1]]) / 2
        op_dict = {'x': X, 'y': Y, 'z': Z}
        return sp.kron(I, sp.kron(op_dict[op], sp.eye(2**(L-i-1))), format='csr')

    for i in range(L - 1):  # Open boundary conditions
        Sx_i = spin_op(i, 'x')
        Sx_j = spin_op(i + 1, 'x')
        Sy_i = spin_op(i, 'y')
        Sy_j = spin_op(i + 1, 'y')
        Sz_i = spin_op(i, 'z')
        Sz_j = spin_op(i + 1, 'z')
        
        H += J * (Sx_i @ Sx_j + Sy_i @ Sy_j) + Jz * (Sz_i @ Sz_j)

    # Add external magnetic field term -h * S^z
    for i in range(L):
        Sz_i = spin_op(i, 'z')
        H += -h * Sz_i

    H = H.tocsr()
    eigvals, eigvecs = spla.eigsh(H, k=1, which='SA')  # Compute lowest eigenvalues
    return eigvals, eigvecs
"""
# Example usage
L = 15  # Number of sites
J = -1
Jz = -1
h = 0.  # External magnetic field strength
eigvals, eigvecs = heisenberg_exact_diag(L, J, Jz, h)
print(f"Ground Energy: {eigvals[0]:.6f}")
"""

