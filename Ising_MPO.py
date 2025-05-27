# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 14:11:43 2025

@author: 13thi
"""


import numpy as np
from MPO import MPO
_ = np.newaxis

class IsingMPO(MPO):
    def __init__(self, L, J=1., h=0.):
        """
        Initialize the Heisenberg MPO with specific parameters.

        Parameters:
        L  : Number of sites.
        J and Jz : Coupling constants.
        h : External magnetic field.
        """
        super().__init__(L, 2, 3)
        self.tensors = self._initialize_Ising_mpo(J,h)

    def _initialize_Ising_mpo(self, J, h):
        """
        Creates an MPO representation of the 1D Ising model with transverse field.
        
        Parameters:
        -----------
        L : int
            Number of lattice sites.
        J : float, optional
            Coupling strength (default J=1).
        h : float, optional
            Transverse field strength (default h=0).
    
        Returns:
        --------
        mpo : list of numpy arrays
            List of L tensors representing the MPO.
        """
         
    
        # Pauli matrices
        Sz = np.array([[1, 0], [0, -1]], dtype=float)/2  # S^z
        I = np.eye(2)
    
        # MPO tensor (shape: D x d x d x D)
        W = np.zeros((3,2,2,3), dtype=float)
        
        W[0, :, :, 0] = I   # Identity term
        W[1, :, :, 0] = Sz  # S^z for interaction
        W[2, :, :, 0] = -h * Sz  # field term
        W[2, :, :, 1] = J * Sz  # J S^z term
        W[2, :, :, 2] = I   # Identity term
        
    
        # Boundary tensors
        W_1 = W[_,2, :, :, :]
        W_L = W[:, :, :, 0,_]
    

        return [W_1] + [W] * (self.length - 2) + [W_L]

