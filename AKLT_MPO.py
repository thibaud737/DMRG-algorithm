# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 18:28:30 2025

@author: 13thi
"""

import numpy as np
from MPO import MPO
_ = np.newaxis

class AKLTMPO(MPO):
    def __init__(self, L):
        """
        Initialize the AKLT MPO with specific parameters.

        Parameters:
        L  : Number of sites.
        
        """
        super().__init__(L, 3, 5)
        self.tensors = self._initialize_AKLT_mpo()

    def _initialize_AKLT_mpo(self):
        # Define spin-1 operators
        I = np.eye(3)  # 3x3 identity matrix
        Sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])  # S^z
        Sp = np.array([[0, np.sqrt(2), 0], [0, 0, np.sqrt(2)], [0, 0, 0]])  # S^+
        Sm = Sp.T  # S^- is transpose of S^+
        
        # Initialize MPO tensor W with zeros
        W = np.zeros((5,3,3,5))
        
        # Fill in the MPO tensor elements
        W[0, :, :, 0] = I   # Identity (left boundary)
        W[4, :, :, 0] = I   # Identity (right boundary)
        W[4, :, :, 1] = (2/3) * Sz
        W[4, :, :, 2] = (np.sqrt(2)/3) * Sm
        W[4, :, :, 3] = (np.sqrt(2)/3) * Sp
        W[4, :, :, 4] = (1/3) * I
            
    
        # Boundary tensors
        W_1 = W[_,4, :, :, :]
        W_L = W[:, :, :, 0,_]
        
        

        return [W_1] + [W] * (self.length - 2) + [W_L]

