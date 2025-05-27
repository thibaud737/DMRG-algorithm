# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 15:10:58 2025

@author: Thibaud Lesquereux
"""
import numpy as np
from MPO import MPO
_ = np.newaxis

class HeisenbergMPO(MPO):
    def __init__(self, L, J=1.0, Jz=1.0, h=0.0):
        """
        Initialize the Heisenberg MPO with specific parameters.

        Parameters:
        L  : Number of sites.
        J and Jz : Coupling constants.
        h : External magnetic field.
        """
        super().__init__(L, 2, 5)
        self.tensors = self._initialize_heisenberg_mpo(J, Jz, h)

    def _initialize_heisenberg_mpo(self, J, Jz, h):
        """
        Parameters
        ----------
        L : length of the system
        J : xy-direction coupling constant. The default is 1.0.
        Jz : z-direction coupling constant. The default is 1.0.
        h : external magnetic field. The default is 0.0.
        
        
        recall that H = sum_i=1 to L-1 [ 
       (J / 2) * (S-_i * S+_i+1 + S+_i * S-_i+1) 
       + Jz * Sz[i] * Sz[i+1] 
       -h* Sz[i]]
        
        Returns
        -------
        the MPO of the heisenberg model

        """
        # Pauli matrices
        I  = np.array([[1, 0], [0, 1]])
        Sz = np.array([[1, 0], [0, -1]]) / 2
        Sx = np.array([[0,1], [1,0]])/2
        S_plus = np.array([[0, 1], [0, 0]])
        S_minus = np.array([[0, 0], [1, 0]])
        W = np.zeros((5, 2, 2, 5)) #tensor W i.e. 5x5 matrix of 2x2 operators
    
        W[0, :, :, 0] = I
        W[1, :, :, 0] = S_plus 
        W[2, :, :, 0] = S_minus
        W[3, :, :, 0] = Sz
        W[4, :, :, 0] = -h * Sz
        W[4, :, :, 1] = J/2 * S_minus
        W[4, :, :, 2] = J/2 * S_plus
        W[4, :, :, 3] = Jz * Sz
        W[4, :, :, 4] = I
        

        # Boundary tensor
        W_1=W[_,4,:,:,:]
        W_L  = W[:, :, :,0,_]

        return [W_1] + [W] * (self.length - 2) + [W_L]

