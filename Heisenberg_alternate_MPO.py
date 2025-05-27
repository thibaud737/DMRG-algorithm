# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 18:54:40 2025

@author: 13thi
"""

import numpy as np
from MPO import MPO
_ = np.newaxis

class Heisenberg_alternateMPO(MPO):
    def __init__(self, L, J=1.0, Jz=1.0, h=.0, delta=.0, delta_z=.0):
        """
        Initialize the MPO of the hamiltonian H = sum_i=1 to L-1 [ 
       (J / 2)*(1+(-1)^i*delta) * (S-_i * S+_i+1 + S+_i * S-_i+1) 
       + Jz*(1+(-1)^i*delta_z) * Sz[i] * Sz[i+1] 
       -h* Sz[i]]
        .

        Parameters:
        L  : Number of sites.
        J and Jz : Coupling constants.
        h : External magnetic field.
        δ: delta parameter of the hamiltonian above
        δ_z: delta_z parameter of the hamiltonian above
        
        """
        super().__init__(L, 2, 5)
        self.tensors = self._initialize_heisenberg_mpo(J, Jz, h, delta, delta_z)

    def _initialize_heisenberg_mpo(self, J, Jz, h, delta, delta_z):
        """
        recall that H = sum_i=1 to L-1 [ 
       (J / 2)*(1+(-1)^i*delta) * (S-_i * S+_i+1 + S+_i * S-_i+1) 
       + Jz*(1+(-1)^i*delta_z) * Sz[i] * Sz[i+1] 
       -h* Sz[i]]
        
        Parameters
        ----------
        L : length of the system
        J : xy-direction coupling constant. The default is 1.0.
        Jz : z-direction coupling constant. The default is 1.0.
        h : external magnetic field. The default is 0.0.
        δ: delta parameter of the hamiltonian above
        δ_z: delta_z parameter of the hamiltonian above
        
        Returns
        -------
        The MPO of the hamiltonian above

        """

        return [self.tensor_operator(i,J/2*(1+delta*(-1)**i),Jz*(1+delta_z*(-1)**i),h) 
                for i in range(self.length)]
        
        

    def tensor_operator(self, site, a,b,h):
        """
        Parameters
        ----------
        L : length of the system
        J : xy-direction coupling constant. The default is 1.0.
        Jz : z-direction coupling constant. The default is 1.0.
        h : external magnetic field. The default is 0.0.
        δ: delta parameter of the hamiltonian above
        δ_z: delta_z parameter of the hamiltonian above

        returns
        ------- 
        the tensor operator at site i of the Hamiltonian model.
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
        W[4, :, :, 1] = a * S_minus
        W[4, :, :, 2] = a * S_plus
        W[4, :, :, 3] = b * Sz
        W[4, :, :, 4] = I
        

        # Boundary tensor
        W_1=W[_,4,:,:,:]
        W_L  = W[:, :, :,0,_]

        if site == 0:
            return  W_1
        elif site == self.length - 1:
            return W_L
        else:
            return W
        
        
        
        
        