# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:03:07 2025

@author: thibaud Lesquereux
"""
import numpy as np
from MPS import MPS
from scipy.sparse.linalg import LinearOperator, eigsh
import matplotlib.pyplot as plt
import opt_einsum as oe
import time
import sys
_=np.newaxis


class MPO:
    """Matrix Product Operator (MPO) representation."""
    
    def __init__(self, L, d=2, D=3):
        """
        Initialize a random MPO with given number of sites.
        
        Parameters:
        L : length of the system
        d : dimension of the state 
        D : Bond dimension of the matrices
        """
        self.length = L
        self.state_dim = d
        self.bond_dim = D
        self.tensors = self._create_random_mpo(L,d,D)
        
        
    def _create_random_mpo(self,L, d, D):
        """
        Parameters
        ----------
        L : length of the system
        d : dimension of the state
        D : Bond dimension of the matrices

        Returns
        -------
        mps :list of L random matrices (tensor)

        """
        mpo = [np.random.randn(1, d, d, D)]
        for i in range(1, L - 1):
            mpo.append(np.random.randn(D, d, d, D))
        mpo.append(np.random.randn(D, d, d, 1))
        return mpo

    
    
    def apply_to(self, mps):
        """
        Parameters
        ----------
        take a MPS instance

        Raises
        ------
        ValueError if the MPO and the MPS don't have the same lenght
        

        Returns
        -------
        Applies MPO to MPS, returning a new transformed MPS.

        """
        if self.length != mps.length and self.state_dim != mps.state_dim:
            raise ValueError("MPS and MPO must have the same lenght and/or state dimension.")
            
        new_mps = MPS(mps.length, mps.state_dim, mps.bond_dim)
        
        
        for i in range(self.length):
            new_mps.tensors[i] = oe.contract("ijkl ,mkn->imjln", self.tensors[i] , mps.tensors[i]) # apply MPO tensor to MPS tensor
            new_mps.tensors[i]= new_mps.tensors[i].reshape(
                new_mps.tensors[i].shape[0]*new_mps.tensors[i].shape[1] ,
                new_mps.tensors[i].shape[2] ,
                new_mps.tensors[i].shape[3]*new_mps.tensors[i].shape[4]) #reshape the resulting tensor into a MPS tensor
        
        
        return new_mps  

    
    def multiply_mpo(self, other):
        """
        Parameters
        ----------
        other : a MPO instance

        Raises
        ------
        ValueError if the two MPO don't have the same lenght

        Returns
        -------
        new_mpo : the multiplication of two MPOs .

        """
        if self.length != other.length:
            raise ValueError("MPOs must have the same lenght.")
        
        #same shit as apply_to()
        new_tensors = []
        for i in range(self.length):
            new_tensor = oe.contract("ijkl ,mkno-> imjnlo", self.tensors[i], other.tensors[i])
            new_tensors.append(new_tensor.reshape(new_tensor.shape[0]*new_tensor.shape[1], new_tensor.shape[2], new_tensor.shape[3], -1))
        
        # Create the resulting MPO
        new_mpo = MPO(self.length, self.state_dim, self.bond_dim)
        new_mpo.tensors = new_tensors
        return new_mpo
    
    
    
    
    # Overload * for MPO and for MPS
    def __mul__(self, other):
        """
        Parameters
        ----------
        other : either a MPO or a MPS

        Raises
        ------
        ValueError if MPO is multiply with smt else than MPS or MPO

        Returns
        -------
        either a new MPO wich is the multiplication of the two previous MPO
        or returning a new transformed MPS after apllying the MPO to the MPS

        """
        if isinstance(other, MPS):
            # Apply MPO to MPS
            return self.apply_to(other)
        elif isinstance(other, MPO):
            # Multiply MPO with MPO (combine their tensors)
            return self.multiply_mpo(other)
        #else:
            #raise ValueError("MPO can only be multiplied with MPS or MPO.")
            
            


    def left_contraction(self, mps, site):
        """
        Compute the left contraction L_{a_{ℓ},a'_{ℓ},b_{ℓ}}
        
        Parameters:
        -----------
        mps : list of MPS tensors, 
        mpo : list of MPO tensors,
        site : int,  the site ℓ where the contraction ends
        works only for ℓ>1
        Returns:
        --------
        L : np.ndarray, left contraction tensor
        """
        # Initialize left contraction at the first site
        L = oe.contract('ijk, ljmn, omp -> knp', mps.tensors[0].conj(), self.tensors[0], mps.tensors[0])
        
        # Iteratively contract up to site ℓ-1
        for i in range(1, site+1):
            L = oe.contract('ijk ,ilm ,jlno ,knp->mop', L, mps.tensors[i].conj(), self.tensors[i], mps.tensors[i])
        
        return L

        
       

    def right_contraction(self, mps, site):
        """
        Compute the right contraction R_{a_{ℓ},b_{ℓ},a'_{ℓ}}
        
        Parameters:
        -----------
        mps : list of MPS tensors,
        mpo : list of MPO tensors, 
        site : int, the site ℓ where the contraction ends
    
        Returns:
        --------
        R : np.ndarray, right contraction tensor 
        """
        L = mps.length  # Total number of sites
        
        # Initialize right contraction at the last site
        R = oe.contract('ijk, ljmn, omp -> ilo ', mps.tensors[-1].conj(), self.tensors[-1], mps.tensors[-1])
        
        # Iteratively contract from site ℓ+1 down to site ℓ
        for i in range(L-2, site-1, -1):
            R = oe.contract('ijk ,ljmn , omp, knp->ilo', mps.tensors[i].conj(), self.tensors[i], mps.tensors[i], R)
            
        return R

     
  

    def effective_hamiltonian(self, mps, site):
        """
        calculate the effective Hmailtonian and optimze the MPS at site
        Parameters
        ----------
        mps : MPS

        Returns
        -------
        None.

        """
        start_time = time.time()

        #contraction of MPS and MPO
        R=self.right_contraction(mps, site+1)
        L=self.left_contraction(mps,site-1)

        def H_eff_matvec(v):
            """
            Parameters
            ----------
            v : vector to be multiplied by the effective Hamiltonian

            Returns
            -------
            result : H_eff @ v, the result of the multiplication
            """
            #reshape the input vector to match the tensor shape
            l,d,r=mps.tensors[site].shape
            v = v.reshape(d,l,r)
            v = v.transpose((1,0,2))
            v = v.reshape(l,d,r)

            # Perform the matrix-vector multiplication
            if site == self.length - 1:
                Hv = oe.contract('ijk,jlmn,kmp->li', L, self.tensors[site], v)
            
            elif site == 0:
                Hv = oe.contract('jlmn,kmp ,onp->lo', self.tensors[site], v, R)
           
            else:
                Hv = oe.contract('ijk,jlmn,kmp ,onp->lio', L, self.tensors[site], v, R)
            # Reshape the result back to a vector
            return Hv.flatten()
        
        # Solve the generalized eigenvalue problem
        shape = mps.tensors[site].shape
        size = shape[0]*shape[1]*shape[2]

        H_eff_op = LinearOperator((size, size), matvec=H_eff_matvec)

        mid_time = time.time()
        # Solve for the ground state
        _, eigenvectors = eigsh(H_eff_op, k=1, which='SA',v0=mps.tensors[site])
        ground_state_vector = eigenvectors[:, 0]
        
        
        # Reshape back to the MPS tensor format
        l,d,r=mps.tensors[site].shape
        ground_state_vector = ground_state_vector.reshape(d,l,r)
        ground_state_vector = ground_state_vector.transpose((1,0,2))
        mps.tensors[site]=ground_state_vector.reshape(mps.tensors[site].shape)
        
        end_time = time.time()
        #print(f"Time taken for effective Hamiltonian site {site+1}: {mid_time-start_time:.3f}, {end_time - mid_time:.3f} seconds")
        sys.stdout.write(f"\rSite: {site+1}")
    

    def right_sweep(self, mps):
        """
        Perform a right sweep in the DMRG algorithm.

        Parameters:
        -----------
        mps : The MPS instance.

        Returns:
        --------
        mps : The updated MPS after the right sweep.
            """
        
        for site in range(self.length-1):
             self.effective_hamiltonian(mps,site)
             mps.left_normalize(site)
             
        
        return mps 
           
        
                
    def left_sweep(self, mps):
        """
        Perform a left sweep in the DMRG algorithm.
        Parameters
        ----------
        mps : The MPS instance.

        Returns
        -------
        mps :The updated MPS after the left sweep.

        """
        
        for site in range(self.length-1, 0,-1):
                self.effective_hamiltonian(mps,site)
                mps.right_normalize(site)
               
        return mps 
    

        
    def variational_ground_state(self,mps, max_iter=10, tol=1e-6):
        """
        Parameters
        ----------
        mps : MPS
            The MPS instance.
        max_iter : int, optional
            Maximum number of iterations (default is 10).
        tol : float, optional
            Convergence tolerance (default is 1e-6).
    
        Returns
        -------
        enregy: the ground state energy
        mps : MPS
            The MPS of the ground state.
        """
        
        mps=mps.right_canonical_form(0)
        
        for i in range(max_iter):
            
            mps = self.right_sweep(mps)
            mps = self.left_sweep(mps)
            
            if self.convergence(mps, tol):
                print(f"\nA convergé en {i+1} sweep ")
                break
        print(f"\nvariance of the energy: {self.var_energy(mps)}") 
        return self.energy(mps), mps


    def energy(self, mps):
        """
        <ψ|H|ψ>/<ψ|ψ>
    
        Parameters:
        -----------
        mps : MPS instance."
        
        Returns
        -------
        energy of the mps w.r.t the MPO
        """
        E=oe.contract('ijk, ljmn, omp->knp', mps.tensors[0].conj(),self.tensors[0],mps.tensors[0])
        for i in range(1, self.length):
            E=oe.contract('ilo, ijk, ljmn, omp->knp',E, mps.tensors[i].conj(),self.tensors[i],mps.tensors[i])
        
        E=E.reshape(-1)[0]/(mps*mps)
        return E
    
    
    def var_energy(self,mps):
        """
        <ψ|H^2|ψ>/<ψ|ψ>-(<ψ|H|ψ>/<ψ|ψ>)^2
        
        Parameters
        ----------
        mps : MPS instance

        Returns
        -------
        variance of the energy w.r.t the MPO
        """
        #E =  self.energy(mps)
        #H2 = self*self
        #E2= (mps*H2*mps)
        #E2=E2/(mps*mps)
        E=oe.contract('ijk, ljmn, omp->knp', mps.tensors[0].conj(),self.tensors[0],mps.tensors[0])
        E2=oe.contract('ijk,ljmn, ompq,rps->knqs', mps.tensors[0].conj(),self.tensors[0].conj(),self.tensors[0],mps.tensors[0])
        for i in range(1, self.length):
            E2=oe.contract('ilor, ijk, ljmn, ompq, rps->knqs',E2, mps.tensors[i].conj(),self.tensors[i].conj(),self.tensors[i],mps.tensors[i])
            E=oe.contract('ilo, ijk, ljmn, omp->knp',E, mps.tensors[i].conj(),self.tensors[i],mps.tensors[i])
        E2=E2.reshape(-1)[0]/(mps*mps)
        E=E.reshape(-1)[0]/(mps*mps)
        return E2-E*E
        
        
         
    def __repr__(self):
        return f"MPO(length={self.length}, tensors={self.tensors})"



    def convergence(self, mps, epsilon=1e-1):
        """
        Check if the variance of the energy is less than epsilon.
    
        Parameters:
        -----------
        mps : MPS
            The matrix product state to evaluate.
        epsilon : float
            The tolerance for convergence.
    
        Returns:
        --------
        bool
            True if the variance is less than epsilon, False otherwise.
        float
            The variance of the energy.
        """
        
        # Check if the variance is less than epsilon
        if np.abs(self.var_energy(mps)) > epsilon:
            return False
        else:
            return True
    
