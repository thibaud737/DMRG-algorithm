# -*- coding: utf-8 -*-
"""
edited by Thibaud Lesquereux.
"""
import numpy as np
import copy
import importlib
import opt_einsum as oe
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
_ = np.newaxis


class MPS:
    """Matrix Product State (MPS) representation."""
    
    def __init__(self, L, d=2, D=10):
        """
        Initialize a random MPS with given number of sites.
        
        Parameters:
         L : length of the system
         d : dimension of the state
         D : Bond dimension of the matrices

        """
        self.length = L
        self.state_dim = d
        self.bond_dim = D
        self.tensors =self._create_random_mps(L, d, D)
        
        
    def _create_random_mps(self, L, d, D):
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
        mps = [np.random.randn(1, d, D)]
        for i in range(1, L - 1):
            mps.append(np.random.randn(D, d, D))
        mps.append(np.random.randn(D, d, 1))
        return mps
    
    def _create_mps(self,L,d,D):
        """
        Parameters
        ----------
        L : length of the system
        d : dimension of the state
        D : Bond dimension of the matrices

        Returns
        -------
        mps :list of L  matrices (tensor)

        """
        mps = [np.zeros((1, d, D))]
        for i in range(1, L - 1):
            mps.append(np.zeros((D, d, D)))
        mps.append(np.zeros((D, d, 1)))
        return mps
    
    

    def __repr__(self):
        return f"MPS(length={self.length}, tensors={self.tensors})"


    
    def apply_to(self, mpo):
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
        if self.length != mpo.length and self.state_dim != mpo.state_dim:
            raise ValueError("MPS and MPO must have the same lenght and/or state dimension.")
            
        new_mps = MPS(self.length, self.state_dim, self.bond_dim)
        
        
        for i in range(self.length):
            new_mps.tensors[i] = oe.contract("mjn, ijkl->imkln", self.tensors[i].conj() , mpo.tensors[i]) # apply MPO tensor to MPS tensor
            new_mps.tensors[i]= new_mps.tensors[i].reshape(
                new_mps.tensors[i].shape[0]*new_mps.tensors[i].shape[1] ,
                new_mps.tensors[i].shape[2] ,
                new_mps.tensors[i].shape[3]*new_mps.tensors[i].shape[4]) #reshape the resulting tensor into a MPS tensor
        
        
        return new_mps


    
    def __mul__(self, other):
        """
        Parameters
        ----------
        other : either a MPO or a MPS
    
        Returns
        -------
        - Scalar product if other is an MPS instance ie <ψ|ϕ>
        - Transformed MPS in bra <ψ| if other is an MPO instance and multiplied by the MPO 
        """
        
        if isinstance(other, MPS):
            # Handle MPS * MPS case (scalar product)
            # Initialize left contraction ⟨ψ|φ⟩_0
            L = oe.contract('ijk, ijm -> km', self.tensors[0].conj(), other.tensors[0])
            for i in range(1,self.length):
                L = oe.contract('km, kij, mil ->jl ', L, self.tensors[i].conj(),other.tensors[i])
                                  
            return L[0,0]
    
        else:
            
            return self.apply_to(other)
    
        #else:
            #raise ValueError("MPS can only be multiplied with another MPS or MPO.")


    def left_normalize_QR(self,site):
        """
        apply left QR decomposition to the site
        """
        A = self.tensors[site]
        D_left, d, D_right = A.shape
        A = A.reshape(D_left * d, D_right)
        Q, R = np.linalg.qr(A)
        self.tensors[site] = Q.reshape(D_left, d, -1)
        if site + 1 < self.length:
            self.tensors[site + 1] = np.tensordot(R, self.tensors[site + 1], axes=(1, 0))
        
    def right_normalize_QR(self,site):
        """
        apply right QR decomposition to the site
        """
        A = self.tensors[site]
        D_left, d, D_right = A.shape
        A = A.reshape(D_left, d * D_right)
        Q, R = np.linalg.qr(A.T)
        self.tensors[site] = Q.T.reshape(-1, d, D_right)
        if site > 0:
            self.tensors[site - 1] = np.tensordot(self.tensors[site - 1], R.T, axes=(2, 0))


    def left_normalize(self, site):
        """
        Parameters
        ----------
        mps : MPS instance
        site : int, the site ℓ where the transfomation is apply
        
        returns
        -------
        the site-th matrix in its left canonical form
        """
        A=self.tensors
        B = A[site].reshape(-1, A[site].shape[2])  # reshape left and d index
        U, S, V_t = np.linalg.svd(B, full_matrices=False) #carry out SVD
        self.tensors[site] = U.reshape(A[site].shape[0], A[site].shape[1],-1)  #reshape  into tensor/ update mps
        S_diag = np.diag(S)
        
        #multiply S and V with the next matrix state 
        self.tensors[site+1]=oe.contract('ij, jk, klm -> ilm', S_diag, V_t, self.tensors[site+1]) 
        
    def right_normalize(self,site):
        """
        Parameters
        ----------
        mps : MPS instance
        site : int, the site ℓ where the transfomation is apply
        
        returns
        -------
        the site-th matrix in its right canonical form
        """
        A= self.tensors
        B = A[site].reshape(A[site].shape[0], -1)  # reshape right and d index
        U, S, V_t = np.linalg.svd(B, full_matrices=False) # carry out SVD
        self.tensors[site] = V_t.reshape(-1, A[site].shape[1], A[site].shape[2]) #reshape  into tensor/ update mps
        S_diag = np.diag(S)
        
        #multiply U and S with the previous matrix state
        self.tensors[site-1]= oe.contract('ijk , kl -> ijl', self.tensors[site-1], U @ S_diag)
                
       

    def left_canonical_form(self,site):
            """
            Parameters
            ----------
            mps : a list of L matrices
            site : int, the site ℓ where the transfomation ends
            
            (the matrix are in canonical form up to site-1 
             and the ℓ-th matrix is multiplied by the last S@V_t)
            
            Returns
            -------
            the MPS in its left canonical form
    
            """
            mps = copy.deepcopy(self)
            A= copy.deepcopy(self.tensors)
            for i in range(site):
                #print("li=",i)
                B = A[i].reshape(-1, A[i].shape[2])  # reshape left and d index
                U, S, V_t = np.linalg.svd(B, full_matrices=False) #carry out SVD
                mps.tensors[i] = U.reshape(A[i].shape[0], A[i].shape[1],-1)  #reshape  into tensor/ update mps
                S_diag = np.diag(S)
                
                #multiply S and V with the next matrix state
                A[i+1]=oe.contract('ij , jkl -> ikl', S_diag @ V_t, A[i+1])
                mps.tensors[i+1]=oe.contract('ij , jkl -> ikl', S_diag @ V_t, self.tensors[i+1]) 
                     
                
            return mps  
        
    
    
    def right_canonical_form(self,site):
        """
        Parameters
        ----------
        mps : a list of L matrices
        site : int, the site ℓ where the transfomation ends
        
        (the matrix are in canonical form L down to site+1 
         and the ℓ-th matrix is multiplied by the last U@S)
        
        Returns
        -------
        the MPS in its right canonical form
        """
        mps = copy.deepcopy(self) 
        A= copy.deepcopy(self.tensors)
        for i in range(self.length-1,site, -1):
            #print("ri=", i)
            
            B = A[i].reshape(A[i].shape[0], -1)  # reshape right and d index
            U, S, V_t = np.linalg.svd(B, full_matrices=False) # carry out SVD
            mps.tensors[i] = V_t.reshape(-1, A[i].shape[1], A[i].shape[2]) #reshape  into tensor/ update mps
            S_diag = np.diag(S)
            
            #multiply U and S with the previous matrix state
            A[i-1]= oe.contract('ijk , kl -> ijl', A[i-1], U @ S_diag)
            mps.tensors[i-1]= oe.contract('ijk , kl -> ijl', self.tensors[i-1], U @ S_diag)
            
        return mps
    

    
    

    def mixed_canonical_form(self, site):
        """
        Transforms an MPS into mixed canonical form 
    
        Parameters:
            -----------
            mps : a list of L matrices
            site : int the site ℓ 
            The site index around which the MPS is brought to mixed canonical form
    
        returns:
        --------
        The MPS in mixed canonical form.
        """
        # Apply left-canonical form up to the site
        mps=self.left_canonical_form(site)
        
        
        # Apply right-canonical form up to the site
        mps=mps.right_canonical_form(site)
        return mps

    
    def left_mps_contraction(self, site):
        """
        Compute the left contraction of the MPS up to site

        parameters:
        ----------
        site : int, the site ℓ where the contraction ends
        mps : MPS instance
        returns:
        --------
        L : the left contraction of the MPS up to site
        """
        L = oe.contract('ijk, ijm -> km', self.tensors[0].conj(), self.tensors[0])
        for i in range(1, site+1):
            L = oe.contract('km, kij, mil ->jl ', L, self.tensors[i].conj(), self.tensors[i])
            
        return L
    
    def right_mps_contraction(self, site):
        """
        Compute the right contraction of the MPS down to site
        parameters:
        ----------
        site : int, the site ℓ where the contraction ends
        mps : MPS instance
        returns:
        --------
        R : the right contraction of the MPS down to site
        """
        R = oe.contract('ijk, ljm -> il', self.tensors[-1].conj(), self.tensors[-1])
        for i in range(self.length-2, site-1, -1):
            R = oe.contract('jki, mkl, il ->jm ', self.tensors[i].conj(), self.tensors[i],R)
        return R
    
    


    def effective_N(self,site):
        """
        Compute the effective N tensor at site ℓ
        (used to solve generalized eigenvalue problem for DMRG. if MPS in mixed canonical form N=Id)
        Parameters:
        ----------
        site : int, the site ℓ where the effective N tensor is computed
        mps : MPS instance
        returns:
        --------
        N : the effective N tensor at site ℓ
        """
        # Create a Kronecker delta tensor
        delta = np.eye(self.state_dim)

        # Compute left and right MPS contractions
        psi_A = self.left_mps_contraction(site-1)
        psi_B = self.right_mps_contraction(site+1)
        
        # Compute the effective N tensor
        N = oe.contract('ij, kl, mn -> miknjl', psi_A, psi_B, delta)
        N = N.reshape(delta.shape[0]*psi_A.shape[0]*psi_B.shape[0],-1)

        if site==0:
             N = oe.contract('ij,kl -> kilj', psi_B, delta)
             N = N.reshape(delta.shape[0]*psi_B.shape[0],-1)
        if site==self.length-1:  
             N = oe.contract('ij, kl -> kilj', psi_A, delta)
             N = N.reshape(delta.shape[0]*psi_A.shape[0],-1)
        
        if N.all() != ((N + N.conj().T)/2).all():
            raise ValueError("N not hermitian")
            
        return N


   
    
    def mps_to_statevector(self):
        """
        Converts an MPS to a full state vector.

        Returns:
            psi : np.ndarray of shape (d^L,)
        """
        psi = self.tensors[0] 

        for  A in self.tensors[1:]:
            # Contract virtual bond
            psi = np.tensordot(psi, A, axes=[-1, 0]) 

        # Now shape is (1, d, d, ..., d, 1)
        psi = np.squeeze(psi)
        psi = psi.reshape(-1)  # flatten into (d^L,)
        return psi

    def full_density_matrix(self):
        """
        Compute the full density matrix ρ = |ψ⟩⟨ψ| (for a small MPS length).

        Returns:
            rho : (d^L, d^L) matrix
        """
        psi = self.mps_to_statevector()
        rho = np.outer(psi, psi.conj())
        return rho
    
    def reduced_density_matrix(self, site):
        """
        Compute the reduced density matrix rho at site ℓ of an MPS.

        Parameters:
            mps : MPS instance
            site : the site ℓ at which to compute the density matrix

        Returns:
            rho : np.ndarray of shape (d, d), the reduced density matrix at site
        """
        # Compute the left and right contractions
        L = self.left_mps_contraction(site-1)
        R= self.right_mps_contraction(site+1)
        if site==0:
            rho= oe.contract('ikl, jmn, ln ->km ',self.tensors[site], self.tensors[site].conj(), R)
        if site==self.length-1:
            rho= oe.contract('ij, ikl, jmn ->km ', L,self.tensors[site], self.tensors[site].conj())
        else:
            rho= oe.contract('ij,ikl,jmn, ln ->km ', L,self.tensors[site], self.tensors[site].conj(), R)  

        return rho.reshape(self.state_dim, self.state_dim)
        
        
        
    def entanglement_entropy(self, site):
        """
        Compute the entanglement entropy of the MPS at site ℓ using SVD

        Parameters:
        ----------
        site : int, the site ℓ where the entanglement entropy is computed
        mps : MPS instance
        returns:
        --------
        S : the entanglement entropy at site ℓ
        """
        
        mps = self.mixed_canonical_form(site)
        A = mps.tensors[site]
        B = A.reshape( -1,A.shape[2])  # reshape left and d index
        _, S,_ = np.linalg.svd(B, full_matrices=False)
        p = S**2
        p = p[p > 0]
        print(np.sum(p))
        return -np.sum(p * np.log(p))
        

    def entanglement_entropy_function(self, plot=True):
        """
        Compute the entanglement entropy as a function of ℓ
        (this function is overloaded in phase_diagram.py and use stored data)
        Parameters:
        ----------
        mps : MPS instance
        returns:
        --------
        S : the entanglement entropy as function of ℓ
        the fit parameters of the entanglement entropy
        """
        
    
        S = []
        L=self.length
        for i in range(L):
            S.append(self.entanglement_entropy(i))
    
        
        # Fit the Calabrese-Cardy formula
        def calabrese_cardy(x, c, A):
            return (c / 6) * np.log(L/np.pi*np.sin(np.pi*x/L)) + A

        # Perform the fit
        x = np.linspace(1e-1, L-1e-1 ,100)
        
        l=np.arange(1e-1, L-1e-1)
        S= np.array(S)

        # Reduce the data to the central region
        
        mid = len(S) // 2
        half = len(S) // 3
        S1 = S[mid - half: mid + half]
        l1 = l[mid - half: mid + half]
        
        popt, pcov = curve_fit(lambda x, c, A: calabrese_cardy(x, c, A), l1, S1, p0=[1, 0])

        if plot:
            # Plot the fit
            plt.figure(figsize=(10, 6))
            plt.plot(l, S, 'o', label="Entanglement entropy")
            plt.plot(x, calabrese_cardy(x, *popt), label=f"Fit: c={popt[0]:.2f}, A={popt[1]:.2f}", color="red")
            plt.xlabel("ℓ")
            plt.ylabel("Entanglement entropy")
            plt.title("Calabrese-Cardy Fit")
            plt.legend()
            plt.grid()
            plt.show()
        
        return S, popt[0], popt[1]
    
    

    def mean(self, O, site):
        """
        Compute the mean between an operator O and the MPS at site ℓ
        𝑂_ℓ = ⟨ψ|O|ψ⟩

        Parameters:
        ----------
        O : operator
        site : int, the site where the correlation function is computed
        mps : MPS instance
        returns:
        --------
        the mean between O and the MPS at site site
        """
        #check if the operator have the good dimension
        if O.shape[0] != self.state_dim:
            raise ValueError("O must have the same physical dimension as the MPS")
        
        #initialize the left contraction
        if site ==0:
            L = oe.contract('ijk,jl, mln-> kn', self.tensors[0].conj(), O, self.tensors[0])
        else:
             L = oe.contract('ijk, ijm -> km', self.tensors[0].conj(), self.tensors[0])
        
        #left contraction of the MPS up to length-1
        for i in range(1, self.length):
            if i == site:
                L = oe.contract('ij, ikl, km, jmn -> ln', L, self.tensors[i].conj(), O, self.tensors[i])
            else:
                L = oe.contract('km, kij, mil ->jl ', L, self.tensors[i].conj(), self.tensors[i])
                
        return L[0,0]



    def two_site_corr(self, O1, O2, site1, site2):
        """
        Compute the correlator ⟨ψ|O₁ ⊗ O₂|ψ⟩ between two operators O1 and O2 at site1 and site2

        Parameters:
        ----------
        O1 : operator 1
        O2 : operator 2
        site1 : int, the first site where O1 acts
        site2 : int, the second site where O2 acts
        mps : MPS instance
        returns:
        --------
        the 2-site correlation ⟨ψ|O₁ ⊗ O₂|ψ⟩ at site1 and site2 (respectively)
        """
        #check if site1 and site2 are different
        if site1 == site2:
            raise ValueError("site1 and site2 must be different")
        #check if the operator have the good dimension
        if O1.shape[0] != self.state_dim or O2.shape[0] != self.state_dim:
            raise ValueError("O1 and O2 must have the same physical dimension as the MPS")
    
        #initialize the left contraction
        
        if site1 ==0:
            L = oe.contract('ijk,jl, mln-> kn', self.tensors[0].conj(), O1, self.tensors[0])
        elif site2 ==0:
            L = oe.contract('ijk,jl, mln-> kn', self.tensors[0].conj(), O2, self.tensors[0])
        else:
             L = oe.contract('ijk, ijm -> km', self.tensors[0].conj(), self.tensors[0])
        #left contraction of the MPS up to length-1
        for i in range(1, self.length):
            if i == site1:
                L = oe.contract('ij, ikl, km, jmn -> ln', L, self.tensors[i].conj(), O1, self.tensors[i])
            elif i == site2:
                L = oe.contract('ij, ikl, km, jmn -> ln', L, self.tensors[i].conj(), O2, self.tensors[i])
            else:
                L = oe.contract('km, kij, mil ->jl ', L, self.tensors[i].conj(), self.tensors[i])
        return L[0,0]


    def correlation_function(self, O1, O2, plot=True):
        """
        Calculate the two-point correlation function ⟨O1_{L/2} O2_r⟩ - ⟨O1_{L/2}⟩⟨O2_r⟩ as a function of distance r.
        (used to compute the correlation in the critical phase i.e XX model. another general version is in the phase_diagram.py file)
        Parameters
        ----------
        O1 : Operator acting on site L/2.
        O2 : Operator acting on site r.
        plot : bool, If True, plot the correlation function and its fits.

        Returns
        -------
        corr_function : list of Correlation values for r = 1, ..., L//2.
        b : Power-law exponent from the fit a / r^b.
        xi : Correlation length from the fit a * exp(-r/xi) / r^b.
        """
        #compute the correlation function between O1 and O2
        corr_function=[]
        r= range(1,-(-self.length//2)+1)
        for i in range(self.length//2, self.length):
            corr_function.append( self.two_site_corr(O1, O2, self.length//2 -1, i)-self.mean(O1, self.length//2-1)*self.mean(O2, i))
           

        #plot the correlation function
        # Define the fitting function
        def fit_func_exp(r, a, b, c):
            return a * r**(-b) * np.exp(-r/c)

        def fit_func(r, a, b):
            return a * r**(-b)

        # Extract odd values of r and corresponding correlation values
        r= np.arange(1, len(corr_function)+1)
        r_odd = r[::2]
        corr_odd = corr_function[::2]
        
        #fit the bulk
        mid = len(r_odd) // 2
        half = len(r_odd) // 2
        r_odd1 = r_odd[mid - half: mid + half]
        corr_odd1 = corr_odd[mid - half: mid + half]

        # Perform the curve fitting
        popt, pcov = curve_fit(fit_func, r_odd1, corr_odd1, p0=(1, 1))

        # Perform the curve fitting for the exp function
        popt_exp, pcov_exp = curve_fit(fit_func_exp, r_odd1, corr_odd1, p0=(1, 1, 10))

        r_odd=np.abs(r_odd)
        corr_odd=np.abs(corr_odd)

        if plot:
            plt.figure(figsize=(10, 6))
            # Plot the fitted curves
            x= np.linspace(1, -(-self.length//2), 100)
            plt.plot(x, fit_func(x, *popt), label=f"Fit: $a/r^b$, $a={popt[0]:.3f}$, $b={popt[1]:.3f}$", linestyle="--")
            plt.plot(x, fit_func_exp(x, *popt_exp),label=f"Fit: $a \\cdot e^{{-x/\\xi}}/r^b,\\ a={popt_exp[0]:.3f},\\ b={popt_exp[1]:.3f},\\ \\xi={popt_exp[2]:.3f}$" , linestyle="-.")
            plt.plot(x,-1/(np.pi)**2*1/x**2, label="Theory: $-1/(\\pi^2\\cdot r^2)$", linestyle=":")
            # Plot the original data points
            plt.plot(r, corr_function, marker='o')
            plt.legend()
            plt.xlabel("distance r")
            plt.ylabel(r"$\langle S^z_{\frac{L}{2}} S^z_{\frac{L}{2}+r} \rangle$")
            plt.title("Correlation function ")
            plt.grid()
            plt.show()
            
            
            # Create a plot of log(corr_function) as a function of log(r) with only the fits
            plt.figure(figsize=(10, 6))
            x = np.logspace(np.log10(1), np.log10(-(-self.length // 2)), 100)

            plt.loglog(r_odd,np.abs(corr_odd), marker='o')
            plt.loglog(x,fit_func(x, np.abs(popt[0]), popt[1]), label=f"Fit: $a/r^b$, $a={popt[0]:.3f}$, $b={popt[1]:.3f}$",linestyle="--")
            plt.loglog(x,fit_func_exp(x, np.abs(popt_exp[0]), popt_exp[1], popt_exp[2]), label=f"Fit: $a \\cdot e^{{-x/\\xi}}/r^b,\\ a={popt_exp[0]:.3f},\\ b={popt_exp[1]:.3f},\\ \\xi={popt_exp[2]:.3f}$", linestyle="-.")
            plt.loglog(x, 1/(np.pi)**2*1/x**2, label="Theory: $-1/(\\pi^2 \\cdot r^2)$", linestyle=":")
            plt.legend()
            plt.xlabel("log(r)")
            plt.ylabel(r"$log\langle S^z_{\frac{L}{2}} S^z_{\frac{L}{2}+r} \rangle$")
            plt.title("Log-Log Correlation Function Fits")
            plt.grid()
            plt.show()
        
        return corr_function, popt_exp[1], popt_exp[2]




    def Neel_order_parameter(self):
        """
        Compute the Neel order parameter for the MPS
        (this is plot in the phase_diagram.py file)
        𝑁𝑂𝑃 = 1/L sum_ℓ (⟨S^z_odd⟩-⟨S^z_even⟩)
        Parameters:
        ----------
        mps : MPS instance
        returns:
        --------
        the Neel order parameter for the MPS i.e. sum_ℓ (⟨S^z_odd⟩-⟨S^z_even⟩)/L
        """
        # Define the Sz operator for a single site
        Sz = np.array([[1, 0], [0, -1]]) / 2 
        NOP=0     
        #calculate iteratively (⟨S^z_odd⟩-⟨S^z_even⟩)
        for i in range(self.length):
                NOP+=(-1)**(i+1)*self.mean(Sz, i)

        NOP=NOP/self.length
        return NOP
    

    def FM_order_parameter(self):
        """
        Compute the FM order parameter for the MPS
        (this is plot in the phase_diagram.py file)
        FMOP = 1/L sum_ℓ ⟨S^z_ℓ⟩
        Parameters:
        ----------
        mps : MPS instance
        returns:
        --------
        the FM order parameter for the MPS i.e. sum_ℓ ⟨S^z_ℓ⟩/L
        """
        # Define the Sz operator for a single site
        Sz = np.array([[1, 0], [0, -1]]) / 2 
        FMOP=0     
        #calculate iteratively (⟨S^z_i⟩)
        for i in range(self.length):
                FMOP+=self.mean(Sz, i)

        FMOP=FMOP/self.length
        return FMOP


#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------debugging function------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

    

    def debug_left_canonical(self,site):
        A=self.left_canonical_form(site)
        for i in range(site):
           
           I=np.einsum("ijk, ijn->kn", A.tensors[i], A.tensors[i])
           print(f"{i+1}",I)
        
        B=A.left_canonical_form(site)
        for i in range(self.length):
           if not np.allclose(A.tensors[i], B.tensors[i], atol=1e-10):
               print(f"Tensor at site {i+1} changed!")  
        
        
      
    def debug_right_canonical(self,site):
        A=self.right_canonical_form(site)
        for i in range(self.length-1,site-2,-1):
           I=np.einsum("ijk, ljk->il", A.tensors[i], A.tensors[i])
           print(f"{i+1}",I)   
          
        B=A.right_canonical_form(site)
        for i in range(self.length):
           if not np.allclose(A.tensors[i], B.tensors[i], atol=1e-10):
               print(f"Tensor at site {i+1} changed!")   
        
        
        
    def debug_mixed_canonical(self, site):
        A=self.mixed_canonical_form(site)
        for i in range(site):
           I=np.einsum("ijk, ijn->kn", A.tensors[i], A.tensors[i])
           print(f"{i+1}",I)
        for i in range(self.length-1,site-1,-1):
           I=np.einsum("ijk, ljk->il", A.tensors[i], A.tensors[i])
           print(f"{i+1}",I)  
        #print(self*self)
        
    def debug_right_normalize(self,site):
        self.right_normalize(site)
        I=np.einsum("ijk, ljk->il", self.tensors[site], self.tensors[site])
        print(f"{site}",I)  
        
        
    def debug_left_normalize(self,site):
        self.left_normalize(site)
        I=np.einsum("ijk, ijn->kn", self.tensors[site], self.tensors[site])
        print(f"{site}",I)
        
        
    def debug_N(self,site):
        mps=self.mixed_canonical_form(site)
        for i, tensor in enumerate(mps.tensors):
            print(f"Site {i+1} tensor shape:", tensor.shape)
        mps.effective_N(site)
    
    def debug_L_cont(self,site):
        mps=self.mixed_canonical_form(site)
        for i, tensor in enumerate(mps.tensors):
            print(f"Site {i+1} tensor shape:", tensor.shape)
        mps.left_mps_contraction(site)
    
    def debug_R_cont(self,site):
        mps=self.mixed_canonical_form(site)
        for i, tensor in enumerate(mps.tensors):
            print(f"Site {i+1} tensor shape:", tensor.shape)
        mps.right_mps_contraction(site)
        
        