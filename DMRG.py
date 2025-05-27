import numpy as np
from scipy.sparse import linalg as sp_linalg
_=np.newaxis
import time

def main():
   d=2
   L=5
   D=100
   max_iter=1
   J=1
   Jz=0
   h=0
   start_time = time.time()
   """Initialize the MPS """
   mps = [np.ones((1, d, D))]
   for i in range(1, L - 1):
        mps.append(np.ones((D, d, D)))
   mps.append(np.ones((D, d, 1)))
   
   """ R canonicalize the MPS"""
   for i in range(L-1,0, -1):
            
            B = mps[i].reshape(mps[i].shape[0], -1)  # reshape right and d index
            U, S, V_t = np.linalg.svd(B, full_matrices=False) # carry out SVD
            mps[i] = V_t.reshape(-1, mps[i].shape[1], mps[i].shape[2]) #reshape  into tensor/ update mps
            S_diag = np.diag(S)
            
            #multiply U and S with the previous matrix state
            mps[i-1]= np.einsum('ijk , kl -> ijl', mps[i-1], U @ S_diag)
  
    
   
   
   
   "initialize the MPO"
   I  = np.array([[1, 0], [0, 1]])
   Sz = np.array([[1, 0], [0, -1]]) / 2
   #Sx = np.array([[0,1], [1,0]])/2
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
   W_1=W[4,:,:,:].reshape(1,2,2,5)
   W_L  = W[:, :, :,0,].reshape(5,2,2,1)
        
   #for i in range(5):
       #for matrix  in enumerate(W_L[i,:,:,0]):
           #print(matrix)
        
   mpo = [W_1] + [W] * (L - 2) + [W_L] # MPO list
    
   """sweep"""
   for i in range(max_iter):
        """right sweep"""
        for site in range(L-1):
                
            """L contraction"""
            L1 = np.einsum('ijk, ljmn, omp -> knp', mps[0].conj(), mpo[0], mps[0])
            # Iteratively contract up to site ℓ-1
            for i in range(1, site):
                L1 = np.einsum('ijk ,ilm ,jlno ,knp->mop', L1, mps[i].conj(), mpo[i], mps[i])
            
            """R contraction"""
            R = np.einsum('ijk, ljmn, omp -> ilo ', mps[-1].conj(), mpo[-1], mps[-1])
            # Iteratively contract from site ℓ+1 down to site ℓ
            for i in range(L-2, site, -1):
                R = np.einsum('ijk ,ljmn , omp, knp->ilo', mps[i].conj(), mpo[i], mps[i], R)
    
            """update tensor"""
            #effective Hamiltonian 
            H_eff=np.einsum('ijk, jlmn, onp ->liomkp', L1, mpo[site], R)
            H_eff=H_eff.reshape(H_eff.shape[0]*H_eff.shape[1]*H_eff.shape[2],-1)
            #H_eff.shape[3]*H_eff.shape[1]*H_eff.shape[5]
            #print("H2= ", H_eff.shape)
            
            
            if site== L-1:
                H_eff=np.einsum('ijk, jlmn ->limk', L1, mpo[site])
                H_eff=H_eff.reshape(H_eff.shape[0]*H_eff.shape[1],-1)
                
            if site==0:
                H_eff=np.einsum('ijkl,mln ->jmkn', mpo[site],R)
                H_eff=H_eff.reshape(H_eff.shape[0]*H_eff.shape[1],-1)
                #print("H1= ", H_eff.shape)
            
            #print("norm",np.linalg.norm(H_eff -H_eff.conj().T))
            # Solve the generalized eigenvalue problem
            #eigenvalues, eigenvectors =np.linalg.eigh(np.linalg.inv(N)@H_eff)
            eigenvalues, eigenvectors = np.linalg.eigh(H_eff)
            ground_state_index = np.argmin(eigenvalues)
            ground_state_vector = eigenvectors[:, ground_state_index]
    
            # Reshape back to the MPS tensor format
            l,d,r=mps[site].shape
            #ground_state_vector = ground_state_vector.reshape(l,r,d)
            #ground_state_vector = ground_state_vector.transpose((0,2,1))
            
            ground_state_vector = ground_state_vector.reshape(d,l,r)
            ground_state_vector = ground_state_vector.transpose((1,0,2))
            mps[site]=ground_state_vector.reshape(mps[site].shape)
    
            """L normalize the tensor"""
            B = mps[site].reshape(-1, mps[site].shape[2])  # reshape left and d index
            U, S, V_t = np.linalg.svd(B, full_matrices=False) #carry out SVD
            mps[site] = U.reshape(mps[site].shape[0], mps[site].shape[1],-1)  #reshape  into tensor/ update mps
            S_diag = np.diag(S)
            
            #multiply S and V with the next matrix state 
            mps[site+1]=np.einsum('ij, jk, klm -> ilm', S_diag, V_t, mps[site+1]) 
    
    
        
        """left sweep"""
        for site in range(L-1, 0,-1):
            """L contraction"""
            L1 = np.einsum('ijk, ljmn, omp -> knp', mps[0].conj(), mpo[0], mps[0])
            # Iteratively contract up to site ℓ-1
            for i in range(1, site):
                L1 = np.einsum('ijk ,ilm ,jlno ,knp->mop', L1, mps[i].conj(), mpo[i], mps[i])
            
            """R contraction"""
            R = np.einsum('ijk, ljmn, omp -> ilo ', mps[-1].conj(), mpo[-1], mps[-1])
            # Iteratively contract from site ℓ+1 down to site ℓ
            for i in range(L-2, site, -1):
                R = np.einsum('ijk ,ljmn , omp, knp->ilo', mps[i].conj(), mpo[i], mps[i], R)
    
            """update tensor"""
            #effective Hamiltonian 
            H_eff=np.einsum('ijk, jlmn, onp ->liomkp', L1, mpo[site], R)
            H_eff=H_eff.reshape(H_eff.shape[0]*H_eff.shape[1]*H_eff.shape[2],-1)
            #H_eff.shape[3]*H_eff.shape[1]*H_eff.shape[5]
            #print("H2= ", H_eff.shape)
            
            
            if site== L-1:
                H_eff=np.einsum('ijk, jlmn ->limk', L1, mpo[site])
                H_eff=H_eff.reshape(H_eff.shape[0]*H_eff.shape[1],-1)
                
            if site==0:
                H_eff=np.einsum('ijkl,mln ->jmkn', mpo[site],R)
                H_eff=H_eff.reshape(H_eff.shape[0]*H_eff.shape[1],-1)
                #print("H1= ", H_eff.shape)
                
        

            
            #print("norm",np.linalg.norm(H_eff -H_eff.conj().T))
            # Solve the generalized eigenvalue problem
            #eigenvalues, eigenvectors =np.linalg.eigh(np.linalg.inv(N)@H_eff)
            eigenvalues, eigenvectors = np.linalg.eigh(H_eff)
            ground_state_index = np.argmin(eigenvalues)
            ground_state_vector = eigenvectors[:, ground_state_index]
    
            # Reshape back to the MPS tensor format
            l,d,r=mps[site].shape
            #ground_state_vector = ground_state_vector.reshape(l,r,d)
            #ground_state_vector = ground_state_vector.transpose((0,2,1))
            
            ground_state_vector = ground_state_vector.reshape(d,l,r)
            ground_state_vector = ground_state_vector.transpose((1,0,2))
            mps[site]=ground_state_vector.reshape(mps[site].shape)
    
            """R normalize the tensor"""
            B = mps[site].reshape(mps[site].shape[0], -1)  # reshape right and d index
            U, S, V_t = np.linalg.svd(B, full_matrices=False) # carry out SVD
            mps[site] = V_t.reshape(-1, mps[site].shape[1], mps[site].shape[2]) #reshape  into tensor/ update mps
            S_diag = np.diag(S)
            
            #multiply U and S with the previous matrix state
            mps[site-1]= np.einsum('ijk , kl -> ijl', mps[site-1], U @ S_diag)
    

        
        
        """calculate energy"""
        E = np.einsum('ijk, ljmn, omp -> knp', mps[0].conj(), mpo[0], mps[0])
        # Iteratively contract up to site ℓ-1
        for i in range(1, L):
            E = np.einsum('ijk ,ilm ,jlno ,knp->mop', E, mps[i].conj(), mpo[i], mps[i])
            
        norm_mps = np.einsum('ijk, ijm -> km', mps[0].conj(), mps[0])
        for i in range(1, L):
            norm_mps = np.einsum('km, kij, mil ->jl ', norm_mps, mps[i].conj(), mps[i])
        norm_mps=norm_mps[0,0]
        print("mps*mps",norm_mps)
        E = E[0,0] / norm_mps
        print("Energy: ", E)

   end_time = time.time()
   print(f"Time taken: {end_time - start_time:.6f} seconds")    
        
        
main()       