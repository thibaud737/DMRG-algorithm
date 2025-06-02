# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:02:38 2025

@author: thibaud lesquereux
"""
import matplotlib.pyplot as plt
import numpy as np
from MPS import MPS
from MPO import MPO
from Heisenberg_MPO import HeisenbergMPO
from Ising_MPO import IsingMPO
from AKLT_MPO import AKLTMPO
from Heisenberg_alternate_MPO import Heisenberg_alternateMPO
from exact_diag_OBC import heisenberg_exact_diag as ED
import test_conv as test
import phase_diagram as pd
import Data as data

if __name__ == "__main__":

   #simple exemple of the use of the MPS class (without Data storage)
   """
   mps=MPS(L, d, D)
   mpo=HeisenbergMPO(L, J, Jz, h)
   mpo2=IsingMPO(L,Jz,h)
   mpo3= Heisenberg_alternateMPO(L, J, Jz, h, delta, delta_z)
    
   E,mps=mpo3.variational_ground_state(mps,2,1e-5)
   print(f"DMRG Ground Energy: {E:.6f}")
   print("var energy: ", mpo3.var_energy(mps))

   """
   #------------------------------------------------------------------------------------------------
   # the following exemple uses the the MPS's data stored (first stored the data before using it)
   #------------------------------------------------------------------------------------------------

   #data saving exemple
   """
   L, d, D = 250, 2, 100
   J, Jz, h, delta, delta_z = 1, 1, 0, 0.0, 0.0
   data.store_MPS(L, d, D, J, Jz, h, delta, delta_z, model="Heisenberg_alternate")
   mps=data.get_MPS(f"Heisenberg_alternate_(L,D){L,D}_(Jz,J,deltaz, delta,h){Jz,J,delta_z, delta,h}.txt")
   """

   #all convergences test
   """
   #test.test_conv_Chi_var(datas=False)
   #test.test_conv_energy_L(datas=False)
   #test.test_energy_per_site_vs_bond_dim(datas=False)
   #test.test_conv_to_Ising()
   #test.test_energy_DMRG_vs_ED(datas=False)
   """


   #phase diagram of the order parameters
   """
   L, d, D = 51, 2, 100
   J, h, delta, delta_z = 1, 0, 0.0, 0.0
   pd.FM_order_parameter(L, D, Jz, J, delta_z, delta, h,max_Jz=10, datas=True)
   pd.Neel_order_parameter(L, D, Jz, J, delta_z, delta, h,max_Jz=10, datas=True)

   delta_z_values= [-10,-6,-5,-4,-3,-2,-1,-0.5,0,0.5,1,2,3,4,5,6,10]
   pd.dimer_order_parameter(L, D, Jz, J, delta_z_values, delta, h,datas=True)
   """

   #central charge vs L calculation
   """
   L_values =[50,60,80,100,120,150]
   D = 200
   J,Jz, h, delta, delta_z = 1, 0, 0, 0.0, 0.0
   pd.central_charge_vs_L(L_values, D, Jz, J, delta_z, delta, h,datas=False)
   """


   #entropy calculation
   """ 
   L, d, D = 250, 2, 100
   J, Jz, h, delta, delta_z = 1, 1, 0, 0.0, 0.0
   # the MPS need to be stored before
   #data.store_MPS(L, d, D, J, Jz, h, delta, delta_z, model="Heisenberg_alternate")
   data.store_entropy(f"Heisenberg_alternate_(L,D){L,D}_(Jz,J,deltaz, delta,h){Jz,J,delta_z, delta,h}.txt")
   pd.entanglement_entropy_function(f"Heisenberg_alternate_(L,D){L,D}_(Jz,J,deltaz, delta,h){Jz,J,delta_z, delta,h}.txt", plot=True)
   """


   #correlation calculations
   """
   #Correlation function
   L, d, D = 150, 2, 100
   J, Jz, h, delta, delta_z = 1, 1, 0, 0.0, 0.0
   # the MPS need to be stored before
   #data.store_MPS(L, d, D, J, Jz, h, delta, delta_z, model="Heisenberg_alternate")
   data.store_correlation(f"Heisenberg_alternate_(L,D){L,D}_(Jz,J,deltaz, delta,h){Jz,J,delta_z, delta,h}.txt" ,coordinate='Z')
   pd.correlation_function(f"Heisenberg_alternate_(L,D){L,D}_(Jz,J,deltaz, delta,h){Jz,J,delta_z, delta,h}.txt",coordinate='Z', plot=True)
   """
   """
   #correlation length
   L, d, D = 150, 2, 100
   J, Jz, h, delta, delta_z = 1, 1, 0, 0.0, 0.0
   delta_z_values= [-5,-4,-3,-2,-1,0,1,2,3,4,5]
   pd.correlation_length_vs_deltaz(L, D, Jz, J, delta_z_values, delta, h,datas=True)
   """
   """
   #calculation of the critical point diagram
   L, d, D = 100, 2, 100
   J, Jz, h = 1, 1, 0
   delta_z_values = [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]
   delta_values =[-0.3,-0.2,-0.1,0,0.1,0.2,0.3]
   pd.critical_phase_Diagram(L, D, Jz, J, delta_z_values, delta_values, h,datas=True)
   """
   
      




    
    