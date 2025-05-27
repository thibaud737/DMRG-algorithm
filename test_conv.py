# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 19:33:32 2025

@author: 13thi
"""

import matplotlib.pyplot as plt
import numpy as np
from MPS import MPS
from MPO import MPO
from Heisenberg_MPO import HeisenbergMPO
from Ising_MPO import IsingMPO
from AKLT_MPO import AKLTMPO
from exact_diag_OBC import heisenberg_exact_diag as ED
import Data as data
from scipy.optimize import curve_fit
def test_conv_to_Ising():
    """
    Plot the energy as a function of J going to zero and compare
    the Heisenberg MPO, Ising MPO, and exact diagonalization energy.

    L_max ~ 15 if using ED
    """
    L, d, D = 10, 2, 100
    Jz, h = 1, 0  # Ising limit
    tolerance = 1e-6

    J_values = np.linspace(0, 0.1, 10)  
    energies_heisenberg = []
    energies_ising = []
    energies_exact = []

    for J in J_values:
        # Heisenberg MPO
        mps_heisenberg = MPS(L, d, D)
        mpo_heisenberg = HeisenbergMPO(L, J, Jz, h)
        energy_heisenberg, _ = mpo_heisenberg.variational_ground_state(mps_heisenberg, 10, tolerance)
        energies_heisenberg.append(energy_heisenberg)

        # Ising MPO
        mps_ising = MPS(L, d, D)
        mpo_ising = IsingMPO(L, Jz, h)
        energy_ising, _ = mpo_ising.variational_ground_state(mps_ising, 10, tolerance)
        energies_ising.append(energy_ising)
        
        
        # Exact diagonalization
        eigvals, _ = ED(L, J, Jz, h)
        exact_energy = eigvals[0]
        energies_exact.append(exact_energy)
        
    # Plot the energies
    plt.figure(figsize=(8, 6))

    plt.plot(J_values, energies_heisenberg, label="Heisenberg MPO", marker='o')
    plt.plot(J_values, energies_ising, label="Ising MPO", marker='x')
    plt.plot(J_values, energies_exact, label="Exact Diagonalization", linestyle='--')
    plt.xlabel("J")
    plt.ylabel("Energy")
    plt.title("Energy vs J (J -> 0)")
    plt.legend()
    plt.grid()
    plt.show()


def test_conv_Chi_var(datas=True):
    """
    Test the convergence of the energy and variance with respect to the bond dimension (Chi)
    """
    L, d = 100, 2
    J, Jz, h = 1, 0, 0
    max_D = 300


    energies = []
    var = []
    for D in range(100, max_D + 1, 100):
        #if the data is not already stored, store it
        if datas:
            data.store_MPS(L, d, D, J, Jz, h, 0.0, 0.0, model="Heisenberg")
        mps = data.get_MPS(f"Heisenberg_(L,D){L,D}_(Jz,J,deltaz, delta,h){Jz, J,0.0, 0.0,h}.txt")
        mpo = HeisenbergMPO(L, J, Jz, h)
        energy = mpo.energy(mps)
        var.append(mpo.var_energy(mps))
        energies.append(energy)

    # plot the convergence of the energy variance
    plt.plot(range(100, max_D + 1, 100), var,marker='o', label="variance")
    plt.xlabel(r'$\chi$')
    plt.ylabel("Enregy variance")
    plt.legend()
    plt.show()

def test_conv_energy_L(datas=True):
    """
    test the convergence of the energy per site with respect to the system size
    """
    d, D = 2, 100
    J, Jz, h = -1, -1, 0
    max_L = 100


    lengths = range(10, max_L + 1, 10)
    energies_per_L = []

    for L in lengths:
        #if the data is not already stored, store it
        if datas:
            data.store_MPS(L, d, D, J, Jz, h, 0.0, 0.0, model="Heisenberg")
        mps = data.get_MPS(f"Heisenberg_(L,D){L,D}_(Jz,J,deltaz, delta,h){Jz, J,0.0, 0.0,h}.txt")
        mpo = HeisenbergMPO(L, J, Jz, h)
        energy = mpo.energy(mps)
        energies_per_L.append(energy / L)
    # Fit energies_per_L to a function of the form a + b / L^n

    def fit_func(L, a, b, n):
        return a + b / (L ** n)

    popt, pcov = curve_fit(fit_func, lengths, energies_per_L, p0=[-1.0, 1.0, 1.0])
    a_fit, b_fit, n_fit = popt
    fit_energies = fit_func(np.array(lengths), *popt)
    x= np.linspace(min(lengths), max(lengths), 100)
    print(f"Fit parameters: a = {a_fit}, b = {b_fit}, n = {n_fit}")
    #plot the convergence of energy/L
    plt.plot(lengths, energies_per_L, marker='o', label="Energy/L")
    plt.plot(x, fit_func(x,a_fit, b_fit, n_fit), label=f"Fit: a + b/L^n, n={n_fit:.2f}", linestyle='--', color="red")
    plt.xlabel("System size (L)")
    plt.ylabel("Energy per site (E/L)")
    plt.legend()
    plt.show()

def test_energy_per_site_vs_bond_dim(datas=True):
    """
    Test the convergence of the energy per site with respect to the chi dimension
    """
    L, d = 50, 2
    J, Jz, h = -1, -1, 0
    max_D = 300

    energies_per_site = []
    bond_dims = range(100, max_D + 1, 50)

    for D in bond_dims:
        #if the data is not already stored, store it
        if datas:
            data.store_MPS(L, d, D, J, Jz, h, 0.0, 0.0, model="Heisenberg")
        mps = data.get_MPS(f"Heisenberg_(L,D){L,D}_(Jz,J,deltaz, delta,h){Jz, J,0.0, 0.0,h}.txt")
        mpo = HeisenbergMPO(L, J, Jz, h)
        energy = mpo.energy(mps)
        energies_per_site.append(energy / L)

    # plot energy per site vs bond dimension
    plt.plot(bond_dims, energies_per_site, marker='o', label="Energy/L")
    plt.xlabel(r'$\chi$')
    plt.ylabel("Energy per site (E/L)")
    plt.legend()
    plt.show()

def test_energy_DMRG_vs_ED(datas=True):
    """
    Test the energy of the DMRG method against exact diagonalization
    """
    L,d =  15, 2
    J, Jz, h = -1,-1, 0
    delta, delta_z = 0.0, 0.0
    bond_dims =[10,50,100,150]
    energy=[]
    eigvals, _ = ED(L, J, Jz, h)
    energy_ED =eigvals[0]
    print("ED energy: ", energy_ED)
    for D in bond_dims:
        #if the data is not already stored, store it
        if datas:
            data.store_MPS(L, d, D, J, Jz, h, 0.0, 0.0, model="Heisenberg_alternate")
        mps = data.get_MPS(f"Heisenberg_alternate_(L,D){L,D}_(Jz,J,deltaz, delta,h){Jz, J,0.0, 0.0,h}.txt")
        mpo = HeisenbergMPO(L, J, Jz, h)
        energy_DMRG = mpo.energy(mps)
        energy.append(energy_DMRG - energy_ED)

    #plot the difference for different value of L
    plt.plot(bond_dims, energy, marker='o', label="Energy Difference (DMRG - ED)")
    plt.xlabel(r"$\chi$")
    plt.ylabel("Energy Difference (DMRG - ED)")
    plt.title(r"Energy Difference vs $\chi$")
    plt.legend()
    plt.grid()
    plt.show()


def test_correlation(coordinate='Z'):
    """
    Test the correlation function of the Heisenberg model
    """
    L, d,D = 100, 2, 100
    J, Jz, h = 1, 0, 0
    
    tolerance = 1e-6

    mps = MPS(L, d, D)
    mpo = HeisenbergMPO(L, J, Jz, h)
    energy, mps = mpo.variational_ground_state(mps, 10, tolerance)

    #spin operator
    if coordinate == 'X':
        S = np.array([[0, 1], [1, 0]])/2
    elif coordinate == 'Y':
        S = np.array([[0, -1j], [1j, 0]])/2
    elif coordinate == 'Z':
        S = np.array([[1, 0], [0, -1]])/2
    else:
        raise ValueError("Invalid coordinate. Choose 'X', 'Y', or 'Z'.")
    # Compute the correlation function
    corr = mps.correlation_function(S,S)
