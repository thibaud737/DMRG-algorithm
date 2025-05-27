# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 22:21:29 2025

@author: thibaud lesquereux
"""

import matplotlib.pyplot as plt
import numpy as np
from MPS import MPS
from Heisenberg_alternate_MPO import Heisenberg_alternateMPO
import Data as data
from scipy.optimize import curve_fit

def FM_order_parameter(L, D, J, delta_z, delta, h,max_Jz=10, datas=True):
    """
    Compute the FM order parameter as a function of Jz
    Parameters:
    ----------
    max_Jz : maximum value of Jz
    L_values : list of number of sites
    D : bond dimension
    J and Jz : Coupling constants.
    h : External magnetic field.
    delta_z ,delta : parameter of the Hamiltonian
    datas : If True, store the data to a file
    Returns:
    -------
    None
    """

    
    #global look at the FM order parameter
    N = 2*max_Jz+1
    Jz = np.linspace(-max_Jz, max_Jz, N)
    mps=MPS(L, 2, D)
    FMOP=[]
    for i in range(len(Jz)):
        if datas:
                data.store_data(L, 2, D, J, Jz[i], h, delta, delta_z, model="Heisenberg_alternate")
        mps=data.get_data(f"Heisenberg_alternate_(L,D){L,D}_(Jz,J,deltaz, delta,h){Jz[i],J,delta_z, delta,h}.txt")
        FMOP.append(np.abs(mps.FM_order_parameter()))
    
    #plot the FM order parameter
    plt.figure(figsize=(8, 6))
    plt.plot(Jz / J, FMOP, marker='o')
    plt.xlabel('Jz / J')
    plt.ylabel('FM Order Parameter')
    plt.title('FM Order Parameter vs Jz / J')
    plt.grid(True)
    plt.show()
    

def Neel_order_parameter(L, D, J, delta_z, delta, h,max_Jz=10, datas=True):
    """
    Compute the Néel order parameter as a function of Jz
    Parameters:
    ----------
    max_Jz : maximum value of Jz
    L_values : list of number of sites
    D : bond dimension
    J and Jz : Coupling constants.
    h : External magnetic field.
    delta_z ,delta : parameter of the Hamiltonian
    datas : If True, store the data to a file
    Returns:
    -------
    None
    """
    #global look at the Neel order parameter
    N =2*max_Jz+1
    Jz = np.linspace(-max_Jz, max_Jz, N)
    mps=MPS(L, 2, D)
    NOP=[]
    
    for i in range(len(Jz)):
        if datas:
                data.store_data(L, 2, D, J, Jz[i], h, delta, delta_z, model="Heisenberg_alternate")
        mps=data.get_data(f"Heisenberg_alternate_(L,D){L,D}_(Jz,J,deltaz, delta,h){Jz[i],J,delta_z, delta,h}.txt")
        NOP.append(np.abs(mps.Neel_order_parameter()))


    # look closer at the phase transition
    Jz_ = np.linspace(1, 2, max_Jz)
    NOP2=[]
    for i in range(len(Jz_)):
        data.store_data(L, d, D, J, Jz_[i], h, delta, delta_z, model="Heisenberg_alternate")
        mps=data.get_data(f"Heisenberg_alternate_(L,D){L,D}_(Jz,J,deltaz, delta,h){Jz_[i],J,delta_z, delta,h}.txt")
        NOP2.append(np.abs(mps.Neel_order_parameter()))

    # Plot the Neel order parameter
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(Jz / J, NOP, marker='o')
    plt.xlabel('Jz / J')
    plt.ylabel('Neel Order Parameter')
    plt.title('Neel Order Parameter vs Jz / J')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(Jz_ / J, NOP2, marker='o')
    plt.xlabel('Jz / J')
    plt.ylabel('Neel Order Parameter')
    plt.title('Neel Order Parameter vs Jz / J (zoomed in)')
    plt.grid(True)
    plt.show()
    
def entanglement_entropy_function(filename, plot=True):
        """
        Compute the entanglement entropy as a function of ℓ

        Parameters:
        ----------
        mps : MPS instance
        returns:
        --------
        S : the entanglement entropy as function of ℓ
        the fit parameters of the entanglement entropy
        """
        
        S = data.get_entropy(filename)
        L = len(S)
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
        
        return popt[0], popt[1]


   

def central_charge_vs_L(L_values, D, Jz, J, delta_z, delta, h, datas=True):
    """
    Compute the central charge as a function of L
    Parameters:
    ----------
    L_values : list of number of sites
    D : bond dimension
    J and Jz : Coupling constants.
    h : External magnetic field.
    delta_z ,delta : parameter of the Hamiltonian
    datas : If True, store the data to a file
    Returns:
    -------
    None
    """

    if datas:
        # Store the entropy to a file.
        for L in L_values:
                data.store_entropy(f"Heisenberg_alternate_(L,D){L,D}_(Jz,J,deltaz, delta,h){Jz,J,delta_z, delta,h}.txt")
            
    # compute the central charge from the stored data
    central_charges = []
    for L in L_values:
       c,_ = entanglement_entropy_function(f"Heisenberg_alternate_(L,D){L,D}_(Jz,J,deltaz, delta,h){Jz,J,delta_z, delta,h}.txt", plot=False)
       central_charges.append(c)
    # Fit central charge with func = 1 - a / L^n
    def fit_func(L, a, n):
        return 1 - a / (L ** n)

    L_values_arr = np.array(L_values)
    central_charges_arr = np.array(central_charges)
    popt, pcov = curve_fit(fit_func, L_values_arr, central_charges_arr, p0=[3, 1])

    # Plot the fit
    L_fit = np.linspace(min(L_values), max(L_values), 200)
    plt.plot(L_fit, fit_func(L_fit, *popt), label=f'Fit: 1 - {popt[0]:.2f}/L^{popt[1]:.2f}',linestyle='--', color='red')
    plt.legend()
    # Plot the central charge as a function of L
    plt.plot(L_values, central_charges, marker='o')
    plt.xlabel('Chain Length (L)')
    plt.ylabel('Central Charge (c)')
    plt.title('Central Charge vs Chain Length')
    plt.grid()
    plt.show()


def correlation_function(filename, coordinate='Z', plot=True):
        """
        Calculate the two-point correlation function ⟨O1_{L/2} O2_r⟩ - ⟨O1_{L/2}⟩⟨O2_r⟩ as a function of distance r.
        Parameters
        ----------
        filename : name of the file containing the correlation data.
        coordinate : coordinate of the correlation function. Default is 'Z'.
        plot : bool, If True, plot the correlation function and its fits.

        Returns
        -------
        xi : Correlation length from the fit a * exp(-r/xi).
        """
        #compute the correlation function between O1 and O2
        corr_function= data.get_correlation(filename,coordinate=coordinate)

        # Define the fitting function
        def fit_exp(r, a, c):
            return a * np.exp(-r/c)
        

        r= np.arange(1, len(corr_function)+1)
        r_odd = r[::2]
        corr_odd =corr_function[::2]
        
        #fit the bulk
        mid = len(r_odd) // 2
        half = len(r_odd) // 4
        r_odd1 = r_odd[mid - half: mid + half]
        corr_odd1 = np.abs(corr_odd[mid - half: mid + half])

        # Perform the curve fitting
        popt, pcov = curve_fit(fit_exp, r_odd1, corr_odd1, p0=(1, 10))
        
        if plot:
            plt.figure(figsize=(10, 6))
            # Plot the fitted curves
            x= np.linspace(1, len(corr_function), 100)
            plt.plot(x, fit_exp(x, *popt), label= f"Fit: $a \\cdot e^{{-x/\\xi}},\\ a={popt[0]:.3f},\\ \\xi={popt[1]:.3f}$", linestyle="--")
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
            x = np.logspace(np.log10(1), np.log10(len(corr_function)), 100)
            plt.semilogy(r_odd,np.abs(corr_odd), marker='o')
            plt.semilogy(x,fit_exp(x, np.abs(popt[0]), popt[1]), label=f"Fit: $a \\cdot e^{{-x/\\xi}},\\ a={popt[0]:.3f},\\ \\xi={popt[1]:.3f}$", linestyle="--")
            plt.legend()
            plt.xlabel("r")
            plt.ylabel(r"$log\langle S^z_{\frac{L}{2}} S^z_{\frac{L}{2}+r} \rangle$")
            plt.title("Semi-Log Correlation Function Fits")
            plt.grid()
            plt.show()
        
        return  popt[1]

def correlation_length_vs_deltaz(L, D, Jz, J, delta_z_values, delta, h, datas=True, plot=True):  
    """
    Compute the correlation length as a function of delta_z
     
     
    Parameters:
    ----------
    L : number of sites
    D : bond dimension
    J and Jz : Coupling constants.
    h : External magnetic field.
    delta_z_values ,delta : list of parameter of the Hamiltonian
    datas : If True, store the data to a file
    Returns:
    -------
    inverse correlation lengths : list of inverse correlation lengths
    """
    if datas:
        # Store the correlation to a file.
        for delta_z in delta_z_values:
                data.store_MPS(L, 2, D, J, Jz, h, delta, delta_z, model="Heisenberg_alternate")
                data.store_correlation(f"Heisenberg_alternate_(L,D){L,D}_(Jz,J,deltaz, delta,h){Jz,J,delta_z, delta,h}.txt" ,coordinate='Z')
    # compute the correlation length from the stored data
    correlation_lengths = []
    for delta_z in delta_z_values:
       c = correlation_function(f"Heisenberg_alternate_(L,D){L,D}_(Jz,J,deltaz, delta,h){Jz,J,delta_z, delta,h}.txt", plot=False)
       correlation_lengths.append(1/c)

    
    if plot:
        # Plot the correlation length as a function of delta_z
        plt.figure(figsize=(10, 6))
        plt.plot(delta_z_values, correlation_lengths, marker='o')
        plt.xlabel(r'$\delta_z$')
        plt.ylabel(r" $\xi^{-1}$")
        plt.title('Inverse correlation Length vs Delta_z')
        plt.grid()
        plt.show()
    return correlation_lengths

def critical_phase_Diagram(L, D, Jz, J, delta_z_values, delta_values, h, datas=True):
    """
    Compute the critical phase diagram of the alternating Heisenberg model 
    (i.e critical point in detla vs delta_z)
    Parameters:
    ----------
    L : number of sites
    D : bond dimension
    J and Jz : Coupling constants.
    h : External magnetic field.
    delta_z ,delta : parameter of the Hamiltonian
    Returns:
    -------
    None
    """
    critical_points = []
    def find_min(x,y):
        """
        Find the minimum of the correlation_vs_delta_z function and return the delta_z value at which it occurs.
        """
        def fit_func(x, a, b, c):
            return a * np.abs(x - b) + c

        # Perform the curve fitting
        popt, _ = curve_fit(fit_func,x ,y ,p0=[1, 0, 0])
        return popt[1]
    predicted_values=[0.45,0.33,0.16,0,-0.16,-0.33,-0.45,0]
    #delta_z_values = np.arange(predicted_values[0]-1, predicted_values[0]+1.1, 0.5)
    i=1
    for delta in delta_values:
            a = find_min(delta_z_values, correlation_length_vs_deltaz(L, D, Jz, J, delta_z_values, delta, h, datas=datas, plot=False))
            #a = min(correlation_length_vs_deltaz(L, D, Jz, J, delta_z_values, delta, h, datas=datas, plot=False))   
            critical_points.append(a)
            #delta_z_values = np.arange(predicted_values[i]-1, predicted_values[i]+1.1, 0.5)
            i+=1
    # Compute the critical phase diagram
    

    # Fit the critical points with a linear function: delta = a * delta_z + b
    def linear_func(x, a, b):
        return a * x + b

    critical_points_arr = np.array(critical_points)
    delta_values_arr = np.array(delta_values)
    popt, _ = curve_fit(linear_func, critical_points_arr, delta_values_arr)
    a, b = popt

    # Plot the critical points and the fit
    plt.figure(figsize=(10, 6))
    plt.plot(critical_points, delta_values, marker='o', label='Critical Points')
    x_fit = np.linspace(min(critical_points), max(critical_points), 200)
    plt.plot(x_fit, linear_func(x_fit, a, b), 'r--', label=f'Fit: delta = {a:.3f} * delta_z + {b:.3f}')
    plt.xlabel(r'$\delta_z$')
    plt.ylabel(r'$\delta$')
    plt.title('Critical Phase Diagram')
    plt.legend()
    plt.grid()
    plt.show()
   