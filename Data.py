# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 19:28:54 2025

@author: thibaud lesquereux
"""

import numpy as np
from MPS import MPS
from Heisenberg_MPO import HeisenbergMPO
from Ising_MPO import IsingMPO
from AKLT_MPO import AKLTMPO
from Heisenberg_alternate_MPO import Heisenberg_alternateMPO
import os
import time
import sys
import threading

def chrono(start_event, stop_event):
    """
    Function to display a timer in the console.
    Stops when a keyboard interrupt is detected.
    """
    
    start_time = time.time()
    while not start_event.is_set():
        time.sleep(0.01)  
    while not stop_event.is_set():
        elapsed = int(time.time() - start_time)
        minutes, seconds = divmod(elapsed, 60)
        sys.stdout.write(f"\r          Elapsed time : {minutes:02d}:{seconds:02d}")
        sys.stdout.flush()
        time.sleep(1)  # Update every second
    print("\n")  # line to clear the console at the end
   

def save_mps_to_file(filename, mps):
    """
    Save the MPS tensors to a file.

    parameters:
    filename (str): The name of the file to save the MPS to.
    mps (MPS): The MPS object to save.
    """
    # Create the full path
    directory = r"C:\Users\13thi\Documents\EPFL\EPFL\Condensed matter\CTMC 2\data"
    full_path = os.path.join(directory, filename)
 
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Save the MPS tensors to the file
    with open(full_path, 'w') as f:
        # Write the dimensions of the MPS
        f.write(f"{mps.length,mps.state_dim,mps.bond_dim}\n")

        # Write each tensor's shape and data
        for tensor in mps.tensors:
            f.write(f"{tensor.shape}\n")
            np.savetxt(f, tensor.flatten(), newline=' ')
            f.write('\n')

def data( L=100, d=2, D=100, J=1., Jz=.0, h=.0, delta=.0, delta_z=.0, model="Heisenberg_alternate"):
    """
    Store the MPS of a given case in a file.

    Parameters:
    L (int): Length of the chain.
    d (int): Physical dimension of the system.
    D (int): Bond dimension.
    J (float): Coupling constant.
    Jz (float): Coupling constant for the z-direction.
    h (float): Magnetic field strength.
    delta (float): Parameter for the Heisenberg_alternate model.
    delta_z (float): Parameter for the Heisenberg_alternate model.
    model (str): Model type ("Heisenberg", "Ising", "AKLT", "Heisenberg_alternate").

    returns:
    a file containing the MPS tensors.
    """
    
    start_time = time.time()
    # Check if the model is valid
    if model == "Heisenberg":
        mpo = HeisenbergMPO(L, J,Jz,h)
    elif model == "Ising":
        mpo = IsingMPO(L, J, h)
    elif model == "AKLT":
        mpo = AKLTMPO(L)
    elif model == "Heisenberg_alternate":
        mpo = Heisenberg_alternateMPO(L, J, Jz, h, delta, delta_z)
    else:
        raise ValueError("Unknown model of MPO")

    mps = MPS(L, d, D)
    _, mps = mpo.variational_ground_state(mps,3, 1e-6)
    
    filename = f"{model}_(L,D){L,D}_(Jz,J,deltaz, delta,h){Jz,J,delta_z, delta,h}.txt"
    save_mps_to_file(filename, mps)
    end_time = time.time()
    print(f"\nMPS saved as {filename},\nTime taken: {end_time - start_time:.6f} seconds")



def store_MPS( L=100, d=2, D=100, J=1., Jz=.0, h=.0, delta=.0, delta_z=.0, model="Heisenberg_alternate"):
    """
    Store the MPS of a given case in a file with a timer.

    """
    start_event = threading.Event()
    stop_event = threading.Event()
    t = threading.Thread(target=chrono, args=(start_event, stop_event))
    t.start()
    start_event.set()
    data(L, d, D, J, Jz, h, delta, delta_z, model)
    stop_event.set()
    t.join()




def get_MPS(filename, directory=r"C:\Users\13thi\Documents\EPFL\EPFL\Condensed matter\CTMC 2\data"):
    """
    Load the MPS tensors from a file.

    Parameters:
    filename (str): The name of the file to load the MPS from.

    Returns:
    MPS: The loaded MPS object.
    """
    filename = filename.strip()
    full_path = os.path.join(directory, filename)
    
    # Ensure the directory exists
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"The file {filename} does not exist.")
    
   
    with open(full_path, 'r') as f:
        # Read the first line to get the dimensions
        dim = eval(f.readline().strip())

        # Read the rest of the file to get the tensors
        tensors = []

        for line in f:
            if line.startswith("Entropy Data") or line.startswith("Correlation"):
                break
            shape = eval(line.strip())
            line = next(f)
            tensor = np.fromiter(map(float, line.strip().split()), dtype=float).reshape(shape)
            tensors.append(tensor)
            
    mps=MPS(dim[0], dim[1], dim[2])
    mps.tensors=tensors
    print("MPS loaded successfully.")
    return mps



def store_entropy(filename,directory = r"C:\Users\13thi\Documents\EPFL\EPFL\Condensed matter\CTMC 2\data"):
    """
    Save the entropy of the MPS at each site to a file.

    Parameters:
    filename (str): The name of the file to save the entropy to.
    directory (str): The directory where the file will be saved.
    """
    # Create the full path
    full_path = os.path.join(directory, filename)
    
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Calculate entropy at each site
    mps=get_MPS(filename,directory)
    entropies = []
    for site in range(mps.length):
        entropies.append(mps.entanglement_entropy(site))

    # Check if "Entropy Data" already exists in the file
    with open(full_path, 'r') as f:
        lines = f.readlines()

    entropy_data_index = None
    for i, line in enumerate(lines):
        if line.startswith("Entropy Data"):
            entropy_data_index = i
            break

    # If "Entropy Data" exists, overwrite the next line
    if entropy_data_index is not None:
        lines[entropy_data_index + 1] = " ".join(map(str, entropies)) + "\n"
        with open(full_path, 'w') as f:
            f.writelines(lines)
            # Stop if another "Correlation" or "Entropy" section is found after this
            for j in range(entropy_data_index + 2, len(lines)):
                if lines[j].startswith("Correlation"):
                    break
    else:
        # Otherwise, append "Entropy Data" and the entropy values
        with open(full_path, 'a') as f:
            f.write("Entropy Data\n")
            f.write(" ".join(map(str, entropies)) + "\n")
            
    
    print(f"Entropy saved in {filename}")


def get_entropy(filename, directory = r"C:\Users\13thi\Documents\EPFL\EPFL\Condensed matter\CTMC 2\data"):
    """
    Load the entropy of the MPS from a file.

    Parameters:
    filename (str): The name of the file to load the entropy from.
    directory (str): The directory where the file is located.

    Returns:
    list: A list of entanglement entropy values at each site.
    """
    # Create the full path
    full_path = os.path.join(directory, filename)
    
    # Ensure the directory exists
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"The file {full_path} does not exist.")
    
    entropies = []
    with open(full_path, 'r') as f:
        for line in f:
            if line.startswith("Entropy Data"):
                line = next(f) # Skip comment lines
                for x in line.split():
                    entropies.append(float(x))
                    if line.lower().startswith("correlation"):
                        break  # Stop if a correlation section is found
            
    if entropies == []:
        raise ValueError(f"No entropy data found in {filename}.")
    return entropies


def store_correlation(filename, directory = r"C:\Users\13thi\Documents\EPFL\EPFL\Condensed matter\CTMC 2\data",coordinate='Z'):
    """
    Save the correlation function C(r) to a file.

    Parameters:
    filename (str): The name of the file to save the correlation to.
    correlation (list): The correlation function values.
    directory (str): The directory where the file will be saved.
    """
    #spin operator
    if coordinate == 'X':
        S = np.array([[0, 1], [1, 0]])/2
    elif coordinate == 'Y':
        S = np.array([[0, -1j], [1j, 0]])/2
    elif coordinate == 'Z':
        S = np.array([[1, 0], [0, -1]])/2
    else:
        raise ValueError("Invalid coordinate. Choose 'X', 'Y', or 'Z'.")

    # Create the full path
    full_path = os.path.join(directory, filename)
    
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    #compute the correlation function between O1 and O2
    mps=get_MPS(filename,directory)
    corr_function=[]
    r= range(1,-(-mps.length//2+1))
    for i in range(mps.length//2, mps.length):
        corr_function.append( mps.two_site_corr(S, S, mps.length//2-1, i)-mps.mean(S, mps.length//2-1)*mps.mean(S, i))

    # Check if "Correlation Data" already exists in the file
    with open(full_path, 'r') as f:
        lines = f.readlines()

    correlation_data_index = None
    for i, line in enumerate(lines):
        if line.startswith(f"Correlation {coordinate} Data"):
            correlation_data_index = i
            break

    # If "Correlation Data" exists, overwrite the next line
    if correlation_data_index is not None:
        lines[correlation_data_index + 1] = " ".join(map(str, corr_function)) + "\n"
        with open(full_path, 'w') as f:
            f.writelines(lines)
            # Stop if another "Correlation" or "Entropy" section is found after this
            for j in range(correlation_data_index + 2, len(lines)):
                if lines[j].startswith("Correlation") or lines[j].startswith("Entropy"):
                    break
    else:
        # Otherwise, append "Correlation Data" and the Correlation values
        with open(full_path, 'a') as f:
            f.write(f"Correlation {coordinate} Data\n")
            f.write(" ".join(map(str, corr_function)) + "\n")

    print(f"Correlation {coordinate} saved in {filename}")


def get_correlation(filename, directory = r"C:\Users\13thi\Documents\EPFL\EPFL\Condensed matter\CTMC 2\data",coordinate='Z'):
    """
    Load the correlation function C(r) from a file.

    Parameters:
    filename (str): The name of the file to load the correlation from.
    directory (str): The directory where the file is located.
    coordinate (str): The coordinate for which to load the correlation data ('X', 'Y', or 'Z').

    Returns:
    list: A list of correlation values.
    """
    # Create the full path
    full_path = os.path.join(directory, filename)
    
    # Ensure the directory exists
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"The file {filename} does not exist.")
    
    correlation = []
    with open(full_path, 'r') as f:
        for line in f:
            if line.startswith(f"Correlation {coordinate} Data"):
                line = next(f) # Skip comment lines
                for x in line.split():
                    correlation.append(float(x))
                    if line.lower().startswith("correlation") or line.lower().startswith("entropy"):
                        break  # Stop if a correlation/Entropy section is found
    if correlation == []:
        raise ValueError(f"No correlation data found in {filename}.")
    return correlation