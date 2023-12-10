# Import necessary modules
import utils.utils_bohm as u
import pickle
import numpy as np
import scipy.special as sc

# Set grid parameters
N = 200
X, Y, Z = np.mgrid[-1:1:N*1j, -1:1:N*1j, -1:1:N*1j]
Xelse = np.mgrid[-1:1:N*1j]

# Define wavefunction psi_klm
def psi_klm(X, Y, Z, k, l, m, R0):
    mu = 2
    r = np.sqrt(X**2 + Y**2 + Z**2)
    theta = np.arccos(Z/r)
    phi = np.arctan2(Y, X)
    return (r - R0)**l * np.exp(-mu * (r - R0)**2) * (sc.genlaguerre(k, l + 1/2))(2*mu*(r - R0)**2) * sc.sph_harm(m, l, phi, theta)

# Normalize the wavefunction
def normalize_phi(phi):
    cst = np.sum(np.absolute(phi)**2)
    return phi / cst

# Generate normalized wavefunction phi
phi = normalize_phi(psi_klm(X, Y, Z, 7, 7, 1, 0.1))

# Generate random particle samples based on the wavefunction
Samples = u.get_samples(20000, np.absolute(phi)**2, (Xelse, Xelse, Xelse))

# Calculate the velocity field v_phi
vphi = u.v_phi(phi, (X, Y, Z), N)
print("Calculated v_phi")

# Generate particle trajectories
Trajectories = u.get_trajectory_Samples(Samples, vphi, 0.001, 200, 200, 2)
print("Generated particle trajectories")

# Save the particle trajectories to a pickle file
file3 = open('./particle_trajectories.pkl', 'wb')
pickle.dump(Trajectories, file3)
