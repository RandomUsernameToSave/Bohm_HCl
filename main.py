import utils.utils_bohm as u
import pickle
import numpy as np
Number_Samples = 20000

file1 = open('./eigenvalues.pkl','rb')
file2 = open('./eigenvector.pkl', 'rb')
#file3 = open('./particle_trajectories.pkl','wb')
#file4 = open('./particle_sample.pkl','wb')

N= 100

eigenvalues = pickle.load(file1)
eigenvectors = pickle.load(file2)
X, Y, Z = np.mgrid[-1:1:N*1j, -1:1:N*1j, -1:1:N*1j]
Xelse = np.mgrid[-1:1:N*1j]

def get_e(n):
    return eigenvectors.T[n].reshape((N,N,N)).cpu().numpy()

valeur_propre = eigenvalues[4].cpu().numpy()

phi = get_e(0)
print(np.any(np.imag(phi)))
vphi = u.v_phi(phi,Xelse)

#Samples = u.get_samples(Number_Samples,phi,(Xelse,Xelse,Xelse))
#pickle.dump(Samples,file4)


Trajectories = []

#for particle in Samples:

#    Trajectory = u.get_trajectory(particle,vphi,0.1,100,N,valeur_propre)
#    Trajectories.append(Trajectory)

#pickle.dump(Trajectories,file3)
