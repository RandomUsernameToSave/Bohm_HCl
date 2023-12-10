import numpy as np
from random import choices
import scipy.special as sc

def compute_density(Trajectories, N_grid):
    """
    Compute particle density in a 3D grid based on their final positions.

    Parameters:
    - Trajectories: List of lists representing particle positions at each time step.
      Format: [t0 [position], t1 [position], ...]
    - N_grid: Number of grid points in each dimension.

    Returns:
    - density: 3D array representing the particle density in the grid.
      Each grid cell contains the square root of the normalized particle count.
    """
    # Initialize a 3D array to store particle density
    density = np.zeros((N_grid, N_grid, N_grid))

    # Count particle occurrences in the final positions
    for i_particle in range(len(Trajectories[-1])):
        x, y, z = get_nearest_point_index(Trajectories[-1][i_particle], N_grid)
        density[x][y][z] += 1

    # Normalize the density and take the square root
    normalized_density = np.sqrt(density / len(Trajectories[0]))

    return normalized_density


def magouille(grid,N_grid):
    ML = np.zeros_like(grid)
    for i in range(N_grid):
        for j in range(N_grid):
            for k in range(N_grid):
                x = grid[0][i][j][k]
                y = grid[1][i][j][k]
                ML[0][i][j][k]= -y/(x**2+y**2)
                ML[1][i][j][k]= x/(x**2+y**2)

    return np.asarray(ML)

def v_phi(phi,grid,N_grid):
    
    magouille_array = magouille(grid,N_grid)

    return magouille_array # np.imag( np.gradient( phi , 2/N_grid) / phi)

def get_nearest_point_index(X,N):
    """Grid has to be symetric to 0"""
    x = round((X[0]+1)*N/2)
    y = round((X[1]+1)*N/2)
    z = round((X[2]+1)*N/2)

    if x>=N:
        x=N-1
    if y>=N:
        y=N-1
    if z>=N:
        z=N-1
    if x<=-N:
        x=-N+1
    if y<=-N:
        y=-N+1
    if z<=-N:
        z=-N+1

    return x,y,z

def get_trajectory(X0, v_phi, dt ,Number_of_dt, N_grid,En ):
    X = [X0]
    for i in range(Number_of_dt):
        x,y,z = get_nearest_point_index(X[-1],N_grid)

        interest_vphix = v_phi[0][x][y][z]
        interest_vphiy = v_phi[1][x][y][z]
        interest_vphiz = v_phi[2][x][y][z]
        
        vphi_int_vector = np.asarray([interest_vphix,interest_vphiy,interest_vphiz])
        
        next_pos = X[-1] + dt*vphi_int_vector # value of the closest field calculated
        X.append(next_pos)
    return X

def get_trajectory_Samples(Samples, v_phi, dt, Number_of_dt, N_grid, En):
    """
    Calculate particle trajectories based on velocity field data.

    Parameters:
    - Samples: List of initial positions for particles.
    - v_phi: Velocity field data represented as a 4D array (x, y, z, component).
    - dt: Time step for integration.
    - Number_of_dt: Number of time steps.
    - N_grid: Number of grid points.
    - En: Energy parameter (not clear how it's used, might need additional documentation).

    Returns:
    - Trajectories: List of lists representing particle positions at each time step.
      Format: [t0 [position], t1 [position], ...]
    """
    Trajectories = [Samples]

    for i in range(Number_of_dt):
        next_list_positions = []

        for i_particle in range(len(Samples)):
            x, y, z = get_nearest_point_index(Trajectories[-1][i_particle], N_grid)

            # Extract velocity components at the nearest grid point
            interest_vphix = v_phi[0][x][y][z]
            interest_vphiy = v_phi[1][x][y][z]
            interest_vphiz = v_phi[2][x][y][z]

            # Create a velocity vector from the components
            vphi_int_vector = np.asarray([interest_vphix, interest_vphiy, interest_vphiz])

            # Calculate the next position using the velocity vector and time step
            next_pos = Trajectories[-1][i_particle] + dt * vphi_int_vector

            next_list_positions.append(next_pos)

        Trajectories.append(next_list_positions)

    return Trajectories


def divergence(f):
    """
    Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
    :param f: List of ndarrays, where every item of the list is one dimension of the vector field
    :return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
    """
    num_dims = len(f)
    return np.ufunc.reduce(np.add, [np.gradient(f[i],2/200 , axis=i) for i in range(num_dims)])

def potential_Q(phi, N_grid):
    """
    Compute the quantum potential (Q) based on the given wavefunction.

    Parameters:
    - phi: 3D array representing the wavefunction.
    - N_grid: Number of grid points in each dimension.

    Returns:
    - req: Quantum potential (Q) computed based on the given wavefunction.
    """
    # Calculate the gradient of the absolute value of the wavefunction
    gradient_phi = np.gradient(np.absolute(phi), 2/N_grid)

    # Calculate the divergence of the gradient and normalize where necessary
    req = np.divide(-divergence(gradient_phi), np.absolute(phi), out=np.zeros_like(np.absolute(phi)), where=np.absolute(phi) >= 0.0005)

    return req


def total_potential(phi, N_grid):
    """
    Calculate the total potential energy, including the classical potential (V)
    and the quantum potential (Q), based on the given wavefunction.

    Parameters:
    - phi: 3D array representing the wavefunction.
    - N_grid: Number of grid points in each dimension.

    Returns:
    - total_potential: 3D array representing the total potential energy.
    """
    # Create a 3D grid for coordinates
    N = N_grid
    X, Y, Z = np.mgrid[-1:1:N*1j, -1:1:N*1j, -1:1:N*1j]

    # Define the classical potential function
    def get_potential(x, y, z):
        return 0.3 * (np.sqrt(x**2 + y**2 + z**2) - 0.3)**2

    # Calculate the classical potential (V)
    V = get_potential(X, Y, Z)

    # Calculate the quantum potential (Q)
    Q = potential_Q(phi, N_grid)

    # Compute the total potential energy by subtracting V and Q
    total_potential = np.gradient(-V - Q, 2/N_grid)

    return total_potential


def trajectory_from_potential(Samples, phi, dt ,Number_of_dt, N_grid,En,vphi):
    #Speed_traj = get_trajectory_Samples(Samples,vphi,dt,2,N_grid,En)
    Trajectories = [np.asarray(Samples) , np.asarray(Samples)] #Speed_traj
    tot_pot = total_potential(phi,N_grid)

    for i in range(Number_of_dt):
        next_list_positions = []
        
        for i_particle in range(len(Samples)) :
            x,y,z = get_nearest_point_index(Trajectories[-1][i_particle],N_grid)

            interest_vphix = tot_pot[0][x][y][z]
            interest_vphiy = tot_pot[1][x][y][z]
            interest_vphiz = tot_pot[2][x][y][z]
            
            vphi_int_vector = np.asarray([interest_vphix,interest_vphiy,interest_vphiz])
            
            next_pos = 2*Trajectories[-1][i_particle] -Trajectories[-2][i_particle] + vphi_int_vector*(dt**2) # value of the closest field calculated
            next_list_positions.append(next_pos)

        Trajectories.append(next_list_positions)
    return Trajectories

def get_samples(NumberParticles, phi, grid):
    """
    Generate random particle samples based on the given wavefunction values and grid coordinates.

    Parameters:
    - NumberParticles: Number of particle samples to generate.
    - phi: 3D array representing the wavefunction values at each grid point.
    - grid: Tuple (X, Y, Z) representing the 3D grid coordinates (x, y, z).

    Returns:
    - particles: List of randomly chosen particle samples.
      Each particle is represented as a 3D coordinate [x, y, z].
    """
    # Initialize lists to store grid coordinates and corresponding phi values
    value_phi = []
    value_grid = []
    X, Y, Z = grid

    # Iterate over each grid point and store coordinates and phi values
    for x in range(len(X)):
        for y in range(len(Y)):
            for z in range(len(Z)):
                value_grid.append([X[x], Y[y], Z[z]])
                value_phi.append(phi[x][y][z] ** 2)

    # Use the choices function to randomly sample NumberParticles particles
    particles = choices(value_grid, value_phi, k=NumberParticles)

    return particles

def total_potential_V2(Trajectories,N_grid):
    N = N_grid
    X, Y, Z = np.mgrid[-1:1:N*1j, -1:1:N*1j, -1:1:N*1j]

    density = compute_density(Trajectories,N_grid)
    def get_potential(x, y, z):
        return 0.3* (np.sqrt(x**2+y**2+z**2)-0.3)**2 
    
    V = np.asarray(get_potential(X,Y,Z))
    Q = np.asarray(potential_Q(density,N_grid))


    return np.gradient(-V-Q , 2/N_grid)

def trajectory_from_potential_V2(Samples, phi, dt ,Number_of_dt, N_grid,En,vphi):
    
    Trajectories = [np.asarray(Samples),np.asarray(Samples)]
    
    for i in range(Number_of_dt):
        next_list_positions = []
        tot_pot = total_potential_V2(Trajectories,N_grid)
        for i_particle in range(len(Samples)) :
            x,y,z = get_nearest_point_index(Trajectories[-1][i_particle],N_grid)

            interest_vphix = tot_pot[0][x][y][z]
            interest_vphiy = tot_pot[1][x][y][z]
            interest_vphiz = tot_pot[2][x][y][z]
            
            vphi_int_vector = np.asarray([interest_vphix,interest_vphiy,interest_vphiz])
            
            next_pos = 2*Trajectories[-1][i_particle] -Trajectories[-2][i_particle] + vphi_int_vector*dt**2 # value of the closest field calculated
            next_list_positions.append(next_pos)

        Trajectories.append(next_list_positions)
    return Trajectories

def analytical_Q_potential(X,Y,Z,mu,m,wave_function,k=7):
    """Only works for m=l quantum number"""
    r = np.sqrt(X**2+Y**2+Z**2)
    theta = np.arccos(Z/r)
    phi = np.arctan(Y/X)

    L0 = sc.genlaguerre(k , m+1/2)(2*mu*r**2)
    L1 = sc.genlaguerre(k-1 , m+1+1/2)(2*mu*r**2)
    L2 = sc.genlaguerre(k-1 , m+1+1/2)(2*mu*r**2)

    C_r = np.absolute(wave_function)/(r**2)* ( (m*r-2*mu *(1+L1/L0)*r**3 )**2 + m - 6*mu *(1+L1/L0)*r**2 -4*(mu**2)*(r**4)* (-L2/L0 + (L1/L0)**2))
    C_theta = np.absolute(wave_function) * m / (r**2) * ( m * np.cos(theta) / (np.sin(theta)**2) - 1)

    return -(C_r + C_theta ) / np.absolute(phi)

def analytical_total_potential(phi,N_grid,m):
    N = N_grid
    X, Y, Z = np.mgrid[-1:1:N*1j, -1:1:N*1j, -1:1:N*1j]

    def get_potential(x, y, z):
        return 0.3 * (np.sqrt(x**2+y**2+z**2))**2 
    
    V = get_potential(X,Y,Z)
    Q = analytical_Q_potential(X,Y,Z,2,m,phi)
   
    return np.gradient(-V-Q , 2/N_grid)

def trajectory_from_potential_analytical(Samples, phi, dt ,Number_of_dt, N_grid,m):
    #Speed_traj = get_trajectory_Samples(Samples,vphi,dt,2,N_grid,En)
    Trajectories = [np.asarray(Samples) , np.asarray(Samples)] #Speed_traj
    tot_pot = analytical_total_potential(phi,N_grid,m)

    for i in range(Number_of_dt):
        next_list_positions = []
        
        for i_particle in range(len(Samples)) :
            x,y,z = get_nearest_point_index(Trajectories[-1][i_particle],N_grid)

            interest_vphix = tot_pot[0][x][y][z]
            interest_vphiy = tot_pot[1][x][y][z]
            interest_vphiz = tot_pot[2][x][y][z]
            
            vphi_int_vector = np.asarray([interest_vphix,interest_vphiy,interest_vphiz])
            
            next_pos = 2*Trajectories[-1][i_particle] -Trajectories[-2][i_particle] + vphi_int_vector*(dt**2) # value of the closest field calculated
            next_list_positions.append(next_pos)

        Trajectories.append(next_list_positions)
    return Trajectories
