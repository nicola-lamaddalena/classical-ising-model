import numpy as np

def metropolis(lattice, N, J, h, T):
    """Metropolis algorithm for classical Ising model:
    choose a random point in the lattice and compute the variation of energy;
    if it's negative, accept the change; otherwise evaluate the transition probability 
    with the Boltzmann's distribution: exp(-beta E).
    """

    kb = 1.0
    i, j = np.random.randint(0, N, size=2)
    spin = lattice[i, j]
    """ Sum over the nearest neighbors:
            [i-1,j]
               |
    [i,j-1]--[i,j]--[i,j+1]
               |
            [i+1,j]
    """
    neighbors_sum = lattice[i, (j-1)%N] + lattice[i, (j+1)%N] + lattice[(i-1)%N, j] + lattice[(i+1)%N, j]
    delta_energy = 2 * J * spin * neighbors_sum + 2 * h * spin
    if delta_energy < 0:
        #The spin has only two values: +1, -1
        lattice[i, j] *= -1
    else:
        if np.random.random() < np.exp(-delta_energy / (kb * T)):
            lattice[i, j] *= -1
