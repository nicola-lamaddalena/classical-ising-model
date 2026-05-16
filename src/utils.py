import numpy as np

def magnetization(lattice):
    return np.sum(lattice) / lattice.size

def energy(lattice, J):
    neigh_sum = (
            lattice * np.roll(lattice, shift=-1, axis=0)
            + lattice * np.roll(lattice, shift=-1, axis=1)
            )
    return (-J * np.sum(neigh_sum)) / lattice.size
