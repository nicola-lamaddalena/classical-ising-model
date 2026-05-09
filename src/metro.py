import numpy as np

rng = np.random.default_rng(seed=42)

def metropolis(lattice, N, J, h, T, even_mask):
    """Metropolis algorithm for classical Ising model:
    choose a random point in the lattice and compute the variation of energy;
    if it's negative, accept the change; otherwise evaluate the transition probability 
    with the Boltzmann's distribution: exp(-beta E).
    """

    kb = 1.0


    for s in [even_mask, ~even_mask]: # even_mask, not even_mask
        neighbors_sum = np.roll(lattice, shift=1, axis=0) + np.roll(lattice, shift=-1, axis=0) + np.roll(lattice, shift=1, axis=1) + np.roll(lattice, shift=-1, axis=1)
        delta_energy_grid = 2 * J * lattice * neighbors_sum + 2 * h * lattice

        probability = np.exp(-delta_energy_grid / (kb * T))
        accept = (delta_energy_grid <= 0) | (rng.random((N,N)) < probability)
        lattice[s & accept] *= -1

