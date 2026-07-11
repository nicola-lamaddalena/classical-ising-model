import numpy as np

def magnetization(lattice: np.ndarray) -> float:
    return np.sum(lattice) / lattice.size

def energy(lattice: np.ndarray, J: float, h: float = 0.0) -> float:
    neigh_sum = (
            lattice * np.roll(lattice, shift=-1, axis=0)
            + lattice * np.roll(lattice, shift=-1, axis=1)
            )
    return (-J * np.sum(neigh_sum) - h * np.sum(lattice)) / lattice.size

def block_avg(b: int, obs_values: np.ndarray) -> list[float]:
    """
    Compute the mean value of the blocks. Every block has a length of b.
    The values of b are powers of 2.
    """
    block_num = len(obs_values) // b
    avgs = []
    for i in range(0, block_num*b, b):
        avgs.append(np.mean(obs_values[i:i+b]))

    return avgs

obs_values = np.array([
    0.412, 0.398, 0.405, 0.421, 0.437, 0.429, 0.418, 0.402, 0.395, 0.388,
    0.401, 0.415, 0.428, 0.433, 0.440, 0.425, 0.411, 0.397, 0.383, 0.376,
    0.390, 0.404, 0.417, 0.409, 0.395, 0.381, 0.368, 0.374, 0.387, 0.400,
    0.413, 0.426, 0.432, 0.419, 0.406, 0.392, 0.379, 0.365, 0.371, 0.384,
    0.397, 0.410, 0.423, 0.415, 0.401, 0.388, 0.374, 0.361, 0.367, 0.380
])
bs = [2**n for n in range(len(obs_values)) if len(obs_values) // 2**n >= 4]
stds = []
for b in bs:
    avgs = block_avg(b, obs_values=obs_values)
    std = np.std(avgs, ddof=1) / np.sqrt(len(avgs))
    stds.append(std)

print(stds)

def plateau_finder(std_devs: np.ndarray, epsilon: float = 0.1) -> float:
    """
    Find the minimum value of b that reaches the plateau.
    Value of epsilon is arbitrary and is chosen to be 0.1.
    """
    for i in range(len(std_devs)-1):
        if abs((std_devs[i+1] - std_devs[i]) / std_devs[i]) < epsilon:
            if abs((std_devs[i+2]-std_devs[i+1])/std_devs[i+1]) < epsilon:
                if abs((std_devs[i+3]-std_devs[i+2])/std_devs[i+2]) < epsilon:
                    return std_devs[i]
    return 0

