import os, sys, pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

J, H, T = 1.0, 0.0, 2.269

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

def animate(frame, q, U, V, N, frame_text):
    for _ in range(N * N):
        metropolis(V, N, J, H, T)
    q.set_UVC(U, V, V)
    frame_text.set_text(f"Frame: {frame}.")

    return q, frame_text

def main():
    try:
        N = int(sys.argv[1])
    except IndexError: 
        print("Number of points (N) not found. Using N=50")
        N = 50

    std_filename = "ising"
    filename = input("Enter a filename for the animation: ")

    if filename == "":
        filename = std_filename + f"_{N}.gif"
        print(f"Invalid value. Using {filename}.")
    
    else:
        filename = filename.split(".")[0]
        filename += f"_{N}.gif"
        print(f"Valid filename: using {filename}.")

    if os.path.isfile(filename):
        print(f"{filename} found in the directory. Visualizing {filename}.")
        os.system(f"xdg-open ./{filename}")
        sys.exit()

    X, Y = np.meshgrid(np.arange(0, N, 1), np.arange(0, N, 1))
    U, V = np.zeros_like(X), np.random.choice([-1, 1], size=(N, N))
    
    plt.style.use("dark_background")
    fig, ax = plt.subplots()
    props = {"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "none"}
    frame_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="black", bbox=props)
    ax.set_title(f"Lattice with {N} points, J={J}, h={H}, T={T}")
    q = ax.quiver(X, Y, U, V, V, cmap="YlGnBu", pivot="mid")

    ani = FuncAnimation(
        fig,
        animate,
        fargs=(q, U, V, N, frame_text),
        frames=180,
        interval=40,
        blit=True
    )
    
    print(f"Saving {filename}...")
    ani.save(filename=f"./animations/{filename}", writer="pillow")
    inp = input(f"{filename} saved. Press enter to visualize the gif. (enter/no?) ")
    if inp.lower() != "no":
        os.system(f"xdg-open ./animations/{filename}") 

if __name__ == "__main__":
    main()
