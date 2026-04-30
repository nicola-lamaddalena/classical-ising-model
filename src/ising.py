import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from metro import metropolis

J, H, T = 1.0, 0.0, 2.269

ROOT = Path(__file__).resolve().parent.parent
ANIMATIONS_DIR = ROOT / "animations"
SRC_DIR = ROOT / "src"
CONFIGS_DIR = ROOT / "configs"

ANIMATIONS_DIR.mkdir(exist_ok=True)
CONFIGS_DIR.mkdir(exist_ok=True)

def animate(frame, q, U, V, N, frame_text):
    for _ in range(N * N):
        metropolis(V, N, J, H, T)
    q.set_UVC(U, V, V)
    frame_text.set_text(f"Frame: {frame}.")

    return q, frame_text

def main():
    parser = argparse.ArgumentParser(
            description="Classical 2D Ising model simulation",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument("N", type=int, nargs="?", default=50, help="Lattice dimension (NxN")
    parser.add_argument("-f", "--file", type=str, help="Choose a name for the animation file")
    parser.add_argument("-t", "--temperature", type=float, default=2.269, help="Temperature of the system")
    parser.add_argument("-j", "--interaction", type=float, default=1.0, help="Interaction factor J")
    parser.add_argument("--view", action="store_true", help="View existing animation without simulating")
    args = parser.parse_args()

    N = args.N
    T = args.temperature
    J = args.interaction

    user_input = input("Enter a filename for the animation (default: ising): ").strip()
    base_name = user_input if user_input else "ising"
    filename = ANIMATIONS_DIR / f"{Path(base_name).stem}_{N}.gif"
    
    if filename.exists():
        print(f"'{filename.name}' found in {ANIMATIONS_DIR.name}. Visualizing...")
        os.system(f"xdg-open {filename.absolute()}")
        sys.exit()

    print(f"Filename determined: {filename.name}. Starting simulation with N={N}, T={T}, J={J}")

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
    ani.save(filename=filename, writer="pillow")
    inp = input(f"{filename.name} saved. Press enter to visualize, or type 'no': ")
    if inp.lower() != "no":
        os.system(f"xdg-open {filename.absolute()}") 

if __name__ == "__main__":
    main()
