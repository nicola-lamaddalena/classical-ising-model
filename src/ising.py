import json
import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from metro import metropolis
from utils import magnetization, energy

J, H, T = 1.0, 0.0, 2.269

ROOT = Path(__file__).resolve().parent.parent
ANIMATIONS_DIR = ROOT / "animations"
SRC_DIR = ROOT / "src"
CONFIGS_DIR = ROOT / "configs"

ANIMATIONS_DIR.mkdir(exist_ok=True)
CONFIGS_DIR.mkdir(exist_ok=True)
CMAP = ListedColormap(["#d90b26", "#0739ad"])

def animate(frame, q, V, N, J, H, T, frame_text, even_mask, hist_frames, hist_magn, hist_eng, magn_line, eng_line):
    steps_per_frame = 1
    for _ in range(steps_per_frame):
        metropolis(V, N, J, H, T, even_mask)
        hist_frames.append(frame)
        hist_magn.append(magnetization(V))
        hist_eng.append(energy(V, J))
    q.set_data(V)
    magn_line.set_data(hist_frames, hist_magn)
    eng_line.set_data(hist_frames, hist_eng)

    frame_text.set_text(f"Frame: {frame}.")

    return q, frame_text, magn_line, eng_line

def main():
    CONFIG_FILE = CONFIGS_DIR / "configs.json"

    if not CONFIG_FILE.exists():
        print(f"Error: {CONFIG_FILE} not found.")
        return
    
    with open(CONFIG_FILE, "r") as f:
        config_data = json.load(f)

    parser = argparse.ArgumentParser(
            description="Classical 2D Ising model simulation",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    choices = list(config_data.keys()) # update list of configurations 
    parser.add_argument("-M", "--mode", type=str, default="crit", choices=choices, help="Preset configuration (low, critic, high)")
    parser.add_argument("-f", "--file", type=str, help="Choose a name for the animation file")
    parser.add_argument("--view", action="store_true", help="View existing animation without simulating")
    args = parser.parse_args()

    mode = args.mode
    conf = config_data[mode]
    N = conf["N"]
    T = conf["T"]
    J = conf["J"]
    H = conf["H"]
    FRAMES = conf["frames"]

    user_input = input(f"Enter a filename for the animation (default: {mode}): ").strip()
    base_name = user_input if user_input else mode
    filename = ANIMATIONS_DIR / f"{Path(base_name).stem}.gif"
    
    if filename.exists():
        print(f"'{filename.name}' found in {ANIMATIONS_DIR.name}. Visualizing...")
        os.system(f"xdg-open {filename.absolute()}")
        sys.exit()

    print(f"Filename determined: {filename.name}. Starting simulation with N={N}, T={T}, J={J}, h={H}")
    
    start_time = time.time()
    V = np.random.choice([-1, 1], size=(N, N))
    even_mask = np.zeros((N,N), dtype=bool)
    even_mask[::2, ::2] = True # even columns and even rows
    even_mask[1::2, 1::2] = True # odd columns and odd rows
    hist_frames, hist_magn, hist_eng = [], [], []

    plt.style.use("dark_background")
    #fig, ax = plt.subplots(figsize=(10,10))
    #ax.set_xlabel("Lattice width")
    #ax.set_ylabel("Lattice height")
    props = {"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "none"}
    #frame_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="black", bbox=props)
    #ax.set_title(rf"Lattice with ${N} \times{N}$ points, $J={J}$, $h={H}$, $T={T}$")
    #q = ax.imshow(V, cmap=CMAP, interpolation="nearest", origin="lower")
    legend_el = [
            Line2D([0], [0], marker=r"$\uparrow$", color="none", label="Spin up (+1)", markerfacecolor=CMAP(1.0), markersize=15, markeredgecolor="none"), 
            Line2D([0], [0], marker=r"$\downarrow$", color="none", label="Spin down (-1)", markerfacecolor=CMAP(0), markersize=15, markeredgecolor="none")
        ]
    #ax.legend(handles=legend_el, loc="upper right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.1, labelspacing=1.5, framealpha=0.6)
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(12,8), constrained_layout=True)
    gs = GridSpec(nrows=2, ncols=2, width_ratios=[2,1], figure=fig)
    ax_main = fig.add_subplot(gs[:,0])
    ax_top = fig.add_subplot(gs[0,1])
    ax_bottom = fig.add_subplot(gs[1,1])
    ax_top.set_xlim(0, FRAMES)
    ax_top.set_ylim(-1.05, 1.05)
    ax_bottom.set_xlim(0, FRAMES)
    ax_bottom.set_ylim(-2.05, 0.05)
    magn_line, = ax_top.plot([], [], color="white")
    eng_line, = ax_bottom.plot([], [], color="cyan")
    q = ax_main.imshow(V, cmap=CMAP, interpolation="nearest", origin="lower")
    frame_text = ax_main.text(0.02, 0.95, "", transform=ax_main.transAxes, color="black", bbox=props)


    #plt.tight_layout()
    
    ani = FuncAnimation(
        fig,
        animate,
        fargs=(q, V, N, J, H, T, frame_text, even_mask, hist_frames, hist_magn, hist_eng, magn_line, eng_line),
        frames=FRAMES,
        interval=50,
        blit=True
    )
    print(f"Saving {filename}...")
    ani.save(filename=filename, writer="pillow", fps=30, dpi=80)
    elapsed_time = time.time() - start_time
    mins, secs = divmod(elapsed_time, 60)
    inp = input(f"{filename.name} saved in {int(mins)}m {secs:.2f}s. Press enter to visualize, or type 'no': ")
    if inp.lower() != "no":
        os.system(f"xdg-open {filename.absolute()}") 

if __name__ == "__main__":
    main()
