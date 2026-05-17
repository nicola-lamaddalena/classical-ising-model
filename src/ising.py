import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from metro import metropolis
from utils import magnetization, energy

BKG_COLOR = "#E9EEF3"
DOWN_COLOR = "#DB1919"
UP_COLOR = "#1851F0"
MAIN_COLOR = "#191923"

ROOT = Path(__file__).resolve().parent.parent
ANIMATIONS_DIR = ROOT / "animations"
SRC_DIR = ROOT / "src"
CONFIGS_DIR = ROOT / "configs"

ANIMATIONS_DIR.mkdir(exist_ok=True)
CONFIGS_DIR.mkdir(exist_ok=True)
CMAP = ListedColormap([DOWN_COLOR, UP_COLOR])

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
    props = {"boxstyle": "round", "facecolor": BKG_COLOR, "alpha": 0.9, "edgecolor": MAIN_COLOR}

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

    legend_el = [
            Line2D([0], [0], marker=r"$\uparrow$", color="none", label="Spin up (+1)", markerfacecolor=CMAP(1.0), markersize=15, markeredgecolor="none"), 
            Line2D([0], [0], marker=r"$\downarrow$", color="none", label="Spin down (-1)", markerfacecolor=CMAP(0), markersize=15, markeredgecolor="none")
        ]
    fig = plt.figure(figsize=(12,8), constrained_layout=True)
    fig.patch.set_facecolor(BKG_COLOR)
    gs = GridSpec(
            nrows=2, 
            ncols=2, 
            width_ratios=[2,1], 
            figure=fig
        )

    ax_main = fig.add_subplot(gs[:,0])
    ax_main.legend(handles=legend_el, loc="upper right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.1, labelspacing=1.5, framealpha=0.8)
    ax_main.set_title(rf"Lattice with $N={N**2}$ points, $T={T}$, $J={J}$.")
    ax_main.set_xlabel("Lattice width")
    ax_main.set_ylabel("Lattice height")

    ax_top = fig.add_subplot(gs[0,1])
    ax_top.set_title("Average magnetization per spin")
    ax_top.set_xlim(0, FRAMES)
    ax_top.set_ylim(-1.05, 1.05)
    ax_top.set_xlabel("Frames")
    ax_top.set_ylabel("Average magnetization")

    ax_bottom = fig.add_subplot(gs[1,1])
    ax_bottom.set_title("Average energy per spin")
    ax_bottom.set_xlim(0, FRAMES)
    ax_bottom.set_ylim(-2.05, 0.05)
    ax_bottom.set_xlabel("Frames")
    ax_bottom.set_ylabel("Average energy [J]")

    magn_line, = ax_top.plot([], [], color=UP_COLOR)
    eng_line, = ax_bottom.plot([], [], color=DOWN_COLOR)
    q = ax_main.imshow(V, cmap=CMAP, interpolation="nearest", origin="lower")
    frame_text = ax_main.text(0.02, 0.95, "", transform=ax_main.transAxes, color=MAIN_COLOR, bbox=props)
    
    ani = FuncAnimation(
        fig,
        animate,
        fargs=(q, V, N, J, H, T, frame_text, even_mask, hist_frames, hist_magn, hist_eng, magn_line, eng_line),
        frames=FRAMES,
        interval=50,
        blit=True
    )
    print(f"Saving {filename}...")
    ani.save(filename=filename, writer="pillow", fps=30, dpi=150)
    elapsed_time = time.time() - start_time
    mins, secs = divmod(elapsed_time, 60)
    inp = input(f"{filename.name} saved in {int(mins)}m {secs:.2f}s. Press enter to visualize, or type 'no': ")
    if inp.lower() != "no":
        os.system(f"xdg-open {filename.absolute()}") 

if __name__ == "__main__":
    main()
