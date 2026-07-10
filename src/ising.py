import argparse
import json
import os
import subprocess
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
from dataclasses import dataclass, field

BKG_COLOR = "#E9EEF3"
DOWN_COLOR = "#c83e3e"
UP_COLOR = "#3b60e4"
MAIN_COLOR = "#191923"

ROOT = Path(__file__).resolve().parent.parent
ANIMATIONS_DIR = ROOT / "animations"
SRC_DIR = ROOT / "src"
CONFIGS_DIR = ROOT / "configs"
CONFIG_FILE = CONFIGS_DIR / "configs.json"
PROPS = {"boxstyle": "round", "facecolor": BKG_COLOR, "alpha": 0.9, "edgecolor": MAIN_COLOR}

ANIMATIONS_DIR.mkdir(exist_ok=True)
CONFIGS_DIR.mkdir(exist_ok=True)
CMAP = ListedColormap([DOWN_COLOR, UP_COLOR])

@dataclass
class State:
    N: int
    T: float
    J: float
    H: float
    V: np.ndarray
    even_mask: np.ndarray
    hist_frames: list = field(default_factory=list)
    hist_magn: list = field(default_factory=list)
    hist_eng: list = field(default_factory=list)

def animate(frame, q, state, frame_text, magn_line, eng_line, rng):
    steps_per_frame = 1
    for _ in range(steps_per_frame):
        metropolis(state.V, state.N, state.J, state.H, state.T, state.even_mask, rng)
        state.hist_frames.append(frame)
        state.hist_magn.append(magnetization(state.V))
        state.hist_eng.append(energy(state.V, state.J))
    q.set_data(state.V)
    magn_line.set_data(state.hist_frames, state.hist_magn)
    eng_line.set_data(state.hist_frames, state.hist_eng)

    frame_text.set_text(f"Frame: {frame}.")

    return q, frame_text, magn_line, eng_line

def configuration(config_file):
    if not config_file.exists():
        print(f"Error: {config_file} not found.")
        sys.exit(1)
    
    with open(config_file, "r") as f:
        config_data = json.load(f)
    return config_data

def pars_file(config_data):
    parser = argparse.ArgumentParser(
            description="Classical 2D Ising model simulation",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    choices = list(config_data.keys()) # update list of configurations 
    parser.add_argument("-M", "--mode", type=str, default="crit", help=f"Preset configurations ({choices})")
    parser.add_argument("-F", "--file", type=str, help="Choose a name for the animation file")
    parser.add_argument("--view", action="store_true", help="View existing animation without simulating")
    parser.add_argument("--seed", type=int, default=None,
                     help="Random generator seed (default: None)")
    # TODO change view implementation: 
    # "py ising.py -F test --view" should visualize the test animation
    # "py ising.py --view" should print existing animations 
    args = parser.parse_args()


    mode = args.mode
    conf = config_data[mode]
    N = conf["N"]
    T = conf["T"]
    J = conf["J"]
    H = conf["H"]
    FRAMES = conf["frames"]
    therm_steps = conf["therm_steps"]

    return args, mode, N, T, J, H, FRAMES, therm_steps

def init_simulation_state(N, J, T, H):
    V = np.random.choice([-1, 1], size=(N, N))
    even_mask = np.zeros((N, N), dtype=bool)
    even_mask[::2, ::2] = True # even columns and even rows
    even_mask[1::2, 1::2] = True # odd columns and odd rows
    return State(N=N, T=T, J=J, H=H, V=V, even_mask=even_mask)

def thermalize(state: State, therm_steps: int, rng: np.random.Generator):
    for _ in range(therm_steps):
        metropolis(state.V, state.N, state.J, state.H, state.T, state.even_mask, rng)

def open_media(filepath):
    filepath_str = str(filepath)
    
    try:
        if sys.platform == "win32":
            os.startfile(filepath_str)
        elif sys.platform == "darwin":
            subprocess.call(["open", filepath_str]) # macOS
        else:
            subprocess.call(["xdg-open", filepath_str])
    except Exception as e:
        print(f"Cannot open the file: {e}")

def setup_figure(V, N, J, T, FRAMES):
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
    frame_text = ax_main.text(0.02, 0.95, "", transform=ax_main.transAxes, color=MAIN_COLOR, bbox=PROPS)

    return fig, magn_line, eng_line, q, frame_text

def main():

    config_data = configuration(CONFIG_FILE)
    args, mode, N, T, J, H, FRAMES, therm_steps = pars_file(config_data)
    state = init_simulation_state(N, J, T, H)
    base_name = args.file if args.file else mode
    filename = ANIMATIONS_DIR / f"{Path(base_name).stem}.mp4"
    
    if filename.exists():
        print(f"'{filename.name}' found in {ANIMATIONS_DIR.name}. Visualizing...")
        open_media(filename.absolute())
        sys.exit(0)

    print(f"Filename determined: {filename.name}. Starting simulation with N={N}, T={T}, J={J}, h={H}.")
    fig, magn_line, eng_line, q, frame_text = setup_figure(state.V, state.N, state.J, state.T, FRAMES)

    start_time = time.time()
    print(f"Thermalizing with {therm_steps} steps.")
    thermalize(state, therm_steps, args.seed)
    
    ani = FuncAnimation(
        fig,
        animate,
        fargs=(q, state, frame_text, magn_line, eng_line, args.seed),
        frames=FRAMES,
        interval=50,
        blit=True
    )
    print(f"Saving {filename}...")
    ani.save(
            filename=filename, 
            writer="ffmpeg", 
            fps=30, 
            dpi=200,
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-vprofile", "main", "-level", "3.1"]
        )
    elapsed_time = time.time() - start_time
    mins, secs = divmod(elapsed_time, 60)
    inp = input(f"{filename.name} saved in {int(mins)}m {secs:.2f}s. Press enter to visualize, or type 'no': ")
    if inp.lower() != "no":
        open_media(filename.absolute())

if __name__ == "__main__":
    main()
