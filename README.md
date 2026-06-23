# Classical 2D Ising model simulation in Python.

A Python implementation of the 2D classical Ising classical on a square lattice, featuring the Checkerboard Metropolis algorithm vectorized via Numpy, with static rendering of mp4 (Matplotlib) and real-time interactive simulation (PyQt5).

## Usage
Clone the repository and install the required dependencies:
```
git clone https://github.com/nicola-lamaddalena/classical-ising-model.git 
cd classical-ising-model
pip install -r requirements.txt
cd src
```
To generate a pre-rendered mp4 use:
```
python3 ising.py
```
To launch an interactive simulation use:
```
python3 live_ani.py
```

## Implementation
The Metropolis algorithm is implemented using the Checkerboard Metropolis, where the lattice is updated in two steps; this allows computing large lattices (up to $512 \times 512$) efficiently.

## Different configurations
In the *configs* folder there is a JSON file where new configurations can be inserted to visualize different situations. To add a new configuration, simply append your specifications following the structure already present.
To visualize a particular configuration use:
```
python3 ising.py -m name_of_configuration
```
Otherwise, new configurations can be explored using the appropriate flags. To see the available flags use:
```
python3 ising.py -h
```
or
```
python3 ising.py --help
```

## Demo
Here is a demo created in Matplotlib with lattice dimensions $128 \times 128$ at the critical temperature.
![Ising Model Simulation](./animations/crit_demo.mp4)

