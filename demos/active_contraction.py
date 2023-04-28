"""

Åshild Telle / Simula Research Labratory / 2022

Script for simulating active contraction; over (parts of) one cardiac cycle.
Demostrates how to run an active contraction, saving spatial output to a
given output folder.

"""


from argparse import ArgumentParser
import numpy as np
import dolfin as df
from mpi4py import MPI

from emimechanicalmodel import (
    load_mesh,
    compute_active_component,
    EMIModel,
    Monitor,
)


# compute active stress, pre-computed from the Rice model

time_max = 500
num_time_steps = 125
time = np.linspace(0, time_max, num_time_steps)  # ms
active_values = compute_active_component(time)

# load mesh, subdomains
mesh_file = "meshes/tile_connected_10p0.h5"
mesh, volumes = load_mesh(mesh_file)

# initiate EMI model

model = EMIModel(
    mesh,
    volumes,
    experiment="contraction",
)

# track spatial variables through a monitor, save to output_folder

output_folder = "demo_active_contraction"
monitor = Monitor(model, output_folder)


# then run the simulation
for i in range(num_time_steps):
    time_pt, a_str = time[i], active_values[i]

    if verbose >= 1 and MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Time step {i+1} / {num_time_steps}", flush=True)

    model.update_active_fn(a_str)
    model.solve(project=True)

    monitor.update_scalar_functions(time_pt)
    monitor.update_xdmf_files(i)


monitor.save_and_close()
