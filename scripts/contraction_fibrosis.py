"""

Åshild Telle / Simula Research Labratory / 2022


"""

import os
from argparse import ArgumentParser
import numpy as np
import dolfin as df
import sys

from mpi4py import MPI

from emimechanicalmodel.mesh_setup import write_collagen_to_file

from emimechanicalmodel import (
    compute_active_component,
    load_mesh,
    load_mesh_with_collagen_structure,
    EMIModelCollagen,
)

from parameter_setup import (
    add_emi_holzapfel_arguments,
    add_default_arguments,
    add_stretching_arguments,
    setup_monitor,
)


# stretch array; increase from 0 to max

import sys
mesh_file = sys.argv[1] # "meshes_fibrosis_paper/baseline_geometry.h5"
experiment = "contraction"

time_max = int(sys.argv[2])
num_time_steps = time_max 
time = np.linspace(0, time_max, num_time_steps)

# compute active stress, given from the Rice model

active_values = compute_active_component(time)
active_values *= 1200

scale_i = float(sys.argv[3])
scale_e = float(sys.argv[4])

# load mesh, subdomains

mesh, volumes, collagen = load_mesh_with_collagen_structure(mesh_file)
a_i=scale_i*1.0
b_i=20.0
a_if=scale_i*4.0
b_if=30.0

a_e_endo=scale_e*0.5
a_ef_endo=scale_e*0.5
a_e_peri=scale_e*1.0
a_ef_peri=scale_e*4.0

b_e=30.0
b_ef=30.0
b_ef_peri=40.0


material_params = {
    "a_i": a_i,
    "b_i": b_i,
    "a_e_endo": a_e_endo,
    "b_e_endo": b_e,
    "a_e_peri": a_e_peri,
    "b_e_peri": b_e,
    "a_if": a_if,
    "b_if": b_if,
    "a_ef_endo": a_ef_endo,
    "b_ef_endo": b_ef,
    "a_ef_peri": a_ef_peri,
    "b_ef_peri": b_ef_peri,
    "collagen_dist" : collagen,
}
# initiate EMI model

model = EMIModelCollagen(
    mesh,
    volumes,
    active_model="active_stress",
    material_parameters=material_params,
    material_model="holzapfel_collagen",
    experiment=experiment,
    #robin_bcs_value=0.1,
)

# setup parameters - define the parameter space to explore
output_folder = "/data1/aashild/EMI/fibrosis_paper_revisions_Sept22"

mat_folder = material_params.copy()
del mat_folder["collagen_dist"]


monitor = setup_monitor(
        f"active_{experiment}",
        output_folder,
        model,
        mesh_file,
        mat_folder,
        num_time_steps,
        time_max,
)

for i in range(num_time_steps):
    time_pt, a_str = time[i], active_values[i]

    print(f"Time step {i+1} / {num_time_steps}", flush=True)

    model.update_active_fn(a_str)

    peak_time = 136
    project = (i==peak_time)
    model.solve(project=project)

    if project:
        monitor.update_xdmf_files(i)

    monitor.update_scalar_functions(time_pt)


monitor.save_and_close()

