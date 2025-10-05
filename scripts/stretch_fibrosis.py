"""

Åshild Telle / Simula Research Labratory / 2022

Script for simulating passive deformation, i.e. stretch and shear experiments.

"""

import os
from argparse import ArgumentParser
import numpy as np
import dolfin as df
import sys

from mpi4py import MPI

from emimechanicalmodel.mesh_setup import write_collagen_to_file

from emimechanicalmodel import (
    load_mesh_with_collagen_structure,
    EMIModel,
)

from parameter_setup import (
    add_emi_holzapfel_arguments,
    add_default_arguments,
    add_stretching_arguments,
    setup_monitor,
)


# stretch array; increase from 0 to max

mesh_file = sys.argv[1]
experiment = sys.argv[2]
scale_i = float(sys.argv[3])
scale_e = float(sys.argv[4])

if "stretch" in experiment:
    strain = 0.15
else:
    strain = 0.4

num_steps = 31

stretch = np.linspace(0, strain, num_steps)

# load mesh, subdomains

mesh, volumes, collagen = load_mesh_with_collagen_structure(mesh_file)

# initiate EMI model
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


model = EMIModel(
    mesh,
    volumes,
    experiment=experiment,
    material_parameters=material_params,
    material_model="holzapfel_collagen",
)
# setup parameters - define the parameter space to explore

output_folder = "/data1/aashild/EMI/fibrosis_paper_revisions_Sept13"

mat_folder = material_params.copy()
del mat_folder["collagen_dist"]

monitor = setup_monitor(
        f"passive_{experiment}",
        output_folder,
        model,
        mesh_file,
        mat_folder,
        num_steps,
        strain,
)
# then run the simulation
load = np.zeros_like(stretch)

for (i, st_val) in enumerate(stretch):

    print(f"Step {i+1} / {num_steps}", flush=True)

    model.assign_stretch(st_val)
    
    project = (i==num_steps-1)

    model.solve(project=project)

    monitor.update_scalar_functions(st_val)
    if project:
        monitor.update_xdmf_files(i)

    load[i] = model.evaluate_normal_load()


monitor.save_and_close()

#import matplotlib.pyplot as plt
#plt.ylim(-2, 10)
#plt.plot(stretch, load)
#plt.show()

