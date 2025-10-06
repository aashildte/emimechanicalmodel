"""

Åshild Telle / Simula Research Labratory / 2022

Script for simulating passive deformation, i.e. stretch and shear experiments.

"""

import os
from argparse import ArgumentParser
import numpy as np
import dolfin as df

from mpi4py import MPI


from emimechanicalmodel import (
    TissueModel,
)

from parameter_setup import (
    add_emi_holzapfel_arguments,
    add_default_arguments,
    add_stretching_arguments,
    setup_monitor,
)


def read_cl_args():

    parser = ArgumentParser()

    add_default_arguments(parser)
    add_stretching_arguments(parser)
    add_emi_holzapfel_arguments(parser)

    pp = parser.parse_args()

    return (
        pp.a_i,
        pp.b_i,
        pp.a_e,
        pp.b_e,
        pp.a_f,
        pp.b_f,
        pp.mesh_file,
        pp.output_folder,
        pp.dir_stretch,
        pp.strain,
        pp.num_steps,
        pp.plot_at_peak,
        pp.plot_all_steps,
        pp.verbose,
    )


# read in (relevant) parameters from the command line

(
    a_i,
    b_i,
    a_e,
    b_e,
    a_f,
    b_f,
    mesh_file,
    output_folder,
    experiment,
    strain,
    num_steps,
    plot_at_peak,
    plot_all_steps,
    verbose,
) = read_cl_args()


# stretch array; increase from 0 to max

stretch = np.linspace(0, strain, num_steps)

N_10 = 0
for i, s in enumerate(stretch):
    if s >= 0.1:
        N_10 = i
        break

peak_index = num_steps - 1

# load mesh, subdomains

#mesh = df.UnitSquareMesh(3, 3)
mesh = df.UnitCubeMesh(3, 3, 3)

# initiate model
        
material_params = {}

model = TissueModel(
    mesh,
    material_model="Bellini",
    material_parameters=material_params,
    experiment=experiment,
    verbose=verbose,
)

# setup parameters - define the parameter space to explore

# then run the simulation

load = np.zeros_like(stretch)

for (i, st_val) in enumerate(stretch):

    print(f"Step {i+1} / {num_steps}", flush=True)

    model.assign_stretch(st_val)

    project = (plot_all_steps) or (plot_at_peak and i == peak_index)
    model.solve(project=project)
    
    load[i] = model.evaluate_normal_load()
    print(load[i])

import matplotlib.pyplot as plt
plt.plot(stretch, load, color="black", label=r"$LA_{ant}$")
plt.plot(stretch[N_10], load[N_10], "o", color="black", label=r"$LA_{ant}$")

material_params = {
        "c_iso" : 1.52,
        "k1" :2.36,
        "k2" : 12.65,
        "k3" : 1.56,
        "k4" : 7.17,
}

model = TissueModel(
    mesh,
    material_model="Bellini",
    material_parameters=material_params,
    experiment=experiment,
    verbose=verbose,
)

# setup parameters - define the parameter space to explore

# then run the simulation

load = np.zeros_like(stretch)

for (i, st_val) in enumerate(stretch):

    print(f"Step {i+1} / {num_steps}", flush=True)

    model.assign_stretch(st_val)

    project = (plot_all_steps) or (plot_at_peak and i == peak_index)
    model.solve(project=project)
    
    load[i] = model.evaluate_normal_load()
    print(load[i])

plt.plot(stretch, load, "--", color="black", label=r"$LA_{post}$")
plt.plot(stretch[N_10], load[N_10], "o", color="black", label=r"$LA_{post}$")

material_params = {
        "c_iso" : 0.63,
        "k1" : 3.27,
        "k2" : 7.62,
        "k3" : 1.75,
        "k4" : 3.24,
}

model = TissueModel(
    mesh,
    material_model="Bellini",
    material_parameters=material_params,
    experiment=experiment,
    verbose=verbose,
)

# setup parameters - define the parameter space to explore

# then run the simulation

load = np.zeros_like(stretch)

for (i, st_val) in enumerate(stretch):

    print(f"Step {i+1} / {num_steps}", flush=True)

    model.assign_stretch(st_val)

    project = (plot_all_steps) or (plot_at_peak and i == peak_index)
    model.solve(project=project)
    
    load[i] = model.evaluate_normal_load()
    print(load[i])

plt.plot(stretch, load, "-", color="tab:red", label=r"$LA_{ant}$, VTP")
plt.plot(stretch[N_10], load[N_10], "o", color="tab:red", label=r"$LA_{ant}$, VTP")

material_params = {
        "c_iso" : 2.37,
        "k1" : 5.38,
        "k2" : 11.37,
        "k3" : 2.11,
        "k4" : 4.03,
}

model = TissueModel(
    mesh,
    material_model="Bellini",
    material_parameters=material_params,
    experiment=experiment,
    verbose=verbose,
)

# setup parameters - define the parameter space to explore

# then run the simulation

load = np.zeros_like(stretch)

for (i, st_val) in enumerate(stretch):

    print(f"Step {i+1} / {num_steps}", flush=True)

    model.assign_stretch(st_val)

    project = (plot_all_steps) or (plot_at_peak and i == peak_index)
    model.solve(project=project)
    
    load[i] = model.evaluate_normal_load()
    print(load[i])

plt.plot(stretch, load, "--", color="tab:red", label=r"$LA_{post}$, VTP")
plt.plot(stretch[N_10], load[N_10], "o", color="tab:red", label=r"$LA_{post}$, VTP")

plt.legend()

plt.show()
