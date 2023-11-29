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
peak_index = num_steps - 1

# load mesh, subdomains

mesh = df.UnitSquareMesh(3, 3)
#mesh = df.UnitCubeMesh(3, 3, 3)

# initiate model
        
"""
material_params = {
    "a": 2.113,
    "b": 4.319,
    "a_f": 6.595,
    "b_f": 4.340, #4.340,
    "a_s": 0.00082, #0.00082,
    "b_s": 0.004,
    "a_fs": 0.393,
    "b_fs": 1.154,
}
"""

material_params = {
    "a": 0.059,
    "b": 8.023,
    "a_f": 18.472,
    "b_f": 16.026,
    "a_s": 2.481,
    "b_s": 11.120,
    "a_fs": 0.216,
    "b_fs": 11.436,
}


model = TissueModel(
    mesh,
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
np.save(f"schematics_2D/holzapfel_{experiment}.npy", load)
