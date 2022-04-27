"""

Åshild Telle / Simula Research Labratory / 2021

Script for simulating uniaxial stretching, in both directions.

This is used in Fig. 7, 8, 10, 11 and 14 in the paper.

"""

import os
from argparse import ArgumentParser
import numpy as np
import dolfin as df

from mpi4py import MPI


from emimechanicalmodel import (
    load_mesh,
    EMIModel,
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

mesh, volumes = load_mesh(mesh_file, verbose)

# initiate EMI model

material_params = {
    "a_i": a_i,
    "b_i": b_i,
    "a_e": a_e,
    "b_e": b_e,
    "a_if": a_f,
    "b_if": b_f,
}

model = EMIModel(
    mesh,
    volumes,
    material_parameters=material_params,
    experiment=experiment,
    verbose=verbose,
)

# setup parameters - define the parameter space to explore

enable_monitor = bool(output_folder)  # save output if != None

if enable_monitor:
    monitor = setup_monitor(
        f"stretch_emi_{experiment}",
        output_folder,
        model,
        mesh_file,
        material_params,
        num_steps,
        strain,
    )
else:
    monitor = None

if verbose < 2:
    df.set_log_level(60)  # remove information about convergence

# then run the simulation

for (i, st_val) in enumerate(stretch):

    if verbose >= 1 and MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Step {i+1} / {num_steps}", flush=True)

    model.assign_stretch(st_val)

    project = (plot_all_steps) or (plot_at_peak and i == peak_index)
    model.solve(project=project)

    if enable_monitor:
        monitor.update_scalar_functions(st_val)
        if project:
            monitor.update_xdmf_files(i)

if enable_monitor:
    monitor.save_and_close()
