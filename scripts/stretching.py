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
        pp.a_s,
        pp.b_s,
        pp.a_fs,
        pp.b_fs,
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
    a_s,
    b_s,
    a_fs,
    b_fs,
    mesh_file,
    output_folder,
    dir_stretch,
    strain,
    num_steps,
    plot_at_peak,
    plot_all_steps,
    verbose,
) = read_cl_args()


# stretch array; increase from 0 to max

stretch = np.linspace(0, strain, num_steps)
peak_index = num_steps - 1

assert dir_stretch in ["xdir", "ydir"], "Error: set 'd' to be 'xdir' or 'ydir'"

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
    "a_is": a_s,
    "b_is": b_s,
    "a_ifs": a_fs,
    "b_ifs": b_fs,
}


if dir_stretch == "xdir":
    experiment = "xstretch"
else:
    experiment = "ystretch"

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
        f"stretch_emi_{dir_stretch}",
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
    model.solve(project=enable_monitor)

    if enable_monitor:
        monitor.update_scalar_functions(st_val)

        if plot_all_steps or (plot_at_peak and i == peak_index):
            monitor.update_xdmf_files(st_val)

if enable_monitor:
    monitor.save_and_close()
