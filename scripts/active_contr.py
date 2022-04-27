#!/usr/bin/env python3


"""

Åshild Telle / Simula Research Labratory / 2020

Script for simulating active contraction; over (parts of) one cardiac cycle.

This is used for all experiments involving active contraction in the paper.

"""

from argparse import ArgumentParser
import numpy as np
import dolfin as df
from mpi4py import MPI


from emimechanicalmodel import (
    load_mesh,
    compute_active_component,
    EMIModel,
)

from parameter_setup import (
    add_emi_holzapfel_arguments,
    add_default_arguments,
    add_active_arguments,
    setup_monitor,
)

def read_cl_args():

    parser = ArgumentParser()

    add_default_arguments(parser)
    add_active_arguments(parser)
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
        pp.time_max,
        pp.num_time_steps,
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
    a_if,
    b_if,
    mesh_file,
    output_folder,
    time_max,
    num_time_steps,
    plot_at_peak,
    plot_all_steps,
    verbose,
) = read_cl_args()

# compute active stress, given from the Rice model

time = np.linspace(0, time_max, num_time_steps)  # ms
active_values, scaling_value = compute_active_component(time)
active_values *= 0.28 / scaling_value  # max 0.28
peak_index = np.argmax(active_values)

# load mesh, subdomains

mesh, volumes = load_mesh(mesh_file, verbose)

# initiate EMI model

material_params = {
    "a_i": a_i,
    "b_i": b_i,
    "a_e": a_e,
    "b_e": b_e,
    "a_if": a_if,
    "b_if": b_if,
}

model = EMIModel(
    mesh,
    volumes,
    material_parameters=material_params,
    experiment="contr",
    verbose=verbose,
)

enable_monitor = bool(output_folder)  # save output if != None

if enable_monitor:
    monitor = setup_monitor(
        "active_emi",
        output_folder,
        model,
        mesh_file,
        material_params,
        num_time_steps,
        time_max,
    )
else:
    monitor = None

if verbose < 2:
    df.set_log_level(60)  # remove information about convergence

# then run the simulation
for i in range(num_time_steps):
    time_pt, a_str = time[i], active_values[i]

    if verbose >= 1 and MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Time step {i+1} / {num_time_steps}", flush=True)
    
    project = plot_all_steps or (plot_at_peak and i == peak_index)

    model.update_active_fn(a_str)
    model.solve(project=project)

    if enable_monitor:
        monitor.update_scalar_functions(time_pt)

        if project:
            monitor.update_xdmf_files(i)

if enable_monitor:
    monitor.save_and_close()
