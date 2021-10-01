#!/usr/bin/env python3


"""

Åshild Telle / Simula Research Labratory / 2021

Script for exploring parameter uniqueness for the material parameters assigned to our EMI model.

This script is used for Fig. 6 in the paper.

"""

import os
from argparse import ArgumentParser
import numpy as np
import dolfin as df
from mpi4py import MPI

from emimechanicalmodel import (
    load_mesh,
    EMIModel,
    assign_discrete_values,
)

from parameter_setup import (
    add_default_arguments,
    add_stretching_arguments,
)


from mat_model_params import add_emi_holzapfel_arguments


def read_cl_args():

    parser = ArgumentParser()

    parser.add_argument(
        "--stage",
        default=1,
        type=int,
        help="Stage of parameter change (default: 1)",
    )

    parser.add_argument(
        "--factor_stage_2",
        default=0,
        type=int,
        help="Change of af/bf/as/bs parameter (default: 0)",
    )

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
        pp.verbose,
        pp.stage,
        pp.factor_stage_2,
    )

def setup_output_folders(output_folder, dir_strech, mesh_file, strain):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    _, meshid = os.path.split(mesh_file)
    meshid = meshid.split(".")[0]

    output_folder += f"/mesh_{meshid}_strain_{strain}_dir_{dir_stretch}"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    return output_folder


def init_EMI_model(mesh_file, dir_strech, verbose, material_params, change):
    mesh, volumes = load_mesh(mesh_file, verbose)

    if dir_stretch == "xdir":
        experiment = "xstretch"
        material_params["a_if"] = a_f * (1 - change)
        material_params["b_if"] = b_f * (1 - change)
    else:
        experiment = "ystretch"
        material_params["a_is"] = a_s * (1 - change)
        material_params["b_is"] = b_s * (1 - change)

    model = EMIModel(
        mesh,
        volumes,
        experiment,
        material_parameters=material_params,
        verbose=verbose,
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
    verbose,
    stage,
    factor_ind,
) = read_cl_args()
    
material_params = {
    "a_i": a_i,
    "b_i": b_i,
    "a_e": a_e * (1 - change),
    "b_e": b_e * (1 - change),
    "a_if": a_f,
    "b_if": b_f,
    "a_is": a_s,
    "b_is": b_s,
    "a_ifs": a_fs,
    "b_ifs": b_fs,
}

assert output_folder is not None, "Error: Please specify --output_folder (-o)"
assert dir_stretch in ["xdir", "ydir"], "Error: set 'd' to be 'xdir' or 'ydir'"

change = 0.25

output_folder = setup_output_folders(output_folder, dir_stretch, mesh_file, strain)
model = init_EMI_model(mesh_file, dir_stretch, verbose, material_params, change)
subdomain_map = model.subdomain_map

# stretch array; increase from 0 to max

stretch = np.linspace(0, strain, num_steps)
mul_factor = np.linspace(1 - change, 1 + change, 501)

# setup parameters - define the parameter space to explore

if verbose < 2:
    df.set_log_level(60)  # remove information about convergence

if stage == 1:
    for (i, st_val) in enumerate(stretch):

        print(f"Step {i+1} / {num_steps}", flush=True)

        model.assign_stretch(st_val)
        model.solve()

    for (j, factor) in enumerate(mul_factor):
        print(f"Step {j}/{len(mul_factor)}", flush=True)
        if dir_stretch == "xdir":
            assign_discrete_values(model.mat_model.a_f, subdomain_map, factor * a_f, 0)
            model.mat_model.b_f.vector()[:] = factor * b_f
        else:
            assign_discrete_values(model.mat_model.a_s, subdomain_map, factor * a_s, 0)
            model.mat_model.b_s.vector()[:] = factor * b_s

        model.solve()

        filename = (
            f"{output_folder}/uniqueness_{dir_stretch}_stage_{stage}_factor_{j}.h5"
        )

        sol_file = df.HDF5File(MPI.COMM_WORLD, filename, "w")
        sol_file.write(model.state, "/solution")
        sol_file.close()

elif stage == 2:
    model.assign_stretch(strain)

    if dir_stretch == "xdir":
        assign_discrete_values(
            model.mat_model.a_f, subdomain_map, mul_factor[factor_ind] * a_f, 0
        )
        model.mat_model.b_f.vector()[:] = mul_factor[factor_ind] * b_f
    elif dir_stretch == "ydir":
        assign_discrete_values(
            model.mat_model.a_s, subdomain_map, mul_factor[factor_ind] * a_s, 0
        )
        model.mat_model.b_s.vector()[:] = mul_factor[factor_ind] * b_s

    loads = np.zeros_like(mul_factor)

    filename = f"{output_folder}/uniqueness_{dir_stretch}_stage_{stage-1}_factor_{factor_ind}.h5"

    sol_file = df.HDF5File(MPI.COMM_WORLD, filename, "r")
    sol_file.read(model.state, "/solution")
    sol_file.close()

    for (j, factor) in enumerate(mul_factor):
        print(f"Step {j}/{len(mul_factor)}", flush=True)

        assign_discrete_values(model.mat_model.a, subdomain_map, a_i, a_e * factor)
        assign_discrete_values(model.mat_model.b, subdomain_map, b_i, b_e * factor)

        model.solve()

        if dir_stretch == "xdir":
            loads[j] = model.evaluate_load_yz()
        elif dir_stretch == "ydir":
            loads[j] = model.evaluate_load_xz()

    results_folder = f"{output_folder}/results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    output_filename = f"{results_folder}/uniqueness_{factor_ind}.npy"
    np.save(output_filename, loads)
