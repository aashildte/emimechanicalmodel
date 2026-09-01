"""

Åshild Telle / Simula Research Laboratory / 2022

This script initiates the second phase of the Sobol analysis.
It reads in a parameter combination from file, performs a virtual
experiment, and saves load and stress values to a new file.

"""


from SALib.sample import saltelli
import numpy as np
import dolfin as df
import argparse

from emimechanicalmodel import (
    load_mesh_sarcomere,
    compute_active_component,
    SarcomereModel,
)

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# Optimization options for the form compiler
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["representation"] = "uflacs"
df.parameters["form_compiler"]["quadrature_degree"] = 4
df.set_log_level(60)




def go_to_contraction(model, active_values):
    for (step, st) in enumerate(active_values):
        print(step, st)
        model.update_active_fn(st)
        model.solve(project=False)
        model.evaluate_average_shortening()

def evaluate_model(model, active_transient):
    go_to_contraction(model, active_transient)
    
    return_values = [
            model.evaluate_average_shortening(),
            model.evaluate_normal_load(),
            ]
    
    for d in [1, 2, 3, 4, 5]:
        return_values += [
            model.evaluate_subdomain_stress_fiber_dir(d),
            model.evaluate_subdomain_stress_sheet_dir(d),
            model.evaluate_subdomain_stress_fiber_sheet_dir(d),
            model.evaluate_lambda(d),
            model.evaluate_lambda_T(d),
            model.evaluate_shear_angle(d)
        ]
    
    return return_values

def init_model(isometric, X, mesh_file):
    
    mesh, volumes, angles = load_mesh_sarcomere(mesh_file)
    nucleus_angles = np.load(mesh_file.split(".")[0] + "_nuclei_angles.npy", allow_pickle=True).item()["angles"]
    
    a_i_sarcomeres, a_i_zlines, a_i_cytoskeleton, a_i_connections, a_i_nucleus = X

    material_params = {
        "a_i_sarcomeres" : a_i_sarcomeres,
        "a_if_sarcomeres" : 5.0,
        "a_i_zlines": a_i_zlines,
        "a_i_connections" : a_i_connections,
        "a_i_cytoskeleton" : a_i_cytoskeleton,
        "a_if_cytoskeleton" : 5.0,
        "a_i_nucleus" : a_i_nucleus,
        }
    
    model = SarcomereModel(
        mesh,
        volumes,
        sarcomere_angles=angles,
        nucleus_angles=nucleus_angles,
        material_parameters=material_params,
        experiment="contraction",
        active_model="active_stress",
        compressibility_model="nearly_incompressible",
        isometric=isometric,
    )

    return model


def init_active_transient(sarcomere_scale):
    time = np.linspace(0, 137, 137)
    active_values = compute_active_component(time)
    active_values *= 750*sarcomere_scale

    return active_values

def sobol_analysis(meshfile, isometric, i, input_folder, output_folder):
    fname = f"{input_folder}/parameter_set_{i}.npy"
    
    X = np.load(fname)
    sarcomere_scale = X[0]
    
    model = init_model(isometric, X, meshfile)
    active_transient = init_active_transient(sarcomere_scale)       # active tension scales with sarcomere stiffness
    
    outputs = evaluate_model(model, active_transient)

    fout = f"{output_folder}/results_iso_{isometric}_{i}.npy"
    np.save(fout, np.array(outputs))


parser = argparse.ArgumentParser()
    
parser.add_argument("--isometric",
        action="store_true",
        default=False)

parser.add_argument(
    "-i",
    "--variable-count",
    type=int,
    default=0,
    help="Parameter set # to use for this simulation.",
)

parser.add_argument(
    "-if",
    "--input-folder",
    type=str,
    default="sobol_analysis",
    help="Get all input files, i.e., all parameter combinations here",
)

parser.add_argument(
    "-of",
    "--output-folder",
    type=str,
    default="sobol_analysis",
    help="Save all output files, i.e., all resulting metrics here",
)

parser.add_argument("--meshfile", type=str, default="meshes/cells_with_nucleus_v2/cellidealized_with_2nuclei.h5")

args = parser.parse_args()

isometric = args.isometric
i = args.variable_count
input_folder = args.input_folder
output_folder = args.output_folder
meshfile = args.meshfile

sobol_analysis(meshfile, isometric, i, input_folder, output_folder)
