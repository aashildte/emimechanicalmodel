"""

Åshild Telle / Simula Research Laboratory / 2022

This script initiates the second phase of the Sobol analysis.
It reads in a parameter combination from file, performs a virtual
experiment, and saves load and stress values to a new file.

"""


from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np
import dolfin as df
import argparse

from emimechanicalmodel import load_mesh, EMIModel

# Optimization options for the form compiler
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["representation"] = "uflacs"
df.parameters["form_compiler"]["quadrature_degree"] = 4
df.set_log_level(60)


def go_to_stretch(model, stretch):
    for (step, st) in enumerate(stretch):
        print(step, st, flush=True)
        model.assign_stretch(st)
        model.solve()


def go_to_contraction(model, active_values):
    for (step, st) in enumerate(active_values):
        print(step, st, flush=True)
        model.update_active_fn(st)
        model.solve()


def evaluate_model(X, model, stretch_values, contr):
    a_i, b_i, a_e, b_e, a_if, b_if = X

    model.mat_model.a_i.assign(a_i)
    model.mat_model.b_i.assign(b_i)
    model.mat_model.a_e.assign(a_e)
    model.mat_model.b_e.assign(b_e)
    model.mat_model.a_if.assign(a_if)
    model.mat_model.b_if.assign(b_if)

    model.state.vector()[:] = 0  # reset

    if contr:
        go_to_contraction(model, stretch_values)
    else:
        go_to_stretch(model, stretch_values)

    return np.array(
        [
            model.evaluate_normal_load(),
            model.evaluate_shear_load(),
            model.evaluate_subdomain_stress_fibre_dir(1),
            model.evaluate_subdomain_stress_transfibre_dir(1),
            model.evaluate_subdomain_stress_normal_dir(1),
            model.evaluate_subdomain_stress_fibre_dir(0),
            model.evaluate_subdomain_stress_transfibre_dir(0),
            model.evaluate_subdomain_stress_normal_dir(0),
        ]
    )


def init_model():
    mesh, volumes = load_mesh("meshes/tile_connected_5p0.h5")
    model = EMIModel(
        mesh,
        volumes,
        experiment=mode,
    )

    return model


def init_stretch(mode):
    if "stretch" in mode:
        stretch = np.linspace(0, 0.1, 50)
    elif "shear" in mode:
        stretch = np.linspace(0, 0.4, 1000)
    else:
        time = np.linspace(0, 137, 65)
        active_values, scaling_value = emi.compute_active_component(time)
        active_values *= 0.28 / scaling_value  # max 0.28
        stretch = active_values

    return stretch


def sobol_analysis(mode, i, input_folder, output_folder):
    model = init_model()
    stretch = init_stretch(mode)

    fname = f"{input_folder}/parameter_set_{i}.npy"
    X = np.load(fname)

    outputs = evaluate_model(X, model, stretch, mode == "contr")

    fout = f"{output_folder}/results_{mode}_{i}.npy"
    np.save(fout, np.array(outputs))


parser = argparse.ArgumentParser()

parser.add_argument(
    "-M",
    "--mode",
    type=str,
    default="contr",
    help='Deformation mode (valid options; "contr" \
                            "stretch_ff" "shear_fs" "shear_fn" "shear_sf" "stretch_ss" \
                            "shear_sn" "shear_nf" "shear_ns" "stretch_nn")',
)

parser.add_argument(
    "-i",
    "--variable_count",
    type=int,
    default=0,
    help="Parameter set # to use for this simulation.",
)

parser.add_argument(
    "-if",
    "--input_folder",
    type=str,
    default="sobol_analysis",
    help="Get all input files, i.e., all parameter combinations here",
)

parser.add_argument(
    "-of",
    "--output_folder",
    type=str,
    default="sobol_analysis",
    help="Save all output files, i.e., all resulting metrics here",
)

args = parser.parse_args()

mode = args.mode
i = args.variable_count
input_folder = parser.input_folder
output_folder = parser.output_folder

possible_modes = [
    "stretch_ff",
    "shear_fs",
    "shear_fn",
    "shear_sf",
    "stretch_ss",
    "shear_sn",
    "shear_nf",
    "shear_ns",
    "stretch_nn",
    "contr",
]
assert mode in possible_modes, "Error: Unknown mode"

sobol_analysis(mode, i, input_folder, output_folder)
