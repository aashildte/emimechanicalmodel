"""

Script for estimating optimal values for a_i, b_i, a_e, b_e, a_if and b_if.

"""


import os

from argparse import ArgumentParser
import numpy as np
import dolfin as df
from scipy.optimize import minimize

from functools import partial
from collections import defaultdict

from emimechanicalmodel import (
    load_mesh,
    TissueModel,
    EMIModel,
)

from parameter_setup import (
    add_default_arguments,
)


# Optimization options for the form compiler
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["representation"] = "uflacs"
df.parameters["form_compiler"]["quadrature_degree"] = 4

df.set_log_level(60)

def go_to_stretch(model, stretch):
    loads = np.zeros_like(stretch)

    for (step, st) in enumerate(stretch):
        #print(f"Forward step : {step}/{len(stretch)}", flush=True)
        model.assign_stretch(st)
        model.solve()
    
        loads[step] = model.evaluate_load()

    return loads


def initiate_tissue_model(tissue_params, experiment):    
    model = TissueModel(
        mesh,
        experiment=experiment,
        material_parameters=tissue_params,
    )

    return model


def initiate_emi_model(emi_params, experiment):
    model = EMIModel(
        mesh,
        volumes,
        experiment=experiment,
        material_parameters=emi_params,
    )

    return model

def cost_function(material_parameters, target_loads, stretch_values, experiments, mesh, volume):
    b = material_parameters[0]
    cf = 0

    tissue_params = {
        "a": 0.074,
        "b": b,
        "a_f": 2.628,
        "b_f": 5.214,
        "a_s": 0.438,
        "b_s": 3.002,
        "a_fs": 0.062,
        "b_fs": 3.476,
    }


    print("current values: ", tissue_params)

    cf = 0

    for (i, experiment) in enumerate(experiments):

        print(f"Experiment {i}: {experiment}")

        try:
            model = initiate_tissue_model(tissue_params, experiment) 
            emi_load = go_to_stretch(model, stretch_values)
        except RuntimeError:
            return np.inf

        cf += np.linalg.norm(target_loads[experiment] - emi_load)

    print(f"Current cost fun value: {cf**0.5}")

    return cf**0.5


parser = ArgumentParser()
add_default_arguments(parser)
pp = parser.parse_args()

mesh_file = pp.mesh_file
output_folder = pp.output_folder

mesh, volumes = load_mesh(mesh_file)

tissue_params = {
    "a": 0.074,
    "b": 4.878,
    "a_f": 2.628,
    "b_f": 5.214,
    "a_s": 0.438,
    "b_s": 3.002,
    "a_fs": 0.062,
    "b_fs": 3.476,
}

target_stretch = 0.05
stretch_values = np.linspace(0, target_stretch, 5)

experiments = ["stretch_ff", "stretch_ss"] #, "stretch_nn", "shear_ns", "shear_fn", "shear_sf", "shear_nf", "shear_fs", "shear_sn"]
experiments.reverse()

tissue_models = []
target_loads = defaultdict(dict)

for experiment in experiments:
    model = initiate_tissue_model(tissue_params, experiment)
    load_values = go_to_stretch(model, stretch_values)

    target_loads[experiment] = load_values

print("Init values tissue:", target_loads, flush=True)

#a_i = 0.074
#b_i = 4.878
#a_e = 0.074
#b_e = 4.878
#a_if = 2.628
#b_if = 5.214


a = 1
b = 10
af = 1
bf = 10

bounds = [(0, np.inf) for _ in range(1)]

opt = minimize(cost_function, np.array((b,)), (target_loads, stretch_values, experiments, mesh, volumes), bounds=bounds)
print(opt)
