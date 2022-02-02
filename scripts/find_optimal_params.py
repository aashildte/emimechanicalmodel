"""

Script for estimating optimal values for a_i, b_i, a_e, b_e, a_if and b_if.

"""


import os

from argparse import ArgumentParser
import numpy as np
import dolfin as df
from scipy.optimize import minimize

from functools import partial

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


def go_to_stretch(model, stretch, xdir):

    loads = np.zeros_like(stretch)

    for (step, st) in enumerate(stretch):
        print(f"Forward step : {step}/{len(stretch)}", flush=True)
        model.assign_stretch(st)
        model.solve()
        # TODO we need this automathed
        if xdir:
            loads[step] = model.evaluate_load_yz()
        else:
            loads[step] = model.evaluate_load_xz()

    return loads[-1]

def initiate_models(tissue_params, emi_params):
    
    tissue_models = []
    emi_models = []

    experiments = ["xstretch"] #, "ystretch", "xystretch", "yzstretch"]

    for experiment in experiments:

        tissue_model = TissueModel(
            mesh,
            experiment=experiment,
            material_parameters=tissue_params,
        )

        emi_model = EMIModel(
            mesh,
            volumes,
            experiment=experiment,
            material_parameters=emi_params,
        )
        tissue_models.append(tissue_model)
        emi_models.append(emi_model)

    return tissue_models, emi_models


def cost_function(x0, target_loads, emi_models):

    cf = 0

    for (model, target) in zip(emi_models, target_loads):
        model.mat_model.set_a_if(x0)
        model.solve()
        load_emi = model.evaluate_load_yz()

        cf += (load_emi - target)**2

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

x0 = 2.628

emi_params = {
    "a_i": 0.074,
    "b_i": 4.878,
    "a_e": 1,
    "b_e": 10,
    "a_if": x0,
    "b_if": 5.214,
}

tissue_models, emi_models = initiate_models(
    tissue_params, emi_params
)

target_stretch = 0.05
stretch = np.arange(0, target_stretch, 0.01)

tissue_load = go_to_stretch(tissue_models[0], stretch, xdir=True)
emi_load = go_to_stretch(emi_models[0], stretch, xdir=True)

tissue_loads = [tissue_load]
emi_loads = [emi_load]

print("Init values EMI:", emi_loads, flush=True)
print("Init values Tissue:", tissue_loads, flush=True)

opt = minimize(cost_function, x0, (tissue_loads, emi_models))

print(opt)
