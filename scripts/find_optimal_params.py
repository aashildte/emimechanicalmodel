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


def initiate_tissue_model(mesh, tissue_params, experiment):    
    model = TissueModel(
        mesh,
        experiment=experiment,
        material_parameters=tissue_params,
    )

    return model


def initiate_emi_model(mesh, volumes, emi_params, experiment):
    model = EMIModel(
        mesh,
        volumes,
        experiment=experiment,
        material_parameters=emi_params,
    )

    return model

def save_state_values(targets, experiments, models):
    original_states = defaultdict(dict)

    for target_stretch in targets:
        for experiment in experiments:
            model = models[target_stretch][experiment]
            state_values = model.state.vector()[:]
            original_states[target_stretch][experiment] = state_values

    return original_states

def values_to_states(targets, experiments, models, original_states):
    for target_stretch in targets:
        for experiment in experiments:
            model = models[target_stretch][experiment]
            state_values = original_states[target_stretch][experiment]
            model.state.vector()[:] = state_values

def cost_function(params, stretch_values, experiments, target_loads, models):
    a_i, b_i, a_e, b_e, a_if, b_if = params
    cf = 0

    print(f"Current mat param values: a_i = {a_i}, b_i = {b_i}, a_e = {a_e}, b_e = {b_e}, a_if = {a_if}, b_if = {b_if}")


    for experiment in experiments:
        model = models[experiment]
        model.state.vector()[:] = 0       # reset
    
        model.mat_model.a_i.assign(a_i)
        model.mat_model.b_i.assign(b_i)
        model.mat_model.a_e.assign(a_e)
        model.mat_model.b_e.assign(b_e)
        model.mat_model.a_if.assign(a_if)
        model.mat_model.b_if.assign(b_if)

        try:
            load = go_to_stretch(model, stretch_values)
        except RuntimeError:
            return np.inf

        
        target_load = target_loads[experiment]
        cf += (np.linalg.norm(load - target_load)**2) #/ np.linalg.norm(target_load)**2)

    print(f"Current cost fun value: {cf**0.5}")

    return cf**2


parser = ArgumentParser()
add_default_arguments(parser)
pp = parser.parse_args()

mesh_file = pp.mesh_file
output_folder = pp.output_folder

mesh, volumes = load_mesh(mesh_file)

tissue_params = {
    "a": df.Constant(0.074),
    "b": df.Constant(4.878),
    "a_f": df.Constant(2.628),
    "b_f": df.Constant(5.214),
    "a_s": df.Constant(0.438),
    "b_s": df.Constant(3.002),
    "a_fs": df.Constant(0.062),
    "b_fs": df.Constant(3.476),
}

target = 0.2
stretch_values = np.linspace(0, target, 20)

experiments = ["stretch_ff", "stretch_ss", "stretch_nn", "shear_fs", "shear_sf", "shear_ns", "shear_fn", "shear_sf", "shear_nf", "shear_fs", "shear_sn"]

target_loads = {}

for experiment in experiments:
    model = initiate_tissue_model(mesh, tissue_params, experiment)
    load_values = go_to_stretch(model, stretch_values)

    target_loads[experiment] = load_values

a_i = 1.074
b_i = 4.878
a_e = 1.074
b_e = 4.878
a_if = 2.628
b_if = 5.214

emi_params = {
    "a_i": df.Constant(a_i),
    "b_i": df.Constant(b_i),
    "a_e": df.Constant(a_e),
    "b_e": df.Constant(b_e),
    "a_if": df.Constant(a_if),
    "b_if": df.Constant(b_if),
}
emi_models = {}

for experiment in experiments:
    model = initiate_emi_model(mesh, volumes, emi_params, experiment)
    emi_models[experiment] = model

params = [a_i, b_i, a_e, b_e, a_if, b_if]

# this outer loop is to avoid divergence issues
num_attempts = 0
while num_attempts < len(stretch_values) - 2:
    #bounds = [(max(1E-2, p - 1), p + 1) for p in params]
    bounds = [(0.01, 10) for p in params]

    opt = minimize(
            cost_function,
            np.array(params),
            (stretch_values, experiments, target_loads, emi_models),
            bounds=bounds,
            )
    params = opt.x
    num_attempts += 1

    print("Optimization problem nr : ", num_attempts)
    print(opt)
