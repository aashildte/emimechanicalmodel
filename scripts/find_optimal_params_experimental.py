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

def load_experimental_data_stretch(fin):
    data = np.loadtxt(fin, delimiter=",", skiprows=1)

    displacement = data[:,0] * 1E-3 # m
    force = data[:,1]               # N

    # approximately right
    width = 10*1E-3           # m
    area = width**2           # m2

    stretch = displacement / width           # fraction
    load = force / area *1E-3                # kPa

    # only consider tensile displacement for now
    i = 0
    while displacement[i] < 0:
        i+=1
    
    print("num steps: ", len(stretch[i:]))

    return stretch[i:], load[i:]

def load_experimental_data_shear(fin):
    data = np.loadtxt(fin, delimiter=",", skiprows=1)

    displacement = data[:,0] * 1E-3 # m
    normal_force = data[:,1]               # N
    shear_force = data[:,2]               # N

    # approximately right
    width = 10*1E-3           # m
    area = width**2           # m2

    stretch = displacement / width           # fraction
    normal_load = normal_force / area *1E-3                # kPa
    shear_load = shear_force / area *1E-3                # kPa

    # only consider tensile displacement for now
    i = 0
    while displacement[i] < 0:
        i+=1
    
    print("num steps: ", len(stretch[i:]))

    return stretch[i:], normal_load[i:], shear_load[i:]


def go_to_stretch(model, stretch, experiment):
    normal_loads = np.zeros_like(stretch)
    shear_loads = np.zeros_like(stretch)

    for (step, st) in enumerate(stretch):
        #print(f"Forward step : {step}/{len(stretch)}", flush=True)
        model.assign_stretch(st)
        model.solve()
    
        normal_loads[step] = model.evaluate_normal_load()
        shear_loads[step] = model.evaluate_shear_load()

    return normal_loads, shear_loads


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

def cost_function(params, experiments, experimental_data, models, i):
    a_i, b_i, a_e, b_e, a_if, b_if, a_esn, b_esn = params
    cf = 0

    print(f"Current mat param values: a_i = {a_i}, b_i = {b_i}, a_e = {a_e}, b_e = {b_e}, a_if = {a_if}, b_if = {b_if}, a_esn = {a_esn}, b_esn = {b_esn}")

    for experiment in experiments:
        #print("experiment: ", experiment)
        model = models[experiment]
        model.state.vector()[:] = 0       # reset
    
        model.mat_model.a_i.assign(a_i)
        model.mat_model.b_i.assign(b_i)
        model.mat_model.a_e.assign(a_e)
        model.mat_model.b_e.assign(b_e)
        model.mat_model.a_if.assign(a_if)
        model.mat_model.b_if.assign(b_if)
        model.mat_model.a_esn.assign(a_esn)
        model.mat_model.b_esn.assign(b_esn)

        stretch_values = experimental_data[experiment]["stretch_values"][:i]
        try:
            normal_load, shear_load = go_to_stretch(model, stretch_values, experiment)
        except RuntimeError:
            return np.inf

        target_normal_load = experimental_data[experiment]["normal_values"][:i]
        cf += (np.linalg.norm(normal_load - target_normal_load)**2) #/ np.linalg.norm(target_load)**2)
        
        target_shear_load = experimental_data[experiment]["shear_values"][:i]
        cf += (np.linalg.norm(shear_load - target_shear_load)**2) #/ np.linalg.norm(target_load)**2)

    print(f"Current cost fun value: {cf**0.5}")

    return cf**2


parser = ArgumentParser()
add_default_arguments(parser)
pp = parser.parse_args()

sample = "Sample1"

mesh_file = "meshes/tile_cubic.h5"
output_folder = pp.output_folder

mesh, volumes = load_mesh(mesh_file)

experimental_data = defaultdict(dict)

experiments = ["stretch_ff", "stretch_ss", "stretch_nn", "shear_fs", "shear_sf", "shear_fn", "shear_nf", "shear_sn", "shear_ns"]

for experiment in experiments:
    mode = experiment.split("_")[1].upper()
    
    fin = f"Data/LeftVentricle_MechanicalTesting/LeftVentricle_ForceDisplacement/LeftVentricle_{sample}/LeftVentricle_{sample}_{mode}.csv"

    if "stretch" in experiment:
        stretch_values, normal_values = load_experimental_data_stretch(fin)
        shear_values = -1*np.ones_like(normal_values)
    else:
        stretch_values, normal_values, shear_values = load_experimental_data_shear(fin)


    experimental_data[experiment]["stretch_values"] = stretch_values
    experimental_data[experiment]["normal_values"] = normal_values
    experimental_data[experiment]["shear_values"] = shear_values


a_i = 1.0 #1.074
b_i = 5.0 #4.878
a_e = 1.0 #1.074
b_e = 5.0 #4.878
a_if = 1.0 #2.628
b_if = 5.0 #5.214
a_esn = 1.0 #2.628
b_esn = 5.0 #5.214

emi_params = {
    "a_i": df.Constant(a_i),
    "b_i": df.Constant(b_i),
    "a_e": df.Constant(a_e),
    "b_e": df.Constant(b_e),
    "a_if": df.Constant(a_if),
    "b_if": df.Constant(b_if),
    "a_esn": df.Constant(a_esn),
    "b_esn": df.Constant(b_esn),
}

emi_models = {}

for experiment in experiments:
    model = initiate_emi_model(mesh, volumes, emi_params, experiment)
    emi_models[experiment] = model

params = [a_i, b_i, a_e, b_e, a_if, b_if, a_esn, b_esn]

num_attempts = 0

while num_attempts < 11:
    bounds = [(max(1E-2, p - 1), p + 1) for p in params]

    opt = minimize(
            cost_function,
            np.array(params),
            (experiments, experimental_data, emi_models, 5*num_attempts+5),
            bounds=bounds,
            )
    params = opt.x
    
    print(opt)
    print(params)
    num_attempts += 1
