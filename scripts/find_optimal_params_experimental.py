"""

Script for estimating optimal values for a_i, b_i, a_e, b_e, a_if and b_if.

"""


import os
import sys

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

def get_dimensions(fin):
    data_all = np.loadtxt(fin, delimiter=",", skiprows=2, usecols=range(1,10))

    dimensions = defaultdict(dict)
    areas = defaultdict(dict)

    for sample in range(1, 12):
        data = data_all[sample - 1]
        dimensions["FN"] = data[0]
        dimensions["FS"] = data[1]
        dimensions["FF"] = data[2]
        dimensions["SN"] = data[3]
        dimensions["SF"] = data[4]
        dimensions["SS"] = data[5]
        dimensions["NS"] = data[6]
        dimensions["NF"] = data[7]
        dimensions["NN"] = data[8]
        
        areas["FN"] = data[1]*data[2]
        areas["FS"] = data[0]*data[2]
        areas["FF"] = data[0]*data[1]
        
        areas["SN"] = data[4]*data[5]
        areas["SF"] = data[3]*data[5]
        areas["SS"] = data[3]*data[4]

        areas["NS"] = data[7]*data[8]
        areas["NF"] = data[6]*data[8]
        areas["NN"] = data[6]*data[7]

    return dimensions, areas

def load_experimental_data_stretch(fin, width, area):
    data = np.loadtxt(fin, delimiter=",", skiprows=1)

    displacement = data[:,0] * 1E-3 # m
    force = data[:,1]               # N

    # approximately right
    width *= 1E-3             # m
    area  *= 1E-6             # m2

    stretch = displacement / width           # fraction
    load = force / area *1E-3                # kPa

    # only consider tensile displacement for now
    i = 0
    while displacement[i] < 0:
        i+=1
    
    return stretch[i:], load[i:]

def load_experimental_data_shear(fin, width, area):
    data = np.loadtxt(fin, delimiter=",", skiprows=1)

    displacement = data[:,0] * 1E-3 # m
    shear_force = data[:,1]               # N
    normal_force = data[:,2]               # N

    # approximately right
    width *= 1E-3             # m
    area  *= 1E-6             # m2

    stretch = displacement / width           # fraction
    normal_load = normal_force / area *1E-3                # kPa
    shear_load = shear_force / area *1E-3                # kPa

    # only consider tensile displacement for now
    i = 0
    while displacement[i] < 0:
        i+=1
    
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

def cost_function(params, experiments, experimental_data, models):
    a_i, b_i, a_e, b_e, a_if, b_if = params
    cf = 0

    print(f"Current mat param values: a_i = {a_i}, b_i = {b_i}, a_e = {a_e}, b_e = {b_e}, a_if = {a_if}, b_if = {b_if}", flush=True)

    for experiment in experiments:
        print("experiment: ", experiment)
        model = models[experiment]
        model.state.vector()[:] = 0       # reset
    
        model.mat_model.a_i.assign(a_i)
        model.mat_model.b_i.assign(b_i)
        model.mat_model.a_e.assign(a_e)
        model.mat_model.b_e.assign(b_e)
        model.mat_model.a_if.assign(a_if)
        model.mat_model.b_if.assign(b_if)

        stretch_values = experimental_data[experiment]["stretch_values"]
        try:
            normal_load, shear_load = go_to_stretch(model, stretch_values, experiment)
        except RuntimeError:
            cf += 500 + 10*np.random.random()  # just a high number
            continue

        target_normal_load = experimental_data[experiment]["normal_values"]
        cf += (np.linalg.norm(normal_load - target_normal_load)**2)
        
        target_shear_load = experimental_data[experiment]["shear_values"]
        cf += (np.linalg.norm(shear_load - target_shear_load)**2)

    print(f"Current cost fun value: {cf**0.5}", flush=True)

    return cf**0.5

fname_dims = "Data/LeftVentricle_MechanicalTesting/LeftVentricle_Dimensions_mm.csv"
dimensions, areas = get_dimensions(fname_dims)

sample = sys.argv[1] #"Sample1"
print(f"Running experiments for sample {sample}", flush=True)

#mesh_file = f"meshes/tile_5.0.h5"
mesh_file = f"meshes/tile_connected.h5"

mesh, volumes = load_mesh(mesh_file)

experimental_data = defaultdict(dict)

experiments = ["stretch_ff", "stretch_ss", "stretch_nn", "shear_fs", "shear_sf", "shear_fn", "shear_nf", "shear_sn", "shear_ns"]

for experiment in experiments:
    mode = experiment.split("_")[1].upper()
    
    fin = f"Data/LeftVentricle_MechanicalTesting/LeftVentricle_ForceDisplacement/LeftVentricle_{sample}/LeftVentricle_{sample}_{mode}.csv"

    if "stretch" in experiment:
        stretch_values, normal_values = load_experimental_data_stretch(fin, dimensions[mode], areas[mode])
        shear_values = -1*np.ones_like(normal_values)
    else:
        stretch_values, normal_values, shear_values = load_experimental_data_shear(fin, dimensions[mode], areas[mode])


    experimental_data[experiment]["stretch_values"] = stretch_values
    experimental_data[experiment]["normal_values"] = normal_values
    experimental_data[experiment]["shear_values"] = shear_values

a_i = 1.0
b_i = 15.0
a_e = 1.0
b_e = 15.0
a_if = 1.0
b_if = 15.0

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

params = [a_i, b_i, a_e, b_e, a_if, b_if, a_esn, b_esn]

bounds = [(0.01, 100), (0.01, 100), (0.01, 100), (0.01, 100), (0.0, 100), (0.01, 100)]

opt = minimize(
        cost_function,
        np.array(params),
        (experiments, experimental_data, emi_models),
        bounds=bounds,
        )
params = opt.x

print(opt)
print(params)
