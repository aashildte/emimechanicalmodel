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

    print(dimensions)
    print(areas)

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
    
    print("num steps: ", len(stretch[i:]))

    return stretch[i:], load[i:]

def load_experimental_data_shear(fin, width, area):
    data = np.loadtxt(fin, delimiter=",", skiprows=1)

    displacement = data[:,0] * 1E-3 # m
    normal_force = data[:,2]               # N
    shear_force = data[:,1]               # N

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


def initiate_emi_model(mesh, emi_params, experiment):
    model = TissueModel(
        mesh,
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

def cost_function(params, experiments, experimental_data, models, norm):
    a, b, a_f, b_f, a_s, b_s, a_fs, b_fs = params
    cf = 0

    print(f"Current mat param values: a {a}, b {b}, a_f {a_f}, b_f {b_f}, a_s {a_s}, b_s {b_s}, a_fs {a_fs}, b_fs {b_fs}", flush=True)

    for experiment in experiments:
        #print("experiment: ", experiment)
        model = models[experiment]
        model.state.vector()[:] = 0       # reset
    
        model.mat_model.a.assign(a)
        model.mat_model.b.assign(b)
        model.mat_model.a_f.assign(a_f)
        model.mat_model.b_f.assign(b_f)
        model.mat_model.a_s.assign(a_s)
        model.mat_model.b_s.assign(b_s)
        model.mat_model.a_fs.assign(a_fs)
        model.mat_model.b_fs.assign(b_fs)

        stretch_values = experimental_data[experiment]["stretch_values"]
        try:
            normal_load, shear_load = go_to_stretch(model, stretch_values, experiment)
        except IndentationError:
            print("crashed; adding + 500")
            print(cf)
            cf += 500      # just a high value
            continue

        target_normal_load = experimental_data[experiment]["normal_values"]

        if norm == "L1":
            cf += np.sum(np.abs(normal_load - target_normal_load))
        elif norm == "L2":
            cf += np.linalg.norm(normal_load - target_normal_load)
        elif norm == "L2sq":
            cf += np.linalg.norm(normal_load - target_normal_load)**2
        
        target_shear_load = experimental_data[experiment]["shear_values"]
        
        if norm == "L1":
            cf += np.sum(np.abs(shear_load - target_shear_load))
        elif norm == "L2":
            cf += np.linalg.norm(shear_load - target_shear_load)
        elif norm == "L2sq":
            cf += np.linalg.norm(shear_load - target_shear_load)**2


    if norm == "L2sq":
        print("check: ")
        print(cf)
        print(cf**0.5)

        print(f"Current cost fun value: {cf**0.5}", flush=True)
        return cf**0.5
    else:
        print(f"Current cost fun value: {cf}", flush=True)
        return cf


fname_dims = "Data/LeftVentricle_MechanicalTesting/LeftVentricle_Dimensions_mm.csv"
dimensions, areas = get_dimensions(fname_dims)

sample = "Sample" + sys.argv[1]
print(f"Running experiments for sample {sample}", flush=True)

norm = sys.argv[2]

mesh = df.UnitCubeMesh(3, 3, 3)

experimental_data = defaultdict(dict)

experiments = ["stretch_ff", "stretch_ss", "stretch_nn", "shear_fs", "shear_sf", "shear_fn", "shear_nf", "shear_sn", "shear_ns"]

for experiment in experiments:
    mode = experiment.split("_")[1].upper()

    width = dimensions[mode]
    area = areas[mode]

    fin = f"Data/LeftVentricle_MechanicalTesting/LeftVentricle_ForceDisplacement/LeftVentricle_{sample}/LeftVentricle_{sample}_{mode}.csv"

    if "stretch" in experiment:
        stretch_values, normal_values = load_experimental_data_stretch(fin, width, area)
        shear_values = -1*np.ones_like(normal_values)
    else:
        stretch_values, normal_values, shear_values = load_experimental_data_shear(fin, width, area)

    experimental_data[experiment]["stretch_values"] = stretch_values
    experimental_data[experiment]["normal_values"] = normal_values
    experimental_data[experiment]["shear_values"] = shear_values

a = 1.0
b = 5.0
a_f = 15.0
b_f = 15.0
a_s = 1.0
b_s = 5.0
a_fs = 1.0
b_fs = 5.0

emi_params = {
    "a": df.Constant(a),
    "b": df.Constant(b),
    "a_f": df.Constant(a_f),
    "b_f": df.Constant(b_f),
    "a_s": df.Constant(a_s),
    "b_s": df.Constant(b_s),
    "a_fs": df.Constant(a_fs),
    "b_fs": df.Constant(b_fs),
}

emi_models = {}

for experiment in experiments:
    model = initiate_emi_model(mesh, emi_params, experiment)
    emi_models[experiment] = model

params = [a, b, a_f, b_f, a_s, b_s, a_fs, b_fs]

bounds = [(0.0001, 100), (0.0001, 100), (0.0, 100), (0.0001, 100), (0.0, 100), (0.0001, 100), (0.0, 100), (0.0001, 100)]

opt = minimize(
        cost_function,
        np.array(params),
        (experiments, experimental_data, emi_models, norm),
        bounds=bounds,
        )
params = opt.x

print(opt)
print(params)
