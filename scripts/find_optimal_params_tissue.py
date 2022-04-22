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

from load_data import *

# Optimization options for the form compiler
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["representation"] = "uflacs"
df.parameters["form_compiler"]["quadrature_degree"] = 4

df.set_log_level(60)


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
            print("crashed; adding + 500", cf)
            cf += 500      # just a high value
            continue

        target_normal_load = experimental_data[experiment]["normal_values"]
        cf += np.linalg.norm(normal_load - target_normal_load)
        
        target_shear_load = experimental_data[experiment]["shear_values"]        
        cf += np.linalg.norm(shear_load - target_shear_load)

    print(f"Current cost fun value: {cf}", flush=True)
    return cf


fname_dims = "Data/LeftVentricle_MechanicalTesting/LeftVentricle_Dimensions_mm.csv"
dimensions, areas = get_dimensions(fname_dims)

sample = "Sample" + sys.argv[1]
print(f"Running experiments for sample {sample}", flush=True)

norm = sys.argv[2]

mesh = df.UnitCubeMesh(1, 1, 1)

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
    model = initiate_tissue_model(mesh, emi_params, experiment)
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
