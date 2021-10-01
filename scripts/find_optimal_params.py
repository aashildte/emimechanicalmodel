"""

Script for estimating optimal values for a_f / b_f, or a_s, b_s;
using a probabilistic scheme. We compare with the tissue level model,
then change the EMI model parameters to match these.

This is used for Fig. 5 in the paper.

"""


import os

from argparse import ArgumentParser
import numpy as np
import dolfin as df


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


def go_to_stretch(mat_model, stretch, xdir):

    loads = np.zeros_like(stretch)

    for (step, st) in enumerate(stretch):
        print(f"Forward step : {step}/{len(stretch)}", flush=True)
        mat_model.assign_stretch(st)
        mat_model.solve()

        if xdir:
            loads[step] = mat_model.evaluate_load_yz()
        else:
            loads[step] = mat_model.evaluate_load_xz()

    return loads


def get_initial_values(
    tissue_model_x, tissue_model_y, emi_model_x, emi_model_y, target_stretch
):
    num_ev_points = 2  # 10 %, 20 %
    num_steps = 10 * num_ev_points + 1

    total_stretch = np.linspace(0, target_stretch, num_steps)

    # calculate stretch; using a larger number of steps, for convergence

    stretch_values = np.zeros(num_ev_points)
    emi_x_loads = np.zeros(num_ev_points)
    emi_y_loads = np.zeros(num_ev_points)
    tissue_x_loads = np.zeros(num_ev_points)
    tissue_y_loads = np.zeros(num_ev_points)

    states_emi_x = []
    states_emi_y = []

    for num in range(num_ev_points):
        stretch = total_stretch[num * 10 : (1 + num) * 10 + 1]
        stretch_values[num] = stretch[-1]
        tissue_x_loads[num] = go_to_stretch(tissue_model_x, stretch, True)[-1]
        tissue_y_loads[num] = go_to_stretch(tissue_model_y, stretch, False)[-1]
        emi_x_loads[num] = go_to_stretch(emi_model_x, stretch, True)[-1]
        emi_y_loads[num] = go_to_stretch(emi_model_y, stretch, False)[-1]

        st_values = emi_model_x.state.vector()[:]
        states_emi_x.append(st_values)

        st_values = emi_model_y.state.vector()[:]
        states_emi_y.append(st_values)

    states_emi_x, states_emi_y = np.array(states_emi_x), np.array(states_emi_y)

    return (
        stretch_values,
        tissue_x_loads,
        tissue_y_loads,
        emi_x_loads,
        emi_y_loads,
        states_emi_x,
        states_emi_y,
    )


def get_adjustment_factors(n, N, baseline_emi_values, tissue_loads):
    # 4 possible cases; adjust up or down according to comparison
    # at 2 selected points

    A = baseline_emi_values[0]
    B = baseline_emi_values[1]
    C = tissue_loads[0]
    D = tissue_loads[1]

    if A < C and B < D:
        print("case 1")
        factor_a = np.random.random()
        factor_b = 0.1 * np.random.random()
    elif A > C and B > D:
        print("case 2")
        factor_a = np.random.random() - 1
        factor_b = 0.1 * (np.random.random() - 1)
    elif A > C and B < D:
        print("case 3")
        factor_a = np.random.random() - 1
        factor_b = 0.1 * np.random.random()
    elif A < C and B > D:
        print("case 4")
        factor_a = np.random.random()
        factor_b = 0.1 * (np.random.random() - 1)

    factor_a *= reg_factor * (1 - n / N) ** 2
    factor_b *= reg_factor * (1 - n / N) ** 2

    print("multiplicative factors: ", (1 + factor_a), (1 + factor_b), flush=True)
    return factor_a, factor_b


def get_new_loads(emi_model, stretch_values, states, xdir):

    emi_loads = np.zeros_like(stretch_values)

    i = 0
    for (st_val, state) in zip(stretch_values, states):
        emi_model.assign_stretch(st_val)
        emi_model.state.vector()[:] = state
        emi_model.solve()

        if xdir:
            emi_loads[i] = emi_model.evaluate_load_yz()
        else:
            emi_loads[i] = emi_model.evaluate_load_xz()
        i += 1

    return emi_loads


def initiate_models(tissue_params, emi_params):
    tissue_model_x = TissueModel(
        mesh,
        experiment="xstretch",
        material_parameters=tissue_params,
    )

    tissue_model_y = TissueModel(
        mesh,
        experiment="ystretch",
        material_parameters=tissue_params,
    )

    emi_model_x = EMIModel(
        mesh,
        volumes,
        experiment="xstretch",
        material_parameters=emi_params,
    )

    emi_model_y = EMIModel(
        mesh,
        volumes,
        experiment="ystretch",
        material_parameters=emi_params,
    )

    return tissue_model_x, tissue_model_y, emi_model_x, emi_model_y


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

emi_params = {
    "a_i": 0.074,
    "b_i": 4.878,
    "a_e": 1,
    "b_e": 10,
    "a_if": 2.628,
    "b_if": 5.214,
    "a_is": 0.438,
    "b_is": 3.002,
    "a_ifs": 0.062,
    "b_ifs": 3.476,
}


tissue_model_x, tissue_model_y, emi_model_x, emi_model_y = initiate_models(
    tissue_params, emi_params
)

target_stretch = 0.2

(
    stretch_values,
    tissue_x_loads,
    tissue_y_loads,
    emi_x_loads,
    emi_y_loads,
    states_x,
    states_y,
) = get_initial_values(
    tissue_model_x, tissue_model_y, emi_model_x, emi_model_y, target_stretch
)

print("Init values EMI:", emi_x_loads, emi_y_loads, flush=True)
print("Init values Tissue:", tissue_x_loads, tissue_y_loads, flush=True)

n, N = 0, 1000
tol = 1e-4
reg_factor = 0.01

diff = np.linalg.norm(emi_x_loads - tissue_x_loads) + np.linalg.norm(
    emi_y_loads - tissue_y_loads
)

baseline_emi_x_values = emi_x_loads.copy()
baseline_emi_y_values = emi_y_loads.copy()

while n < N and diff > tol:

    n += 1

    print("new iteration ..\n", flush=True)
    print("baselines: ", baseline_emi_x_values, baseline_emi_y_values)

    if n % 2:
        a_old = emi_model_x.mat_model.a_f.vector()[:]
        b_old = emi_model_x.mat_model.b_f.vector()[:]

        factor_a, factor_b = get_adjustment_factors(
            n, N, baseline_emi_x_values, tissue_x_loads
        )

        emi_model_x.mat_model.a_f.vector()[:] *= 1 + factor_a
        emi_model_x.mat_model.b_f.vector()[:] *= 1 + factor_b
        emi_model_y.mat_model.a_f.vector()[:] *= 1 + factor_a
        emi_model_y.mat_model.b_f.vector()[:] *= 1 + factor_b

    else:
        a_old = emi_model_y.mat_model.a_s.vector()[:]
        b_old = emi_model_y.mat_model.b_s.vector()[:]

        factor_a, factor_b = get_adjustment_factors(
            n, N, baseline_emi_y_values, tissue_y_loads
        )

        emi_model_x.mat_model.a_s.vector()[:] *= 1 + factor_a
        emi_model_x.mat_model.b_s.vector()[:] *= 1 + factor_b
        emi_model_y.mat_model.a_s.vector()[:] *= 1 + factor_a
        emi_model_y.mat_model.b_s.vector()[:] *= 1 + factor_b

    emi_loads_x = get_new_loads(emi_model_x, stretch_values, states_x, True)
    emi_loads_y = get_new_loads(emi_model_y, stretch_values, states_y, False)

    new_diff = np.linalg.norm(tissue_x_loads - emi_loads_x) + np.linalg.norm(
        tissue_y_loads - emi_loads_y
    )

    print("new diff: ", new_diff, flush=True)
    print("new loads: ", emi_loads_x, emi_loads_y)

    if new_diff > diff:
        if n % 2:
            emi_model_x.mat_model.a_f.vector()[:] = a_old
            emi_model_x.mat_model.b_f.vector()[:] = b_old
            emi_model_y.mat_model.a_f.vector()[:] = a_old
            emi_model_y.mat_model.b_f.vector()[:] = b_old
            print(
                f"Atempt {n}/{N}; a_f : {max(a_old)}; b_f : {max(b_old)}; diff : {diff}",
                flush=True,
            )
        else:
            emi_model_x.mat_model.a_s.vector()[:] = a_old
            emi_model_x.mat_model.b_s.vector()[:] = b_old
            emi_model_y.mat_model.a_s.vector()[:] = a_old
            emi_model_y.mat_model.b_s.vector()[:] = b_old

            print(
                f"Atempt {n}/{N}; a_s : {max(a_old)}; b_s : {max(b_old)}; diff : {diff}",
                flush=True,
            )
    else:
        diff = new_diff
        baseline_emi_x_values[:] = emi_loads_x[:]
        baseline_emi_y_values[:] = emi_loads_y[:]
        print("\nfound new values!\n", flush=True)

        a_f = np.max(emi_model_x.mat_model.a_f.vector()[:])
        b_f = np.max(emi_model_x.mat_model.b_f.vector()[:])
        a_s = np.max(emi_model_y.mat_model.a_s.vector()[:])
        b_s = np.max(emi_model_y.mat_model.b_s.vector()[:])

        print(f"new a_f, b_f, a_s, b_s: {a_f}, {b_f}, {a_s}, {b_s}")

        print(f"Calculated load values: {emi_loads_x}, {emi_loads_y}", flush=True)
        print(f"Tissue load values: {tissue_x_loads}, {tissue_y_loads}", flush=True)
        print(f"Atempt {n}/{N}; diff : {diff}", flush=True)

emi_params["a_f"] = a_f
emi_params["b_f"] = b_f
emi_params["a_s"] = a_s
emi_params["b_s"] = b_s

# save values in output folder, if indicated

if output_folder is not None:

    tissue_model_x, tissue_model_y, emi_model_x, emi_model_y = initiate_models(
        tissue_params, emi_params
    )

    target_stretch = 0.2
    N = 30

    stretch_values = np.linspace(0, target_stretch, N)

    tissue_loads_x = go_to_stretch(tissue_model_x, stretch_values, True)
    tissue_loads_y = go_to_stretch(tissue_model_y, stretch_values, False)
    emi_loads_x = go_to_stretch(emi_model_x, stretch_values, True)
    emi_loads_y = go_to_stretch(emi_model_y, stretch_values, False)

    output_values = {
        "stretch": stretch_values,
        "tissue_loads_x": tissue_loads_x,
        "tissue_loads_y": tissue_loads_y,
        "emi_loads_x": emi_loads_x,
        "emi_loads_y": emi_loads_y,
    }

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    np.save(f"{output_folder}/emi_tissue_comp.npy", output_values)
