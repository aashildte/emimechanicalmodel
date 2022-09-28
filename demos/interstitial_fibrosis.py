"""

Åshild Telle / Simula Research Laboratory / 2022

Script for simulating active contraction; over (parts of) one cardiac cycle,
with increasing extracellular stiffness, meant to resemble interstinial fibrosis.

"""

import numpy as np
import dolfin as df
from mpi4py import MPI
import matplotlib.pyplot as plt

from emimechanicalmodel import (
    load_mesh,
    compute_active_component,
    EMIModel,
)

df.parameters["form_compiler"]["quadrature_degree"] = 4


# simulate 500 ms with active contraction from the Rice model

num_time_steps = 125

time = np.linspace(0, 500, num_time_steps)  # ms
active_values, scaling_value = compute_active_component(time)
print(scaling_value)
active_values *= 0.28 / scaling_value

# load mesh, subdomains

mesh_file = "meshes/tile_connected_10p0.h5"
mesh, volumes = load_mesh(mesh_file)
cell_id = 1        # while matrix_id = 0

# specifiy material parameters

material_parameters_org = {
    "a_i": 5.7,
    "b_i": 11.67,
    "a_e": 1.52,
    "b_e": 16.31,
    "a_if": 19.83,
    "b_if": 24.72,
}


# explore a_e parameters scaled by the following factors:

stiffness_scaling = [0.5, 1, 2, 4]

for i, scale in enumerate(stiffness_scaling):

    material_parameters = material_parameters_org.copy()
    material_parameters["a_e"]*= scale

    # initiate EMI model

    model = EMIModel(
        mesh,
        volumes,
        experiment="contr",
        material_parameters=material_parameters,
    )

    # track stresses in the fiber direction

    stress_fiber_dir = np.zeros_like(time)

    # then run the simulation
    for i in range(num_time_steps):
        print(f"Time step {i+1} / {num_time_steps}", flush=True)

        time_pt, a_str = time[i], active_values[i]

        model.update_active_fn(a_str)
        model.solve()

        stress_fiber_dir[i] = model.evaluate_subdomain_stress_fibre_dir(cell_id)

    plt.plot(time, stress_fiber_dir)

plt.legend(stiffness_scaling)
plt.show()
