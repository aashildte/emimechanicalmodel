"""

Åshild Telle / Simula Research Laboratory / 2022

Script for stretching in the fiber, cross-fiber and normal directions,
reproducing stress/strain curves along these directions.

"""

import numpy as np
import matplotlib.pyplot as plt
import dolfin as df
from mpi4py import MPI


from emimechanicalmodel import (
    load_mesh,
    EMIModel,
)


# load mesh, subdomains

mesh_file = "meshes/tile_connected_10p0.h5"
mesh, volumes = load_mesh(mesh_file)


# specifiy material parameters

material_parameters = {
    "a_i": 5.7,
    "b_i": 11.67,
    "a_e": 1.52,
    "b_e": 16.31,
    "a_if": 19.83,
    "b_if": 24.72,
}

# define stretching modes

def_modes = ["stretch_ff", "stretch_ss", "stretch_nn"]

# and stretching range (0.1 = 10 %)

stretch_values = np.linspace(0, 0.1, 10)

for mode in def_modes:
    # initiate EMI model

    model = EMIModel(
        mesh,
        volumes,
        experiment=mode,
        material_parameters=material_parameters,
    )

    load_values = np.zeros_like(stretch_values)

    for i, st_val in enumerate(stretch_values):
        model.assign_stretch(st_val)
        model.solve()

        load_values[i] = model.evaluate_normal_load()

    plt.plot(100*stretch_values, load_values)

plt.legend(["Fiber dir. stretch", "Sheet dir. stretch", "Normal dir. stretch"])
plt.xlabel("Stretch (%)")
plt.ylabel("Load (kPa")
plt.show()
