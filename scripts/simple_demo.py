#!/usr/bin/env python3

"""

Åshild Telle / Simula Research Labratory / 2021

Simulation with set of standard parameters, for a simple test-run of the code.

This script isn't used anywhere, but provdies a good starting point if you
want to experiment with the code.

"""

import os
import numpy as np

from emimechanicalmodel import (
    load_mesh,
    compute_active_component,
    EMIModel,
)

# compute active stress, given from the Rice model

num_time_steps = 400

time = np.linspace(0, 1000, num_time_steps)  # ms
active_component, scaling_value = compute_active_component(time)
active_component *= 0.28 / scaling_value  # scale to 1 kPa at peak

# load mesh, subdomains
mesh_file = os.path.join(os.path.dirname(__file__), "..", "meshes", "tile_1p0.h5")
mesh, volumes = load_mesh(mesh_file)

# and finally

model = EMIModel(mesh, volumes, experiment="contr")

for a_str in active_component:
    model.update_active_fn(a_str)
    model.solve()
