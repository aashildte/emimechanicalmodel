"""

Åshild Telle / Simula Research Labratory / 2022

Script for simulating passive deformation, i.e. stretch and shear experiments.

"""

import os
from argparse import ArgumentParser
import numpy as np
import dolfin as df
import matplotlib.pyplot as plt

from mpi4py import MPI


from emimechanicalmodel import (
    TissueModel,
)

from parameter_setup import (
    add_emi_holzapfel_arguments,
    add_default_arguments,
    add_stretching_arguments,
    setup_monitor,
)


def read_cl_args():

    parser = ArgumentParser()

    add_default_arguments(parser)
    add_stretching_arguments(parser)
    add_emi_holzapfel_arguments(parser)

    pp = parser.parse_args()

    return (
        pp.a_i,
        pp.b_i,
        pp.a_e,
        pp.b_e,
        pp.a_f,
        pp.b_f,
        pp.mesh_file,
        pp.output_folder,
        pp.dir_stretch,
        pp.strain,
        pp.num_steps,
        pp.plot_at_peak,
        pp.plot_all_steps,
        pp.verbose,
    )


# read in (relevant) parameters from the command line

(
    a_i,
    b_i,
    a_e,
    b_e,
    a_f,
    b_f,
    mesh_file,
    output_folder,
    experiment,
    strain,
    num_steps,
    plot_at_peak,
    plot_all_steps,
    verbose,
) = read_cl_args()


# stretch array; increase from 0 to max

stretch = np.linspace(0, strain, num_steps)


peak_index = num_steps - 1

# load mesh, subdomains

#mesh = df.UnitSquareMesh(3, 3)
mesh = df.UnitCubeMesh(3, 3, 3)

# initiate model
params_all = [
            [0.58, 43.4, 30.7, 29.8, 60.3],
            [1.03, 58.0, 24.6, 30.0, 113.8],
            [0.54, 45.0, 34.0, 19.0, 52.6],
            [1.63, 55.5, 31.5, 33.0, 195.0],
            [2.58, 80.0, 23.0, 17.0, 310.0],
        ]

labels = ["Control", "1 wk", "2 wk", "3 wk", "4 wk"]

colors = ["black", "darkred", "firebrick", "indianred", "lightcoral"]

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 3))

final_values = []

for stretch_dir, axis in zip(["stretch_ff", "stretch_ss"], axes):
    
    final = []

    for params, label, color in zip(params_all, labels, colors):
        
        c, B1, B2, B3, kappa = params

        K = c*(B1 + B2 + B3) 
        print(K, kappa)

        material_params = {
            "C" : c,
            "b_f" : B1,
            "b_t" : B2,
            "b_ft" : B3,
        }

        comp_params = { "kappa" : kappa }

        model = TissueModel(
            mesh,
            material_model="guccione",
            compressibility_model = "nearly_incompressible",
            compressibility_parameters = comp_params,
            material_parameters=material_params,
            experiment=stretch_dir,
            verbose=verbose,
        )

        # setup parameters - define the parameter space to explore

        # then run the simulation

        load = np.zeros_like(stretch)

        for (i, st_val) in enumerate(stretch):

            print(f"Step {i+1} / {num_steps}", flush=True)

            model.assign_stretch(st_val)

            project = (plot_all_steps) or (plot_at_peak and i == peak_index)
            model.solve(project=project)
            
            load[i] = model.evaluate_normal_load()
            print(load[i])
        final.append(load[-1])
        
        stretch_percent = [100*s for s in stretch]
        axis.plot(stretch_percent, load, label=label, color=color)
        axis.plot(stretch_percent[-1], load[-1], "o", color=color)
        axis.grid('on')

    final_values.append(final)

axis.legend()
axes[0].set_xlabel("Stretch (%)")
axes[1].set_xlabel("Stretch (%)")
axes[0].set_ylabel("Load (kPa)")

plt.tight_layout()
plt.savefig("stretch_curves_mendiola.pdf")

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
axes[0].plot(labels, final_values[0], "-", color="tab:red")
axes[1].plot(labels, final_values[1], "-", color="tab:red")
axes[2].plot(labels, [f/s for (f, s) in zip(final_values[0], final_values[1])], "-", color="tab:red")

for i in range(5):
    axes[0].plot(labels[i], final_values[0][i], "-o", color=colors[i])
    axes[1].plot(labels[i], final_values[1][i], "-o", color=colors[i])
    axes[2].plot(labels[i], final_values[0][i]/final_values[1][i], "-o", color=colors[i])

for ax in axes:
    ax.grid('on')

axes[0].set_ylabel("Load (kPa)")
axes[1].set_ylabel("Load (kPa)")
axes[2].set_ylabel("Ratio (-)")

axes[0].set_title(r"Myocyte fiber direction ($L_F$)")
axes[1].set_title(r"Myocyte fiber direction ($L_S$)")
axes[2].set_title(r"Anisotropy ratio ($L_F/L_S$)")

plt.tight_layout()
plt.savefig("final_load_values_mendiola.pdf")
plt.show()
