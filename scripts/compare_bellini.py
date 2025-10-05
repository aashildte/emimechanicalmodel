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

material_params = [
    {
        "c_iso" : 1.64,
        "k1" : 2.08,
        "k2" : 3.67,
        "k3" : 1.13,
        "k4" : 1.25,
    }, 
    {
        "c_iso" : 0.63,
        "k1" : 3.27,
        "k2" : 7.62,
        "k3" : 1.75,
        "k4" : 3.24,
    },
    {
        "c_iso" : 1.52,
        "k1" :2.36,
        "k2" : 12.65,
        "k3" : 1.56,
        "k4" : 7.17,
    },
    {
        "c_iso" : 2.37,
        "k1" : 5.38,
        "k2" : 11.37,
        "k3" : 2.11,
        "k4" : 4.03,
    },
    ]


fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 3))

final_values = []

colors = ["black", "black", "tab:red", "tab:red"]
labels = [r"$LA_{ant}$", r"$LA_{ant}$, VTP", r"$LA_{post}$", r"$LA_{post}$, VTP"]
markers = ["-", "--", "-", "--"]

for stretch_dir, axis in zip(["stretch_ff", "stretch_ss"], axes):
    
    final = []

    for params, label, color, marker in zip(material_params, labels, colors, markers):
        model = TissueModel(
            mesh,
            material_model="Bellini",
            material_parameters=params,
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

        stretch_percentage = [100*s for s in stretch]
        axis.plot(stretch_percentage, load, marker, color=color, label=label)
        axis.plot(stretch_percentage[-1], load[-1], "o", color=color)
        axis.grid('on')
    final_values.append(final)

axis.legend()

axes[0].set_xlabel("Stretch (%)")
axes[1].set_xlabel("Stretch (%)")
axes[0].set_ylabel("Load (kPa)")

plt.tight_layout()
plt.savefig("stretch_curves_bellini.pdf")

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

axes[0].plot(labels[:2], final_values[0][:2], "-", color="black")
axes[1].plot(labels[:2], final_values[1][:2], "-", color="black")
axes[2].plot(labels[:2], [f/s for (f, s) in zip(final_values[0][:2], final_values[1][:2])], "-", color="black")

axes[0].plot(labels[2:], final_values[0][2:], "-", color="tab:red")
axes[1].plot(labels[2:], final_values[1][2:], "-", color="tab:red")
axes[2].plot(labels[2:], [f/s for (f, s) in zip(final_values[0][2:], final_values[1][2:])], "-", color="tab:red")

for i in range(4):
    axes[0].plot(labels[i], final_values[0][i], "o", color=colors[i])
    axes[1].plot(labels[i], final_values[1][i], "o", color=colors[i])
    axes[2].plot(labels[i], final_values[0][i]/final_values[1][i], "o", color=colors[i])

for ax in axes:
    ax.grid('on')

axes[0].set_ylabel("Load (kPa)")
axes[1].set_ylabel("Load (kPa)")
axes[2].set_ylabel("Ratio (-)")

axes[0].set_title(r"Myocyte fiber direction ($L_F$)")
axes[1].set_title(r"Myocyte transverse direction ($L_S$)")
axes[2].set_title(r"Anisotropy ratio($L_F/L_S$)")

plt.tight_layout()
plt.savefig("final_load_values_bellini.pdf")
plt.show()
