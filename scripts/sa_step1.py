"""

Åshild Telle / Simula Research Laboratory / 2022

This script initiates the first phase of the Sobol analysis.
It defines the parameter space and saves all parameter
combinations to separate output files.

"""

import os
from SALib.sample import saltelli
import numpy as np


def init_SA_problem(N):
    problem = {
        "num_vars": 6,
        "names": ["a_i", "b_i", "a_e", "b_e", "a_if", "b_if"],
        "bounds": [[0.1, 30] for _ in range(6)],
    }

    param_values = saltelli.sample(problem, N, calc_second_order=False)

    return problem, param_values


def initiate_sa(N, output_folder):
    problem, param_values = init_SA_problem(N)

    metrics = [
        "intracellular_stress_fiber_dir",
        "intracellular_stress_sheet_dir",
        "intracellular_stress_normal_dir",
        "extracellular_stress_fiber_dir",
        "extracellular_stress_sheet_dir",
        "extracellular_stress_normal_dir",
    ]

    Ys = {}
    Si = {}
    threads = []

    for metric in metrics:
        Ys[metric] = np.zeros([param_values.shape[0]])

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    for i, X in enumerate(param_values):
        fout = folder + f"/parameter_set_{i}.npy"
        print(X)
        np.save(fout, np.array(X))


parser = argparse.ArgumentParser()

parser.add_argument(
    "-N",
    "--num_variables",
    type=int,
    default=512,
    help="N value for generation of the parameter space",
)

parser.add_argument(
    "-of",
    "--output_folder",
    type=str,
    default="sobol_analysis",
    help="Save all output files, i.e., all parameter combinations here",
)

args = parser.parse_args()

N = args.num_variables
output_folder = args.output_folder
initiate_sa(N, output_folder)
