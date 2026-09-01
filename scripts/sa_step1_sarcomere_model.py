"""

Åshild Telle / Simula Research Laboratory / 2022

This script initiates the first phase of the Sobol analysis.
It defines the parameter space and saves all parameter
combinations to separate output files.

"""

import os
from SALib.sample import saltelli
import numpy as np
from argparse import ArgumentParser

def init_SA_problem(N):
    problem = {
        "num_vars": 5,
        "names": ["a_i_sarcomeres", "a_i_zlines", "a_i_cytoskeleton", "a_i_connections", "a_i_nucleus"],
        "bounds": [[0.03, 30] for _ in range(5)],
    }
    
    param_values = saltelli.sample(problem, N, calc_second_order=False)

    return problem, param_values


def initiate_sa(N, output_folder):
    problem, param_values = init_SA_problem(N)

    metrics = ["relative_shortening", "normal_stress"]

    descriptions = [
                "sarcomeres",
                "zlines",
                "cytoskeleton",
                "connections",
                "nucleus",
                ]

    for desc in descriptions:
        metrics += [
            f"lambda_F_{desc}"
            f"lambda_T_{desc}"
            f"lambda_FT_{desc}"
            f"stress_xdir_{desc}"
            f"stress_ydir_{desc}"
            f"stress_xydir_{desc}"
        ]

    Ys = {}
    Si = {}
    threads = []

    for metric in metrics:
        Ys[metric] = np.zeros([param_values.shape[0]])

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    for i, X in enumerate(param_values):
        fout = output_folder + f"/parameter_set_{i}.npy"
        #print(X)
        np.save(fout, np.array(X))

if __name__ == "__main__":

    parser = ArgumentParser()

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
