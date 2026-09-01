"""
Åshild Telle / Simula Research Laboratory / 2026

Generic Sobol analysis for the new sarcomere model output format.
Each result file contains a 30-element vector. We treat each index
as an independent metric and compute Sobol indices for all of them.

Input:
    /data1/aashild/sobol_analysis/step1_params
    /data1/aashild/sobol_analysis/step2_results

Output:
    sensitivity_analysis.npy
"""

import numpy as np
from SALib.analyze import sobol
from sa_step1_sarcomere_model import init_SA_problem
import argparse
import os


def sobol_analysis_generic(N_base, input_folder, iso):
    problem, _ = init_SA_problem(N_base)

    k = 5
    #N = (2 * k + 2) * N_base
    N = (k + 2) * N_base
    M = 32
    Y = np.zeros((N, M))

    for i in range(N):
        fname = os.path.join(input_folder, f"results_iso_{iso}_{i}.npy")
        try:
            arr = np.load(fname)
            print(arr)
        except Exception:
            print(f"Missing or unreadable file: {fname}")
            arr = np.zeros(30) 
            exit(1)
        if arr.shape[0] != M:
            print(f"Unexpected shape in {fname}: {arr.shape}")
            exit(1)
        
        globals_ = arr[:2]
        domains  = arr[2:].reshape(5, 6)
        domains  = domains[:, [3, 4, 5, 0, 1, 2]]
        arr      = np.concatenate([globals_, domains.reshape(-1)])

        Y[i, :] = arr
    
    S = {}
    for m in range(M):
        #print(Y[:,m])
        S[m] = sobol.analyze(problem, Y[:, m], calc_second_order=False)
    
    #print(S)
    #exit()

    return S

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-N",
        "--total_variable_count",
        type=int,
        required=True,
        help="Number of parameter sets."
    )

    parser.add_argument(
        "-if",
        "--input_folder",
        type=str,
        required=True,
        help="Folder containing results_iso_False_*.npy"
    )

    parser.add_argument(
        "-of",
        "--output_file",
        type=str,
        default="sensitivity_analysis.npy",
        help="Output file for Sobol indices"
    )

    parser.add_argument(
            "--isometric",
            default=False,
            action="store_true")

    args = parser.parse_args()

    S = sobol_analysis_generic(args.total_variable_count, args.input_folder, args.isometric)
    np.save(args.output_file, S)
    print(f"Sobol analysis saved to {args.output_file}")


if __name__ == "__main__":
    main()

