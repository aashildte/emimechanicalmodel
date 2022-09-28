"""

Åshild Telle / Simula Research Laboratory / 2022

This script initiates the third and final phase of the Sobol analysis.
For each mode, it reads in results from previously performed stretch,
shear or contraction experiments, the performs a Sobol analysis.
The results are saved as npy files.

"""


from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np
import dolfin as df
import emimechanicalmodel as emi
import argparse

from sa_step1 import init_SA_problem


def sobol_analysis_stresses_single_mode(mode, N, input_folder):
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

    for i, X in enumerate(param_values):
        fin = f"{input_folder}/results_{mode}_{i}.npy"

        try:
            outputs = np.load(fin)
        except:
            print(f"Unable to load data for {mode}_{i}", flush=True)
    
        for j, metric in enumerate(metrics):
            Ys[metric][i] = outputs[j+2]

        fin.close()

    for metric in metrics:
        Si[metric] = sobol.analyze(problem, Ys[metric], calc_second_order=False)
        print("metric: ", metric, Si[metric])

    return Si


def sobol_analysis_loads_single_mode(mode, N, input_folder):
    problem, param_values = init_SA_problem(N)

    Y_normal = np.zeros([param_values.shape[0]])
    Y_shear = np.zeros([param_values.shape[0]])

    threads = []

    for i, X in enumerate(param_values):
        fin = f"{input_folder}/results_{mode}_{i}.npy"

        try:
            outputs = np.load(fin)
        except:
            print(f"Unable to load data for {mode}_{i}", flush=True)
        
        fin.close()
        
        Y_normal[i] = outputs[0]
        Y_shear[i] = outputs[1]

    Si_normal = sobol.analyze(problem, Y_normal, calc_second_order=False)
    Si_shear = sobol.analyze(problem, Y_shear, calc_second_order=False)
    
    return {"normal" : Si_normal, "shear" : Si_shear}


def perform_sobol_analysis(N, input_folder, output_folder):
    sa = {}
    modes = ["stretch_ff", "shear_fs", "shear_fn", "shear_sf", "stretch_ss", "shear_sn", "shear_nf", "shear_ns", "stretch_nn"]
    
    for mode in modes:
        sa_sa[mode] = sobol_analysis_loads_single_mode(mode, N, input_folder)

    np.save(f"{output_folder}/sensitivity_analysis_load.npy", sa)

    sa = {}
    modes = ["stretch_ff", "contr"]
        
    for mode in modes:
        sa[mode] = sobol_analysis_stresses_single_mode(mode, N, input_folder)

    np.save(f"{output_folder}/sensitivity_analysis_stress.npy", sa)


parser = argparse.ArgumentParser()

parser.add_argument("-M", '--mode', type=str, default="contr",
                    help='Deformation mode (valid options; "contr" \
                            "stretch_ff" "shear_fs" "shear_fn" "shear_sf" "stretch_ss" \
                            "shear_sn" "shear_nf" "shear_ns" "stretch_nn")')

parser.add_argument("-N", '--total_variable_count', type=int, default=0,
                    help='Number of parameter set # to use for this simulation.')

parser.add_argument("-if", "--input_folder", type=str, default="sobol_analysis",
                    help='Get all input files, i.e., all parameter combinations here')

parser.add_argument("-of", "--output_folder", type=str, default="sobol_analysis",
                    help='Save all output files, i.e., all resulting metrics here')

args = parser.parse_args()

mode = args.mode
N = args.total_variable_count
input_folder = parser.input_folder
output_folder = parser.output_folder

perform_sobol_analysis(N, input_folder, output_folder)
