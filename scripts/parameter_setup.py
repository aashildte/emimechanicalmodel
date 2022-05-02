"""

Åshild Telle / Simula Research Labratory / 2021

"""

import os

from emimechanicalmodel import Monitor


def add_emi_holzapfel_arguments(parser):
    """

    Options for changing material parameters from the command line.

    """

    parser.add_argument(
        "-ai",
        "--a_i",
        default=5.70,
        type=float,
        help="Stress/strain scaling parameter for the intracellular domain",
    )

    parser.add_argument(
        "-bi",
        "--b_i",
        default=11.67,
        type=float,
        help="Stress/strain scaling parameter for the intracellular domain",
    )

    parser.add_argument(
        "-ae",
        "--a_e",
        default=1.52,
        type=float,
        help="Stress/strain scaling parameter for the extracellular domain",
    )

    parser.add_argument(
        "-be",
        "--b_e",
        default=16.31,
        type=float,
        help="Stress/strain scaling parameter for the extracellular domain",
    )

    parser.add_argument(
        "-af",
        "--a_f",
        default=19.83,
        type=float,
        help="Stress/strain scaling parameter for the intracellular domain",
    )

    parser.add_argument(
        "-bf",
        "--b_f",
        default=24.72,
        type=float,
        help="Stress/strain scaling parameter for the intracellular domain",
    )



def add_default_arguments(parser):
    """

    Options for changing various options (mesh, output folder, etc.) from the command line.

    """

    parser.add_argument(
        "-m",
        "--mesh_file",
        default="meshes/tile_connected.h5",
        type=str,
        help="Mesh file (h5 format)",
    )

    parser.add_argument(
        "-o",
        "--output_folder",
        default=None,
        type=str,
        help="Specify folder to store results in.",
    )

    parser.add_argument(
        "-a",
        "--plot_all_steps",
        default=False,
        action="store_true",
        help="Save spatial quantities for all time steps",
    )
    
    parser.add_argument(
        "--project_to_subspaces",
        default=False,
        action="store_true",
        help="Save spatial quantities for each subdomain",
    )

    parser.add_argument(
        "-p",
        "--plot_at_peak",
        default=False,
        action="store_true",
        help="Save spatial quantities at peak (max value)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        default=1,
        type=int,
        help="Level of verbose output; 0 gives very little; 1 time steps; 2 fenics output.",
    )


def add_stretching_arguments(parser):
    """

    Options for changing stretch setup from the command line.

    """

    parser.add_argument(
        "-d",
        "--dir_stretch",
        default="stretch_ff",
        type=str,
        help="Direction for stretch; can be 'stretch_ff', 'stretch_ss', 'stretch_nn', " + \
              "'shear_fn', 'shear_nf', 'shear_fs', 'shear_sf', 'shear_ns', or 'shear_sn'.",
    )

    parser.add_argument(
        "-str",
        "--strain",
        default=0.2,
        type=float,
        help="Strain we want to reach; typically in range 0.1-0.3",
    )

    parser.add_argument(
        "-N",
        "--num_steps",
        default=30,
        type=int,
        help="Number of steps for stretching",
    )


def add_active_arguments(parser):
    """

    Options for changing active contraction experiment from the command line.

    """

    parser.add_argument(
        "-tm",
        "--time_max",
        default=500,
        type=int,
        help="Length of cardiac cycle to simulate, in ms; integer between 1 and \
                1000 (max contraction = 138, default 500).",
    )

    parser.add_argument(
        "-t",
        "--num_time_steps",
        default=500,
        type=int,
        help="Number of time steps to simulate (default 500)",
    )


def setup_monitor(
    experiment,
    output_folder,
    model,
    mesh_file,
    material_params,
    num_steps,
    to_value,
):
    """

    Setup function for instance of Monitor class, based on cl options.

    """

    _, meshid = os.path.split(mesh_file)
    meshid = meshid.split(".")[0]

    param_space = {
        "experiment": experiment,
        "mesh": meshid,
        "num_steps": num_steps,
        "to_value": to_value,
    }

    monitor = Monitor(model, output_folder, {**param_space, **material_params})

    return monitor
