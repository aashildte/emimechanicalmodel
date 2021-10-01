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
        default=0.074,
        type=float,
        help="Stress/strain scaling parameter for the intracellular domain",
    )

    parser.add_argument(
        "-bi",
        "--b_i",
        default=4.878,
        type=float,
        help="Stress/strain scaling parameter for the intracellular domain",
    )

    parser.add_argument(
        "-ae",
        "--a_e",
        default=1,
        type=float,
        help="Stress/strain scaling parameter for the extracellular domain",
    )

    parser.add_argument(
        "-be",
        "--b_e",
        default=10,
        type=float,
        help="Stress/strain scaling parameter for the extracellular domain",
    )

    parser.add_argument(
        "-af",
        "--a_f",
        default=4.071,
        type=float,
        help="Stress/strain scaling parameter for the intracellular domain",
    )

    parser.add_argument(
        "-bf",
        "--b_f",
        default=5.433,
        type=float,
        help="Stress/strain scaling parameter for the intracellular domain",
    )

    parser.add_argument(
        "-as",
        "--a_s",
        default=0.309,
        type=float,
        help="Stress/strain scaling parameter for the intracellular domain",
    )

    parser.add_argument(
        "-bs",
        "--b_s",
        default=2.634,
        type=float,
        help="Stress/strain scaling parameter for the intracellular domain",
    )

    parser.add_argument(
        "-afs",
        "--a_fs",
        default=0.062,
        type=float,
        help="Stress/strain scaling parameter for the intracellular domain",
    )

    parser.add_argument(
        "-bfs",
        "--b_fs",
        default=3.476,
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
        default=os.path.join(os.path.dirname(__file__), "..", "meshes", "tile_1p0.h5"),
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
        help="Do spatial plots at all time steps",
    )

    parser.add_argument(
        "-p",
        "--plot_at_peak",
        default=False,
        action="store_true",
        help="Do spatial plots at peak (max value)",
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
        default="xdir",
        type=str,
        help="Direction for stretch; 'xdir' or 'ydir'",
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
        default=1000,
        type=int,
        help="Length of cardiac cycle to simulate, in ms; integer between 1 and \
                1000 (max contraction = 138).",
    )

    parser.add_argument(
        "-t",
        "--num_time_steps",
        default=1000,
        type=int,
        help="Number of time steps to simulate",
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
