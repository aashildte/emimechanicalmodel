"""

Åshild Telle / UW / 2024

Script for simulating active contraction; over (parts of) one cardiac cycle.

"""

from argparse import ArgumentParser
import numpy as np
import dolfin as df
from mpi4py import MPI


from emimechanicalmodel import (
    load_mesh,
    load_mesh_sarcomere,
    compute_active_component,
    SarcomereModel,
)

from parameter_setup import (
    add_emi_holzapfel_arguments,
    add_default_arguments,
    add_active_arguments,
    setup_monitor,
)


def read_cl_args():

    parser = ArgumentParser()

    add_default_arguments(parser)
    add_active_arguments(parser)
    
    parser.add_argument("--remap-connections",
            action="store_true",
            default=False)

    parser.add_argument("--fraction-sarcomeres-disabled",
            type=float,
            default=0.0)

    parser.add_argument("--increase-titin-stiffness",
            action="store_true",
            default=False)
    
    parser.add_argument("--robin",
            type=float,
            default=0.0)

    parser.add_argument("--z-line-scale-factor",
            type=float,
            default=1.0)
    
    parser.add_argument("--connections-scale-factor",
            type=float,
            default=1.0)
    
    parser.add_argument("--sarcomere-scale-factor",
            type=float,
            default=1.0)
    
    parser.add_argument("--sarcomere-scale-factor-af",
            type=float,
            default=1.0)
    
    parser.add_argument("--ECM-scale-factor",
            type=float,
            default=1.0)
    
    parser.add_argument("--cytoskeleton-scale-factor",
            type=float,
            default=1.0)
    
    parser.add_argument("--nucleus-scale-factor",
            type=float,
            default=1.0)
    
    parser.add_argument("--tension-scale-factor",
            type=float,
            default=1.0)
    
    parser.add_argument("--kappa-nucleus",
            type=float,
            default=1000.0)
    
    parser.add_argument("--isometric",
            action="store_true",
            default=False)

    pp = parser.parse_args()

    return (
        pp.mesh_file,
        pp.output_folder,
        pp.time_max,
        pp.num_time_steps,
        pp.plot_at_peak,
        pp.plot_all_steps,
        pp.verbose,
        pp.remap_connections,
        pp.fraction_sarcomeres_disabled,
        pp.increase_titin_stiffness,
        pp.robin,
        pp.z_line_scale_factor,
        pp.connections_scale_factor,
        pp.sarcomere_scale_factor,
        pp.sarcomere_scale_factor_af,
        pp.ECM_scale_factor,
        pp.cytoskeleton_scale_factor,
        pp.nucleus_scale_factor,
        pp.tension_scale_factor,
        pp.isometric,
        pp.kappa_nucleus,
    )


# read in (relevant) parameters from the command line

(
    mesh_file,
    output_folder,
    time_max,
    num_time_steps,
    plot_at_peak,
    plot_all_steps,
    verbose,
    remap_connections,
    fraction_sarcomeres_disabled,
    increase_titin_stiffness,
    rb,
    zline_scale,
    connections_scale,
    sarcomere_scale,
    sarcomere_scale_af,
    ECM_scale,
    cytoskeleton_scale,
    nucleus_scale,
    tension_scale,
    isometric,
    kappa_nucleus,
) = read_cl_args()

# compute active stress, given from the Rice model

time = np.linspace(0, time_max, num_time_steps)  # ms
active_values = compute_active_component(time)
active_values *= 750*tension_scale

peak_index = np.argmax(active_values)
# load mesh, subdomains

mesh, volumes, angles = load_mesh_sarcomere(mesh_file, verbose)
nucleus_angles = np.load(mesh_file.split(".")[0] + "_nuclei_angles.npy", allow_pickle=True).item()["angles"]

enable_monitor = True  # save output if != None

material_params = {
        "a_i_sarcomeres" : 1.0*sarcomere_scale,
        "a_if_sarcomeres" : 5.0*sarcomere_scale_af,
        "a_i_zlines": 29.94146484, #8.0*zline_scale,
        "a_i_connections" : 1.0*connections_scale,
        "a_i_cytoskeleton" : 0.25*cytoskeleton_scale,
        "a_if_cytoskeleton" : 5.0,
        "a_i_nucleus" : 1.0*nucleus_scale,
        }

model = SarcomereModel(
    mesh,
    volumes,
    sarcomere_angles=angles,
    nucleus_angles=nucleus_angles,
    material_parameters=material_params,
    experiment="contraction",
    active_model="active_stress",
    compressibility_model="nearly_incompressible",
    verbose=verbose,
    isometric=isometric,
)

material_params["Ta"] = tension_scale

if enable_monitor:
    monitor = setup_monitor(
        f"active_contraction_exp",
        output_folder,
        model,
        mesh_file,
        material_params,
        num_time_steps,
        time_max,
    )
else:
    monitor = None

if verbose < 2:
    df.set_log_level(60)  # remove information about convergence

# then run the simulation
for i in range(num_time_steps):
    time_pt, a_str = time[i], active_values[i]

    if verbose >= 1 and MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Time step {i+1} / {num_time_steps}", flush=True)
   
    if i==136:
        project = True
    else:
        project = False
        #project = (i==(num_time_steps - 1))
    
    #project=True
    model.update_active_fn(a_str)
    model.solve(project=project)
    

    if enable_monitor:
        monitor.update_scalar_functions(time_pt)

        if project:
            monitor.update_xdmf_files(i)
    
if enable_monitor:
    monitor.save_and_close()
