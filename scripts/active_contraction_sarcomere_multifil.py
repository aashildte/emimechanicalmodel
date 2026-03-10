"""

Åshild Telle / UW / 2024

Script for simulating active contraction; over (parts of) one cardiac cycle.

"""

from argparse import ArgumentParser
import numpy as np
import dolfin as df
from mpi4py import MPI
from collections import defaultdict
import multifil as mf
import matplotlib.pyplot as plt

from emimechanicalmodel import (
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

    pp = parser.parse_args()

    return (
        pp.mesh_file,
        pp.output_folder,
        pp.time_max,
        pp.num_time_steps,
        pp.plot_at_peak,
        pp.plot_all_steps,
        pp.verbose,
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
) = read_cl_args()

# compute active stress, given from the Rice model

time = np.linspace(0, time_max, num_time_steps)  # ms
active_values = compute_active_component(time)
active_values *= 95
peak_index = np.argmax(active_values)
print(np.max(active_values))
# load mesh, subdomains

"""
import matplotlib.pyplot as plt
plt.plot(active_values)
plt.show()
exit()
"""
mesh, volumes, angles = load_mesh_sarcomere(mesh_file, verbose)
enable_monitor = True  # save output if != None

# initiate model
material_params = {}

model = SarcomereModel(
    mesh,
    volumes,
    angles,
    material_parameters=material_params,
    experiment="contraction",
    active_model="active_stress",
    verbose=verbose,
    fraction_sarcomeres_disabled=0.0,
    robin_bcs_value=3.0,
)

if enable_monitor:
    monitor = setup_monitor(
        "active_contraction",
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


calcium_transient = 4*active_values[:]/max(active_values)
calcium_transient = 8 - calcium_transient

sarcomeres = [mf.hs.hs(lattice_spacing=15, z_line=1000, pCa=calcium_transient[0]) for _ in range(5)]
"""
for t in range(200):          # just stabilize
    for sarc in sarcomeres:
        sarc.timestep()
"""
sarcomeres_org_length = [sarc.z_line for sarc in sarcomeres]
sarcomeres_org_force = [sarc.axial_force() for sarc in sarcomeres]
value_map = {}

axial_force = {}
rel_shortenings = {}
for s in range(5):
    axial_force[s] = []
    rel_shortenings[s] = []

#t = 0
#model.solve(project=True)
#monitor.update_scalar_functions(t)
#monitor.update_xdmf_files(t)


# then run the simulation
for t in range(num_time_steps):
    #time_pt, a_str = time[i], active_values[i]
    if verbose >= 1 and MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Time step {t+1} / {num_time_steps}", flush=True)
    
    rel_shortening_all = model.compute_regional_shortening()
    print("relative shortening: ", rel_shortening_all) 

    for s, sarc in enumerate(sarcomeres):
        key = int(1000+s)
        rel_shortening = rel_shortening_all[key]

        rel_shortenings[s].append(rel_shortening)

        print("rel shortening: ", s, rel_shortening, sarc.z_line)

    for s, sarc in enumerate(sarcomeres):
        #print("sarcomere:", s)
        axial_force[s].append(max(0, 2*1E-3*(sarc.axial_force() - sarcomeres_org_force[s])))      # kPa? pN to ..

        value_map[1000+s] = axial_force[s][-1]
        #print(axial_force, value_map, sarcomeres_org_force[s])
        #value_map[1000+s] = active_values[t]

        sarc.ca = 10**(-calcium_transient[t])
        sarc.pCa = calcium_transient[t]
        
        """ 
        for i in range(5):
            rel_shortening_ramp = i/5*rel_shortenings[s][-1] + (5-i)/5*rel_shortenings[s][-2]
            sarc.z_line = sarcomeres_org_length[s]*(1 + rel_shortening_ramp) # rel_shortenings[s][-1])
            sarc.timestep()
        """
        sarc.z_line = sarcomeres_org_length[s]*(1 + rel_shortenings[s][-1])
        
        sarc.timestep()
        if s==0:
            print("ca: ", sarc.ca)
            print("axial force: ", sarc.axial_force())
            print("applied force: ", value_map[1000])
    #exit()
    #print(value_map)
        
    model.update_active_fn(value_map)


    try:
        model.solve(project=True)
    except RuntimeError:

        fig, axes = plt.subplots(2, sharex=True)

        for s in range(5):
            axes[0].plot(axial_force[s])
            axes[1].plot([100*r for r in rel_shortenings[s]])

        axes[0].grid('on')
        axes[1].grid('on')

        axes[0].set_ylabel("Active tension (kPa (?))")
        axes[1].set_ylabel("Relative_shortening (%)")
        axes[1].set_xlabel("Time (ms)")

        axes[0].legend(["Sarc. 1", "Sarc. 2", "Sarc. 3", "Sarc. 4", "Sarc. 5"])

        plt.show()
        exit()

    monitor.update_scalar_functions(t)
    monitor.update_xdmf_files(t)
    

fig, axes = plt.subplots(2)

for s in range(5):
    axes[0].plot(axial_force[s])
    axes[1].plot([100*r for r in rel_shortenings[s]])

    axes[0].grid('on')
    axes[1].grid('on')

    axes[0].set_ylabel("Active tension (kPa (?))")
    axes[1].set_ylabel("Relative_shortening (%)")
    axes[1].set_xlabel("Time (ms)")

    axes[0].legend(["Sarc. 1", "Sarc. 2", "Sarc. 3", "Sarc. 4", "Sarc. 5"])

plt.show()

if enable_monitor:
    monitor.save_and_close()
