"""

Åshild Telle / Simula Research Laboratory / 2020

"""


import os
from functools import partial
import numpy as np
from mpi4py import MPI
import dolfin as df

from .emi_model import EMIModel


class Monitor:
    """

    Class specifically developed for tracking the different values
    for the output required for the paper plots. This might not be
    the most relevant nor interesting code to look at.

    Args:
        cardiac model: model to consider; a list called 'tracked_variables'
            must be present, which will be the functions we save here
        output_folder (str): save results here
        param_space (dict): options given at command line; a subfolder
            based on these will be created

    """

    def __init__(self, cardiac_model, output_folder, param_space={}):

        self.cardiac_model = cardiac_model
        self._define_full_path(output_folder, param_space)
        self._init_xdmf_files()
        self._init_tracked_scalar_functions()

    def _define_full_path(self, output_folder, param_space):
        info_str = "output"

        for key in param_space.keys():
            info_str += f"_{str(param_space[key])}"

        info_str = info_str.replace(" ", "_")
        info_str = info_str.replace("/", "_")
        info_str = info_str.replace(".", "p")

        self.full_path = os.path.join(output_folder, f"{info_str}")

    def _init_xdmf_files(self):
        """

        Defines xdmf files for the tracked quantities + initiates
        the dictionary in which these will be saved, at given
        tracking points.

        """
        states = self.cardiac_model.tracked_variables

        path_xdmf_files = os.path.join(self.full_path, "xdmf_files")

        os.makedirs(path_xdmf_files, exist_ok=True)
        self.path_xdmf_files = path_xdmf_files

        functions = {}
        xdmf_files = {}

        for state in states:

            # get some filename-friendly labels
            label = state.name()
            name = label.split(" (")[0].replace(" ", "_")
            name = name.replace(".", "p")

            filename_xdmf = os.path.join(path_xdmf_files, f"{name}.xdmf")

            # save information in a dictionary

            functions[name] = state
            xdmf_files[name] = df.XDMFFile(MPI.COMM_WORLD, filename_xdmf)

        (self.functions, self.xdmf_files) = (functions, xdmf_files)
        # also write subdomains to file

        f_subdomains = df.XDMFFile(os.path.join(path_xdmf_files, "subdomains.xdmf"))
        f_subdomains.write(self.cardiac_model.volumes)
        f_subdomains.close()

    def update_xdmf_files(self, it_number):
        """

        Write checkpoint for all xdmf files.

        """
        functions, xdmf_files = (self.functions, self.xdmf_files)

        # save to paraview files
        for name in xdmf_files.keys():
            # print(name, it_number, flush=True)
            xdmf_files[name].write_checkpoint(
                functions[name], name, it_number, append=True
            )

    def _init_tracked_scalar_functions(self):
        model = self.cardiac_model

        scalar_functions = {
            "normal_load": model.evaluate_normal_load,
            "shear_load": model.evaluate_shear_load,
        }

        # per subdomain; not actively used but let's keep it here in case we
        # need it again
        """        
        for subdomain_id in model.subdomains:
            scalar_functions[f"stress_xdir_subdomain_{subdomain_id}"] = partial(
                model.evaluate_subdomain_stress_fibre_dir, subdomain_ids=subdomain_id
            )
            scalar_functions[f"stress_ydir_subdomain_{subdomain_id}"] = partial(
                model.evaluate_subdomain_stress_sheet_dir,
                subdomain_ids=subdomain_id,
            )
            scalar_functions[f"stress_zdir_subdomain_{subdomain_id}"] = partial(
                model.evaluate_subdomain_stress_normal_dir, subdomain_ids=subdomain_id
            )

            scalar_functions[f"strain_xdir_subdomain_{subdomain_id}"] = partial(
                model.evaluate_subdomain_strain_fibre_dir, subdomain_ids=subdomain_id
            )
            scalar_functions[f"strain_ydir_subdomain_{subdomain_id}"] = partial(
                model.evaluate_subdomain_strain_sheet_dir,
                subdomain_ids=subdomain_id,
            )
            scalar_functions[f"strain_zdir_subdomain_{subdomain_id}"] = partial(
                model.evaluate_subdomain_strain_normal_dir, subdomain_ids=subdomain_id
            )
        """

        # intracellular/extracellular/both:

        intracellular_subdomains = [1] # list(model.subdomains)[:]
        #intracellular_subdomains.remove(0)
        extracellular_subdomain = [0]

        descriptions = ["intracellular"] #, "intracellular", "whole_domain"]
        subdomains = [
            #extracellular_subdomain,
            intracellular_subdomains,
            #model.subdomains,
        ]

        for (desc, subdomain) in zip(descriptions, subdomains):
            # then across all subdomains:
            scalar_functions[f"stress_xdir_{desc}"] = partial(
                model.evaluate_subdomain_stress_fibre_dir, subdomain_ids=subdomain
            )
            scalar_functions[f"stress_ydir_{desc}"] = partial(
                model.evaluate_subdomain_stress_sheet_dir, subdomain_ids=subdomain
            )
            scalar_functions[f"stress_zdir_{desc}"] = partial(
                model.evaluate_subdomain_stress_normal_dir, subdomain_ids=subdomain
            )

            scalar_functions[f"strain_xdir_{desc}"] = partial(
                model.evaluate_subdomain_strain_fibre_dir, subdomain_ids=subdomain
            )
            scalar_functions[f"strain_ydir_{desc}"] = partial(
                model.evaluate_subdomain_strain_sheet_dir,
                subdomain_ids=model.subdomains,
            )
            scalar_functions[f"strain_zdir_{desc}"] = partial(
                model.evaluate_subdomain_strain_normal_dir, subdomain_ids=subdomain
            )

            scalar_functions[f"active_tension_{desc}"] = partial(
                model.evaluate_active_tension, subdomain_ids=subdomain
            )

        scalar_functions[f"relative_shortening"] = model.evaluate_average_shortening

        scalar_values = {"states": []}

        for key in scalar_functions.keys():
            scalar_values[key] = []

        self.scalar_functions, self.scalar_values = scalar_functions, scalar_values

    def update_scalar_functions(self, state_pt):
        """

        Tracks certain values, as scalars; call this after every time step/
        weak form solve.

        """
        scalar_functions, scalar_values = self.scalar_functions, self.scalar_values

        for key in scalar_functions.keys():
            fun = scalar_functions[key]
            scalar_values[key].append(fun())

        scalar_values["states"].append(state_pt)

    def save_and_close(self):
        """

        Closes all files; call this when the simulation is finalized.

        """
        xdmf_files, scalar_values = self.xdmf_files, self.scalar_values

        # Only the first MPI rank will save the numpy arrays to file
        mesh = self.cardiac_model.mesh
        comm = mesh.mpi_comm()
        if comm.rank == 0:
            npy_sv_filename = os.path.join(
                self.full_path, "output_scalar_variables.npy"
            )

            for (key, value) in scalar_values.items():
                scalar_values[key] = np.array(value)

            np.save(npy_sv_filename, scalar_values)

        for _file in list(xdmf_files.values()):
            _file.close()
