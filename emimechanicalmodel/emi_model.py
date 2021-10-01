"""

Åshild Telle / Simula Research Laboratory / 2021

"""

import dolfin as df
from mpi4py import MPI

from emimechanicalmodel.cardiac_model import CardiacModel
from emimechanicalmodel.mesh_setup import assign_discrete_values
from emimechanicalmodel.emi_holzapfelmaterial import EMIHolzapfelMaterial


class EMIModel(CardiacModel):
    """

    Module for our EMI model.

    Note: One of fix_x_bnd, fix_y_bnd and fix_middle must be true.
    No more than one of these can be true.

    Args:
        mesh (df.Mesh): Domain to be used
        experiment (str): Which experiment - "contr", "xstretch" or "ystretch"
        material_properties: parameters to underlying material model,
            default empty dictionary which means default values will be used
        verbose (int): Set to 0 (no verbose output; default), 1 (some),
            or 2 (quite a bit)
    """

    def __init__(
        self,
        mesh,
        volumes,
        experiment,
        material_parameters={},
        verbose=0,
    ):
        # mesh properties, subdomains
        self.volumes = volumes
        mpi_comm = mesh.mpi_comm()

        self.num_subdomains = int(
            mpi_comm.allreduce(max(volumes.array()), op=MPI.MAX) + 1
        )

        if verbose == 2:
            print("Number of subdomains: ", self.num_subdomains)

        U = df.FunctionSpace(mesh, "DG", 0)
        subdomain_map = volumes.array()  # only works for DG-0

        mat_model = EMIHolzapfelMaterial(U, subdomain_map, **material_parameters)

        self.U, self.subdomain_map, self.mat_model = U, subdomain_map, mat_model

        super().__init__(
            mesh,
            experiment,
            verbose,
        )

    def _define_active_strain(self):
        self.active_fn = df.Function(self.U, name="Active strain (-)")
        self.active_fn.vector()[:] = 0  # initial value

    def update_active_fn(self, value):
        assign_discrete_values(self.active_fn, self.subdomain_map, value, 0)
