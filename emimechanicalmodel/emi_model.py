"""

Åshild Telle / Simula Research Laboratory / 2021

Implementation of the EMI model; mostly defined through heritage.
Whatever is implemented here is unique for EMI; compare to the corresponding
TissueModel for homogenized version.

"""

import dolfinx as df
from mpi4py import MPI

from emimechanicalmodel.cardiac_model import CardiacModel
from emimechanicalmodel.mesh_setup import assign_discrete_values
from emimechanicalmodel.emi_holzapfelmaterial import EMIHolzapfelMaterial
from emimechanicalmodel.proj_fun import ProjectionFunction


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
        project_to_subspaces=False,
    ):
        # mesh properties, subdomains
        self.volumes = volumes
        self.project_to_subspaces = project_to_subspaces

        mpi_comm = mesh.comm

        self.num_subdomains = int(
            mpi_comm.allreduce(max(volumes.values), op=MPI.MAX) + 1
        )

        if verbose == 2:
            print("Number of subdomains: ", self.num_subdomains)

        U = df.fem.FunctionSpace(mesh, ("DG", 0))
        subdomain_map = volumes.values       # only works for DG-0

        mat_model = EMIHolzapfelMaterial(U, subdomain_map, **material_parameters)

        self.U, self.subdomain_map, self.mat_model = U, subdomain_map, mat_model

        super().__init__(
            mesh,
            experiment,
            verbose,
        )

    def _define_active_strain(self):
        self.active_fn = df.fem.Function(self.U, name="Active_strain")
        self.active_fn.vector.array[:] = 0  # initial value

    def update_active_fn(self, value):
        assign_discrete_values(self.active_fn, self.subdomain_map, value, 0)

    def _define_projections(self):
        mesh = self.mesh
        
        # define function spaces

        V_DG = df.fem.VectorFunctionSpace(mesh, ("DG", 2))
        T_DG = df.fem.TensorFunctionSpace(mesh, ("DG", 2))
        
        u_DG = df.fem.Function(V_DG, name="Displacement (µm)")
        E_DG = df.fem.Function(T_DG, name="Strain")
        sigma_DG = df.fem.Function(T_DG, name="Cauchy stress (kPa)")
        P_DG = df.fem.Function(T_DG, name="Piola-Kirchhoff stress (kPa)")

        u_proj = ProjectionFunction(self.u, u_DG)
        E_proj = ProjectionFunction(self.E, E_DG)
        sigma_proj = ProjectionFunction(self.sigma, sigma_DG)
        P_proj = ProjectionFunction(self.P, P_DG)
        
        self.tracked_variables = [u_DG, E_DG, sigma_DG, P_DG]
        self.projections = [u_proj, E_proj, sigma_proj, P_proj]
