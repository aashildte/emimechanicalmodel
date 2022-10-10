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

        #mpi_comm = mesh.mpi_comm()
        mpi_comm = MPI.COMM_WORLD       # TODO doublecheck that this works in parallel

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
        self.active_fn = df.fem.Function(self.U, name="Active strain (-)")
        self.active_fn.vector.array[:] = 0  # initial value

    def update_active_fn(self, value):
        assign_discrete_values(self.active_fn, self.subdomain_map, value, 0)

    def _get_subdomains(self):
        num_subdomains = self.num_subdomains
        
        # works with master branch of FEniCS:
        #subdomains = [df.MeshView(self.volumes, i) for i in range(num_subdomains)]
        
        subdomains = [df.SubMesh(self.mesh, self.volumes, i) for i in range(num_subdomains)]
        return subdomains

    def _define_projections(self):
        mesh = self.mesh
        
        # define function spaces

        V_DG = df.VectorFunctionSpace(mesh, "DG", 2)
        T_DG = df.TensorFunctionSpace(mesh, "DG", 2)
        
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

        if self.project_to_subspaces:
            subspaces_variables, subspaces_projections = \
                    self._define_submesh_projections(u_DG, E_DG, sigma_DG, P_DG)
            self.tracked_variables += subspaces_variables
            self.projections += subspaces_projections


    def _define_submesh_projections(self, u_DG, E_DG, sigma_DG, P_DG):

        subdomains = self._get_subdomains()
        num_subdomains = self.num_subdomains

        V_CG_subdomains = [df.VectorFunctionSpace(sd, "CG", 2) for sd in subdomains]
        T_CG_subdomains = [df.TensorFunctionSpace(sd, "CG", 2) for sd in subdomains]
        

        u_subdomains = [
            df.fem.Function(V_CG_subdomains[i], name=f"Displacement subdomain {i} (µm)")
            for i in range(num_subdomains)
        ]


        E_subdomains = [
            df.fem.Function(T_CG_subdomains[i], name=f"Strain subdomain {i} (-)")
            for i in range(num_subdomains)
        ]


        sigma_subdomains = [
            df.fem.Function(T_CG_subdomains[i], name=f"Cauchy stress subdomain {i} (kPa)")
            for i in range(num_subdomains)
        ]


        P_subdomains = [
            df.fem.Function(
                T_CG_subdomains[i], name=f"Piola-Kirchhoff stress subdomain {i} (kPa)"
            )
            for i in range(num_subdomains)
        ]

        # then projection objects

        u_proj_subdomains = [ProjectionFunction(u_DG, u_sub) for u_sub in u_subdomains]
        E_proj_subdomains = [ProjectionFunction(E_DG, E_sub) for E_sub in E_subdomains]

        sigma_proj_subdomains = [
            ProjectionFunction(sigma_DG, s_sub) for s_sub in sigma_subdomains
        ]

        P_proj_subdomains = [ProjectionFunction(P_DG, P_sub) for P_sub in P_subdomains]

        tracked_variables = u_subdomains + E_subdomains + sigma_subdomains + P_subdomains
        projections = u_proj_subdomains + E_proj_subdomains + sigma_proj_subdomains + P_proj_subdomains

        return tracked_variables, projections
