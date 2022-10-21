"""

Åshild Telle / Simula Research Laboratory / 2022

Implementation of the EMI model; mostly defined through heritage.
Whatever is implemented here is unique for EMI; compare to the corresponding
TissueModel for homogenized version.

"""

import dolfin as df
from mpi4py import MPI

from emimechanicalmodel.cardiac_model import CardiacModel
from emimechanicalmodel.mesh_setup import assign_discrete_values
from emimechanicalmodel.emi_holzapfelmaterial import EMIHolzapfelMaterial
from emimechanicalmodel.emi_guccionematerial import EMIGuccioneMaterial
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
        material_model="holzapfel",
        active_model="active_strain",
        compressibility_model="incompressible",
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

        if material_model=="holzapfel":
            mat_model = EMIHolzapfelMaterial(U, subdomain_map, **material_parameters)
        elif material_model=="guccione":
            mat_model = EMIGuccioneMaterial(U, subdomain_map, **material_parameters)
        else:
            print("Error: Uknown material model.")

        self.U, self.subdomain_map, self.mat_model = U, subdomain_map, mat_model

        super().__init__(
            mesh,
            experiment,
            active_model,
            compressibility_model,
            verbose,
        )

    def _define_active_strain(self):
        """

        Defines an active strain function for active contraction;
        supposed to be updated by update_active_fn step by step.
        This function gives us "gamma" in the active strain approach.

        """

        self.active_fn = df.Function(self.U, name="Active strain (-)")
        self.active_fn.vector()[:] = 0  # initial value

    def update_active_fn(self, value):
        """

        Updates the above active active function.

        Args:
            value (float) – value to be assigned to the active strain functionn
               defined as non-zero over the intracellular domain

        """

        assign_discrete_values(self.active_fn, self.subdomain_map, value, 0)

    def _define_projections(self):
        """

        Defines projection objects which tracks different variables of
        interest as CG functions, defined as scalars, vectors, or tensors.

        If project is set to true in the solve call, these will be updated,
        and (for efficiency) not otherwise.

        """

        mesh = self.mesh

        # define function spaces

        U_DG = df.FunctionSpace(mesh, "DG", 1)
        V_DG = df.VectorFunctionSpace(mesh, "DG", 2)
        T_DG = df.TensorFunctionSpace(mesh, "DG", 2)

        p_DG = df.Function(U_DG, name="Hydrostatic pressure (kPa))")
        u_DG = df.Function(V_DG, name="Displacement (µm)")
        E_DG = df.Function(T_DG, name="Strain")
        sigma_DG = df.Function(T_DG, name="Cauchy stress (kPa)")
        P_DG = df.Function(T_DG, name="Piola-Kirchhoff stress (kPa)")

        p_proj = ProjectionFunction(self.p, p_DG)
        u_proj = ProjectionFunction(self.u, u_DG)
        E_proj = ProjectionFunction(self.E, E_DG)
        sigma_proj = ProjectionFunction(self.sigma, sigma_DG)
        P_proj = ProjectionFunction(self.P, P_DG)

        self.tracked_variables = [u_DG, p_DG, E_DG, sigma_DG, P_DG]
        self.projections = [u_proj, p_proj, E_proj, sigma_proj, P_proj]
