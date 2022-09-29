"""

Åshild Telle / Simula Research Laboratory / 2022

Implementation of a general homogenized tissue model; mostly defined through heritage.
Whatever is implemented here applies for a homogenized model; compare to the
corresponding EMIModel.

"""

import dolfin as df

from emimechanicalmodel.cardiac_model import CardiacModel
from emimechanicalmodel.holzapfelmaterial import HolzapfelMaterial
from emimechanicalmodel.proj_fun import ProjectionFunction


class TissueModel(CardiacModel):
    """

    Module for modeling cardiac tissue in general; simpler tissue model,
    implemented for comaparison. Assuming homogeneous active strain/stress.

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
        experiment,
        material_parameters={},
        verbose=0,
    ):

        self.mat_model = HolzapfelMaterial(**material_parameters)
        self.num_subdomains = 1

        # necessesary for volume integrals etc.
        self.volumes = df.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
        self.volumes.array()[:] = 0

        super().__init__(
            mesh,
            experiment,
            verbose,
        )

    def _define_active_strain(self):
        """

        Defines an active strain function for active contraction;
        supposed to be updated by update_active_fn step by step.
        This function gives us "gamma" in the active strain approach.

        """

        self.active_fn = df.Constant(0.0, name="Active strain (-)")


    def update_active_fn(self, value):
        """

        Updates the above active active function.

        Args:
            value (float) – value to be assigned to the active strain function

        """
        self.active_fn.assign(value)



    def _define_projections(self):
        """
        
        Defines projection objects which tracks different variables of
        interest as CG functions, defined as scalars, vectors, or tensors.

        If project is set to true in the solve call, these will be updated,
        and (for efficiency) not otherwise.

        """

        mesh = self.mesh
        
        # define function spaces

        U_CG = df.VectorFunctionSpace(mesh, "CG", 1)
        V_CG = df.VectorFunctionSpace(mesh, "CG", 2)
        T_CG = df.TensorFunctionSpace(mesh, "CG", 2)

        # define functions
        
        p = df.Function(U_CG, name="Hydrostatic pressure (kPa)")
        u = df.Function(V_CG, name="Displacement (µm)")
        
        E = df.Function(T_CG, name="Strain") 
        sigma = df.Function(T_CG, name="Cauchy stress (kPa)")
        P = df.Function(T_CG, name="Piola-Kirchhoff stress (kPa)")

        # then projection objects

        p_proj = ProjectionFunction(self.p, p)
        u_proj = ProjectionFunction(self.u, u)
        E_proj = ProjectionFunction(self.E, E) 
        sigma_proj = ProjectionFunction(self.sigma, sigma)
        P_proj = ProjectionFunction(self.P, P)

        self.projections = [u_proj, p_proj, E_proj, sigma_proj, P_proj]

        # gather tracked functions into a list for easy access
        self.tracked_variables = [u, p, E, sigma, P]
