"""

Åshild Telle / Simula Research Laboratory / 2021

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
        self.active_fn = df.Constant(0.0, name="Active strain (-)")

    def update_active_fn(self, value):
        self.active_fn.assign(value)

    def _define_projections(self):
        mesh = self.mesh
        
        # define function spaces

        V_CG = df.VectorFunctionSpace(mesh, "CG", 2)
        T_CG = df.TensorFunctionSpace(mesh, "CG", 2)

        # define functions

        u = df.Function(V_CG, name="Displacement (µm)")
        
        E = df.Function(T_CG, name="Strain") 
        sigma = df.Function(T_CG, name="Cauchy stress (kPa)")
        P = df.Function(T_CG, name="Piola-Kirchhoff stress (kPa)")

        # then projection objects

        u_proj = ProjectionFunction(self.u, u)
        E_proj = ProjectionFunction(self.E, E) 
        sigma_proj = ProjectionFunction(self.sigma, sigma)
        P_proj = ProjectionFunction(self.P, P)

        self.projections = [u_proj, E_proj, sigma_proj, P_proj]

        # gather tracked functions into a list for easy access
        self.tracked_variables = [u, E, sigma, P]
