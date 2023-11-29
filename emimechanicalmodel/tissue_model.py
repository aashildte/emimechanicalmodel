"""

Åshild Telle / Simula Research Laboratory / 2022

Implementation of a general homogenized tissue model; mostly defined through heritage.
Whatever is implemented here applies for a homogenized model; compare to the
corresponding EMIModel.

"""

import dolfin as df

from emimechanicalmodel.cardiac_model import CardiacModel
from emimechanicalmodel.holzapfelmaterial import HolzapfelMaterial
from emimechanicalmodel.guccionematerial import GuccioneMaterial
from emimechanicalmodel.fomovskymaterial import FomovskyMaterial
from emimechanicalmodel.compressibility import (
    IncompressibleMaterial,
    NearlyIncompressibleMaterial,
)
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
        material_model="holzapfel",
        material_parameters={},
        active_model="active_strain",
        compressibility_model="incompressible",
        compressibility_parameters={},
        verbose=0,
    ):

        self.num_subdomains = 1
        self.subdomains = [0]

        dim = mesh.topology().dim()
        self.volumes = df.MeshFunction("size_t", mesh, dim, 0)
        self.volumes.array()[:] = 0

        material_parameters["dim"] = dim

        if material_model == "holzapfel":
            mat_model = HolzapfelMaterial(**material_parameters)
        elif material_model == "guccione":
            mat_model = GuccioneMaterial(**material_parameters)
        elif material_model == "fomovsky":
            mat_model = FomovskyMaterial(**material_parameters)
        else:
            print(
                "Error: Uknown material model; please specify as 'holzapfel' or 'guccione' or 'fomovsky'."
            )

        if compressibility_model == "incompressible":
            comp_model = IncompressibleMaterial()
        elif compressibility_model == "nearly_incompressible":
            comp_model = NearlyIncompressibleMaterial(**compressibility_parameters)
        else:
            print(
                "Error: Unknown material model; please specify as 'incompressible' or 'nearly_incompressible'."
            )

        self.mat_model, self.comp_model = mat_model, comp_model

        # necessesary for volume integrals etc.
        self.volumes = df.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
        self.volumes.array()[:] = 0

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

        U_CG = df.FunctionSpace(mesh, "CG", 1)
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
