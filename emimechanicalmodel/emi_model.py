"""

Åshild Telle / Simula Research Laboratory & University of Washington / 2022–2023

Implementation of the EMI model; mostly defined through heritage.
Whatever is implemented here is unique for EMI; compare to the corresponding
TissueModel for homogenized version.

"""

import dolfin as df
from mpi4py import MPI
import numpy as np

from emimechanicalmodel.cardiac_model import CardiacModel
from emimechanicalmodel.mesh_setup import assign_discrete_values
from emimechanicalmodel.emi_holzapfelmaterial import EMIHolzapfelMaterial
from emimechanicalmodel.emi_holzapfelmaterial_collagen import EMIMatrixHolzapfelMaterial
from emimechanicalmodel.emi_guccionematerial import EMIGuccioneMaterial
from emimechanicalmodel.compressibility import IncompressibleMaterial, EMINearlyIncompressibleMaterial
from emimechanicalmodel.proj_fun import ProjectionFunction


class EMIModel(CardiacModel):
    """

    Module for our EMI model.

    Note: One of fix_x_bnd, fix_y_bnd and fix_middle must be true.
    No more than one of these can be true.

    Args:
        mesh (df.Mesh): Domain to be used
        experiment (str): Which experiment - "contraction", "stretch_ff", "shear_fs", ...
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
        material_parameters={},
        active_model="active_strain",
        compressibility_model="incompressible",
        compressibility_parameters={},
        verbose=0,
    ):
        # mesh properties, subdomains
        self.verbose = verbose
        self.volumes = volumes
        self.set_subdomains(volumes)

        U = df.FunctionSpace(mesh, "DG", 0)
        subdomain_map = volumes.array()  # only works for DG-0
        self.xi_i = df.Function(U)
        assign_discrete_values(self.xi_i, subdomain_map, 1, 0)

        if material_model=="holzapfel":
            mat_model = EMIHolzapfelMaterial(U, subdomain_map, **material_parameters)
        elif material_model=="holzapfel_collagen":
            mat_model = EMIMatrixHolzapfelMaterial(U, subdomain_map, **material_parameters)
        elif material_model=="guccione":
            mat_model = EMIGuccioneMaterial(U, subdomain_map, **material_parameters)
        else:
            print("Error: Uknown material model; please specify as 'holzapfel', 'holzapfel_collagen', or 'guccione'.")


        if compressibility_model=="incompressible":
            comp_model = IncompressibleMaterial()
        elif compressibility_model=="nearly_incompressible":
            comp_model = EMINearlyIncompressibleMaterial(U, subdomain_map, **compressibility_parameters)
        else:
            print("Error: Unknown material model; please specify as 'incompressible' or 'nearly_incompressible'.")


        self.U, self.subdomain_map, self.mat_model, self.comp_model = \
                U, subdomain_map, mat_model, comp_model

        super().__init__(
            mesh,
            experiment,
            active_model,
            compressibility_model,
            verbose,
        )
        

    def set_subdomains(self, volumes):
        mpi_comm = MPI.COMM_WORLD
        rank = mpi_comm.Get_rank()
        
        local_subdomains = set(volumes.array())
        subdomains = mpi_comm.gather(local_subdomains, root=0)

        if rank == 0:
            global_subdomains = []
            for s in subdomains:
                global_subdomains += s
            global_subdomains = list(set(global_subdomains))
        else:
            global_subdomains = None

        self.subdomains = mpi_comm.bcast(global_subdomains, root=0)
        self.num_subdomains = max(self.subdomains) 

        if self.verbose == 2:
            print(f"Local subdomains (rank {rank}):{local_subdomains}")  
            print("Number of subdomains in total: ", self.num_subdomains)
            print(f"Global subdomains:{self.subdomains}")  


    def _define_active_fn(self):
        """

        Defines an active strain/stress function for active contraction;
        supposed to be updated by update_active_fn step by step.
        This function gives us "gamma" in the active strain approach,
        and "T_a" in the active stress approach.

        """
        self.active_fn = df.Function(self.U, name="Active tension")
        self.active_fn.vector()[:] = 0  # initial value


    def update_active_fn(self, value):
        """

        Updates the above active active function.

        Args:
            value (float) – value to be assigned to the active strain functionn
               defined as non-zero over the intracellular domain

        """
        assign_discrete_values(self.active_fn, self.subdomain_map, value, 0) 

    
    def evaluate_collagen_stress_fiber_direction(self):
        """

        Evaluates the collagen fiber direction stress (fiber direction normal component).

        """

        assert isinstance(self.mat_model, EMIMatrixHolzapfelMaterial), "Error: Collagen fibers not defined."

        unit_vector, _ = self.mat_model.collagen_field
        return self.evaluate_subdomain_stress(unit_vector, 0)

    
    def evaluate_collagen_stress_transverse_direction(self):
        """

        Evaluates the collagen fiber direction stress (transverse fiber direction normal component).

        """

        assert isinstance(self.mat_model, EMIMatrixHolzapfelMaterial), "Error: Collagen fibers not defined."

        _, unit_vector = self.mat_model.collagen_field
        return self.evaluate_subdomain_stress(unit_vector, 0)
    

    def evaluate_collagen_strain_fiber_direction(self):
        """

        Evaluates the collagen fiber direction strain (fiber direction normal component).

        """

        assert isinstance(self.mat_model, EMIMatrixHolzapfelMaterial), "Error: Collagen fibers not defined."

        unit_vector, _ = self.mat_model.collagen_field
        return self.evaluate_subdomain_strain(unit_vector, 0)

    
    def evaluate_collagen_strain_transverse_direction(self):
        """

        Evaluates the collagen fiber direction strain (transverse fiber direction normal component).

        """

        assert isinstance(self.mat_model, EMIMatrixHolzapfelMaterial), "Error: Collagen fibers not defined."

        _, unit_vector = self.mat_model.collagen_field
        return self.evaluate_subdomain_strain(unit_vector, 0)


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

        if isinstance(self.mat_model, EMIMatrixHolzapfelMaterial):
            sigma_collagen = df.Function(U_DG, name="Cauchy stress (kPa)")
            E_collagen = df.Function(U_DG, name="Strain (-)")
            e1, _ = self.mat_model.collagen_field

            sigma_collagen_proj = ProjectionFunction(df.inner(self.sigma*e1, e1), sigma_collagen)
            E_collagen_proj = ProjectionFunction(df.inner(self.E*e1, e1), E_collagen)

        self.u_DG = u_DG
        self.p_DG = p_DG
        self.E_DG = E_DG
        self.sigma_DG = sigma_DG
        self.PiolaKirchhoff_DG = P_DG

        self.tracked_variables = [u_DG, p_DG, E_DG, sigma_DG, P_DG]
        self.projections = [u_proj, p_proj, E_proj, sigma_proj, P_proj]

        if isinstance(self.mat_model, EMIMatrixHolzapfelMaterial):
            self.tracked_variables += [sigma_collagen, E_collagen]
            self.projections += [sigma_collagen_proj, E_collagen_proj]

