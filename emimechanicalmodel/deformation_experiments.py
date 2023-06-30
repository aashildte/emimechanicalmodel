"""

Åshild Telle / University of Washington, Simula Research Laboratory / 2023–2022

Implementation of boundary conditions ++ for different passive deformation modes.

TODO:
    - Maybe deformations in the normal direction might make sense if we
      consider a cross-section in this direction?

"""

import dolfin as df
from mpi4py import MPI
import numpy as np
from virtualss import (
    define_boundary_conditions,
    evaluate_normal_load,
    evaluate_shear_load,
)


class DeformationExperiment:
    """

    Class for handling setup for different deformation experiments
    including boundary conditions and evaluation of load on the surface.

    Args:
        mesh (df.Mesh): Mesh used
        V_CG (df.VectorFunctionSpace): Function space for displacement
        experiment (str): Which experiment - ...
        verbose (int): print level; 0, 1 or 2
    """

    def __init__(
        self,
        mesh,
        V_CG,
    ):
        self.V_CG = V_CG
        self.mesh = mesh
        self.normal_vector = df.FacetNormal(mesh)

    def _evaluate_load(self, F, P, wall_idt, unit_vector):
        return -1

    def evaluate_normal_load(self, F, P):
        return -1

    def evaluate_shear_load(self, F, P):
        return -1

    def assign_stretch(self, stretch_value):
        """

        Assign a given stretch or a shear, as a fraction (usually between 0 and 1),
        which will be imposed as Dirichlet BC in the relevant direction.

        """
        print(stretch_value)
        self.bcsfun.k = stretch_value


class Contraction(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)

        bcs, bcsfun, ds = define_boundary_conditions( # TODO fix
            "stretch_ff", "fixed_base", mesh, V_CG
        )

        self.ds = ds

    @property
    def bcs(self):
        return []


class StretchFF(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)

        bcs, bcsfun, ds = define_boundary_conditions(
            "stretch_ff", "fixed_base", mesh, V_CG
        )

        self.bcs = bcs
        self.bcsfun = bcsfun
        self.ds = ds

    def evaluate_normal_load(self, F, P):
        return evaluate_normal_load(F, P, self.mesh, self.ds, 2)


class StretchSS(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        
        bcs, bcsfun, ds = define_boundary_conditions(
            "stretch_ss", "fixed_base", mesh, V_CG
        )

        self.bcs = bcs
        self.bcsfun = bcsfun
        self.ds = ds

    def evaluate_normal_load(self, F, P):
        return evaluate_normal_load(F, P, self.mesh, self.ds, 4)


class StretchNN(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        
        bcs, bcsfun, ds = define_boundary_conditions(
            "stretch_nn", "fixed_base", mesh, V_CG
        )

        self.bcs = bcs
        self.bcsfun = bcsfun
        self.ds = ds

    def evaluate_normal_load(self, F, P):
        return evaluate_normal_load(F, P, self.mesh, self.ds, 6)


class ShearFN(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        
        bcs, bcsfun, ds = define_boundary_conditions(
            "shear_fn", "fixed_base", mesh, V_CG
        )

        self.bcs = bcs
        self.bcsfun = bcsfun
        self.ds = ds

    def evaluate_normal_load(self, F, P):
        return evaluate_normal_load(F, P, self.mesh, self.ds, 2)

    def evaluate_shear_load(self, F, P):
        return evaluate_shear_load(F, P, self.mesh, self.ds, 6)


class ShearFS(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        
        bcs, bcsfun, ds = define_boundary_conditions(
            "shear_fs", "fixed_base", mesh, V_CG
        )

        self.bcs = bcs
        self.bcsfun = bcsfun
        self.ds = ds

    def evaluate_normal_load(self, F, P):
        return evaluate_normal_load(F, P, self.mesh, self.ds, 2)

    def evaluate_shear_load(self, F, P):
        return evaluate_shear_load(F, P, self.mesh, self.ds, 4)


class ShearSF(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        
        bcs, bcsfun, ds = define_boundary_conditions(
            "shear_fs", "fixed_base", mesh, V_CG
        )

        self.bcs = bcs
        self.bcsfun = bcsfun
        self.ds = ds

    def evaluate_normal_load(self, F, P):
        return evaluate_normal_load(F, P, self.mesh, self.ds, 4)

    def evaluate_shear_load(self, F, P):
        return evaluate_shear_load(F, P, self.mesh, self.ds, 2)


class ShearSN(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        
        bcs, bcsfun, ds = define_boundary_conditions(
            "shear_fs", "fixed_base", mesh, V_CG
        )

        self.bcs = bcs
        self.bcsfun = bcsfun
        self.ds = ds

    def evaluate_normal_load(self, F, P):
        return evaluate_normal_load(F, P, self.mesh, self.ds, 4)

    def evaluate_shear_load(self, F, P):
        return evaluate_shear_load(F, P, self.mesh, self.ds, 6)


class ShearNS(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        
        bcs, bcsfun, ds = define_boundary_conditions(
            "shear_ns", "fixed_base", mesh, V_CG
        )

        self.bcs = bcs
        self.bcsfun = bcsfun
        self.ds = ds

    def evaluate_normal_load(self, F, P):
        return evaluate_normal_load(F, P, self.mesh, self.ds, 6)
    
    def evaluate_shear_load(self, F, P):
        return evaluate_shear_load(F, P, self.mesh, self.ds, 4)


class ShearNF(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        
        bcs, bcsfun, ds = define_boundary_conditions(
            "shear_nf", "fixed_base", mesh, V_CG
        )

        self.bcs = bcs
        self.bcsfun = bcsfun
        self.ds = ds

    def evaluate_normal_load(self, F, P):
        return evaluate_normal_load(F, P, self.mesh, self.ds, 6)
    
    def evaluate_shear_load(self, F, P):
        return evaluate_shear_load(F, P, self.mesh, self.ds, 2)


