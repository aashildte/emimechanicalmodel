"""

Åshild Telle / Simula Research Laboratory / 2022

Implementation of boundary conditions ++ for different passive deformation modes.

"""

import numpy as np
import dolfinx as df
import ufl
from mpi4py import MPI
from petsc4py import PETSc

class DeformationExperiment():
    """

    Class for handling setup for different deformation experiments
    including boundary conditions and evaluation of load on the surface.

    Args:
        mesh (df.Mesh): Mesh used
        state_space (df.VectorFunctionSpace): Function space for displacement
        experiment (str): Which experiment - ...
        verbose (int): print level; 0, 1 or 2
    """

    def __init__(
        self,
        mesh,
        state_space,
    ):
        self.state_space = state_space
        self.mesh = mesh
        self.surface_normal = ufl.FacetNormal(mesh)
        self.stretch = df.fem.Constant(mesh, PETSc.ScalarType(0))
        self.dimensions = self.get_dimensions(mesh)
        self.boundaries, self.ds = self.get_boundary_markers(mesh, self.dimensions)
        self.normal_vector = ufl.FacetNormal(mesh)

    def _evaluate_load(self, F, P, wall_idt, unit_vector):
        load = df.inner(P*self.normal_vector, unit_vector)
        total_load = df.assemble(load * self.ds(wall_idt))
        area = df.assemble(
            df.det(F)
            * df.inner(df.inv(F).T * unit_vector, unit_vector)
            * self.ds(wall_idt)
        )

        return total_load / area
    
    def evaluate_normal_load(self, F, P):
        return -1
    
    def evaluate_shear_load(self, F, P):
        return -1

    def get_dimensions(self, mesh):
        mpi_comm = MPI.COMM_WORLD # TODO ? mesh.mpi_comm()
        coords = mesh.geometry.x

        xcoords = coords[:, 0]
        ycoords = coords[:, 1]
        zcoords = coords[:, 2]

        xmin = mpi_comm.allreduce(min(xcoords), op=MPI.MIN)
        xmax = mpi_comm.allreduce(max(xcoords), op=MPI.MAX)
        ymin = mpi_comm.allreduce(min(ycoords), op=MPI.MIN)
        ymax = mpi_comm.allreduce(max(ycoords), op=MPI.MAX)
        zmin = mpi_comm.allreduce(min(zcoords), op=MPI.MIN)
        zmax = mpi_comm.allreduce(max(zcoords), op=MPI.MAX)

        length = xmax - xmin
        width = ymax - ymin
        height = zmax - zmin

        print(f"length={length}, " + f"width={width}, " + f"height={height}")

        dimensions = [[xmin, xmax], [ymin, ymax], [zmin, zmax]]

        return dimensions

    def get_boundary_markers(self, mesh, dimensions):

        # functions for walls

        xmin = lambda x : np.isclose(x[0], dimensions[0][0])
        ymin = lambda x : np.isclose(x[1], dimensions[1][0])
        zmin = lambda x : np.isclose(x[2], dimensions[2][0])
        xmax = lambda x : np.isclose(x[0], dimensions[0][1])
        ymax = lambda x : np.isclose(x[1], dimensions[1][1])
        zmax = lambda x : np.isclose(x[2], dimensions[2][1])

        # define subdomains

        boundaries = {
            "x_min": {"bnd_fun": xmin, "idt": 1},
            "x_max": {"bnd_fun": xmax, "idt": 2},
            "y_min": {"bnd_fun": ymin, "idt": 3},
            "y_max": {"bnd_fun": ymax, "idt": 4},
            "z_min": {"bnd_fun": zmin, "idt": 5},
            "z_max": {"bnd_fun": zmax, "idt": 6},
        }

        # Mark boundary subdomains
 
        fdim = 2  # 2D surfaces

        facets = []
        values = []

        for (key, bnd) in boundaries.items():
            wall_fn = bnd["bnd_fun"]
            wall_dofs = df.mesh.locate_entities_boundary(mesh, fdim, wall_fn)
 
            facets.append(wall_dofs)
            values.append(np.full_like(wall_dofs, bnd["idt"]))
            boundaries[key]["dofs"] = wall_dofs

        marked_facets = np.hstack(facets)
        marked_values = np.hstack(values)
        sorted_facets = np.argsort(marked_facets)

        boundary_markers = df.mesh.meshtags(
                mesh,
                fdim,
                marked_facets[sorted_facets],
                marked_values[sorted_facets],
                )

        self.facet_tag = boundary_markers

        # Redefine boundary measure
        metadata = {"quadrature_degree": 4}
        ds = ufl.Measure(
                "ds",
                domain=mesh,
                subdomain_data=boundary_markers,
                metadata=metadata,
                )

        return boundaries, ds

    def _define_deformation_bcs(self, surface_fixed, surface_moving):
        """
        
        Defines Dirichlet BC for two sides, in a pairwise manner.

        Args:
            surface_fixed - dictionary for properties of the fixed surface
            surface_moving - dictionary for properties of the fixed surface

        Both surface dictionary objects needs an "idt" tag which corresponds to
        the meshfunction defined over all surfaces of the mesh (top, bottom, ...).

        """
        state_space, stretch = self.state_space, self.stretch
        mesh = self.mesh
        
        V0, _ = state_space.sub(0).collapse()
        dofs_fixed =  df.fem.locate_dofs_geometrical((state_space.sub(0), V0), surface_fixed["bnd_fun"])
        dofs_moving = df.fem.locate_dofs_geometrical((state_space.sub(0), V0), surface_moving["bnd_fun"])
            
        
        u_fixed = df.fem.Function(V0)
        u_fixed.vector.array[:] = 0
        
        u_moving = df.fem.Function(V0)
        u_moving.vector.array[:] = 0

        bcs = [
            df.fem.dirichletbc(u_fixed, dofs_fixed, state_space.sub(0)),
            df.fem.dirichletbc(u_moving, dofs_moving, state_space.sub(0)),
        ]

        self.bcsfun = u_moving      # this will be updated by respective subclasses

        return bcs



    def assign_stretch(self, stretch_value):
        """

        Assign a given stretch or a shear, as a fraction (usually between 0 and 1),
        which will be imposed as Dirichlet BC in the relevant direction.

        """

        self.stretch.assign(stretch_value * self.stretch_length)
 

class Contraction(DeformationExperiment):
    @property
    def bcs(self):
        return []


class StretchFF(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        min_v, max_v = self.dimensions[0]
        self.L = max_v - min_v

    def assign_stretch(self, stretch_value):
        self.bcsfun.vector.array[:] = stretch_value*self.L #, 0, 0)

    def evaluate_normal_load(self, F, P):
        unit_vector = df.as_vector([1.0, 0.0, 0.0])
        wall_idt = self.boundaries["x_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    @property
    def stretch_length(self):
        min_v, max_v = self.dimensions[0]
        return max_v - min_v

    @property
    def bcs(self):
        fixed_side = self.boundaries["x_min"]
        moving_side = self.boundaries["x_max"]
        
        return self._define_deformation_bcs(fixed_side, moving_side)


class StretchSS(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        min_v, max_v = self.dimensions[1]
        self.L = max_v - min_v

    def assign_stretch(self, stretch_value):
        self.bcsfun.value = (0, stretch_value*self.L, 0)

    def evaluate_normal_load(self, F, P):
        unit_vector = df.as_vector([0.0, 1.0, 0.0])
        wall_idt = self.boundaries["y_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    @property
    def stretch_length(self):
        min_v, max_v = self.dimensions[1]
        return max_v - min_v
    
    @property
    def bcs(self):
        fixed_side = self.boundaries["y_min"]
        moving_side = self.boundaries["y_max"]
        
        return self._define_deformation_bcs(fixed_side, moving_side)


class StretchNN(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        min_v, max_v = self.dimensions[2]
        self.L = max_v - min_v

    def assign_stretch(self, stretch_value):
        self.bcsfun.value = (0, 0, stretch_value*self.L)

    def evaluate_normal_load(self, F, P):
        unit_vector = df.as_vector([0.0, 0.0, 1.0])
        wall_idt = self.boundaries["z_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    @property
    def stretch_length(self):
        min_v, max_v = self.dimensions[2]
        return max_v - min_v
    
    @property
    def bcs(self):
        fixed_side = self.boundaries["z_min"]
        moving_side = self.boundaries["z_max"]
        
        return self._define_deformation_bcs(fixed_side, moving_side)


class ShearNS(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        min_v, max_v = self.dimensions[2]
        self.L = max_v - min_v

    def assign_stretch(self, stretch_value):
        self.bcsfun.value = (0, stretch_value*self.L, 0)

    def evaluate_normal_load(self, F, P):
        unit_vector = df.as_vector([0.0, 0.0, 1.0])
        wall_idt = self.boundaries["z_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    def evaluate_shear_load(self, F, P):
        unit_vector = df.as_vector([0.0, 1.0, 0.0])
        wall_idt = self.boundaries["z_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    @property
    def bcs(self):
        fixed_side = self.boundaries["z_min"]
        moving_side = self.boundaries["z_max"]
        
        return self._define_deformation_bcs(fixed_side, moving_side)


class ShearNF(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        min_v, max_v = self.dimensions[2]
        self.L = max_v - min_v

    def assign_stretch(self, stretch_value):
        self.bcsfun.value = (stretch_value*self.L, 0, 0)

    def evaluate_normal_load(self, F, P):
        unit_vector = df.as_vector([0.0, 0.0, 1.0])
        wall_idt = self.boundaries["z_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    def evaluate_shear_load(self, F, P):
        unit_vector = df.as_vector([1.0, 0.0, 0.0])
        wall_idt = self.boundaries["z_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)
    
    @property
    def bcs(self):
        fixed_side = self.boundaries["z_min"]
        moving_side = self.boundaries["z_max"]
        
        return self._define_deformation_bcs(fixed_side, moving_side)


class ShearFN(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        min_v, max_v = self.dimensions[0]
        self.L = max_v - min_v

    def assign_stretch(self, stretch_value):
        self.bcsfun.value = (0, 0, stretch_value*self.L)

    def evaluate_normal_load(self, F, P):
        unit_vector = df.as_vector([1.0, 0.0, 0.0])
        wall_idt = self.boundaries["x_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    def evaluate_shear_load(self, F, P):
        unit_vector = df.as_vector([0.0, 0.0, 1.0])
        wall_idt = self.boundaries["x_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)
    
    @property
    def bcs(self):
        fixed_side = self.boundaries["x_min"]
        moving_side = self.boundaries["x_max"]
        
        return self._define_deformation_bcs(fixed_side, moving_side)


class ShearFS(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        min_v, max_v = self.dimensions[0] 
        self.L = max_v - min_v

    def assign_stretch(self, stretch_value):
        self.bcsfun.value = (0, stretch_value*self.L, 0)

    def assign_stretch(self, stretch_value):
        self.bcsfun.k = stretch_value

    def evaluate_normal_load(self, F, P):
        unit_vector = df.as_vector([1.0, 0.0, 0.0])
        wall_idt = self.boundaries["x_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    def evaluate_shear_load(self, F, P):
        unit_vector = df.as_vector([0.0, 1.0, 0.0])
        wall_idt = self.boundaries["x_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)
    
    @property
    def bcs(self):
        fixed_side = self.boundaries["x_min"]
        moving_side = self.boundaries["x_max"]
        
        return self._define_deformation_bcs(fixed_side, moving_side)


class ShearSF(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        min_v, max_v = self.dimensions[1]
        self.L = max_v - min_v

    def assign_stretch(self, stretch_value):
        self.bcsfun.value = (stretch_value*self.L, 0, 0)

    def evaluate_normal_load(self, F, P):
        unit_vector = df.as_vector([0.0, 1.0, 0.0])
        wall_idt = self.boundaries["y_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    def evaluate_shear_load(self, F, P):
        unit_vector = df.as_vector([1.0, 0.0, 0.0])
        wall_idt = self.boundaries["y_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    @property
    def bcs(self):
        fixed_side = self.boundaries["y_min"]
        moving_side = self.boundaries["y_max"]
        
        return self._define_deformation_bcs(fixed_side, moving_side)


class ShearSN(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        min_v, max_v = self.dimensions[1]
        self.L = max_v - min_v

    def assign_stretch(self, stretch_value):
        self.bcsfun.value = (0, 0, stretch_value*self.L)

    def evaluate_normal_load(self, F, P):
        unit_vector = df.as_vector([0.0, 1.0, 0.0])
        wall_idt = self.boundaries["y_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    def evaluate_shear_load(self, F, P):
        unit_vector = df.as_vector([0.0, 0.0, 1.0])
        wall_idt = self.boundaries["y_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    @property
    def bcs(self):
        fixed_side = self.boundaries["y_min"]
        moving_side = self.boundaries["y_max"]
        
        return self._define_deformation_bcs(fixed_side, moving_side)
