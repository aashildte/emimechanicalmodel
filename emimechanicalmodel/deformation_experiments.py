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
        self.V_CG2 = V_CG
        self.surface_normal = df.FacetNormal(mesh)
        self.stretch = df.Constant(0)
        self.dimensions = self.get_dimensions(mesh)
        self.boundaries, self.ds = self.get_boundary_markers(mesh, self.dimensions)
        self.normal_vector = df.FacetNormal(mesh)

    def _evaluate_load(self, F, P, wall_idt, unit_vector):
        load = df.inner(P * self.normal_vector, unit_vector)
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

        self.dim = mesh.topology().dim()

        mpi_comm = mesh.mpi_comm()
        coords = mesh.coordinates()[:]

        xcoords = coords[:, 0]
        ycoords = coords[:, 1]

        xmin = mpi_comm.allreduce(min(xcoords), op=MPI.MIN)
        xmax = mpi_comm.allreduce(max(xcoords), op=MPI.MAX)
        ymin = mpi_comm.allreduce(min(ycoords), op=MPI.MIN)
        ymax = mpi_comm.allreduce(max(ycoords), op=MPI.MAX)
        
        length = xmax - xmin
        width = ymax - ymin
        
        if self.dim > 2:
            zcoords = coords[:, 2]
            
            zmin = mpi_comm.allreduce(min(zcoords), op=MPI.MIN)
            zmax = mpi_comm.allreduce(max(zcoords), op=MPI.MAX)
            height = zmax - zmin
            
            print(f"Domain length={length}, " + f"width={width}, " + f"height={height}")
            dimensions = [[xmin, xmax], [ymin, ymax], [zmin, zmax]]

        else:    
            print(f"Domain length={length}, " + f"width={width}")
            dimensions = [[xmin, xmax], [ymin, ymax]]

        return dimensions

    def get_boundary_markers(self, mesh, dimensions):
        # define subdomains

        boundaries = {
            "x_min": {"subdomain": Wall(0, "min", dimensions), "idt": 1},
            "x_max": {"subdomain": Wall(0, "max", dimensions), "idt": 2},
            "y_min": {"subdomain": Wall(1, "min", dimensions), "idt": 3},
            "y_max": {"subdomain": Wall(1, "max", dimensions), "idt": 4},
        }

        if self.dim > 2:
            boundaries["z_min"] = {"subdomain": Wall(2, "min", dimensions), "idt": 5}
            boundaries["z_max"] = {"subdomain": Wall(2, "max", dimensions), "idt": 6}

        # Mark boundary subdomains

        boundary_markers = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundary_markers.set_all(0)

        for bnd_pair in boundaries.items():
            bnd = bnd_pair[1]
            bnd["subdomain"].mark(boundary_markers, bnd["idt"])

        # df.File("boundaries.pvd") << boundary_markers

        # Redefine boundary measure
        ds = df.Measure("ds", domain=mesh, subdomain_data=boundary_markers)

        return boundaries, ds

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
        L = max_v - min_v

        if self.dim == 2:
            self.bcsfun = df.Expression(("k*L", 0), L=L, k=0, degree=2)
        else:
            self.bcsfun = df.Expression(("k*L", 0, 0), L=L, k=0, degree=2)

    def assign_stretch(self, stretch_value):
        self.bcsfun.k = stretch_value

    def evaluate_normal_load(self, F, P):        
        if self.dim == 3:
            unit_vector = df.as_vector([1.0, 0.0, 0.0])
        else:
            unit_vector = df.as_vector([1.0, 0.0])
        
        wall_idt = self.boundaries["x_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    @property
    def stretch_length(self):
        min_v, max_v = self.dimensions[0]
        return max_v - min_v

    @property
    def bcs(self):
        boundaries, V_CG2, stretch = self.boundaries, self.V_CG2, self.stretch

        xmin = boundaries["x_min"]["subdomain"]
        xmax = boundaries["x_max"]["subdomain"]

        const = df.Constant(np.zeros(self.dim))

        bcs = [
            df.DirichletBC(V_CG2, const, xmin),
            df.DirichletBC(V_CG2, self.bcsfun, xmax),
        ]

        return bcs


class StretchSS(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        min_v, max_v = self.dimensions[1]
        L = max_v - min_v

        if self.dim == 2:
            self.bcsfun = df.Expression((0, "k*L"), L=L, k=0, degree=2)
        else:
            self.bcsfun = df.Expression((0, "k*L", 0), L=L, k=0, degree=2)

    def assign_stretch(self, stretch_value):
        self.bcsfun.k = stretch_value

    def evaluate_normal_load(self, F, P):
        if self.dim == 3:
            unit_vector = df.as_vector([0.0, 1.0, 0.0])
        else:
            unit_vector = df.as_vector([0.0, 1.0])

        wall_idt = self.boundaries["y_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    @property
    def stretch_length(self):
        min_v, max_v = self.dimensions[1]
        return max_v - min_v

    @property
    def bcs(self):
        boundaries, V_CG2, stretch = self.boundaries, self.V_CG2, self.stretch

        ymin = boundaries["y_min"]["subdomain"]
        ymax = boundaries["y_max"]["subdomain"]
       
        const = df.Constant(np.zeros(self.dim))

        bcs = [
            df.DirichletBC(V_CG2, const, ymin),
            df.DirichletBC(V_CG2, self.bcsfun, ymax),
        ]

        return bcs


class StretchNN(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        min_v, max_v = self.dimensions[2]
        L = max_v - min_v

        assert self.dim == 3, "Error: This deformation only makes sense in three dimensions."

        self.bcsfun = df.Expression((0, 0, "k*L"), L=L, k=0, degree=2)

    def assign_stretch(self, stretch_value):
        self.bcsfun.k = stretch_value

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
        boundaries, V_CG2, stretch = self.boundaries, self.V_CG2, self.stretch

        zmin = boundaries["z_min"]["subdomain"]
        zmax = boundaries["z_max"]["subdomain"]

        bcs = [
            df.DirichletBC(V_CG2, df.Constant((0.0, 0.0, 0.0)), zmin),
            df.DirichletBC(V_CG2, self.bcsfun, zmax),
        ]

        return bcs


class ShearNS(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        min_v, max_v = self.dimensions[2]
        L = max_v - min_v
        
        assert self.dim == 3, "Error: This deformation only makes sense in three dimensions."

        self.bcsfun = df.Expression(
            (0, "k*(x[2] - min_v)", 0), min_v=min_v, k=0, degree=2
        )

    def assign_stretch(self, stretch_value):
        self.bcsfun.k = stretch_value

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
        boundaries, V_CG2, stretch = self.boundaries, self.V_CG2, self.stretch

        ymin = boundaries["y_min"]["subdomain"]
        ymax = boundaries["y_max"]["subdomain"]
        zmin = boundaries["z_min"]["subdomain"]
        zmax = boundaries["z_max"]["subdomain"]

        boundaries = [zmin, zmax]
        # boundaries = [ymin, ymax, zmin, zmax]
        bcs = [df.DirichletBC(V_CG2, self.bcsfun, bnd) for bnd in boundaries]

        return bcs


class ShearNF(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        min_v, max_v = self.dimensions[2]
        
        assert self.dim == 3, "Error: This deformation only makes sense in three dimensions."

        self.bcsfun = df.Expression(
            ("k*(x[2] - min_v)", 0, 0), min_v=min_v, k=0, degree=2
        )

    def assign_stretch(self, stretch_value):
        self.bcsfun.k = stretch_value

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
        boundaries, V_CG2, stretch = self.boundaries, self.V_CG2, self.stretch

        xmin = boundaries["x_min"]["subdomain"]
        xmax = boundaries["x_max"]["subdomain"]
        zmin = boundaries["z_min"]["subdomain"]
        zmax = boundaries["z_max"]["subdomain"]

        boundaries = [zmin, zmax]
        # boundaries = [xmin, xmax, zmin, zmax]
        bcs = [df.DirichletBC(V_CG2, self.bcsfun, bnd) for bnd in boundaries]

        return bcs


class ShearFN(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        min_v, max_v = self.dimensions[0]

        self.bcsfun = df.Expression(
            (0, 0, "k*(x[0] - min_v)"), min_v=min_v, k=0, degree=2
        )
        
        assert self.dim == 3, "Error: This deformation only makes sense in three dimensions."

    def assign_stretch(self, stretch_value):
        self.bcsfun.k = stretch_value

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
        boundaries, V_CG2, stretch = self.boundaries, self.V_CG2, self.stretch

        xmin = boundaries["x_min"]["subdomain"]
        xmax = boundaries["x_max"]["subdomain"]
        zmin = boundaries["z_min"]["subdomain"]
        zmax = boundaries["z_max"]["subdomain"]

        boundaries = [xmin, xmax]
        # boundaries = [xmin, xmax, zmin, zmax]
        bcs = [df.DirichletBC(V_CG2, self.bcsfun, bnd) for bnd in boundaries]

        return bcs


class ShearFS(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        min_v, max_v = self.dimensions[0]

        self.bcsfun = df.Expression(
            (0, "k*(x[0] - min_v)", 0), min_v=min_v, k=0, degree=2
        )

    def assign_stretch(self, stretch_value):
        self.bcsfun.k = stretch_value

    def evaluate_normal_load(self, F, P):
        if self.dim == 3:
            unit_vector = df.as_vector([1.0, 0.0, 0.0])
        else:
            unit_vector = df.as_vector([1.0, 0.0])
        
        wall_idt = self.boundaries["x_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    def evaluate_shear_load(self, F, P):
        if self.dim == 3:
            unit_vector = df.as_vector([0.0, 1.0, 0.0])
        else:
            unit_vector = df.as_vector([0.0, 1.0])
        
        wall_idt = self.boundaries["x_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    @property
    def bcs(self):
        boundaries, V_CG2, stretch = self.boundaries, self.V_CG2, self.stretch

        xmin = boundaries["x_min"]["subdomain"]
        xmax = boundaries["x_max"]["subdomain"]
        zmin = boundaries["z_min"]["subdomain"]
        zmax = boundaries["z_max"]["subdomain"]

        boundaries = [xmin, xmax]
        # boundaries = [xmin, xmax, zmin, zmax]
        bcs = [df.DirichletBC(V_CG2, self.bcsfun, bnd) for bnd in boundaries]

        return bcs


class ShearSF(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        min_v, max_v = self.dimensions[1]

        self.bcsfun = df.Expression(
            ("k*(x[1] - min_v)", 0, 0), min_v=min_v, k=0, degree=2
        )

    def assign_stretch(self, stretch_value):
        self.bcsfun.k = stretch_value

    def evaluate_normal_load(self, F, P):
        if self.dim == 2:
            unit_vector = df.as_vector([0.0, 1.0])
        elif self.dim == 3:
            unit_vector = df.as_vector([0.0, 1.0, 0.0])
 
        wall_idt = self.boundaries["y_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    def evaluate_shear_load(self, F, P):
        if self.dim == 2:
            unit_vector = df.as_vector([1.0, 0.0])
        elif self.dim == 3:
            unit_vector = df.as_vector([1.0, 0.0, 0.0])
        
        wall_idt = self.boundaries["y_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    @property
    def bcs(self):
        boundaries, V_CG2, stretch = self.boundaries, self.V_CG2, self.stretch

        ymin = boundaries["y_min"]["subdomain"]
        ymax = boundaries["y_max"]["subdomain"]
        zmin = boundaries["z_min"]["subdomain"]
        zmax = boundaries["z_max"]["subdomain"]

        boundaries = [ymin, ymax]
        bcs = [df.DirichletBC(V_CG2, self.bcsfun, bnd) for bnd in boundaries]

        return bcs


class ShearSN(DeformationExperiment):
    def __init__(self, mesh, V_CG):
        super().__init__(mesh, V_CG)
        min_v, max_v = self.dimensions[1]

        self.bcsfun = df.Expression(
            (0, 0, "k*(x[1] - min_v)"), min_v=min_v, k=0, degree=2
        )
        
        assert self.dim == 3, "Error: This deformation only makes sense in three dimensions."

    def assign_stretch(self, stretch_value):
        self.bcsfun.k = stretch_value

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
        boundaries, V_CG2, stretch = self.boundaries, self.V_CG2, self.stretch

        ymin = boundaries["y_min"]["subdomain"]
        ymax = boundaries["y_max"]["subdomain"]
        zmin = boundaries["z_min"]["subdomain"]
        zmax = boundaries["z_max"]["subdomain"]

        boundaries = [ymin, ymax]
        # boundaries = [ymin, ymax, zmin, zmax]
        bcs = [df.DirichletBC(V_CG2, self.bcsfun, bnd) for bnd in boundaries]

        return bcs


class Wall(df.SubDomain):
    """

    Subdomain class; extracts coordinates for the six walls. Assumes
    all boundaryes are aligned with the x, y and z axes.

    Params:
        index: 0, 1 or 2 for x, y or z
        minmax: 'min' or 'max'; for smallest and largest values for
            chosen dimension
        dimensions: 3 x 2 array giving dimensions of the domain,
            logically following the same flow as index and minmax

    """

    def __init__(self, index, minmax, dimensions):
        super().__init__()

        assert minmax in ["min", "max"], "Error: Let minmax be 'min' or 'max'."

        # extract coordinate for min or max in the direction we're working in
        index_coord = dimensions[index][0 if minmax == "min" else 1]

        self.index, self.index_coord = index, index_coord

    def inside(self, x, on_boundary):
        index, index_coord = self.index, self.index_coord

        return df.near(x[index], index_coord, eps=1e-10) and on_boundary
