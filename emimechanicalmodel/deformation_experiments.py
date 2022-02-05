"""

Åshild Telle / Simula Research Laboratory / 2021

"""

import dolfin as df
from mpi4py import MPI


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

    def get_dimensions(self, mesh):
        mpi_comm = mesh.mpi_comm()
        coords = mesh.coordinates()[:]

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
        # define subdomains

        boundaries = {
            "x_min": {"subdomain": Wall(0, "min", dimensions), "idt": 1},
            "x_max": {"subdomain": Wall(0, "max", dimensions), "idt": 2},
            "y_min": {"subdomain": Wall(1, "min", dimensions), "idt": 3},
            "y_max": {"subdomain": Wall(1, "max", dimensions), "idt": 4},
            "z_min": {"subdomain": Wall(2, "min", dimensions), "idt": 5},
            "z_max": {"subdomain": Wall(2, "max", dimensions), "idt": 6},
        }

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

        Assign a given stretch, as a fraction (between 0 and 1), which
        will be imposed as Dirichlet BC in the relevant direction.

        """

        self.stretch.assign(stretch_value * self.stretch_length)


class Contraction(DeformationExperiment):
    
    @property
    def bcs(self):
        return []


class StretchFF(DeformationExperiment):
    def evaluate_load(self, F, P):
        unit_vector = df.as_vector([1.0, 0.0, 0.0])
        wall_idt = self.boundaries["x_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    @property
    def stretch_length(self):
        min_v, max_v = self.dimensions[0]
        return max_v - min_v

    @property
    def bcs(self):
        boundaries, V_CG2, stretch = self.boundaries, self.V_CG2, self.stretch

        bnd_xmin = boundaries["x_min"]["subdomain"]
        bnd_ymin = boundaries["y_min"]["subdomain"]
        bnd_zmin = boundaries["z_min"]["subdomain"]
        bnd_xmax = boundaries["x_max"]["subdomain"]

        bcs = [
            df.DirichletBC(V_CG2.sub(0), df.Constant(0), bnd_xmin),
            df.DirichletBC(V_CG2.sub(1), df.Constant(0), bnd_ymin),
            df.DirichletBC(V_CG2.sub(2), df.Constant(0), bnd_zmin),
            df.DirichletBC(V_CG2.sub(0), stretch, bnd_xmax),
        ]

        return bcs


class StretchSS(DeformationExperiment):
    def evaluate_load(self, F, P):
        unit_vector = df.as_vector([0.0, 1.0, 0.0])
        wall_idt = self.boundaries["y_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    @property
    def stretch_length(self):
        min_v, max_v = self.dimensions[1]
        return max_v - min_v

    @property
    def bcs(self):
        boundaries, V_CG2, stretch = self.boundaries, self.V_CG2, self.stretch

        bnd_xmin = boundaries["x_min"]["subdomain"]
        bnd_ymin = boundaries["y_min"]["subdomain"]
        bnd_zmin = boundaries["z_min"]["subdomain"]
        bnd_ymax = boundaries["y_max"]["subdomain"]

        bcs = [
            df.DirichletBC(V_CG2.sub(0), df.Constant(0), bnd_xmin),
            df.DirichletBC(V_CG2.sub(1), df.Constant(0), bnd_ymin),
            df.DirichletBC(V_CG2.sub(2), df.Constant(0), bnd_zmin),
            df.DirichletBC(V_CG2.sub(1), stretch, bnd_ymax),
        ]

        return bcs


class StretchNN(DeformationExperiment):
    def evaluate_load(self, F, P):
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

        bnd_xmin = boundaries["x_min"]["subdomain"]
        bnd_ymin = boundaries["y_min"]["subdomain"]
        bnd_zmin = boundaries["z_min"]["subdomain"]
        bnd_zmax = boundaries["z_max"]["subdomain"]

        bcs = [
            df.DirichletBC(V_CG2.sub(0), df.Constant(0), bnd_xmin),
            df.DirichletBC(V_CG2.sub(1), df.Constant(0), bnd_ymin),
            df.DirichletBC(V_CG2.sub(2), df.Constant(0), bnd_zmin),
            df.DirichletBC(V_CG2.sub(2), stretch, bnd_zmax),
        ]

        return bcs


class ShearNS(DeformationExperiment):
    def evaluate_load(self, F, P):
        unit_vector = df.as_vector([0.0, 1.0, 0.0])
        wall_idt = self.boundaries["z_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    @property
    def stretch_length(self):
        min_v, max_v = self.dimensions[2]
        return max_v - min_v

    @property
    def bcs(self):
        boundaries, V_CG2, stretch = self.boundaries, self.V_CG2, self.stretch

        bnd_ymin = boundaries["y_min"]["subdomain"]
        bnd_ymax = boundaries["y_max"]["subdomain"]
        bnd_zmin = boundaries["z_min"]["subdomain"]
        bnd_zmax = boundaries["z_max"]["subdomain"]

        bcs = [
            df.DirichletBC(V_CG2.sub(0), df.Constant(0), bnd_ymin),
            df.DirichletBC(V_CG2.sub(2), df.Constant(0), bnd_ymin),
            df.DirichletBC(V_CG2.sub(0), df.Constant(0), bnd_ymax),
            df.DirichletBC(V_CG2.sub(2), df.Constant(0), bnd_ymax),
            df.DirichletBC(V_CG2.sub(0), df.Constant(0), bnd_zmin),
            df.DirichletBC(V_CG2.sub(1), df.Constant(0), bnd_zmin),
            df.DirichletBC(V_CG2.sub(2), df.Constant(0), bnd_zmin),
            df.DirichletBC(V_CG2.sub(0), df.Constant(0), bnd_zmax),
            df.DirichletBC(V_CG2.sub(1), stretch, bnd_zmax),
            df.DirichletBC(V_CG2.sub(2), df.Constant(0), bnd_zmax),
        ]

        return bcs


class ShearNF(DeformationExperiment):
    def evaluate_load(self, F, P):
        unit_vector = df.as_vector([1.0, 0.0, 0.0])
        wall_idt = self.boundaries["z_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    @property
    def stretch_length(self):
        min_v, max_v = self.dimensions[2]
        return max_v - min_v

    @property
    def bcs(self):
        boundaries, V_CG2, stretch = self.boundaries, self.V_CG2, self.stretch

        bnd_xmin = boundaries["x_min"]["subdomain"]
        bnd_xmax = boundaries["x_max"]["subdomain"]
        bnd_zmin = boundaries["z_min"]["subdomain"]
        bnd_zmax = boundaries["z_max"]["subdomain"]

        bcs = [
            df.DirichletBC(V_CG2.sub(1), df.Constant(0), bnd_xmin),
            df.DirichletBC(V_CG2.sub(2), df.Constant(0), bnd_xmin),
            df.DirichletBC(V_CG2.sub(1), df.Constant(0), bnd_xmax),
            df.DirichletBC(V_CG2.sub(2), df.Constant(0), bnd_xmax),
            df.DirichletBC(V_CG2.sub(0), df.Constant(0), bnd_zmin),
            df.DirichletBC(V_CG2.sub(1), df.Constant(0), bnd_zmin),
            df.DirichletBC(V_CG2.sub(2), df.Constant(0), bnd_zmin),
            df.DirichletBC(V_CG2.sub(0), stretch, bnd_zmax),
            df.DirichletBC(V_CG2.sub(1), df.Constant(0), bnd_zmax),
            df.DirichletBC(V_CG2.sub(2), df.Constant(0), bnd_zmax),
        ]

        return bcs


class ShearFN(DeformationExperiment):
    def evaluate_load(self, F, P):
        unit_vector = df.as_vector([0.0, 0.0, 1.0])
        wall_idt = self.boundaries["x_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    @property
    def stretch_length(self):
        min_v, max_v = self.dimensions[2]
        return max_v - min_v

    @property
    def bcs(self):
        boundaries, V_CG2, stretch = self.boundaries, self.V_CG2, self.stretch

        bnd_xmin = boundaries["x_min"]["subdomain"]
        bnd_xmax = boundaries["x_max"]["subdomain"]
        bnd_zmin = boundaries["z_min"]["subdomain"]
        bnd_zmax = boundaries["z_max"]["subdomain"]

        bcs = [
            df.DirichletBC(V_CG2.sub(0), df.Constant(0), bnd_zmin),
            df.DirichletBC(V_CG2.sub(1), df.Constant(0), bnd_zmin),
            df.DirichletBC(V_CG2.sub(0), df.Constant(0), bnd_zmax),
            df.DirichletBC(V_CG2.sub(1), df.Constant(0), bnd_zmax),
            df.DirichletBC(V_CG2.sub(0), df.Constant(0), bnd_xmin),
            df.DirichletBC(V_CG2.sub(1), df.Constant(0), bnd_xmin),
            df.DirichletBC(V_CG2.sub(2), df.Constant(0), bnd_xmin),
            df.DirichletBC(V_CG2.sub(0), df.Constant(0), bnd_xmax),
            df.DirichletBC(V_CG2.sub(1), df.Constant(0), bnd_xmax),
            df.DirichletBC(V_CG2.sub(2), stretch, bnd_xmax),
        ]

        return bcs


class ShearFS(DeformationExperiment):
    def evaluate_load(self, F, P):
        unit_vector = df.as_vector([0.0, 1.0, 0.0])
        wall_idt = self.boundaries["x_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    @property
    def stretch_length(self):
        min_v, max_v = self.dimensions[2]
        return max_v - min_v

    @property
    def bcs(self):
        boundaries, V_CG2, stretch = self.boundaries, self.V_CG2, self.stretch

        bnd_xmin = boundaries["x_min"]["subdomain"]
        bnd_xmax = boundaries["x_max"]["subdomain"]
        bnd_zmin = boundaries["z_min"]["subdomain"]
        bnd_zmax = boundaries["z_max"]["subdomain"]

        bcs = [
            df.DirichletBC(V_CG2.sub(0), df.Constant(0), bnd_zmin),
            df.DirichletBC(V_CG2.sub(2), df.Constant(0), bnd_zmin),
            df.DirichletBC(V_CG2.sub(0), df.Constant(0), bnd_zmax),
            df.DirichletBC(V_CG2.sub(2), df.Constant(0), bnd_zmax),
            df.DirichletBC(V_CG2.sub(0), df.Constant(0), bnd_xmin),
            df.DirichletBC(V_CG2.sub(1), df.Constant(0), bnd_xmin),
            df.DirichletBC(V_CG2.sub(2), df.Constant(0), bnd_xmin),
            df.DirichletBC(V_CG2.sub(0), df.Constant(0), bnd_xmax),
            df.DirichletBC(V_CG2.sub(1), stretch, bnd_xmax),
            df.DirichletBC(V_CG2.sub(2), df.Constant(0), bnd_xmax),
        ]

        return bcs


class ShearSF(DeformationExperiment):
    def evaluate_load(self, F, P):
        unit_vector = df.as_vector([1.0, 0.0, 0.0])
        wall_idt = self.boundaries["y_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    @property
    def stretch_length(self):
        min_v, max_v = self.dimensions[2]
        return max_v - min_v

    @property
    def bcs(self):
        boundaries, V_CG2, stretch = self.boundaries, self.V_CG2, self.stretch

        bnd_ymin = boundaries["y_min"]["subdomain"]
        bnd_ymax = boundaries["y_max"]["subdomain"]
        bnd_zmin = boundaries["z_min"]["subdomain"]
        bnd_zmax = boundaries["z_max"]["subdomain"]

        bcs = [
            df.DirichletBC(V_CG2.sub(1), df.Constant(0), bnd_zmin),
            df.DirichletBC(V_CG2.sub(2), df.Constant(0), bnd_zmin),
            df.DirichletBC(V_CG2.sub(1), df.Constant(0), bnd_zmax),
            df.DirichletBC(V_CG2.sub(2), df.Constant(0), bnd_zmax),
            df.DirichletBC(V_CG2.sub(0), df.Constant(0), bnd_ymin),
            df.DirichletBC(V_CG2.sub(1), df.Constant(0), bnd_ymin),
            df.DirichletBC(V_CG2.sub(2), df.Constant(0), bnd_ymin),
            df.DirichletBC(V_CG2.sub(0), stretch, bnd_ymax),
            df.DirichletBC(V_CG2.sub(1), df.Constant(0), bnd_ymax),
            df.DirichletBC(V_CG2.sub(2), df.Constant(0), bnd_ymax),
        ]

        return bcs


class ShearSN(DeformationExperiment):
    def evaluate_load(self, F, P):
        unit_vector = df.as_vector([0.0, 0.0, 1.0])
        wall_idt = self.boundaries["y_max"]["idt"]

        return self._evaluate_load(F, P, wall_idt, unit_vector)

    @property
    def stretch_length(self):
        min_v, max_v = self.dimensions[2]
        return max_v - min_v

    @property
    def bcs(self):
        boundaries, V_CG2, stretch = self.boundaries, self.V_CG2, self.stretch

        bnd_ymin = boundaries["y_min"]["subdomain"]
        bnd_ymax = boundaries["y_max"]["subdomain"]
        bnd_zmin = boundaries["z_min"]["subdomain"]
        bnd_zmax = boundaries["z_max"]["subdomain"]

        bcs = [
            df.DirichletBC(V_CG2.sub(0), df.Constant(0), bnd_zmin),
            df.DirichletBC(V_CG2.sub(1), df.Constant(0), bnd_zmin),
            df.DirichletBC(V_CG2.sub(0), df.Constant(0), bnd_zmax),
            df.DirichletBC(V_CG2.sub(1), df.Constant(0), bnd_zmax),
            df.DirichletBC(V_CG2.sub(0), df.Constant(0), bnd_ymin),
            df.DirichletBC(V_CG2.sub(1), df.Constant(0), bnd_ymin),
            df.DirichletBC(V_CG2.sub(2), df.Constant(0), bnd_ymin),
            df.DirichletBC(V_CG2.sub(0), df.Constant(0), bnd_ymax),
            df.DirichletBC(V_CG2.sub(1), df.Constant(0), bnd_ymax),
            df.DirichletBC(V_CG2.sub(2), stretch, bnd_ymax),
        ]

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
