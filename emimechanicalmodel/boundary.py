"""

Åshild Telle / Simula Research Laboratory / 2021

"""

import dolfin as df
from mpi4py import MPI


class Boundary:
    """

    Class for handling boundary conditions

    Args:
        mesh (df.Mesh): Mesh used
        V_CG (df.VectorFunctionSpace): Function space for displacement
        experiment (str): Which experiment - "contr", "xstretch" or "ystretch"
        verbose (int): print level; 0, 1 or 2
    """

    def __init__(
        self,
        mesh,
        V_CG,
        experiment,
        verbose=0,
    ):
        self.verbose = verbose

        self.surface_normal = df.FacetNormal(mesh)
        self.stretch = df.Constant(0)

        self._set_dimensions(mesh)
        self._define_boundary_markers(mesh)
        self.wall, self.stretch_length, self.bcs = self._define_boundary_conditions(
            experiment, V_CG
        )

    def _evaluate_load(self, F, load, wall):
        idt = self.boundaries[wall]["id"]
        surface_normal = self.surface_normal

        total_load = df.assemble(load * self.ds(idt))
        area = df.assemble(
            df.det(F)
            * df.inner(df.inv(F).T * surface_normal, surface_normal)
            * self.ds(idt)
        )

        return total_load / area

    def evaluate_load_yz(self, F, load):
        """

        Calculates load, ie surface stresses, on the 'x_max' wall;
        relevant for stretch in the fiber direction.

        """

        return self._evaluate_load(F, load, "x_max")

    def evaluate_load_xz(self, F, load):
        """

        Calculates load, ie surface stresses, on the 'y_max' wall;
        relevant for stretch in the sheetfiber direction.

        """

        return self._evaluate_load(F, load, "y_max")

    def evaluate_load_xy(self, F, load):
        """

        Calculates load, ie surface stresses, on the 'z_max' wall;
        relevant for potential stretch in the normal direction.

        """
        return self._evaluate_load(F, load, "z_max")

    def _set_dimensions(self, mesh):
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

        dims = [[xmin, xmax], [ymin, ymax], [zmin, zmax]]

        if self.verbose >= 1:
            length = xmax - xmin
            width = ymax - ymin
            height = zmax - zmin

            print(f"length={length}, " + f"width={width}, " + f"height={height}")

        self.dimensions = dims

    def _define_boundary_markers(self, mesh):
        dims = self.dimensions
        # define subdomains

        boundaries = {
            "x_min": {"subdomain": Wall(0, "min", dims), "id": 1},
            "x_max": {"subdomain": Wall(0, "max", dims), "id": 2},
            "y_min": {"subdomain": Wall(1, "min", dims), "id": 3},
            "y_max": {"subdomain": Wall(1, "max", dims), "id": 4},
            "z_min": {"subdomain": Wall(2, "min", dims), "id": 5},
            "z_max": {"subdomain": Wall(2, "max", dims), "id": 6},
        }

        # Mark boundary subdomains

        boundary_markers = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundary_markers.set_all(0)

        for bnd_pair in boundaries.items():
            bnd = bnd_pair[1]
            bnd["subdomain"].mark(boundary_markers, bnd["id"])

        # df.File("boundaries.pvd") << boundary_markers

        # Redefine boundary measure
        ds = df.Measure("ds", domain=mesh, subdomain_data=boundary_markers)

        self.boundaries, self.ds = boundaries, ds

    def assign_stretch(self, stretch_value):
        """

        Assign a given stretch, as a fraction (between 0 and 1), which
        will be imposed as Dirichlet BC in the relevant direction.

        """
        self.stretch.assign(stretch_value * self.stretch_length)

    def _define_x_bnd_cond(self, V):
        boundaries = self.boundaries

        bnd_xmin = boundaries["x_min"]["subdomain"]
        bnd_ymin = boundaries["y_min"]["subdomain"]
        bnd_zmin = boundaries["z_min"]["subdomain"]
        bnd_xmax = boundaries["x_max"]["subdomain"]

        bcs = [
            df.DirichletBC(V.sub(0), df.Constant(0), bnd_xmin),
            df.DirichletBC(V.sub(1), df.Constant(0), bnd_ymin),
            df.DirichletBC(V.sub(2), df.Constant(0), bnd_zmin),
            df.DirichletBC(V.sub(0), self.stretch, bnd_xmax),
        ]

        xmin, xmax = self.dimensions[0]
        length = xmax - xmin

        return "x_max", length, bcs

    def _define_y_bnd_cond(self, V):
        boundaries = self.boundaries

        bnd_xmin = boundaries["x_min"]["subdomain"]
        bnd_ymin = boundaries["y_min"]["subdomain"]
        bnd_zmin = boundaries["z_min"]["subdomain"]
        bnd_ymax = boundaries["y_max"]["subdomain"]

        bcs = [
            df.DirichletBC(V.sub(0), df.Constant(0), bnd_xmin),
            df.DirichletBC(V.sub(1), df.Constant(0), bnd_ymin),
            df.DirichletBC(V.sub(2), df.Constant(0), bnd_zmin),
            df.DirichletBC(V.sub(1), self.stretch, bnd_ymax),
        ]

        ymin, ymax = self.dimensions[1]
        width = ymax - ymin

        return "y_max", width, bcs

    def _define_boundary_conditions(self, experiment, V):
        assert experiment in [
            "contr",
            "xstretch",
            "ystretch",
        ], "Error: experiment must be 'contr', 'xstretch' or 'ystretch', " + \
                f"current value: {experiment}"

        if experiment == "contr":
            wall, stretch_length, bcs = None, 0, []  # enforce in weak form instead

        if experiment == "xstretch":
            wall, stretch_length, bcs = self._define_x_bnd_cond(V)

        if experiment == "ystretch":
            wall, stretch_length, bcs = self._define_y_bnd_cond(V)

        return wall, stretch_length, bcs


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
