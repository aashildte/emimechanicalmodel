"""

Åshild Telle, James Trotter / Simula Research Laboratory / 2021

"""

import dolfinx as df
import ufl

class ProjectionFunction:
    """

    Class for projecting Fenics symbolic functions to actual functions.

    Args:
        org_fun (ufl form): Fenics symbolic relation between functions
        proj_fun (df.Function): Fenics function in a certain function space

    """

    def __init__(self, org_fun, proj_fun):
        self.org_fun = org_fun
        self.proj_fun = proj_fun
        self.proj_space = proj_fun.function_space

        self._define_proj_matrices()

    def _define_proj_matrices(self):
        v = self.org_fun
        V = self.proj_space

        # Define bilinear form for projection and assemble matrix
        mesh = V.mesh
        dx = ufl.dx(mesh, metadata={"quadrature_degree" : 4})
        w = ufl.TestFunction(V)
        Pv = ufl.TrialFunction(V)
        a = df.fem.form(ufl.inner(w, Pv) * dx)
        self.proj_A = df.fem.petsc.assemble_matrix(a)

        # Define linear form for projection
        mesh = V.mesh
        w = ufl.TestFunction(V)
        Pv = ufl.TrialFunction(V)
        self.proj_L = ufl.inner(w, v) * dx

    def project(self):
        """

        Performs the actual projection; call this after solving
        the weak form.

        """

        v = self.org_fun
        V = self.proj_space

        solver_type = "cg"
        preconditioner_type = "jacobi"

        # Assemble right-hand side vector and solve
        b = df.assemble(self.proj_L)

        df.cpp.la.solve(
            self.proj_A,
            self.proj_fun.vector(),
            b,
            solver_type,
            preconditioner_type,
        )
