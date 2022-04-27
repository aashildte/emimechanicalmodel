"""

Åshild Telle, James Trotter / Simula Research Laboratory / 2021

"""

import dolfin as df

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
        self.proj_space = proj_fun.function_space()

        self._define_proj_matrices()

    def _define_proj_matrices(self):
        v = self.org_fun
        V = self.proj_space

        # Define bilinear form for projection and assemble matrix
        mesh = V.mesh()
        dx = df.dx(mesh)
        w = df.TestFunction(V)
        Pv = df.TrialFunction(V)
        a = df.inner(w, Pv) * dx
        self.proj_A = df.assemble(a)

        # Define linear form for projection
        mesh = V.mesh()
        dx = df.dx(mesh)
        w = df.TestFunction(V)
        Pv = df.TrialFunction(V)
        self.proj_L = df.inner(w, v) * dx

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
