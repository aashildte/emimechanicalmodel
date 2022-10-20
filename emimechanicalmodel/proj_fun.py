"""

Åshild Telle, James Trotter / Simula Research Laboratory / 2022

"""

import dolfinx as df
import ufl
from petsc4py import PETSc

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
        mesh = V.mesh
        dx = ufl.dx(mesh, metadata={"quadrature_degree" : 4})

        # Define bilinear form for projection and assemble matrix
        w = ufl.TestFunction(V)
        Pv = ufl.TrialFunction(V)
        a = df.fem.form(ufl.inner(w, Pv) * dx)
        self.proj_A = df.fem.petsc.assemble_matrix(a)
        self.proj_A.assemble()

        # Define linear form for projection
        L = ufl.inner(w, v)*dx
        self._rhs = df.fem.form(L)

        self.solver = PETSc.KSP().create(mesh.comm)
        self.solver.setOperators(self.proj_A)

        self.b = df.fem.Function(V)
 
        opts = PETSc.Options()
        option_prefix = self.solver.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "cg"
        opts[f"{option_prefix}pc_type"] = "jacobi"
        self.solver.setFromOptions()


    def project(self):
        """

        Gathers the RHS and erforms the actual projection; call this
        after solving the weak form.

        """

        df.fem.petsc.assemble_vector(self.b.vector, self._rhs)
        self.solver.solve(self.b.vector, x = self.proj_fun.vector)
