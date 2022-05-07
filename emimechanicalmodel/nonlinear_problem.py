"""

James Trotter, Åshild Telle / Simula Research Laboratory / 2021

"""

import dolfin as df

class NonlinearProblem(df.NonlinearProblem):
    """

    Class extending dolfin's NonLinearProblem class,
    makes it possible to save matrices instead if reinitializing
    them for every time step iteration (?).

    Args:
        J: bilinear form - RHS
        F: linear form - LHS
        bcs - boundary conditions (list; can be empty)

    """
    def __init__(self, J, F, bcs):
        self.bilinear_form = J
        self.linear_form = F
        self.bcs = bcs
        self.n = 0
        df.NonlinearProblem.__init__(self)

    def F(self, b, x):
        """
        
        Assembles linear form (LHS)

        Args:
            b - vector of linear system
            x - ?

        """
        df.assemble(self.linear_form, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        """

        Assembles bilinear form (RHS)

        Args:
            A - matrix of linear system
            x - ?

        """
        df.assemble(self.bilinear_form, tensor=A)
        for bc in self.bcs:
            bc.apply(A)


class NewtonSolver(df.NewtonSolver):
    """

    Class extending dolfin's NewtonSolver,setting several of
    the parameters explicitly + making sure we use
    superlu_dist for solving the problem.

    Args:
        mesh - mesh used for the simulation
        verbose - sets printing level; default 0
            verbose > 2 forwards printing level to
            PETScOptions

    """
    def __init__(self, mesh, verbose=0):

        df.NewtonSolver.__init__(
            self, mesh.mpi_comm(), df.PETScKrylovSolver(), df.PETScFactory.instance()
        )

        self.parameters.update(
            {
                "error_on_nonconvergence": False,
                "relative_tolerance": 1e-5,
                "absolute_tolerance": 1e-5,
                "maximum_iterations": 50,
                "relaxation_parameter": 1.0,
            }
        )

        if verbose > 2:
            df.PETScOptions.set("ksp_monitor")
            df.PETScOptions.set("log_view")
            df.PETScOptions.set("ksp_view")
            df.PETScOptions.set("pc_view")

        PETScOptions_params = {
            "ksp_type": "preonly",
            "ksp_norm_type": "unpreconditioned",
            "ksp_error_if_not_converged": False,
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "superlu_dist",
            "mat_superlu_dist_equil": True,
            "mat_superlu_dist_rowperm": "LargeDiag_MC64",
            "mat_superlu_dist_colperm": "PARMETIS",
            "mat_superlu_dist_parsymbfact": True,
            "mat_superlu_dist_replacetinypivot": True,
            "mat_superlu_dist_fact": "DOFACT",
            "mat_superlu_dist_iterrefine": True,
            "mat_superlu_dist_statprint": True,
        }

        for (key, value) in PETScOptions_params.items():
            df.PETScOptions.set(key, value)

        self.linear_solver().set_from_options()

    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)
