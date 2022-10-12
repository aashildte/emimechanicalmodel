"""

Åshild Telle, James D. Trotter / Simula Research Laboratory / 2022

This is the main implementation; all continuum mechanics is defined through this code.
We employ an active strain formulation, through an active function, and  express the
stress-strain relationship through a strain energy function. These both need to be
specified in classes implementing CardiacModel through heritage.

"""

from abc import ABC, abstractmethod
import dolfinx as df
import ufl
import numpy as np

from emimechanicalmodel.deformation_experiments import (
    Contraction,
    StretchFF,
    StretchSS,
    StretchNN,
    ShearNS,
    ShearNF,
    ShearFN,
    ShearFS,
    ShearSF,
    ShearSN,
)

from emimechanicalmodel.nonlinear_problem import NewtonSolver, NonlinearProblem


class CardiacModel(ABC):
    def __init__(
        self,
        mesh,
        experiment,
        verbose=0,
    ):
        """

        Module for modeling cardiac tissue in general; meant as an abstract
        module inherited by TissueModel and EMIModel respectively.

        Args:
            mesh (df.Mesh): Domain to be used
            experiment (str): Which experiment - "contr", "xstretch" or "ystretch"
            verbose (int): Set to 0 (no verbose output; default), 1 (some),
                or 2 (quite a bit)
        """
        # mesh properties
        self.mesh = mesh

        # write things out or not
        self.verbose = verbose

        # define variational form
        self._define_active_strain()
        self._define_state_space(experiment)
        self._define_kinematic_variables(experiment)

        # boundary conditions

        exp_dict = {
            "contr": Contraction,
            "stretch_ff": StretchFF,
            "stretch_ss": StretchSS,
            "stretch_nn": StretchNN,
            "shear_ns": ShearNS,
            "shear_nf": ShearNF,
            "shear_fn": ShearFN,
            "shear_fs": ShearFS,
            "shear_sf": ShearSF,
            "shear_sn": ShearSN,
        }

        self.exp = exp_dict[experiment](mesh, self.V_CG)

        self.fiber_dir = ufl.as_vector([1, 0, 0])
        self.sheet_dir = ufl.as_vector([0, 1, 0])
        self.normal_dir = ufl.as_vector([0, 0, 1])

        self.bcs = self.exp.bcs

        # define solver and initiate tracked variables
        self._define_solver(verbose)
        self._define_projections()

    def _define_solver(self, verbose):
        J = df.fem.formmanipulations.derivative(self.weak_form, self.state)
        self.problem = NonlinearProblem(J, self.weak_form, self.bcs)
        self._solver = NewtonSolver(self.mesh, verbose=verbose)

    def _define_state_space(self, experiment):

        # needs to be called before setting exp conds + weak form

        mesh = self.mesh
 
        P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)

        if experiment == "contr":
            R = ufl.VectorElement("Real", mesh.ufl_cell(), 0, 6)
            state_space = df.fem.FunctionSpace(mesh, P2 * P1 * R)
        else:
            state_space = df.fem.FunctionSpace(mesh, P2 * P1)

        self.V_CG = state_space.sub(0)
        self.state_space = state_space

        if self.verbose:
            dofs = state_space.dofmap.index_map.size_global
            print("Degrees of freedom: ", dofs, flush=True)

    def assign_stretch(self, stretch_value):
        self.exp.assign_stretch(stretch_value)

    @abstractmethod
    def _define_active_strain(self):
        pass

    @abstractmethod
    def update_active_fn(self, value):
        pass

    def _calculate_P(self, F, J):
        active_fn, mat_model = self.active_fn, self.mat_model

        sqrt_fun = (1 - active_fn)**(-0.5)
        F_a = ufl.as_tensor(
            ((1 - active_fn, 0, 0), (0, sqrt_fun, 0), (0, 0, sqrt_fun))
        )

        F_e = ufl.variable(F * ufl.inv(F_a))
        psi = mat_model.passive_component(F_e) 

        P = ufl.det(F_a) * ufl.diff(psi, F_e) * ufl.inv(F_a.T) + self.p * J * ufl.inv(F.T)

        return P

    @abstractmethod
    def _define_projections(self):
        pass

    def _define_kinematic_variables(self, experiment):
        state_space = self.state_space
        
        state = df.fem.Function(state_space, name="state")
        test_functions = ufl.TestFunctions(state_space)

        if experiment == "contr":
            u, p, r = state.split()
            v, q, _ = test_functions
        else:
            u, p = state.split()
            v, q = test_functions

        self.state = state
        self.p = p

        # Kinematics
        d = len(u)
        I = ufl.Identity(d)  # Identity tensor

        F = ufl.variable(I + ufl.grad(u))   # Deformation gradient; TODO check if ufl.variable is actually needed
        J = ufl.det(F)
        P = self._calculate_P(F, J)

        C = F.T * F  # the right Cauchy-Green tensor
        E = 0.5 * (C - I)  # the Green-Lagrange strain tensor

        sigma = (1 / ufl.det(F)) * P * F.T

        dx = ufl.dx(metadata={"quadrature_degree": 4})

        # weak form
        weak_form = ufl.inner(P, ufl.grad(v)) * dx

        if experiment == "contr":
            weak_form += self.remove_rigid_motion_term(u, r, state, test_state, dx)

        weak_form += q * (J - 1) * dx  # incompressible term
        
        (self.F, self.E, self.sigma, self.P, self.u, self.weak_form,) = (
            F,
            E,
            sigma,
            P,
            u,
            weak_form,
        )

    def remove_rigid_motion_term(self, u, r, state, test_state, dx):

        position = df.SpatialCoordinate(self.mesh)

        RM = [
            df.Constant((1, 0, 0)),
            df.Constant((0, 1, 0)),
            df.Constant((0, 0, 1)),
            df.cross(position, df.Constant((1, 0, 0))),
            df.cross(position, df.Constant((0, 1, 0))),
            df.cross(position, df.Constant((0, 0, 1))),
        ]

        Pi = sum(df.dot(u, zi) * r[i] * dx for i, zi in enumerate(RM))

        return df.derivative(Pi, state, test_state)

    def solve(self, project=True):
        # just keep the simple version here for easy comparison:
        
        df.solve(
            self.weak_form == 0,
            self.state,
            self.exp.bcs,
            solver_parameters={
                "newton_solver": {
                    "absolute_tolerance": 1e-5,
                    "maximum_iterations": 10,
                }
            },
            form_compiler_parameters={"optimize": True},
            metadata={"quadrature_degree": 4},
        )
        """
        self._solver.solve(self.problem, self.state.vector())
        """
        # save stress and strain to fenics functions
        if project:
            for proj_fun in self.projections:
                proj_fun.project()


    def calculate_volume(self, subdomain_id):
        return self.integrate_subdomain(1, subdomain_id)

    def integrate_subdomain(self, fun, subdomain_id):
        dx = df.Measure("dx", domain=self.mesh, subdomain_data=self.volumes, metadata={"quadrature_degree": 4})
        return df.assemble(fun * dx(int(subdomain_id)))

    def evaluate_subdomain_stress(self, unit_vector, subdomain_id):
        v = self.F * unit_vector
        v /= df.sqrt(df.dot(v, v))
        stress = df.inner(v, self.sigma * v)
        return self.integrate_subdomain(stress, subdomain_id) / self.calculate_volume(
            subdomain_id
        )

    def evaluate_normal_load(self):
        return self.exp.evaluate_normal_load(self.F, self.sigma)

    def evaluate_shear_load(self):
        return self.exp.evaluate_shear_load(self.F, self.sigma)

    def evaluate_subdomain_stress_fibre_dir(self, subdomain_id):
        unit_vector = self.fiber_dir
        return self.evaluate_subdomain_stress(unit_vector, subdomain_id)

    def evaluate_subdomain_stress_transfibre_dir(self, subdomain_id):
        unit_vector = self.sheet_dir
        return self.evaluate_subdomain_stress(unit_vector, subdomain_id)

    def evaluate_subdomain_stress_normal_dir(self, subdomain_id):
        unit_vector = self.normal_dir
        return self.evaluate_subdomain_stress(unit_vector, subdomain_id)

    def evaluate_subdomain_strain(self, unit_vector, subdomain_id):
        strain = df.inner(unit_vector, self.E * unit_vector)
        return self.integrate_subdomain(strain, subdomain_id) / self.calculate_volume(
            subdomain_id
        )

    def evaluate_subdomain_strain_fibre_dir(self, subdomain_id):
        unit_vector = self.fiber_dir
        return self.evaluate_subdomain_strain(unit_vector, subdomain_id)

    def evaluate_subdomain_strain_transfibre_dir(self, subdomain_id):
        unit_vector = self.sheet_dir
        return self.evaluate_subdomain_strain(unit_vector, subdomain_id)

    def evaluate_subdomain_strain_normal_dir(self, subdomain_id):
        unit_vector = self.normal_dir
        return self.evaluate_subdomain_strain(unit_vector, subdomain_id)
