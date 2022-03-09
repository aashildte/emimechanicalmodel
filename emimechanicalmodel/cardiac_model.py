"""

Åshild Telle, James D. Trotter / Simula Research Laboratory / 2021

"""

from abc import ABC, abstractmethod
import dolfin as df

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
from emimechanicalmodel.proj_fun import ProjectionFunction
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
        self._set_fenics_parameters()
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

        self.fiber_dir = df.as_vector([1, 0, 0])
        self.sheet_dir = df.as_vector([0, 1, 0])
        self.normal_dir = df.as_vector([0, 0, 1])

        self.bcs = self.exp.bcs

        # define solver and initiate tracked variables
        self._define_solver(verbose)
        self._define_projections()

    def _define_solver(self, verbose):
        J = df.fem.formmanipulations.derivative(self.weak_form, self.state)
        self.problem = NonlinearProblem(J, self.weak_form, self.bcs)
        self._solver = NewtonSolver(self.mesh, verbose=verbose)

    def _set_fenics_parameters(self):
        """

        Default parameters for form compiler + solver

        """

        df.parameters["form_compiler"]["cpp_optimize"] = True
        df.parameters["form_compiler"]["representation"] = "uflacs"
        df.parameters["form_compiler"]["quadrature_degree"] = 4

    def _define_state_space(self, experiment):

        # needs to be called before setting exp conds + weak form

        mesh = self.mesh

        P1 = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        P2 = df.VectorElement("Lagrange", mesh.ufl_cell(), 2)

        if experiment == "contr":
            P3 = df.VectorElement("Real", mesh.ufl_cell(), 0, 6)
            state_space = df.FunctionSpace(mesh, df.MixedElement([P1, P2, P3]))
        else:
            state_space = df.FunctionSpace(mesh, df.MixedElement([P1, P2]))

        self.V_CG = state_space.sub(1)
        self.state_space = state_space

        if self.verbose:
            dofs = len(state_space.dofmap().dofs())
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

        sqrt_fun = (df.Constant(1) - active_fn) ** (-0.5)
        F_a = df.as_tensor(
            ((df.Constant(1) - active_fn, 0, 0), (0, sqrt_fun, 0), (0, 0, sqrt_fun))
        )

        F_e = df.variable(F * df.inv(F_a))
        psi = mat_model.passive_component(F_e)

        P = df.det(F_a) * df.diff(psi, F_e) * df.inv(F_a.T) + self.p * J * df.inv(F.T)

        return P

    def _define_projections(self):
        mesh = self.mesh
        mat_model = self.mat_model

        # define function spaces

        U_DG = df.FunctionSpace(mesh, "DG", 2)
        V_DG = df.VectorFunctionSpace(mesh, "DG", 2)
        V_CG = df.VectorFunctionSpace(mesh, "CG", 2)
        T_DG = df.TensorFunctionSpace(mesh, "DG", 2)

        # define functions

        u_DG = df.Function(V_DG, name="Displacement ($\mu$m)")
        u_CG = df.Function(V_CG, name="Displacement ($\mu$m)")
        E_DG = df.Function(T_DG, name="Strain")
        sigma_DG = df.Function(T_DG, name="Cauchy stress (kPa)")
        P_DG = df.Function(T_DG, name="Piola-Kirchhoff stress (kPa)")

        # then projection objects

        u_proj_DG = ProjectionFunction(self.u, u_DG)
        u_proj_CG = ProjectionFunction(self.u, u_CG)
        E_proj = ProjectionFunction(self.E, E_DG)
        sigma_proj = ProjectionFunction(self.sigma, sigma_DG)
        P_proj = ProjectionFunction(self.P, P_DG)

        self.projections = [u_proj_DG, u_proj_CG, E_proj, sigma_proj, P_proj]

        self.u_DG, self.u_CG, self.E_DG, self.sigma_DG, self.P_DG = u_DG, u_CG, E_DG, sigma_DG, P_DG

        # gather tracked functions into a list for easy access

        self.tracked_variables = [u_DG, u_CG, E_DG, sigma_DG, P_DG]


    def _define_kinematic_variables(self, experiment):
        state_space = self.state_space

        state = df.Function(state_space, name="state")
        test_state = df.TestFunction(state_space)

        if experiment == "contr":
            p, u, r = df.split(state)
            q, v, _ = df.split(test_state)
        else:
            p, u = df.split(state)
            q, v = df.split(test_state)

        self.state = state
        self.p = p

        # Kinematics
        d = len(u)
        I = df.Identity(d)  # Identity tensor

        F = df.variable(I + df.grad(u))  # Deformation gradient
        J = df.det(F)
        P = self._calculate_P(F, J)

        C = F.T * F  # the right Cauchy-Green tensor
        E = 0.5 * (C - I)  # the Green-Lagrange strain tensor

        N = df.FacetNormal(self.mesh)
        sigma = (1 / df.det(F)) * P * F.T

        # weak form
        weak_form = df.inner(P, df.grad(v)) * df.dx

        if experiment == "contr":
            weak_form += self.remove_rigid_motion_term(u, r, state, test_state)

        weak_form += q * (J - 1) * df.dx  # incompressible term

        (self.F, self.E, self.sigma, self.P, self.u, self.weak_form,) = (
            F,
            E,
            sigma,
            P,
            u,
            weak_form,
        )

    def remove_rigid_motion_term(self, u, r, state, test_state):

        position = df.SpatialCoordinate(self.mesh)

        RM = [
            df.Constant((1, 0, 0)),
            df.Constant((0, 1, 0)),
            df.Constant((0, 0, 1)),
            df.cross(position, df.Constant((1, 0, 0))),
            df.cross(position, df.Constant((0, 1, 0))),
            df.cross(position, df.Constant((0, 0, 1))),
        ]

        Pi = sum(df.dot(u, zi) * r[i] * df.dx for i, zi in enumerate(RM))

        return df.derivative(Pi, state, test_state)

    def solve(self, project=True):
        # just keep the simple version here for easy comparison:
        df.solve(self.weak_form == 0, self.state, self.exp.bcs)
        
        #self._solver.solve(self.problem, self.state.vector())
        

        # save stress and strain to fenics functions
        if project:
            for proj_fun in self.projections:
                proj_fun.project()

    def calculate_volume(self, subdomain_id):
        return self.integrate_subdomain(1, subdomain_id)

    def integrate_subdomain(self, fun, subdomain_id):
        dx = df.Measure("dx", domain=self.mesh, subdomain_data=self.volumes)
        return df.assemble(fun * dx(int(subdomain_id)))

    def evaluate_subdomain_stress(self, unit_vector, subdomain_id):
        v = self.F * unit_vector
        v /= df.sqrt(df.dot(v, v))
        stress = df.inner(v, self.sigma * v)
        return self.integrate_subdomain(stress, subdomain_id) / self.calculate_volume(
            subdomain_id
        )

    def evaluate_normal_load(self):
        return self.exp.evaluate_normal_load(self.F, self.P)
    
    def evaluate_shear_load(self):
        return self.exp.evaluate_shear_load(self.F, self.P)

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
