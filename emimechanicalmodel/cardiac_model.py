"""

Åshild Telle, James D. Trotter / Simula Research Laboratory / 2022

This is the main implementation; all continuum mechanics is defined through this code.
We employ an active strain formulation, through an active function, and  express the
stress-strain relationship through a strain energy function. These both need to be
specified in classes implementing CardiacModel through heritage.

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

from emimechanicalmodel.nonlinear_problem import NewtonSolver, NonlinearProblem


class CardiacModel(ABC):
    def __init__(
        self,
        mesh,
        experiment,
        active_model,
        compressibility_model,
        verbose=0,
    ):
        """

        Module for modeling cardiac tissue in general; meant as an abstract
        module inherited by TissueModel and EMIModel respectively.

        Args:
            mesh (df.Mesh): Domain to be used
            experiment (str): Which experiment - "contraction", "stretch_ff, "strain_fs", ...
            active_model (str): Active model - "active_stress" or "active_strain"
            compressibility_model (str): Compressibility model - "incompressible" or "nearly_incompressible"
            verbose (int): Set to 0 (no verbose output; default), 1 (some),
                or 2 (quite a bit)

        """
        
        self.mesh = mesh
        self.verbose = verbose
        self.active_model = active_model
        self.compressibility_model = compressibility_model
        self.experiment_str = experiment

        # set directions, assuming alignment with the Cartesian axes

        self.fiber_dir = df.as_vector([1, 0, 0])
        self.sheet_dir = df.as_vector([0, 1, 0])
        self.normal_dir = df.as_vector([0, 0, 1])

        # define variational form
        self._define_active_strain()
        self._set_fenics_parameters()
        self._define_state_space()
        self._define_state_functions()
        self._define_kinematic_variables()

        # boundary conditions

        exp_dict = {
            "contraction": Contraction,
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

        self.deformation = exp_dict[experiment](mesh, self.V_CG)
        self.bcs = self.deformation.bcs

        # define solver and initiate tracked variables
        self._define_solver(verbose)
        self._define_projections()

    def _define_solver(self, verbose):
        """

        Defines the problem (initiates the matrices) to be used
        for solving our mechanical problem.

        Args:
            verbose - 0, 1, 2; how much information to pront

        """

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


    def _define_state_space(self):
        """

        Defines function spaces for test and trial functions.
        This function needs to be called before defining boundary
        conditions and weak form.

        Args:
            experiment (str): what kind of experiment to be performed
                (if this is "contraction" we create a subspace for avoiding
                rigid motion using Lagrangian multipliers)

        """

        mesh = self.mesh

        P1 = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        P2 = df.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P3 = df.VectorElement("Real", mesh.ufl_cell(), 0, 6)

        mixed_elements = [P2]

        if self.compressibility_model == "incompressible":
            mixed_elements += [P1]

        if self.experiment_str == "contraction":
            mixed_elements += [P3]

        state_space = df.FunctionSpace(mesh, df.MixedElement(mixed_elements))

        if len(mixed_elements) == 1:
            self.V_CG = state_space
        else:
            self.V_CG = state_space.sub(0)

        self.state_space = state_space

        if self.verbose:
            dofs = len(state_space.dofmap().dofs())
            print("Degrees of freedom: ", dofs, flush=True)


    def _define_state_functions(self):
        """

        Defines function spaces for test and trial functions.
        This function needs to be called before defining boundary
        conditions and weak form.

        Args:
            experiment (str): what kind of experiment to be performed
                (if this is "contraction" we create a subspace for avoiding
                rigid motion using Lagrangian multipliers)

        """

        state_space = self.state_space

        state = df.Function(state_space, name="state")
        test_state = df.TestFunction(state_space)

        u = p = r = v = q = s = df.Constant(0)        # if these are declared they can be ufl variables!

        if self.compressibility_model == "incompressible":
            if self.experiment_str == "contraction":
                u, p, r = df.split(state)
                v, q, s = df.split(test_state)
            else:
                u, p = df.split(state)
                v, q = df.split(test_state)
            self.p = p
        else:
            if self.experiment_str == "contraction":
                u, r = df.split(state)
                v,  _ = df.split(test_state)
            else:
                u = state
                v = test_state

        self.state = state
        self.test_state = test_state
        
        self.u, self.p, self.r, self.v, self.q, self.s = u, p, r, v, q, s

        self.state_functions = [u, p, r]
        self.test_state_functions = [v, q, s]


    def assign_stretch(self, stretch_value):
        """

        Assigns stretch / shear values to the experiments,
        if applicable for the given deformation mode.

        Args:
            stretch_value (float); relative to the size
            of the domain (0.1 = 10% of length between fixed bcs)

        """

        self.deformation.assign_stretch(stretch_value)

    @abstractmethod
    def _define_active_strain(self):
        """

        Defines an active strain function for active contraction;
        supposed to be updated by update_active_fn step by step.
        This function gives us "gamma" in the active strain approach,
        and will be differently defined for tissue and emi models
        respectively.

        """
        pass

    @abstractmethod
    def update_active_fn(self, value):
        """

        Updates the above active active function.

        Args:
            value (float) – value to be assigned to the active strain function

        """
        pass


    def _calculate_P(self, F):
        """

        Defines the first Piola-Kirchhoff stress tensor:
        ..math::
            \mathbf{P} &=& \mathrm{det} (\mathbf{F_a}) \frac{\partial \psi (\mathbf{F_p})}{\partial \mathbf{F_p}} \mathbf{F_a}^{-T} + J p \mathbf{F}^{-T}

        This is calculated either using an active strain or an active stress
        approach. For all passive deformation modes, these should be equal.

        Args:
            F (ufl form) - deformation tensor
        """

        active_fn, comp_model, active_model, mat_model = \
                self.active_fn, self.comp_model, self.active_model, self.mat_model

        assert active_model in ["active_stress", "active_strain"], \
                "Error: Unknown active model, please specify as 'active_stress' or 'active_strain'."
        
        J = df.det(F)

        if active_model == "active_stress":
            e1 = self.fiber_dir
            
            C = pow(J, -float(2) / 3) * F.T * F 
            I4 = df.inner(C * e1, e1)

            psi_active = active_fn / 2.0 * (I4 - 1)
            psi_passive = mat_model.get_strain_energy_term(F)
            psi_comp = comp_model.get_strain_energy_term(F, self.p)
 
            psi = psi_active + psi_passive + psi_comp
            P = df.diff(psi, F)
        else:

            sqrt_fun = (df.Constant(1) - active_fn) ** (-0.5)
            F_a = df.as_tensor(
                ((df.Constant(1) - active_fn, 0, 0), (0, sqrt_fun, 0), (0, 0, sqrt_fun))
            )

            F_e = df.variable(F * df.inv(F_a))
            psi_passive = mat_model.get_strain_energy_term(F_e)
            psi_comp = comp_model.get_strain_energy_term(F_e, self.p)
            psi = psi_passive + psi_comp

            P = df.det(F_a) * df.diff(psi, F_e) * df.inv(F_a.T)

        return P


    @abstractmethod
    def _define_projections(self):
        pass

    def _define_kinematic_variables(self):
        """

        Defines test and trial functions, as well as derived quantities
        and the weak form which we aim to solve.

        """

        u, p, r = self.state_functions
        v, q, _ = self.test_state_functions

        # Kinematics
        d = len(u)
        I = df.Identity(d)  # Identity tensor

        F = df.variable(I + df.grad(u))  # Deformation gradient
        P = self._calculate_P(F)

        C = F.T * F  # the right Cauchy-Green tensor
        E = 0.5 * (C - I)  # the Green-Lagrange strain tensor

        sigma = (1 / df.det(F)) * P * F.T

        # then define the weak form
        weak_form = self._elasticity_term(P, v)

        if self.compressibility_model == "incompressible":
            weak_form += self._pressure_term(q, F)

        if self.experiment_str == "contraction":
            state, test_state = self.state, self.test_state
            weak_form += self._remove_rigid_motion_term(u, r, state, test_state)

        (self.F, self.E, self.sigma, self.P, self.u, self.weak_form,) = (
            F,
            E,
            sigma,
            P,
            u,
            weak_form,
        )

    def _elasticity_term(self, P, v):
        """

        Defines the part of the weak form which defines an equilibrium of
        stresses.

        Args:
            P (ufl form): Piola-Kirchhoff stress tensor
            v (dolfin function): test function for displacement

        Returns:
            weak form term

        """

        return df.inner(P, df.grad(v)) * df.dx


    def _pressure_term(self, q, F):
        """

        Defines the part of the weak form imposing the incompressibility part.

        Args:
            q (dolfin function): test function for the pressure
            F: deformation tensor

        Returns:
            weak form term

        """
        return q * (df.det(F) - 1) * df.dx


    def _remove_rigid_motion_term(self, u, r, state, test_state):
        """

        Defines the part of the weak form which removes rigid motion,
        i.e. non-uniqueness in translation and rotation.

        Args:
            u (dolfin funciton): displacement function
            r (dolfin function): test function for the space of rigid motion
            state, triplet: all three trial functions
            test_state, triplet: all three test functions

        Returns:
            weak form term

        """

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
        """

        The main function to make anything happen; this will solve the
        equation and find appropriate values for u and p solving the
        weak form subject to imposed boundary conditions.

        Typically you would update a given variable, e.g. the active function
        or boundary conditions, then call this function.

        Args:
            project (bool): If True, we update all projection after solving

        """

        # as per default we are using the manual implementation which
        # should be at least as fast; however, we will
        # just keep a simple version here for easy comparison:
         
        df.solve(
            self.weak_form == 0,
            self.state,
            self.bcs,
            solver_parameters={
                "newton_solver": {
                    "absolute_tolerance": 1e-5,
                    "maximum_iterations": 10,
                }
            },
            form_compiler_parameters={"optimize": True},
        )
        """
        self._solver.solve(self.problem, self.state.vector())
        """
        # save stress and strain to fenics functions
        if project:
            for proj_fun in self.projections:
                proj_fun.project()

    def integrate_subdomain(self, fun, subdomain_id):
        """

        Args:
            fun (ufl form): function f that we want to integrate
            subdomain_id (int): cell idt or ECM idt

        Returns:
            ..math::
            \int_{\Omega_subdomain_id} f dx

        """

        dx = df.Measure("dx", domain=self.mesh, subdomain_data=self.volumes)
        return df.assemble(fun * dx(int(subdomain_id)))

    def calculate_volume(self, subdomain_id):
        """

        Args:
            subdomain_id (int): cell idt or ECM idt

        Returns:
            ..math::
            \int_{\Omega_subdomain_id} 1 dx

        """

        return self.integrate_subdomain(1, subdomain_id)

    def evaluate_subdomain_stress(self, unit_vector, subdomain_id):
        """

        Args:
            unit_vector (dolfin vector): vector e determining the direction to
               evaluate the Cauchy stress in
            subdomain_id (int): cell idt or ECM idt

        Returns:
            ..math::
            \int_{\Omega_subdomain_id} v \cdot \sigma v dx

        where
            ..math::
            v = \frac{F \cdot e} \frac{|| F \cdot e ||}

        (see eq. (17) in the paper)

        """
        v = self.F * unit_vector
        v /= df.sqrt(df.dot(v, v))
        stress = df.inner(v, self.sigma * v)
        return self.integrate_subdomain(stress, subdomain_id) / self.calculate_volume(
            subdomain_id
        )

    def evaluate_normal_load(self):
        """

        Evaluates the load in the normal direction of the direction of deformation,
        i.e., in the direction of the normal of the surface being moved.

        Returns:
            normal load L (float)

        """

        return self.deformation.evaluate_normal_load(self.F, self.sigma)

    def evaluate_shear_load(self):
        """

        Evaluates the load in the shear direction of the direction of deformation,
        i.e., in the direction the surface being moved moves.

        Returns:
            shear load L (float)

        """

        return self.deformation.evaluate_shear_load(self.F, self.sigma)

    def evaluate_subdomain_stress_fibre_dir(self, subdomain_id):
        """

        Args:
            subdomain_id (int): cell idt or ECM idt

        Returns:
            ..math:: \overline{\sigma_{ff}}
            (see eq. (17) in the paper)

        """
        unit_vector = self.fiber_dir
        return self.evaluate_subdomain_stress(unit_vector, subdomain_id)

    def evaluate_subdomain_stress_sheet_dir(self, subdomain_id):
        """

        Args:
            subdomain_id (int): cell idt or ECM idt

        Returns:
            ..math:: \overline{\sigma_{ss}}
            (see eq. (17) in the paper)

        """
        unit_vector = self.sheet_dir
        return self.evaluate_subdomain_stress(unit_vector, subdomain_id)

    def evaluate_subdomain_stress_normal_dir(self, subdomain_id):
        """

        Args:
            subdomain_id (int): cell idt or ECM idt

        Returns:
            ..math:: \overline{\sigma_{nn}}
            (see eq. (17) in the paper)

        """
        unit_vector = self.normal_dir
        return self.evaluate_subdomain_stress(unit_vector, subdomain_id)

    def evaluate_subdomain_strain(self, unit_vector, subdomain_id):
        """

        Args:
            unit_vector (dolfin vector): vector e determining the direction to
               evaluate the Cauchy stress in
            subdomain_id (int): cell idt or ECM idt

        Returns:
            ..math::
            \int_{\Omega_subdomain_id} e \cdot E e dx

        (see eq. (16) in the paper)

        """
        strain = df.inner(unit_vector, self.E * unit_vector)
        return self.integrate_subdomain(strain, subdomain_id) / self.calculate_volume(
            subdomain_id
        )

    def evaluate_subdomain_strain_fibre_dir(self, subdomain_id):
        """

        Args:
            subdomain_id (int): cell idt or ECM idt

        Returns:
            ..math:: \overline{E_{ff}}
            (see eq. (16) in the paper)

        """
        unit_vector = self.fiber_dir
        return self.evaluate_subdomain_strain(unit_vector, subdomain_id)

    def evaluate_subdomain_strain_sheet_dir(self, subdomain_id):
        """

        Args:
            subdomain_id (int): cell idt or ECM idt

        Returns:
            ..math:: \overline{E_{ss}}
            (see eq. (16) in the paper)

        """
        unit_vector = self.sheet_dir
        return self.evaluate_subdomain_strain(unit_vector, subdomain_id)

    def evaluate_subdomain_strain_normal_dir(self, subdomain_id):
        """

        Args:
            subdomain_id (int): cell idt or ECM idt

        Returns:
            ..math:: \overline{E_{nn}}
            (see eq. (16) in the paper)

        """
        unit_vector = self.normal_dir
        return self.evaluate_subdomain_strain(unit_vector, subdomain_id)
