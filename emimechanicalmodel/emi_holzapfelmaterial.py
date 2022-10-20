"""

Åshild Telle / Simula Research Laboratiry / 2022

Material model; the Holzapfel-Odgen model adapted to the EMI framework.

"""

import dolfinx as df
import ufl
from petsc4py import PETSc

from .mesh_setup import assign_discrete_values


class EMIHolzapfelMaterial:
    """

    Adaption of Holzapfel material model to the EMI framework; simply let all material parameters
    be discrete functions, assigned to be zero for anisotropic terms.

    Args:
        U - function space for discrete function; DG-0 is a good choice
        subdomain_map - mapping from volume array to U; for DG-0 this is trivial
        a_i ... b_if - material properties; see paper

    """
    def __init__(
        self,
        U,
        subdomain_map,
        a_i=5.70,
        b_i=11.67,
        a_e=1.52,
        b_e=16.31,
        a_if=19.83,
        b_if=24.72,
    ):
        mesh = U.mesh

        # convert to constants

        a_i = df.fem.Constant(mesh, PETSc.ScalarType(a_i))
        b_i = df.fem.Constant(mesh, PETSc.ScalarType(b_i))
        a_if = df.fem.Constant(mesh, PETSc.ScalarType(a_if))
        b_if = df.fem.Constant(mesh, PETSc.ScalarType(b_if))
        a_e = df.fem.Constant(mesh, PETSc.ScalarType(a_e))
        b_e = df.fem.Constant(mesh, PETSc.ScalarType(b_e))

        # assign material paramters via characteristic functions
        xi_i = df.fem.Function(U)
        assign_discrete_values(xi_i, subdomain_map, 1, 0)

        xi_e = df.fem.Function(U)
        assign_discrete_values(xi_e, subdomain_map, 0, 1)

        a = a_i*xi_i + a_e*xi_e
        b = b_i*xi_i + b_e*xi_e
        a_f = a_if*xi_i
        b_f = b_if        # set everywhere to avoid division by zero error

        # these are constants and can be altered from the outside
        self.a_i, self.b_i, self.a_if, self.b_if, self.a_e, self.b_e = \
                a_i, b_i, a_if, b_if, a_e, b_e

        # these are fenics functions defined over all of omega, not likely to be accessed
        self._a, self._b, self._a_f, self._b_f = a, b, a_f, b_f
        self.xi_e = xi_e
        self.xi_i = xi_i

    def passive_component(self, F):

        a, b, a_f, b_f = (
            self._a,
            self._b,
            self._a_f,
            self._b_f,
        )

        e1 = ufl.as_vector([1.0, 0.0, 0.0])

        J = ufl.det(F)
        C = pow(J, -float(2) / 3) * F.T * F

        IIFx = ufl.tr(C)
        I4e1 = ufl.inner(C * e1, e1)

        cond = lambda a: ufl.conditional(a > 0, a, 0)

        W_hat = a / (2 * b) * (ufl.exp(b * (IIFx - 3)) - 1)
        W_f = a_f / (2 * b_f) * (ufl.exp(b_f * cond(I4e1 - 1) ** 2) - 1)

        return W_hat + W_f
