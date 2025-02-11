"""

Åshild Telle / Simula Research Laboratiry / 2022

Material model; the Holzapfel-Odgen model adapted to the EMI framework.

"""

import numpy as np
import dolfin as df

try:
    import ufl
except ModuleNotFoundError:
    import ufl_legacy as ufl

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
        a_i=df.Constant(5.70),
        b_i=df.Constant(11.67),
        a_e=df.Constant(1.52),
        b_e=df.Constant(16.31),
        a_if=df.Constant(19.83),
        b_if=df.Constant(24.72),
    ):
        # these are df.Constants, which can be changed from the outside
        self.a_i, self.a_e, self.b_i, self.b_e, self.a_if, self.b_if = (
            a_i,
            a_e,
            b_i,
            b_e,
            a_if,
            b_if,
        )

        self.dim = U.mesh().topology().dim()

        # assign material paramters via characteristic functions
        xi_i = df.Function(U)
        assign_discrete_values(xi_i, subdomain_map, 1, 0)

        xi_e = df.Function(U)
        assign_discrete_values(xi_e, subdomain_map, 0, 1)

        a = a_i * xi_i + a_e * xi_e
        b = b_i * xi_i + b_e * xi_e
        a_f = a_if * xi_i
        b_f = b_if  # set everywhere to avoid division by zero error

        # these are fenics functions defined over all of omega, not likely to be accessed
        self._a, self._b, self._a_f, self._b_f = a, b, a_f, b_f

    def get_strain_energy_term(self, F):

        a, b, a_f, b_f = (
            self._a,
            self._b,
            self._a_f,
            self._b_f,
        )

        if self.dim == 2:
            e1 = df.as_vector([1.0, 0.0])
        elif self.dim == 3:
            e1 = df.as_vector([1.0, 0.0, 0.0])

        J = df.det(F)
        C = J ** 2 * F.T * F
        J_iso = pow(J, -1.0 / float(self.dim))
        C_iso = J_iso ** 2 * F.T * F

        IIFx = df.tr(C_iso)
        I4e1 = df.inner(C_iso * e1, e1)

        cond = lambda a: ufl.conditional(a > 0, a, 0)

        W_hat = a / (2 * b) * (df.exp(b * (IIFx - self.dim)) - 1)
        W_f = a_f / (2 * b_f) * (df.exp(b_f * cond(I4e1 - 1) ** 2) - 1)

        return W_hat + W_f
