"""

Åshild Telle / Simula Research Laboratory / 2022

Material model; the Guccione model adapted to the EMI framework.

"""

import dolfin as df
import ufl

from .mesh_setup import assign_discrete_values


class EMIGuccioneMaterial:
    """

    Adaption of Holzapfel material model to the EMI framework; simply let all material parameters
    be discrete functions, assigned to be zero for anisotropic terms.

    Args:
        U - function space for discrete function; DG-0 is a good choice
        subdomain_map - mapping from volume array to U; for DG-0 this is trivial
        C_i ... b_e - material properties; see paper

    """

    def __init__(
        self,
        U,
        subdomain_map,
        C_i=df.Constant(4),
        C_e=df.Constant(2),
        b_if=df.Constant(40),
        b_it=df.Constant(5),
        b_ift=df.Constant(5),
        b_e=df.Constant(10),
    ):

        self.C_i = C_i
        self.C_e = C_e
        self.b_if = b_if
        self.b_it = b_it
        self.b_ift = b_ift
        self.b_e = b_e

        self.dim = U.mesh().topology().dim()

        # assign material paramters via characteristic functions
        xi_i = df.Function(U)
        assign_discrete_values(xi_i, subdomain_map, 1, 0)

        xi_e = df.Function(U)
        assign_discrete_values(xi_e, subdomain_map, 0, 1)

        C = C_i * xi_i + C_e * xi_e
        b_f = b_if * xi_i + b_e * xi_e
        b_t = b_it * xi_i + b_e * xi_e
        b_ft = b_ift * xi_i + b_e * xi_e

        # these are fenics functions defined over all of omega, not likely to be accessed
        self._C, self._b_f, self._b_t, self._b_ft = C, b_f, b_t, b_ft

    def get_strain_energy_term(self, F):
        C_ss, b_f, b_t, b_ft = self._C, self._b_f, self._b_t, self._b_ft

        if self.dim == 2:
            e1 = df.as_vector([1.0, 0.0])
            e2 = df.as_vector([0.0, 1.0])
        elif self.dim == 3:
            e1 = df.as_vector([1.0, 0.0, 0.0])
            e2 = df.as_vector([0.0, 1.0, 0.0])
            e3 = df.as_vector([0.0, 0.0, 1.0])

        I = df.Identity(self.dim)

        J = df.det(F)
        J_iso = pow(J, -1.0 / float(self.dim))
        C = J_iso ** 2 * F.T * F

        E = 0.5 * (C - I)

        E11, E12 = (
            df.inner(E * e1, e1),
            df.inner(E * e1, e2),
        )
        E21, E22 = (
            df.inner(E * e2, e1),
            df.inner(E * e2, e2),
        )

        if self.dim == 3:
            E13 = df.inner(E * e1, e3)
            E23 = df.inner(E * e2, e3)

            E31, E32, E33 = (
                df.inner(E * e3, e1),
                df.inner(E * e3, e2),
                df.inner(E * e3, e3),
            )

        Q = b_f * E11 ** 2 + b_t * (E22 ** 2 + E33 ** 2) + b_ft * (E12 ** 2 + E21 ** 2)

        if self.dim == 3:
            Q += b_t * (E23 ** 2 + E32 ** 2)
            Q += b_ft * (E13 ** 2 + E31 ** 2)

        # passive strain energy
        Wpassive = C_ss / 2.0 * (df.exp(Q) - 1)

        return Wpassive
