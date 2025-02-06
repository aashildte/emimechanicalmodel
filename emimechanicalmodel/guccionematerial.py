"""

Åshild Telle / Simula Research Laboratory / 2022

"""

import numpy as np
import dolfin as df

try:
    import ufl
except ModuleNotFoundError:
    import ufl_legacy as ufl


class GuccioneMaterial:
    """

    Underlying material model, using Holzapfels strain energy function.

    Args:
        C ... b_ts : Material properties

    """

    def __init__(
        self,
        C=df.Constant(0.5),
        b_f=df.Constant(8),
        b_t=df.Constant(2),
        b_ft=df.Constant(4),
        dim=3,
    ):

        self.C = C
        self.b_s = b_f
        self.b_t = b_t
        self.b_ft = b_ft
        self.dim = dim

        self._C, self._b_f, self._b_t, self._b_ft = C, b_f, b_t, b_ft

    def get_strain_energy_term(self, F):
        C_ss, b_f, b_t, b_ft = self.C, self.b_f, self.b_t, self.b_ft

        if self.dim == 2:
            e1 = df.as_vector([1.0, 0.0])
            e2 = df.as_vector([0.0, 1.0])
        else:
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
            E13 = (df.inner(E * e1, e3),)
            E23 = (df.inner(E * e2, e3),)

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
