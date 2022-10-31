"""

Åshild Telle / Simula Research Laboratory / 2022

"""

import dolfin as df
import ufl

class GuccioneMaterial():
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
    ):

        self.C = C
        self.b_s = b_f
        self.b_t = b_t
        self.b_ft = b_ft

        self._C, self._b_f, self._b_t, self._b_ft = C, b_f, b_t, b_ft

    def get_strain_energy_term(self, F):
        C_ss, b_f, b_t, b_ft = self.C, self.b_f, self.b_t, self.b_ft

        e1 = df.as_vector([1.0, 0.0, 0.0])
        e2 = df.as_vector([0.0, 1.0, 0.0])
        e3 = df.as_vector([0.0, 0.0, 1.0])

        I = df.Identity(3)
        J = df.det(F)
        C = pow(J, -float(2) / 3) * F.T * F
        E = 0.5 * (C - I)

        E11, E12, E13 = (
            df.inner(E * e1, e1),
            df.inner(E * e1, e2),
            df.inner(E * e1, e3),
        )
        E21, E22, E23 = (
            df.inner(E * e2, e1),
            df.inner(E * e2, e2),
            df.inner(E * e2, e3),
        )
        E31, E32, E33 = (
            df.inner(E * e3, e1),
            df.inner(E * e3, e2),
            df.inner(E * e3, e3),
        )

        Q = (
            b_f * E11 ** 2
            + b_t * (E22 ** 2 + E33 ** 2 + E23 ** 2 + E32 ** 2)
            + b_ft * (E12 ** 2 + E21 ** 2 + E13 ** 2 + E31 ** 2)
        )

        # passive strain energy
        Wpassive = C_ss / 2.0 * (df.exp(Q) - 1)

        return Wpassive
