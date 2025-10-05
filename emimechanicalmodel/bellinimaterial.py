"""


Material model based on formulation used by Bellini et al (2013).

"""

import numpy as np
import dolfin as df

try:
    import ufl
except ModuleNotFoundError:
    import ufl_legacy as ufl


class BelliniMaterial:
    """

    Underlying material model

    """

    def __init__(
        self,
        c_iso=df.Constant(1.65),
        k1=df.Constant(2.08),
        k2=df.Constant(3.67),
        k3=df.Constant(1.13),
        k4=df.Constant(1.25),
        dim=3,
    ):
        self.dim = dim

        self.c_iso, self.k1, self.k2, self.k3, self.k4 = c_iso, k1, k2, k3, k4


    def get_strain_energy_term(self, F):
        c_iso, k1, k2, k3, k4 = self.c_iso, self.k1, self.k2, self.k3, self.k4

        if self.dim == 3:
            e1 = df.as_vector([1.0, 0.0, 0.0])
            e2 = df.as_vector([0.0, 1.0, 0.0])
        else:
            e1 = df.as_vector([1.0, 0.0])
            e2 = df.as_vector([0.0, 1.0])

        J = df.det(F)
        J_iso = pow(J, -1.0 / float(self.dim))
        C = J_iso ** 2 * F.T * F

        IIFx = df.tr(C)
        I4e1 = df.inner(C * e1, e1)
        I4e2 = df.inner(C * e2, e2)

        W_hat = c_iso*(IIFx - self.dim)
        W_f = k1 / (2 * k2) * (df.exp(k2 * (I4e1 - 1) ** 2) - 1)
        W_s = k3 / (2 * k4) * (df.exp(k4 * (I4e2 - 1) ** 2) - 1)
        W_ani = W_f + W_s

        return W_hat + W_ani
