"""

Åshild Telle / Simula Research Laboratory / 2022

Material model based on the Holzapfel-Odgen model (2009).

"""

import numpy as np
import dolfin as df
import ufl


class HolzapfelMaterial:
    """

    Underlying material model, using Holzapfels strain energy function.

    Args:
        a ... b_fs : Material properties, see original paper

    """

    def __init__(
        self,
        a=df.Constant(0.074),
        b=df.Constant(4.878),
        a_f=df.Constant(2.628),
        b_f=df.Constant(5.214),
        a_s=df.Constant(0.438),
        b_s=df.Constant(3.002),
        a_fs=df.Constant(0.062),
        b_fs=df.Constant(3.476),
        dim=3,
    ):
        
        self.dim = dim

        self.a, self.b, self.a_f, self.b_f, self.a_s, self.b_s, self.a_fs, self.b_fs = (
            a,
            b,
            a_f,
            b_f,
            a_s,
            b_s,
            a_fs,
            b_fs,
        )

    def get_strain_energy_term(self, F):
        a, b, a_f, b_f, a_s, b_s, a_fs, b_fs = (
            self.a,
            self.b,
            self.a_f,
            self.b_f,
            self.a_s,
            self.b_s,
            self.a_fs,
            self.b_fs,
        )
        
        if self.dim == 3:
            e1 = df.as_vector([1.0, 0.0, 0.0])
            e2 = df.as_vector([0.0, 1.0, 0.0])
        else:
            e1 = df.as_vector([1.0, 0.0])
            e2 = df.as_vector([0.0, 1.0])

        J = df.det(F)
        C = pow(J, -float(2) / 3) * F.T * F

        IIFx = df.tr(C)
        I4e1 = df.inner(C * e1, e1)
        I4e2 = df.inner(C * e2, e2)
        I8e1e2 = df.inner(C * e1, e2)

        cond = lambda a: ufl.conditional(a > 0, a, 0)

        W_hat = a / (2 * b) * (df.exp(b * (IIFx - 3)) - 1)
        W_f = a_f / (2 * b_f) * (df.exp(b_f * cond(I4e1 - 1) ** 2) - 1)
        W_s = a_s / (2 * b_s) * (df.exp(b_s * cond(I4e2 - 1) ** 2) - 1)
        W_fs = a_fs / (2 * b_fs) * (df.exp(b_fs * (I8e1e2**2)) - 1)
        W_ani = W_f + W_s + W_fs

        return W_hat + W_ani
