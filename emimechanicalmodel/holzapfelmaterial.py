"""

Åshild Telle / Simula Research Laboratory / 2022

Material model based on the Holzapfel-Odgen model (2009).

"""

import ufl


class HolzapfelMaterial:
    """

    Underlying material model, using Holzapfels strain energy function.

    Args:
        a ... b_fs : Material properties, see original paper

    """

    def __init__(
        self,
        a=0.074,
        b=4.878,
        a_f=2.628,
        b_f=5.214,
        a_s=0.438,
        b_s=3.002,
        a_fs=0.062,
        b_fs=3.476,
    ):

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

    def passive_component(self, F):
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

        e1 = ufl.as_vector([1.0, 0.0, 0.0])
        e2 = ufl.as_vector([0.0, 1.0, 0.0])

        J = ufl.det(F)
        C = pow(J, -float(2) / 3) * F.T * F

        IIFx = ufl.tr(C)
        I4e1 = ufl.inner(C * e1, e1)
        I4e2 = ufl.inner(C * e2, e2)
        I8e1e2 = ufl.inner(C * e1, e2)

        cond = lambda a: ufl.conditional(a > 0, a, 0)

        W_hat = a / (2 * b) * (ufl.exp(b * (IIFx - 3)) - 1)
        W_f = a_f / (2 * b_f) * (ufl.exp(b_f * cond(I4e1 - 1) ** 2) - 1)
        W_s = a_s / (2 * b_s) * (ufl.exp(b_s * cond(I4e2 - 1) ** 2) - 1)
        W_fs = a_fs / (2 * b_fs) * (ufl.exp(b_fs * (I8e1e2**2)) - 1)
        W_ani = W_f + W_s + W_fs

        return W_hat + W_ani
