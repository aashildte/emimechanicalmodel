"""

Åshild Telle / Simula Research Laboratiry / 2022

Material model; the Holzapfel-Odgen model adapted to the EMI framework.

"""

import numpy as np
import dolfin as df
import ufl

class FomovskyMaterial:

    def __init__(
        self,
        dim = 2,
        CC = df.Constant(400)
    ):
        self.dim = dim
        self.CC = CC
    
    def get_strain_energy_term(self, F):
        print(self.dim)
        CC = self.CC

        if self.dim == 2:
            e1 = df.as_vector([1.0, 0.0])
        elif self.dim == 3:
            e1 = df.as_vector([1.0, 0.0, 0.0])

        J = df.det(F)
        J_iso = pow(J, -1.0 / float(self.dim))
        C = J_iso ** 2 * F.T * F

        IIFx = df.tr(C)
        W_hat = CC * (IIFx - self.dim) #** 2
        
        return W_hat
