"""

Åshild Telle / Simula Research Laboratiry / 2021

"""

import dolfin as df
import ufl

from .holzapfelmaterial import HolzapfelMaterial
from .mesh_setup import assign_discrete_values


class EMIHolzapfelMaterial:
    """

    Adaption of Holzapfel material model to the EMI framework; simply let all material parameters
    be discrete functions, assigned to be zero for anisotropic terms.

    Args:
        fun_space - function space for discrete function; DG-0 is a good choice
        subdomain_map - mapping from volume array to fun_space; for DG-0 this is trivial
        a_i ... b_f - material properties; see paper    

    """
    def __init__(
        self,
        fun_space,
        subdomain_map,
        a_i=0.074,
        b_i=4.878,
        a_e=1,
        b_e=10,
        a_if=4.071,
        b_if=5.433,
    ):
        a_fun = df.Function(fun_space, name="a")
        assign_discrete_values(a_fun, subdomain_map, a_i, a_e)

        b_fun = df.Function(fun_space, name="b")
        assign_discrete_values(b_fun, subdomain_map, b_i, b_e)

        a_f_fun = df.Function(fun_space, name="a_f")
        assign_discrete_values(a_f_fun, subdomain_map, a_if, 0)

        b_f_fun = df.Function(fun_space, name="b_f")
        assign_discrete_values(b_f_fun, subdomain_map, b_if, 1)

        # these are just floats; needed for set functions (below)
        self._a_i, self._a_e, self._b_i, self._b_e, self._a_if, self._b_if = \
                a_i, a_e, b_i, b_e, a_if, b_if

        # these are fenics functions
        self.a_fun, self.b_fun, self.a_f_fun, self.b_f_fun = \
                a_fun, b_fun, a_f_fun, b_f_fun
        
        # and this will be needed if we're updating any of the fenics functions
        self.subdomain_map = subdomain_map

    def passive_component(self, F):
        a, b, a_f, b_f = (
            self.a_fun,
            self.b_fun,
            self.a_f_fun,
            self.b_f_fun,
        )
        e1 = df.as_vector([1.0, 0.0, 0.0])
        e2 = df.as_vector([0.0, 1.0, 0.0])

        J = df.det(F)
        C = pow(J, -float(2) / 3) * F.T * F

        IIFx = df.tr(C)
        I4e1 = df.inner(C * e1, e1)
        I4e2 = df.inner(C * e2, e2)
        I8e1e2 = df.inner(C * e1, e2)

        cond = lambda a: ufl.conditional(a > 0, a, 0)

        W_hat = a / (2 * b) * (df.exp(b * (IIFx - 3)) - 1)
        W_f = a_f / (2 * b_f) * (df.exp(b_f * cond(I4e1 - 1) ** 2) - 1)

        # TODO we don't need these anymore
        self.I1, self.I4e1, self.I4e2, self.I8e1e2 = IIFx, I4e1, I4e2, I8e1e2

        return W_hat + W_f

    def set_a_i(self, a_i):
        a_e = self._a_e

        self._a_i = a_i
        assign_discrete_values(self.a_fun, self.subdomain_map, a_i, a_e)
    
    def set_a_e(self, a_e):
        a_i = self._a_i

        self._a_e = a_e
        assign_discrete_values(self.a_fun, self.subdomain_map, a_i, a_e)
    
    def set_b_i(self, b_i):
        b_e = self._b_e

        self._b_i = b_i
        assign_discrete_values(self.b_fun, self.subdomain_map, b_i, b_e)
    
    def set_b_e(self, b_e):
        b_i = self._b_i

        self._b_e = b_e
        assign_discrete_values(self.b_fun, self.subdomain_map, b_i, b_e)
    
    def set_a_if(self, a_if):
        self._a_if = a_if
        assign_discrete_values(self.a_f_fun, self.subdomain_map, a_if, 0)
    
    def set_b_if(self, b_if):
        self._b_if = b_if
        assign_discrete_values(self.b_f_fun, self.subdomain_map, b_if, 1)
