"""

Åshild Telle / Simula Research Laboratiry / 2021

"""

import dolfin as df

from .holzapfelmaterial import HolzapfelMaterial
from .mesh_setup import assign_discrete_values


class EMIHolzapfelMaterial(HolzapfelMaterial):
    """

    Adaption of Holzapfel material model to the EMI framework; simply let all material parameters
    be discrete functions, assigned to be zero for anisotropic terms.

    Args:
        fun_space - function space for discrete function; DG-0 is a good choice
        subdomain_map - mapping from volume array to fun_space; for DG-0 this is trivial
        a_i ... b_fs - material properties; see paper
    
    Note that, unfortunelately, as per now it is not possible to change parameters by
    assigning them post-init; in that case you need to change it for both subspaces by
    modifying a_fun, b_fun, etc. (self.a_f, self.b_f, ...)

    """
    def __init__(
        self,
        fun_space,
        subdomain_map,
        a_i=df.Constant(0.074),
        b_i=df.Constant(4.878),
        a_e=df.Constant(1),
        b_e=df.Constant(10),
        a_if=df.Constant(4.071),
        b_if=df.Constant(5.433),
        a_is=df.Constant(0.309),
        b_is=df.Constant(2.634),
        a_ifs=df.Constant(0.062),
        b_ifs=df.Constant(3.476),
    ):
        a_fun = df.Function(fun_space, name="a")
        assign_discrete_values(a_fun, subdomain_map, a_i, a_e)

        b_fun = df.Function(fun_space, name="b")
        assign_discrete_values(b_fun, subdomain_map, b_i, b_e)

        a_f_fun = df.Function(fun_space, name="a_f")
        assign_discrete_values(a_f_fun, subdomain_map, a_if, 0)

        b_f_fun = df.Function(fun_space, name="b_f")
        assign_discrete_values(b_f_fun, subdomain_map, b_if, b_if)

        a_s_fun = df.Function(fun_space, name="a_s")
        assign_discrete_values(a_s_fun, subdomain_map, a_is, 0)

        b_s_fun = df.Function(fun_space, name="b_s")
        assign_discrete_values(b_s_fun, subdomain_map, b_is, b_is)

        a_fs_fun = df.Function(fun_space, name="a_fs")
        assign_discrete_values(a_fs_fun, subdomain_map, a_ifs, 0)

        b_fs_fun = df.Function(fun_space, name="b_fs")
        assign_discrete_values(b_fs_fun, subdomain_map, b_ifs, b_ifs)

        super().__init__(
            a_fun,
            b_fun,
            a_f_fun,
            b_f_fun,
            a_s_fun,
            b_s_fun,
            a_fs_fun,
            b_fs_fun,
        )
