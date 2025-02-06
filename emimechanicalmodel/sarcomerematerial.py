"""

Åshild Telle / UW / 2024

Material model; the Holzapfel-Odgen model adapted to the EMI framework
including intracellular substructures (sarcomere, z-lines, connections, cytoskeleton).

"""

import numpy as np
import dolfin as df

try:
    import ufl
except ModuleNotFoundError:
    import ufl_legacy as ufl

def assign_discrete_values(function, subdomain_map, subdomain_value):
    """

    Assigns function values to a function based on a subdomain map;
    usually just element by element in a DG-0 function.

    Args:
        function (df.Function): function to be changed
        subdomain_map (df.MeshFunction): subdomain division,
            extracellular space expected to have value 0,
            intracellular space expected to have values >= 1
        value_i: to be assigned to omega_i
        value_e: to be assigned to omega_e

    Note that all cells are assigned the same value homogeneously.

    """

    function.vector()[:] = np.where(
        subdomain_map == subdomain_value,
        1,
        0,
    )


class EMIHolzapfelMaterial_with_substructures:
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
        a_i_sarcomeres=df.Constant(2.85),
        a_if_sarcomeres=df.Constant(9.914),
        a_i_zlines=df.Constant(11.4),
        a_i_cytoskeleton=df.Constant(5.70),
        a_if_cytoskeleton=df.Constant(19.83),
        a_i_connections=df.Constant(5.7),
        b=df.Constant(11.67),
        b_f=df.Constant(24.72),
    ):
        self.dim = U.mesh().topology().dim()

        # assign material paramters via characteristic functions
        xi_sarcomeres = df.Function(U)
        xi_zlines = df.Function(U)
        xi_cytoskeleton = df.Function(U)
        xi_connections = df.Function(U)
        
        assign_discrete_values(xi_sarcomeres, subdomain_map, 1)
        assign_discrete_values(xi_zlines, subdomain_map, 2)
        assign_discrete_values(xi_cytoskeleton, subdomain_map, 3)
        assign_discrete_values(xi_connections, subdomain_map, 4)

        a = a_i_sarcomeres * xi_sarcomeres + \
            a_i_zlines * xi_zlines + \
            a_i_cytoskeleton * xi_cytoskeleton + \
            a_i_connections * xi_connections
        
        a_f = a_if_sarcomeres * xi_sarcomeres + \
              a_if_cytoskeleton * xi_cytoskeleton

        self.a, self.b, self.a_f, self.b_f = a, b, a_f, b_f

    def get_strain_energy_term(self, F):

        a, b, a_f, b_f = (
            self.a,
            self.b,
            self.a_f,
            self.b_f,
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

        #cond = lambda a: ufl.conditional(a > 0, a, 0)

        W_hat = a / (2 * b) * (df.exp(b * (IIFx - self.dim)) - 1)
        W_f = a_f / (2 * b_f) * (df.exp(b_f * (I4e1 - 1) ** 2) - 1)

        return W_hat + W_f
