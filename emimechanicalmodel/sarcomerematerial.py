"""

Åshild Telle / UW / 2024

Material model; the Holzapfel-Odgen model adapted to the EMI framework
including intracellular substructures (sarcomere, z-lines, connections, cytoskeleton).

"""

import numpy as np
import dolfin as df
import ufl

def assign_discrete_values(function, subdomain_map, subdomain_value):
    """

    Assigns function values to a function based on a subdomain map;
    usually just element by element in a DG-0 function.

    Here assuming all subunits have idts in a 1000-range

    Args:
        function (df.Function): function to be changed
        subdomain_map (df.MeshFunction): subdomain division,
            extracellular space expected to have value 0,
            intracellular space expected to have values >= 1
        value_i: to be assigned to omega_i
        value_e: to be assigned to omega_e

    """

    function.vector()[:] = np.where(np.logical_and(subdomain_map >= subdomain_value, subdomain_map < (subdomain_value + 1000)),
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
        subdomains,
        a_i_sarcomeres=df.Constant(2.85),
        a_if_sarcomeres=df.Constant(9.914),
        a_i_zlines=df.Constant(11.4),
        b_i_zlines=df.Constant(11.67),
        a_i_cytoskeleton=df.Constant(0.01),
        a_if_cytoskeleton=df.Constant(0.1),
        a_i_connections=df.Constant(5.7),
        b_i_connections=df.Constant(11.67),
        b=df.Constant(11.67),
        b_f=df.Constant(24.72),
        disabled_connection_fraction=1.0,
        disabled_connection_stiffness=0.0001,

    ):
        self.dim = U.mesh().topology().dim()
        self.subdomain_map = subdomain_map
        self.subdomains = subdomains

        # assign material paramters via characteristic functions
        xi_sarcomeres = df.Function(U)
        xi_zlines = df.Function(U)
        xi_cytoskeleton = df.Function(U)
        xi_connections = df.Function(U)
        
        assign_discrete_values(xi_sarcomeres, subdomain_map, 1000)
        assign_discrete_values(xi_zlines, subdomain_map, 2000)
        assign_discrete_values(xi_cytoskeleton, subdomain_map, 3000)
        assign_discrete_values(xi_connections, subdomain_map, 4000)
        
        a_i_connections_map = self._assign_connection_stiffness(disabled_connection_fraction, a_i_connections, disabled_connection_stiffness)
        a_i_connections_fun = df.Function(U)
        a_i_connections_fun.vector()[:] = a_i_connections_map

        total = xi_sarcomeres.vector()[:] + xi_zlines.vector()[:] + xi_cytoskeleton.vector()[:] + xi_connections.vector()[:]
        assert sum(total) == len(total), "Error: A part of the domain is not assigned material properties."

        a = a_i_sarcomeres * xi_sarcomeres + \
            a_i_zlines * xi_zlines + \
            a_i_cytoskeleton * xi_cytoskeleton + \
            a_i_connections_fun
        
        a_f = a_if_sarcomeres * xi_sarcomeres + \
              a_if_cytoskeleton * xi_cytoskeleton

        b = b * xi_sarcomeres + \
            b_i_zlines * xi_zlines + \
            b * xi_cytoskeleton + \
            b_i_connections * xi_connections

        self.a, self.b, self.a_f, self.b_f = a, b, a_f, b_f

    def _assign_connection_stiffness(self, disabled_connection_fraction, a_i_functional, a_i_dysfunctional):

        subdomain_array = self.subdomain_map[:]
        vector = np.zeros_like(subdomain_array)

        # find highest number sarcomere subdomain
        for i in self.subdomains:
            if 4000 <= i < 5000:
                #vector = np.where(subdomain_array == i, 1.0, vector)
                
                if np.random.random() < disabled_connection_fraction:
                    #np.random.seed(i)
                    #dysf_value = max(0, np.random.normal(a_i_dysfunctional, 0.01))
                    #print("connection stiffness: ", dysf_value)
                    vector = np.where(subdomain_array == i, a_i_dysfunctional, vector)
                else:
                    vector = np.where(subdomain_array == i, a_i_functional, vector)
                
        return vector

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
