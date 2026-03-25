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

def assign_discrete_values(function, subdomain_map, subdomain_value_min, subdomain_value_max):
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
    #function.vector()[:] = np.where(np.logical_and(subdomain_map >= subdomain_value_min, subdomain_map <= subdomain_value_max),
    #    1,
    #    0,
    #)

    mask = np.logical_and(subdomain_map >= subdomain_value_min, subdomain_map <= subdomain_value_max)
    local = function.vector().get_local().copy()
    local[:] = 0
    local[mask] = 1
    function.vector().set_local(local)
    function.vector().apply("insert")



class EMIHolzapfelMaterial_with_substructures:
    """

    Adaption of Holzapfel material model to the EMI framework; simply let all material parameters
    be discrete functions, assigned to be zero for anisotropic terms.

    Args:
        U - function space for discrete function; DG-0 is a good choice
        subdomain_map - mapping from volume array to U; for DG-0 this is trivial
        a and b values - material properties

    """

    def __init__(
        self,
        U,
        subdomain_map,
        subdomains,
        a_i_sarcomeres=df.Constant(1.0),
        a_if_sarcomeres=df.Constant(5.0),
        a_i_zlines=df.Constant(4.0),
        a_i_cytoskeleton=df.Constant(0.5),
        a_if_cytoskeleton=df.Constant(2.0),
        a_i_connections=df.Constant(4.0),
        a_i_nucleus=df.Constant(1.6),
        a_e=df.Constant(1.0),
        b=df.Constant(5.0),
        b_f=df.Constant(5.0),
    ):
        self.subdomain_map = subdomain_map
        self.subdomains = subdomains
        self.dim = U.mesh().topology().dim()
        
        assert isinstance(subdomain_map, np.ndarray)


        # assign material paramters via characteristic functions
        xi_ECM = df.Function(U)
        xi_sarcomeres = df.Function(U)
        xi_zlines = df.Function(U)
        xi_cytoskeleton = df.Function(U)
        xi_connections = df.Function(U)
        xi_nucleus = df.Function(U)
        
        assign_discrete_values(xi_ECM, subdomain_map, 0, 0)
        assign_discrete_values(xi_sarcomeres, subdomain_map, 1, 999)
        assign_discrete_values(xi_zlines, subdomain_map, 1000, 1000) 
        assign_discrete_values(xi_cytoskeleton, subdomain_map, 2000, 2000)
        assign_discrete_values(xi_connections, subdomain_map, 3000, 3001)
        #assign_discrete_values(xi_nucleus, subdomain_map, 5000, 0)
        assign_discrete_values(xi_nucleus, subdomain_map, 4000, 4000) # or "substrate"
        
        total = xi_sarcomeres.vector()[:] + xi_zlines.vector()[:] + xi_cytoskeleton.vector()[:] + xi_connections.vector()[:] + xi_nucleus.vector()[:] + xi_ECM.vector()[:]         
       
        subdomains = list(set(subdomain_map))
        subdomains.sort()
        print("subdomains: ", subdomains)
        assert sum(total) == len(total), "Error: A part of the domain is not assigned material properties."
        
        a = (
            a_i_sarcomeres    * xi_sarcomeres
          + a_i_zlines        * xi_zlines
          + a_i_cytoskeleton  * xi_cytoskeleton
          + a_i_connections   * xi_connections
          + a_i_nucleus       * xi_nucleus
          + a_e               * xi_ECM
        )

        a_f = (
            a_if_sarcomeres   * xi_sarcomeres
          + a_if_cytoskeleton * xi_cytoskeleton
          + a_if_cytoskeleton * xi_connections
        )

        self.a, self.b, self.a_f, self.b_f = a, b, a_f, b_f

    
    def get_strain_energy_term(self, F, e1=df.as_vector([1.0, 0.0, 0.0])):

        a, b, a_f, b_f = (
            self.a,
            self.b,
            self.a_f,
            self.b_f,
        )

        dim = 3
        J = df.det(F)
        C = F.T * F
        J_iso = J**(-2.0/dim)
        C_iso = J_iso * C

        IIFx = df.tr(C_iso)
        I4e1 = df.inner(C_iso * e1, e1)

        cond = lambda a: ufl.conditional(a > 0, a, 0)

        W_hat = a / (2 * b) * (df.exp(b * (IIFx - self.dim)) - 1)
        W_f = a_f / (2 * b_f) * (df.exp(b_f * cond(I4e1 - 1) ** 2) - 1)

        return W_hat + W_f
