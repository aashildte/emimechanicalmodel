"""

Åshild Telle / Simula Research Laboratory / 2022

"""

from abc import ABC, abstractmethod
from .mesh_setup import assign_discrete_values
import dolfin as df
import numpy as np


try:
    import ufl
except ModuleNotFoundError:
    import ufl_legacy as ufl


class CompressibleMaterial(ABC):
    def __init__(self, kappa=1000):
        self.kappa = kappa 

    def get_strain_energy_term(self, F, p=None):
        """
        Args:
            J - determinant of the deformation tensor
            p - hydrostatic pressure (no purpose here)

        Returns;
            psi_incompressible; contribution to the total strain anergy function

        """
        J = ufl.det(F)
        return self.kappa * (J * df.ln(J) - J + 1)


class IncompressibleMaterial(CompressibleMaterial):
    def __init__(self):
        pass

    def get_strain_energy_term(self, F, p):
        """
        Args:
            J - determinant of the deformation tensor
            p - hydrostatic pressure

        Returns;
            psi_incompressible; contribution to the total strain anergy function

        """
        J = ufl.det(F)
        return p * (J - 1)


class SarcomereNearlyIncompressibleMaterial(CompressibleMaterial):
    def __init__(
            self,
            U,
            subdomain_map,
            kappa_sarcomeres=df.Constant(1000),
            kappa_zlines=df.Constant(1000),
            kappa_connections=df.Constant(1000),
            kappa_cytoskeleton=df.Constant(1000),
            kappa_nucleus=df.Constant(1000),
            ):
    
        # assign material paramters via characteristic functions
        #xi_ECM = df.Function(U)
        xi_sarcomeres = df.Function(U)
        xi_zlines = df.Function(U)
        xi_cytoskeleton = df.Function(U)
        xi_connections = df.Function(U)
        xi_nucleus = df.Function(U)
        
        assign_discrete_values(xi_sarcomeres, subdomain_map, 1, 1999)
        assign_discrete_values(xi_zlines, subdomain_map, 2000, 2000)
        assign_discrete_values(xi_cytoskeleton, subdomain_map, 3000, 3000)
        assign_discrete_values(xi_connections, subdomain_map, 4000, 4000)
        assign_discrete_values(xi_nucleus, subdomain_map, 5000, 5002)

        total = xi_sarcomeres.vector()[:] + xi_zlines.vector()[:] + xi_cytoskeleton.vector()[:] + xi_connections.vector()[:] + xi_nucleus.vector()[:] #+ xi_ECM.vector()[:]  
        subdomains = list(set(subdomain_map))
        subdomains.sort()
        print("subdomains: ", subdomains)
        print(sum(total))
        print(len(total))
        assert sum(total) == len(total), "Error: A part of the domain is not assigned compressibility properties."

        #self.kappa = df.Constant(1000)
        self.kappa = kappa_sarcomeres*xi_sarcomeres + \
                     kappa_zlines*xi_zlines + \
                     kappa_connections*xi_connections + \
                     kappa_cytoskeleton*xi_cytoskeleton + \
                     kappa_nucleus*xi_nucleus



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


class EMINearlyIncompressibleMaterial(CompressibleMaterial):
    def __init__(
        self,
        U,
        subdomain_map,
        kappa_i=df.Constant(10000),
        kappa_e=df.Constant(100),
    ):
        """

        Args:
            U - function space for discrete function; DG-0 is a good choice
            subdomain_map - mapping from volume array to U; for DG-0 this is trivial

            kappa_i - incompressibility parameter for omega i
            kappa_e - incompressibility parameter for omega e

        """
        xi_i = df.Function(U)
        assign_discrete_values(xi_i, subdomain_map, 1, 0)

        xi_e = df.Function(U)
        assign_discrete_values(xi_e, subdomain_map, 0, 1)

        self.kappa = kappa_i * xi_i + kappa_e * xi_e



class NearlyIncompressibleMaterial(CompressibleMaterial):
    """

    This works for tissue-level models as well as for the EMI model
    in cases where kappa_i = kappa_e = a constant.

    """

    def __init__(self, kappa=df.Constant(1000)):
        """
        Args:
            kappa - incompressibility parameter

        """
        self.kappa = kappa
