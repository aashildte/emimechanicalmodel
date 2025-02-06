"""

Åshild Telle / Simula Research Laboratory / 2022

"""

from abc import ABC, abstractmethod
from .mesh_setup import assign_discrete_values
import dolfin as df

try:
    import ufl
except ModuleNotFoundError:
    import ufl_legacy as ufl

class CompressibleMaterial(ABC):
    @abstractmethod
    def get_strain_energy_term(J, p):
        pass


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
