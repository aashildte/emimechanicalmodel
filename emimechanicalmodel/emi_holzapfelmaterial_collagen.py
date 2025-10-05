"""

Åshild Telle / University of Washington / 2023

Material model; the Holzapfel-Odgen model adapted to the EMI framework including
an anisotropic component following the assumed collagen alignment in the matrix.

"""
import matplotlib.pyplot as plt

import numpy as np
import dolfin as df

try:
    import ufl
except ModuleNotFoundError:
    import ufl_legacy as ufl


def assign_discrete_values(function, subdomain_map, value_i, value_e):
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

    id_extra = 0
    function.vector()[:] = np.where(
        subdomain_map == id_extra,
        value_e,
        value_i,
    )


class EMIMatrixHolzapfelMaterial:
    """

    Adaption of Holzapfel material model to the EMI framework; simply let all material parameters
    be discrete functions, assigned to be zero for anisotropic terms.

    Args:
        U - function space for discrete function; DG-0 is a good choice
        subdomain_map - mapping from volume array to U; for DG-0 this is trivial
        collagen_dist - df function
        a_i, ..., b_ef - mateiral parameters for intracellular, endomysial, and perimysial subdomains
    """

    def __init__(
        self,
        U,
        subdomain_map,
        collagen_dist,
        a_i=df.Constant(5.70),
        b_i=df.Constant(11.67),
        a_e_endo=df.Constant(5.70),
        b_e_endo=df.Constant(11.67),
        a_e_peri=df.Constant(5.70),
        b_e_peri=df.Constant(11.67),
        a_if=df.Constant(19.83),
        b_if=df.Constant(24.72),
        a_ef_endo=df.Constant(19.83),
        b_ef_endo=df.Constant(24.72),
        a_ef_peri=df.Constant(19.83),
        b_ef_peri=df.Constant(24.72),
    ):
        # these are df.Constants, which can be changed from the outside
        self.a_i, self.a_e, self.b_i, self.b_e, self.a_if, self.b_if, self.a_ef, self.b_ef = (
            a_i,
            a_e_endo,
            b_i,
            b_e_endo,
            a_if,
            b_if,
            a_ef_endo,
            b_ef_endo,
        )

        self.dim = U.mesh().topology().dim()
        
        if self.dim == 2:
            self.e1 = df.as_vector([1.0, 0.0])
            self.e2 = df.as_vector([1.0, 0.0])
        elif self.dim == 3:
            self.e1 = df.as_vector([1.0, 0.0, 0.0])
            self.e2 = df.as_vector([0.0, 1.0, 0.0])

        self.U = U
        self.mesh = U.mesh()
        
        # assign material paramters via characteristic functions
        xi_i = df.Function(U)

        xi_i.vector()[:] = np.where(
            subdomain_map > 2,
            1,
            0,
        )
        
        xi_e_endo = df.Function(U)
        xi_e_endo.vector()[:] = np.where(
            subdomain_map == 1,
            1,
            0,
        )
        
        xi_e_peri = df.Function(U)
        xi_e_peri.vector()[:] = np.where(
            subdomain_map == 2,
            1,
            0,
        )

        volumes_total = df.Function(U)
        volumes_total.vector()[:] += xi_i.vector()[:]
        volumes_total.vector()[:] += xi_e_endo.vector()[:]
        volumes_total.vector()[:] += xi_e_peri.vector()[:]


        df.File("cells.pvd") << xi_i
        df.File("endo.pvd") << xi_e_endo
        df.File("peri.pvd") << xi_e_peri

        max_v = max(volumes_total.vector()[:])
        min_v = min(volumes_total.vector()[:])
        
        assert abs(max_v - 1) < 1E-14 and abs(min_v - 1) < 1E-14, f"Error: not all elements are assigned a tag. Max, min: {max_v}, {min_v}"

        self.collagen_field = self.calculate_collagen_fiber_direction(collagen_dist)

        self.xi_i, self.xi_e_endo, self.xi_e_peri = xi_i, xi_e_endo, xi_e_peri


    def calculate_collagen_fiber_direction(self, theta):
        R = df.as_matrix(
            (
                (df.cos(theta), -df.sin(theta)),
                (df.sin(theta), df.cos(theta)),
            )
        )

        return R*self.e1, R*self.e1


    def get_strain_energy_term(self, F):
        
        a_i, a_e, b_i, b_e, a_if, b_if, a_ef, b_ef = self.a_i, self.a_e, self.b_i, self.b_e, self.a_if, self.b_if, self.a_ef, self.b_ef
        xi_i, xi_e_endo, xi_e_peri = self.xi_i, self.xi_e_endo, self.xi_e_peri
       
        ecm_f, _ = self.collagen_field

        
        J = df.det(F)
        C = J ** 2 * F.T * F
        J_iso = pow(J, -1.0 / float(self.dim))
        C_iso = J_iso ** 2 * F.T * F

        IIFx = df.tr(C_iso)
        I4_myocytes = df.inner(C_iso * self.e1, self.e1)
        I4_matrix = df.inner(C_iso * ecm_f, ecm_f)

        cond = lambda a: ufl.conditional(a > 0, a, 0)

        W_myocytes = a_i*xi_i / (2 * b_i) * (df.exp(b_i * (IIFx - self.dim)) - 1) + \
                     a_if*xi_i / (2 * b_if) * (df.exp(b_if * cond(I4_myocytes - 1) ** 2) - 1)
        
        W_matrix_endo = a_e*xi_e_endo / (2 * b_e) * (df.exp(b_e * (IIFx - self.dim)) - 1) + \
                     a_ef*xi_e_endo / (2 * b_ef) * (df.exp(b_ef * cond(I4_matrix - 1) ** 2) - 1)
        
        W_matrix_peri = a_e*xi_e_peri / (2 * b_e) * (df.exp(b_e * (IIFx - self.dim)) - 1) + \
                        a_ef*xi_e_peri / (2 * b_ef) * (df.exp(b_ef * cond(I4_matrix - 1) ** 2) - 1)

        return W_myocytes + W_matrix_endo + W_matrix_peri
