"""

Åshild Telle / University of Washington / 2023

Material model; the Holzapfel-Odgen model adapted to the EMI framework including
an anisotropic component following the assumed collagen alignment in the matrix.

"""
import matplotlib.pyplot as plt

import numpy as np
import dolfin as df
import ufl

from .mesh_setup import assign_discrete_values


class EMIMatrixHolzapfelMaterial:
    """

    Adaption of Holzapfel material model to the EMI framework; simply let all material parameters
    be discrete functions, assigned to be zero for anisotropic terms.

    Args:
        U - function space for discrete function; DG-0 is a good choice
        subdomain_map - mapping from volume array to U; for DG-0 this is trivial

    """

    def __init__(
        self,
        U,
        subdomain_map,
        collagen_dist,
        a_i=df.Constant(5.70),
        b_i=df.Constant(11.67),
        a_e=df.Constant(5.70),
        b_e=df.Constant(11.67),
        a_if=df.Constant(19.83),
        b_if=df.Constant(24.72),
        a_ef=df.Constant(19.83),
        b_ef=df.Constant(24.72),
    ):
        # these are df.Constants, which can be changed from the outside
        self.a_i, self.a_e, self.b_i, self.b_e, self.a_if, self.b_if, self.a_ef, self.b_ef = (
            a_i,
            a_e,
            b_i,
            b_e,
            a_if,
            b_if,
            a_ef,
            b_ef,
        )

        self.dim = U.mesh().topology().dim()
        
        if self.dim == 2:
            self.e1 = df.as_vector([1.0, 0.0])
        elif self.dim == 3:
            self.e1 = df.as_vector([1.0, 0.0, 0.0])

        self.U = U
        #self.V = V
        self.mesh = U.mesh()

        # assign material paramters via characteristic functions
        xi_i = df.Function(U)
        assign_discrete_values(xi_i, subdomain_map, 1, 0)

        xi_e = df.Function(U)
        assign_discrete_values(xi_e, subdomain_map, 0, 1)

        self.collagen_field = self.calculate_collagen_fiber_direction(collagen_dist)

        V = df.VectorFunctionSpace(U.mesh(), "CG", 2)
        output_folder = "conttraction_experiments_spatial"
        df.File(f"{output_folder}/fiber_direction.pvd") << df.project(self.collagen_field, V)

        self.xi_i, self.xi_e = xi_i, xi_e

    def calculate_collagen_fiber_direction(self, theta):
        R = df.as_matrix(
            (
                (df.cos(theta), -df.sin(theta)),
                (df.sin(theta), df.cos(theta)),
            )
        )

        return R*self.e1


    def get_strain_energy_term(self, F):
        
        a_i, a_e, b_i, b_e, a_if, b_if, a_ef, b_ef = self.a_i, self.a_e, self.b_i, self.b_e, self.a_if, self.b_if, self.a_ef, self.b_ef
        xi_i, xi_e = self.xi_i, self.xi_e
       
        ecm_f = self.collagen_field

        
        J = df.det(F)
        C = J ** 2 * F.T * F
        J_iso = pow(J, -1.0 / float(self.dim))
        C_iso = J_iso ** 2 * F.T * F

        IIFx = df.tr(C)
        I4_myocytes = df.inner(C_iso * self.e1, self.e1)
        I4_matrix = df.inner(C_iso * ecm_f, ecm_f)

        cond = lambda a: ufl.conditional(a > 0, a, 0)

        W_myocytes = a_i*xi_i / (2 * b_i) * (df.exp(b_i * (IIFx - self.dim)) - 1) + \
                     a_if*xi_i / (2 * b_if) * (df.exp(b_if * cond(I4_myocytes - 1) ** 2) - 1)
        
        W_matrix = a_e*xi_e / (2 * b_e) * (df.exp(b_e * (IIFx - self.dim)) - 1) + \
                     a_ef*xi_e / (2 * b_ef) * (df.exp(b_ef * cond(I4_matrix - 1) ** 2) - 1)


        return W_myocytes + W_matrix
