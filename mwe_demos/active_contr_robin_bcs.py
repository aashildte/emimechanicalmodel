"""

Åshild Telle / UW / 2024

"""

import numpy as np
import dolfin as df
import ufl
from mpi4py import MPI

df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["representation"] = "uflacs"
df.parameters["form_compiler"]["quadrature_degree"] = 4
df.parameters["ghost_mode"] = "shared_facet"

def load_mesh(mesh_file):
    """

    Loads mesh and subdomains from h5 file.

    Args:
        mesh_file (str): path to h5 file

    Returns:
        mesh (df.Mesh): Mesh to be used in simulation
        volumes (df.MeshFunction): Subdomain partition

    """

    comm = MPI.COMM_WORLD
    h5_file = df.HDF5File(comm, mesh_file, "r")
    mesh = df.Mesh()
    h5_file.read(mesh, "mesh", False)

    volumes = df.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    h5_file.read(volumes, "volumes")

    print(
        "Number of nodes: %g, number of elements: %g"
        % (mesh.num_vertices(), mesh.num_cells())
    )

    return mesh, volumes


def assign_discrete_values(
    function, subdomain_map, intracellular_value, extracellular_value
):
    """

    Assigns function values to a function based on a subdomain map; usually
    just element by element in a DG-0 function.

    Args:
        function (df.Function): function to be changed
        subdomain_map (df.MeshFunction): subdomain division,
            extracellular space expected to have value 0,
            intracellular space expected to have values >= 1
        value1: to be assigned to omega_i
        value2: to be assigned to omega_e

    """

    extracellular_domain_id = 0

    function.vector()[:] = np.where(
        subdomain_map.array()[:] == extracellular_domain_id,
        extracellular_value,
        intracellular_value,
    )


def psi_holzapfel(
    F,
    a = 5.7,
    b=11.67,
    a_f=19.83,
    b_f=24.72
):
    """

    Defines the discrete strain energy function as given by assigned
    material parameters.

    Args:
        F (ufl form): deformation tensor
        a, b, a_f, b_f - material parameters

    Returns:
        psi (ufl form)

    """
    cond = lambda a: ufl.conditional(a > 0, a, 0)

    e1 = df.as_vector([1.0, 0.0, 0.0])

    J = df.det(F)
    C = pow(J, -float(2) / 3) * F.T * F

    IIFx = df.tr(C)
    I4e1 = df.inner(C * e1, e1)

    W_hat = a / (2 * b) * (df.exp(b * (IIFx - 3)) - 1)
    W_f = a_f / (2 * b_f) * (df.exp(b_f * cond(I4e1 - 1) ** 2) - 1)

    return W_hat + W_f


def define_weak_form(mesh, active_fun):
    """

    Defines function spaces (P1 x P2 x RM) and functions to solve for,
    as well as the weak form for the problem itself.

    Args:
        mesh (df.Mesh): domain to solve equations over
        active_fun (ufl form): active strain function

    Returns:
        weak form (ufl form), state, displacement

    """

    P1 = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    P2 = df.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    R = df.VectorElement("Real", mesh.ufl_cell(), 0, 6)

    state_space = df.FunctionSpace(mesh, df.MixedElement([P2, P1, R]))

    state = df.Function(state_space, name="state")
    test_state = df.TestFunction(state_space)

    u, p, r = df.split(state)
    v, q, _ = df.split(test_state)

    # Kinematics
    d = len(u)
    I = df.Identity(d)  # Identity tensor
    F = df.variable(I + df.grad(u))  # Deformation gradient
    J = df.det(F)

    # Weak form
    weak_form = 0
    weak_form += elasticity_term(active_fun, F, J, p, v)
    weak_form += pressure_term(q, J)
    weak_form += robin_bnd_cond_term(u, v) 

    return weak_form, state, u


def elasticity_term(active_fun, F, J, p, v):
    """

    First term of the weak form

    Args:
        active_fun (ufl form): active strain function
        F (ufl form): deformation tensor
        J (ufl form): Jacobian
        p (df.Function): pressure
        v (df.TestFunction): test function for displacement

    Returns:
        component of weak form (ufl form)

    """

    sqrt_fun = (1 - active_fun) ** (-0.5)
    F_a = df.as_tensor(((1 - active_fun, 0, 0), (0, sqrt_fun, 0), (0, 0, sqrt_fun)))

    F_e = df.variable(F * df.inv(F_a))
    psi = psi_holzapfel(F_e) + p * (df.det(F_e) - 1)

    P = df.det(F_a) * df.diff(psi, F_e) * df.inv(F_a.T)

    return df.inner(P, df.grad(v)) * df.dx


def pressure_term(q, J):
    """

    Second term of the weak form

    Args:
        q (df.TestFunction): test function for pressure
        J (ufl form): Jacobian

    Returns:
        component of weak form (ufl form)

    """
    return q * (J - 1) * df.dx


def robin_bnd_cond_term(u, v):
    robin_value = df.Constant(1)
    return df.inner(robin_value * u, v) * df.ds


if __name__ == "__main__":
    mesh, volumes = load_mesh("cell_3D.h5")
    active_strain = np.linspace(0, 0.02, 2)

    U_DG0 = df.FunctionSpace(mesh, "DG", 0)

    active_fun = df.Function(U_DG0, name="Active strain")
    active_fun.vector()[:] = 0  # initial value

    weak_form, state, u = define_weak_form(mesh, active_fun)

    for a in active_strain:
        assign_discrete_values(active_fun, volumes, a, 0)  # a in omega_i, 0 in omega_e
        df.solve(weak_form == 0, state)

