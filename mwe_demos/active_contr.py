"""

This MWE/standalone example demonstrates a contraction for a single cell,
using free Neumann boundary conditions on all of the outer boundary.

This features
- an active contraction example
- how to assign values discretey, both for material parameters and active strain
- how to set up the weak formulation, including all 3 parts displayed in the paper

while ignoring
- stretching setup / fixed boundary conditon options
- all HPC options
- all monitoring options
- all options that are needed to reproduce paper plots

Overall, it might provide a good persemoneous starting point for understanding
the core of our model.

Åshild Telle / Simula Research Laboratory / 2021

"""

import numpy as np
import dolfin as df
import ufl
from mpi4py import MPI

df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["representation"] = "uflacs"
df.parameters["form_compiler"]["quadrature_degree"] = 4


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


def discrete_material_params(fun_space, subdomain_map):
    """

    Defines material parameters based on subdomain partition; instead
    of using two strain energy functions we define each material parameter
    as a discontinous function.

    Args:
        fun_space (df.FunctionSpace): function space in which each
            parameter will live in
        subdomain_map_(df.MeshFunction): corresponding subdomain partition

    """

    a_i = 5.7
    b_i = 11.67
    a_e = 1.52
    b_e = 16.31
    a_if = 19.83
    b_if = 24.72

    a_fun = df.Function(fun_space, name="a")
    assign_discrete_values(a_fun, subdomain_map, a_i, a_e)

    b_fun = df.Function(fun_space, name="b")
    assign_discrete_values(b_fun, subdomain_map, b_i, b_e)

    a_f_fun = df.Function(fun_space, name="a_f")
    assign_discrete_values(a_f_fun, subdomain_map, a_if, 0)

    b_f_fun = df.Function(fun_space, name="b_f")
    assign_discrete_values(b_f_fun, subdomain_map, b_if, 1)

    return {
        "a": a_fun,
        "b": b_fun,
        "a_f": a_f_fun,
        "b_f": b_f_fun,
    }


def psi_holzapfel(
    F,
    mat_params,
):
    """

    Defines the discrete strain energy function as given by assigned
    material parameters; note that the strain energy function is defined
    over both subspaces, and that the two different strain energy
    functions are given by defining each parameter discontinously.

    Args:
        F (ufl form): deformation tensor
        mat_params (dict): material parameters

    Returns:
        psi (ufl form)

    """

    a, b, a_f, b_f, a_s, b_s, a_fs, b_fs = (
        mat_params["a"],
        mat_params["b"],
        mat_params["a_f"],
        mat_params["b_f"],
    )

    cond = lambda a: ufl.conditional(a > 0, a, 0)

    e1 = df.as_vector([1.0, 0.0, 0.0])

    J = df.det(F)
    C = pow(J, -float(2) / 3) * F.T * F

    IIFx = df.tr(C)
    I4e1 = df.inner(C * e1, e1)

    W_hat = a / (2 * b) * (df.exp(b * (IIFx - 3)) - 1)
    W_f = a_f / (2 * b_f) * (df.exp(b_f * cond(I4e1 - 1) ** 2) - 1)

    return W_hat + W_f


def define_weak_form(mesh, active_fun, mat_params):
    """

    Defines function spaces (P1 x P2 x RM) and functions to solve for,
    as well as the weak form for the problem itself.

    Args:
        mesh (df.Mesh): domain to solve equations over
        active_fun (ufl form): active strain function
        mat_params (dict): material parameters

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
    weak_form += elasticity_term(active_fun, F, J, p, v, mat_params)
    weak_form += pressure_term(q, J)
    weak_form += rigid_motion_term(mesh, u, r, state, test_state)

    return weak_form, state, u


def elasticity_term(active_fun, F, J, p, v, mat_params):
    """

    First term of the weak form

    Args:
        active_fun (ufl form): active strain function
        F (ufl form): deformation tensor
        J (ufl form): Jacobian
        p (df.Function): pressure
        v (df.TestFunction): test function for displacement
        mat_params (dict): material parameters

    Returns:
        component of weak form (ufl form)

    """

    sqrt_fun = (1 - active_fun) ** (-0.5)
    F_a = df.as_tensor(((1 - active_fun, 0, 0), (0, sqrt_fun, 0), (0, 0, sqrt_fun)))

    F_e = df.variable(F * df.inv(F_a))
    psi = psi_holzapfel(F_e, mat_params)

    P = df.det(F_a) * df.diff(psi, F_e) * df.inv(F_a.T) + p * J * df.inv(F.T)

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


def rigid_motion_term(mesh, u, r, state, test_state):
    """

    Third term of the weak form

    Args:
        mesh (df.Mesh): domain to solve problem in
        u (df.Function): Displacement
        r (df.TestFunction): Test function for rigid motion space
        state: (u, p, r) – functions to solve for
        test_state: (v, q, s) – corresponding test functions

    Returns:
        component of weak form (ufl form)

    """

    position = df.SpatialCoordinate(mesh)

    RM = [
        df.Constant((1, 0, 0)),
        df.Constant((0, 1, 0)),
        df.Constant((0, 0, 1)),
        df.cross(position, df.Constant((1, 0, 0))),
        df.cross(position, df.Constant((0, 1, 0))),
        df.cross(position, df.Constant((0, 0, 1))),
    ]

    Pi = sum(df.dot(u, zi) * r[i] * df.dx for i, zi in enumerate(RM))

    return df.derivative(Pi, state, test_state)


if __name__ == "__main__":
    mesh, volumes = load_mesh("cell_3D.h5")
    active_strain = np.linspace(0, 0.2, 20)

    U_DG0 = df.FunctionSpace(mesh, "DG", 0)

    mat_params = discrete_material_params(U_DG0, volumes)

    active_fun = df.Function(U_DG0, name="Active strain")
    active_fun.vector()[:] = 0  # initial value

    weak_form, state, u = define_weak_form(mesh, active_fun, mat_params)

    # just for plotting purposes
    disp_file = df.XDMFFile("contraction_example/u_emi.xdmf")
    V_CG2 = df.VectorFunctionSpace(mesh, "CG", 2)
    u_fun = df.Function(V_CG2, name="Displacement")

    for a in active_strain:
        assign_discrete_values(active_fun, volumes, a, 0)  # a in omega_i, 0 in omega_e
        df.solve(weak_form == 0, state)

        # plotting again
        u_fun.assign(df.project(u, V_CG2))

        disp_file.write_checkpoint(u_fun, "Displacement (µm)", a, append=True)

    disp_file.close()
