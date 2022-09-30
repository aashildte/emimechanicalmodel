"""

This MWE/standalone example demonstrates a contraction for a single cell,
using free Neumann boundary conditions on all of the outer boundary.

This features
- a stretching example, for stretching in the fiber direction
- how to assign material parameters discretely, giving two different
  strain energy functions
- how to set up the weak formulation, including the 2 parts relevant for stretching

while ignoring
- active contraction setup + sheet-fiber direction stretch
- all HPC options
- all monitoring options
- all options that are needed to reproduce paper plots

Overall, it might provide a good persemoneous starting point for understanding
the core of our model.

Åshild Telle / Simula Research Laboratory / 2021

"""

import os
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


def define_weak_form(mesh, stretch_fun, mat_params):
    """

    Defines function spaces (P1 x P2 x RM) and functions to solve for,
    as well as the weak form for the problem itself.

    Args:
        mesh (df.Mesh): domain to solve equations over
        stretch_fun (ufl form): function that assigns Dirichlet bcs
                on wall to be stretched/extended
        mat_params (dict): material parameters

    Returns:
        weak form (ufl form), state, displacement, boundary conditions

    """

    P1 = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    P2 = df.VectorElement("Lagrange", mesh.ufl_cell(), 2)

    state_space = df.FunctionSpace(mesh, df.MixedElement([P2, P1]))

    state = df.Function(state_space, name="state")
    test_state = df.TestFunction(state_space)

    u, p = df.split(state)
    v, q = df.split(test_state)

    # Kinematics
    d = len(u)
    I = df.Identity(d)  # Identity tensor
    F = df.variable(I + df.grad(u))  # Deformation gradient
    J = df.det(F)

    # Weak form
    weak_form = 0
    weak_form += elasticity_term(F, J, p, v, mat_params)
    weak_form += pressure_term(q, J)

    V = state_space.split()[1]
    bcs = define_bcs(V, mesh, stretch_fun)

    return weak_form, state, u, bcs


def elasticity_term(F, J, p, v, mat_params):
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

    psi = psi_holzapfel(F, *mat_params)
    P = df.diff(psi, F) + p * J * df.inv(F.T)

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


def define_bcs(V, mesh, stretch_fun):
    """

    Defines boundary conditions based on displacement, assuming the domain
    has a box-like shape. We'll keep the displacement on the sides defined
    by lowest x coord, y coords and z coords fixed in their respective
    planes, while stretching the side defined by the highest x coord.

    Args:
        V (df.VectorFunctionSpace): function space for displacement
        mesh (df.Mesh): Domain in which we solve the problem
        stretch_fun (ufl form): function that assigns Dirichlet bcs
                on wall to be stretched/extended

    Returns:
        List of boundary conditions
    """

    coords = mesh.coordinates()
    xmin = min(coords[:, 0])
    xmax = max(coords[:, 0])
    ymin = min(coords[:, 1])
    zmin = min(coords[:, 2])

    xmin_bnd = f"on_boundary && near(x[0], {xmin})"
    xmax_bnd = f"on_boundary && near(x[0], {xmax})"
    ymin_bnd = f"on_boundary && near(x[1], {ymin})"
    zmin_bnd = f"on_boundary && near(x[2], {zmin})"

    bcs = [
        df.DirichletBC(V.sub(0), 0, xmin_bnd),
        df.DirichletBC(V.sub(1), 0, ymin_bnd),
        df.DirichletBC(V.sub(2), 0, zmin_bnd),
        df.DirichletBC(V.sub(0), stretch_fun, xmax_bnd),
    ]

    return bcs


mesh, volumes = load_mesh("cell_3D.h5")
stretch = np.linspace(0, 0.2, 21)
stretch_fun = df.Constant(0)

U_DG0 = df.FunctionSpace(mesh, "DG", 0)
mat_params = discrete_material_params(U_DG0, volumes)

weak_form, state, u, bcs = define_weak_form(mesh, stretch_fun, mat_params)

# just for plotting purposes
disp_file = df.XDMFFile("stretching_example/u_emi.xdmf")
V_CG2 = df.VectorFunctionSpace(mesh, "CG", 2)
u_fun = df.Function(V_CG2, name="Displacement")

for s in stretch:
    stretch_fun.assign(s)
    df.solve(weak_form == 0, state, bcs=bcs)

    # plotting again
    u_fun.assign(df.project(u, V_CG2))

    disp_file.write_checkpoint(u_fun, "Displacement (µm)", s, append=True)

disp_file.close()
