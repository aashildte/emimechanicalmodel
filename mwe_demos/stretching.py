"""

Åshild Telle / Simula Research Laboratory / 2022

"""

import numpy as np
import dolfinx as df
import ufl
from mpi4py import MPI
from petsc4py import PETSc


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

    with df.io.XDMFFile(comm, mesh_file, "r") as xdmf_file:
        mesh = xdmf_file.read_mesh(name="mesh")
        volumes = xdmf_file.read_meshtags(mesh, name="mesh")
    
    num_vertices = mesh.geometry.index_map().size_global
    num_cells = mesh.topology.index_map(3).size_global
    print(f"Number of nodes: {num_vertices}, number of elements: {num_cells}")

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

    function.vector.array[:] = np.where(
        subdomain_map.values[:] == extracellular_domain_id,
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

    a_fun = df.fem.Function(fun_space, name="a")
    assign_discrete_values(a_fun, subdomain_map, a_i, a_e)


    b_fun = df.fem.Function(fun_space, name="b")
    assign_discrete_values(b_fun, subdomain_map, b_i, b_e)

    a_f_fun = df.fem.Function(fun_space, name="a_f")
    assign_discrete_values(a_f_fun, subdomain_map, a_if, 0)

    b_f_fun = df.fem.Function(fun_space, name="b_f")
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

    a, b, a_f, b_f = (
        mat_params["a"],
        mat_params["b"],
        mat_params["a_f"],
        mat_params["b_f"],
    )


    cond = lambda a: ufl.conditional(a > 0, a, 0)

    e1 = ufl.as_vector([1.0, 0.0, 0.0])

    J = ufl.det(F)
    C = pow(J, -float(2) / 3) * F.T * F

    IIFx = ufl.tr(C)
    I4e1 = ufl.inner(C * e1, e1)

    W_hat = a / (2 * b) * (ufl.exp(b * (IIFx - 3)) - 1)
    W_f = a_f / (2 * b_f) * (ufl.exp(b_f * cond(I4e1 - 1) ** 2) - 1)

    return W_hat + W_f


def define_weak_form(mesh, mat_params):
    """

    Defines function spaces (P1 x P2) and functions to solve for, as well
    as the weak form for the problem itself. This assumes a fully incompressible
    formulation, solving for the displacement and the hydrostatic pressure.

    Args:
        mesh (df.Mesh): domain to solve equations over
        mat_params (dict): material parameters

    Returns:
        weak form (ufl form), state, displacement, boundary conditions
        stretch_fun (ufl form): function that assigns Dirichlet bcs
                on wall to be stretched/extended

    """
    
    
    P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)

    state_space = df.fem.FunctionSpace(mesh, P2 * P1)
    
    state = df.fem.Function(state_space)
    test_state = ufl.TestFunctions(state_space)

    u, p = ufl.split(state)
    v, q = test_state
    
    # Kinematics
    d = len(u)
    I = ufl.Identity(d)                # Identity tensor
    F = ufl.variable(I + ufl.grad(u))  # Deformation gradient
    J = ufl.det(F)
    
    # Weak form
    
    metadata = {"quadrature_degree": 4}
    dx = ufl.Measure("dx", domain=mesh, metadata=metadata)
    
    weak_form = 0
    weak_form += elasticity_term(F, J, p, v, mat_params, dx)
    weak_form += pressure_term(q, J, dx) 
    
    bcs, stretch_fun = define_bcs(state_space, mesh)
    
    return weak_form, state, bcs, stretch_fun


def elasticity_term(F, J, p, v, mat_params, dx):
    """

    First term of the weak form

    Args:
        F (ufl form): deformation tensor
        J (ufl form): Jacobian
        p (df.Function): pressure
        v (df.TestFunction): test function for displacement
        mat_params (dict): material parameters
        

    Returns:
        component of weak form (ufl form)

    """

    psi = psi_holzapfel(F, mat_params)
    P = ufl.diff(psi, F) + p * J * ufl.inv(F.T)
    
    return ufl.inner(P, ufl.grad(v)) * dx


def pressure_term(q, J, dx):
    """

    Second term of the weak form

    Args:
        q (df.TestFunction): test function for pressure
        J (ufl form): Jacobian

    Returns:
        component of weak form (ufl form)

    """
    return q * (J - 1) * dx


def define_bcs(state_space, mesh):
    """

    Defines boundary conditions based on displacement, assuming the domain
    has a box-like shape. We'll keep the displacement on the sides defined
    by lowest x coord, y coords and z coords fixed in their respective
    planes, while stretching the side defined by the highest x coord.

    Args:
        state_space (FunctionSpace): function space for displacement and pressure
        mesh (df.Mesh): Domain in which we solve the problem

    Returns:
        List of boundary conditions
        stretch_fun (ufl form): function that assigns Dirichlet bcs
                on wall to be stretched/extended
    """
    

    coords = mesh.geometry.x
    xmin = min(coords[:, 0])
    xmax = max(coords[:, 0])
    ymin = min(coords[:, 1])
    zmin = min(coords[:, 2])

    xmin_bnd = lambda x : np.isclose(x[0], xmin)
    xmax_bnd = lambda x : np.isclose(x[0], xmax)
    ymin_bnd = lambda x : np.isclose(x[1], ymin)
    zmin_bnd = lambda x : np.isclose(x[2], zmin)

    fdim = 2 
    bcs = []
    
    # first define the fixed boundaries
    
    bnd_funs = [xmin_bnd, ymin_bnd, zmin_bnd]
    components = [0, 1, 2]

    V0, _ = state_space.sub(0).collapse()

    for bnd_fun, comp in zip(bnd_funs, components):
        V_c, _ = V0.sub(comp).collapse()
        u_fixed = df.fem.Function(V_c)
        u_fixed.vector.array[:] = 0
        dofs = df.fem.locate_dofs_geometrical((state_space.sub(0).sub(comp),V_c), bnd_fun)
        bc = df.fem.dirichletbc(u_fixed, dofs, state_space.sub(0).sub(comp))
        bcs.append(bc)
    
    # then the moving one
    V0, _ = state_space.sub(0).collapse()
    V0x, _ = V0.sub(0).collapse()
    
    stretch_fun = df.fem.Function(V0x)
    stretch_fun.vector.array[:] = 0

    boundary_facets = df.mesh.locate_entities_boundary(mesh, fdim, xmax_bnd)
    dofs = df.fem.locate_dofs_topological((state_space.sub(0).sub(0), V0x), fdim, boundary_facets)
    
    bc = df.fem.dirichletbc(stretch_fun, dofs, state_space.sub(0).sub(0))
    bcs.append(bc)

    return bcs, stretch_fun 



mesh, volumes = load_mesh("single_cell.xdmf")

U_DG0 = df.fem.FunctionSpace(mesh, ("DG", 0))
mat_params = discrete_material_params(U_DG0, volumes)

weak_form, state, bcs, stretch_fun = define_weak_form(mesh, mat_params)


problem = df.fem.petsc.NonlinearProblem(weak_form, state, bcs)
solver = df.nls.petsc.NewtonSolver(mesh.comm, problem)

solver.rtol=1e-4
solver.atol=1e-4
solver.convergence_criterium = "incremental"

stretch_values = np.linspace(0, 0.1, 10)

fout = df.io.XDMFFile(mesh.comm, "displacement.xdmf", "w")
fout.write_mesh(mesh)

for s in stretch_values:
    print(f"Domain stretch: {100*s:.5f} %")
    stretch_fun.vector.array[:] = s
    solver.solve(state)
    u, _ = state.split()

    fout.write_function(u, s)

fout.close()
