"""

Åshild Telle / Simula Research Laboratory / 2021

"""

import numpy as np
from mpi4py import MPI

import dolfin as df


def load_mesh(mesh_file: str, verbose=1):
    """

    Loads mesh and submeshes (mesh function) from given mesh file.
    Subdomain visualisation is saved in a pvd / vtu file,

    Args:
        mesh_file - h5 file
        verbose - 0 or 1, print more information for 1

    Returns:
        mesh - dolfin mesh
        volumes - dolfin mesh function, defining the two subdomains

    """

    comm = MPI.COMM_WORLD
    h5_file = df.HDF5File(comm, mesh_file, "r")
    mesh = df.Mesh()
    h5_file.read(mesh, "mesh", False)

    dim = mesh.topology().dim()
    volumes = df.MeshFunction("size_t", mesh, dim, 0)

    # this needs to match whatever the subdomain is called in the mesh file
    if dim == 3:
        h5_file.read(volumes, "volumes")
    else:
        h5_file.read(volumes, "subdomains")

    if verbose > 0:
        print("Mesh and subdomains loaded successfully.")
        print(
            "Number of nodes: %g, number of elements: %g"
            % (mesh.num_vertices(), mesh.num_cells())
        )

    return mesh, volumes

def load_mesh_sarcomere(mesh_file: str, verbose=1):
    """

    Loads mesh and submeshes (mesh function) from given mesh file.
    Subdomain visualisation is saved in a pvd / vtu file,

    Args:
        mesh_file - h5 file
        verbose - 0 or 1, print more information for 1

    Returns:
        mesh - dolfin mesh
        volumes - dolfin mesh function, defining the two subdomains

    """

    comm = MPI.COMM_WORLD
    h5_file = df.HDF5File(comm, mesh_file, "r")
    mesh = df.Mesh()
    h5_file.read(mesh, "mesh", False)

    dim = mesh.topology().dim()
    volumes = df.MeshFunction("size_t", mesh, dim, 0)

    # this needs to match whatever the subdomain is called in the mesh file
    if dim == 3:
        h5_file.read(volumes, "volumes")
    else:
        h5_file.read(volumes, "subdomains")

    if verbose > 0:
        print("Mesh and subdomains loaded successfully.")
        print(
            "Number of nodes: %g, number of elements: %g"
            % (mesh.num_vertices(), mesh.num_cells())
        )

    sarcomere_file = mesh_file.split(".h")[0] + ".npy"
    sarcomere_angles = np.load(sarcomere_file)

    return mesh, volumes, sarcomere_angles


def write_collagen_to_file(mesh_file):
    comm = MPI.COMM_WORLD
    h5_file = df.HDF5File(comm, mesh_file, "r")
    mesh = df.Mesh()
    h5_file.read(mesh, "mesh", False)

    collagen_dist = df.MeshFunction("double", mesh, 0)
    
    # this needs to match whatever the subdomain is called in the mesh file
    h5_file.read(collagen_dist, "collagen_dist")

    V = df.FunctionSpace(mesh, "CG", 1)
    theta = df.Function(V)
    theta.vector()[:] = collagen_dist.array()[:]

    name = mesh_file.split(".")[0]
    fid = df.HDF5File(comm, f"{name}_collagen.h5", "w")
    fid.write(theta, "collagen_dist")
    fid.close()
    print("Done!")


def load_mesh_with_collagen_structure(mesh_file, verbose=1):
    """

    Loads mesh and submeshes (mesh function) from given mesh file.
    Subdomain visualisation is saved in a pvd / vtu file,

    Args:
        mesh_file - h5 file
        verbose - 0 or 1, print more information for 1

    Returns:
        mesh - dolfin mesh
        volumes - dolfin mesh function, defining the two subdomains

    """

    comm = MPI.COMM_WORLD
    h5_file = df.HDF5File(comm, mesh_file, "r")
    mesh = df.Mesh()
    h5_file.read(mesh, "mesh", False)

    dim = mesh.topology().dim()
    volumes = df.MeshFunction("size_t", mesh, dim, 0)
    #collagen_dist = df.MeshFunction("double", mesh, 0)
        
    # this needs to match whatever the subdomain is called in the mesh file
    if dim == 3:
        raise NotImplementedError("TODO")
    h5_file.read(volumes, "subdomains")
    #h5_file.read(collagen_dist, "collagen_dist")
    """

    V = df.FunctionSpace(mesh, "CG", 1)
    theta = df.Function(V)

        # in parallel 
    rank = comm.Get_rank()
    values = []                           
    visited = []                                                     

    v = theta.vector()
    dofmap = V.dofmap()
    node_min, node_max = v.local_range()                                            
   
    values = theta.vector()[:]

    for cell in df.cells(mesh):                                                        
        dofs = dofmap.cell_dofs(cell.index())
        for dof in dofs:
            global_dof = dofmap.local_to_global_index(dof)
            if dof not in visited and node_min <= global_dof <= node_max:
                #print(dof)
                values[dof] = collagen_dist[dof]
                visited.append(dof)

    theta.vector().set_local(values)
    #df.File("collagen_distribution.pvd") << theta
    """
    V = df.FunctionSpace(mesh, "CG", 1)
    theta = df.Function(V)

    name = mesh_file.split(".")[0]
    fid = df.HDF5File(comm, f"{name}_collagen.h5", "r")
    fid.read(theta, "collagen_dist")
    fid.close()

    if verbose > 0:
        print("Mesh and subdomains loaded successfully.")
        print(
            "Number of nodes: %g, number of elements: %g"
            % (mesh.num_vertices(), mesh.num_cells())
        )

    return mesh, volumes, theta


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
