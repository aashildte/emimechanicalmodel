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
    if dim==3:
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
