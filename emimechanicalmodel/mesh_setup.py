"""

Åshild Telle / Simula Research Laboratory / 2021

"""

import numpy as np
from mpi4py import MPI

import dolfinx as df


def load_mesh(mesh_file: str, verbose=1):
    """

    Loads mesh and submeshes (mesh function) from given mesh file.
    Subdomain visualisation is saved in a pvd / vtu file,

    Args:
        mesh_file - h5 file
        path_vtu_file - path, where to save subdomain file;
            default None = no output subdomain file will be saved

    Returns:
        mesh - dolfin mesh
        volumes - dolfin mesh function, defining the two subdomains

    """

    comm = MPI.COMM_WORLD
    encoding = df.io.XDMFFile.Encoding.HDF5
    
    with df.io.XDMFFile(comm, mesh_file, "r") as xdmf_file:
        mesh = xdmf_file.read_mesh(name="mesh")
        volumes = xdmf_file.read_meshtags(mesh, name="mesh")

    if verbose > 0:
        num_vertices = mesh.geometry.index_map().size_global
        num_cells = mesh.topology.index_map(3).size_global
        print(f"Number of nodes: {num_vertices}, number of elements: {num_cells}")

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

    function.vector[:] = np.where(
        subdomain_map == id_extra,
        value_e,
        value_i,
    )
