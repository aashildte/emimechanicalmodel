
import dolfinx as df
from mpi4py import MPI
import numpy as np

from emimechanicalmodel import (
    EMIModel,
)

def test_emi_volume():
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)

    dim = mesh.topology.dim
    N = mesh.topology.index_map(dim).size_local
    indices = np.arange(0, N, 1, dtype=np.int32)
    values = np.zeros(N, dtype=np.int32)
    
    volumes = df.cpp.mesh.MeshTags_int32(mesh, dim, indices, values)

    # just change the array slightly
    
    volumes.values[0] = 1
    volumes.values[0] = 2
    
    model = EMIModel(
        mesh, volumes, experiment="stretch_ff"
    )
    
    for idt in set(volumes.values):
        vol = model.calculate_volume(idt)
        assert vol > 0

if __name__ == "__main__":
    test_emi_volume()
