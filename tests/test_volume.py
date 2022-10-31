import dolfin as df
from mpi4py import MPI

from emimechanicalmodel import (
    EMIModel,
)


def test_emi_volume():
    mesh = df.UnitCubeMesh(1, 1, 1)
    volumes = df.MeshFunction("size_t", mesh, 3)

    # just change the array slightly

    volumes.array()[0] = 1
    volumes.array()[1] = 2

    model = EMIModel(mesh, volumes, experiment="contraction")

    for idt in set(volumes.array()):
        vol = model.calculate_volume(idt)
        assert vol > 0


if __name__ == "__main__":
    test_emi_volume()
