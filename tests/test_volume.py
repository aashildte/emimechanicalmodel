import dolfin as df
from mpi4py import MPI

from emimechanicalmodel import (
    EMIModel,
)

import dolfin as df
from emimechanicalmodel import EMIModel
from emimechanicalmodel import SarcomereModel


def test_sarcomere_passive_two_subdomains_zero_u():
    mesh = df.UnitSquareMesh(2, 2)
    vols = df.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    for c in df.cells(mesh):
        vols[c] = 1 if c.midpoint().x() < 0.5 else 2

    model = SarcomereModel(
        mesh,
        vols,
        experiment="contraction",
    
    )

    model.update_active_fn(0.0)
    model.solve()

    u = model.state.split()[0]
    
    assert u.vector().norm("linf") < 1e-12

    Vsig = df.TensorFunctionSpace(mesh, "DG", 0)
    sig_proj = df.project(model.sigma, Vsig)
    assert sig_proj.vector().norm("linf") < 1e-12



def test_emi_passive_two_subdomains_zero_u():
    mesh = df.UnitSquareMesh(2, 2)
    vols = df.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    for c in df.cells(mesh):
        vols[c] = 1 if c.midpoint().x() < 0.5 else 2

    model = EMIModel(
        mesh,
        vols,
        experiment="contraction",
    )

    model.update_active_fn(0.0)
    model.solve()

    u = model.state.split()[0]
    assert u.vector().norm("linf") < 1e-12
    
    Vsig = df.TensorFunctionSpace(mesh, "DG", 0)
    sig_proj = df.project(model.sigma, Vsig)
    assert sig_proj.vector().norm("linf") < 1e-12




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
    #test_emi_volume()
    #test_emi_passive_two_subdomains_zero_u()
    test_sarcomere_passive_two_subdomains_zero_u()
