import os
import dolfin as df
from mpi4py import MPI
import numpy as np

from emimechanicalmodel import EMIModel

def test_emi_active():
    mesh = df.UnitCubeMesh(1, 1, 1)
    volumes = df.MeshFunction('size_t', mesh, 3)
    volumes.array()[0] = 1

    model = EMIModel(
        mesh, volumes, experiment="contr"
    )

    active_value = 0.001
    
    model.update_active_fn(active_value)
    model.solve(project=False)

    assert abs(np.max(model.active_fn.vector()[:]) - active_value) < 1E-10
    assert abs(np.min(model.active_fn.vector()[:]) - 0) < 1E-10


def test_emi_proj_strain():
    mesh = df.UnitCubeMesh(1, 1, 1)
    volumes = df.MeshFunction('size_t', mesh, 3)
    volumes.array()[0] = 1

    model = EMIModel(
        mesh, volumes, experiment="contr"
    )

    active_value = 0.001
    
    model.update_active_fn(active_value)
    model.solve(project=True)

    assert model.evaluate_subdomain_strain_fibre_dir(1) < 0


def test_emi_proj_stress():
    mesh = df.UnitCubeMesh(1, 1, 1)
    volumes = df.MeshFunction('size_t', mesh, 3)
    volumes.array()[0] = 1

    model = EMIModel(
        mesh, volumes, experiment="contr"
    )

    active_value = 0.001

    assert model.evaluate_subdomain_stress_fibre_dir(1) > 0


def test_emi_xstretch():
    mesh = df.UnitCubeMesh(1, 1, 1)
    volumes = df.MeshFunction('size_t', mesh, 3)
    volumes.array()[0] = 1
    
    model = EMIModel(
        mesh, volumes, experiment="xstretch"
    )

    stretch_value = 0.05
    model.assign_stretch(stretch_value)
    model.solve()
    
    assert(model.evaluate_load_yz() > 0)

def test_emi_ystretch():
    mesh = df.UnitCubeMesh(3, 3, 3)
    volumes = df.MeshFunction('size_t', mesh, 3)
    volumes.array()[0] = 1
    
    model = EMIModel(
        mesh, volumes, experiment="ystretch"
    )
    
    stretch_value = 0.05
    model.assign_stretch(stretch_value)
    model.solve()
    
    assert(model.evaluate_load_xz() > 0)


if __name__ == "__main__":
    test_emi_active()
    test_emi_xstretch()
    test_emi_ystretch()
    test_emi_proj_strain()
    test_emi_proj_stress()
