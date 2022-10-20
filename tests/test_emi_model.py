import os
import pytest
import dolfinx as df
from mpi4py import MPI
import numpy as np

from emimechanicalmodel import EMIModel

def test_emi_active():
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    
    U = df.FunctionSpace(mesh, ("DG", 0))
    volumes = df.Function(U)
    volumes.vector.array[0] = 1

    model = EMIModel(
        mesh, volumes, experiment="contr"
    )

    active_value = 0.001
    
    model.update_active_fn(active_value)
    model.solve(project=False)

    assert abs(np.max(model.active_fn.vector()[:]) - active_value) < 1E-10
    assert abs(np.min(model.active_fn.vector()[:]) - 0) < 1E-10


def test_emi_proj_strain():
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    
    U = df.FunctionSpace(mesh, ("DG", 0))
    volumes = df.Function(U)
    volumes.vector.array[0] = 1

    model = EMIModel(
        mesh, volumes, experiment="contr"
    )

    active_value = 0.001
    
    model.update_active_fn(active_value)
    model.solve(project=True)

    assert model.evaluate_subdomain_strain_fibre_dir(1) < 0


def test_emi_proj_stress():
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    
    U = df.FunctionSpace(mesh, ("DG", 0))
    volumes = df.Function(U)
    volumes.vector.array[0] = 1

    model = EMIModel(
        mesh, volumes, experiment="contr"
    )

    active_value = 0.001

    assert model.evaluate_subdomain_stress_fibre_dir(1) > 0


@pytest.mark.parametrize(
    ("deformation_mode"),
    [
        ("stretch_ff"),
        ("stretch_ss"),
        ("stretch_nn"),
        ("shear_fs"),
        ("shear_sf"),
        ("shear_nf"),
        ("shear_fn"),
        ("shear_sn"),
        ("shear_ns"),
    ],
)
def test_emi_deformation(deformation_mode):
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    
    U = df.FunctionSpace(mesh, ("DG", 0))
    volumes = df.Function(U)
    volumes.vector.array[0] = 1
    
    model = EMIModel(
        mesh, volumes, experiment=deformation_mode,
    )

    stretch_value = 0.05
    model.assign_stretch(stretch_value)
    model.solve()

    if "stretch" in deformation_mode:
        assert(model.evaluate_normal_load() > 0)
    else:
        assert(model.evaluate_shear_load() > 0)

if __name__ == "__main__":
    test_emi_active()
    test_emi_proj_strain()
    test_emi_proj_stress()
    test_emi_deformation("shear_fs")
