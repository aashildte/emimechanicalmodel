import os
import pytest
import dolfin as df
from mpi4py import MPI
import numpy as np

from emimechanicalmodel import EMIModel


def test_emi_active_strain():
    mesh = df.UnitCubeMesh(1, 1, 1)
    volumes = df.MeshFunction("size_t", mesh, 3)
    volumes.array()[0] = 1

    model = EMIModel(mesh, volumes, experiment="contraction")

    active_value = 0.001

    model.update_active_fn(active_value)
    model.solve(project=False)

    u, _, _ = model.state.split(deepcopy=True)

    U = df.FunctionSpace(mesh, "DG", 0)
    active = df.project(model.active_fn, U)

    assert abs(np.max(active.vector()[:]) - active_value) < 1e-10
    assert abs(np.min(active.vector()[:]) - 0) < 1e-10
    assert np.linalg.norm(u.vector()[:]) > 0


def test_emi_active_stress():
    mesh = df.UnitCubeMesh(1, 1, 1)
    volumes = df.MeshFunction("size_t", mesh, 3)
    volumes.array()[0] = 1

    model = EMIModel(
        mesh, volumes, active_model="active_stress", experiment="contraction"
    )

    active_value = 0.001

    model.update_active_fn(active_value)
    model.solve(project=False)

    u, _, _ = model.state.split(deepcopy=True)

    U = df.FunctionSpace(mesh, "DG", 0)
    active = df.project(model.active_fn, U)
   
    assert abs(np.max(active.vector()[:]) - active_value) < 1e-10
    assert abs(np.min(active.vector()[:]) - 0) < 1e-10
    assert np.linalg.norm(u.vector()[:]) > 0


def test_emi_proj_strain():
    mesh = df.UnitCubeMesh(1, 1, 1)
    volumes = df.MeshFunction("size_t", mesh, 3)
    volumes.array()[0] = 1

    model = EMIModel(mesh, volumes, experiment="contraction")

    active_value = 0.001

    model.update_active_fn(active_value)
    model.solve(project=True)

    assert model.evaluate_subdomain_strain_fibre_dir(1) < 0


def test_emi_proj_stress():
    mesh = df.UnitCubeMesh(1, 1, 1)
    volumes = df.MeshFunction("size_t", mesh, 3)
    volumes.array()[0] = 1

    model = EMIModel(mesh, volumes, experiment="contraction")

    active_value = 0.001

    assert model.evaluate_subdomain_stress_fibre_dir(1) > 0


@pytest.mark.parametrize(
    ("deformation_mode"),
    [
        ("stretch_ff"),
        ("stretch_ss"),
        ("shear_fs"),
        ("shear_sf"),
    ],
)
def test_emi_deformation_2D(deformation_mode):
    mesh = df.UnitSquareMesh(1, 1)
    volumes = df.MeshFunction("size_t", mesh, 2)
    volumes.array()[0] = 1

    model = EMIModel(mesh, volumes, experiment=deformation_mode)
    
    stretch_value = 0.01
    model.assign_stretch(stretch_value)
    model.solve()
    
    if "stretch" in deformation_mode:
        assert model.evaluate_normal_load() > 0
    else:
        print("load: ", model.evaluate_shear_load())
        assert model.evaluate_shear_load() > 0


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
def test_emi_deformation_3D(deformation_mode):
    mesh = df.UnitCubeMesh(1, 1, 1)
    volumes = df.MeshFunction("size_t", mesh, 3)
    volumes.array()[0] = 1

    model = EMIModel(mesh, volumes, experiment=deformation_mode)
    
    stretch_value = 0.01
    model.assign_stretch(stretch_value)
    model.solve()
    
    if "stretch" in deformation_mode:
        assert model.evaluate_normal_load() > 0
    else:
        assert model.evaluate_shear_load() > 0


if __name__ == "__main__":
    #test_emi_active_stress()
    #test_emi_active_strain()
    #test_emi_proj_strain()
    #test_emi_proj_stress()
    for deformation_mode in [
        "stretch_ff",
        #"shear_fs",
        #"shear_fn",
        #"shear_sf",
        "stretch_ss",
        #"shear_sn",
        #"shear_nf",
        #"shear_ns",
        "stretch_nn",
    ]:
        test_emi_deformation_3D(deformation_mode)
