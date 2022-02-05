import os
import pytest
import dolfin as df
from mpi4py import MPI
import numpy as np

from emimechanicalmodel import TissueModel

def test_tissue_active():
    mesh = df.UnitCubeMesh(1, 1, 1)

    model = TissueModel(
        mesh, experiment="contr"
    )

    active_value = 0.001
    
    model.update_active_fn(active_value)
    model.solve(project=False)

    assert abs(float(model.active_fn) - active_value) < 1E-10


def test_tissue_proj_strain():
    mesh = df.UnitCubeMesh(1, 1, 1)

    model = TissueModel(
        mesh, experiment="contr"
    )

    active_value = 0.001
    
    model.update_active_fn(active_value)
    model.solve(project=True)

    assert model.evaluate_subdomain_strain_fibre_dir(0) < 0


def test_tissue_proj_stress():
    mesh = df.UnitCubeMesh(1, 1, 1)

    model = TissueModel(
        mesh, experiment="contr"
    )

    active_value = 0.001
    
    model.update_active_fn(active_value)
    model.solve(project=True)

    assert model.evaluate_subdomain_stress_fibre_dir(0) > 0


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
def test_tissue_deformation(deformation_mode):
    mesh = df.UnitCubeMesh(1, 1, 1)
    
    model = TissueModel(
        mesh, experiment=deformation_mode,
    )

    stretch_value = 0.05
    model.assign_stretch(stretch_value)
    model.solve()
    
    print(model.evaluate_load())

    assert(model.evaluate_load() > 0)


if __name__ == "__main__":
    test_tissue_active()
    test_tissue_proj_strain()
    test_tissue_proj_stress()
    test_tissue_deformation("shear_fs")
    test_tissue_deformation("shear_sf")
    test_tissue_deformation("shear_nf")
    test_tissue_deformation("shear_fn")
    test_tissue_deformation("shear_sn")
    test_tissue_deformation("shear_ns")
