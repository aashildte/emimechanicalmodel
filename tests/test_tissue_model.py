import os
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


def test_tissue_xstretch():
    mesh = df.UnitCubeMesh(1, 1, 1)
    
    model = TissueModel(
        mesh, experiment="xstretch"
    )

    stretch_value = 0.05
    model.assign_stretch(stretch_value)
    model.solve()
    
    assert(model.evaluate_load_yz() > 0)


def test_tissue_ystretch():
    mesh = df.UnitCubeMesh(1, 1, 1)
    
    model = TissueModel(
        mesh, experiment="ystretch"
    )
    
    stretch_value = 0.05
    model.assign_stretch(stretch_value)
    model.solve()
    
    assert(model.evaluate_load_xz() > 0)


if __name__ == "__main__":
    test_tissue_active()
    test_tissue_xstretch()
    test_tissue_ystretch()
    test_tissue_proj_strain()
    test_tissue_proj_stress()
