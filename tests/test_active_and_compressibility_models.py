import os
import pytest
import dolfin as df
from mpi4py import MPI
import numpy as np

from emimechanicalmodel import TissueModel


@pytest.mark.parametrize(
    ["active_model", "compressibility_model"],
    [
        ("active_strain", "incompressible"),
        ("active_stress", "incompressible"),
        ("active_strain", "nearly_incompressible"),
        ("active_stress", "nearly_incompressible"),
    ],
)
def test_compressibility_active(active_model, compressibility_model):
    mesh = df.UnitCubeMesh(1, 1, 1)
    
    model = TissueModel(
            mesh,
            active_model=active_model,
            compressibility_model=compressibility_model,
            experiment="contraction",
            )
    
    active_value = 0.001

    model.update_active_fn(active_value)
    model.solve(project=False)
    
    V = df.VectorFunctionSpace(mesh, "CG", 2)
    u = df.project(model.u, V)

    assert np.isclose(float(model.active_fn), active_value)
    assert np.linalg.norm(u.vector()[:]) > 0


@pytest.mark.parametrize(
    "compressibility_model",
    [
        "incompressible",
        "nearly_incompressible",
    ],
)
def test_compressibility_stretch(compressibility_model):
    mesh = df.UnitCubeMesh(1, 1, 1)
    
    model = TissueModel(
            mesh,
            compressibility_model=compressibility_model,
            experiment="stretch_ff",
            )
    
    stretch_value = 0.001

    model.assign_stretch(stretch_value)
    model.solve(project=False)
    
    V = df.VectorFunctionSpace(mesh, "CG", 2)
    u = df.project(model.u, V)

    assert np.linalg.norm(u.vector()[:]) > 0

if __name__ == "__main__":
    test_compressibility_active("active_strain", "incompressible")
    test_compressibility_active("active_stress", "incompressible")
    test_compressibility_active("active_strain", "nearly_incompressible")
    test_compressibility_active("active_stress", "nearly_incompressible")
    test_compressibility_stretch("incompressible")
    test_compressibility_stretch("nearly_incompressible")
