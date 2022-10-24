import os
import pytest
import dolfin as df
from mpi4py import MPI
import numpy as np

from emimechanicalmodel import TissueModel


@pytest.mark.parametrize(
    ["active_model", "compressibility_model", "experiment"],
    [
        ("active_strain", "incompressible", "contraction"),
        ("active_stress", "incompressible", "contraction"),
        ("active_strain", "nearly_incompressible", "contraction"),
        ("active_stress", "nearly_incompressible", "contraction"),
        ("active_strain", "incompressible", "stretch_ff"),
        ("active_stress", "incompressible", "stretch_ff"),
        ("active_strain", "nearly_incompressible", "stretch_ff"),
        ("active_stress", "nearly_incompressible", "stretch_ff"),
    ],
)
def test_active_and_compressibility(active_model, compressibility_model, experiment):
    mesh = df.UnitCubeMesh(1, 1, 1)
    
    model = TissueModel(
            mesh,
            active_model=active_model,
            compressibility_model=compressibility_model,
            experiment=experiment,
            )

    active_value = 0.001

    model.update_active_fn(active_value)
    model.solve(project=False)
    
    V = df.VectorFunctionSpace(mesh, "CG", 2)
    u = df.project(model.u, V)

    assert np.isclose(float(model.active_fn), active_value)
    assert np.linalg.norm(u.vector()[:]) > 0

if __name__ == "__main__":
    test_active_and_compressibility("active_strain", "incompressible", "contraction")
    test_active_and_compressibility("active_stress", "incompressible", "contraction")
    test_active_and_compressibility("active_strain", "nearly_incompressible", "contraction")
    test_active_and_compressibility("active_stress", "nearly_incompressible", "contraction")
    test_active_and_compressibility("active_strain", "incompressible", "stretch_ff")
    test_active_and_compressibility("active_stress", "incompressible", "stretch_ff")
    test_active_and_compressibility("active_strain", "nearly_incompressible", "stretch_ff")
    test_active_and_compressibility("active_stress", "nearly_incompressible", "stretch_ff")
