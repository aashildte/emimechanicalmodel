import os
import pytest
import dolfin as df
from mpi4py import MPI
import numpy as np

from emimechanicalmodel import TissueModel


@pytest.mark.parametrize(
    ["active_model", "compressible_model"],
    [
        ["active_strain", "incompressible"],
        ["active_stress", "incompressible"],
    ],
)
def test_active_contraction(active_model, compressibility_model):
    mesh = df.UnitCubeMesh(1, 1, 1)

    model = TissueModel(
            mesh,
            active_model=active_model,
            compressibility_model=compressibility_model,
            experiment="contr",
            )

    active_value = 0.001

    model.update_active_fn(active_value)
    model.solve(project=False)
    
    u, p, _ = model.state.split(deepcopy=True)

    assert np.isclose(float(model.active_fn), active_value)
    assert np.linalg.norm(u.vector()[:]) > 0


if __name__ == "__main__":
    test_active_model("active_stress", "incompressible")
    test_active_model("active_strain", "incompressible")
