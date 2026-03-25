import pytest
import dolfin as df
from emimechanicalmodel import EMIModel, TissueModel, SarcomereModel

MODELS = [
    (EMIModel, 2),
    (EMIModel, 3),
    (SarcomereModel, 2),
    (SarcomereModel, 3),
]

ACTIVE = ["active_stress", "active_strain"]

@pytest.mark.parametrize("ModelClass,dim", MODELS)
@pytest.mark.parametrize("active_model", ACTIVE)
def test_fiber_strain_vs_transverse(ModelClass, dim, active_model):

    # Mesh
    if dim == 3:
        mesh = df.UnitCubeMesh(1, 1, 1)
    elif dim == 2:
        mesh = df.UnitSquareMesh(1, 1)
    else:
        raise ValueError("dim must be 2 or 3")

    volumes = df.MeshFunction("size_t", mesh, mesh.topology().dim())
    volumes.array()[:] = 1

    # Model
    model = ModelClass(
        mesh,
        volumes,
        experiment="contraction",
        active_model=active_model,
        compressibility_model="nearly_incompressible",
    )

    active = 0.01
    model.update_active_fn(active)
    model.solve(project=True)

    # Strains
    E_ff = model.evaluate_subdomain_strain_fibre_dir(1)
    E_ss = model.evaluate_subdomain_strain_sheet_dir(1)

    # Assertions
    assert E_ff < 0
    print(ModelClass.__name__, dim, active_model, abs(E_ff), abs(E_ss))
    assert abs(E_ff) > abs(E_ss)


if __name__ == "__main__":
    test_fiber_strain_vs_transverse(EMIModel, 2, "active_stress")

