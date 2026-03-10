
import pytest
import dolfin as df
from emimechanicalmodel import EMIModel, TissueModel, SarcomereModel


@pytest.mark.parametrize(
    "ModelClass,dim",
    [(EMIModel, 2), (EMIModel, 3), (SarcomereModel, 2), (SarcomereModel, 3)],
    ids=["EMIModel_2","EMIModel_3","SarcomereModel_2","SarcomereModel_3"],
)
def test_fiber_strain_vs_transverse(ModelClass, dim):
    if dim == 3:
        mesh = df.UnitCubeMesh(1, 1, 1)
    elif dim == 2:
        mesh = df.UnitSquareMesh(1, 1)
    else:
        raise ValueError("dim must be 2 or 3")

    volumes = df.MeshFunction("size_t", mesh, mesh.topology().dim())
    volumes.array()[:] = 1

    model = ModelClass(mesh, volumes, experiment="contraction")
    active = 0.001
    model.update_active_fn(active)
    model.solve(project=True)

    E_ff = model.evaluate_subdomain_strain_fibre_dir(1)
    E_ss = model.evaluate_subdomain_strain_sheet_dir(1)

    assert E_ff < 0
    # assert abs(E_ff) > abs(E_ss) - not yet


if __name__ == "__main__":

    test_fiber_strain_vs_transverse(EMIModel, 2)
    test_fiber_strain_vs_transverse(EMIModel, 3)
    test_fiber_strain_vs_transverse(SarcomereModel, 2)
    test_fiber_strain_vs_transverse(SarcomereModel, 3)
