import pytest
import dolfin as df
from math import isclose

from emimechanicalmodel import TissueModel, EMIModel


@pytest.mark.parametrize(
    ("mat_param_name", "mat_param_value", "experiment"),
    [
        ("a", 0.074, "stretch_ff"),
        ("b", 4.878, "stretch_ff"),
        ("a_f", 2.628, "stretch_ff"),
        ("b_f", 5.214, "stretch_ff"),
        ("a_s", 0.438, "stretch_ss"),
        ("b_s", 3.002, "stretch_ss"),
        ("a_fs", 0.062, "shear_fs"),
        ("b_fs", 3.476, "shear_fs"),
    ],
)
def test_tissue_params(mat_param_name, mat_param_value, experiment):

    mesh = df.UnitCubeMesh(1, 1, 1)

    material_parameters = {mat_param_name: df.Constant(mat_param_value)}

    model = TissueModel(
        mesh,
        material_parameters=material_parameters,
        experiment=experiment,
    )

    model.assign_stretch(0.02)
    model.solve()
    load1 = model.evaluate_normal_load()

    model.mat_model.__dict__[mat_param_name].assign(2 * mat_param_value)
    model.solve()
    load2 = model.evaluate_normal_load()

    print(load1, load2)

    assert not isclose(
        load1, load2
    ), f"No sensitivity for material parameter {mat_param_name}."
        

@pytest.mark.parametrize(
    ("mat_param_name", "mat_param_value", "experiment"),
    [
        ("a_i", 5.7, "stretch_ff"),
        ("b_i", 11.67, "stretch_ff"),
        ("a_e", 1.52, "stretch_ff"),
        ("b_e", 16.31, "stretch_ff"),
        ("a_if", 19.83, "stretch_ff"),
        ("b_if", 24.72, "stretch_ff"),
    ],
)
def test_emi_holzapfel_params(mat_param_name, mat_param_value, experiment):

    mesh = df.UnitCubeMesh(1, 1, 1)
    volumes = df.MeshFunction("size_t", mesh, 3)
    volumes.array()[0] = 1

    material_parameters = {mat_param_name: df.Constant(mat_param_value)}

    model = EMIModel(
        mesh,
        volumes,
        material_parameters=material_parameters,
        experiment=experiment,
    )

    model.assign_stretch(0.02)
    model.solve()
    load1 = model.evaluate_normal_load()

    model.mat_model.__dict__[mat_param_name].assign(2 * mat_param_value)
    model.solve()

    load2 = model.evaluate_normal_load()

    assert not isclose(
        load1, load2
    ), f"No sensitivity for material parameter {mat_param_name}."


@pytest.mark.parametrize(
    ("mat_param_name", "mat_param_value", "experiment"),
    [
        ("C_i", 0.5, "stretch_ff"),
        ("C_e", 0.5, "stretch_ff"),
        ("b_if", 8, "stretch_ff"),
        ("b_it", 2, "stretch_ff"),
        ("b_ift", 4, "stretch_ff"),
        ("b_e", 2, "stretch_ff"),
    ],
)
def test_emi_guccione_params(mat_param_name, mat_param_value, experiment):

    mesh = df.UnitCubeMesh(1, 1, 1)
    volumes = df.MeshFunction("size_t", mesh, 3)
    volumes.array()[0] = 1

    material_parameters = {mat_param_name: df.Constant(mat_param_value)}

    model = EMIModel(
        mesh,
        volumes,
        material_model="guccione",
        material_parameters=material_parameters,
        experiment=experiment,
    )

    model.assign_stretch(0.02)
    model.solve()
    load1 = model.evaluate_normal_load()

    model.mat_model.__dict__[mat_param_name].assign(2 * mat_param_value)
    model.solve()

    load2 = model.evaluate_normal_load()

    assert not isclose(
        load1, load2
    ), f"No sensitivity for material parameter {mat_param_name}."


if __name__ == "__main__":
    test_emi_guccione_params("C_i", 0.5, "stretch_ff")
    #test_tissue_params("a_fs", 0.062, "shear_fs"),
