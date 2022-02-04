
import pytest
import dolfin as df
from math import isclose

from emimechanicalmodel import TissueModel

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
def test_tissue_model(mat_param_name, mat_param_value, experiment):

    mesh = df.UnitCubeMesh(1, 1, 1)

    material_parameters = {mat_param_name : df.Constant(mat_param_value)}
    
    model = TissueModel(
        mesh,
        material_parameters=material_parameters,
        experiment=experiment,
    )

    model.assign_stretch(0.05)
    model.solve()
    load1 = model.evaluate_load()  
    
    model.mat_model.__dict__[mat_param_name].assign(2*mat_param_value)
    model.solve()
    load2 = model.evaluate_load()
    
    assert not isclose(load1, load2), \
            f"No sensitivity for material parameter {mat_param_name}."
