"""

Analytic solution of simple shear experiments, checked against Dokos et al (appendix)

Bit of code that is useful for finding bugs:

    e1 = df.as_vector([1, 0, 0])
    e2 = df.as_vector([0, 1, 0])
    e3 = df.as_vector([0, 0, 1])
    
    E = model.E
    
    E11 = df.assemble(df.inner(E*e1, e1)*df.dx(mesh))
    E12 = df.assemble(df.inner(E*e1, e2)*df.dx(mesh))
    E13 = df.assemble(df.inner(E*e1, e3)*df.dx(mesh))
    E22 = df.assemble(df.inner(E*e2, e2)*df.dx(mesh))
    E23 = df.assemble(df.inner(E*e2, e3)*df.dx(mesh))
    E33 = df.assemble(df.inner(E*e3, e3)*df.dx(mesh))

    print([E11, E12, E13])
    print([E12, E22, E23])
    print([E13, E23, E33])
"""

import dolfin as df
from emimechanicalmodel import TissueModel, ShearFN, ShearNF, ShearFS, ShearSF, ShearNS, ShearSN

def test_tissue_shear_ns():
    mesh = df.UnitCubeMesh(3, 3, 3)
    
    model = TissueModel(
        mesh, experiment="shear_ns",
    )

    k = 0.05
    model.assign_stretch(k)
    model.solve()

    E_expected = df.as_tensor(
        [[0, 0, 0],
         [0, 0, 0.5*k],
         [0, 0.5*k, 0.5*k**2]])
    
    diff_E = model.E - E_expected
    diff_norm = df.assemble(df.inner(diff_E, diff_E)*df.dx)
   
    assert diff_norm < 1e-14, "Error: Incorrect ns shear"

def test_tissue_shear_nf():
    mesh = df.UnitCubeMesh(1, 1, 1)
    
    model = TissueModel(
        mesh, experiment="shear_nf",
    )

    k = 0.05
    model.assign_stretch(k)
    model.solve()

    E_expected = df.as_tensor(
        [[0, 0, 0.5*k],
         [0, 0, 0],
         [0.5*k, 0, 0.5*k**2]])
    
    diff_E = model.E - E_expected
    diff_norm = df.assemble(df.inner(diff_E, diff_E)*df.dx)
   
    assert diff_norm < 1e-14, "Error: Incorrect nf shear"


def test_tissue_shear_sn():
    mesh = df.UnitCubeMesh(1, 1, 1)
    
    model = TissueModel(
        mesh, experiment="shear_sn",
    )

    k = 0.05
    model.assign_stretch(k)
    model.solve()

    E_expected = df.as_tensor(
        [[0, 0, 0],
         [0, 0.5*k**2, 0.5*k],
         [0, 0.5*k, 0]])
    
    diff_E = model.E - E_expected
    diff_norm = df.assemble(df.inner(diff_E, diff_E)*df.dx)
   
    assert diff_norm < 1e-14, "Error: Incorrect sn shear"


def test_tissue_shear_sf():
    mesh = df.UnitCubeMesh(1, 1, 1)
    
    model = TissueModel(
        mesh, experiment="shear_sf",
    )

    k = 0.05
    model.assign_stretch(k)
    model.solve()

    E_expected = df.as_tensor(
        [[0, 0.5*k, 0],
         [0.5*k, 0.5*k**2, 0],
         [0, 0, 0]])
    
    diff_E = model.E - E_expected
    diff_norm = df.assemble(df.inner(diff_E, diff_E)*df.dx)
   
    assert diff_norm < 1e-14, "Error: Incorrect sf shear"


def test_tissue_shear_fn():
    mesh = df.UnitCubeMesh(1, 1, 1)
    
    model = TissueModel(
        mesh, experiment="shear_fn",
    )

    k = 0.05
    model.assign_stretch(k)
    model.solve()

    E_expected = df.as_tensor(
        [[0.5*k**2, 0, 0.5*k],
         [0, 0, 0],
         [0.5*k, 0, 0]])
    
    diff_E = model.E - E_expected
    diff_norm = df.assemble(df.inner(diff_E, diff_E)*df.dx)
   
    assert diff_norm < 1e-14, "Error: Incorrect fn shear"


def test_tissue_shear_fs():
    mesh = df.UnitCubeMesh(1, 1, 1)
    
    model = TissueModel(
        mesh, experiment="shear_fs",
    )

    k = 0.05
    model.assign_stretch(k)
    model.solve()

    E_expected = df.as_tensor(
        [[0.5*k**2, 0.5*k, 0],
         [0.5*k, 0, 0],
         [0, 0, 0]])
    
    diff_E = model.E - E_expected
    diff_norm = df.assemble(df.inner(diff_E, diff_E)*df.dx)
   
    assert diff_norm < 1e-14, "Error: Incorrect fs shear"


if __name__ == "__main__":
    test_tissue_shear_ns()
    test_tissue_shear_nf()
    test_tissue_shear_sn()
    test_tissue_shear_sf()
    test_tissue_shear_fn()
    test_tissue_shear_fs()
