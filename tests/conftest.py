import pytest
import opendiff.materials as mat
import numpy as np

#clean tmp dir


def pytest_runtest_teardown(item):
    if item.rep_call.passed:
        if "tmpdir" in item.funcargs:
            tmpdir = item.funcargs["tmpdir"]
            if tmpdir.check():
                tmpdir.remove()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call,):
    outcome = yield
    rep = outcome.get_result()
    #rep.when --> setup, call or teardown
    setattr(item, f"rep_{rep.when}", rep)


@pytest.fixture
def xs_aiea3d():
    fuel1 = [[1.5, 0.01, 0, 1, 0., 0.020],
             [0.4, 0.085, 0.135, 0, 0., 0.]]

    fuel1_cr = [[1.5, 0.01, 0, 1, 0., 0.020],
                [0.4, 0.130, 0.135, 0, 0., 0.]]

    fuel2 = [[1.5, 0.01, 0, 1, 0., 0.020],
             [0.4, 0.080, 0.135, 0, 0., 0.]]

    refl = [[2.0, 0.0, 0, 0, 0., 0.040],
            [0.3, 0.01, 0., 0, 0., 0.]]

    refl_cr = [[2.0, 0.0, 0, 0, 0., 0.040],
               [0.3, 0.055, 0., 0, 0., 0.]]

    void = [[1e10, 0., 0., 0, 0., 0.],
            [1e10, 0., 0., 0, 0., 0.]]

    all_mat = [fuel1, fuel1_cr, fuel2, refl, refl_cr, void]
    mat_names = ["fuel1", "fuel1_cr", "fuel2", "refl", "refl_cr", "void"]
    reac_names = ["D", "SIGA", "NU_SIGF", "CHI", "1", "2"]

    return all_mat, mat_names, reac_names


@pytest.fixture
def macrolib_1d(xs_aiea3d):
    all_mat, mat_names, reac_names = xs_aiea3d
    mat_lib = mat.Materials(all_mat, mat_names, reac_names)

    x = [0, 25]

    # we mesh it
    nb_cells = 20
    geometry = ["fuel1"] * nb_cells
    x_mesh = np.linspace(x[0], x[1], nb_cells+1)

    geometry = np.array([[geometry]])
    # print(geometry.shape)
    # print(geometry)

    macrolib = mat.Macrolib(mat_lib, geometry)

    return macrolib, x_mesh
