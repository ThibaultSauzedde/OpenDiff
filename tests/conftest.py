import pytest
import opendiff.materials as mat
import numpy as np

# clean tmp dir


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
    # rep.when --> setup, call or teardown
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


@pytest.fixture
def macrolib_2d(xs_aiea3d):
    all_mat, mat_names, reac_names = xs_aiea3d
    mat_lib = mat.Materials(all_mat, mat_names, reac_names)

    x = [0, 20*9]
    y = [0, 20*9]
    pblm = np.array([["fuel1_cr", "fuel1",    "fuel1",    "fuel1",    "fuel1_cr", "fuel1",    "fuel1",    "fuel2",   "refl"],
                     ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",   "refl"],
                     ["fuel1",    "fuel1",    "fuel1_cr", "fuel1",    "fuel1",    "fuel1",    "fuel2",    "fuel2",   "refl"],
                     ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",    "refl",    "refl"],
                     ["fuel1_cr", "fuel1",    "fuel1",    "fuel1",    "fuel1_cr", "fuel2",    "fuel2",    "refl",    "refl"],
                     ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",    "fuel2",    "refl",     "refl",    "refl"],
                     ["fuel1",    "fuel1",    "fuel2",    "fuel2",    "fuel2",    "refl",     "refl",     "refl",    "refl"],
                     ["fuel2",    "fuel2",    "fuel2",    "refl",     "refl",     "refl",     "refl",     "refl",    "refl"],
                     ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "refl"]])
    # pblm = np.concatenate((pblm, pblm[::-1, :]), axis=1)

    # print(pblm)

    #we mesh it 
    nb_div_pmat_x = 1
    nb_div_pmat_y = 1
    shape = (pblm.shape[0]*nb_div_pmat_y, pblm.shape[1]*nb_div_pmat_x)
    geom = np.empty(shape, dtype='U16')
    for i, row in enumerate(pblm):
        for j, value in enumerate(row):
            geom[i*nb_div_pmat_x:(i+1)*nb_div_pmat_x, j *
                nb_div_pmat_x:(j+1)*nb_div_pmat_x] = value  

    x_mesh = np.linspace(x[0], x[1], geom.shape[1]+1)
    y_mesh = np.linspace(y[0], y[1], geom.shape[0]+1)

    geometry = np.array([geom])

    macrolib = mat.Macrolib(mat_lib, geometry)

    return macrolib, x_mesh, y_mesh


@pytest.fixture
def macrolib_3d(xs_aiea3d):
    all_mat, mat_names, reac_names = xs_aiea3d
    mat_lib = mat.Materials(all_mat, mat_names, reac_names)

    x = [0, 20*9]
    y = [0, 20*9]
    z_delta = [0, 20., 260, 80, 20]
    z = np.cumsum(z_delta)

    pblm3 = np.array([ ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "refl"],
                    ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "refl"],
                    ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "refl"],
                    ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "refl"],
                    ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "void"],
                    ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "void"],
                    ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "void",     "void",    "void"],
                    ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "void",     "void",    "void"],
                    ["refl",     "refl",     "refl",     "refl",     "void",     "void",     "void",     "void",    "void"]])

    pblm2 = np.array([ ["fuel1_cr", "fuel1",    "fuel1",    "fuel1",    "fuel1_cr", "fuel1",    "fuel1",    "fuel2",   "refl"],
                    ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",   "refl"],
                    ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",    "fuel2",   "refl"],
                    ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",    "refl",    "refl"],
                    ["fuel1_cr", "fuel1",    "fuel1",    "fuel1",    "fuel1_cr", "fuel2",    "fuel2",    "refl",    "void"],
                    ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",    "fuel2",    "refl",     "refl",    "void"],
                    ["fuel1",    "fuel1",    "fuel2",    "fuel2",    "fuel2",    "refl",     "refl",     "void",    "void"],
                    ["fuel2",    "fuel2",    "fuel2",    "refl",     "refl",     "refl",     "void",     "void",    "void"],
                    ["refl",     "refl",     "refl",     "refl",     "void",     "void",     "void",     "void",    "void"]])

    pblm1 = np.array([ ["fuel1_cr", "fuel1",    "fuel1",    "fuel1",    "fuel1_cr", "fuel1",    "fuel1",    "fuel2",   "refl"],
                    ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",   "refl"],
                    ["fuel1",    "fuel1",    "fuel1_cr", "fuel1",    "fuel1",    "fuel1",    "fuel2",    "fuel2",   "refl"],
                    ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",    "refl",    "refl"],
                    ["fuel1_cr", "fuel1",    "fuel1",    "fuel1",    "fuel1_cr", "fuel2",    "fuel2",    "refl",    "void"],
                    ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",    "fuel2",    "refl",     "refl",    "void"],
                    ["fuel1",    "fuel1",    "fuel2",    "fuel2",    "fuel2",    "refl",     "refl",     "void",    "void"],
                    ["fuel2",    "fuel2",    "fuel2",    "refl",     "refl",     "refl",     "void",     "void",    "void"],
                    ["refl",     "refl",     "refl",     "refl",     "void",     "void",     "void",     "void",    "void"]])

    pblm0 = np.array([ ["refl_cr",  "refl",     "refl",     "refl",     "refl_cr",  "refl",     "refl",     "refl",    "refl"],
                    ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "refl"],
                    ["refl",     "refl",     "refl_cr",  "refl",     "refl",     "refl",     "refl",     "refl",    "refl"],
                    ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "refl"],
                    ["refl_cr",  "refl",     "refl",     "refl",     "refl_cr",  "refl",     "refl",     "refl",    "void"],
                    ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "void"],
                    ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "void",    "void"],
                    ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "void",     "void",    "void"],
                    ["refl",     "refl",     "refl",     "refl",     "void",     "void",     "void",     "void",    "void"]])

    #we mesh it
    pblm = []
    nb_div_pmat_x = 1
    nb_div_pmat_y = 1
    z_mesh = [[0., 20.],
              [21,  280],
              [281, 360], 
              [361, 380.]]
    z_mesh_r = z_mesh

    for pblm_i, z_mesh_i in zip([pblm3, pblm2, pblm1, pblm0], z_mesh_r):
        shape = (1, pblm_i.shape[0]*nb_div_pmat_y, pblm_i.shape[1]*nb_div_pmat_x)
        geom = np.empty(shape, dtype='U16')
        for i, row in enumerate(pblm_i):
            for j, value in enumerate(row):
                geom[0, i*nb_div_pmat_x:(i+1)*nb_div_pmat_x, j *
                    nb_div_pmat_y:(j+1)*nb_div_pmat_y] = value
        for k in range(len(z_mesh_i)):
            if z_mesh_i[k] == 0.:
                continue
            pblm.append(geom)

    pblm = np.concatenate(pblm, axis=0)

    x_mesh = np.linspace(x[0], x[1], pblm.shape[2]+1)
    y_mesh = np.linspace(y[0], y[1], pblm.shape[1]+1)
    z_mesh = np.array([y for x in z_mesh_r for y in x])

    macrolib = mat.Macrolib(mat_lib, pblm)
    # print(pblm)
    return macrolib, x_mesh, y_mesh, z_mesh


@pytest.fixture
def macrolib_3d_refine(xs_aiea3d):
    all_mat, mat_names, reac_names = xs_aiea3d
    mat_lib = mat.Materials(all_mat, mat_names, reac_names)

    x = [0, 20*9]
    y = [0, 20*9]
    z_delta = [0, 20., 260, 80, 20]
    z = np.cumsum(z_delta)

    pblm3 = np.array([ ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "refl"],
                    ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "refl"],
                    ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "refl"],
                    ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "refl"],
                    ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "void"],
                    ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "void"],
                    ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "void",     "void",    "void"],
                    ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "void",     "void",    "void"],
                    ["refl",     "refl",     "refl",     "refl",     "void",     "void",     "void",     "void",    "void"]])

    pblm2 = np.array([ ["fuel1_cr", "fuel1",    "fuel1",    "fuel1",    "fuel1_cr", "fuel1",    "fuel1",    "fuel2",   "refl"],
                    ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",   "refl"],
                    ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",    "fuel2",   "refl"],
                    ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",    "refl",    "refl"],
                    ["fuel1_cr", "fuel1",    "fuel1",    "fuel1",    "fuel1_cr", "fuel2",    "fuel2",    "refl",    "void"],
                    ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",    "fuel2",    "refl",     "refl",    "void"],
                    ["fuel1",    "fuel1",    "fuel2",    "fuel2",    "fuel2",    "refl",     "refl",     "void",    "void"],
                    ["fuel2",    "fuel2",    "fuel2",    "refl",     "refl",     "refl",     "void",     "void",    "void"],
                    ["refl",     "refl",     "refl",     "refl",     "void",     "void",     "void",     "void",    "void"]])

    pblm1 = np.array([ ["fuel1_cr", "fuel1",    "fuel1",    "fuel1",    "fuel1_cr", "fuel1",    "fuel1",    "fuel2",   "refl"],
                    ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",   "refl"],
                    ["fuel1",    "fuel1",    "fuel1_cr", "fuel1",    "fuel1",    "fuel1",    "fuel2",    "fuel2",   "refl"],
                    ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",    "refl",    "refl"],
                    ["fuel1_cr", "fuel1",    "fuel1",    "fuel1",    "fuel1_cr", "fuel2",    "fuel2",    "refl",    "void"],
                    ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",    "fuel2",    "refl",     "refl",    "void"],
                    ["fuel1",    "fuel1",    "fuel2",    "fuel2",    "fuel2",    "refl",     "refl",     "void",    "void"],
                    ["fuel2",    "fuel2",    "fuel2",    "refl",     "refl",     "refl",     "void",     "void",    "void"],
                    ["refl",     "refl",     "refl",     "refl",     "void",     "void",     "void",     "void",    "void"]])

    pblm0 = np.array([ ["refl_cr",  "refl",     "refl",     "refl",     "refl_cr",  "refl",     "refl",     "refl",    "refl"],
                    ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "refl"],
                    ["refl",     "refl",     "refl_cr",  "refl",     "refl",     "refl",     "refl",     "refl",    "refl"],
                    ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "refl"],
                    ["refl_cr",  "refl",     "refl",     "refl",     "refl_cr",  "refl",     "refl",     "refl",    "void"],
                    ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "void"],
                    ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "void",    "void"],
                    ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "void",     "void",    "void"],
                    ["refl",     "refl",     "refl",     "refl",     "void",     "void",     "void",     "void",    "void"]])

    #we mesh it
    pblm = []
    nb_div_pmat_x = 5
    nb_div_pmat_y = 5
    z_mesh = [[0., 5, 10, 13, 16, 18, 19, 20.],
            [21, 22, 24, 27, 30., 35, 40, 50, 60, 70, 80, 90, 100.,
            110., 120., 130., 140., 150., 160., 170., 180., 190., 200.,
            210, 220, 230, 240, 250, 260, 265, 270., 273, 276, 278, 279, 280],
            [281, 282, 284, 287, 300, 305, 310, 320, 330, 340, 345, 350, 353, 356, 358, 359, 360], 
            [361, 362, 364, 367, 370, 375, 380.]]
    z_mesh_r = z_mesh

    for pblm_i, z_mesh_i in zip([pblm3, pblm2, pblm1, pblm0], z_mesh_r):
        shape = (1, pblm_i.shape[0]*nb_div_pmat_y, pblm_i.shape[1]*nb_div_pmat_x)
        geom = np.empty(shape, dtype='U16')
        for i, row in enumerate(pblm_i):
            for j, value in enumerate(row):
                geom[0, i*nb_div_pmat_x:(i+1)*nb_div_pmat_x, j *
                    nb_div_pmat_y:(j+1)*nb_div_pmat_y] = value
        for k in range(len(z_mesh_i)):
            if z_mesh_i[k] == 0.:
                continue
            pblm.append(geom)

    pblm = np.concatenate(pblm, axis=0)

    x_mesh = np.linspace(x[0], x[1], pblm.shape[2]+1)
    dx = x_mesh[1:]-x_mesh[:-1]
    y_mesh = np.linspace(y[0], y[1], pblm.shape[1]+1)
    dy = y_mesh[1:]-y_mesh[:-1]

    z_mesh = np.array([y for x in z_mesh_r for y in x])

    dz = z_mesh[1:]-z_mesh[:-1]
    # surf = np.multiply.outer(dx, dy)
    # vol = np.multiply.outer(surf, dz)
    # vol_1d = vol.reshape(-1)

    macrolib = mat.Macrolib(mat_lib, pblm)

    return macrolib, x_mesh, y_mesh, z_mesh
