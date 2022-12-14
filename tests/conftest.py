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
    fuel1 = [[1.5, 0.01, 0, 1, 0., 0.020, 202 * 1.60218e-19 * 1e6, 2.4],
             [0.4, 0.085, 0.135, 0, 0., 0., 202 * 1.60218e-19 * 1e6, 2.4]]

    fuel1_cr = [[1.5, 0.01, 0, 1, 0., 0.020, 202 * 1.60218e-19 * 1e6, 2.4],
                [0.4, 0.130, 0.135, 0, 0., 0., 202 * 1.60218e-19 * 1e6, 2.4]]

    fuel2 = [[1.5, 0.01, 0, 1, 0., 0.020, 202 * 1.60218e-19 * 1e6, 2.4],
             [0.4, 0.080, 0.135, 0, 0., 0., 202 * 1.60218e-19 * 1e6, 2.4]]

    refl = [[2.0, 0.0, 0, 0, 0., 0.040, 202 * 1.60218e-19 * 1e6, 2.4],
            [0.3, 0.01, 0., 0, 0., 0., 202 * 1.60218e-19 * 1e6, 2.4]]

    refl_cr = [[2.0, 0.0, 0, 0, 0., 0.040, 202 * 1.60218e-19 * 1e6, 2.4],
               [0.3, 0.055, 0., 0, 0., 0., 202 * 1.60218e-19 * 1e6, 2.4]]

    void = [[1e10, 0., 0., 0, 0., 0., 202 * 1.60218e-19 * 1e6, 2.4],
            [1e10, 0., 0., 0, 0., 0., 202 * 1.60218e-19 * 1e6, 2.4]]

    all_mat = {"fuel1": fuel1, "fuel1_cr": fuel1_cr, "fuel2": fuel2, "refl": refl, "refl_cr": refl_cr, "void": void}
    middles = {mat_name: mat_name for mat_name, _ in all_mat.items()}
    isot_reac_names = [("ISO", "D"), ("ISO", "SIGA"), ("ISO", "NU_SIGF"), ("ISO", "CHI"),
                       ("ISO", "1"), ("ISO", "2"), ("ISO", "EFISS"), ("ISO", "NU")]

    return all_mat, middles, isot_reac_names


@pytest.fixture
def xs_aiea3d_pert_mat(xs_aiea3d):
    all_mat, middles, isot_reac_names = xs_aiea3d
    materials = {mat_name: mat.Material(values, isot_reac_names) for mat_name, values in all_mat.items()}
    middles = mat.Middles(materials, middles)
    middles_pert = mat.Middles(middles)
    # middles_pert.multXsValue("fuel1", 2, "NU_SIGF", "ISO", 1.02)
    # middles_pert.multXsValue("fuel1", 2, "SIGR", "ISO", 1.01)
    sigr = middles_pert.getXsValue("fuel1", 2, "NU_SIGF")*1.02
    middles_pert.setXsValue("fuel1", 2, "NU_SIGF", "ISO", sigr)

    sigr = middles_pert.getXsValue("fuel1", 2, "SIGR")*1.01
    middles_pert.setXsValue("fuel1", 2, "SIGR", "ISO", sigr)
    return middles_pert


def get_1d_geom(nb_cells=20):
    x = [0, 25]

    geometry = ["fuel1"] * nb_cells
    x_mesh = np.linspace(x[0], x[1], nb_cells+1)

    geometry = np.array([[geometry]])
    # print(geometry.shape)
    # print(geometry)

    return geometry, x_mesh


@pytest.fixture
def nmid_geom_1d(nb_cells=20):
    x = [0, 20*9*2]
    pblm = ["refl", "fuel2", "fuel1", "fuel1", "fuel1_cr", "fuel1", "fuel1", "fuel1", "fuel1_cr"]
    pblm += pblm[::-1]
    
    #we mesh it 
    pblm_meshed = []
    for i in pblm: 
        pblm_meshed.extend([i]*nb_cells)
    
    x_mesh = np.linspace(x[0], x[1], len(pblm_meshed)+1)

    pblm_meshed = [[pblm_meshed]]

    return pblm_meshed, x_mesh

@pytest.fixture
def macrolib_1d(xs_aiea3d):
    all_mat, middles, isot_reac_names = xs_aiea3d
    materials = {mat_name: mat.Material(values, isot_reac_names) for mat_name, values in all_mat.items()}
    middles = mat.Middles(materials, middles)
    geometry, x_mesh = get_1d_geom()
    macrolib = mat.Macrolib(middles, geometry)

    return macrolib, x_mesh

@pytest.fixture
def macrolib_1d_nmid(xs_aiea3d, nmid_geom_1d):
    all_mat, middles, isot_reac_names = xs_aiea3d
    materials = {mat_name: mat.Material(values, isot_reac_names) for mat_name, values in all_mat.items()}
    middles = mat.Middles(materials, middles)
    geometry, x_mesh = nmid_geom_1d
    macrolib = mat.Macrolib(middles, geometry)

    return macrolib, x_mesh

@pytest.fixture
def macrolib_1d_pert(xs_aiea3d_pert_mat):
    geometry, x_mesh = get_1d_geom()
    return mat.Macrolib(xs_aiea3d_pert_mat, geometry)

@pytest.fixture
def macrolib_1d_nmid_pert(xs_aiea3d_pert_mat, nmid_geom_1d):
    geometry, x_mesh = nmid_geom_1d
    return mat.Macrolib(xs_aiea3d_pert_mat, geometry)

@pytest.fixture
def macrolib_1d_refine(xs_aiea3d):
    all_mat, middles, isot_reac_names = xs_aiea3d
    materials = {mat_name: mat.Material(values, isot_reac_names) for mat_name, values in all_mat.items()}
    middles = mat.Middles(materials, middles)
    geometry, x_mesh = get_1d_geom(50)
    macrolib = mat.Macrolib(middles, geometry)

    return macrolib, x_mesh


@pytest.fixture
def macrolib_1d_pert_refine(xs_aiea3d_pert_mat):
    geometry, x_mesh = get_1d_geom(50)
    return mat.Macrolib(xs_aiea3d_pert_mat, geometry)


def get_2d_geom(nb_div_pmat_x=1, nb_div_pmat_y=1):
    x = [0, 20*9]
    y = [0, 20*9]
    pblm = np.array([["fuel1_cr", "fuel1",    "fuel1",    "fuel1",    "fuel1_cr", "fuel1",    "fuel1",    "fuel2",   "refl"],
                     ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",
                         "fuel1",    "fuel1",    "fuel2",   "refl"],
                     ["fuel1",    "fuel1",    "fuel1_cr", "fuel1",    "fuel1",
                         "fuel1",    "fuel2",    "fuel2",   "refl"],
                     ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",
                         "fuel1",    "fuel2",    "refl",    "refl"],
                     ["fuel1_cr", "fuel1",    "fuel1",    "fuel1",
                         "fuel1_cr", "fuel2",    "fuel2",    "refl",    "refl"],
                     ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",
                         "fuel2",    "refl",     "refl",    "refl"],
                     ["fuel1",    "fuel1",    "fuel2",    "fuel2",    "fuel2",
                         "refl",     "refl",     "refl",    "refl"],
                     ["fuel2",    "fuel2",    "fuel2",    "refl",     "refl",
                         "refl",     "refl",     "refl",    "refl"],
                     ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "refl"]])
    # pblm = np.concatenate((pblm, pblm[::-1, :]), axis=1)

    # print(pblm)

    #we mesh it
    shape = (pblm.shape[0]*nb_div_pmat_y, pblm.shape[1]*nb_div_pmat_x)
    geom = np.empty(shape, dtype='U16')
    for i, row in enumerate(pblm):
        for j, value in enumerate(row):
            geom[i*nb_div_pmat_x:(i+1)*nb_div_pmat_x, j *
                nb_div_pmat_x:(j+1)*nb_div_pmat_x] = value  

    x_mesh = np.linspace(x[0], x[1], geom.shape[1]+1)
    y_mesh = np.linspace(y[0], y[1], geom.shape[0]+1)

    geometry = np.array([geom])

    return geometry, x_mesh, y_mesh


@pytest.fixture
def macrolib_2d(xs_aiea3d):
    all_mat, middles, isot_reac_names = xs_aiea3d
    materials = {mat_name: mat.Material(values, isot_reac_names) for mat_name, values in all_mat.items()}
    middles = mat.Middles(materials, middles)
    geometry, x_mesh, y_mesh = get_2d_geom()
    macrolib = mat.Macrolib(middles, geometry)
    return macrolib, x_mesh, y_mesh


@pytest.fixture
def macrolib_2d_pert(xs_aiea3d_pert_mat):
    geometry, x_mesh, y_mesh = get_2d_geom()
    macrolib = mat.Macrolib(xs_aiea3d_pert_mat, geometry)
    return macrolib


@pytest.fixture
def macrolib_2d_refine(xs_aiea3d):
    all_mat, middles, isot_reac_names = xs_aiea3d
    materials = {mat_name: mat.Material(values, isot_reac_names) for mat_name, values in all_mat.items()}
    middles = mat.Middles(materials, middles)
    geometry, x_mesh, y_mesh = get_2d_geom(20, 20)
    macrolib = mat.Macrolib(middles, geometry)
    return macrolib, x_mesh, y_mesh


@pytest.fixture
def macrolib_2d_pert_refine(xs_aiea3d_pert_mat):
    geometry, x_mesh, y_mesh = get_2d_geom(20, 20)
    macrolib = mat.Macrolib(xs_aiea3d_pert_mat, geometry)
    return macrolib


def get_3d_geom(nb_div_pmat_x=1, nb_div_pmat_y=1, z_mesh=None):
    x = [0, 20*9]
    y = [0, 20*9]
    z_delta = [0, 20., 260, 80, 20]
    z = np.cumsum(z_delta)

    pblm3 = np.array([["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "refl"],
                      ["refl",     "refl",     "refl",     "refl",     "refl",
                          "refl",     "refl",     "refl",    "refl"],
                      ["refl",     "refl",     "refl",     "refl",     "refl",
                          "refl",     "refl",     "refl",    "refl"],
                      ["refl",     "refl",     "refl",     "refl",     "refl",
                       "refl",     "refl",     "refl",    "refl"],
                      ["refl",     "refl",     "refl",     "refl",     "refl",
                       "refl",     "refl",     "refl",    "void"],
                      ["refl",     "refl",     "refl",     "refl",     "refl",
                       "refl",     "refl",     "refl",    "void"],
                      ["refl",     "refl",     "refl",     "refl",     "refl",
                       "refl",     "void",     "void",    "void"],
                      ["refl",     "refl",     "refl",     "refl",     "refl",
                       "refl",     "void",     "void",    "void"],
                      ["refl",     "refl",     "refl",     "refl",     "void",     "void",     "void",     "void",    "void"]])

    pblm2 = np.array([["fuel1_cr", "fuel1",    "fuel1",    "fuel1",    "fuel1_cr", "fuel1",    "fuel1",    "fuel2",   "refl"],
                      ["fuel1",    "fuel1",    "fuel1",    "fuel1",
                          "fuel1",    "fuel1",    "fuel1",    "fuel2",   "refl"],
                      ["fuel1",    "fuel1",    "fuel1",    "fuel1",
                          "fuel1",    "fuel1",    "fuel2",    "fuel2",   "refl"],
                      ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",
                       "fuel1",    "fuel2",    "refl",    "refl"],
                      ["fuel1_cr", "fuel1",    "fuel1",    "fuel1",    "fuel1_cr",
                       "fuel2",    "fuel2",    "refl",    "void"],
                      ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",
                       "fuel2",    "refl",     "refl",    "void"],
                      ["fuel1",    "fuel1",    "fuel2",    "fuel2",    "fuel2",
                       "refl",     "refl",     "void",    "void"],
                      ["fuel2",    "fuel2",    "fuel2",    "refl",     "refl",
                       "refl",     "void",     "void",    "void"],
                      ["refl",     "refl",     "refl",     "refl",     "void",     "void",     "void",     "void",    "void"]])

    pblm1 = np.array([["fuel1_cr", "fuel1",    "fuel1",    "fuel1",    "fuel1_cr", "fuel1",    "fuel1",    "fuel2",   "refl"],
                      ["fuel1",    "fuel1",    "fuel1",    "fuel1",
                          "fuel1",    "fuel1",    "fuel1",    "fuel2",   "refl"],
                      ["fuel1",    "fuel1",    "fuel1_cr", "fuel1",
                          "fuel1",    "fuel1",    "fuel2",    "fuel2",   "refl"],
                      ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",
                       "fuel1",    "fuel2",    "refl",    "refl"],
                      ["fuel1_cr", "fuel1",    "fuel1",    "fuel1",    "fuel1_cr",
                       "fuel2",    "fuel2",    "refl",    "void"],
                      ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",
                       "fuel2",    "refl",     "refl",    "void"],
                      ["fuel1",    "fuel1",    "fuel2",    "fuel2",    "fuel2",
                       "refl",     "refl",     "void",    "void"],
                      ["fuel2",    "fuel2",    "fuel2",    "refl",     "refl",
                       "refl",     "void",     "void",    "void"],
                      ["refl",     "refl",     "refl",     "refl",     "void",     "void",     "void",     "void",    "void"]])

    pblm0 = np.array([["refl_cr",  "refl",     "refl",     "refl",     "refl_cr",  "refl",     "refl",     "refl",    "refl"],
                      ["refl",     "refl",     "refl",     "refl",     "refl",
                          "refl",     "refl",     "refl",    "refl"],
                      ["refl",     "refl",     "refl_cr",  "refl",     "refl",
                          "refl",     "refl",     "refl",    "refl"],
                      ["refl",     "refl",     "refl",     "refl",     "refl",
                       "refl",     "refl",     "refl",    "refl"],
                      ["refl_cr",  "refl",     "refl",     "refl",     "refl_cr",
                       "refl",     "refl",     "refl",    "void"],
                      ["refl",     "refl",     "refl",     "refl",     "refl",
                       "refl",     "refl",     "refl",    "void"],
                      ["refl",     "refl",     "refl",     "refl",     "refl",
                       "refl",     "refl",     "void",    "void"],
                      ["refl",     "refl",     "refl",     "refl",     "refl",
                       "refl",     "void",     "void",    "void"],
                      ["refl",     "refl",     "refl",     "refl",     "void",     "void",     "void",     "void",    "void"]])

    #we mesh it
    pblm = []
    if z_mesh is None:
        z_mesh = [[0., 20.],
                  [21,  280],
                  [281, 360],
                  [361, 380.]]
    z_mesh_r = z_mesh

    for pblm_i, z_mesh_i in zip([pblm3, pblm2, pblm1, pblm0], z_mesh_r):
        shape = (1, pblm_i.shape[0]*nb_div_pmat_y,
                 pblm_i.shape[1]*nb_div_pmat_x)
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

    return pblm, x_mesh, y_mesh, z_mesh


@pytest.fixture
def macrolib_3d(xs_aiea3d):
    all_mat, middles, isot_reac_names = xs_aiea3d
    materials = {mat_name: mat.Material(values, isot_reac_names) for mat_name, values in all_mat.items()}
    middles = mat.Middles(materials, middles)
    pblm, x_mesh, y_mesh, z_mesh = get_3d_geom()
    macrolib = mat.Macrolib(middles, pblm)
    # print(pblm)
    return macrolib, x_mesh, y_mesh, z_mesh


@pytest.fixture
def macrolib_3d_pert(xs_aiea3d_pert_mat):
    pblm, x_mesh, y_mesh, z_mesh = get_3d_geom()
    macrolib = mat.Macrolib(xs_aiea3d_pert_mat, pblm)
    return macrolib, x_mesh, y_mesh, z_mesh


@pytest.fixture
def macrolib_3d_refine(xs_aiea3d):
    all_mat, middles, isot_reac_names = xs_aiea3d
    materials = {mat_name: mat.Material(values, isot_reac_names) for mat_name, values in all_mat.items()}
    middles = mat.Middles(materials, middles)
    nb_div_pmat_x = 5
    nb_div_pmat_y = 5
    z_mesh = [[0., 5, 10, 13, 16, 18, 19, 20.],
              [21, 22, 24, 27, 30., 35, 40, 50, 60, 70, 80, 90, 100.,
               110., 120., 130., 140., 150., 160., 170., 180., 190., 200.,
               210, 220, 230, 240, 250, 260, 265, 270., 273, 276, 278, 279, 280],
              [281, 282, 284, 287, 300, 305, 310, 320, 330,
                  340, 345, 350, 353, 356, 358, 359, 360],
              [361, 362, 364, 367, 370, 375, 380.]]
    pblm, x_mesh, y_mesh, z_mesh = get_3d_geom(
        nb_div_pmat_x, nb_div_pmat_y, z_mesh)
    macrolib = mat.Macrolib(middles, pblm)
    # print(pblm)
    return macrolib, x_mesh, y_mesh, z_mesh


@pytest.fixture
def macrolib_3d_pert_refine(xs_aiea3d_pert_mat):
    nb_div_pmat_x = 5
    nb_div_pmat_y = 5
    z_mesh = [[0., 5, 10, 13, 16, 18, 19, 20.],
            [21, 22, 24, 27, 30., 35, 40, 50, 60, 70, 80, 90, 100.,
            110., 120., 130., 140., 150., 160., 170., 180., 190., 200.,
            210, 220, 230, 240, 250, 260, 265, 270., 273, 276, 278, 279, 280],
            [281, 282, 284, 287, 300, 305, 310, 320, 330, 340, 345, 350, 353, 356, 358, 359, 360], 
            [361, 362, 364, 367, 370, 375, 380.]]
    pblm, x_mesh, y_mesh, z_mesh = get_3d_geom(
        nb_div_pmat_x, nb_div_pmat_y, z_mesh)
    macrolib = mat.Macrolib(xs_aiea3d_pert_mat, pblm)
    return macrolib, x_mesh, y_mesh, z_mesh
