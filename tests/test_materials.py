import pytest
import numpy as np
import numpy.testing as npt

import opendiff.materials as mat


def test_materials(xs_aiea3d):
    all_mat, mat_names, reac_names = xs_aiea3d
    mat_lib = mat.Materials(all_mat, mat_names, reac_names)
    assert mat_lib.getReacNames() == [
        'D', 'SIGA', 'NU_SIGF', 'CHI', '1', '2', 'SIGR']

    assert mat_lib.getMatNames() == mat_names

    fuel1 = np.array([[1.5,   0.01,  0.,    1.,    0.,    0.02,  0.03],
                      [0.4,   0.085, 0.135, 0.,    0.,    0.,    0.085]])
    npt.assert_almost_equal(mat_lib.getMaterial("fuel1"), fuel1)

    mat_dict = mat_lib.getMaterials()
    mat_dict_ref = {'fuel1': np.array([[1.5, 0.01, 0., 1., 0., 0.02, 0.03],
                                       [0.4, 0.085, 0.135, 0., 0., 0., 0.085]]),
                    'fuel1_cr': np.array([[1.5, 0.01, 0., 1., 0., 0.02, 0.03],
                                          [0.4, 0.13, 0.135, 0., 0., 0., 0.13]]),
                    'fuel2': np.array([[1.5, 0.01, 0., 1., 0., 0.02, 0.03],
                                       [0.4, 0.08, 0.135, 0., 0., 0., 0.08]]),
                    'refl': np.array([[2., 0., 0., 0., 0., 0.04, 0.04],
                                      [0.3, 0.01, 0., 0., 0., 0., 0.01]]),
                    'refl_cr': np.array([[2., 0., 0., 0., 0., 0.04, 0.04],
                                         [0.3, 0.055, 0., 0., 0., 0., 0.055]]),
                    'void': np.array([[1.e+10, 0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00],
                                      [1.e+10, 0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00, 0.e+00]])}
    for key, value in mat_dict_ref.items():
        assert key in mat_dict.keys()
        npt.assert_almost_equal(mat_dict[key], value)

    assert mat_lib.getValue("fuel1", 0, 'SIGR') == pytest.approx(0.03)
    assert mat_lib.getValue("fuel1", 1, 'SIGR') == pytest.approx(0.085)

    assert mat_lib.getNbGroups() == 2

    # waiting for tests in openDiff
    # with pytest.raises(KeyError) as e_info:
    #     print(mat_lib.getValue("fuel1", 2, 'SIGR'))
    #     print(mat_lib.getValue("fuel1", 2, 'SIGR'))

    assert mat_lib.getReactionIndex("D") == 0
    assert mat_lib.getReactionIndex("CHI") == 3

    mat_lib.addMaterial(all_mat[0], "fuel1_mix", [
                        'NU_SIGF', 'CHI', '1', '2', 'D', 'SIGA', ])

    npt.assert_almost_equal(mat_lib.getMaterial("fuel1_mix"), [[0.,    1.,    0.,    0.02,  1.5,   0.01,  1.01],
                                                               [0.135, 0.,    0.,    0.,    0.4,   0.085, 0.4]])


def test_macrolib_1d(macrolib_1d):
    macrolib, _ = macrolib_1d
    assert macrolib.getNbGroups() == 2
    assert macrolib.getReacNames() == [
        'D', 'SIGA', 'NU_SIGF', 'CHI', '1', '2', 'SIGR']

    nusif_2 = [[[0.135, 0.135, 0.135, 0.135, 0.135]]]
    nusif_1 = [[[0., 0., 0., 0., 0.]]]
    tr_12 = [[[0.02, 0.02, 0.02, 0.02, 0.02]]]

    npt.assert_almost_equal(macrolib.getValues(1, 'NU_SIGF'), nusif_1)
    npt.assert_almost_equal(macrolib.getValues(2, 'NU_SIGF'), nusif_2)
    npt.assert_almost_equal(macrolib.getValues(1, '2'), tr_12)

    npt.assert_almost_equal(macrolib.getValues1D(1, 'NU_SIGF'), nusif_1[0][0])
    npt.assert_almost_equal(macrolib.getValues1D(2, 'NU_SIGF'), nusif_2[0][0])
    npt.assert_almost_equal(macrolib.getValues1D(1, '2'), tr_12[0][0])


# def test_macrolib_2d(macrolib_2d):


# def test_macrolib_3d(macrolib_3d):
