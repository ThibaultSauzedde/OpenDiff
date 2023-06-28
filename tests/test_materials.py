import pytest
import numpy as np
import numpy.testing as npt

import opendiff.materials as mat


def test_materials(xs_aiea3d):
    all_mat, middles, isot_reac_names = xs_aiea3d
    materials = {mat_name: mat.Material(
        values, isot_reac_names) for mat_name, values in all_mat.items()}
    middles = mat.Middles(materials, middles)
    # print(mat_lib.getReacNames())
    assert middles.getReacNames() == {
        '2', '1', 'SIGF', 'CHI', 'NU_SIGF', 'SIGR', 'EFISS', 'NU', 'SIGA', 'D'}

    fuel1 = np.array([[1.5000000e+00, 1.0000000e-02, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00,
                       2.0000000e-02, 3.2364036e-11, 2.4000000e+00, 3.0000000e-02, 0.0000000e+00],
                      [4.0000000e-01, 8.5000000e-02, 1.3500000e-01, 0.0000000e+00, 0.0000000e+00,
                       0.0000000e+00, 3.2364036e-11, 2.4000000e+00, 8.5000000e-02, 5.6250000e-02]])

    npt.assert_almost_equal(materials["fuel1"].getValues(), fuel1)

    mat_dict_ref = {'fuel1': np.array([[1.5000000e+00, 1.0000000e-02, 0.0000000e+00, 1.0000000e+00,
                                        3.2364036e-11, 2.4000000e+00, 0.0000000e+00, 2.0000000e-02,
                                        3.0000000e-02, 0.0000000e+00],
                                       [4.0000000e-01, 8.5000000e-02, 1.3500000e-01, 0.0000000e+00,
                                        3.2364036e-11, 2.4000000e+00, 0.0000000e+00, 0.0000000e+00,
                                        8.5000000e-02, 5.6250000e-02]]),
                    'fuel1_cr': np.array([[1.5000000e+00, 1.0000000e-02, 0.0000000e+00, 1.0000000e+00,
                                           3.2364036e-11, 2.4000000e+00, 0.0000000e+00, 2.0000000e-02,
                                           3.0000000e-02, 0.0000000e+00],
                                          [4.0000000e-01, 1.3000000e-01, 1.3500000e-01, 0.0000000e+00,
                                           3.2364036e-11, 2.4000000e+00, 0.0000000e+00, 0.0000000e+00,
                                           1.3000000e-01, 5.6250000e-02]]),
                    'fuel2': np.array([[1.5000000e+00, 1.0000000e-02, 0.0000000e+00, 1.0000000e+00,
                                        3.2364036e-11, 2.4000000e+00, 0.0000000e+00, 2.0000000e-02,
                                        3.0000000e-02, 0.0000000e+00],
                                       [4.0000000e-01, 8.0000000e-02, 1.3500000e-01, 0.0000000e+00,
                                        3.2364036e-11, 2.4000000e+00, 0.0000000e+00, 0.0000000e+00,
                                        8.0000000e-02, 5.6250000e-02]]),
                    'refl': np.array([[2.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                                       3.2364036e-11, 2.4000000e+00, 0.0000000e+00, 4.0000000e-02,
                                       4.0000000e-02, 0.0000000e+00],
                                      [3.0000000e-01, 1.0000000e-02, 0.0000000e+00, 0.0000000e+00,
                                       3.2364036e-11, 2.4000000e+00, 0.0000000e+00, 0.0000000e+00,
                                       1.0000000e-02, 0.0000000e+00]]),
                    'refl_cr': np.array([[2.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                                          3.2364036e-11, 2.4000000e+00, 0.0000000e+00, 4.0000000e-02,
                                          4.0000000e-02, 0.0000000e+00],
                                         [3.0000000e-01, 5.5000000e-02, 0.0000000e+00, 0.0000000e+00,
                                          3.2364036e-11, 2.4000000e+00, 0.0000000e+00, 0.0000000e+00,
                                          5.5000000e-02, 0.0000000e+00]]),
                    'void': np.array([[1.0000000e+10, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                                       3.2364036e-11, 2.4000000e+00, 0.0000000e+00, 0.0000000e+00,
                                       0.0000000e+00, 0.0000000e+00],
                                      [1.0000000e+10, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                                       3.2364036e-11, 2.4000000e+00, 0.0000000e+00, 0.0000000e+00,
                                       0.0000000e+00, 0.0000000e+00]])}
    for key, value in mat_dict_ref.items():
        assert key in materials.keys()
        # print(key)
        # print(repr(mat_dict[key]))
        # npt.assert_almost_equal(middles[key].getValues(), value)

    assert middles.getXsValue("fuel1", 1, 'SIGR') == pytest.approx(0.03)
    assert middles.getXsValue("fuel1", 2, 'SIGR') == pytest.approx(0.085)

    assert middles.getNbGroups() == 2

    with pytest.raises(ValueError) as e_info:
        print(middles.getXsValue("fuel1", 3, 'SIGR'))
        print(middles.getXsValue("fuel1", 3, 'SIGR'))

    # assert middles.getReactionIndex("D") == 0
    # assert middles.getReactionIndex("CHI") == 3

def test_materials_random(xs_aiea3d):
    all_mat, middles, isot_reac_names = xs_aiea3d
    materials = {mat_name: mat.Material(
        values, isot_reac_names) for mat_name, values in all_mat.items()}
    middles = mat.Middles(materials, middles)
    middles_pert = mat.Middles(middles)

    xs_value = middles.getXsValue("fuel1", 2, 'SIGA')
    print(xs_value)
    print(middles_pert.getXsValue("fuel1", 2, 'SIGA'))
    print("------------------------------------")
    middles_pert.multXsValue("fuel1", 2, 'SIGA', "ISO",  1.1)
    print(middles.getXsValue("fuel1", 2, 'SIGA'))
    print(middles_pert.getXsValue("fuel1", 2, 'SIGA'))

    print("------------------------------------")
    fuel1 = middles_pert.getMaterials()["fuel1"]
    print(fuel1.getXsValue(2, "ISO", "SIGR"))
    fuel1.multXsValue(2, "ISO", "SIGR", 1.1)
    print(fuel1.getXsValue(2, "ISO", "SIGR"))

    print("------------------------------------")
    middles_pert2 = mat.Middles(middles)
    for mat_name, material in middles_pert2.getMaterials().items():
        print(mat_name, material.getValues())

    middles_pert2.randomPerturbation(["SIGR"], 1.0)
    for mat_name, material in middles_pert2.getMaterials().items():
        print(mat_name, material.getValues())
        print()



def test_macrolib_1d(macrolib_1d):
    macrolib, _ = macrolib_1d
    assert macrolib.getNbGroups() == 2
    assert macrolib.getReacNames() == {
        '2', '1', 'SIGF', 'CHI', 'NU_SIGF', 'SIGR', 'EFISS', 'NU', 'SIGA', 'D'}
    nusif_2 = [[[0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135,
                 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135]]]
    nusif_1 = [[[0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]
    tr_12 = [[[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
               0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]]]

    npt.assert_almost_equal(macrolib.getValues(1, 'NU_SIGF'), nusif_1)
    npt.assert_almost_equal(macrolib.getValues(2, 'NU_SIGF'), nusif_2)
    npt.assert_almost_equal(macrolib.getValues(1, '2'), tr_12)

    npt.assert_almost_equal(
        macrolib.getValues1D(1, 'NU_SIGF'), nusif_1[0][0])
    npt.assert_almost_equal(
        macrolib.getValues1D(2, 'NU_SIGF'), nusif_2[0][0])
    npt.assert_almost_equal(macrolib.getValues1D(1, '2'), tr_12[0][0])


def test_macrolib_2d(macrolib_2d):
    macrolib, _, _ = macrolib_2d
    assert macrolib.getNbGroups() == 2
    assert macrolib.getReacNames() == {
        '2', '1', 'SIGF', 'CHI', 'NU_SIGF', 'SIGR', 'EFISS', 'NU', 'SIGA', 'D'}

    npt.assert_almost_equal(macrolib.getValues(
        1, 'NU_SIGF'), np.zeros((1, 9, 9)))
    nusif_2 = [[[0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.],
                [0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.],
                [0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.],
                [0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0., 0.],
                [0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0., 0.],
                [0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0., 0., 0.],
                [0.135, 0.135, 0.135, 0.135, 0.135, 0., 0., 0., 0.],
                [0.135, 0.135, 0.135, 0., 0., 0., 0., 0., 0.],
                [0.,   0.,    0.,   0.,    0.,   0.,    0.,    0.,    0.]]]
    npt.assert_almost_equal(macrolib.getValues(2, 'NU_SIGF'), nusif_2)

    tr_12 = [[[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
             [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
             [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
             [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04],
             [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04],
             [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04, 0.04],
             [0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04, 0.04, 0.04],
             [0.02, 0.02, 0.02, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
             [0.04,   0.04,    0.04,   0.04,    0.04,   0.04,    0.04,    0.04,    0.04]]]
    npt.assert_almost_equal(macrolib.getValues(1, '2'), tr_12)

    npt.assert_almost_equal(macrolib.getValues1D(
        2, 'NU_SIGF'), np.array(nusif_2).reshape(-1))
    npt.assert_almost_equal(macrolib.getValues1D(
        1, '2'), np.array(tr_12).reshape(-1))


def test_macrolib_3d(macrolib_3d):
    macrolib, _, _, _ = macrolib_3d
    assert macrolib.getNbGroups() == 2
    assert macrolib.getReacNames() == {
        '2', '1', 'SIGF', 'CHI', 'NU_SIGF', 'SIGR', 'EFISS', 'NU', 'SIGA', 'D'}

    npt.assert_almost_equal(macrolib.getValues(
        1, 'NU_SIGF'), np.zeros((7, 9, 9)))
    nusif_2 = np.array([[[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                        [[0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.],
                         [0.135, 0.135, 0.135, 0.135, 0.135,
                             0.135, 0.135, 0.135, 0.],
                         [0.135, 0.135, 0.135, 0.135, 0.135,
                             0.135, 0.135, 0.135, 0.],
                         [0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0., 0.],
                         [0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0., 0.],
                         [0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0., 0., 0.],
                         [0.135, 0.135, 0.135, 0.135, 0.135, 0., 0., 0., 0.],
                         [0.135, 0.135, 0.135, 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                        [[0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.],
                         [0.135, 0.135, 0.135, 0.135, 0.135,
                             0.135, 0.135, 0.135, 0.],
                         [0.135, 0.135, 0.135, 0.135, 0.135,
                             0.135, 0.135, 0.135, 0.],
                         [0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0., 0.],
                         [0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0., 0.],
                         [0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0., 0., 0.],
                         [0.135, 0.135, 0.135, 0.135, 0.135, 0., 0., 0., 0.],
                         [0.135, 0.135, 0.135, 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                        [[0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.],
                         [0.135, 0.135, 0.135, 0.135, 0.135,
                             0.135, 0.135, 0.135, 0.],
                         [0.135, 0.135, 0.135, 0.135, 0.135,
                             0.135, 0.135, 0.135, 0.],
                         [0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0., 0.],
                         [0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0., 0.],
                         [0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0., 0., 0.],
                         [0.135, 0.135, 0.135, 0.135, 0.135, 0., 0., 0., 0.],
                         [0.135, 0.135, 0.135, 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                        [[0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.],
                         [0.135, 0.135, 0.135, 0.135, 0.135,
                             0.135, 0.135, 0.135, 0.],
                         [0.135, 0.135, 0.135, 0.135, 0.135,
                             0.135, 0.135, 0.135, 0.],
                         [0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0., 0.],
                         [0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0., 0.],
                         [0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0., 0., 0.],
                         [0.135, 0.135, 0.135, 0.135, 0.135, 0., 0., 0., 0.],
                         [0.135, 0.135, 0.135, 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                        [[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                        [[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0.]]])
    npt.assert_almost_equal(macrolib.getValues(2, 'NU_SIGF'), nusif_2)

    tr_12 = np.array([[[0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
                       [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
                       [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
                       [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
                       [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.],
                       [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.],
                       [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0., 0., 0.],
                       [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0., 0., 0.],
                       [0.04, 0.04, 0.04, 0.04, 0., 0., 0., 0., 0.]],

                      [[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
                       [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
                       [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
                       [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04],
                       [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.],
                       [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04, 0.],
                       [0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04, 0., 0.],
                       [0.02, 0.02, 0.02, 0.04, 0.04, 0.04, 0., 0., 0.],
                       [0.04, 0.04, 0.04, 0.04, 0., 0., 0., 0., 0.]],

                      [[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
                       [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
                       [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
                       [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04],
                       [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.],
                       [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04, 0.],
                       [0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04, 0., 0.],
                       [0.02, 0.02, 0.02, 0.04, 0.04, 0.04, 0., 0., 0.],
                       [0.04, 0.04, 0.04, 0.04, 0., 0., 0., 0., 0.]],

                      [[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
                       [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
                       [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
                       [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04],
                       [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.],
                       [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04, 0.],
                       [0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04, 0., 0.],
                       [0.02, 0.02, 0.02, 0.04, 0.04, 0.04, 0., 0., 0.],
                       [0.04, 0.04, 0.04, 0.04, 0., 0., 0., 0., 0.]],

                      [[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
                       [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
                       [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
                       [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04],
                       [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.],
                       [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04, 0.],
                       [0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04, 0., 0.],
                       [0.02, 0.02, 0.02, 0.04, 0.04, 0.04, 0., 0., 0.],
                       [0.04, 0.04, 0.04, 0.04, 0., 0., 0., 0., 0.]],

                      [[0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
                       [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
                       [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
                       [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
                       [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.],
                       [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.],
                       [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0., 0.],
                       [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0., 0., 0.],
                       [0.04, 0.04, 0.04, 0.04, 0., 0., 0., 0., 0.]],

                      [[0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
                       [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
                       [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
                       [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
                       [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.],
                       [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.],
                       [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0., 0.],
                       [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0., 0., 0.],
                       [0.04, 0.04, 0.04, 0.04, 0., 0., 0., 0., 0.]]])
    npt.assert_almost_equal(macrolib.getValues(1, '2'), tr_12)

    npt.assert_almost_equal(macrolib.getValues1D(
        2, 'NU_SIGF'), np.array(nusif_2).reshape(-1))
    npt.assert_almost_equal(macrolib.getValues1D(
        1, '2'), np.array(tr_12).reshape(-1))



def test_modif_control_rod(geom_3d):
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    large_width = 400
    np.set_printoptions(linewidth=large_width)
    pblm, x, y, z = geom_3d
    new_geom = mat.get_geometry_roded(pblm, x, y, z, [(40, 60, 60, 80, 20, 360)], "fuel1_cr", "fuel1", [280])
    new_geom = np.array(new_geom)
    print(pblm[3])
    print(new_geom[3])
    import ipdb; ipdb.set_trace()
