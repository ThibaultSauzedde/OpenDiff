import pytest
import numpy as np
import numpy.testing as npt

import opendiff.materials as mat


def test_materials(xs_aiea3d):
    all_mat, mat_names, reac_names = xs_aiea3d
    mat_lib = mat.Materials(all_mat, mat_names, reac_names)
    # print(mat_lib.getReacNames())
    assert mat_lib.getReacNames() == [
        'D', 'SIGA', 'NU_SIGF', 'CHI', 'EFISS', 'NU', '1', '2', 'SIGR', 'SIGF']

    assert mat_lib.getMatNames() == mat_names

    fuel1 = np.array([[1.5,   0.01,  0.,    1., 202 * 1.60218e-19 * 1e6, 2.4,    0.,    0.02,  0.03, 0.],
                      [0.4,   0.085, 0.135, 0., 202 * 1.60218e-19 * 1e6, 2.4,    0.,    0.,    0.085, 5.625e-02]])
    print(mat_lib.getMaterial("fuel1"))
    npt.assert_almost_equal(mat_lib.getMaterial("fuel1"), fuel1)

    mat_dict = mat_lib.getMaterials()
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
        assert key in mat_dict.keys()
        # print(key)
        # print(repr(mat_dict[key]))
        npt.assert_almost_equal(mat_dict[key], value)

    assert mat_lib.getValue("fuel1", 1, 'SIGR') == pytest.approx(0.03)
    assert mat_lib.getValue("fuel1", 2, 'SIGR') == pytest.approx(0.085)

    assert mat_lib.getNbGroups() == 2

    with pytest.raises(ValueError) as e_info:
        print(mat_lib.getValue("fuel1", 3, 'SIGR'))
        print(mat_lib.getValue("fuel1", 3, 'SIGR'))

    assert mat_lib.getReactionIndex("D") == 0
    assert mat_lib.getReactionIndex("CHI") == 3

    mat_lib.addMaterial(all_mat[0], "fuel1_mix", [
                        'NU_SIGF', 'CHI', '1', '2', 'D', 'SIGA', 'EFISS', 'NU'])

    # print(repr(mat_lib.getMaterial("fuel1_mix")))
    npt.assert_almost_equal(mat_lib.getMaterial("fuel1_mix"), [[0.00000000e+00, 1.00000000e+00, 3.23640360e-11, 2.40000000e+00,
                                                                1.50000000e+00, 1.00000000e-02, 0.00000000e+00, 2.00000000e-02,
                                                                1.02000000e+00, 3.23640360e-09],
                                                               [1.35000000e-01, 0.00000000e+00, 3.23640360e-11, 2.40000000e+00,
                                                                4.00000000e-01, 8.50000000e-02, 0.00000000e+00, 0.00000000e+00,
                                                                0.00000000e+00, 3.80753365e-10]])


def test_macrolib_1d(macrolib_1d):
    macrolib, _ = macrolib_1d
    assert macrolib.getNbGroups() == 2
    assert macrolib.getReacNames() == [
        'D', 'SIGA', 'NU_SIGF', 'CHI', 'EFISS', 'NU', '1', '2', 'SIGR', 'SIGF']

    nusif_2 = [[[0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135]]]
    nusif_1 = [[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]
    tr_12 = [[[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]]]

    npt.assert_almost_equal(macrolib.getValues(1, 'NU_SIGF'), nusif_1)
    npt.assert_almost_equal(macrolib.getValues(2, 'NU_SIGF'), nusif_2)
    npt.assert_almost_equal(macrolib.getValues(1, '2'), tr_12)

    npt.assert_almost_equal(macrolib.getValues1D(1, 'NU_SIGF'), nusif_1[0][0])
    npt.assert_almost_equal(macrolib.getValues1D(2, 'NU_SIGF'), nusif_2[0][0])
    npt.assert_almost_equal(macrolib.getValues1D(1, '2'), tr_12[0][0])


def test_macrolib_2d(macrolib_2d):
    macrolib, _, _ = macrolib_2d
    assert macrolib.getNbGroups() == 2
    assert macrolib.getReacNames() == [
        'D', 'SIGA', 'NU_SIGF', 'CHI', 'EFISS', 'NU', '1', '2', 'SIGR', 'SIGF']


    npt.assert_almost_equal(macrolib.getValues(1, 'NU_SIGF'), np.zeros((1, 9, 9)))
    nusif_2 = [[[0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.],
                [0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.],
                [0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.],
                [0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.   , 0.],
                [0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.   , 0.],
                [0.135, 0.135, 0.135, 0.135, 0.135, 0.135, 0.   , 0.   , 0.],
                [0.135, 0.135, 0.135, 0.135, 0.135, 0.   , 0.   , 0.   , 0.],
                [0.135, 0.135, 0.135, 0.   , 0.   , 0.   , 0.   , 0.   , 0.],
                [0. ,   0.,    0. ,   0.,    0. ,   0.,    0.,    0.,    0.]]]
    npt.assert_almost_equal(macrolib.getValues(2, 'NU_SIGF'), nusif_2)

    tr_12 = [[[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
             [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
             [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
             [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04  , 0.04],
             [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04  , 0.04],
             [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04  , 0.04  , 0.04],
             [0.02, 0.02, 0.02, 0.02, 0.02, 0.04  , 0.04  , 0.04  , 0.04],
             [0.02, 0.02, 0.02, 0.04  , 0.04  , 0.04  , 0.04  , 0.04  , 0.04],
             [0.04,   0.04,    0.04,   0.04,    0.04,   0.04,    0.04,    0.04,    0.04]]]
    npt.assert_almost_equal(macrolib.getValues(1, '2'), tr_12)

    npt.assert_almost_equal(macrolib.getValues1D(2, 'NU_SIGF'), np.array(nusif_2).reshape(-1))
    npt.assert_almost_equal(macrolib.getValues1D(1, '2'), np.array(tr_12).reshape(-1))

def test_macrolib_3d(macrolib_3d):
    macrolib, _, _, _ = macrolib_3d
    assert macrolib.getNbGroups() == 2
    assert macrolib.getReacNames() == [
        'D', 'SIGA', 'NU_SIGF', 'CHI', 'EFISS', 'NU', '1', '2', 'SIGR', 'SIGF']

    npt.assert_almost_equal(macrolib.getValues(1, 'NU_SIGF'), np.zeros((7, 9, 9)))
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
        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.  ],
        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.  ],
        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.  , 0.  , 0.  ],
        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.  , 0.  , 0.  ],
        [0.04, 0.04, 0.04, 0.04, 0.  , 0.  , 0.  , 0.  , 0.  ]],

       [[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.  ],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04, 0.  ],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04, 0.  , 0.  ],
        [0.02, 0.02, 0.02, 0.04, 0.04, 0.04, 0.  , 0.  , 0.  ],
        [0.04, 0.04, 0.04, 0.04, 0.  , 0.  , 0.  , 0.  , 0.  ]],

       [[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.  ],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04, 0.  ],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04, 0.  , 0.  ],
        [0.02, 0.02, 0.02, 0.04, 0.04, 0.04, 0.  , 0.  , 0.  ],
        [0.04, 0.04, 0.04, 0.04, 0.  , 0.  , 0.  , 0.  , 0.  ]],

       [[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.  ],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04, 0.  ],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04, 0.  , 0.  ],
        [0.02, 0.02, 0.02, 0.04, 0.04, 0.04, 0.  , 0.  , 0.  ],
        [0.04, 0.04, 0.04, 0.04, 0.  , 0.  , 0.  , 0.  , 0.  ]],

       [[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.  ],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04, 0.  ],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.04, 0.  , 0.  ],
        [0.02, 0.02, 0.02, 0.04, 0.04, 0.04, 0.  , 0.  , 0.  ],
        [0.04, 0.04, 0.04, 0.04, 0.  , 0.  , 0.  , 0.  , 0.  ]],

       [[0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.  ],
        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.  ],
        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.  , 0.  ],
        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.  , 0.  , 0.  ],
        [0.04, 0.04, 0.04, 0.04, 0.  , 0.  , 0.  , 0.  , 0.  ]],

       [[0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.  ],
        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.  ],
        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.  , 0.  ],
        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.  , 0.  , 0.  ],
        [0.04, 0.04, 0.04, 0.04, 0.  , 0.  , 0.  , 0.  , 0.  ]]])
    npt.assert_almost_equal(macrolib.getValues(1, '2'), tr_12)

    npt.assert_almost_equal(macrolib.getValues1D(2, 'NU_SIGF'), np.array(nusif_2).reshape(-1))
    npt.assert_almost_equal(macrolib.getValues1D(1, '2'), np.array(tr_12).reshape(-1))
