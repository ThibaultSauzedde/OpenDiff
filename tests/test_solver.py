import pytest
import numpy as np
import numpy.testing as npt
import scipy

import opendiff.solver as solver
from opendiff import set_log_level, log_level


def test_solverPI_1d(macrolib_1d):
    set_log_level(log_level.debug)
    macrolib, x_mesh = macrolib_1d
    ref_eigenvector = np.array([0.02422983, 0.00530779, 0.07209338, 0.01579275, 0.11818324,
                                0.02588911, 0.16136532, 0.03534839, 0.20057686, 0.04393778,
                                0.23485256, 0.05144582, 0.26334835, 0.05768763, 0.28536217,
                                0.06250944, 0.30035136, 0.06579242, 0.30794603, 0.06745558,
                                0.30795833, 0.06745782, 0.30038708, 0.06579893, 0.2854179,
                                0.06251959, 0.26341873, 0.05770044, 0.23493079, 0.05146005,
                                0.20065531, 0.04395203, 0.16143628, 0.03536127, 0.11823968,
                                0.02589934, 0.07212968, 0.01579933, 0.02424235, 0.00531005])
    ref_eigenvalue = 0.5513156
    s = solver.SolverPowerIt(x_mesh, macrolib, -1., -1.)

    s.solve(inner_solver="SparseLU")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

    s.solve(inner_solver="LeastSquaresConjugateGradient")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

    s.solve(inner_solver="BiCGSTAB")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

    s.solve(inner_solver="BiCGSTAB", inner_precond="IncompleteLUT")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

    s.solve(inner_solver="GMRES")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)


def test_solverSlepc_1d(macrolib_1d):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh = macrolib_1d
    ref_eigenvector = np.array([0.02422983, 0.00530779, 0.07209338, 0.01579275, 0.11818324,
                                0.02588911, 0.16136532, 0.03534839, 0.20057686, 0.04393778,
                                0.23485256, 0.05144582, 0.26334835, 0.05768763, 0.28536217,
                                0.06250944, 0.30035136, 0.06579242, 0.30794603, 0.06745558,
                                0.30795833, 0.06745782, 0.30038708, 0.06579893, 0.2854179,
                                0.06251959, 0.26341873, 0.05770044, 0.23493079, 0.05146005,
                                0.20065531, 0.04395203, 0.16143628, 0.03536127, 0.11823968,
                                0.02589934, 0.07212968, 0.01579933, 0.02424235, 0.00531005])
    ref_eigenvalue = 0.5513156
    s = solver.SolverSlepc(x_mesh, macrolib, -1., -1.)
    s.solve()
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

    # s.solve(solver="power") not working
    # assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    # npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
    #                         decimal=4)

    s.solve(solver="arnoldi")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

    # s.solve(solver="arpack") arpack is missing
    # assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    # npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
    #                         decimal=4)

    s.solve(inner_solver="ibcgs")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)
