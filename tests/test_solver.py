import pytest
import numpy as np
import numpy.testing as npt
import scipy

import opendiff.solver as solver
from opendiff import set_log_level, log_level


def test_solverPI_1d(macrolib_1d, datadir):
    set_log_level(log_level.debug)
    macrolib, x_mesh = macrolib_1d
    # np.savetxt("/home/ts249161/dev/these/openDiff/tests/test_solver/ev_1d.txt", ref_eigenvector)
    ref_eigenvector = np.loadtxt(datadir / "ev_1d.txt")

    ref_eigenvalue = 0.5513156
    s = solver.SolverPowerIt(x_mesh, macrolib, -1., -1.)

    s.solve(inner_solver="SparseLU")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

    # s.solve(inner_solver="LeastSquaresConjugateGradient")
    # assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    # npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
    #                         decimal=4)

    s.solve(inner_solver="BiCGSTAB")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

    s.solve(inner_solver="BiCGSTAB", inner_precond="IncompleteLUT")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

    s.solve(inner_solver="GMRES")
    assert 0.5513096713596358 == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)


def test_solverSlepc_1d(macrolib_1d, datadir):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh = macrolib_1d
    # np.savetxt("/home/ts249161/dev/these/openDiff/tests/test_solver/ev_slepc_1d.txt", ref_eigenvector)
    ref_eigenvector = np.loadtxt(datadir / "ev_slepc_1d.txt")

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


def test_solverPI_2d(macrolib_2d, datadir):
    set_log_level(log_level.debug)
    macrolib, x_mesh, y_mesh = macrolib_2d

    ref_eigenvalue = 1.0256210451968997

    ref_eigenvector = np.loadtxt(datadir / "ev_2d.txt")
    s = solver.SolverPowerIt(x_mesh, y_mesh, macrolib, 1., -1., 1., -1.)

    s.solve(inner_solver="SparseLU")
    # print(s.getEigenValues())
    # print(s.getEigenVectors()[0])
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-5)
    # np.savetxt(
    #     "/home/ts249161/dev/these/openDiff/tests/test_solver/ev_2d.txt", s.getEigenVectors()[0])
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

    s.solve(inner_solver="LeastSquaresConjugateGradient")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-5)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

    s.solve(inner_solver="BiCGSTAB")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-5)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

    s.solve(inner_solver="BiCGSTAB", inner_precond="IncompleteLUT")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-5)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

    s.solve(inner_solver="GMRES")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-4)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=3)


def test_solverSlepc_2d(macrolib_2d, datadir):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh, y_mesh = macrolib_2d

    # np.savetxt("/home/ts249161/dev/these/openDiff/tests/test_solver/ev_slepc_2d.txt", ref_eigenvector)
    ref_eigenvector = np.loadtxt(datadir / "ev_slepc_2d.txt")
    # delta_pi = (100*(ref_eigenvector-ref_eigenvector_pi)/ref_eigenvector_pi).reshape((len(y_mesh)-1, len(x_mesh)-1, macrolib.getNbGroups()))
    # print(delta_pi[:, :, 0])
    # print(delta_pi[:, :, 1])

    ref_eigenvalue = 1.0256209309983306  # 1 pcm de diff avec pi
    s = solver.SolverSlepc(x_mesh, y_mesh, macrolib, 1., -1., 1., -1.)
    s.solve()
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

    # not convertged enough to get he same results
    # # s.solve(solver="power") not working
    # # assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    # # npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
    # #                         decimal=4)

    # s.solve(solver="arnoldi")
    # assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    # npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
    #                         decimal=4)

    # # s.solve(solver="arpack") arpack is missing
    # # assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    # # npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
    # #                         decimal=4)

    # s.solve(inner_solver="ibcgs")
    # assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    # npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
    #                         decimal=4)


def test_solverPI_3d(macrolib_3d, datadir):
    set_log_level(log_level.debug)
    macrolib, x_mesh, y_mesh, z_mesh = macrolib_3d
    ref_eigenvalue = 1.1151426441284367
    ref_eigenvector = np.loadtxt(datadir / "ev_3d.txt")
    s = solver.SolverPowerIt(x_mesh, y_mesh, z_mesh,
                             macrolib, 1., 0., 1., 0., 0., 0.)
    print(s.getVolumes())
    s.solve(inner_solver="SparseLU")
    # np.set_printoptions(threshold=100000, edgeitems=10, linewidth=140)
    # print(s.getEigenValues())
    # print(repr(s.getEigenVectors()[0]))
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    # np.savetxt(
    #     "/home/ts249161/dev/these/openDiff/tests/test_solver/ev_3d.txt", s.getEigenVectors()[0])
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

    # s.solve(inner_solver="LeastSquaresConjugateGradient")
    # print(s.getEigenValues()[0])
    # assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    # npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
    #                         decimal=4)

    # s.solve(inner_solver="BiCGSTAB", outer_max_iter=1000)
    # assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)

    s.solve(inner_solver="BiCGSTAB", inner_precond="IncompleteLUT")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)

    s.solve(inner_solver="GMRES")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-4)


def test_solverSlepc_3d(macrolib_3d, datadir):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh, y_mesh, z_mesh = macrolib_3d
    # np.savetxt("/home/ts249161/dev/these/openDiff/tests/test_solver/ev_slepc_3d.txt", ref_eigenvector)
    ref_eigenvector = np.loadtxt(datadir / "ev_slepc_3d.txt")
    ref_eigenvalue = 1.1151426949100507
    s = solver.SolverSlepc(x_mesh, y_mesh, z_mesh,
                           macrolib, 1., 0., 1., 0., 0., 0.)
    s.solve()
    # np.set_printoptions(threshold=100000, edgeitems=10, linewidth=140)
    # print(s.getEigenValues())
    # print(repr(s.getEigenVectors()[0]))
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    ev_0 = s.getEigenVectors()[0]
    npt.assert_almost_equal(ev_0, ref_eigenvector,
                            decimal=4)

    s.solve(solver="arnoldi")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)

    s.solve(inner_solver="ibcgs")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)

    # Test power
    # np.savetxt(
    #     "/home/ts249161/dev/these/openDiff/tests/test_solver/power_3d.txt", power_ref.reshape(-1))
    power_ref = np.loadtxt(datadir / "power_3d.txt")

    power = s.getPower()
    power_sum_0 = power.sum()
    # print(repr(power))
    npt.assert_almost_equal(power.reshape(-1), power_ref,
                            decimal=4)

    s.normPower()
    power = s.getPower()
    power_sum_1 = power.sum()
    assert power_sum_1 == pytest.approx(1, abs=1e-6)
    ev0 = s.getEigenVector(0)
    assert ev0.shape == (7, 9, 9, 2)
    npt.assert_almost_equal(power_sum_0 / power_sum_1 * s.getEigenVectors()[0],  ev_0,
                            decimal=4)

    s_star = solver.SolverSlepc(s)
    s_star.makeAdjoint()
    s_star.solve()
    s.normPhiStarMPhi(s_star)
    ev0 = s.getEigenVectors()[0]
    ev0_star = s_star.getEigenVectors()[0]
    M = s.getM()
    assert ev0_star.dot(M.dot(ev0)) == pytest.approx(1, abs=1e-6)
    
@pytest.mark.integtest
def test_solverPI_3d_refine_lu(macrolib_3d_refine):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh, y_mesh, z_mesh = macrolib_3d_refine
    s = solver.SolverPowerIt(x_mesh, y_mesh, z_mesh, macrolib, 1., 0., 1., 0., 0., 0.)
    s.solve(inner_solver="SparseLU")

@pytest.mark.integtest
def test_solverPI_3d_refine_BiCGSTAB(macrolib_3d_refine):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh, y_mesh, z_mesh = macrolib_3d_refine
    s = solver.SolverPowerIt(x_mesh, y_mesh, z_mesh, macrolib, 1., 0., 1., 0., 0., 0.)
    s.solve(inner_solver="BiCGSTAB")

@pytest.mark.integtest
def test_solverPI_3d_refine_BiCGSTAB_lu(macrolib_3d_refine):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh, y_mesh, z_mesh = macrolib_3d_refine
    s = solver.SolverPowerIt(x_mesh, y_mesh, z_mesh, macrolib, 1., 0., 1., 0., 0., 0.)
    s.solve(inner_solver="BiCGSTAB", inner_precond="IncompleteLUT")

@pytest.mark.integtest
def test_solverPI_3d_refine_GMRES(macrolib_3d_refine):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh, y_mesh, z_mesh = macrolib_3d_refine
    s = solver.SolverPowerIt(x_mesh, y_mesh, z_mesh, macrolib, 1., 0., 1., 0., 0., 0.)
    s.solve(inner_solver="GMRES")

@pytest.mark.integtest
def test_solverSlepc_3d_refine(macrolib_3d_refine):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh, y_mesh, z_mesh = macrolib_3d_refine
    s = solver.SolverSlepc(x_mesh, y_mesh, z_mesh, macrolib, 1., 0., 1., 0., 0., 0.)
    s.solve(outer_max_iter=10000, inner_max_iter=200, tol=1e-6)

@pytest.mark.integtest
def test_solverSlepc_3d_refine_arnoldi(macrolib_3d_refine):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh, y_mesh, z_mesh = macrolib_3d_refine
    s = solver.SolverSlepc(x_mesh, y_mesh, z_mesh, macrolib, 1., 0., 1., 0., 0., 0.)
    s.solve(solver="arnoldi", outer_max_iter=10000, inner_max_iter=200, tol=1e-6)

@pytest.mark.integtest
def test_solverSlepc_3d_refine_ibcgs(macrolib_3d_refine):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh, y_mesh, z_mesh = macrolib_3d_refine
    s = solver.SolverSlepc(x_mesh, y_mesh, z_mesh, macrolib, 1., 0., 1., 0., 0., 0.)
    s.solve(inner_solver="ibcgs", outer_max_iter=10000, inner_max_iter=200, tol=1e-6)
