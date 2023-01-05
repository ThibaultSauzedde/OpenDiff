import pytest
import numpy as np
import numpy.testing as npt
import scipy

import opendiff.solver as solver
from opendiff import set_log_level, log_level


def test_remove_ev(macrolib_1d, datadir):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh = macrolib_1d
    s = solver.SolverFullSlepc(x_mesh, macrolib, -1., -1.)
    s.solve(nb_eigen_values=10)

    ref_ev = [0.5513156303206834, 0.19788648486212612, 0.0801425345747261, 0.03723735959701421, 0.019399267510645364, 0.011102238300568805,
              0.006868208610792799, 0.00453559108479465, 0.0031663713907822784, 0.0023192887514495615, 0.0017719996736606217]

    assert len(s.getEigenValues()) == 11
    assert len(s.getEigenVectors()) == 11
    npt.assert_almost_equal(s.getEigenValues(), ref_ev, decimal=5)

    s.removeEigenVectors([0])
    npt.assert_almost_equal(s.getEigenValues(), ref_ev[1:], decimal=5)
    assert len(s.getEigenVectors()) == 10

    s.solve(nb_eigen_values=10)
    s.removeEigenVectors([0, 1, 2])
    npt.assert_almost_equal(s.getEigenValues(), ref_ev[3:], decimal=5)
    assert len(s.getEigenVectors()) == 8

    s.solve(nb_eigen_values=10)
    s.removeEigenVectors([0, 3, 5, 8, 9])
    npt.assert_almost_equal(
        s.getEigenValues(), np.array(ref_ev)[[1, 2, 4, 6, 7, 10]], decimal=5)
    assert len(s.getEigenVectors()) == 6


def test_isOrthogonal(macrolib_1d, datadir):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh = macrolib_1d
    s = solver.SolverFullSlepc(x_mesh, macrolib, -1., -1.)
    s.solve(nb_eigen_values=10)
    assert s.isOrthogonal() == True

    s.solve(nb_eigen_values=20)
    assert s.isOrthogonal() == False


def test_handleDenegeratedEigenvalues(macrolib_1d_refine, datadir):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh = macrolib_1d_refine
    s = solver.SolverFullSlepc(x_mesh, macrolib, -1., -1.)
    s.solve(nb_eigen_values=60)
    # print(s.isOrthogonal())

    s.handleDenegeratedEigenvalues()  # todo: find a test which need this function
    # print(s.isOrthogonal())

    # s.solve(nb_eigen_values=20)
    # assert s.handleDenegeratedEigenvalues() == False


def test_solverPI_1d(macrolib_1d, datadir):
    set_log_level(log_level.debug)
    macrolib, x_mesh = macrolib_1d
    ref_eigenvector = np.loadtxt(datadir / "ev_1d.txt")

    ref_eigenvalue = 0.5513156
    s = solver.SolverFullPowerIt(x_mesh, macrolib, -1., -1.)

    s.solve(inner_solver="SparseLU")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    # np.savetxt("/home/ts249161/dev/these/openDiff/tests/test_solver/ev_1d.txt",
    #            s.getEigenVectors()[0])

    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

    s.solve(inner_solver="LeastSquaresConjugateGradient", outer_max_iter=1000, inner_max_iter=50)
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

    s.solve(inner_solver="BiCGSTAB")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-5)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

    s.solve(inner_solver="BiCGSTAB", inner_precond="IncompleteLUT")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

    s.solve(inner_solver="GMRES")
    assert 0.5513096713596358 == pytest.approx(s.getEigenValues()[0], abs=1e-5)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

def test_solverCondPI_1d(macrolib_1d, datadir):
    set_log_level(log_level.debug)
    macrolib, x_mesh = macrolib_1d
    ref_eigenvector = np.loadtxt(datadir / "ev_1d.txt")
    print(x_mesh)

    ref_eigenvector = np.loadtxt(datadir / "ev_1d.txt")
    ref_eigenvalue = 0.5513156
    s = solver.SolverCondPowerIt(x_mesh, macrolib, -1., -1.)

    s.solve(inner_solver="BiCGSTAB")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-5)
    coeff = s.getEigenVectors()[0][0] / ref_eigenvector[0]
    npt.assert_almost_equal(s.getEigenVectors()[0]/coeff, ref_eigenvector,
                            decimal=4)

    s.solve(inner_solver="LeastSquaresConjugateGradient", outer_max_iter=1000, inner_max_iter=50)
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    npt.assert_almost_equal(s.getEigenVectors()[0]/coeff, ref_eigenvector,
                            decimal=4)

    s.solve(inner_solver="BiCGSTAB", inner_precond="IncompleteLUT")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    npt.assert_almost_equal(s.getEigenVectors()[0]/coeff, ref_eigenvector,
                            decimal=4)

    s.solve(inner_solver="GMRES")
    assert 0.5513096713596358 == pytest.approx(s.getEigenValues()[0], abs=1e-5)
    npt.assert_almost_equal(s.getEigenVectors()[0]/coeff, ref_eigenvector,
                            decimal=4)


def test_SolverFullSlepc_1d(macrolib_1d, datadir):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh = macrolib_1d
    ref_eigenvector = np.loadtxt(datadir / "ev_slepc_1d.txt")

    ref_eigenvalue = 0.5513156
    s = solver.SolverFullSlepc(x_mesh, macrolib, -1., -1.)
    s.solve()
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    # np.savetxt("/home/ts249161/dev/these/openDiff/tests/test_solver/ev_slepc_1d.txt",
    #            s.getEigenVectors()[0])
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
    s = solver.SolverFullPowerIt(x_mesh, y_mesh, macrolib, 1., -1., 1., -1.)

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
                            decimal=3)

    s.solve(inner_solver="BiCGSTAB", outer_max_iter=1000, inner_max_iter=50)
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=3e-5)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=3)

    s.solve(inner_solver="BiCGSTAB", inner_precond="IncompleteLUT")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-5)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

    s.solve(inner_solver="GMRES", outer_max_iter=1000, inner_max_iter=50)
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-4)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

def test_solverCondPI_2d(macrolib_2d, datadir):
    set_log_level(log_level.debug)
    macrolib, x_mesh, y_mesh = macrolib_2d

    ref_eigenvalue = 1.0256210451968997

    ref_eigenvector = np.loadtxt(datadir / "ev_2d.txt")
    s = solver.SolverCondPowerIt(x_mesh, y_mesh, macrolib, 1., -1., 1., -1.)
    s.solve(inner_solver="BiCGSTAB", tol_eigen_vectors=1e-7)

    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-5)
    coeff = s.getEigenVectors()[0][0] / ref_eigenvector[0]
    npt.assert_almost_equal(s.getEigenVectors()[0]/coeff, ref_eigenvector,
                            decimal=4)

    s.solve(inner_solver="BiCGSTAB")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-5)
    npt.assert_almost_equal(s.getEigenVectors()[0]/coeff, ref_eigenvector,
                            decimal=4)

    s.solve(inner_solver="BiCGSTAB", inner_precond="IncompleteLUT")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-5)
    npt.assert_almost_equal(s.getEigenVectors()[0]/coeff, ref_eigenvector,
                            decimal=4)

    s.solve(inner_solver="GMRES", outer_max_iter=1000, inner_max_iter=50)
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-5)
    npt.assert_almost_equal(s.getEigenVectors()[0]/coeff, ref_eigenvector,
                            decimal=4)

                  
def test_SolverFullSlepc_2d(macrolib_2d, datadir):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh, y_mesh = macrolib_2d

    ref_eigenvector = np.loadtxt(datadir / "ev_slepc_2d.txt")
    # delta_pi = (100*(ref_eigenvector-ref_eigenvector_pi)/ref_eigenvector_pi).reshape((len(y_mesh)-1, len(x_mesh)-1, macrolib.getNbGroups()))
    # print(delta_pi[:, :, 0])
    # print(delta_pi[:, :, 1])

    ref_eigenvalue = 1.0256209309983306
    s = solver.SolverFullSlepc(x_mesh, y_mesh, macrolib, 1., -1., 1., -1.)
    s.solve()
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    # np.savetxt("/home/ts249161/dev/these/openDiff/tests/test_solver/ev_slepc_2d.txt", s.getEigenVectors()[0])
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

    # not converged enough to get he same results
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
    s = solver.SolverFullPowerIt(x_mesh, y_mesh, z_mesh,
                             macrolib, 1., 0., 1., 0., 0., 0.)

    s.solve(inner_solver="SparseLU", outer_max_iter=10000)
    # np.set_printoptions(threshold=100000, edgeitems=10, linewidth=140)
    # print(s.getEigenValues())
    # print(repr(s.getEigenVectors()[0]))
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    # np.savetxt(
    #     "/home/ts249161/dev/these/openDiff/tests/test_solver/ev_3d.txt", s.getEigenVectors()[0])
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

    s.solve(inner_solver="LeastSquaresConjugateGradient", outer_max_iter=10000, inner_max_iter=50)
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    # npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
    #                         decimal=4)

    s.solve(inner_solver="BiCGSTAB", outer_max_iter=3000, inner_max_iter=50, tol_inner=1e-5)
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-5)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4) # convergence issue


    s.solve(inner_solver="BiCGSTAB", inner_precond="IncompleteLUT", outer_max_iter=10000, inner_max_iter=50)
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

    s.solve(inner_solver="GMRES", outer_max_iter=1000, inner_max_iter=500)
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-5)
    npt.assert_almost_equal(s.getEigenVectors()[0], ref_eigenvector,
                            decimal=4)

def test_solverCondPI_3d(macrolib_3d, datadir):
    set_log_level(log_level.debug)
    macrolib, x_mesh, y_mesh, z_mesh = macrolib_3d
    ref_eigenvalue = 1.1151426441284367
    ref_eigenvector = np.loadtxt(datadir / "ev_3d.txt")
    s = solver.SolverCondPowerIt(x_mesh, y_mesh, z_mesh,
                             macrolib, 1., 0., 1., 0., 0., 0.)

    s.solve(inner_solver="SparseLU", tol_eigen_vectors=1e-7, outer_max_iter=1000, inner_max_iter=500)
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    coeff = s.getEigenVectors()[0][0] / ref_eigenvector[0]
    npt.assert_almost_equal(s.getEigenVectors()[0]/coeff, ref_eigenvector,
                            decimal=4)
                            
    s.solve(inner_solver="LeastSquaresConjugateGradient", tol_eigen_vectors=1e-7, outer_max_iter=1000, inner_max_iter=500)
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    coeff = s.getEigenVectors()[0][0] / ref_eigenvector[0]
    # npt.assert_almost_equal(s.getEigenVectors()[0]/coeff, ref_eigenvector,
    #                         decimal=4)

    s.solve(inner_solver="BiCGSTAB", tol_inner=1e-7, tol_eigen_vectors=1e-7, outer_max_iter=2000, inner_max_iter=1000)
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-4)
    coeff = s.getEigenVectors()[0][0] / ref_eigenvector[0]
    npt.assert_almost_equal(s.getEigenVectors()[0]/coeff, ref_eigenvector,
                            decimal=4)


    s.solve(inner_solver="BiCGSTAB", inner_precond="IncompleteLUT", tol_eigen_vectors=1e-7, outer_max_iter=1000, inner_max_iter=500)
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    coeff = s.getEigenVectors()[0][0] / ref_eigenvector[0]
    npt.assert_almost_equal(s.getEigenVectors()[0]/coeff, ref_eigenvector,
                            decimal=4)

    # s.solve(inner_solver="GMRES", tol_eigen_vectors=1e-7, outer_max_iter=1000, inner_max_iter=500)
    # assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-5)
    # npt.assert_almost_equal(s.getEigenVectors()[0]/coeff, ref_eigenvector,
    #                         decimal=4) # diverge !!! 

def test_SolverFullSlepc_3d(macrolib_3d, datadir):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh, y_mesh, z_mesh = macrolib_3d
    ref_eigenvector = np.loadtxt(datadir / "ev_slepc_3d.txt")
    ref_eigenvalue = 1.1151426949100507
    s = solver.SolverFullSlepc(x_mesh, y_mesh, z_mesh,
                           macrolib, 1., 0., 1., 0., 0., 0.)
    s.solve()
    # np.set_printoptions(threshold=100000, edgeitems=10, linewidth=140)
    # print(s.getEigenValues())
    # print(repr(s.getEigenVectors()[0]))
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)
    ev_0 = s.getEigenVectors()[0]
    # np.savetxt("/home/ts249161/dev/these/openDiff/tests/test_solver/ev_slepc_3d.txt", ev_0)

    npt.assert_almost_equal(ev_0, ref_eigenvector,
                            decimal=4)

    s.solve(solver="arnoldi")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)

    s.solve(inner_solver="ibcgs")
    assert ref_eigenvalue == pytest.approx(s.getEigenValues()[0], abs=1e-6)

    # Test power
    power_ref = np.loadtxt(datadir / "power_3d.txt")

    power = s.getPower()
    power_sum_0 = power.sum()
    # print(repr(power))
    # np.savetxt(
    #     "/home/ts249161/dev/these/openDiff/tests/test_solver/power_3d.txt", power.reshape(-1))
    npt.assert_almost_equal(power.reshape(-1), power_ref,
                            decimal=4)

    s.normPower()
    power = s.getPower()
    power_sum_1 = power.sum()
    assert power_sum_1 == pytest.approx(1, abs=1e-6)
    ev0 = s.getEigenVector(0)
    assert ev0.shape == (2, 7, 9, 9)
    npt.assert_almost_equal(power_sum_0 / power_sum_1 * s.getEigenVectors()[0],  ev_0,
                            decimal=4)

    s_star = solver.SolverFullSlepc(s)
    s_star.makeAdjoint()
    s_star.solve()
    s.normPhiStarMPhi(s_star)
    ev0 = s.getEigenVectors()[0]
    ev0_star = s_star.getEigenVectors()[0]
    M = s.getM()
    assert ev0_star.dot(M.dot(ev0)) == pytest.approx(1, abs=1e-6)
    
#-------------------------------------------------------------
#Fixed source problem
#-------------------------------------------------------------


def test_solverFixedSource_1d(macrolib_1d, datadir):
    set_log_level(log_level.debug)
    macrolib, x_mesh = macrolib_1d
    ref_eigenvector = np.loadtxt(datadir / "ev_1d.txt")

    ref_eigenvalue = 0.5513156
    s = solver.SolverFullPowerIt(x_mesh, macrolib, -1., -1.)
    s_star = solver.SolverFullPowerIt(s)
    s_star.makeAdjoint()
    s.solve(inner_solver="SparseLU", acceleration="chebyshev")
    s_star.solve(inner_solver="SparseLU", acceleration="chebyshev")
    ev = s.getEigenVector(0)

    source = np.zeros_like(ev)
    # source[0, 0, 0, 2:4] = 0.1
    # source[0, 0, 0, -4:-2] = 0.1
    source[0, 0, 0, 10:12] = 0.1

    source_flat = np.ravel(source)
    s_fixed_source = solver.SolverFullFixedSource(s, s_star, source_flat)
    s_fixed_source.solve(inner_solver="SparseLU", acceleration="chebyshev", outer_max_iter=20)
    gamma = s_fixed_source.getGamma()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(ev[0, 0, 0, :], label="ev0")
    ax.plot(ev[1, 0, 0, :], label="ev1")
    ax.plot(source[0, 0, 0, :], label="src0")
    ax.plot(source[1, 0, 0, :], label="src1")
    ax.plot(gamma[0, 0, 0, :], "--", label="gamma0")
    ax.plot(gamma[1, 0, 0, :], "--", label="gamma1")
    plt.legend()

    source = np.zeros_like(ev)
    source[1, 0, 0, 10:12] = 0.1
    source_flat = np.ravel(source)
    s_fixed_source = solver.SolverFullFixedSource(s, s_star, source_flat)
    s_fixed_source.makeAdjoint()
    s_fixed_source.solve(inner_solver="SparseLU", acceleration="chebyshev", outer_max_iter=20)
    gamma = s_fixed_source.getGamma()
    ev = s_star.getEigenVector(0)

    fig, ax = plt.subplots()
    ax.plot(ev[0, 0, 0, :], label="ev0")
    ax.plot(ev[1, 0, 0, :], label="ev1")
    ax.plot(source[0, 0, 0, :], label="src0")
    ax.plot(source[1, 0, 0, :], label="src1")
    ax.plot(gamma[0, 0, 0, :], "--", label="gamma0")
    ax.plot(gamma[1, 0, 0, :], "--", label="gamma1")
    plt.legend()
    plt.show()


def test_cheb_acc():
    dom_ratio = 0.988
    print()
    for p in range(1, 7):
        if p == 1:
            alpha = 2/(2-dom_ratio)
            beta = 0.
            print(alpha, beta)
        else:
            gamma = np.arccosh(2/dom_ratio-1)
            alpha = 4/dom_ratio * np.cosh((p-1)*gamma) / np.cosh(p*gamma)
            beta = np.cosh((p-2)*gamma) / np.cosh(p*gamma) 
            beta_hebert = (1-dom_ratio/2)*alpha - 1
            print(alpha, beta, beta_hebert, gamma)

    # for p in range(1, 7):
    #     if p == 1:
    #         alpha = 2/dom_ratio
    #         rho = 1/(alpha-1)

    #         r0 = alpha*rho
    #         r1 = rho
    #         print(r0, r1)
    #     else:
    #         rho = 4/(4*(alpha-1)-rho)
    #         alpha = rho/(rho-alpha)
    #         r0 = alpha * r0
    #         r1 = -rho
    #         r2 = 1 - r0 - r1
    #         print(r0, r1, r2)


@pytest.mark.integtest
def test_solverPI_3d_refine_lu(macrolib_3d_refine):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh, y_mesh, z_mesh = macrolib_3d_refine
    s = solver.SolverFullPowerIt(x_mesh, y_mesh, z_mesh, macrolib, 1., 0., 1., 0., 0., 0.)
    s.solve(inner_solver="SparseLU", acceleration="chebyshev")

@pytest.mark.integtest
def test_solverPI_3d_refine_BiCGSTAB(macrolib_3d_refine):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh, y_mesh, z_mesh = macrolib_3d_refine
    s = solver.SolverFullPowerIt(x_mesh, y_mesh, z_mesh, macrolib, 1., 0., 1., 0., 0., 0.)
    s.solve(inner_solver="BiCGSTAB")

@pytest.mark.integtest
def test_solverPI_3d_refine_BiCGSTAB_lu(macrolib_3d_refine):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh, y_mesh, z_mesh = macrolib_3d_refine
    s = solver.SolverFullPowerIt(x_mesh, y_mesh, z_mesh, macrolib, 1., 0., 1., 0., 0., 0.)
    s.solve(inner_solver="BiCGSTAB", inner_precond="IncompleteLUT")

@pytest.mark.integtest
def test_solverPI_3d_refine_GMRES(macrolib_3d_refine):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh, y_mesh, z_mesh = macrolib_3d_refine
    s = solver.SolverFullPowerIt(x_mesh, y_mesh, z_mesh, macrolib, 1., 0., 1., 0., 0., 0.)
    s.solve(inner_solver="GMRES")

@pytest.mark.integtest
def test_SolverFullSlepc_3d_refine(macrolib_3d_refine):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh, y_mesh, z_mesh = macrolib_3d_refine
    s = solver.SolverFullSlepc(x_mesh, y_mesh, z_mesh, macrolib, 1., 0., 1., 0., 0., 0.)
    s.solve(outer_max_iter=10000, inner_max_iter=200, tol=1e-6)

@pytest.mark.integtest
def test_SolverFullSlepc_3d_refine_arnoldi(macrolib_3d_refine):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh, y_mesh, z_mesh = macrolib_3d_refine
    s = solver.SolverFullSlepc(x_mesh, y_mesh, z_mesh, macrolib, 1., 0., 1., 0., 0., 0.)
    s.solve(solver="arnoldi", outer_max_iter=10000, inner_max_iter=200, tol=1e-6)

@pytest.mark.integtest
def test_SolverFullSlepc_3d_refine_ibcgs(macrolib_3d_refine):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh, y_mesh, z_mesh = macrolib_3d_refine
    s = solver.SolverFullSlepc(x_mesh, y_mesh, z_mesh, macrolib, 1., 0., 1., 0., 0., 0.)
    s.solve(inner_solver="ibcgs", outer_max_iter=10000, inner_max_iter=200, tol=1e-6)
