import pytest
import numpy as np
import numpy.testing as npt
import scipy
import matplotlib.pyplot as plt

import opendiff.materials as mat
import opendiff.solver as solver
import opendiff.perturbation as pert

from opendiff import set_log_level, log_level

import grid_post_process as pp

def test_pert_first_order_1d(macrolib_1d_refine, macrolib_1d_pert_refine, datadir):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh = macrolib_1d_refine

    nb_eigen = 49
    ref_eigenvalue = 0.5513156
    s = solver.SolverSlepc(x_mesh, macrolib, -1., -1.)
    s.solve(nb_eigen_values=nb_eigen, inner_max_iter=500,
            tol=1e-10, tol_inner=1e-4)

    s_star = solver.SolverSlepc(s)
    s_star.makeAdjoint()
    s_star.solve(nb_eigen_values=nb_eigen, inner_max_iter=500,
                 tol=1e-10, tol_inner=1e-4)

    egval = s.getEigenValues()

    pert.checkBiOrthogonality(s, s_star, 1e-10, True)

    s_pert = solver.SolverSlepc(
        x_mesh, macrolib_1d_pert_refine, -1., -1.)  # zero flux albedo
    s_recons = solver.SolverSlepc(s_pert)
    s_pert.solve(nb_eigen_values=1, inner_max_iter=500,
                 tol=1e-10, tol_inner=1e-4)
    egval_pert = s_pert.getEigenValues()
    # np.savetxt("/home/ts249161/dev/these/openDiff/tests/test_solver/ev_slepc_1d.txt", ref_eigenvector)
    # ref_eigenvector = np.loadtxt(datadir / "ev_slepc_1d.txt")

    egvec_recons, egval_recons, a = pert.firstOrderPerturbation(
        s, s_star, s_recons, "PhiStarMPhi")
    # egvec_recons, egval_recons, a = pert.firstOrderPerturbation(
    #     s, s_star, s_recons, "power")

    # print(egval_recons)
    # print(egval_pert[0])
    # print(egval[0])

    # print("sens", 1e5*(egval_pert[0]-egval[0])/(egval[0]*egval_pert[0]))
    # print("sens recons", 1e5*(egval_recons-egval[0])/(egval[0]*egval_recons))
    # print("delta recons", 1e5*(egval_recons -
    #       egval_pert[0])/(egval_pert[0]*egval_recons))

    assert abs(1e5*(egval_recons -
                    egval_pert[0])/(egval_pert[0]*egval_recons)) < 18

    egvec_recons, egval_recons, a = pert.firstOrderPerturbation(
        s, s_star, s_recons, "power")

    # print("sens", 1e5*(egval_pert[0]-egval[0])/(egval[0]*egval_pert[0]))
    # print("sens recons", 1e5*(egval_recons-egval[0])/(egval[0]*egval_recons))
    # print("delta recons", 1e5*(egval_recons -
    #       egval_pert[0])/(egval_pert[0]*egval_recons))

    assert abs(1e5*(egval_recons -
                    egval_pert[0])/(egval_pert[0]*egval_recons)) < 18
    #
    # The flux reconstruction is not working
    #

    # s_pert.normPower()
    # s.normPower()
    # s_recons.normPower()

    # egvect_pert = s_pert.getEigenVector(0)
    # egvect = s.getEigenVector(0)
    # egvect_recons = s_recons.getEigenVector(0)

    # x_mean = (x_mesh[:-1] + x_mesh[1:])/2.

    # fig, axs = plt.subplots(5, 10, sharex=True, figsize=(12, 7))
    # for i in range(nb_eigen):
    #     k = i % 5
    #     j = i // 10
    #     phi_i = s.getEigenVector(i)
    #     phi_i = phi_i/phi_i.sum()*100
    #     axs[j, k].plot(x_mean, np.real(phi_i[0, 0, :, 0]))
    #     axs[j, k].plot(x_mean, np.real(phi_i[0, 0, :, 1]))
    #     axs[j, k].set_title(f"i = {i} / k = {egval[i]:.5f}")

    # # add a big axis, hide frame
    # fig.add_subplot(111, frameon=False)
    # # hide tick and tick label of the big axis
    # plt.tick_params(labelcolor='none', which='both', top=False,
    #                 bottom=False, left=False, right=False)
    # plt.xlabel("x (cm)")
    # plt.ylabel("$\phi$ (U.A.)")

    # fig, ax = plt.subplots(figsize=(15, 10))
    # ax.plot(x_mean, egvect[0, 0, :, 0], "--", label="init - 1", color='red')
    # ax.plot(x_mean, egvect[0, 0, :, 1], "--", label="init - 2", color='red')

    # ax.plot(x_mean, egvect_pert[0, 0, :, 0], label="pert - 1", color='blue')
    # ax.plot(x_mean, egvect_pert[0, 0, :, 1], label="pert - 2", color='orange')
    # ax.plot(x_mean, egvect_recons[0, 0, :, 0],
    #         "-.", label="recons - 1", color='orange')
    # ax.plot(x_mean, egvect_recons[0, 0, :, 1],
    #         "-.", label="recons - 2", color='blue')

    # ax.set_xlabel("x (cm")
    # ax.set_ylabel("$\phi$ (U.A.)")
    # ax.legend()

    # fig, ax = plt.subplots(1, 2, sharey=True, figsize=(20, 7))
    # ax[0].plot(x_mean, 100*(egvect_pert[0, 0, :, 0]-egvect[0, 0, :, 0]) / egvect[0, 0, :, 0],
    #            label="$(\phi_{pert,1} - \phi_{init,1}) / \phi_{init,1}}$", color='blue')
    # ax[0].plot(x_mean, 100*(egvect_pert[0, 0, :, 1]-egvect[0, 0, :, 1]) / egvect[0, 0, :, 1],
    #            label="$(\phi_{pert,2} - \phi_{init,2}) / \phi_{init,2}}$", color='orange')
    # ax[0].legend(fontsize=15)
    # ax[0].set_xlabel("x (cm)")
    # ax[0].set_ylabel("Ecarts (%)")

    # ax[1].plot(x_mean, 100*(egvect_recons[0, 0, :, 0]-egvect_pert[0, 0, :, 0]) / egvect_pert[0, 0, :, 0],
    #            label="$(\phi_{recons,1} - \phi_{pert,1}) / \phi_{pert,1}}$", color='blue')
    # ax[1].plot(x_mean, 100*(egvect_recons[0, 0, :, 1]-egvect_pert[0, 0, :, 1]) / egvect_pert[0, 0, :, 1],
    #            label="$(\phi_{recons,2} - \phi_{pert,2}) / \phi_{pert,2}}$", color='orange')
    # ax[1].legend(fontsize=15)
    # ax[1].set_xlabel("x (cm)")
    # fig.tight_layout()
    # plt.show()


def test_pert_first_order_2d(macrolib_2d_refine, macrolib_2d_pert_refine, datadir):
    solver.init_slepc()
    set_log_level(log_level.warning)
    macrolib, x_mesh, y_mesh = macrolib_2d_refine

    nb_eigen = 49
    s = solver.SolverSlepc(x_mesh, y_mesh, macrolib, 1., -1., 1., -1.)
    s.solve(nb_eigen_values=nb_eigen, inner_max_iter=500,
            tol=1e-10, tol_inner=1e-4)

    s_star = solver.SolverSlepc(s)
    s_star.makeAdjoint()
    s_star.solve(nb_eigen_values=nb_eigen, inner_max_iter=500,
                 tol=1e-10, tol_inner=1e-4)

    egval = s.getEigenValues()

    pert.checkBiOrthogonality(s, s_star, 1e-10, True)

    s_pert = solver.SolverSlepc(
        x_mesh, y_mesh, macrolib_2d_pert_refine, 1., -1., 1., -1.)
    s_recons = solver.SolverSlepc(s_pert)
    s_pert.solve(nb_eigen_values=1, inner_max_iter=500,
                 tol=1e-10, tol_inner=1e-4)
    egval_pert = s_pert.getEigenValues()
    # np.savetxt("/home/ts249161/dev/these/openDiff/tests/test_solver/ev_slepc_1d.txt", ref_eigenvector)
    # ref_eigenvector = np.loadtxt(datadir / "ev_slepc_1d.txt")

    egvec_recons, egval_recons, a = pert.firstOrderPerturbation(
        s, s_star, s_recons, "PhiStarMPhi")
    # egvec_recons, egval_recons, a = pert.firstOrderPerturbation(
    #     s, s_star, s_recons, "power")

    # print(egval_recons)
    # print(egval_pert[0])
    # print(egval[0])

    # print("sens", 1e5*(egval_pert[0]-egval[0])/(egval[0]*egval_pert[0]))
    # print("sens recons", 1e5*(egval_recons-egval[0])/(egval[0]*egval_recons))
    # print("delta recons", 1e5*(egval_recons -
    #       egval_pert[0])/(egval_pert[0]*egval_recons))

    assert abs(1e5*(egval_recons -
                    egval_pert[0])/(egval_pert[0]*egval_recons)) == pytest.approx(7.863762548633915, abs=0.1)

    s_pert.normPower()
    s.normPower()
    s_recons.normPower()

    egvect_pert = s_pert.getEigenVector(0)
    egvect = s.getEigenVector(0)
    egvect_recons = s_recons.getEigenVector(0)

    delta = 100*(egvect_pert-egvect)/egvect
    delta_recons = 100*(egvect_recons-egvect_pert)/egvect_pert

    # print(np.max(np.abs(delta)))
    # print(np.max(np.abs(delta_recons)))

    assert np.max(np.abs(delta_recons)) == pytest.approx(
        1.080209457369273, abs=1e-6)
    # np.savetxt(
    #     "/home/ts249161/dev/these/openDiff/tests/test_perturbation/delta_recons_2d.txt", delta_recons.reshape(-1))
    # np.savetxt(
    #     "/home/ts249161/dev/these/openDiff/tests/test_perturbation/delta_2d.txt", delta.reshape(-1))
    delta_ref = np.loadtxt(datadir / "delta_2d.txt")
    delta_recons_ref = np.loadtxt(datadir / "delta_recons_2d.txt")

    npt.assert_almost_equal(delta.reshape(-1), delta_ref,
                            decimal=8)
    npt.assert_almost_equal(delta_recons.reshape(-1), delta_recons_ref,
                            decimal=8)

    egvec_recons, egval_recons, a = pert.firstOrderPerturbation(
        s, s_star, s_recons, "power")

    s_recons.normPower()
    delta = 100*(egvect_pert-egvect)/egvect
    delta_recons = 100*(egvect_recons-egvect_pert)/egvect_pert
    assert np.max(np.abs(delta_recons)) == pytest.approx(1.080209457369273, abs=1e-6)
    npt.assert_almost_equal(delta.reshape(-1), delta_ref,
                            decimal=8)
    npt.assert_almost_equal(delta_recons.reshape(-1), delta_recons_ref,
                            decimal=8)

    # print(egval_recons)
    # print(egval_pert[0])
    # print(egval[0])

    # print("sens", 1e5*(egval_pert[0]-egval[0])/(egval[0]*egval_pert[0]))
    # print("sens recons", 1e5*(egval_recons-egval[0])/(egval[0]*egval_recons))
    # print("delta recons", 1e5*(egval_recons -
    #       egval_pert[0])/(egval_pert[0]*egval_recons))

    # print(np.max(delta))
    # print(np.max(delta_recons))

    # print(a)
    # print("----------------------------------------------")
    # pp.plot_map2d(delta_recons[:, :, :, 0].sum(axis=0), [x_mesh, y_mesh],
    #               show=False, x_label=None, y_label=None, cbar=False, show_stat_data=True, show_edge=False, show_xy=False, sym=True, stat_data_size=12)
    # pp.plot_map2d(delta_recons[:, :, :, 1].sum(axis=0), [x_mesh, y_mesh],
    #               show=True, x_label=None, y_label=None, cbar=False, show_stat_data=True, show_edge=False, show_xy=False, sym=True, stat_data_size=12)


def test_pert_high_order_1d(macrolib_1d_refine, macrolib_1d_pert_refine, datadir):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh = macrolib_1d_refine

    nb_eigen = 49

    s = solver.SolverSlepc(x_mesh, macrolib, -1., -1.)
    s.solve(nb_eigen_values=nb_eigen, inner_max_iter=500,
            tol=1e-10, tol_inner=1e-4)

    s_star = solver.SolverSlepc(s)
    s_star.makeAdjoint()
    s_star.solve(nb_eigen_values=nb_eigen, inner_max_iter=500,
                 tol=1e-10, tol_inner=1e-4)

    egval = s.getEigenValues()

    pert.checkBiOrthogonality(s, s_star, 1e-10, True)

    s_pert = solver.SolverSlepc(
        x_mesh, macrolib_1d_pert_refine, -1., -1.)  # zero flux albedo
    s_recons = solver.SolverSlepc(s_pert)
    s_pert.solve(nb_eigen_values=1, inner_max_iter=500,
                 tol=1e-10, tol_inner=1e-4)
    egval_pert = s_pert.getEigenValues()
    # np.savetxt("/home/ts249161/dev/these/openDiff/tests/test_solver/ev_slepc_1d.txt", ref_eigenvector)
    # ref_eigenvector = np.loadtxt(datadir / "ev_slepc_1d.txt")

    egvec_recons, egval_recons, a = pert.highOrderPerturbation(
        3, s, s_star, s_recons)

    print(egval_recons)
    print(egval_pert[0])
    print(egval[0])

    print("sens", 1e5*(egval_pert[0]-egval[0])/(egval[0]*egval_pert[0]))
    print("sens recons", 1e5*(egval_recons-egval[0])/(egval[0]*egval_recons))
    print("delta recons", 1e5*(egval_recons -
          egval_pert[0])/(egval_pert[0]*egval_recons))

    # assert abs(1e5*(egval_recons -
    #                 egval_pert[0])/(egval_pert[0]*egval_recons)) < 1

    assert abs(1e5*(egval_recons -
                    egval_pert[0])/(egval_pert[0]*egval_recons)) == pytest.approx(0.0015, abs=0.1)

def test_pert_high_order_2d(macrolib_2d_refine, macrolib_2d_pert_refine, datadir):
    solver.init_slepc()
    set_log_level(log_level.warning)
    macrolib, x_mesh, y_mesh = macrolib_2d_refine

    nb_eigen = 50
    s = solver.SolverSlepc(x_mesh, y_mesh, macrolib, 1., -1., 1., -1.)
    s.solve(nb_eigen_values=nb_eigen, inner_max_iter=500,
            tol=1e-10, tol_inner=1e-4)

    s_star = solver.SolverSlepc(s)
    s_star.makeAdjoint()
    s_star.solve(nb_eigen_values=nb_eigen, inner_max_iter=500,
                 tol=1e-10, tol_inner=1e-4)

    egval = s.getEigenValues()

    pert.checkBiOrthogonality(s, s_star, 1e-10, True)

    s_pert = solver.SolverSlepc(
        x_mesh, y_mesh, macrolib_2d_pert_refine, 1., -1., 1., -1.)
    s_recons = solver.SolverSlepc(s_pert)
    s_pert.solve(nb_eigen_values=1, inner_max_iter=500,
                 tol=1e-10, tol_inner=1e-4)
    egval_pert = s_pert.getEigenValues()

    # np.savetxt("/home/ts249161/dev/these/openDiff/tests/test_solver/ev_slepc_1d.txt", ref_eigenvector)
    # ref_eigenvector = np.loadtxt(datadir / "ev_slepc_1d.txt")

    egvec_recons, egval_recons, a = pert.highOrderPerturbation(
        5, s, s_star, s_recons)
    # egvec_recons, egval_recons, a = pert.firstOrderPerturbation(
    #     s, s_star, s_recons, "power")
    # print(s.getPower().sum())
    # print(s_recons.getPower().sum())
    # print(egval_recons)
    # print(egval_pert[0])
    # print(egval[0])

    # print("sens", 1e5*(egval_pert[0]-egval[0])/(egval[0]*egval_pert[0]))
    # print("sens recons", 1e5*(egval_recons-egval[0])/(egval[0]*egval_recons))
    # print("delta recons", 1e5*(egval_recons -
    #       egval_pert[0])/(egval_pert[0]*egval_recons))

    assert 1e5*(egval_recons -
          egval_pert[0])/(egval_pert[0]*egval_recons) == pytest.approx(0.18, abs=0.1)

    s_pert.normPower()
    s.normPower()
    s_recons.normPower()

    egvect_pert = s_pert.getEigenVector(0)
    egvect = s.getEigenVector(0)
    egvect_recons = s_recons.getEigenVector(0)

    delta = 100*(egvect_pert-egvect)/egvect
    delta_recons = 100*(egvect_recons-egvect_pert)/egvect_pert

    # print(np.max(np.abs(delta)))
    # print(np.max(np.abs(delta_recons)))

    assert np.max(np.abs(delta_recons)) == pytest.approx(1.0863853333508067, abs=1e-6)
    # np.savetxt(
    #     "/home/ts249161/dev/these/openDiff/tests/test_perturbation/delta_recons_high_2d.txt", delta_recons.reshape(-1))
    delta_ref = np.loadtxt(datadir / "delta_2d.txt")
    delta_recons_ref = np.loadtxt(datadir / "delta_recons_high_2d.txt")

    npt.assert_almost_equal(delta.reshape(-1), delta_ref,
                            decimal=5)
    npt.assert_almost_equal(delta_recons.reshape(-1), delta_recons_ref,
                            decimal=5)
    # pp.plot_map2d(delta_recons[:, :, :, 0].sum(axis=0), [x_mesh, y_mesh],
    #               show=False, x_label=None, y_label=None, cbar=False, show_stat_data=True, show_edge=False, show_xy=False, sym=True, stat_data_size=12)
    # pp.plot_map2d(delta_recons[:, :, :, 1].sum(axis=0), [x_mesh, y_mesh],
    #               show=True, x_label=None, y_label=None, cbar=False, show_stat_data=True, show_edge=False, show_xy=False, sym=True, stat_data_size=12)
