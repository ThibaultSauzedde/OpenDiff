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


def test_checkBiOrthogonality(macrolib_1d_refine):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh = macrolib_1d_refine

    nb_eigen = 90
    s = solver.SolverFullSlepc(x_mesh, macrolib, -1., -1.)
    s.solve(nb_eigen_values=nb_eigen, inner_max_iter=500,
            tol=1e-10, tol_inner=1e-4)

    s_star = solver.SolverFullSlepc(s)
    s_star.makeAdjoint()
    s_star.solve(nb_eigen_values=nb_eigen, inner_max_iter=500,
                 tol=1e-10, tol_inner=1e-4)

    assert len(s.getEigenValues()) == 91
    assert len(s_star.getEigenValues()) == 91

    pert.checkBiOrthogonality(s, s_star, 1e-10, False, True)

    assert len(s.getEigenValues()) == 51
    assert len(s_star.getEigenValues()) == 51

    pert.checkBiOrthogonality(s, s_star, 1e-10, True)


def test_pert_first_order_1d(macrolib_1d_refine, macrolib_1d_pert_refine, datadir):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh = macrolib_1d_refine

    nb_eigen = 49
    ref_eigenvalue = 0.5513156
    s = solver.SolverFullSlepc(x_mesh, macrolib, -1., -1.)
    s.solve(nb_eigen_values=nb_eigen, inner_max_iter=500,
            tol=1e-10, tol_inner=1e-4)

    s_star = solver.SolverFullSlepc(s)
    s_star.makeAdjoint()
    s_star.solve(nb_eigen_values=nb_eigen, inner_max_iter=500,
                 tol=1e-10, tol_inner=1e-4)

    egval = s.getEigenValues()

    pert.checkBiOrthogonality(s, s_star, 1e-10, True)

    s_pert = solver.SolverFullSlepc(
        x_mesh, macrolib_1d_pert_refine, -1., -1.)  # zero flux albedo
    s_recons = solver.SolverFullSlepc(s_pert)
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
    s = solver.SolverFullSlepc(x_mesh, y_mesh, macrolib, 1., -1., 1., -1.)
    s.solve(nb_eigen_values=nb_eigen, inner_max_iter=500,
            tol=1e-10, tol_inner=1e-4)

    s_star = solver.SolverFullSlepc(s)
    s_star.makeAdjoint()
    s_star.solve(nb_eigen_values=nb_eigen, inner_max_iter=500,
                 tol=1e-10, tol_inner=1e-4)

    egval = s.getEigenValues()

    pert.checkBiOrthogonality(s, s_star, 1e-10, True)

    s_pert = solver.SolverFullSlepc(
        x_mesh, y_mesh, macrolib_2d_pert_refine, 1., -1., 1., -1.)
    s_recons = solver.SolverFullSlepc(s_pert)
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
    assert np.max(np.abs(delta_recons)) == pytest.approx(
        1.080209457369273, abs=1e-6)
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

    s = solver.SolverFullSlepc(x_mesh, macrolib, -1., -1.)
    s.solve(nb_eigen_values=nb_eigen, inner_max_iter=500,
            tol=1e-10, tol_inner=1e-4)

    s_star = solver.SolverFullSlepc(s)
    s_star.makeAdjoint()
    s_star.solve(nb_eigen_values=nb_eigen, inner_max_iter=500,
                 tol=1e-10, tol_inner=1e-4)

    egval = s.getEigenValues()

    pert.checkBiOrthogonality(s, s_star, 1e-10, True)

    s_pert = solver.SolverFullSlepc(
        x_mesh, macrolib_1d_pert_refine, -1., -1.)  # zero flux albedo
    s_recons = solver.SolverFullSlepc(s_pert)
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
    s = solver.SolverFullSlepc(x_mesh, y_mesh, macrolib, 1., -1., 1., -1.)
    s.solve(nb_eigen_values=nb_eigen, inner_max_iter=500,
            tol=1e-10, tol_inner=1e-4)

    s_star = solver.SolverFullSlepc(s)
    s_star.makeAdjoint()
    s_star.solve(nb_eigen_values=nb_eigen, inner_max_iter=500,
                 tol=1e-10, tol_inner=1e-4)

    egval = s.getEigenValues()

    pert.checkBiOrthogonality(s, s_star, 1e-10, True)

    s_pert = solver.SolverFullSlepc(
        x_mesh, y_mesh, macrolib_2d_pert_refine, 1., -1., 1., -1.)
    s_recons = solver.SolverFullSlepc(s_pert)
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

    assert np.max(np.abs(delta_recons)) == pytest.approx(
        1.0863853333508067, abs=1e-6)
    # np.savetxt(
    #     "/home/ts249161/dev/these/openDiff/tests/test_perturbation/delta_recons_high_2d.txt", delta_recons.reshape(-1))
    delta_ref = np.loadtxt(datadir / "delta_2d.txt")
    delta_recons_ref = np.loadtxt(datadir / "delta_recons_high_2d.txt")

    npt.assert_almost_equal(delta.reshape(-1), delta_ref,
                            decimal=4)
    npt.assert_almost_equal(delta_recons.reshape(-1), delta_recons_ref,
                            decimal=5)
    # pp.plot_map2d(delta_recons[:, :, :, 0].sum(axis=0), [x_mesh, y_mesh],
    #               show=False, x_label=None, y_label=None, cbar=False, show_stat_data=True, show_edge=False, show_xy=False, sym=True, stat_data_size=12)
    # pp.plot_map2d(delta_recons[:, :, :, 1].sum(axis=0), [x_mesh, y_mesh],
    #               show=True, x_label=None, y_label=None, cbar=False, show_stat_data=True, show_edge=False, show_xy=False, sym=True, stat_data_size=12)


def test_first_order_gpt_1d(macrolib_1d_nmid, macrolib_1d_nmid_pert, datadir):
    solver.init_slepc()
    set_log_level(log_level.debug)
    macrolib, x_mesh = macrolib_1d_nmid

    s = solver.SolverFullSlepc(x_mesh, macrolib, 0., 0.)
    s.solve(inner_max_iter=500, tol=1e-10, tol_inner=1e-4)

    s_star = solver.SolverFullPowerIt(
        x_mesh, macrolib_1d_nmid_pert, 0., 0.)
    s_star.makeAdjoint()
    s_star.solve(inner_solver="SparseLU", inner_max_iter=500,
                 tol=1e-10, tol_inner=1e-4)

    s_pert = solver.SolverFullPowerIt(
        x_mesh, macrolib_1d_nmid_pert, 0., 0.)  # zero flux albedo
    s_pert.solve(inner_solver="SparseLU", inner_max_iter=500,
                 tol=1e-10, tol_inner=1e-4)

    power = s.normPower(1e6)
    power_pert = s_pert.normPower(1e6)

    ev = s.getEigenVector(0)
    ev_pert = s_pert.getEigenVector(0)

    # get power in python
    sigf = np.concatenate([macrolib.getValues1D(
        1, "SIGF"),  macrolib.getValues1D(2, "SIGF")])
    efiss = np.concatenate([macrolib.getValues1D(
        1, "EFISS"),  macrolib.getValues1D(2, "EFISS")])
    norm = sigf * efiss
    power_python = norm.dot(np.ravel(ev))

    sigf_pert = np.concatenate([macrolib_1d_nmid_pert.getValues1D(
        1, "SIGF"),  macrolib_1d_nmid_pert.getValues1D(2, "SIGF")])
    efiss_pert = np.concatenate([macrolib_1d_nmid_pert.getValues1D(
        1, "EFISS"),  macrolib_1d_nmid_pert.getValues1D(2, "EFISS")])
    norm_pert = sigf_pert * efiss_pert
    power_python_pert = norm_pert.dot(np.ravel(ev_pert))
    # norm_pert * np.ravel(ev_pert)

    norm = s.getPowerNormVector()
    norm_pert = s_pert.getPowerNormVector()

    response = np.zeros_like(norm)
    response[40] = norm[40]
    response[int(response.shape[0]/2) +
             40] = norm[int(response.shape[0]/2) + 40]
    response_power = response.dot(np.ravel(ev))
    response_pert = np.zeros_like(norm_pert)
    response_pert[40] = norm_pert[40]
    response_pert[int(response_pert.shape[0]/2) +
                  40] = norm_pert[int(response_pert.shape[0]/2) + 40]
    response_power_pert = response_pert.dot(np.ravel(ev_pert))

    x_mean = (x_mesh[:-1] + x_mesh[1:])/2.

    delta_power_gpt, gamma_star = pert.firstOrderGPT(s, s_star, s_pert, response, response_pert, norm, norm_pert,
                                                     1e-6, 1e-5, 20000, 500, "SparseLU", "")
    gamma_star = gamma_star.reshape(ev.shape)
    # source = source.reshape(ev.shape)

    fig, ax = plt.subplots()
    ax.plot(x_mean, ev[0, 0, 0, :], label="ev0")
    ax.plot(x_mean, ev[1, 0, 0, :], label="ev1")
    ax.plot(x_mean, ev_pert[0, 0, 0, :], "--", label="ev_pert0")
    ax.plot(x_mean, ev_pert[1, 0, 0, :], "--", label="ev_pert1")
    plt.legend()

    fig, ax = plt.subplots()
    ax.plot(x_mean, power[0, 0, :], label="power")
    ax.plot(x_mean, power_pert[0, 0, :], "--", label="power_pert")
    ax.plot(x_mean[40], response_power, "+", label="power")
    ax.plot(x_mean[40], response_power_pert, "P", label="power_pert")
    ax.plot(x_mean[40], response_power -
            delta_power_gpt, "P", label="power_pert")
    plt.legend()

    fig, ax = plt.subplots()
    # ax.plot(x_mean, source[0, 0, 0, :], label="source0")
    # ax.plot(x_mean, source[1, 0, 0, :], label="source1")
    ax.plot(x_mean, gamma_star[0, 0, 0, :], "--", label="gamma_star0")
    ax.plot(x_mean, gamma_star[1, 0, 0, :], "--", label="gamma_star1")
    plt.legend()

    plt.show()

    # import ipdb
    # ipdb.set_trace()


def test_EpGPT_1d(xs_aiea3d, nmid_geom_1d, macrolib_1d_nmid_pert, datadir):
    solver.init_slepc()
    set_log_level(log_level.info)
    all_mat, middles, isot_reac_names = xs_aiea3d
    materials = {mat_name: mat.Material(
        values, isot_reac_names) for mat_name, values in all_mat.items()}
    middles = mat.Middles(materials, middles)
    geometry, x_mesh = nmid_geom_1d
    macrolib = mat.Macrolib(middles, geometry)

    s_pert = solver.SolverFullPowerIt(
        x_mesh, macrolib_1d_nmid_pert, 0., 0.)  # zero flux albedo
    s_recons = solver.SolverFullPowerIt(s_pert)
    s_pert.solve(inner_solver="SparseLU", inner_max_iter=500,
                 tol=1e-10, tol_inner=1e-4)
    s_pert.normPower(1e6)

    epgpt_1d = pert.EpGPT(x_mesh, middles, geometry,
                          0., 0.)
    null_vect = []

    epgpt_1d.solveReference(1e-6, 1e-5, null_vect, 1.,
                            1e-5, 1000, 100, "SparseLU", "", "")
    # epgpt_1d.createBasis(1e-5, ["D", "SIGA", "NU_SIGF", "CHI"], 10., 1, 1e6,
    #                      1e-6, 1e-5, null_vect, 1.,
    #                      1e-5, 1000, 100, "SparseLU", "", "")
    epgpt_1d.createBasis(1e-3, 0.5, 0.001, 0.001, 0.001, 1e6,
                         1e-6, 1e-5, 1.,
                         1e-5, 1000, 100, "SparseLU", "", "")

    basis = epgpt_1d.getBasis()
    for i in range(len(basis)):
        for j in range(len(basis)):
            if (i == j):
                continue
            test = basis[i].dot(basis[j])
            if test > 1e-6:
                print(test)

    for i in range(len(basis)):
        test = basis[i].dot(basis[i])
        if abs(test-1.) > 1e-6:
            print(test)

    x_mean = (x_mesh[:-1] + x_mesh[1:])/2.
    # fig, ax = plt.subplots()
    # for i in range(len(basis)):
    #     ax.plot(basis[i])
    # plt.show()

    epgpt_1d.calcImportances(1e-5, [], 1e-5, 10000, 100, "SparseLU", "")
    importances = epgpt_1d.getImportances()

    # fig, ax = plt.subplots()
    # for i in range(len(importances)):
    #     ax.plot(importances[i])
    # plt.show()

    epgpt_1d.dump(str(datadir / "./test_epgpt_1d.h5"))

    _, eigenvalue_recons, a = epgpt_1d.firstOrderPerturbation(s_recons, -1)
    eigenvalue = epgpt_1d.getSolver().getEigenValues()[0]
    eigenvalue_pert = s_pert.getEigenValues()[0]

    print(1e5*(eigenvalue_pert-eigenvalue)/(eigenvalue*eigenvalue_pert))
    print(1e5*(eigenvalue_pert-eigenvalue_recons) /
          (eigenvalue_recons*eigenvalue_pert))

    s_pert.normPower(1e6)
    epgpt_1d.getSolver().normPower(1e6)
    s_recons.normPower(1e6)

    print(s_recons.getPower().sum())  # why the power is not exactlyc 1e6 ???

    egvect_pert = s_pert.getEigenVector(0)
    egvect = epgpt_1d.getSolver().getEigenVector(0)
    egvect_recons = s_recons.getEigenVector(0)

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(x_mean, egvect[0, 0, 0, :], "--", label="init - 1", color='red')
    ax.plot(x_mean, egvect[1, 0, 0, :], "--", label="init - 2", color='red')

    ax.plot(x_mean, egvect_pert[0, 0, 0, :], label="pert - 1", color='blue')
    ax.plot(x_mean, egvect_pert[1, 0, 0, :], label="pert - 2", color='orange')
    ax.plot(x_mean, egvect_recons[0, 0, 0, :],
            "-.", label="recons - 1", color='orange')
    ax.plot(x_mean, egvect_recons[1, 0, 0, :],
            "-.", label="recons - 2", color='blue')

    ax.set_xlabel("x (cm")
    ax.set_ylabel("$\phi$ (U.A.)")
    ax.legend()

    delta_egvect_recons = 100 * (egvect_recons-egvect_pert)/egvect_pert

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(x_mean, delta_egvect_recons[0, 0, 0, :],  color='red')
    ax.plot(x_mean, delta_egvect_recons[1, 0, 0, :], color='red')
    ax.set_xlabel("x (cm")
    ax.set_ylabel("$\phi$ (U.A.)")
    ax.legend()
    plt.show()


def test_python_EpGPT_1d(xs_aiea3d, nmid_geom_1d, datadir):
    solver.init_slepc()
    set_log_level(log_level.warning)
    all_mat, middles, isot_reac_names = xs_aiea3d
    materials = {mat_name: mat.Material(
        values, isot_reac_names) for mat_name, values in all_mat.items()}
    middles = mat.Middles(materials, middles)
    geometry, x_mesh = nmid_geom_1d
    macrolib = mat.Macrolib(middles, geometry)
    # epgpt_1d = pert.EpGPT(x_mesh, middles, geometry,
    #                       0., 0.)
    # null_vect = []
    # epgpt_1d.createBasis(1e-5, ["D", "SIGA", "NU_SIGF", "CHI"], 1., 1e6,
    #                      1e-6, 1e-5, null_vect, 1.,
    #                      1e-5, 1000, 100, "SparseLU", "")
    # basis = epgpt_1d.getBasis()
    # basis_coeff = epgpt_1d.getBasisCoeff()
    tol = 1e-6
    tol_eigen_vectors = 1e-5
    v0 = []
    tol_inner = 1e-4
    outer_max_iter = 1000
    inner_max_iter = 100
    inner_solver = "SparseLU"
    solver_ref = solver.SolverFullPowerIt(
        x_mesh, macrolib, 0., 0.)
    solver_ref.solve(tol, tol_eigen_vectors, 1, v0, 1.,
                     tol_inner, outer_max_iter, inner_max_iter, inner_solver, "")
    solver_ref.normPower(1e6)

    basis = []
    for i in range(500):
        middles_pert = mat.Middles(middles)
        middles_pert.randomPerturbation(["D", "SIGA", "NU_SIGF", "CHI"], 10.)
        macrolib_pert = mat.Macrolib(middles_pert, geometry)
        solver_i = solver.SolverFullPowerIt(
            x_mesh, macrolib_pert, 0., 0.)
        v0 = solver_ref.getEigenVectors()[0]
        solver_i.solve(tol, tol_eigen_vectors, 1, v0, solver_ref.getEigenValues()[0],
                       tol_inner, outer_max_iter, inner_max_iter, inner_solver, "")
        solver_i.normPower(1e6)
        # print("ev", 1e5 *(solver_ref.getEigenValues()[0] - solver_i.getEigenValues()[0]) / (solver_ref.getEigenValues()[0] * solver_i.getEigenValues()[0]))

        delta_ev = solver_i.getEigenVectors(
        )[0] - solver_ref.getEigenVectors()[0]
        basis_size_test = len(basis)
        delta_ev_recons = np.zeros_like(delta_ev)
        for k in range(basis_size_test):
            coeff = basis[k].dot(delta_ev)
            delta_ev_recons += coeff * basis[k]

        test = np.linalg.norm(delta_ev - delta_ev_recons) / \
            np.linalg.norm(delta_ev)
        print(
            f"The reconstruction precision is {test:.2e} with a basis size {basis_size_test}")

        if (test > 1e-5):
            u_i = np.copy(delta_ev)
            for k in range(len(basis)):
                u_i -= basis[k].dot(u_i) * basis[k]
            u_i /= np.linalg.norm(u_i)
            basis.append(u_i)

        # fig, ax = plt.subplots()
        # ax.plot(delta_ev)
        # ax.plot(delta_ev_recons)
        # # plt.show()

        # fig, ax = plt.subplots()
        # ax.plot(u_i)
        # plt.show()

    for i in range(len(basis)):
        for j in range(len(basis)):
            if (i == j):
                continue
            test = basis[i].dot(basis[j])
            if test > 1e-6:
                print(test)

    for i in range(len(basis)):
        test = basis[i].dot(basis[i])
        if abs(test-1.) > 1e-6:
            print(test)

    import ipdb
    ipdb.set_trace()

    # x_mean = (x_mesh[:-1] + x_mesh[1:])/2.
    # fig, ax = plt.subplots()
    # for i in range(len(basis)):
    #     ax.plot(basis[i])
    # plt.show()

    # import ipdb
    # ipdb.set_trace()
