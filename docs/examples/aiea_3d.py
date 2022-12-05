import ipdb
import numpy as np

import opendiff.materials as mat
import opendiff.solver as solver
import opendiff.perturbation as pert

from opendiff import set_log_level, log_level

import grid_post_process as pp


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
all_mat = [fuel1, fuel1_cr, fuel2, refl, refl_cr, void]
mat_names = ["fuel1", "fuel1_cr", "fuel2", "refl", "refl_cr", "void"]
reac_names = ["D", "SIGA", "NU_SIGF", "CHI", "1", "2", "EFISS", "NU"]

mat_lib = mat.Materials(all_mat, mat_names, reac_names)
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
                  ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",
                      "fuel1",    "fuel1",    "fuel2",   "refl"],
                  ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",
                      "fuel1",    "fuel2",    "fuel2",   "refl"],
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
                  ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",
                      "fuel1",    "fuel1",    "fuel2",   "refl"],
                  ["fuel1",    "fuel1",    "fuel1_cr", "fuel1",    "fuel1",
                      "fuel1",    "fuel2",    "fuel2",   "refl"],
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
# we mesh it
pblm = []
nb_div_pmat_x = 5
nb_div_pmat_y = 5
z_mesh = [[0., 5, 10, 13, 16, 18, 19, 20.],
          [21, 22, 24, 27, 30., 35, 40, 50, 60, 70, 80, 90, 100.,
           110., 120., 130., 140., 150., 160., 170., 180., 190., 200.,
           210, 220, 230, 240, 250, 260, 265, 270., 273, 276, 278, 279, 280],
          [281, 282, 284, 287, 300, 305, 310, 320, 330,
              340, 345, 350, 353, 356, 358, 359, 360],
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

# ipdb.set_trace()
# surf = np.multiply.outer(dx, dy)
# vol = np.multiply.outer(surf, dz)
# vol_1d = vol.reshape(-1)
macrolib = mat.Macrolib(mat_lib, pblm)
solver.init_slepc()
set_log_level(log_level.debug)
# s = solver.SolverSlepc(x_mesh, y_mesh, z_mesh, macrolib, 1., 0., 1., 0., 0., 0.)
# # s.solve(solver="power",
# #         outer_max_iter=10000, inner_max_iter=500, tol=1e-10, tol_inner=1e-4)
# s.solve(
#         outer_max_iter=10000, inner_max_iter=500, tol=1e-10, tol_inner=1e-4)

# s = solver.SolverSlepc(x_mesh, y_mesh, z_mesh,
#                        macrolib, 1., 0., 1., 0., 0., 0.)
# s_star = solver.SolverSlepc(s)
# s_star.makeAdjoint()
# s.solve(tol=1e-15, nb_eigen_values=200, inner_max_iter=10, tol_inner=1e-3)
# s_star.solve(tol=1e-15, nb_eigen_values=200, inner_max_iter=10, tol_inner=1e-3)

s = solver.SolverCondPowerIt(x_mesh, y_mesh, z_mesh, macrolib, 1., 0., 1., 0., 0., 0.)
s_star = solver.SolverCondPowerIt(s)

# s = solver.SolverFullPowerIt(x_mesh, y_mesh, z_mesh, macrolib, 1., 0., 1., 0., 0., 0.)
# s_star = solver.SolverFullPowerIt(s)
# s_star.solve(inner_solver="BiCGSTAB", inner_max_iter=500, tol_inner=1e-5)

# s_star.makeAdjoint()
# s.solve(inner_solver="SparseLU", outer_max_iter=1000)
s.solve(inner_solver="SimplicialLDLT", inner_max_iter=1000,
        tol_inner=1e-7, outer_max_iter=50)

# s.solve(inner_solver="BiCGSTAB", inner_max_iter=10, tol_inner=1e-7, outer_max_iter=20)
# s.solve(inner_solver="BiCGSTAB", inner_precond="IncompleteLUT",
#         inner_max_iter=50, tol_inner=1e-3, outer_max_iter=1000)
# s.solve(inner_solver="GMRES", inner_max_iter=500, tol_inner=1e-3, outer_max_iter=1000)
# s_star.solve(inner_solver="BiCGSTAB", inner_max_iter=10, tol_inner=1e-3)

ev0 = s.getEigenVector(0)
import ipdb; ipdb.set_trace()
pp.plot_map2d(ev0.sum(axis=0).sum(axis=0),
              [x_mesh, y_mesh], show_stat_data=False, show_edge=False, show=True)

# ev0_star = s_star.getEigenVector(0)
# pp.plot_map2d(ev0_star.sum(axis=-1).sum(axis=0), [x_mesh, y_mesh], show_stat_data=False, show_edge=False)


# s.solve(inner_solver="BiCGSTAB", inner_precond="IncompleteLUT", inner_max_iter=1000, tol_inner=1e-3)
# s.solve(inner_solver="GMRES", inner_max_iter=1000, tol_inner=1e-3)
# s.solve(inner_solver="SparseLU", inner_max_iter=1000, tol_inner=1e-3)


# import grid_post_process as pp
# ev0 = s.getEigenVector(0, macrolib)
# pp.plot_map2d(ev0[:, :, :, 0].sum(axis=0), [x_mesh, y_mesh], show_stat_data=False, show_edge=False)
# pp.plot_map2d(ev0[:, :, :, 1].sum(axis=0), [x_mesh, y_mesh], show_stat_data=False, show_edge=False)
# pp.plot_map2d(ev0.sum(axis=-1).sum(axis=0), [x_mesh, y_mesh], show_stat_data=False, show_edge=False)
# power = s.getPower(macrolib)
# pp.plot_map2d(power[:, :, :].sum(axis=0), [x_mesh, y_mesh],
#               show_stat_data=False, show_edge=False)
# # !pp.plot_map2d(macrolib.getValues(2, "NU_SIGF").sum(axis=0), [x_mesh, y_mesh], show_stat_data=False, show_edge=False)$
# # !pp.plot_map2d(macrolib.getValues(2, "Efiss").sum(axis=0), [x_mesh, y_mesh], show_stat_data=False, show_edge=False)

ipdb.set_trace()
pert.checkBiOrthogonality(s, s_star, 1e-5, False)

mat_lib_pert = mat.Materials(mat_lib)
sigr = mat_lib_pert.getValue("fuel1_cr", 1, "SIGR")*1.01
mat_lib_pert.setValue("fuel1_cr", 1, "SIGR", sigr)

macrolib_pert = mat.Macrolib(mat_lib_pert, pblm)
s_pert = solver.SolverSlepc(x_mesh, y_mesh, z_mesh,
                            macrolib_pert, 1., 0., 1., 0., 0., 0.)
s_pert.solve(tol=1e-15, nb_eigen_values=1, inner_max_iter=10, tol_inner=1e-3)
s_recons = solver.SolverSlepc(s_pert)

egvec_recons, egval_recons, a = pert.firstOrderPerturbation(
    s, s_star, s_recons, "PhiStarMPhi")

s_pert.normPower()
s.normPower()
s_recons.normPower()

ev0 = s.getEigenVector(0)
ev0_star = s_star.getEigenVector(0)
ev0_recons = s_recons.getEigenVector(0)
ev0_pert = s_pert.getEigenVector(0)

pp.plot_map2d(ev0_recons.sum(axis=-1).sum(axis=0),
              [x_mesh, y_mesh], show_stat_data=False, show_edge=False)
pp.plot_map2d(100*(ev0_pert.sum(axis=-1).sum(axis=0)-ev0.sum(axis=-1).sum(axis=0))/ev0.sum(axis=-1).sum(axis=0),
              [x_mesh, y_mesh], show=True, x_label=None, y_label=None, cbar=False, show_stat_data=True, show_edge=False, show_xy=False, sym=True, stat_data_size=12)
pp.plot_map2d(100*(ev0_recons.sum(axis=-1).sum(axis=0)-ev0.sum(axis=-1).sum(axis=0))/ev0.sum(axis=-1).sum(axis=0),
              [x_mesh, y_mesh], show=True, x_label=None, y_label=None, cbar=False, show_stat_data=True, show_edge=False, show_xy=False, sym=True, stat_data_size=12)

pp.plot_map2d(100*(ev0_recons.sum(axis=-1).sum(axis=0)-ev0_pert.sum(axis=-1).sum(axis=0))/ev0_pert.sum(axis=-1).sum(axis=0),
              [x_mesh, y_mesh], show=True, x_label=None, y_label=None, cbar=False, show_stat_data=True, show_edge=False, show_xy=False, sym=True, stat_data_size=12)
pp.plot_map2d(100*(ev0_recons[:, :, :, 0].sum(axis=0)-ev0_pert[:, :, :, 0].sum(axis=0))/ev0_pert[:, :, :, 0].sum(axis=0), [x_mesh, y_mesh],
              show=True, x_label=None, y_label=None, cbar=False, show_stat_data=True, show_edge=False, show_xy=False, sym=True, stat_data_size=12)
pp.plot_map2d(100*(ev0_recons[:, :, :, 1].sum(axis=0)-ev0_pert[:, :, :, 1].sum(axis=0))/ev0_pert[:, :, :, 1].sum(axis=0), [x_mesh, y_mesh],
              show=True, x_label=None, y_label=None, cbar=False, show_stat_data=True, show_edge=False, show_xy=False, sym=True, stat_data_size=12)

ipdb.set_trace()
