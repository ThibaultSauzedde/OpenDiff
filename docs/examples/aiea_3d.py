import numpy as np

import opendiff.materials as mat
import opendiff.solver as solver
from opendiff import set_log_level, log_level


fuel1 = [[1.5, 0.01, 0, 1, 0., 0.020],
         [0.4, 0.085, 0.135, 0, 0., 0.]]
fuel1_cr = [[1.5, 0.01, 0, 1, 0., 0.020],
            [0.4, 0.130, 0.135, 0, 0., 0.]]
fuel2 = [[1.5, 0.01, 0, 1, 0., 0.020],
         [0.4, 0.080, 0.135, 0, 0., 0.]]
refl = [[2.0, 0.0, 0, 0, 0., 0.040],
        [0.3, 0.01, 0., 0, 0., 0.]]
refl_cr = [[2.0, 0.0, 0, 0, 0., 0.040],
           [0.3, 0.055, 0., 0, 0., 0.]]
void = [[1e10, 0., 0., 0, 0., 0.],
        [1e10, 0., 0., 0, 0., 0.]]
all_mat = [fuel1, fuel1_cr, fuel2, refl, refl_cr, void]
mat_names = ["fuel1", "fuel1_cr", "fuel2", "refl", "refl_cr", "void"]
reac_names = ["D", "SIGA", "NU_SIGF", "CHI", "1", "2"]

mat_lib = mat.Materials(all_mat, mat_names, reac_names)
x = [0, 20*9]
y = [0, 20*9]
z_delta = [0, 20., 260, 80, 20]
z = np.cumsum(z_delta)
pblm3 = np.array([ ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "refl"],
                ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "refl"],
                ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "refl"],
                ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "refl"],
                ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "void"],
                ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "void"],
                ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "void",     "void",    "void"],
                ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "void",     "void",    "void"],
                ["refl",     "refl",     "refl",     "refl",     "void",     "void",     "void",     "void",    "void"]])
pblm2 = np.array([ ["fuel1_cr", "fuel1",    "fuel1",    "fuel1",    "fuel1_cr", "fuel1",    "fuel1",    "fuel2",   "refl"],
                ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",   "refl"],
                ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",    "fuel2",   "refl"],
                ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",    "refl",    "refl"],
                ["fuel1_cr", "fuel1",    "fuel1",    "fuel1",    "fuel1_cr", "fuel2",    "fuel2",    "refl",    "void"],
                ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",    "fuel2",    "refl",     "refl",    "void"],
                ["fuel1",    "fuel1",    "fuel2",    "fuel2",    "fuel2",    "refl",     "refl",     "void",    "void"],
                ["fuel2",    "fuel2",    "fuel2",    "refl",     "refl",     "refl",     "void",     "void",    "void"],
                ["refl",     "refl",     "refl",     "refl",     "void",     "void",     "void",     "void",    "void"]])
pblm1 = np.array([ ["fuel1_cr", "fuel1",    "fuel1",    "fuel1",    "fuel1_cr", "fuel1",    "fuel1",    "fuel2",   "refl"],
                ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",   "refl"],
                ["fuel1",    "fuel1",    "fuel1_cr", "fuel1",    "fuel1",    "fuel1",    "fuel2",    "fuel2",   "refl"],
                ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",    "refl",    "refl"],
                ["fuel1_cr", "fuel1",    "fuel1",    "fuel1",    "fuel1_cr", "fuel2",    "fuel2",    "refl",    "void"],
                ["fuel1",    "fuel1",    "fuel1",    "fuel1",    "fuel2",    "fuel2",    "refl",     "refl",    "void"],
                ["fuel1",    "fuel1",    "fuel2",    "fuel2",    "fuel2",    "refl",     "refl",     "void",    "void"],
                ["fuel2",    "fuel2",    "fuel2",    "refl",     "refl",     "refl",     "void",     "void",    "void"],
                ["refl",     "refl",     "refl",     "refl",     "void",     "void",     "void",     "void",    "void"]])
pblm0 = np.array([ ["refl_cr",  "refl",     "refl",     "refl",     "refl_cr",  "refl",     "refl",     "refl",    "refl"],
                ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "refl"],
                ["refl",     "refl",     "refl_cr",  "refl",     "refl",     "refl",     "refl",     "refl",    "refl"],
                ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "refl"],
                ["refl_cr",  "refl",     "refl",     "refl",     "refl_cr",  "refl",     "refl",     "refl",    "void"],
                ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",    "void"],
                ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "void",    "void"],
                ["refl",     "refl",     "refl",     "refl",     "refl",     "refl",     "void",     "void",    "void"],
                ["refl",     "refl",     "refl",     "refl",     "void",     "void",     "void",     "void",    "void"]])
#we mesh it
pblm = []
nb_div_pmat_x = 5
nb_div_pmat_y = 5
z_mesh = [[0., 5, 10, 13, 16, 18, 19, 20.],
        [21, 22, 24, 27, 30., 35, 40, 50, 60, 70, 80, 90, 100.,
        110., 120., 130., 140., 150., 160., 170., 180., 190., 200.,
        210, 220, 230, 240, 250, 260, 265, 270., 273, 276, 278, 279, 280],
        [281, 282, 284, 287, 300, 305, 310, 320, 330, 340, 345, 350, 353, 356, 358, 359, 360], 
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
# surf = np.multiply.outer(dx, dy)
# vol = np.multiply.outer(surf, dz)
# vol_1d = vol.reshape(-1)
macrolib = mat.Macrolib(mat_lib, pblm)
solver.init_slepc()
set_log_level(log_level.debug)
# s = solver.SolverSlepc(x_mesh, y_mesh, z_mesh, macrolib, 1., 0., 1., 0., 0., 0.)
# s.solve(outer_max_iter=10000, inner_max_iter=1, tol=1e-6, tol_inner=1e-4)

s = solver.SolverPowerIt(x_mesh, y_mesh, z_mesh, macrolib, 1., 0., 1., 0., 0., 0.)
s.solve(inner_solver="BiCGSTAB", inner_max_iter=1000, tol_inner=1e-3)
# s.solve(inner_solver="BiCGSTAB", inner_precond="IncompleteLUT", inner_max_iter=1000, tol_inner=1e-3)
# s.solve(inner_solver="GMRES", inner_max_iter=1000, tol_inner=1e-3)
# s.solve(inner_solver="SparseLU", inner_max_iter=1000, tol_inner=1e-3)