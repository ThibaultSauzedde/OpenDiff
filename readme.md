# OpendDiff: a simple C++ solver wrapped in python for the neutron diffusion problem and perturbation theories 

OpendDiff is able to solve the forward, adjoint and inhomogeneous neutron diffusion problem using a finite difference scheme.

The matrix problem is then organized either in its full form or in the condensed form. 

The full form is:

$$F \phi_i=\lambda_i B \phi_i,$$

with $\lambda_i$ the eigenvalue, $\phi_i$ the eigenvector or the neutron flux, $F$ the fission operator, $B$ the term regrouping the removal, scattering, and diffusion operators, and $i \in [0, R]$ with $R$ the total number of eigenvectors.

The condensed form is described in the following manual [DIF3D: A CODE TO SOLVE ONE-, TWO-, AND THREE-DIMENSIONAL FINITE-DIFFERENCE DIFFUSION THEORY PROBLEMS](https://www.osti.gov/biblio/7157044). 

The outer iterations can be accelerated by the chebychev methods.  

## Installation 

To install OpenDiff, one can use pip : 

    pip install . 

Before, the following C++ dependencies have to be installed or downloaded (for header only libraries):

- [eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
- [slepc](https://slepc.upv.es/)
- [HighFive](https://github.com/BlueBrain/HighFive)
- [spdlog](https://github.com/gabime/spdlog)
- [fmt](https://fmt.dev/latest/index.html)
- [pybind11](https://github.com/pybind/pybind11)

And the EIGEN_DIR, HIGHFIVE_DIR, SPDLOG_DIR and FMT_DIR environment variables have to be set:

    export EIGEN_DIR="path_to_eigen"
    export HIGHFIVE_DIR="path_to_highfive"
    export SPDLOG_DIR="path_to_spdlog"
    export FMT_DIR="path_to_fmt"

### Documentation generation

The document can be generated with Doxygen : 

    cd docs/
    Doxygen Doxyfile

## Basic use of OpenDiff

The main steps of a calculation based on OpenDiff are:

- Creation of the materials (a collection of isotopes associated with microscopic cross sections)
- Creation of the middles (a collection of materials associated with the concentration of the isotopes),
- Geometry creation,
- Creation of a library with all the macroscopic cross sections,
- Creation of a solver instance using the geometry, the macroscopic cross sections and the boundary conditions, 
- Call to the solve function of the solver. 

A small example based on the BILIS-2D benchmark is given above:

```python

import numpy as np

import opendiff.materials as mat
import opendiff.solver as solver
from opendiff import set_log_level, log_level

set_log_level(log_level.debug)

#
#Materials and middles creation
#

fuel1 = [[1.4360, 0.0095042, 0.0058708, 1, 0., 0.017754, 202 * 1.60218e-19 * 1e6, 2.4],
         [0.3635, 0.0750058, 0.0960670, 0, 0., 0., 202 * 1.60218e-19 * 1e6, 2.4]]
fuel2 = [[1.4366, 0.0096785, 0.0061908, 1, 0., 0.017621, 202 * 1.60218e-19 * 1e6, 2.4],
         [0.3636, 0.078436, 0.1035800, 0, 0., 0., 202 * 1.60218e-19 * 1e6, 2.4]]
refl = [[1.32, 0.0026562, 0, 1, 0., 0.023106, 202 * 1.60218e-19 * 1e6, 2.4],
        [0.2772, 0.071596, 0, 0, 0., 0., 202 * 1.60218e-19 * 1e6, 2.4]]
fuel3 = [[1.4389, 0.010363, 0.0074527, 1, 0., 0.017101, 202 * 1.60218e-19 * 1e6, 2.4],
         [0.3638, 0.091408, 0.1323600, 0, 0., 0., 202 * 1.60218e-19 * 1e6, 2.4]]
fuel4 = [[1.4381, 0.0100030, 0.0061908, 1, 0., 0.017290, 202 * 1.60218e-19 * 1e6, 2.4],
         [0.3665, 0.0848280, 0.103580, 0, 0., 0., 202 * 1.60218e-19 * 1e6, 2.4]]
fuel5 = [[1.4385, 0.0101320, 0.0064285, 1, 0., 0.017192, 202 * 1.60218e-19 * 1e6, 2.4],
         [0.3665, 0.087314, 0.109110, 0, 0., 0., 202 * 1.60218e-19 * 1e6, 2.4]]
fuel6 = [[1.4389, 0.010165, 0.0061908, 1, 0., 0.017125, 202 * 1.60218e-19 * 1e6, 2.4],
         [0.3679, 0.088024, 0.1035800, 0, 0., 0., 202 * 1.60218e-19 * 1e6, 2.4]]
fuel7 = [[1.4393, 0.010294, 0.0064285, 1, 0., 0.017027, 202 * 1.60218e-19 * 1e6, 2.4],
         [0.3680, 0.09051, 0.1091100, 0, 0., 0., 202 * 1.60218e-19 * 1e6, 2.4]]
void = [[1e10, 0., 0., 0, 0., 0., 202 * 1.60218e-19 * 1e6, 2.4],
        [1e10, 0., 0., 0, 0., 0., 202 * 1.60218e-19 * 1e6, 2.4]]

all_mat = {"fuel1": fuel1, "fuel2": fuel2, "fuel3": fuel3, "fuel4": fuel4,
           "fuel5": fuel5, "fuel6": fuel6, "fuel7": fuel7, "refl": refl, "void": void}

middles = {mat_name: mat_name for mat_name, _ in all_mat.items()}
isot_reac_names = [("ISO", "D"), ("ISO", "SIGA"), ("ISO", "NU_SIGF"), ("ISO", "CHI"),
                   ("ISO", "1"), ("ISO", "2"), ("ISO", "EFISS"), ("ISO", "NU")]
materials = {mat_name: mat.Material(
    values, isot_reac_names) for mat_name, values in all_mat.items()}
middles = mat.Middles(materials, middles)
middles.createIndependantMaterials()

#
# Geometry creation
#

x = [0, 17 * 23.1226]
y = [0, 17 * 23.1226]

nb_div_pmat_y = 20
nb_div_pmat_x = 20

pblm = np.array([["void",  "void",  "void",  "void",  "refl",  "refl",  "refl",  "refl",  "refl",  "refl",  "refl",  "refl",  "refl",  "void",  "void",  "void",  "void"],
                ["void",  "void",  "refl",  "refl",  "refl",  "fuel3", "fuel3", "fuel3", "fuel3", "fuel3", "fuel3", "fuel3", "refl",  "refl",  "refl",  "void",  "void"],
                ["void",  "refl",  "refl",  "fuel3", "fuel3", "fuel7", "fuel1", "fuel1", "fuel1", "fuel1", "fuel1", "fuel7", "fuel3", "fuel3", "refl",  "refl",  "void"],
                ["void",  "refl",  "fuel3", "fuel3", "fuel4", "fuel1", "fuel6", "fuel1", "fuel6", "fuel1", "fuel6", "fuel1", "fuel4", "fuel3", "fuel3", "refl",  "void"],
                ["refl",  "refl",  "fuel3", "fuel4", "fuel2", "fuel7", "fuel2", "fuel7", "fuel1", "fuel7", "fuel2", "fuel7", "fuel2", "fuel4", "fuel3", "refl",  "refl"],
                ["refl",  "fuel3", "fuel7", "fuel1", "fuel7", "fuel2", "fuel7", "fuel2", "fuel5", "fuel2", "fuel7", "fuel2", "fuel7", "fuel1", "fuel7", "fuel3", "refl"],
                ["refl",  "fuel3", "fuel1", "fuel6", "fuel2", "fuel7", "fuel1", "fuel7", "fuel2", "fuel7", "fuel1", "fuel7", "fuel2", "fuel6", "fuel1", "fuel3", "refl"],
                ["refl",  "fuel3", "fuel1", "fuel1", "fuel7", "fuel2", "fuel7", "fuel1", "fuel7", "fuel1", "fuel7", "fuel2", "fuel7", "fuel1", "fuel1", "fuel3", "refl"],
                ["refl",  "fuel3", "fuel1", "fuel6", "fuel1", "fuel5", "fuel2", "fuel7", "fuel1", "fuel7", "fuel2", "fuel5", "fuel1", "fuel6", "fuel1", "fuel3", "refl"],
                ["refl",  "fuel3", "fuel1", "fuel1", "fuel7", "fuel2", "fuel7", "fuel1", "fuel7", "fuel1", "fuel7", "fuel2", "fuel7", "fuel1", "fuel1", "fuel3", "refl"],
                ["refl",  "fuel3", "fuel1", "fuel6", "fuel2", "fuel7", "fuel1", "fuel7", "fuel2", "fuel7", "fuel1", "fuel7", "fuel2", "fuel6", "fuel1", "fuel3", "refl"],
                ["refl",  "fuel3", "fuel7", "fuel1", "fuel7", "fuel2", "fuel7", "fuel2", "fuel5", "fuel2", "fuel7", "fuel2", "fuel7", "fuel1", "fuel7", "fuel3", "refl"],
                ["refl",  "refl",  "fuel3", "fuel4", "fuel2", "fuel7", "fuel2", "fuel7", "fuel1", "fuel7", "fuel2", "fuel7", "fuel2", "fuel4", "fuel3", "refl",  "refl"],
                ["void",  "refl",  "fuel3", "fuel3", "fuel4", "fuel1", "fuel6", "fuel1", "fuel6", "fuel1", "fuel6", "fuel1", "fuel4", "fuel3", "fuel3", "refl",  "void"],
                ["void",  "refl",  "refl",  "fuel3", "fuel3", "fuel7", "fuel1", "fuel1", "fuel1", "fuel1", "fuel1", "fuel7", "fuel3", "fuel3", "refl",  "refl",  "void"],
                ["void",  "void",  "refl",  "refl",  "refl",  "fuel3", "fuel3", "fuel3", "fuel3", "fuel3", "fuel3", "fuel3", "refl",  "refl",  "refl",  "void",  "void"],
                ["void",  "void",  "void",  "void",  "refl",  "refl",  "refl",  "refl",  "refl",  "refl",  "refl",  "refl",  "refl",  "void",  "void",  "void",  "void"]])

shape = (pblm.shape[0]*nb_div_pmat_y, pblm.shape[1]*nb_div_pmat_x)
geom = np.empty(shape, dtype='U16')
for i, row in enumerate(pblm):
    for j, value in enumerate(row):
        geom[i*nb_div_pmat_x:(i+1)*nb_div_pmat_x, j *
             nb_div_pmat_x:(j+1)*nb_div_pmat_x] = value

geometry = np.array([geom])
x_mesh = np.linspace(x[0], x[1], geom.shape[1]+1)
y_mesh = np.linspace(y[0], y[1], geom.shape[0]+1)

#
# Macrolib creation
# 

macrolib = mat.Macrolib(middles, geometry)


#
# Solver
# 

s = solver.SolverCondPowerIt(x_mesh, y_mesh, macrolib, 0., 0., 0., 0.)
s.solve(inner_solver="BiCGSTAB", acceleration="chebyshev", inner_precond="", inner_max_iter=500, tol_inner=1e-6)
```

## Perturbation theories 

to complete

### The modal expansion 

### EpGPT

