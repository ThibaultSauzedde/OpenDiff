// python binding

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include <vector>
#include <string_view>

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

#include <petscmat.h>
#include <slepceps.h>

#include "spdlog/spdlog.h"

#include "materials.h"
#include "macrolib.h"
#include "diff_operator.h"
#include "solver.h"
#include "perturbation.h"

namespace py = pybind11;
using vecd = std::vector<double>;
using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>; // declares a column-major sparse matrix type of double

//todo: use enum instead of string in some cases 


// return 3 dimensional ndarray
template <class T>
py::array_t<T> eigenTensor3D(py::array_t<T> inArray)
{

    // request a buffer descriptor from Python
    py::buffer_info buffer_info = inArray.request();

    // extract data and shape of input array
    T *data = static_cast<T *>(buffer_info.ptr);
    std::vector<ssize_t> shape = buffer_info.shape;

    // wrap ndarray in Eigen::Map:
    // the second template argument is the rank of the tensor and has to be known at compile time
    Eigen::TensorMap<Eigen::Tensor<T, 3>> in_tensor(data, shape[0], shape[1], shape[2]);

    return py::array_t<T, py::array::c_style>(shape,
                                              in_tensor.data()); // data pointer
}

// void PetscVecToEigen(const Vec &pvec, unsigned int nrow, unsigned int ncol, Eigen::MatrixXd &emat)
// {
//     PetscScalar *pdata;
//     // Returns a pointer to a contiguous array containing this
//     processor's portion
//         // of the vector data. For standard vectors this doesn't use any copies.
//         // If the the petsc vector is not in a contiguous array then it will copy
//         // it to a contiguous array.
//         VecGetArray(pvec, &pdata);
//     // Make the Eigen type a map to the data. Need to be mindful of
//     anything that
//         // changes the underlying data location like re-allocations.
//         emat = Eigen::Map<Eigen::MatrixXd>(pdata, nrow, ncol);
//     VecRestoreArray(pvec, &pdata);
// }

PYBIND11_MODULE(opendiff, m)
{
    py::enum_<spdlog::level::level_enum>(m, "log_level")
        .value("debug", spdlog::level::debug)
        .value("info", spdlog::level::info)
        .value("warning", spdlog::level::warn)
        .value("error", spdlog::level::err)
        .export_values();

    m.def("set_log_level", spdlog::set_level);
    py::add_ostream_redirect(m, "ostream_redirect"); // if needed by the user

    py::module eigen = m.def_submodule("eigen", "A simple eigen tensor binding");
    eigen.def("eigenTensor3D", &eigenTensor3D<double>, py::return_value_policy::move,
              py::arg("inArray"));

    py::module materials = m.def_submodule("materials", "A module for materials and macrolib handling.");

    py::class_<mat::Materials>(materials, "Materials")
        .def(py::init<const mat::Materials &>())
        .def(py::init<const std::vector<Eigen::ArrayXXd> &, const std::vector<std::string> &,
                      const std::vector<std::string> &>())
        .def("getReacNames", &mat::Materials::getReacNames)
        .def("getMatNames", &mat::Materials::getMatNames)
        .def("getMaterial", &mat::Materials::getMaterial)
        .def("getMaterials", &mat::Materials::getMaterials)
        .def("getValue", &mat::Materials::getValue)
        .def("setValue", &mat::Materials::setValue)
        .def("getNbGroups", py::overload_cast<>(&mat::Materials::getNbGroups, py::const_))
        .def("getReactionIndex", &mat::Materials::getReactionIndex)
        .def("addMaterial", &mat::Materials::addMaterial);

    py::class_<mat::Macrolib>(materials, "Macrolib")
        // .def(py::init<const mat::Macrolib &>())
        .def(py::init<const mat::Materials &, const std::vector<std::vector<std::vector<std::string>>> &>())
        .def("getReacNames", &mat::Macrolib::getReacNames)
        .def("getNbGroups", &mat::Macrolib::getNbGroups)
        .def("getDim", &mat::Macrolib::getDim)
        .def("getValues", &mat::Macrolib::getValuesPython)
        .def("getValues1D", &mat::Macrolib::getValues1DPython);

    py::module operators = m.def_submodule("operators", "A module for the operators' creation.");
    operators.def("init_petsc", PetscInitializeNoArguments);

    operators.def("diff_removal_op", py::overload_cast<vecd &, mat::Macrolib &>(&operators::diff_removal_op<SpMat, vecd>));
    operators.def("diff_fission_op", py::overload_cast<vecd &, mat::Macrolib &>(&operators::diff_fission_op<SpMat, vecd>));
    operators.def("diff_scatering_op", py::overload_cast<vecd &, mat::Macrolib &>(&operators::diff_scatering_op<SpMat, vecd>));
    operators.def("diff_diffusion_op_1d", py::overload_cast<vecd &, mat::Macrolib &, double, double>(&operators::diff_diffusion_op<SpMat, vecd>));
    operators.def("diff_diffusion_op_2d", py::overload_cast<vecd &, vecd &, mat::Macrolib &, double, double, double, double>(&operators::diff_diffusion_op<SpMat, vecd>));
    operators.def("diff_diffusion_op_3d", py::overload_cast<vecd &, vecd &, vecd &, mat::Macrolib &,
                                                            double, double, double, double, double, double>(&operators::diff_diffusion_op<SpMat, vecd>));

    py::module solver = m.def_submodule("solver", "A module for the solver.");
    solver.def("init_slepc", solver::init_slepc);
    solver.def("end_slepc", SlepcFinalize);

    // py::implicitly_convertible<solver::Solver<SpMat>, solver::SolverPowerIt>();
    // py::implicitly_convertible<solver::Solver<SpMat>, solver::SolverSlepc>();

    py::class_<solver::Solver<SpMat>>(solver, "Solver")
        .def("getVolumes", &solver::Solver<SpMat>::getVolumesPython)
        .def("makeAdjoint", &solver::Solver<SpMat>::makeAdjoint)
        .def("getEigenValues", &solver::Solver<SpMat>::getEigenValues)
        .def("getEigenVectors", &solver::Solver<SpMat>::getEigenVectors)
        .def("getVolumes", &solver::Solver<SpMat>::getVolumesPython)
        .def("getEigenVector", &solver::Solver<SpMat>::getEigenVectorPython)
        .def("getPower", &solver::Solver<SpMat>::getPowerPython)
        .def("normPower", &solver::Solver<SpMat>::normPowerPython,
             py::arg("power_W") = 1.)
        .def("normPhiStarMPhi", &solver::Solver<SpMat>::normPhiStarMPhi)
        .def("getK", &solver::Solver<SpMat>::getK)
        .def("getM", &solver::Solver<SpMat>::getM);

    py::class_<solver::SolverPowerIt, solver::Solver<SpMat>>(solver, "SolverPowerIt")
        .def(py::init<const solver::Solver<SpMat> &>())
        .def(py::init<vecd &, mat::Macrolib &, double, double>())
        .def(py::init<vecd &, vecd &, mat::Macrolib &, double, double, double, double>())
        .def(py::init<vecd &, vecd &, vecd &, mat::Macrolib &, double, double, double, double, double, double>())
        .def("solve", &solver::SolverPowerIt::solve,
             py::arg("tol") = 1e-6, py::arg("tol_eigen_vectors") = 1e-4,
             py::arg("nb_eigen_values") = 1, py::arg("v0") = Eigen::VectorXd(),
             py::arg("tol_inner") = 1e-4, py::arg("outer_max_iter") = 500,
             py::arg("inner_max_iter") = 20, py::arg("inner_solver") = "BiCGSTAB",
             py::arg("inner_precond") = "");

    py::class_<solver::SolverSlepc, solver::Solver<SpMat>>(solver, "SolverSlepc")
        .def(py::init<const solver::Solver<SpMat> &>())
        .def(py::init<vecd &, mat::Macrolib &, double, double>())
        .def(py::init<vecd &, vecd &, mat::Macrolib &, double, double, double, double>())
        .def(py::init<vecd &, vecd &, vecd &, mat::Macrolib &, double, double, double, double, double, double>())
        .def("solve", &solver::SolverSlepc::solveIterative,
             py::arg("tol") = 1e-6, py::arg("tol_eigen_vectors") = 1e-4,
             py::arg("nb_eigen_values") = 1, py::arg("v0") = Eigen::VectorXd(),
             py::arg("tol_inner") = 1e-4, py::arg("outer_max_iter") = 500,
             py::arg("inner_max_iter") = 20, py::arg("solver") = "krylovschur",
             py::arg("inner_solver") = "", py::arg("inner_precond") = "");

    py::module perturbation = m.def_submodule("perturbation", "A module for the perturbation.");
    perturbation.def("checkBiOrthogonality", &perturbation::checkBiOrthogonality,
                     py::arg("solver"), py::arg("solver_star"),
                     py::arg("max_eps") = 1e-6, py::arg("raise_error") = false);
    perturbation.def("firstOrderPerturbation", &perturbation::firstOrderPerturbation);
    perturbation.def("highOrderPerturbation", &perturbation::highOrderPerturbationPython);
}