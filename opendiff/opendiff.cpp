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
using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;      // declares a column-major sparse matrix type of double
using SpMatSimple = Eigen::SparseMatrix<float, Eigen::RowMajor>; // declares a column-major sparse matrix type of float
using tuple_str = std::tuple<std::string, std::string>;
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

    py::class_<mat::Material>(materials, "Material")
        .def(py::init<const mat::Material &>())
        .def(py::init<const Eigen::ArrayXXd &, const std::vector<std::string> &, const std::vector<std::string> &>())
        .def(py::init<const Eigen::ArrayXXd &, const std::vector<tuple_str> &>())
        .def("getReacNames", &mat::Material::getReacNames)
        .def("getIsotNames", &mat::Material::getIsotNames)
        .def("getNbGroups", &mat::Material::getNbGroups)
        .def("getValues", &mat::Material::getValues)
        .def("getXsValue", &mat::Material::getXsValue)
        .def("setXsValue", &mat::Material::setXsValue)
        .def("multXsValue", &mat::Material::multXsValue)
        .def("getIndex", py::overload_cast<>(&mat::Material::getIndex, py::const_))
        .def("getIndex", py::overload_cast<const std::string &, const std::string &>(&mat::Material::getIndex, py::const_));

    py::class_<mat::Middles>(materials, "Middles")
        .def(py::init<const mat::Middles &>())
        .def(py::init<std::map<std::string, mat::Material> &, const std::map<std::string, std::string> &>())
        .def(py::init<std::map<std::string, mat::Material> &, const std::map<std::string, std::string> &,
                      const std::map<std::string, std::map<std::string, double>> &>())
        .def("createIndependantMaterials", &mat::Middles::createIndependantMaterials)
        .def("createIndependantMiddlesByPlane", &mat::Middles::createIndependantMiddlesByPlane)
        .def("getMiddles", &mat::Middles::getMiddles)
        .def("getMaterials", &mat::Middles::getMaterials)
        .def("getConcentrations", &mat::Middles::getConcentrations)
        .def("getReacNames", &mat::Middles::getReacNames)
        .def("getNbGroups", &mat::Middles::getNbGroups)
        .def("addMaterial", &mat::Middles::addMaterial)
        .def("addMiddleNoConc", py::overload_cast<const std::string, const std::string>(&mat::Middles::addMiddle))
        .def("addMiddle", py::overload_cast<const std::string, const std::string, std::map<std::string, double> &>(&mat::Middles::addMiddle))
        .def("setXsValue", &mat::Middles::setXsValue)
        .def("getXsValue", py::overload_cast<const std::string, const int, const std::string &, const std::string &>(&mat::Middles::getXsValue, py::const_))
        .def("getXsValue", py::overload_cast<const std::string, const int, const std::string &>(&mat::Middles::getXsValue, py::const_))
        .def("multXsValue", py::overload_cast<const std::string, const int, const std::string &, const std::string &, double>(&mat::Middles::multXsValue))
        .def("multXsValue", py::overload_cast<const std::string, const int, const std::string &, double>(&mat::Middles::multXsValue))
        .def("randomPerturbation", &mat::Middles::randomPerturbationPython);

    py::class_<mat::Macrolib>(materials, "Macrolib")
        // .def(py::init<const mat::Macrolib &>())
        .def(py::init<const mat::Middles &, const std::vector<std::vector<std::vector<std::string>>> &>())
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

    py::class_<solver::Solver>(solver, "Solver")
        .def("dump", &solver::Solver::dump, py::arg("file_name"), py::arg("suffix") = "")
        .def("load", &solver::Solver::load, py::arg("file_name"), py::arg("suffix") = "")
        .def("getVolumes", &solver::Solver::getVolumesPython)
        .def("makeAdjoint", &solver::Solver::makeAdjoint)
        .def("getEigenValues", &solver::Solver::getEigenValues)
        .def("getDominanceRatio", &solver::Solver::getDominanceRatio)
        .def("setDominanceRatio", &solver::Solver::setDominanceRatio)
        .def("getEigenVectors", &solver::Solver::getEigenVectors)
        .def("getVolumes", &solver::Solver::getVolumesPython)
        .def("getEigenVector", &solver::Solver::getEigenVectorPython)
        .def("getPower", &solver::Solver::getPowerPython)
        .def("getPowerNormVector", &solver::Solver::getPowerNormVector)
        .def("normVector", &solver::Solver::normVector)
        .def("normPower", &solver::Solver::normPowerPython,
             py::arg("power_W") = 1.)
        .def("normPhiStarMPhi", &solver::Solver::normPhiStarMPhi)
        .def("normPhi", &solver::Solver::normPhi)
        .def("removeEigenVectors", &solver::Solver::removeEigenVectors)
        .def("handleDenegeratedEigenvalues", &solver::Solver::handleDenegeratedEigenvalues,
             py::arg("max_eps") = 1e-6)
        .def("isOrthogonal", &solver::Solver::isOrthogonal,
             py::arg("max_eps") = 1e-6, py::arg("raise_error") = false);

    py::class_<solver::SolverFull<SpMat>, solver::Solver>(solver, "SolverFull")
        .def("getK", &solver::SolverFull<SpMat>::getK)
        .def("getM", &solver::SolverFull<SpMat>::getM);

    py::class_<solver::SolverFullPowerIt, solver::SolverFull<SpMat>>(solver, "SolverFullPowerIt")
        .def(py::init<const solver::SolverFullPowerIt &>())
        .def(py::init<vecd &, mat::Macrolib &, double, double>())
        .def(py::init<vecd &, vecd &, mat::Macrolib &, double, double, double, double>())
        .def(py::init<vecd &, vecd &, vecd &, mat::Macrolib &, double, double, double, double, double, double>())
        .def("solve", &solver::SolverFullPowerIt::solve,
             py::arg("tol") = 1e-6, py::arg("tol_eigen_vectors") = 1e-5,
             py::arg("nb_eigen_values") = 1, py::arg("v0") = Eigen::VectorXd(), py::arg("ev0") = 1.0,
             py::arg("tol_inner") = 1e-4, py::arg("outer_max_iter") = 500,
             py::arg("inner_max_iter") = 20, py::arg("inner_solver") = "BiCGSTAB",
             py::arg("inner_precond") = "", py::arg("acceleration") = "chebyshev");

    py::class_<solver::SolverFullSlepc, solver::SolverFull<SpMat>>(solver, "SolverFullSlepc")
        .def(py::init<const solver::SolverFullSlepc &>())
        .def(py::init<vecd &, mat::Macrolib &, double, double>())
        .def(py::init<vecd &, vecd &, mat::Macrolib &, double, double, double, double>())
        .def(py::init<vecd &, vecd &, vecd &, mat::Macrolib &, double, double, double, double, double, double>())
        .def("solve", py::overload_cast<double, int, const Eigen::VectorXd &, double, double, int, int, std::string, std::string, std::string>(&solver::SolverFullSlepc::solve),
             py::arg("tol") = 1e-6,
             py::arg("nb_eigen_values") = 1, py::arg("v0") = Eigen::VectorXd(), py::arg("ev0") = 1.0,
             py::arg("tol_inner") = 1e-4, py::arg("outer_max_iter") = 500,
             py::arg("inner_max_iter") = 20, py::arg("solver") = "krylovschur",
             py::arg("inner_solver") = "", py::arg("inner_precond") = "");

    py::class_<solver::SolverCond<SpMat>, solver::Solver>(solver, "SolverCond");

    py::class_<solver::SolverCondPowerIt, solver::SolverCond<SpMat>>(solver, "SolverCondPowerIt")
        .def(py::init<const solver::SolverCondPowerIt &>())
        .def(py::init<vecd &, mat::Macrolib &, double, double>())
        .def(py::init<vecd &, vecd &, mat::Macrolib &, double, double, double, double>())
        .def(py::init<vecd &, vecd &, vecd &, mat::Macrolib &, double, double, double, double, double, double>())
        .def("solve", &solver::SolverCondPowerIt::solve,
             py::arg("tol") = 1e-6, py::arg("tol_eigen_vectors") = 1e-5,
             py::arg("nb_eigen_values") = 1, py::arg("v0") = Eigen::VectorXd(), py::arg("ev0") = 1.0,
             py::arg("tol_inner") = 1e-4, py::arg("outer_max_iter") = 500,
             py::arg("inner_max_iter") = 20, py::arg("inner_solver") = "GMRES",
             py::arg("inner_precond") = "", py::arg("acceleration") = "");

    py::class_<solver::SolverFullFixedSource, solver::SolverFull<SpMat>>(solver, "SolverFullFixedSource")
        .def(py::init<const solver::SolverFullFixedSource &>())
        .def(py::init<const solver::SolverFull<SpMat> &, const solver::SolverFull<SpMat> &, const Eigen::VectorXd &>())
        .def("getGamma", &solver::SolverFullFixedSource::getGammaPython)
        .def("solve", &solver::SolverFullFixedSource::solve,
             py::arg("tol") = 1e-6, py::arg("tol_eigen_vectors") = 1e-5,
             py::arg("nb_eigen_values") = 1, py::arg("v0") = Eigen::VectorXd(), py::arg("ev0") = 1.0,
             py::arg("tol_inner") = 1e-4, py::arg("outer_max_iter") = 500,
             py::arg("inner_max_iter") = 20, py::arg("inner_solver") = "GMRES",
             py::arg("inner_precond") = "", py::arg("acceleration") = "chebyshev");

    py::module perturbation = m.def_submodule("perturbation", "A module for the perturbation.");
    perturbation.def("checkBiOrthogonality", &perturbation::checkBiOrthogonality<solver::SolverFull<SpMat>>,
                     py::arg("solver"), py::arg("solver_star"), py::arg("ev0") = 1.0,
                     py::arg("max_eps") = 1e-6, py::arg("raise_error") = false);
    perturbation.def("handleDegeneratedEigenvalues", &perturbation::handleDegeneratedEigenvalues<solver::SolverFull<SpMat>>);
    perturbation.def("firstOrderPerturbation", &perturbation::firstOrderPerturbation<solver::SolverFull<SpMat>>);
    perturbation.def("highOrderPerturbation", &perturbation::highOrderPerturbationPython<solver::SolverFull<SpMat>>);
    perturbation.def("firstOrderGPT", py::overload_cast<const solver::SolverFull<SpMat> &, const solver::SolverFull<SpMat> &, const solver::SolverFull<SpMat> &,
                                                        Eigen::VectorXd &, Eigen::VectorXd &,
                                                        Eigen::VectorXd &, Eigen::VectorXd &,
                                                        double, double, int, int,
                                                        std::string, std::string, std::string>(&perturbation::firstOrderGPT<solver::SolverFull<SpMat>>));
    // operators.def("diff_diffusion_op_3d", py::overload_cast<vecd &, vecd &, vecd &, mat::Macrolib &,
    //                                                         double, double, double, double, double, double>(&operators::diff_diffusion_op<SpMat, vecd>));
    // &perturbation::firstOrderGPT<solver::SolverFull<SpMat>>);

    py::class_<perturbation::EpGPT<solver::SolverFullPowerIt>>(perturbation, "EpGPT")
        .def(py::init<const perturbation::EpGPT<solver::SolverFullPowerIt> &>())
        .def(py::init<vecd &, mat::Middles &, const std::vector<std::vector<std::vector<std::string>>> &, double, double>())
        .def(py::init<vecd &, vecd &, mat::Middles &, const std::vector<std::vector<std::vector<std::string>>> &, double, double, double, double>())
        .def(py::init<vecd &, vecd &, vecd &, mat::Middles &, const std::vector<std::vector<std::vector<std::string>>> &, double, double, double, double, double, double>())
        .def("createBasis", &perturbation::EpGPT<solver::SolverFullPowerIt>::createBasis)
        .def("solveReference", &perturbation::EpGPT<solver::SolverFullPowerIt>::solveReference)
        .def("calcImportances", &perturbation::EpGPT<solver::SolverFullPowerIt>::calcImportances)
        .def("firstOrderPerturbation", &perturbation::EpGPT<solver::SolverFullPowerIt>::firstOrderPerturbation)
        .def("highOrderPerturbation", &perturbation::EpGPT<solver::SolverFullPowerIt>::highOrderPerturbation,
             py::arg("solver_pert"), py::arg("tol_eigen_value") = 1e-5, py::arg("max_iter") = 100, py::arg("basis_size") = -1)
        .def("getBasis", &perturbation::EpGPT<solver::SolverFullPowerIt>::getBasis)
        .def("getImportances", &perturbation::EpGPT<solver::SolverFullPowerIt>::getImportances)
        .def("getN_star", &perturbation::EpGPT<solver::SolverFullPowerIt>::getN_star)
        .def("getSolver", &perturbation::EpGPT<solver::SolverFullPowerIt>::getSolver)
        .def("getSolverStar", &perturbation::EpGPT<solver::SolverFullPowerIt>::getSolverStar)
        .def("setBasis", &perturbation::EpGPT<solver::SolverFullPowerIt>::setBasis)
        .def("setImportances", &perturbation::EpGPT<solver::SolverFullPowerIt>::setImportances)
        .def("setN_star", &perturbation::EpGPT<solver::SolverFullPowerIt>::setN_star)
        .def("setSolver", &perturbation::EpGPT<solver::SolverFullPowerIt>::setSolver)
        .def("setSolverStar", &perturbation::EpGPT<solver::SolverFullPowerIt>::setSolverStar)
        .def("dump", &perturbation::EpGPT<solver::SolverFullPowerIt>::dump)
        .def("load", &perturbation::EpGPT<solver::SolverFullPowerIt>::load);
}