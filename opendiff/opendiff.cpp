// python binding

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <vector>
#include <string_view>

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

#include <petscmat.h>
#include <slepceps.h>

#include "materials.h"
#include "macrolib.h"
#include "diff_operator.h"

namespace py = pybind11;

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
    py::module eigen = m.def_submodule("eigen", "A simple eigen tensor binding");
    eigen.def("eigenTensor3D", &eigenTensor3D<double>, py::return_value_policy::move,
              py::arg("inArray"));

    py::module materials = m.def_submodule("materials", "A module for materials and macrolib handling.");

    py::class_<mat::Materials>(materials, "Materials")
        .def(py::init<const mat::Materials &>())
        .def(py::init<const std::vector<Eigen::ArrayXXf> &, const std::vector<std::string> &,
                      const std::vector<std::string> &>())
        .def("getReacNames", &mat::Materials::getReacNames)
        .def("getMatNames", &mat::Materials::getMatNames)
        .def("getMaterial", &mat::Materials::getMaterial)
        .def("getMaterials", &mat::Materials::getMaterials)
        .def("getValue", &mat::Materials::getValue)
        .def("getNbGroups", py::overload_cast<>(&mat::Materials::getNbGroups, py::const_))
        .def("getValue", &mat::Materials::getReactionIndex)
        .def("addMaterial", &mat::Materials::addMaterial);

    py::class_<mat::Macrolib>(materials, "Macrolib")
        // .def(py::init<const mat::Macrolib &>())
        .def(py::init<const mat::Materials &, const std::vector<std::vector<std::vector<std::string>>> &>())
        .def("getReacNames", &mat::Macrolib::getReacNames)
        .def("getNbGroups", &mat::Macrolib::getNbGroups)
        .def("getGeometryNDim", &mat::Macrolib::getGeometryNDim)
        .def("getValues", &mat::Macrolib::getValuesPython)
        .def("getValues1D", &mat::Macrolib::getValues1DPython);

    py::module operators = m.def_submodule("operators", "A module for the operators' creation.");
    operators.def("init_petsc", PetscInitializeNoArguments);
    operators.def("diff_removal_op", &operators::diff_removal_op);
    operators.def("diff_fission_op", &operators::diff_fission_op);
    operators.def("diff_scatering_op", &operators::diff_scatering_op);

    typedef std::vector<double> vecd;

    operators.def("diff_diffusion_op_1d", py::overload_cast<vecd &, mat::Macrolib &, double, double>(&operators::diff_diffusion_op));
    operators.def("diff_diffusion_op_2d", py::overload_cast<vecd &, vecd &, mat::Macrolib &, double, double, double, double>(&operators::diff_diffusion_op));
    operators.def("diff_diffusion_op_3d", py::overload_cast<vecd &, vecd &, vecd &, mat::Macrolib &, double, double, double, double, double, double>(&operators::diff_diffusion_op));
}