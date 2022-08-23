#ifndef DIFF_OPERATOR_H
#define DIFF_OPERATOR_H

#include <vector>
#include <string_view>
#include <string>
#include <map>
#include <iostream>

#include <pybind11/eigen.h>

#include <Eigen/Sparse>
#include <petscmat.h>

#include "macrolib.h"

namespace py = pybind11;

namespace operators
{
    typedef Eigen::Triplet<double> Triplet;
    typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
    typedef std::vector<double> vecd;

    //matrix creation helper
    SpMat matrix_from_coeff(const std::vector<Triplet> &coefficients, int matrix_size);
    Mat matrix_from_coeff_petsc(const std::vector<Triplet> &coefficients, int matrix_size);

    //removal op todo: maybe add function with 1dview of a tensor??
    std::vector<Triplet> diff_removal_op_triplet(vecd &volumes_1d, mat::Macrolib &macrolib);
    SpMat diff_removal_op(vecd &volumes_1d, mat::Macrolib &macrolib);
    Mat diff_removal_op_petsc(vecd &volumes_1d, mat::Macrolib &macrolib);

    //fission op
    std::vector<Triplet> diff_fission_op_triplet(vecd &volumes_1d, mat::Macrolib &macrolib);
    SpMat diff_fission_op(vecd &volumes_1d, mat::Macrolib &macrolib);
    Mat diff_fission_op_petsc(vecd &volumes_1d, mat::Macrolib &macrolib);

    //scatering op
    std::vector<Triplet> diff_scatering_op_triplet(vecd &volumes_1d, mat::Macrolib &macrolib);
    SpMat diff_scatering_op(vecd &volumes_1d, mat::Macrolib &macrolib);
    Mat diff_scatering_op_petsc(vecd &volumes_1d, mat::Macrolib &macrolib);

    //diffusion op
    std::vector<Triplet> diff_diffusion_op_triplet(vecd &dx, mat::Macrolib &macrolib,
                                                   double albedo_x0, double albedo_xn);
    SpMat diff_diffusion_op(vecd &dx, mat::Macrolib &macrolib,
                            double albedo_x0, double albedo_xn);
    Mat diff_diffusion_op_petsc(vecd &dx, mat::Macrolib &macrolib,
                                double albedo_x0, double albedo_xn);

    std::vector<Triplet> diff_diffusion_op_triplet(vecd &dx, vecd &dy, mat::Macrolib &macrolib,
                                                   double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn);
    SpMat diff_diffusion_op(vecd &dx, vecd &dy, mat::Macrolib &macrolib,
                            double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn);
    Mat diff_diffusion_op_petsc(vecd &dx, vecd &dy, mat::Macrolib &macrolib,
                                double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn);

    std::vector<Triplet> diff_diffusion_op_triplet(vecd &dx, vecd &dy, vecd &dz, mat::Macrolib &macrolib,
                                                   double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn);
    SpMat diff_diffusion_op(vecd &dx, vecd &dy, vecd &dz, mat::Macrolib &macrolib,
                            double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn);
    Mat diff_diffusion_op_petsc(vecd &dx, vecd &dy, vecd &dz, mat::Macrolib &macrolib,
                                double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn);

} // namespace operators

#endif // DIFF_OPERATOR_H