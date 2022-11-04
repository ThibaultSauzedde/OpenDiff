#ifndef DIFF_OPERATOR_H
#define DIFF_OPERATOR_H

#include <vector>
#include <string_view>
#include <string>
#include <map>
#include <iostream>
#include "spdlog/spdlog.h"
#include <limits>

#include <pybind11/eigen.h>

#include <Eigen/Sparse>
#include <petscmat.h>

#include "macrolib.h"

namespace py = pybind11;

namespace operators
{
    using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    using Tensor1D = Eigen::Tensor<double, 1, Eigen::RowMajor>;
    using Tensor2D = Eigen::Tensor<double, 2, Eigen::RowMajor>;
    using Tensor3D = Eigen::Tensor<double, 3, Eigen::RowMajor>;
    using Tensor4D = Eigen::Tensor<double, 4, Eigen::RowMajor>;

    // using Triplet = Eigen::Triplet<double>;
    typedef Eigen::Triplet<double> Triplet;

    //
    // template for creating the matrix content in triplet for a given nrj group (usefull for the condensed form)
    //


    // removal op 
    template <typename V>
    void diff_removal_op_triplet(std::vector<Triplet> &coefficients, const int i_grp, V &volumes_1d, mat::Macrolib &macrolib,
                                 int offset_i = 0, int offset_j = 0);
    // fission op
    template <typename V>
    void diff_fission_op_triplet(std::vector<Triplet> &coefficients, const int i_grp, const int i_grp_p, V &volumes_1d, mat::Macrolib &macrolib,
                                 int offset_i = 0, int offset_j = 0);
    // scatering op
    template <typename V>
    void diff_scatering_op_triplet(std::vector<Triplet> &coefficients, const int i_grp, const int i_grp_p, V &volumes_1d, mat::Macrolib &macrolib,
                                   int offset_i = 0, int offset_j = 0);
    // diffusion op 1d
    template <typename V>
    void diff_diffusion_op_triplet(std::vector<Triplet> &coefficients, const int i_grp, V &dx, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn,
                                   int offset_i = 0, int offset_j = 0);
    // diffusion op 2d
    template <typename V>
    void diff_diffusion_op_triplet(std::vector<Triplet> &coefficients, const int i_grp, V &dx, V &dy, mat::Macrolib &macrolib,
                                   double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn,
                                   int offset_i = 0, int offset_j = 0);
    // diffusion op 3d
    template <typename V>
    void diff_diffusion_op_triplet(std::vector<Triplet> &coefficients, const int i_grp, V &dx, V &dy, V &dz, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn,
                                   double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn,
                                   int offset_i = 0, int offset_j = 0);

    // --------------------------------------------------------
                
    template <typename V>
    std::vector<Triplet> diff_removal_op_triplet(const int i_grp, V &volumes_1d, mat::Macrolib &macrolib,
                                                 int offset_i = 0, int offset_j = 0);
    // fission op
    template <typename V>
    std::vector<Triplet> diff_fission_op_triplet(const int i_grp, const int i_grp_p, V &volumes_1d, mat::Macrolib &macrolib,
                                                 int offset_i = 0, int offset_j = 0);

    template <typename V>
    std::vector<Triplet> diff_fission_op_triplet(const int i_grp, V &volumes_1d, mat::Macrolib &macrolib);

    template <typename V>
    std::vector<Triplet> diff_fission_spectrum_op_triplet(const int i_grp, V &volumes_1d, mat::Macrolib &macrolib);

    // scatering op
    template <typename V>
    std::vector<Triplet> diff_scatering_op_triplet(const int i_grp, const int i_grp_p, V &volumes_1d, mat::Macrolib &macrolib,
                                                   int offset_i = 0, int offset_j = 0);
    // diffusion op 1d
    template <typename V>
    std::vector<Triplet> diff_diffusion_op_triplet(const int i_grp, V &dx, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn,
                                                   int offset_i = 0, int offset_j = 0);
    // diffusion op 2d
    template <typename V>
    std::vector<Triplet> diff_diffusion_op_triplet(const int i_grp, V &dx, V &dy, mat::Macrolib &macrolib,
                                                   double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn,
                                                   int offset_i = 0, int offset_j = 0);
    // diffusion op 3d
    template <typename V>
    std::vector<Triplet> diff_diffusion_op_triplet(const int i_grp, V &dx, V &dy, V &dz, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn,
                                                   double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn,
                                                   int offset_i = 0, int offset_j = 0);


    //
    // template for creating the matrix content in triplet 
    //

    template <typename V>
    std::vector<Triplet> diff_removal_op_triplet(V &volumes_1d, mat::Macrolib &macrolib);
    // fission op
    template <typename V>
    std::vector<Triplet> diff_fission_op_triplet(V &volumes_1d, mat::Macrolib &macrolib);
    // scatering op
    template <typename V>
    std::vector<Triplet> diff_scatering_op_triplet(V &volumes_1d, mat::Macrolib &macrolib);
    // diffusion op 1d
    template <typename V>
    std::vector<Triplet> diff_diffusion_op_triplet(V &dx, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn);
    // diffusion op 2d
    template <typename V>
    std::vector<Triplet> diff_diffusion_op_triplet(V &dx, V &dy, mat::Macrolib &macrolib,
                                                   double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn);
    // diffusion op 3d
    template <typename V>
    std::vector<Triplet> diff_diffusion_op_triplet(V &dx, V &dy, V &dz, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn,
                                                   double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn);

    //
    // template for filling the slepc or eigen matrices
    //

    // template for filling the matrix (for slepc and eigen matrix)
    template <typename T>
    T matrix_from_coeff(const std::vector<Triplet> &coefficients, int matrix_size);

    template <typename T>
    void matrix_from_coeff(T &A, const std::vector<Triplet> &coefficients);

    // slepc specialisation
    template <>
    inline void matrix_from_coeff(Mat &A, const std::vector<Triplet> &coefficients);

    //REALLY SLOW, do not use it !!!! we convert eigen matrix to petsc one in the solve...
    template <>
    inline Mat matrix_from_coeff(const std::vector<Triplet> &coefficients, int matrix_size);


    //
    // template operators for one group (for slepc and eigen matrix)
    //

    template <typename T, typename V>
    T diff_removal_op(const int i_grp, V &volumes_1d, mat::Macrolib &macrolib);

    template <typename T, typename V>
    T diff_fission_op(const int i_grp, const int i_grp_p, V &volumes_1d, mat::Macrolib &macrolib);

    template <typename T, typename V>
    T diff_fission_op(const int i_grp, V &volumes_1d, mat::Macrolib &macrolib);

    template <typename T, typename V>
    T diff_fission_spectrum_op(const int i_grp, V &volumes_1d, mat::Macrolib &macrolib);    

    template <typename T, typename V>
    T diff_scatering_op(const int i_grp, const int i_grp_p, V &volumes_1d, mat::Macrolib &macrolib);

    template <typename T, typename V>
    T diff_diffusion_op(const int i_grp, V &dx, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn);

    template <typename T, typename V>
    T diff_diffusion_op(const int i_grp, V &dx, V &dy, mat::Macrolib &macrolib,
                        double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn);

    template <typename T, typename V>
    T diff_diffusion_op(const int i_grp, V &dx, V &dy, V &dz, mat::Macrolib &macrolib,
                        double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn);

    //
    // template operators (for slepc and eigen matrix)
    //

    template <typename T, typename V>
    T diff_removal_op(V &volumes_1d, mat::Macrolib &macrolib);

    template <typename T, typename V>
    T diff_fission_op(V &volumes_1d, mat::Macrolib &macrolib);

    template <typename T, typename V>
    T diff_scatering_op(V &volumes_1d, mat::Macrolib &macrolib);

    template <typename T, typename V>
    T diff_diffusion_op(V &dx, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn);

    template <typename T, typename V>
    T diff_diffusion_op(V &dx, V &dy, mat::Macrolib &macrolib,
                        double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn);

    template <typename T, typename V>
    T diff_diffusion_op(V &dx, V &dy, V &dz, mat::Macrolib &macrolib,
                        double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn);


    //
    //template for creating M matrix (K is the fission operator)
    //
    template <typename T, typename V>
    T setup_m_operators(T &D, V volumes, mat::Macrolib &macrolib);

    template <>
    inline Mat setup_m_operators(Mat &D, Tensor1D volumes, mat::Macrolib &macrolib);

    //
    //template for creating the cond operators 
    //
    template <typename T, typename V>
    void setup_cond_operators(std::vector<T> &F, std::vector<T> &chi, std::vector<T> &A, std::vector<std::vector<T>> &S,
                           T &D, V volumes, mat::Macrolib &macrolib);

    #include "diff_operator.inl"

} // namespace operators

#endif // DIFF_OPERATOR_H