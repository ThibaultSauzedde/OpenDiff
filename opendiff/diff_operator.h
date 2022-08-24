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
    // using Triplet = Eigen::Triplet<double>;
    using SpMat = Eigen::SparseMatrix<double>; // declares a column-major sparse matrix type of double
    using vecd = std::vector<double>;
    typedef Eigen::Triplet<double> Triplet;

    // removal op todo: maybe add function with 1dview of a tensor??
    std::vector<Triplet> diff_removal_op_triplet(vecd &volumes_1d, mat::Macrolib &macrolib);
    //fission op
    std::vector<Triplet> diff_fission_op_triplet(vecd &volumes_1d, mat::Macrolib &macrolib);
    //scatering op
    std::vector<Triplet> diff_scatering_op_triplet(vecd &volumes_1d, mat::Macrolib &macrolib);
    //diffusion op
    std::vector<Triplet> diff_diffusion_op_triplet(vecd &dx, mat::Macrolib &macrolib,
                                                   double albedo_x0, double albedo_xn);
    std::vector<Triplet> diff_diffusion_op_triplet(vecd &dx, vecd &dy, mat::Macrolib &macrolib,
                                                   double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn);
    std::vector<Triplet> diff_diffusion_op_triplet(vecd &dx, vecd &dy, vecd &dz, mat::Macrolib &macrolib,
                                                   double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn);


    // template for filling the matrix (for slepc and eigen matrix)
    template <class T>
    T matrix_from_coeff(const std::vector<Triplet> &coefficients, int matrix_size)
    {
        T A(matrix_size, matrix_size);
        A.setFromTriplets(coefficients.begin(), coefficients.end());
        return A;
    }
    template <class T> 
    void matrix_from_coeff(T &A, const std::vector<Triplet> &coefficients)
    {
        A.setFromTriplets(coefficients.begin(), coefficients.end());
    }

    // slepc specialisation 
    template <> 
    inline void matrix_from_coeff(Mat &A, const std::vector<Triplet> &coefficients)
    {
        PetscInt Istart, Iend;
        MatSetFromOptions(A);
        MatSetUp(A);
        MatGetOwnershipRange(A, &Istart, &Iend);
        for (auto t : coefficients)
        {
            MatSetValue(A, t.row(), t.col(), t.value(), INSERT_VALUES);
        }

        MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); 

        // MatView(A, PETSC_VIEWER_STDOUT_WORLD);
    }

    template <>
    inline Mat matrix_from_coeff(const std::vector<Triplet> &coefficients, int matrix_size)
    {
        Mat A;

        MatCreate(PETSC_COMM_WORLD, &A);
        MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, matrix_size, matrix_size);

        matrix_from_coeff<Mat>(A, coefficients);

        return A;
    }


    // template operators (for slepc and eigen matrix)
    template <class T>
    T diff_removal_op(vecd &volumes_1d, mat::Macrolib &macrolib)
    {
        auto coefficients = diff_removal_op_triplet(volumes_1d, macrolib);
        int matrix_size = macrolib.getNbGroups() * static_cast<int>(volumes_1d.size());
        return matrix_from_coeff<T>(coefficients, matrix_size);
    }

    template <class T> T diff_fission_op(vecd &volumes_1d, mat::Macrolib &macrolib)
    {
        auto coefficients = diff_fission_op_triplet(volumes_1d, macrolib);
        int matrix_size = macrolib.getNbGroups() * static_cast<int>(volumes_1d.size());
        return matrix_from_coeff<T>(coefficients, matrix_size);
    }

    template <class T> T diff_scatering_op(vecd &volumes_1d, mat::Macrolib &macrolib)
    {
        auto coefficients = diff_scatering_op_triplet(volumes_1d, macrolib);
        int matrix_size = macrolib.getNbGroups() * static_cast<int>(volumes_1d.size());
        return matrix_from_coeff<T>(coefficients, matrix_size);
    }

    template <class T> T diff_diffusion_op(vecd &dx, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn)
    {
        auto coefficients = diff_diffusion_op_triplet(dx, macrolib, albedo_x0, albedo_xn);
        int matrix_size = macrolib.getNbGroups() * static_cast<int>(dx.size());
        return matrix_from_coeff<T>(coefficients, matrix_size);
    }

    template <class T>
    T diff_diffusion_op(vecd &dx, vecd &dy, mat::Macrolib &macrolib,
                        double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn)
    {
        auto coefficients = diff_diffusion_op_triplet(dx, dy, macrolib, albedo_x0, albedo_xn, albedo_y0, albedo_yn);
        int matrix_size = macrolib.getNbGroups() * static_cast<int>(dx.size() * dy.size());
        return matrix_from_coeff<T>(coefficients, matrix_size);
    }

    template <class T>
    T diff_diffusion_op(vecd &dx, vecd &dy, vecd &dz, mat::Macrolib &macrolib,
                        double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn)
    {
        auto coefficients = diff_diffusion_op_triplet(dx, dy, dz, macrolib, albedo_x0, albedo_xn,
                                                      albedo_y0, albedo_yn, albedo_z0, albedo_zn);
        int matrix_size = macrolib.getNbGroups() * static_cast<int>(dx.size() * dy.size() * dz.size());
        return matrix_from_coeff<T>(coefficients, matrix_size);
    }

    template <class T>
    void diff_removal_op(T &A, vecd &volumes_1d, mat::Macrolib &macrolib)
    {
        auto coefficients = diff_removal_op_triplet(volumes_1d, macrolib);
        matrix_from_coeff<T>(A, coefficients);
    }

    template <class T>
    void diff_fission_op(T &A, vecd &volumes_1d, mat::Macrolib &macrolib)
    {
        auto coefficients = diff_fission_op_triplet(volumes_1d, macrolib);
        matrix_from_coeff<T>(A, coefficients);
    }

    template <class T>
    void diff_scatering_op(T &A, vecd &volumes_1d, mat::Macrolib &macrolib)
    {
        auto coefficients = diff_scatering_op_triplet(volumes_1d, macrolib);
        matrix_from_coeff<T>(A, coefficients);
    }

    template <class T> void diff_diffusion_op(T &A, vecd &dx, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn)
    {
        auto coefficients = diff_diffusion_op_triplet(dx, macrolib, albedo_x0, albedo_xn);
        matrix_from_coeff<T>(A, coefficients);
    }

    template <class T> void diff_diffusion_op(T &A, vecd &dx, vecd &dy, mat::Macrolib &macrolib,
                           double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn)
    {
        auto coefficients = diff_diffusion_op_triplet(dx, dy, macrolib, albedo_x0, albedo_xn, albedo_y0, albedo_yn);
        matrix_from_coeff<T>(A, coefficients);
    }

    template <class T>
    void diff_diffusion_op(T &A, vecd &dx, vecd &dy, vecd &dz, mat::Macrolib &macrolib,
                           double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn)
    {
        auto coefficients = diff_diffusion_op_triplet(dx, dy, dz, macrolib, albedo_x0, albedo_xn,
                                                      albedo_y0, albedo_yn, albedo_z0, albedo_zn);
        matrix_from_coeff<T>(A, coefficients);
    }

} // namespace operators

#endif // DIFF_OPERATOR_H