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
    typedef Eigen::Triplet<double> Triplet;

    // template for creating the matrix content in triplet

    // removal op
    template <typename V>
    std::vector<Triplet> diff_removal_op_triplet(V &volumes_1d, mat::Macrolib &macrolib)
    {
        auto nb_groups = macrolib.getNbGroups();
        int nb_cells = static_cast<int>(volumes_1d.size());

        std::vector<Triplet> coefficients{};
        coefficients.reserve(nb_groups * nb_cells);

        for (int g{0}; g < nb_groups; ++g)
        {
            for (int i{0}; i < nb_cells; ++i)
            {
                coefficients.push_back(Triplet(g + i * nb_groups, g + i * nb_groups, macrolib.getValues1D(g + 1, "SIGR")[i] * volumes_1d[i]));
            }
        }

        return coefficients;
    }

    // fission op
    template <typename V>
    std::vector<Triplet> diff_fission_op_triplet(V &volumes_1d, mat::Macrolib &macrolib)
    {
        auto nb_groups = macrolib.getNbGroups();
        int nb_cells = static_cast<int>(volumes_1d.size());

        std::vector<Triplet> coefficients{};
        coefficients.reserve(nb_groups * nb_groups * nb_cells);

        for (int i{0}; i < nb_cells; ++i)
        {
            for (int g{0}; g < nb_groups; ++g)
            {
                for (int gp{0}; gp < nb_groups; ++gp)
                {
                    auto t = Triplet(i * nb_groups + g, i * nb_groups + gp, macrolib.getValues1D(g + 1, "CHI")[i] * macrolib.getValues1D(gp + 1, "NU_SIGF")[i] * volumes_1d[i]);
                    coefficients.push_back(t);
                }
            }
        }

        return coefficients;
    }

    // scatering op
    template <typename V>
    std::vector<Triplet> diff_scatering_op_triplet(V &volumes_1d, mat::Macrolib &macrolib)
    {
        auto nb_groups = macrolib.getNbGroups();
        int nb_cells = static_cast<int>(volumes_1d.size());

        std::vector<Triplet> coefficients{};
        coefficients.reserve(nb_groups * nb_groups * nb_cells);

        for (int i{0}; i < nb_cells; ++i)
        {
            for (int g{0}; g < nb_groups; ++g)
            {
                for (int gp{0}; gp < nb_groups; ++gp)
                {
                    auto t = Triplet(i * nb_groups + g, i * nb_groups + gp, macrolib.getValues1D(gp + 1, std::to_string(g + 1))[i] * volumes_1d[i]);
                    coefficients.push_back(t);
                }
            }
        }

        return coefficients;
    }

    // diffusion op 1d
    template <typename V>
    std::vector<Triplet> diff_diffusion_op_triplet(V &dx, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn)
    {
        auto nb_groups = macrolib.getNbGroups();
        int nb_cells = static_cast<int>(dx.size());

        // left and right C
        Eigen::Tensor<double, 2> diff_coeff_1d(nb_groups, nb_cells);
        for (int grp{0}; grp < nb_groups; ++grp)
        {
            diff_coeff_1d.chip(grp, 0) = macrolib.getValues1D(grp + 1, "D");
        }

        auto C_x0 = 2 * (diff_coeff_1d.chip(0, 1) * (1 - albedo_x0)) / (4 * diff_coeff_1d.chip(0, 1) * (1 + albedo_x0) + dx[0] * (1 - albedo_x0));
        auto C_xn = 2 * (diff_coeff_1d.chip(nb_cells - 1, 1) * (1 - albedo_xn)) / (4 * diff_coeff_1d.chip(nb_cells - 1, 1) * (1 + albedo_xn) + dx[nb_cells - 1] * (1 - albedo_xn));

        // midle C
        Eigen::Tensor<double, 2> C(nb_cells + 1, nb_groups);
        for (int i{0}; i < nb_cells - 1; ++i)
        {
            auto C_i = 2 * (diff_coeff_1d.chip(i, 1) * diff_coeff_1d.chip(i + 1, 1)) /
                       (dx[i + 1] * diff_coeff_1d.chip(i, 1) + dx[i] * diff_coeff_1d.chip(i + 1, 1));
            C.chip(i + 1, 0) = C_i;
        }
        C.chip(0, 0) = C_x0;
        C.chip(nb_cells, 0) = C_xn;

        // create the coefficients (only diagonals and sub diagonals)
        std::vector<Triplet> coefficients{};
        coefficients.reserve(nb_groups * nb_cells + 2 * nb_groups * (nb_cells - 1));

        for (int g{0}; g < nb_groups; ++g)
        {
            for (int i{0}; i < nb_cells; ++i)
            {
                int id = g + i * nb_groups;
                // diagonal term
                coefficients.push_back(Triplet(id, id, -(C(i, g) + C(i + 1, g))));

                // sub diagonal terms
                if (i != nb_cells - 1)
                {
                    coefficients.push_back(Triplet(id, id + nb_groups, C(i, g)));
                    coefficients.push_back(Triplet(id + nb_groups, id, C(i, g)));
                }
            }
        }

        return coefficients;
    }

    // diffusion op 2d
    template <typename V>
    std::vector<Triplet> diff_diffusion_op_triplet(V &dx, V &dy, mat::Macrolib &macrolib,
                                                   double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn)
    {
        auto nb_groups = macrolib.getNbGroups();
        int dx_size = static_cast<int>(dx.size());
        int dy_size = static_cast<int>(dy.size());
        int nb_cells = dx_size * dy_size;

        // left and right C
        Eigen::Tensor<double, 3> diff_coeff_2d(nb_groups, dy_size, dx_size);
        for (int grp{0}; grp < nb_groups; ++grp)
        {
            diff_coeff_2d.chip(grp, 0) = macrolib.getValues(grp + 1, "D").chip(0, 0); // chip(0,0) remove the null z axis
        }

        // left and right C
        auto C_x0 = 2 * (diff_coeff_2d.chip(0, 2) * (1 - albedo_x0)) /
                    (4 * diff_coeff_2d.chip(0, 2) * (1 + albedo_x0) + dx[0] * (1 - albedo_x0));
        auto C_xn = 2 * (diff_coeff_2d.chip(dx_size - 1, 2) * (1 - albedo_xn)) /
                    (4 * diff_coeff_2d.chip(dx_size - 1, 2) * (1 + albedo_xn) + dx[dx_size - 1] * (1 - albedo_xn));

        // down and up C
        auto C_y0 = 2 * (diff_coeff_2d.chip(0, 1) * (1 - albedo_y0)) /
                    (4 * diff_coeff_2d.chip(0, 1) * (1 + albedo_y0) + dy[0] * (1 - albedo_y0));
        auto C_yn = 2 * (diff_coeff_2d.chip(dy_size - 1, 1) * (1 - albedo_yn)) /
                    (4 * diff_coeff_2d.chip(dy_size - 1, 1) * (1 + albedo_yn) + dy[dy_size - 1] * (1 - albedo_yn));

        // midle C
        Eigen::Tensor<double, 3> C_x(nb_groups, dy_size, dx_size + 1);
        for (int i{0}; i < dx_size + 1; ++i)
        {
            for (int j{0}; j < dy_size; ++j)
            {
                Eigen::array<Eigen::Index, 3> offsets = {0, j, i};
                Eigen::array<Eigen::Index, 3> offsets_m = {0, j, i - 1};
                Eigen::array<Eigen::Index, 3> extents = {nb_groups, 1, 1};

                if (i == 0)
                {
                    C_x.chip(j, 1).chip(i, 1) = C_x0.chip(j, 1); // C_x.chip(j, 1).chip(i, 1) == C_x.slice(offsets, extents) bit it is lvalue !
                }
                else if (i == dx_size)
                {
                    C_x.chip(j, 1).chip(i, 1) = C_xn.chip(j, 1);
                }
                else
                {
                    C_x.chip(j, 1).chip(i, 1) = 2 * (diff_coeff_2d.slice(offsets, extents) * diff_coeff_2d.slice(offsets_m, extents)) /
                                                (dx[i] * diff_coeff_2d.slice(offsets_m, extents) + dx[i - 1] * diff_coeff_2d.slice(offsets, extents));
                }
            }
        }

        Eigen::Tensor<double, 3> C_y(nb_groups, dy_size, dx_size + 1);
        for (int i{0}; i < dx_size; ++i)
        {
            for (int j{0}; j < dy_size + 1; ++j)
            {
                Eigen::array<Eigen::Index, 3> offsets = {0, j, i};
                Eigen::array<Eigen::Index, 3> offsets_m = {0, j - 1, i};
                Eigen::array<Eigen::Index, 3> extents = {nb_groups, 1, 1};

                if (j == 0)
                {
                    C_y.chip(j, 1).chip(i, 1) = C_y0.chip(i, 1); // C_y.chip(j, 1).chip(i, 1) == C_y.slice(offsets, extents) bit it is lvalue !
                }
                else if (j == dy_size)
                {
                    C_y.chip(j, 1).chip(i, 1) = C_yn.chip(i, 1);
                }
                else
                {
                    C_y.chip(j, 1).chip(i, 1) = 2 * (diff_coeff_2d.slice(offsets, extents) * diff_coeff_2d.slice(offsets_m, extents)) /
                                                (dy[j] * diff_coeff_2d.slice(offsets_m, extents) + dy[j - 1] * diff_coeff_2d.slice(offsets, extents));
                }
            }
        }

        // create the coefficients (only diagonals and sub diagonals)
        std::vector<Triplet> coefficients{};
        coefficients.reserve(nb_groups * nb_cells + 2 * nb_groups * (dx_size - 1) * dy_size + 2 * nb_groups * dx_size * (dy_size - 1));

        for (int g{0}; g < nb_groups; ++g)
        {
            for (int i{0}; i < dx_size; ++i)
            {
                for (int j{0}; j < dy_size; ++j)
                {
                    // diagonal term
                    int id = g + i * nb_groups + j * nb_groups * dx_size;
                    coefficients.push_back(Triplet(id, id,
                                                   -dy[j] * (C_x(g, j, i) + C_x(g, j, i + 1)) - dx[i] * (C_y(g, j, i) + C_y(g, j + 1, i))));

                    // nb_groups sub diagonal terms
                    if (i != dx_size - 1)
                    {
                        coefficients.push_back(Triplet(id, id + nb_groups,
                                                       dy[j] * C_x(g, j, i + 1)));
                        coefficients.push_back(Triplet(id + nb_groups, id,
                                                       dy[j] * C_x(g, j, i + 1)));
                    }

                    // nb_groups* dx_size sub diagonal terms
                    if (j != dy_size - 1)
                    {
                        coefficients.push_back(Triplet(id, id + nb_groups * dx_size,
                                                       dx[j] * C_y(g, j + 1, i)));
                        coefficients.push_back(Triplet(id + nb_groups * dx_size, id,
                                                       dx[j] * C_y(g, j + 1, i)));
                    }
                }
            }
        }

        return coefficients;
    }

    // diffusion op 3d
    template <typename V>
    std::vector<Triplet> diff_diffusion_op_triplet(V &dx, V &dy, V &dz, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn,
                                                   double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn)
    {
        auto nb_groups = macrolib.getNbGroups();
        int dx_size = static_cast<int>(dx.size());
        int dy_size = static_cast<int>(dy.size());
        int dz_size = static_cast<int>(dz.size());
        int nb_cells = dx_size * dy_size * dz_size;

        // left and right C
        Eigen::Tensor<double, 4> diff_coeff_3d(nb_groups, dz_size, dy_size, dx_size);
        for (int grp{0}; grp < nb_groups; ++grp)
        {
            diff_coeff_3d.chip(grp, 0) = macrolib.getValues(grp + 1, "D");
        }

        // left and right C
        auto C_x0 = 2 * (diff_coeff_3d.chip(0, 3) * (1 - albedo_x0)) /
                    (4 * diff_coeff_3d.chip(0, 3) * (1 + albedo_x0) + dx[0] * (1 - albedo_x0));
        auto C_xn = 2 * (diff_coeff_3d.chip(dx_size - 1, 3) * (1 - albedo_xn)) /
                    (4 * diff_coeff_3d.chip(dx_size - 1, 3) * (1 + albedo_xn) + dx[dx_size - 1] * (1 - albedo_xn));

        // down and up C
        auto C_y0 = 2 * (diff_coeff_3d.chip(0, 2) * (1 - albedo_y0)) /
                    (4 * diff_coeff_3d.chip(0, 2) * (1 + albedo_y0) + dx[0] * (1 - albedo_y0));
        auto C_yn = 2 * (diff_coeff_3d.chip(dy_size - 1, 2) * (1 - albedo_yn)) /
                    (4 * diff_coeff_3d.chip(dy_size - 1, 2) * (1 + albedo_yn) + dx[dy_size - 1] * (1 - albedo_yn));

        // down and up C
        auto C_z0 = 2 * (diff_coeff_3d.chip(0, 1) * (1 - albedo_z0)) /
                    (4 * diff_coeff_3d.chip(0, 1) * (1 + albedo_z0) + dz[0] * (1 - albedo_z0));
        auto C_zn = 2 * (diff_coeff_3d.chip(dz_size - 1, 1) * (1 - albedo_zn)) /
                    (4 * diff_coeff_3d.chip(dz_size - 1, 1) * (1 + albedo_zn) + dz[dz_size - 1] * (1 - albedo_zn));

        // midle C
        Eigen::Tensor<double, 4> C_x(nb_groups, dz_size, dy_size, dx_size + 1);
        for (int i{0}; i < dx_size + 1; ++i)
        {
            for (int j{0}; j < dy_size; ++j)
            {
                for (int k{0}; k < dz_size; ++k)
                {
                    Eigen::array<Eigen::Index, 4> offsets = {0, k, j, i};
                    Eigen::array<Eigen::Index, 4> offsets_m = {0, k, j, i - 1};
                    Eigen::array<Eigen::Index, 4> extents = {nb_groups, 1, 1, 1};

                    Eigen::array<Eigen::Index, 3> offsets_c = {0, k, j};
                    Eigen::array<Eigen::Index, 3> extents_c = {nb_groups, 1, 1};

                    if (i == 0)
                    {
                        C_x.chip(k, 1).chip(j, 1).chip(i, 1) = C_x0.slice(offsets_c, extents_c); // C_x.chip(j, 1).chip(i, 1) == C_x.slice(offsets, extents) bit it is lvalue !
                    }
                    else if (i == dx_size)
                    {
                        C_x.chip(k, 1).chip(j, 1).chip(i, 1) = C_xn.slice(offsets_c, extents_c);
                    }
                    else
                    {
                        C_x.chip(k, 1).chip(j, 1).chip(i, 1) = 2 * (diff_coeff_3d.slice(offsets, extents) * diff_coeff_3d.slice(offsets_m, extents)) /
                                                               (dx[i] * diff_coeff_3d.slice(offsets_m, extents) + dx[i - 1] * diff_coeff_3d.slice(offsets, extents));
                    }
                }
            }
        }

        Eigen::Tensor<double, 4> C_y(nb_groups, dz_size, dy_size + 1, dx_size);
        for (int i{0}; i < dx_size; ++i)
        {
            for (int j{0}; j < dy_size + 1; ++j)
            {
                for (int k{0}; k < dz_size; ++k)
                {
                    Eigen::array<Eigen::Index, 4> offsets = {0, k, j, i};
                    Eigen::array<Eigen::Index, 4> offsets_m = {0, k, j - 1, i};
                    Eigen::array<Eigen::Index, 4> extents = {nb_groups, 1, 1, 1};

                    Eigen::array<Eigen::Index, 3> offsets_c = {0, k, i};
                    Eigen::array<Eigen::Index, 3> extents_c = {nb_groups, 1, 1};

                    if (i == 0)
                    {
                        C_y.chip(k, 1).chip(j, 1).chip(i, 1) = C_y0.slice(offsets_c, extents_c); // C_y.chip(j, 1).chip(i, 1) == C_y.slice(offsets, extents) bit it is lvalue !
                    }
                    else if (i == dx_size)
                    {
                        C_y.chip(k, 1).chip(j, 1).chip(i, 1) = C_yn.slice(offsets_c, extents_c);
                    }
                    else
                    {
                        C_y.chip(k, 1).chip(j, 1).chip(i, 1) = 2 * (diff_coeff_3d.slice(offsets, extents) * diff_coeff_3d.slice(offsets_m, extents)) /
                                                               (dy[j] * diff_coeff_3d.slice(offsets_m, extents) + dy[j - 1] * diff_coeff_3d.slice(offsets, extents));
                    }
                }
            }
        }

        Eigen::Tensor<double, 4> C_z(nb_groups, dz_size + 1, dy_size, dx_size);
        for (int i{0}; i < dx_size; ++i)
        {
            for (int j{0}; j < dy_size; ++j)
            {
                for (int k{0}; k < dz_size + 1; ++k)
                {
                    Eigen::array<Eigen::Index, 4> offsets = {0, k, j, i};
                    Eigen::array<Eigen::Index, 4> offsets_m = {0, k - 1, j, i};
                    Eigen::array<Eigen::Index, 4> extents = {nb_groups, 1, 1, 1};

                    Eigen::array<Eigen::Index, 3> offsets_c = {0, j, i};
                    Eigen::array<Eigen::Index, 3> extents_c = {nb_groups, 1, 1};

                    if (i == 0)
                    {
                        C_z.chip(k, 1).chip(j, 1).chip(i, 1) = C_z0.slice(offsets_c, extents_c); // C_z.chip(j, 1).chip(i, 1) == C_z.slice(offsets, extents) bit it is lvalue !
                    }
                    else if (i == dx_size)
                    {
                        C_z.chip(k, 1).chip(j, 1).chip(i, 1) = C_zn.slice(offsets_c, extents_c);
                    }
                    else
                    {
                        C_z.chip(k, 1).chip(j, 1).chip(i, 1) = 2 * (diff_coeff_3d.slice(offsets, extents) * diff_coeff_3d.slice(offsets_m, extents)) /
                                                               (dz[k] * diff_coeff_3d.slice(offsets_m, extents) + dz[k - 1] * diff_coeff_3d.slice(offsets, extents));
                    }
                }
            }
        }

        // create the coefficients (only diagonals and sub diagonals)
        std::vector<Triplet> coefficients{};
        coefficients.reserve(nb_groups * nb_cells + 2 * nb_groups * (dx_size - 1) * dy_size * dz_size +
                             2 * nb_groups * dx_size * (dy_size - 1) * dz_size + 2 * nb_groups * dx_size * dy_size * (dz_size - 1));

        for (int g{0}; g < nb_groups; ++g)
        {
            for (int i{0}; i < dx_size; ++i)
            {
                for (int j{0}; j < dy_size; ++j)
                {
                    for (int k{0}; k < dz_size; ++k)
                    {
                        // diagonal term
                        int id = g + i * nb_groups + j * nb_groups * dx_size + k * nb_groups * dx_size * dy_size;
                        coefficients.push_back(Triplet(id, id,
                                                       -dy[j] * dz[k] * (C_x(g, k, j, i) + C_x(g, k, j, i + 1)) -
                                                           dx[i] * dz[k] * (C_y(g, k, j, i) + C_y(g, k, j + 1, i)) -
                                                           dx[i] * dy[j] * (C_z(g, k, j, i) + C_z(g, k + 1, j, i))));

                        // nb_groups sub diagonal terms
                        if (i != dx_size - 1)
                        {
                            coefficients.push_back(Triplet(id, id + nb_groups,
                                                           dy[j] * dz[k] * C_x(g, k, j, i + 1)));
                            coefficients.push_back(Triplet(id + nb_groups, id,
                                                           dy[j] * dz[k] * C_x(g, k, j, i + 1)));
                        }

                        // nb_groups* dx_size sub diagonal terms
                        if (j != dy_size - 1)
                        {
                            coefficients.push_back(Triplet(id, id + nb_groups * dx_size,
                                                           dx[i] * dz[k] * C_y(g, k, j + 1, i)));
                            coefficients.push_back(Triplet(id + nb_groups * dx_size, id,
                                                           dx[i] * dz[k] * C_y(g, k, j + 1, i)));
                        }

                        // nb_groups* dx_size * dy_size sub diagonal terms
                        if (k != dz_size - 1)
                        {
                            coefficients.push_back(Triplet(id, id + nb_groups * dx_size * dy_size,
                                                           dx[i] * dy[j] * C_z(g, k + 1, j, i)));
                            coefficients.push_back(Triplet(id + nb_groups * dx_size * dy_size, id,
                                                           dx[i] * dy[j] * C_z(g, k + 1, j, i)));
                        }
                    }
                }
            }
        }

        return coefficients;
    }

    // template for filling the matrix (for slepc and eigen matrix)
    template <typename T>
    T matrix_from_coeff(const std::vector<Triplet> &coefficients, int matrix_size)
    {
        T A(matrix_size, matrix_size);
        A.setFromTriplets(coefficients.begin(), coefficients.end());
        return A;
    }
    template <typename T>
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
    }

    //REALLY SLOW, do not use it !!!! we convert eigen matrix to petsc one in the solve...
    template <>
    inline Mat matrix_from_coeff(const std::vector<Triplet> &coefficients, int matrix_size)
    {
        enum
        {
            IsRowMajor = Eigen::SparseMatrix<double>::IsRowMajor
        };

        typename Eigen::SparseMatrix<double>::IndexVector wi(matrix_size);

        // pass 1: count the nnz per inner-vector
        wi.setZero();
        for (auto it(coefficients.begin()); it != coefficients.end(); ++it)
        {
            eigen_assert(it->row() >= 0 && it->row() < mat.rows() && it->col() >= 0 && it->col() < mat.cols());
            wi(IsRowMajor ? it->col() : it->row())++;
        }

        Mat A;

        MatCreate(PETSC_COMM_WORLD, &A);
        MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, matrix_size, matrix_size);
        MatSeqAIJSetPreallocation(A, 0, wi.data());

        // MatCreateSeqAIJ(PETSC_COMM_WORLD, matrix_size, matrix_size, 0, wi.data(), &A);
        // matrix_from_coeff<Mat>(A, coefficients);
        PetscInt Istart, Iend;
        MatSetFromOptions(A);
        MatSetUp(A);
        MatGetOwnershipRange(A, &Istart, &Iend);

        int i = 0;
        for (auto t : coefficients)
        {
            MatSetValue(A, t.row(), t.col(), t.value(), INSERT_VALUES);
            i++;
        }

        MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

        // MatView(A, PETSC_VIEWER_STDOUT_WORLD);

        return A;
    }

    // template operators (for slepc and eigen matrix)
    template <typename T, typename V>
    T diff_removal_op(V &volumes_1d, mat::Macrolib &macrolib)
    {
        auto coefficients = diff_removal_op_triplet(volumes_1d, macrolib);
        int matrix_size = macrolib.getNbGroups() * static_cast<int>(volumes_1d.size());
        return matrix_from_coeff<T>(coefficients, matrix_size);
    }

    template <typename T, typename V>
    T diff_fission_op(V &volumes_1d, mat::Macrolib &macrolib)
    {
        auto coefficients = diff_fission_op_triplet(volumes_1d, macrolib);
        int matrix_size = macrolib.getNbGroups() * static_cast<int>(volumes_1d.size());
        return matrix_from_coeff<T>(coefficients, matrix_size);
    }

    template <typename T, typename V>
    T diff_scatering_op(V &volumes_1d, mat::Macrolib &macrolib)
    {
        auto coefficients = diff_scatering_op_triplet(volumes_1d, macrolib);
        int matrix_size = macrolib.getNbGroups() * static_cast<int>(volumes_1d.size());
        return matrix_from_coeff<T>(coefficients, matrix_size);
    }

    template <typename T, typename V>
    T diff_diffusion_op(V &dx, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn)
    {
        auto coefficients = diff_diffusion_op_triplet(dx, macrolib, albedo_x0, albedo_xn);
        int matrix_size = macrolib.getNbGroups() * static_cast<int>(dx.size());
        return matrix_from_coeff<T>(coefficients, matrix_size);
    }

    template <typename T, typename V>
    T diff_diffusion_op(V &dx, V &dy, mat::Macrolib &macrolib,
                        double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn)
    {
        auto coefficients = diff_diffusion_op_triplet(dx, dy, macrolib, albedo_x0, albedo_xn, albedo_y0, albedo_yn);
        int matrix_size = macrolib.getNbGroups() * static_cast<int>(dx.size() * dy.size());
        return matrix_from_coeff<T>(coefficients, matrix_size);
    }

    template <typename T, typename V>
    T diff_diffusion_op(V &dx, V &dy, V &dz, mat::Macrolib &macrolib,
                        double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn)
    {
        auto coefficients = diff_diffusion_op_triplet(dx, dy, dz, macrolib, albedo_x0, albedo_xn,
                                                      albedo_y0, albedo_yn, albedo_z0, albedo_zn);
        int matrix_size = macrolib.getNbGroups() * static_cast<int>(dx.size() * dy.size() * dz.size());
        return matrix_from_coeff<T>(coefficients, matrix_size);
    }

    template <typename T, typename V>
    void diff_removal_op(T &A, V &volumes_1d, mat::Macrolib &macrolib)
    {
        auto coefficients = diff_removal_op_triplet(volumes_1d, macrolib);
        matrix_from_coeff<T>(A, coefficients);
    }

    template <typename T, typename V>
    void diff_fission_op(T &A, V &volumes_1d, mat::Macrolib &macrolib)
    {
        auto coefficients = diff_fission_op_triplet(volumes_1d, macrolib);
        matrix_from_coeff<T>(A, coefficients);
    }

    template <typename T, typename V>
    void diff_scatering_op(T &A, V &volumes_1d, mat::Macrolib &macrolib)
    {
        auto coefficients = diff_scatering_op_triplet(volumes_1d, macrolib);
        matrix_from_coeff<T>(A, coefficients);
    }

    template <typename T, typename V>
    void diff_diffusion_op(T &A, V &dx, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn)
    {
        auto coefficients = diff_diffusion_op_triplet(dx, macrolib, albedo_x0, albedo_xn);
        matrix_from_coeff<T>(A, coefficients);
    }

    template <typename T, typename V>
    void diff_diffusion_op(T &A, V &dx, V &dy, mat::Macrolib &macrolib,
                           double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn)
    {
        auto coefficients = diff_diffusion_op_triplet(dx, dy, macrolib, albedo_x0, albedo_xn, albedo_y0, albedo_yn);
        matrix_from_coeff<T>(A, coefficients);
    }

    template <typename T, typename V>
    void diff_diffusion_op(T &A, V &dx, V &dy, V &dz, mat::Macrolib &macrolib,
                           double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn)
    {
        auto coefficients = diff_diffusion_op_triplet(dx, dy, dz, macrolib, albedo_x0, albedo_xn,
                                                      albedo_y0, albedo_yn, albedo_z0, albedo_zn);
        matrix_from_coeff<T>(A, coefficients);
    }

    //template for creating M matrix (K is the fission operator)
    template <typename T, typename V>
    T setup_m_operators(T &D, V volumes, mat::Macrolib &macrolib)
    {
        auto R = operators::diff_removal_op<T, V>(volumes, macrolib);
        auto S = operators::diff_scatering_op<T, V>(volumes, macrolib);
        auto M = R - S - D;
        return M;
    }

    using Tensor1D = Eigen::Tensor<double, 1>;
    template <>
    inline Mat setup_m_operators(Mat &D, Tensor1D volumes, mat::Macrolib &macrolib)
    {
        auto M = operators::diff_removal_op<Mat, Tensor1D>(volumes, macrolib);

        auto S = operators::diff_scatering_op<Mat, Tensor1D>(volumes, macrolib);

        MatAXPY(M, -1.0, S, SAME_NONZERO_PATTERN);
        MatAXPY(M, -1.0, D, SAME_NONZERO_PATTERN);
        return M;
    }

} // namespace operators

#endif // DIFF_OPERATOR_H