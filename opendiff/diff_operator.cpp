#include <Eigen/Sparse>
#include <unsupported/Eigen/CXX11/Tensor>
#include <petscmat.h>

#include "diff_operator.h"
#include "macrolib.h"

namespace py = pybind11;

namespace operators
{
    using Triplet = Eigen::Triplet<double>;
    using SpMat = Eigen::SparseMatrix<double>; // declares a column-major sparse matrix type of double
    using vecd = std::vector<double>;


    //removal op
    std::vector<Triplet> diff_removal_op_triplet(vecd &volumes_1d, mat::Macrolib &macrolib)
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


    //fission op
    std::vector<Triplet> diff_fission_op_triplet(vecd &volumes_1d, mat::Macrolib &macrolib)
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


    //scatering op
    std::vector<Triplet> diff_scatering_op_triplet(vecd &volumes_1d, mat::Macrolib &macrolib)
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


    //diffusion op 1d
    std::vector<Triplet> diff_diffusion_op_triplet(vecd &dx, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn)
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

        //create the coefficients (only diagonals and sub diagonals)
        std::vector<Triplet> coefficients{};
        coefficients.reserve(nb_groups * nb_cells + 2 * nb_groups * (nb_cells - 1));

        for (int g{0}; g < nb_groups; ++g)
        {
            for (int i{0}; i < nb_cells; ++i)
            {
                //diagonal term
                coefficients.push_back(Triplet(g + i * nb_groups, g + i * nb_groups, -(C(i, g) + C(i + 1, g))));

                //sub diagonal terms
                if (i > 0)
                {
                    coefficients.push_back(Triplet(g + i * nb_groups, g + i * nb_groups + nb_groups, C(i, g)));
                    coefficients.push_back(Triplet(g + i * nb_groups + nb_groups, g + i * nb_groups, C(i, g)));
                }
            }
        }

        return coefficients;
    }


    //diffusion op 2d
    std::vector<Triplet> diff_diffusion_op_triplet(vecd &dx, vecd &dy, mat::Macrolib &macrolib,
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

        //create the coefficients (only diagonals and sub diagonals)
        std::vector<Triplet> coefficients{};
        coefficients.reserve(nb_groups * nb_cells + 2 * nb_groups * (dx_size - 1) * dy_size + 2 * nb_groups * dx_size * (dy_size - 1));

        for (int g{0}; g < nb_groups; ++g)
        {
            for (int i{0}; i < dx_size; ++i)
            {
                for (int j{0}; j < dy_size; ++j)
                {
                    //diagonal term
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


    //diffusion op 3d
    std::vector<Triplet> diff_diffusion_op_triplet(vecd &dx, vecd &dy, vecd &dz, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn,
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

        //create the coefficients (only diagonals and sub diagonals)
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
                        //diagonal term
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


} // namespace operators
