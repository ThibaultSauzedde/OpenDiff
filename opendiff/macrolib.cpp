#include <vector>
#include <iostream>
#include <string_view>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "macrolib.h"

// todo: make double a template in order to use simple or double precision !
namespace py = pybind11;

namespace mat
{
    using Tensor3D = Eigen::Tensor<double, 3, Eigen::RowMajor>;

    geometry_vector get_geometry_roded(geometry_vector geometry, vecd x, vecd y, vecd z,
                                       vector_tuple control_rod_pos, std::string rod_middle, std::string unroded_middle,
                                       std::vector<double> control_rod_zpos)
    {
        if (control_rod_pos.size() != control_rod_zpos.size())
            throw std::invalid_argument("The sizes of control_rod_pos and control_rod_zpos must be equals!");
        
        int nb_rod = static_cast<int>(control_rod_pos.size());
        for (int i = 0; i < nb_rod; i++)
        {
            geometry_tuple rod_description = control_rod_pos[i];
            double rod_zpos = control_rod_zpos[i];

            // assert control_rod_pos is ok with x, y and z
            double x1 = std::get<0>(rod_description);
            double x2 = std::get<1>(rod_description);
            double y1 = std::get<2>(rod_description);
            double y2 = std::get<3>(rod_description);
            double z1 = std::get<4>(rod_description);
            double z2 = std::get<5>(rod_description);

            int x1_index = std::distance(x.begin(), std::lower_bound(x.begin(), x.end(), x1 - 1e-5));
            int x2_index = std::distance(x.begin(), std::lower_bound(x.begin(), x.end(), x2 - 1e-5));
            int y1_index = std::distance(y.begin(), std::lower_bound(y.begin(), y.end(), y1 - 1e-5));
            int y2_index = std::distance(y.begin(), std::lower_bound(y.begin(), y.end(), y2 - 1e-5));
            int z1_index = std::distance(z.begin(), std::lower_bound(z.begin(), z.end(), z1 - 1e-5));
            int z2_index = std::distance(z.begin(), std::lower_bound(z.begin(), z.end(), z2 - 1e-5));

            if (x1_index == static_cast<int>(x.size()))
                throw std::invalid_argument("x1 cannot is not in the range of x.");
            if (x2_index == static_cast<int>(x.size()))
                throw std::invalid_argument("x2 cannot is not in the range of x.");

            if (y1_index == static_cast<int>(y.size()))
                throw std::invalid_argument("y1 cannot is not in the range of y.");
            if (y2_index == static_cast<int>(y.size()))
                throw std::invalid_argument("y2 cannot is not in the range of y.");

            if (z1_index == static_cast<int>(z.size()))
                throw std::invalid_argument("z1 cannot is not in the range of z.");
            if (z2_index == static_cast<int>(z.size()))
                throw std::invalid_argument("z2 cannot is not in the range of z.");

            int z_index = std::distance(z.begin(), std::lower_bound(z.begin(), z.end(), rod_zpos -1e-5));

            if (z_index < z1_index || z1_index > z2_index)
                throw std::invalid_argument("The rod position is not between z1 and z2.");
            
            for (int xi = x1_index; xi < x2_index; xi++)
            {
                for (int yi = y1_index; yi < y2_index; yi++)
                {
                    for (int zi = z1_index; zi < z2_index; zi++)
                    {
                        if (zi < z_index)
                            geometry[zi][yi][zi] = unroded_middle;
                        else
                            geometry[zi][yi][zi] = rod_middle;
                    }
                }
            }
        }
    
    return geometry;


    }

    Macrolib::Macrolib(const mat::Middles &middles, const Eigen::Tensor<std::string, 3, Eigen::RowMajor> &geometry)
    {
        setup(middles, geometry);
    }

    Macrolib::Macrolib(const mat::Middles &middles, const geometry_vector &geometry)
    {
        //copy of the geometry in a tensor, it does not cost that much :)

        Eigen::Tensor<std::string, 3, Eigen::RowMajor> geometry_tensor(static_cast<int>(geometry.size()), static_cast<int>(geometry[0].size()),
                                                      static_cast<int>(geometry[0][0].size()));

        int i = 0, j = 0, k = 0;

        auto i_size = geometry[0].size();
        auto j_size = geometry[0][0].size();

        for (std::vector i_val : geometry)
        {
            j = 0;
            if (i_size != i_val.size())
                throw std::invalid_argument("The size of the vector must be the same in each direction ! (y)");

            for (std::vector j_val : i_val)
            {
                k = 0;
                if (j_size != j_val.size())
                    throw std::invalid_argument("The size of the vector must be the same in each direction ! (x)");

                for (auto k_val : j_val)
                {
                    geometry_tensor(i, j, k) = k_val;
                    k++;
                }
                j++;
            }
            i++;
        }
        setup(middles, geometry_tensor);
    }

    void Macrolib::setup(const mat::Middles &middles, const Eigen::Tensor<std::string, 3, Eigen::RowMajor> &geometry)
    {
        m_nb_groups = middles.getNbGroups();
        m_reac_names = middles.getReacNames();
        auto dim_z = geometry.dimension(0), dim_y = geometry.dimension(1), dim_x = geometry.dimension(2);
        m_dim = {dim_x, dim_y, dim_z};
        Eigen::array<Eigen::DenseIndex, 1> one_dim({dim_z * dim_y * dim_x});

        // loop on reactions 
        for (int i_grp = 0; i_grp < m_nb_groups; i_grp++)
        {
            for (auto reac_name : m_reac_names)
            {
                Tensor3D reac_i(dim_z, dim_y, dim_x); // z, y, x
                // loop on the geometry
                for (int i = 0; i < dim_z; ++i)
                {
                    for (int j = 0; j < dim_y; ++j)
                    {
                        for (int k = 0; k < dim_x; ++k)
                        {
                            reac_i(i, j, k) = middles.getXsValue(geometry(i, j, k), i_grp + 1, reac_name);
                        }
                    }
                }
                m_values[{i_grp + 1, reac_name}] = reac_i;
                m_values_1dview[{i_grp + 1, reac_name}] = reac_i.reshape(one_dim);
            }
        }
    }

    void Macrolib::addReaction(const int i_grp, const std::string &reac_name, double values)
    {
        if( find(m_reac_names.begin(), m_reac_names.end(), reac_name) == m_reac_names.end() )
            m_reac_names.insert(reac_name);

        auto dim_z = std::get<2>(m_dim), dim_y = std::get<1>(m_dim), dim_x = std::get<0>(m_dim);
        Eigen::array<Eigen::DenseIndex, 1> one_dim({dim_z * dim_y * dim_x});
        Tensor3D reac_i(dim_z, dim_y, dim_x); // z, y, x
        reac_i.setConstant(values);
        m_values[{i_grp, reac_name}] = reac_i;
        m_values_1dview[{i_grp, reac_name}] = reac_i.reshape(one_dim);
    }

    void Macrolib::addReaction(const int i_grp, const std::string &reac_name, Tensor3D values)
    {
        if( find(m_reac_names.begin(), m_reac_names.end(), reac_name) == m_reac_names.end() )
            m_reac_names.insert(reac_name);

        auto dim_z = std::get<2>(m_dim), dim_y = std::get<1>(m_dim), dim_x = std::get<0>(m_dim);

        if (dim_z != values.dimension(0) || dim_y != values.dimension(1) || dim_x != values.dimension(2))
            throw std::invalid_argument("The size of the values must be identical to the one in the macrolib!");
        
        Eigen::array<Eigen::DenseIndex, 1> one_dim({dim_z * dim_y * dim_x});
        m_values[{i_grp, reac_name}] = values;
        m_values_1dview[{i_grp, reac_name}] = values.reshape(one_dim);
    }

    const py::array_t<double> Macrolib::getValuesPython(const int i_grp, const std::string &reac_name) const
    {
        Tensor3D values = getValues(i_grp, reac_name);

        return py::array_t<double, py::array::c_style>({values.dimension(0), values.dimension(1), values.dimension(2)},
                                                        values.data());
    }

    // todo fix issue with view and use reference for the return param !
    const py::array_t<double> Macrolib::getValues1DPython(const int i_grp, const std::string &reac_name) const
    {
        auto values_1d = getValues1D(i_grp, reac_name);

        return py::array_t<double, py::array::c_style>({values_1d.dimension(0)},
                                                        values_1d.data());
    }

    // todo fix issue with view and use reference for the return param !
    const Eigen::VectorXd Macrolib::getValuesArray(const int i_grp, const std::string &reac_name) const
    {
        auto values_1d = getValues1D(i_grp, reac_name);

        return Eigen::Map<const Eigen::VectorXd>(values_1d.data(), values_1d.dimension(0));
    }

} // namespace mat
