#include <vector>
#include <iostream>
#include <string_view>
#include <string>

#include "macrolib.h"

// todo: make double a template in order to use simple or double precision !
namespace py = pybind11;

namespace mat
{
    Macrolib::Macrolib(const mat::Materials &materials, const Eigen::Tensor<std::string, 3> &geometry)
    {
        setup(materials, geometry);
    }

    Macrolib::Macrolib(const mat::Materials &materials, const std::vector<std::vector<std::vector<std::string>>> &geometry)
    {
        //copy of the geometry in a tensor, it does not cost that much :)

        Eigen::Tensor<std::string, 3> geometry_tensor(static_cast<int>(geometry.size()), static_cast<int>(geometry[0].size()),
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
        setup(materials, geometry_tensor);
    }

    void Macrolib::setup(const mat::Materials &materials, const Eigen::Tensor<std::string, 3> &geometry)
    {
        m_nb_groups = materials.getNbGroups();
        m_reac_names = materials.getReacNames();
        auto dim_z = geometry.dimension(0), dim_y = geometry.dimension(1), dim_x = geometry.dimension(2);
        m_dim = {dim_x, dim_y, dim_z};
        Eigen::array<Eigen::DenseIndex, 1> one_dim({dim_z * dim_y * dim_x});

        // loop on reactionsmaterials.getReacNames()
        for (int i_grp = 0; i_grp < m_nb_groups; i_grp++)
        {
            for (auto reac_name : m_reac_names)
            {
                Eigen::Tensor<double, 3> reac_i(dim_z, dim_y, dim_x); // z, y, x
                // loop on the geometry
                for (int i = 0; i < dim_z; ++i)
                {
                    for (int j = 0; j < dim_y; ++j)
                    {
                        for (int k = 0; k < dim_x; ++k)
                        {
                            reac_i(i, j, k) = materials.getValue(geometry(i, j, k), i_grp, reac_name);
                        }
                    }
                }
                m_values[{i_grp + 1, reac_name}] = reac_i;
                m_values_1dview[{i_grp + 1, reac_name}] = reac_i.reshape(one_dim);
            }
        }
    }

    const py::array_t<double> Macrolib::getValuesPython(const int i_grp, const std::string &reac_name) const
    {
        Eigen::Tensor<double, 3> values = getValues(i_grp, reac_name);

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

} // namespace mat
