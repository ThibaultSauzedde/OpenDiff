#ifndef MACROLIB_H
#define MACROLIB_H

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <unsupported/Eigen/CXX11/Tensor>

#include "materials.h"

namespace py = pybind11;

namespace mat
{
    class Macrolib
    {
    private:
        // values (map of tensor with tuple as keys)
        using pairreac_t = std::pair<int, std::string>; // make pairlist_t an alias

        std::map<pairreac_t, Eigen::Tensor<double, 3>> m_values{};

        std::map<pairreac_t, Eigen::Tensor<double, 1>> m_values_1dview{};

        std::vector<std::string> m_reac_names{};

        int m_nb_groups{-1};

        std::tuple<int, int, int> m_dim{0, 0, 0}; // tuple avec la taille de x, y et z

        void setup(const mat::Materials &materials, const Eigen::Tensor<std::string, 3> &geometry);

    public:
        Macrolib() = delete;
        Macrolib(const Macrolib &copy) = delete;
        Macrolib(const mat::Materials &materials, const Eigen::Tensor<std::string, 3> &geometry);                      // dim are z, y, x
        Macrolib(const mat::Materials &materials, const std::vector<std::vector<std::vector<std::string>>> &geometry); // python wrapping

        const Eigen::Tensor<double, 1> &getValues1D(const int i_grp, const std::string &reac_name) const { return m_values_1dview.at({i_grp, reac_name}); };
        const Eigen::Tensor<double, 3> &getValues(const int i_grp, const std::string &reac_name) const { return m_values.at({i_grp, reac_name}); };
        const py::array_t<double> getValuesPython(const int i_grp, const std::string &reac_name) const;   // python wrapping
        const py::array_t<double> getValues1DPython(const int i_grp, const std::string &reac_name) const; // python wrapping

        const auto getReacNames() { return m_reac_names; };
        const int getNbGroups() { return m_nb_groups; };
    };

} // namespace mat

#endif // MACROLIB_H