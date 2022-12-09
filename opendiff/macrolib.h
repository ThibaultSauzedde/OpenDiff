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
    using Tensor1D = Eigen::Tensor<double, 1, Eigen::RowMajor>;
    using Tensor3D = Eigen::Tensor<double, 3, Eigen::RowMajor>;
    class Macrolib
    {
    private:
        // values (map of tensor with tuple as keys)
        using pairreac_t = std::pair<int, std::string>; // make pairlist_t an alias

        std::map<pairreac_t, Tensor3D> m_values{};

        std::map<pairreac_t, Tensor1D> m_values_1dview{};

        std::set<std::string> m_reac_names{};

        int m_nb_groups{-1};

        std::tuple<int, int, int> m_dim{0, 0, 0}; // tuple avec la taille de x, y et z

        void setup(const mat::Middles &middles, const Eigen::Tensor<std::string, 3, Eigen::RowMajor> &geometry);

    public:
        Macrolib() = default;
        Macrolib(const Macrolib &copy) = default;
        Macrolib(const mat::Middles &middles, const Eigen::Tensor<std::string, 3, Eigen::RowMajor> &geometry); // dim are z, y, x
        Macrolib(const mat::Middles &middles, const std::vector<std::vector<std::vector<std::string>>> &geometry); // python wrapping

        const Tensor1D &getValues1D(const int i_grp, const std::string &reac_name) const
        {
            if (isIn(i_grp, reac_name))
                return m_values_1dview.at({i_grp, reac_name});
            else
                throw std::invalid_argument("The wanted reac name (" + reac_name + ") and nrj group (" + std::to_string(i_grp) + ") is not in the materials");
        };
        const Tensor3D &getValues(const int i_grp, const std::string &reac_name) const
        {
            if (isIn(i_grp, reac_name))
                return m_values.at({i_grp, reac_name});
            else
                throw std::invalid_argument("The wanted reac name (" + reac_name + ") and nrj group (" + std::to_string(i_grp) + ") is not in the materials");
        };
        const py::array_t<double> getValuesPython(const int i_grp, const std::string &reac_name) const;   // python wrapping
        const py::array_t<double> getValues1DPython(const int i_grp, const std::string &reac_name) const; // python wrapping

        bool isIn(const int i_grp, const std::string &reac_name) const
        {
            if (m_values.find({i_grp, reac_name}) == m_values.end())
                return false;
            else
                return true;
        };

        const auto getReacNames() const { return m_reac_names; } ; 
        const int getNbGroups() const { return m_nb_groups; } ;  
        const auto getDim() const { return m_dim; } ; 

        void addReaction(const int i_grp, const std::string &reac_name, double values);
        void addReaction(const int i_grp, const std::string &reac_name, Tensor3D values);

    };

} // namespace mat

#endif // MACROLIB_H