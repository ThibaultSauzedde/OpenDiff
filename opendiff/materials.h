#ifndef MATERIALS_H
#define MATERIALS_H

#include <vector>
#include <string>
#include <map>

#include <Eigen/Dense>

namespace mat
{
    class Materials
    {
    private:
        // vector of eigen matrix
        std::map<std::string, Eigen::ArrayXXd> m_values{};

        // we add the transfert section in it (always at the end), "SIGR" and "SIGF"
        std::vector<std::string> m_reac_names{"D", "SIGA", "NU_SIGF", "CHI", "EFISS", "NU"};

        std::map<std::string, int> m_reac2id{};

        int m_nb_groups{-1};

        std::vector<int> checkReacNamesOrder(const std::vector<std::string> &reac_names);
        void setReactionsNames();
        void getNbGroups(const std::vector<Eigen::ArrayXXd> &values);
        Eigen::ArrayXXd addAdditionalXS(const Eigen::ArrayXXd &mat, const std::vector<int> &ids);

    public:
        Materials() = default;
        Materials(const Materials &copy) = default;
        Materials(const std::vector<Eigen::ArrayXXd> &values, const std::vector<std::string> &names,
                  const std::vector<std::string> &reac_names);

        const Eigen::ArrayXXd &getMaterial(const std::string &name) const
        {
            if (m_values.find(name) == m_values.end())
                throw std::invalid_argument("The wanted material name (" + name + ") is not in the materials");
            else
                return m_values.at(name);
        };
        const auto getMaterials() const { return m_values; };

        // todo: add test (mat_name and reac name exist, igrp is in [0, nb_groups-1])
        const double getValue(const std::string &mat_name, const int i_grp, const std::string &reac_name) const;
        void setValue(const std::string &mat_name, const int i_grp, const std::string &reac_name, double value);

        const auto getReacNames() const { return m_reac_names; };
        const auto getMatNames() const
        {
            std::vector<std::string> keys;
            for (const auto &[k, v] : m_values)
            {
                keys.push_back(k);
            }
            return keys;
        };
        const int getNbGroups() const { return m_nb_groups; };
        const int getReactionIndex(const std::string &reac_name) const
        {
            if (m_reac2id.find(reac_name) == m_reac2id.end())
                throw std::invalid_argument("The wanted reac name " + reac_name + "is not in the materials");
            else
                return m_reac2id.at(reac_name);
        };
        void addMaterial(const Eigen::ArrayXXd &mat, const std::string &name, const std::vector<std::string> &reac_names);
    };

} // namespace mat

#endif // MATERIALS_H