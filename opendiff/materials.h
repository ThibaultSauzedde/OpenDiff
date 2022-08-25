#ifndef MATERIALS_H
#define MATERIALS_H

#include <vector>
#include <string>
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

        // we add the transfert section in it (always at the end) and  "SIGR"
        std::vector<std::string> m_reac_names{"D", "SIGA", "NU_SIGF", "CHI"};

        std::map<std::string, int> m_reac2id{};

        int m_nb_groups{-1};

        std::vector<int> checkReacNamesOrder(const std::vector<std::string> &reac_names);
        void setReactionsNames();
        void getNbGroups(const std::vector<Eigen::ArrayXXd> &values);
        Eigen::ArrayXXd addRemovalXS(const Eigen::ArrayXXd &mat, const std::vector<int> &ids);

    public:
        Materials() = delete;
        Materials(const Materials &copy) = default;
        Materials(const std::vector<Eigen::ArrayXXd> &values, const std::vector<std::string> &names,
                  const std::vector<std::string> &reac_names);

        const Eigen::ArrayXXd &getMaterial(const std::string &name) const { return m_values.at(name); };
        const auto getMaterials() const { return m_values; };

        const double getValue(const std::string &mat_name, const int i_grp, const std::string &reac_name) const;

        const auto getReacNames() const { return m_reac_names; };
        const auto getMatNames() const { return m_values; };
        const int getNbGroups() const { return m_nb_groups; };
        const int getReactionIndex(const std::string &reac_name) const { return m_reac2id.at(reac_name); };
        void addMaterial(const Eigen::ArrayXXd &mat, const std::string &name, const std::vector<std::string> &reac_names);
    };

} // namespace mat

#endif // MATERIALS_H