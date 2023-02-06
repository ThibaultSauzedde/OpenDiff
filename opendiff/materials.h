#ifndef MATERIALS_H
#define MATERIALS_H

#include <vector>
#include <string>
#include <map>
#include <tuple>
#include <set>
#include <random>
#include <algorithm>
#include <chrono>
#include "spdlog/spdlog.h"

#include <Eigen/Dense>

namespace mat
{
    class Material
    {
    private:
        using tuple_str = std::tuple<std::string, std::string>;

        // vector of eigen matrix
        Eigen::ArrayXXd m_values{};

        // we add the transfert section in it (always at the end), "SIGR" and "SIGF"
        std::set<std::string> m_reac_names{"D", "SIGA", "NU_SIGF", "CHI", "EFISS", "NU"};
        std::set<std::string> m_isot_names{}; // the list of unique isot names

        std::vector<tuple_str> m_isot_reac_names{};

        int m_nb_groups{-1};

        void createIndex(const std::vector<std::string> &isot_names, const std::vector<std::string> &reac_names);
        void checkIsotReacNamesOrder(const std::vector<std::string> &isot_names, const std::vector<std::string> &reac_names);
        void setReactionsNames();
        void addAdditionalXS();
        void majAdditionalXS();

    public:
        Material() = default;
        Material(const Material &copy) = default;
        Material(const Eigen::ArrayXXd &values, const std::vector<std::string> &isot_names, const std::vector<std::string> &reac_names);

        Material(const Eigen::ArrayXXd &values, const std::vector<tuple_str> &isot_reac_names);

        const auto &getValues() const { return m_values; };

        const auto &getIndex() const { return m_isot_reac_names; };
        const int getIndex(const std::string &isot_name, const std::string &reac_name) const;

        const double getXsValue(const int i_grp, const std::string &isot_name, const std::string &reac_name) const;
        void setXsValue(const int i_grp, const std::string &isot_name, const std::string &reac_name, double value);
        void multXsValue(const int i_grp, const std::string &isot_name, const std::string &reac_name, double value);

        const auto getReacNames() const { return m_reac_names; };
        const auto getIsotNames() const { return m_isot_names; };
        const int getNbGroups() const { return m_nb_groups; };
    };

    class Middles
    {
    private:
        std::map<std::string, Material> m_materials;
        std::map<std::string, std::string> m_middles;
        std::map<std::string, std::map<std::string, double>> m_conc{};
        int m_nb_groups{-1};
        std::set<std::string> m_reac_names{};

        void checkMiddles();

    public:
        Middles() = default;
        Middles(const Middles &copy) = default;
        Middles(std::map<std::string, Material> &materials, const std::map<std::string, std::string> &middles);
        Middles(std::map<std::string, Material> &materials, const std::map<std::string, std::string> &middles,
                const std::map<std::string, std::map<std::string, double>> &concentrations);

        void createIndependantMaterials();

        std::vector<std::vector<std::vector<std::string>>> createIndependantMiddlesByPlane(
            const std::vector<std::vector<std::vector<std::string>>> &geometry, std::vector<std::string> & ignore_middles, std::vector<int> & z_ids);

        void addMaterial(std::string name, Material value)
        {
            m_materials[name] = value;
            checkMiddles();
        };

        void addMiddle(std::string middle_name, std::string mat_name, std::map<std::string, double> &concentrations)
        {
            m_middles[middle_name] = mat_name;
            m_conc[middle_name] = concentrations;
            checkMiddles();
        };

        void addMiddle(std::string middle_name, std::string mat_name)
        {
            m_middles[middle_name] = mat_name;
            // set the concentration to 1
            for (auto isot_name : m_materials[mat_name].getIsotNames())
                m_conc[middle_name][isot_name] = 1.;
            checkMiddles();
        };

        auto getMiddles() { return m_middles; };
        auto getMaterials() { return m_materials; };
        auto getConcentrations() { return m_conc; };

        auto size() { return m_middles.size(); };

        double getXsValue(const std::string middle_name, const int i_grp, const std::string &reac_name, const std::string &isot_name) const;
        void setXsValue(const std::string middle_name, const int i_grp, const std::string &reac_name, const std::string &isot_name, double value);
        double getXsValue(const std::string middle_name, const int i_grp, const std::string &reac_name) const;
        void multXsValue(const std::string middle_name, const int i_grp, const std::string &reac_name, const std::string &isot_name, double value);
        void multXsValue(const std::string middle_name, const int i_grp, const std::string &reac_name, double value);

        const int getNbGroups() const { return m_nb_groups; };
        const auto getReacNames() const { return m_reac_names; };

        void randomPerturbation(std::vector<std::string> reactions,
                                std::default_random_engine &generator,
                                std::geometric_distribution<int> &middles_distribution,
                                std::uniform_int_distribution<int> &grp_distribution,
                                std::uniform_real_distribution<double> &pert_value_distribution);
        void randomPerturbationPython(std::vector<std::string> reactions, double pert_value_max);
    };

} // namespace mat

#endif // MATERIALS_H