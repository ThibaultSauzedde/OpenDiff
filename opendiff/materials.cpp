#include <vector>
#include <iostream>
#include <string>
#include <string>
#include <map>

#include "materials.h"

namespace mat
{
    Material::Material(const Eigen::ArrayXXd &values, const std::vector<tuple_str> &isot_reac_names)
    {
        m_values = values;
        m_nb_groups = m_values.rows();

        // set the reactions properly
        setReactionsNames();

        // create the index
        m_isot_reac_names = isot_reac_names;
        std::vector<std::string> reac_names{};
        std::vector<std::string> isot_names{};
        for (int i{0}; i < static_cast<int>(isot_reac_names.size()); ++i)
        {
            isot_names.push_back(std::get<0>(isot_reac_names[i]));
            reac_names.push_back(std::get<1>(isot_reac_names[i]));
        }

        // check the reac and isot names
        checkIsotReacNamesOrder(isot_names, reac_names);

        // create an empty eigen array with the size of reac_names (or isot_names) times the number of energy groups
        addAdditionalXS();
    }

    Material::Material(const Eigen::ArrayXXd &values, const std::vector<std::string> &isot_names, const std::vector<std::string> &reac_names)
    {
        m_values = values;
        m_nb_groups = m_values.rows();

        // set the reactions properly
        setReactionsNames();

        // create the index
        createIndex(isot_names, reac_names);

        // check the reac and isot names
        checkIsotReacNamesOrder(isot_names, reac_names);

        // create an empty eigen array with the size of reac_names (or isot_names) times the number of energy groups
        addAdditionalXS();
    }

    void Material::setReactionsNames()
    {
        // set the reactions names
        for (int i{0}; i < m_nb_groups; ++i)
        {
            m_reac_names.insert(std::to_string(i + 1));
        }

        m_reac_names.insert("SIGR");
        m_reac_names.insert("SIGF");
    }

    void Material::createIndex(const std::vector<std::string> &isot_names, const std::vector<std::string> &reac_names)
    {
        if (reac_names.size() != isot_names.size())
            throw std::invalid_argument("There is a different number of reactions and isotopes!");

        for (int i{0}; i < static_cast<int>(reac_names.size()); ++i)
            m_isot_reac_names.push_back(std::make_tuple(isot_names[i], reac_names[i]));
    }

    void Material::checkIsotReacNamesOrder(const std::vector<std::string> &isot_names, const std::vector<std::string> &reac_names)
    {
        if (reac_names.size() != isot_names.size())
            throw std::invalid_argument("There is a different number of reactions and isotopes!");

        // get unique reac and isot names
        m_isot_names = std::set<std::string>(isot_names.begin(), isot_names.end());

        // loop on the isot
        bool isin{false};
        for (std::string isot_name : m_isot_names)
        {
            // loop on the mandatories reac names
            for (std::string reac_name : m_reac_names)
            {
                // loop on the given reac and isot names
                isin = false;
                for (int i{0}; i < static_cast<int>(reac_names.size()); ++i)
                {
                    if (isot_name != isot_names[i])
                        continue;

                    if (reac_name == reac_names[i])
                        isin = true;
                }
                if (isin == false && !((reac_name == "SIGR") || (reac_name == "SIGF")))
                    throw std::invalid_argument("The reaction " + reac_name + " is missing for the isotope " + isot_name + "!");
                else if (isin == true && ((reac_name == "SIGR") || (reac_name == "SIGF")))
                    throw std::invalid_argument("The reaction " + reac_name + " is calculated in this class and should not be given for the isotope " + isot_name + "!");
            }
        }
    }

    void Material::addAdditionalXS()
    {
        // add the removal + sigf xs section
        m_values.conservativeResize(m_values.rows(), m_values.cols() + 2 * static_cast<int>(m_isot_names.size()));
        for (std::string isot_name : m_isot_names)
        {
            m_isot_reac_names.push_back(std::make_tuple(isot_name, "SIGR"));
            m_isot_reac_names.push_back(std::make_tuple(isot_name, "SIGF"));
        }
        majAdditionalXS();
    }

    void Material::majAdditionalXS()
    {
        for (std::string isot_name : m_isot_names)
        {
            int id_siga = getIndex(isot_name, "SIGA");
            int id_nusigf = getIndex(isot_name, "NU_SIGF");
            int id_nu = getIndex(isot_name, "NU");
            int id_sigr = getIndex(isot_name, "SIGR");
            int id_sigf = getIndex(isot_name, "SIGF");

            // sigr = siga
            m_values(Eigen::placeholders::all, id_sigr) = m_values(Eigen::placeholders::all, id_siga);

            // add transfert section to other groups
            for (int grp_orig{0}; grp_orig < m_nb_groups; ++grp_orig)
            {
                for (int grp_dest{0}; grp_dest < m_nb_groups; ++grp_dest)
                {
                    if (grp_orig == grp_dest)
                        continue;

                    int id_grp_dest = getIndex(isot_name, std::to_string(grp_dest + 1));
                    m_values(grp_orig, id_sigr) += m_values(grp_orig, id_grp_dest);
                }
            }
            m_values(Eigen::placeholders::all, id_sigf) = m_values(Eigen::placeholders::all, id_nusigf) / m_values(Eigen::placeholders::all, id_nu);
        }
    }

    const int Material::getIndex(const std::string &isot_name, const std::string &reac_name) const
    {
        auto index_val = std::make_tuple(isot_name, reac_name);
        // get the id in the ref reac names
        int index = std::distance(m_isot_reac_names.begin(),
                                  std::find(m_isot_reac_names.begin(), m_isot_reac_names.end(), index_val));

        if (index == static_cast<int>(m_isot_reac_names.size()))
            throw std::invalid_argument("The wanted (isot name, reac name) = (" + isot_name + ", " + isot_name + ") is not in the material");

        return index;
    };

    const double Material::getXsValue(const int i_grp, const std::string &isot_name, const std::string &reac_name) const
    {
        if (i_grp < 1 || i_grp > m_nb_groups)
            throw std::invalid_argument("The wanted nrj group (" + std::to_string(i_grp) + ") is not in the material");

        return m_values(i_grp - 1, getIndex(isot_name, reac_name));
    }

    void Material::setXsValue(const int i_grp, const std::string &isot_name, const std::string &reac_name, double value)
    {
        if (i_grp < 1 || i_grp > m_nb_groups)
            throw std::invalid_argument("The wanted nrj group (" + std::to_string(i_grp) + ") is not in the material");

        m_values(i_grp - 1, getIndex(isot_name, reac_name)) = value;
        if ((reac_name != "SIGR") && (reac_name != "SIGF"))
            majAdditionalXS();
    }

    //
    // Middles
    //

    Middles::Middles(std::map<std::string, Material> &materials, const std::map<std::string, std::string> &middles)
    {
        m_materials = materials;
        m_middles = middles;

        // set the concentration to 1
        for (const auto &[middle_name, mat_name] : m_middles)
            for (auto isot_name : m_materials[mat_name].getIsotNames())
                m_conc[middle_name][isot_name] = 1.;

        checkMiddles();
    }

    Middles::Middles(std::map<std::string, Material> &materials, const std::map<std::string, std::string> &middles,
                     const std::map<std::string, std::map<std::string, double>> &concentrations)
    {
        m_materials = materials;
        m_middles = middles;
        m_conc = concentrations;
        checkMiddles();
    }

    void Middles::checkMiddles()
    {
        m_reac_names = m_materials.begin()->second.getReacNames();
        m_nb_groups = m_materials.begin()->second.getNbGroups();
        for (const auto &[mat_name, mat] : m_materials)
        {
            if (mat.getNbGroups() != m_nb_groups)
                throw std::invalid_argument("The number if nrj group has to be the same in all the materials ! ");
        }

        for (const auto &[middle_name, mat_name] : m_middles)
        {
            if (m_materials.find(mat_name) == m_materials.end())
                throw std::invalid_argument("The wanted material (" + mat_name + ") is not in the materials!");

            if (m_conc.find(middle_name) == m_conc.end())
                throw std::invalid_argument("The wanted middle (" + middle_name + ") has no concentrations!");

            //check the isot
            auto isot_names = m_materials[mat_name].getIsotNames();
            for (auto isot_name : isot_names)
            {
                if (!m_conc[middle_name].count(isot_name))
                    throw std::invalid_argument("The wanted isot (" + isot_name + ") in middle " + middle_name + "has no concentrations!");
            }
        }
    }

    void Middles::createIndependantMaterials()
    {
        std::set<std::string> unique_mat_name{};
        for (const auto &[middle_name, mat_name] : m_middles)
        {
            // not found
            if (std::find(unique_mat_name.begin(), unique_mat_name.end(), mat_name) == unique_mat_name.end())
                unique_mat_name.insert(mat_name);
            else // already found
            {
                // add a mat with unique name
                int i = 0;
                std::string new_mat_name{mat_name + "_" + std::to_string(i)};
                while (m_materials.count(new_mat_name))
                {
                    i++;
                    new_mat_name = mat_name + "_" + std::to_string(i);
                }

                m_materials[new_mat_name] = Material(m_materials[mat_name]);
                m_middles[middle_name] = new_mat_name;
            }
        }
    }

    double Middles::getXsValue(const std::string middle_name, const int i_grp, const std::string &reac_name, const std::string &isot_name) const
    {
        auto mat = m_materials.at(m_middles.at(middle_name)) ; 
        return mat.getXsValue(i_grp, isot_name, reac_name) * m_conc.at(middle_name).at(isot_name);
    }

    void Middles::setXsValue(const std::string middle_name, const int i_grp, const std::string &reac_name, const std::string &isot_name, double value)
    {
        auto mat = m_materials.at(m_middles.at(middle_name)) ; 
        mat.setXsValue(i_grp, isot_name, reac_name, value);
    }

    double Middles::getXsValue(const std::string middle_name, const int i_grp, const std::string &reac_name) const
    {
        auto mat = m_materials.at(m_middles.at(middle_name)) ;
        auto isot_names = mat.getIsotNames();
        double xs_value{0.};
        // no chekc --> all the isot must have all the wanted reac !
        for (auto isot_name : isot_names)
            xs_value += getXsValue(middle_name, i_grp, reac_name, isot_name);

        return xs_value;
    }

} // namespace mat
