#include <vector>
#include <iostream>
#include <string>
#include <string>
#include <map>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

    void Material::multXsValue(const int i_grp, const std::string &isot_name, const std::string &reac_name, double value)
    {
        if (i_grp < 1 || i_grp > m_nb_groups)
            throw std::invalid_argument("The wanted nrj group (" + std::to_string(i_grp) + ") is not in the material");

        m_values(i_grp - 1, getIndex(isot_name, reac_name)) *= value;
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

    // Middles::Middles(const Middles &copy)
    // {
    //     m_reac_names = copy.m_reac_names;
    //     m_nb_groups = copy.m_nb_groups;
    //     for (const auto &[mat_name, material] : copy.m_materials)
    //         m_materials[mat_name] = Material(material);
    //     for (const auto &[middle_name, mat_name] : copy.m_middles)
    //         m_middles[middle_name] = mat_name;
    //     for (const auto &[middle_name, conc] : copy.m_conc)
    //         m_conc[middle_name] = conc;
    // }

    void Middles::checkMiddles()
    {
        m_reac_names = m_materials.begin()->second.getReacNames();
        m_nb_groups = m_materials.begin()->second.getNbGroups();
        for (const auto &[mat_name, material] : m_materials)
        {
            if (material.getNbGroups() != m_nb_groups)
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

    std::vector<std::vector<std::vector<std::string>>> Middles::createIndependantMiddlesByPlane(
        const std::vector<std::vector<std::vector<std::string>>> &geometry, std::vector<std::string> & ignore_middles, std::vector<int> & z_ids)
    {
        // create a new geometry with name indexed by the plane id 
        std::vector<std::vector<std::vector<std::string>>> new_geometry;

        int i = 0, j = 0, k = 0;

        auto i_size = geometry[0].size();
        auto j_size = geometry[0][0].size();

        
        if (!z_ids.empty() && geometry.size() != z_ids.size())
            throw std::invalid_argument("The size of the z_ids vector must be the same than the z direction !");

        for (std::vector i_val : geometry)
        {
            j = 0;
            std::vector<std::vector<std::string>> vy;
            if (i_size != i_val.size())
                throw std::invalid_argument("The size of the vector must be the same in each direction ! (y)");

            for (std::vector j_val : i_val)
            {
                k = 0;
                std::vector<std::string> vx;
                if (j_size != j_val.size())
                    throw std::invalid_argument("The size of the vector must be the same in each direction ! (x)");

                for (auto k_val : j_val)
                {
                    //not found
                    if (std::find(ignore_middles.begin(), ignore_middles.end(), k_val) == ignore_middles.end())
                    {
                        auto new_name = k_val + "_z" ;
                        if (z_ids.empty())
                            new_name += std::to_string(i) ;
                        else
                            new_name += std::to_string(z_ids[i]) ;

                        // geometry_tensor(i, j, k) = new_name;
                        vx.push_back(new_name);
                        //we add the middles
                        m_middles[new_name] = m_middles[k_val];
                        m_conc[new_name] = m_conc[k_val];
                    }
                    else
                        vx.push_back(k_val);

                    k++;
                }
                vy.push_back(vx);
                j++;
            }
            new_geometry.push_back(vy);
            i++;
        }

        // we create independant materials for the new middles
        createIndependantMaterials();

        return new_geometry ; 
    }

    double Middles::getXsValue(const std::string middle_name, const int i_grp, const std::string &reac_name, const std::string &isot_name) const
    {
        Material material = m_materials.at(m_middles.at(middle_name));
        return material.getXsValue(i_grp, isot_name, reac_name) * m_conc.at(middle_name).at(isot_name);
    }

    void Middles::setXsValue(const std::string middle_name, const int i_grp, const std::string &reac_name, const std::string &isot_name, double value)
    {
        Material &material = m_materials.at(m_middles.at(middle_name));
        material.setXsValue(i_grp, isot_name, reac_name, value);
    }

    double Middles::getXsValue(const std::string middle_name, const int i_grp, const std::string &reac_name) const
    {
        Material material = m_materials.at(m_middles.at(middle_name));
        auto isot_names = material.getIsotNames();
        double xs_value{0.};
        // no check --> all the isot must have all the wanted reac !
        for (auto isot_name : isot_names)
            xs_value += getXsValue(middle_name, i_grp, reac_name, isot_name) * m_conc.at(middle_name).at(isot_name);

        return xs_value;
    }

    void Middles::multXsValue(const std::string middle_name, const int i_grp, const std::string &reac_name, const std::string &isot_name, double value)
    {
        Material &material = m_materials[m_middles.at(middle_name)];
        material.multXsValue(i_grp, isot_name, reac_name, value);
    }

    void Middles::multXsValue(const std::string middle_name, const int i_grp, const std::string &reac_name, double value)
    {
        Material &material = m_materials[m_middles.at(middle_name)];
        auto isot_names = material.getIsotNames();
        // no check --> all the isot must have all the wanted reac !
        for (auto isot_name : isot_names)
            multXsValue(middle_name, i_grp, reac_name, isot_name, value);
    }

    void Middles::randomPerturbation(std::vector<std::string> reactions,
                                     std::default_random_engine &generator,
                                     std::geometric_distribution<int> &middles_distribution,
                                     std::uniform_int_distribution<int> &grp_distribution,
                                     std::uniform_real_distribution<double> &pert_value_distribution)
    {
        auto nb_middles_pert = middles_distribution(generator) + 1 ;
        if (nb_middles_pert >= static_cast<int>(m_middles.size()))
            nb_middles_pert = static_cast<int>(m_middles.size()) - 1;
        spdlog::debug("{} middles are modified", nb_middles_pert);
        for (auto i{0}; i < nb_middles_pert; ++i)
        {
            auto it_middles = m_middles.begin();
            std::advance(it_middles, rand() % m_middles.size());
            std::string middle_name = it_middles->first;

            std::shuffle(reactions.begin(), reactions.end(), generator);
            std::string reac_name = reactions[0];
            int i_grp = grp_distribution(generator);
            double pert_value = 1 + pert_value_distribution(generator) / 100.;
            // test if the value is not null 
            if (getXsValue(middle_name, i_grp+1, reac_name) < 1e-5)
            {
                i -= 1 ; 
                continue ; 
            }
            spdlog::info("Middle {}, group {}, reac {}, pertubation {}", middle_name, i_grp + 1, reac_name, pert_value);
            multXsValue(middle_name, i_grp + 1, reac_name, pert_value);
        }
    }

    void Middles::randomPerturbation(std::vector<std::string> reactions,
                                     std::default_random_engine &generator,
                                     std::normal_distribution<double> &pert_value_distribution)
    {
        for (const auto &[middle_name, mat_name] : m_middles)
        {
            for (std::string reac_name : reactions)
            {
                for (int i{0}; i < m_nb_groups; ++i)
                {
                    multXsValue(middle_name, i + 1, reac_name, pert_value_distribution(generator));
                }
            }
        }
    }

    void Middles::randomPerturbation(std::default_random_engine &generator,
                                     std::normal_distribution<double> &pert_value_distribution)
    {
        for (const auto &[middle_name, mat_name] : m_middles)
        {
            for (std::string reac_name : m_reac_names)
            {
                for (int i{0}; i < m_nb_groups; ++i)
                {
                    multXsValue(middle_name, i + 1, reac_name, pert_value_distribution(generator));
                }
            }
        }
    }

    void Middles::randomPerturbationUniform(std::vector<std::string> reactions, double pert_value_max)
    {
        std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
        std::geometric_distribution<int> middles_distribution(0.5);
        std::uniform_int_distribution<int> grp_distribution(0, getNbGroups() - 1);
        std::uniform_real_distribution<double> pert_value_distribution(-pert_value_max, +pert_value_max);
        randomPerturbation(reactions, generator, middles_distribution,
                           grp_distribution, pert_value_distribution);
    }

    void Middles::randomPerturbationNormal(double mean, double std)
    {
        std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
        std::normal_distribution<double> pert_xs_distribution(mean, std);
        Middles::randomPerturbation(generator, pert_xs_distribution);
    }

} // namespace mat
