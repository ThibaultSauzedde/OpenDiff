#include <vector>
#include <iostream>
#include <string>
#include <string>
#include <map>

#include "materials.h"

namespace mat
{
    Materials::Materials(const std::vector<Eigen::ArrayXXd> &values, const std::vector<std::string> &names,
                         const std::vector<std::string> &reac_names)
    {
        if (names.size() != values.size())
            throw std::invalid_argument("The number of values is != from the number of names!");

        // get the number of groups
        getNbGroups(values);

        // set the reactions properly
        setReactionsNames();

        // check that the reac names are in the same order than m_reac_names
        auto ids = checkReacNamesOrder(reac_names);

        // for each mat: create an empty eigen array with the size of m_reac_names times the number of energy groups
        int i = 0;
        for (Eigen::ArrayXXd mat : values)
        {
            auto new_mat = addAdditionalXS(mat, ids);

            // fill the matrix
            m_values[names[i]] = new_mat;
            i++;
        }
    }

    std::vector<int> Materials::checkReacNamesOrder(const std::vector<std::string> &reac_names)
    {
        std::vector<int> ids(reac_names.size(), -1);

        if ((reac_names.size() + 2) != m_reac_names.size())
            throw std::invalid_argument("There is a wrong number of reaction!");

        int i = 0;
        for (std::string reac_name : m_reac_names)
        {
            if (reac_name == "SIGR" || reac_name == "SIGF")
            {
                i++;
                continue;
            }
            // get the id in the ref reac names
            int ref_i = std::distance(reac_names.begin(), std::find(reac_names.begin(),
                                                                    reac_names.end(), reac_name));

            if (static_cast<std::vector<int>::size_type>(ref_i) == reac_names.size())
            {
                std::cerr << reac_name << "\n";
                throw std::invalid_argument("There is a missing reaction!");
            }
            else
                ids[ref_i] = i;

            i++;
        }

        return ids;
    }

    void Materials::getNbGroups(const std::vector<Eigen::ArrayXXd> &values)
    {
        int i = 0;
        for (Eigen::ArrayXXd mat : values)
        {
            // check array shape
            if (i == 0)
                m_nb_groups = mat.rows();
            else if (m_nb_groups != mat.rows())
                throw std::invalid_argument("The number of groups is not the same for all the materials!");

            i++;
        }
    }

    void Materials::setReactionsNames()
    {
        // set the reactions names
        for (int i{0}; i < m_nb_groups; ++i)
        {
            m_reac_names.push_back(std::to_string(i + 1));
        }

        m_reac_names.push_back("SIGR");
        m_reac_names.push_back("SIGF");

        int i = 0;
        for (auto reac_name : m_reac_names)
        {
            m_reac2id[reac_name] = i;
            i++;
        }
    }

    // todo: check the bounds
    const double Materials::getValue(const std::string &mat_name, const int i_grp, const std::string &reac_name) const
    {
        if (i_grp < 1 || i_grp > m_nb_groups)
            throw std::invalid_argument("The wanted nrj group (" + std::to_string(i_grp) + ") is not in the materials");

        if (m_values.find(mat_name) == m_values.end())
            throw std::invalid_argument("The wanted reac name (" + reac_name + ") is not in the materials");
        else
            return m_values.at(mat_name)(i_grp - 1, getReactionIndex(reac_name));
    }

    void Materials::setValue(const std::string &mat_name, const int i_grp, const std::string &reac_name, double value)
    {
        if (i_grp < 1 || i_grp > m_nb_groups)
            throw std::invalid_argument("The wanted nrj group (" + std::to_string(i_grp) + ") is not in the materials");

        if (m_values.find(mat_name) == m_values.end())
            throw std::invalid_argument("The wanted reac name (" + reac_name + ") is not in the materials");
        else
            m_values.at(mat_name)(i_grp - 1, getReactionIndex(reac_name)) = value;
    }

    Eigen::ArrayXXd Materials::addAdditionalXS(const Eigen::ArrayXXd &mat, const std::vector<int> &ids)
    {
        // check array shape
        if (static_cast<std::vector<int>::size_type>(mat.cols()) != ids.size())
            throw std::invalid_argument("The number of reaction is different in the array and the name's list!");

        // reorder the array
        Eigen::ArrayXXd new_mat = mat(Eigen::placeholders::all, ids);

        // add the removal xs section
        new_mat.conservativeResize(mat.rows(), mat.cols() + 2);

        int id_siga = m_reac2id["SIGA"];
        int id_nusigf = m_reac2id["NU_SIGF"];
        int id_nu = m_reac2id["NU"];
        int id_sigr = m_reac2id["SIGR"];
        int id_sigf = m_reac2id["SIGF"]; // last id

        // sigr = siga
        new_mat(Eigen::placeholders::all, id_sigr) = new_mat(Eigen::placeholders::all, id_siga);

        // add transfert section to other groups
        for (int grp_orig{0}; grp_orig < m_nb_groups; ++grp_orig)
        {
            for (int grp_dest{0}; grp_dest < m_nb_groups; ++grp_dest)
            {
                if (grp_orig == grp_dest)
                    continue;

                int id_grp_dest = m_reac2id[std::to_string(grp_dest + 1)];
                new_mat(grp_orig, id_sigr) += new_mat(grp_orig, id_grp_dest);
            }
        }

        new_mat(Eigen::placeholders::all, id_sigf) = new_mat(Eigen::placeholders::all, id_nusigf) / new_mat(Eigen::placeholders::all, id_nu);

        return new_mat;
    }

    void Materials::addMaterial(const Eigen::ArrayXXd &mat, const std::string &name, const std::vector<std::string> &reac_names)
    {
        // check that the reac names are in the same order than m_reac_names
        auto ids = checkReacNamesOrder(reac_names);

        // check the number of groups
        if (m_nb_groups != mat.rows())
            throw std::invalid_argument("The number of groups is not the same for all the materials!");

        auto new_mat = addAdditionalXS(mat, ids);

        // fill the matrix
        m_values[name] = new_mat;
    }

} // namespace mat
