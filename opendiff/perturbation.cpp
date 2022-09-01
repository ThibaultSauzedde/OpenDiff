#include <string>
#include <tuple>

#include "perturbation.h"

namespace py = pybind11;

namespace perturbation
{
    using vecd = std::vector<double>;
    using vecvec = std::vector<Eigen::VectorXd>;
    using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    using Tensor0D = Eigen::Tensor<double, 0, Eigen::RowMajor>;
    
    bool checkBiOrthogonality(solver::Solver<SpMat> &solver, solver::Solver<SpMat> &solver_star, double max_eps, bool raise_error)
    {
        auto eigen_vectors = solver.getEigenVectors();
        auto eigen_vectors_star = solver_star.getEigenVectors();

        // if (eigen_vectors_star.size() != eigen_vectors.size())
        //     throw std::invalid_argument("The number of eigen vectors has the be identical in solver and solver_star!");
        int vsize = static_cast<int>(std::min(eigen_vectors.size(), eigen_vectors_star.size()));

        auto M = solver.getM();

        std::vector<double> all_test{};
        for (auto i = 0; i < vsize; ++i)
        {
            for (auto j = 0; j < vsize; ++j)
            {
                if (i == j)
                    continue;

                double test = eigen_vectors_star[j].dot(M * eigen_vectors[i]);
                all_test.push_back(std::abs(test));
                if (test > max_eps)
                    spdlog::debug("Biorthogonality test failed for {}, {}: {:.2e}", i, j, test);
            }
        }

        double max_test = *std::max_element(all_test.begin(), all_test.end());
        spdlog::info("Biorthogonality max test : {:.2e}", max_test);
        if (max_test > max_eps && raise_error)
            throw std::invalid_argument("The eigen vector are not bi-orthogonals!");
        else if (max_test > max_eps)
            return false;
        else
            return true;
    }

    std::tuple<Eigen::VectorXd, double, vecd> firstOrderPerturbation(solver::Solver<SpMat> &solver, solver::Solver<SpMat> &solver_star, solver::Solver<SpMat> &solver_pert, std::string norm_method)
    {
        if (norm_method != "power" && norm_method != "PhiMPhiStar")
            throw std::invalid_argument("Invalid method name!");

        if (!(solver.isNormed() && solver.getNormMethod() == norm_method))
            solver.norm(norm_method, solver_star);

        auto K = solver.getK();
        auto M = solver.getM();
        auto K_pert = solver_pert.getK();
        auto M_pert = solver_pert.getM();
        auto eigen_values = solver.getEigenValues();
        auto eigen_vectors = solver.getEigenVectors();
        auto eigen_vectors_star = solver_star.getEigenVectors();
        auto delta_M = (M_pert - M);
        auto delta_L_ev = ((K_pert - K) - delta_M * eigen_values[0]) * eigen_vectors[0];
        int nb_ev = static_cast<int>(std::min(eigen_vectors.size(), eigen_vectors_star.size()));
        Eigen::VectorXd ev_recons(eigen_vectors[0].size());
        ev_recons.setZero();
        auto eval_recons = eigen_values[0];

        std::vector<double> a{};
        if (norm_method == "PhiMPhiStar")
        {
            auto a0 = 1 - eigen_vectors_star[0].dot(delta_M * eigen_vectors[0]) / 2.;
            a.push_back(a0);
            for (int i{1}; i < nb_ev; ++i)
            {
                auto a_i = eigen_vectors_star[i].dot(delta_L_ev) * 1 / (eigen_values[0] - eigen_values[i]);
                a.push_back(a_i);
                ev_recons += a_i * eigen_vectors[i];
            }
            eval_recons += eigen_vectors_star[0].dot(delta_L_ev);
        }
        else if (norm_method == "power") // todo fix it, not working 
        {
            //add unpert eigen vector (temporary)
            solver_pert.setEigenVectors(eigen_vectors); 

            a.push_back(0);
            auto power = solver.getPower(0);
            Tensor0D power_sum = power.sum();
            auto power_pert = solver_pert.getPower(0);
            Tensor0D power_pert_sum = power.sum();
            double a0 = power_sum(0) - power_pert_sum(0);

            for (int i{1}; i < nb_ev; ++i)
            {
                auto coeff = eigen_vectors_star[i].dot(M * eigen_vectors[i]);
                auto a_i = eigen_vectors_star[i].dot(delta_L_ev) * 1/((eigen_values[0]-eigen_values[i])*coeff);
                a.push_back(a_i);
                ev_recons += a_i * eigen_vectors[i];
                auto power_pert_i = solver_pert.getPower(i);
                Tensor0D power_pert_i_sum = power.sum();
                a0 -= power_pert_i_sum(0);
            }
            a0 /= power_pert_sum(0);
            a[0] = 1 + a0;

            eval_recons += eigen_vectors_star[0].dot(delta_L_ev) / (eigen_vectors_star[0].dot(M * eigen_vectors[0]));

        }
        ev_recons += a[0] * eigen_vectors[0];

        solver_pert.clearEigenValues();
        solver_pert.pushEigenValue(eval_recons);
        solver_pert.pushEigenVector(ev_recons);
        return std::make_tuple(ev_recons, eval_recons, a);
    }

//todo: add high order pert

} // namespace perturbation
