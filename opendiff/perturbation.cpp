#include <string>
#include <tuple>
#include "spdlog/fmt/ostr.h"

#include "perturbation.h"

namespace py = pybind11;

namespace perturbation
{
    using vecd = std::vector<double>;
    using vecvec = std::vector<Eigen::VectorXd>;
    using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    using Tensor0D = Eigen::Tensor<double, 0, Eigen::RowMajor>;
    using Tensor1D = Eigen::Tensor<double, 1, Eigen::RowMajor>;
    using Tensor2D = Eigen::Tensor<double, 2, Eigen::RowMajor>;

    bool checkBiOrthogonality(solver::SolverFull<SpMat> &solver, solver::SolverFull<SpMat> &solver_star, double max_eps, bool raise_error, bool remove)
    {
        auto eigen_vectors = solver.getEigenVectors();
        auto eigen_vectors_star = solver_star.getEigenVectors();

        // if (eigen_vectors_star.size() != eigen_vectors.size())
        //     throw std::invalid_argument("The number of eigen vectors has the be identical in solver and solver_star!");
        int vsize = static_cast<int>(std::min(eigen_vectors.size(), eigen_vectors_star.size()));

        auto M = solver.getM();

        std::vector<double> all_test{};
        std::vector<int> ids_remove{};

        for (auto i = 0; i < vsize; ++i)
        {
            for (auto j = 0; j < vsize; ++j)
            {
                if (i == j)
                    continue;

                double test = eigen_vectors_star[j].dot(M * eigen_vectors[i]);
                all_test.push_back(std::abs(test));
                if (std::abs(test) > max_eps)
                {
                    spdlog::debug("Biorthogonality test failed for {}, {}: {:.2e}", i, j, test);
                    ids_remove.push_back(std::max(i, j)); // remove max of both
                }
            }
        }

        if (remove)
        {
            solver.removeEigenVectors(ids_remove);
            solver_star.removeEigenVectors(ids_remove); // the same in order to keep the same order
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

    void handleDegeneratedEigenvalues(solver::SolverFull<SpMat> &solver, solver::SolverFull<SpMat> &solver_star, double max_eps)
    {
        solver.handleDenegeratedEigenvalues(max_eps);
        solver_star.handleDenegeratedEigenvalues(max_eps);
    }

    std::tuple<Eigen::VectorXd, double, vecd> firstOrderPerturbation(solver::SolverFull<SpMat> &solver, solver::SolverFull<SpMat> &solver_star, solver::SolverFull<SpMat> &solver_pert, std::string norm_method)
    {
        if (norm_method != "power" && norm_method != "PhiStarMPhi")
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
        if (norm_method == "PhiStarMPhi")
        {
            // spdlog::debug("a0/2. + 1  = {:.2e}", eigen_vectors_star[0].dot(delta_M * eigen_vectors[0]));
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
            Tensor0D power_pert_sum = power_pert.sum();
            double a0 = power_sum(0);

            for (int i{1}; i < nb_ev; ++i)
            {
                auto coeff = eigen_vectors_star[i].dot(M * eigen_vectors[i]);
                auto a_i = eigen_vectors_star[i].dot(delta_L_ev) * 1/((eigen_values[0]-eigen_values[i])*coeff);
                a.push_back(a_i);
                ev_recons += a_i * eigen_vectors[i];
                auto power_pert_i = solver_pert.getPower(i);
                Tensor0D power_pert_i_sum = power_pert_i.sum();
                a0 -= a_i * power_pert_i_sum(0);
            }
            a0 /= power_pert_sum(0);
            a[0] = a0;

            eval_recons += eigen_vectors_star[0].dot(delta_L_ev) / (eigen_vectors_star[0].dot(M * eigen_vectors[0]));

        }
        ev_recons += a[0] * eigen_vectors[0];

        solver_pert.clearEigenValues();
        solver_pert.pushEigenValue(eval_recons);
        solver_pert.pushEigenVector(ev_recons);
        return std::make_tuple(ev_recons, eval_recons, a);
    }

    std::tuple<Eigen::VectorXd, double, Tensor2D> highOrderPerturbation(int order, solver::SolverFull<SpMat> &solver,
                                                                        solver::SolverFull<SpMat> &solver_star, solver::SolverFull<SpMat> &solver_pert)
    {
        order += 1;
        solver.normPower();

        auto K = solver.getK();
        auto M = solver.getM();
        auto K_pert = solver_pert.getK();
        auto M_pert = solver_pert.getM();
        auto eigen_values = solver.getEigenValues();
        auto eigen_vectors = solver.getEigenVectors();
        auto eigen_vectors_star = solver_star.getEigenVectors();
        auto delta_M = (M_pert - M);
        auto delta_K = (K_pert - K);
        auto delta_L = (delta_K - delta_M * eigen_values[0]);
        int nb_ev = static_cast<int>(std::min(eigen_vectors.size(), eigen_vectors_star.size()));

        Tensor1D eval_recons_list(order);
        eval_recons_list.setZero();

        Tensor2D a(order, nb_ev);
        a.setZero();

        Tensor1D norm(nb_ev);
        norm.setZero();

        Tensor2D ev_star_dm_ev(nb_ev, nb_ev);
        ev_star_dm_ev.setZero();

        Tensor2D ev_star_dl_ev(nb_ev, nb_ev);
        ev_star_dl_ev.setZero();

        // add unpert eigen vector (temporary)
        solver_pert.setEigenVectors(eigen_vectors);

        auto power = solver.getPower(0);
        Tensor0D power_sum = power.sum();
        auto power_pert = solver_pert.getPower(0);
        Tensor0D power_pert_sum = power_pert.sum();
        a(0, 0) = power_sum(0) / power_pert_sum(0);
        eval_recons_list(0) = eigen_values[0];

        // precompute values
        for (int i{0}; i < nb_ev; ++i)
        {
            norm(i) = eigen_vectors_star[i].dot(M * eigen_vectors[i]);
            for (int j{0}; j < nb_ev; ++j)
            {
                ev_star_dm_ev(i, j) = eigen_vectors_star[i].dot(delta_M * eigen_vectors[j]);
                ev_star_dl_ev(i, j) = eigen_vectors_star[i].dot(delta_L * eigen_vectors[j]);
            }
        }

        for (int i{1}; i < order; ++i)
        {
            // eigenvalue calculation
            auto tmp0 = 0.;
            for (int j{0}; j < nb_ev; ++j)
                tmp0 += ev_star_dl_ev(0, j) * a(i - 1, j);

            auto tmp1 = 0.;
            for (int j{0}; j < nb_ev; ++j)
                for (int k{0}; k <= i - 2; ++k)
                    tmp1 += eval_recons_list(i - k - 1) * a(k, j) * ev_star_dm_ev(0, j);

            auto tmp2 = 0.;
            for (int k{1}; k <= i - 1; ++k)
                tmp2 += eval_recons_list(i - k) * a(k, 0);
            eval_recons_list(i) = (tmp0 - tmp1 - tmp2 * norm(0)) / (a(0, 0) * norm(0));

            // a coeff calculation
            auto power_pert_high_order = 0.;
            for (int n{1}; n < nb_ev; ++n)
            {
                auto tmpa_0 = 0.;
                for (int j{0}; j < nb_ev; ++j)
                    tmpa_0 += ev_star_dl_ev(n, j) * a(i - 1, j);

                auto tmpa_1 = 0.;
                for (int j{0}; j < nb_ev; ++j)
                    for (int k{0}; k <= i - 2; ++k)
                        tmpa_1 += eval_recons_list(i - k - 1) * ev_star_dm_ev(n, j) * a(k, j);

                auto tmpa_2 = 0.;
                for (int k{0}; k <= i - 1; ++k)
                    tmpa_2 = eval_recons_list(i - k) * a(k, n);

                a(i, n) = (-tmpa_0 + tmpa_1 + tmpa_2 * norm(n)) / ((eigen_values[n] - eigen_values[0]) * norm(n));

                auto power_pert_n = solver_pert.getPower(n);
                Tensor0D power_pert_n_sum = power_pert_n.sum();
                power_pert_high_order += a(i, n) * power_pert_n_sum(0);
            }

            a(i, 0) = -power_pert_high_order / power_pert_sum(0);
        }

        Tensor0D eval_recons_sum = eval_recons_list.sum();
        double eval_recons = eval_recons_sum(0);
        Eigen::VectorXd ev_recons(eigen_vectors[0].size());
        ev_recons.setZero();

        Eigen::array<int, 1> dims({0});
        Tensor1D a_sum = a.sum(dims);
        for (int n{0}; n < nb_ev; ++n)
            ev_recons += a_sum(n) * eigen_vectors[n];

        solver_pert.clearEigenValues();
        solver_pert.pushEigenValue(eval_recons);
        solver_pert.pushEigenVector(ev_recons);

        return std::make_tuple(ev_recons, eval_recons, a);
    }

    std::tuple<Eigen::VectorXd, double, py::array_t<double>> highOrderPerturbationPython(int order, solver::SolverFull<SpMat> &solver,
                                                                                         solver::SolverFull<SpMat> &solver_star, solver::SolverFull<SpMat> &solver_pert)
    {
        auto [ev_recons, eval_recons, a] = highOrderPerturbation(order, solver, solver_star, solver_pert);
        auto a_python = py::array_t<double, py::array::c_style>({a.dimension(0), a.dimension(1)},
                                                                a.data());
        return std::make_tuple(ev_recons, eval_recons, a_python);
    }

} // namespace perturbation
