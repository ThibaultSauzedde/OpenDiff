#ifndef PERTURBATION_H
#define PERTURBATION_H

#include <tuple>
#include <string>

#include "solver.h"

namespace perturbation
{
    using vecd = std::vector<double>;
    using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;

    bool checkBiOrthogonality(solver::Solver<SpMat> &solver, solver::Solver<SpMat> &solver_star, double max_eps = 1e-6, bool raise_error = false);

    std::tuple<Eigen::VectorXd, double, vecd> firstOrderPerturbation(solver::Solver<SpMat> &solver, solver::Solver<SpMat> &solver_star, solver::Solver<SpMat> &solver_pert, std::string norm_method);

    std::tuple<Eigen::VectorXd, double, vecd>  HighOrderPerturbation(solver::Solver<SpMat> &solver, solver::Solver<SpMat> &solver_star, solver::Solver<SpMat> &solver_pert, std::string norm_method);
} // namespace perturbation

#endif // PERTURBATION_H