#ifndef PERTURBATION_H
#define PERTURBATION_H

#include <tuple>
#include <string>
#include "spdlog/fmt/ostr.h"

#include "solver.h"

namespace perturbation
{
    using vecd = std::vector<double>;
    using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    using Tensor1D = Eigen::Tensor<double, 1, Eigen::RowMajor>;
    using Tensor2D = Eigen::Tensor<double, 2, Eigen::RowMajor>;

    void handleDegeneratedEigenvalues(solver::SolverFull<SpMat> &solver, solver::SolverFull<SpMat> &solver_star, double max_eps = 1e-6);

    bool checkBiOrthogonality(solver::SolverFull<SpMat> &solver, solver::SolverFull<SpMat>&solver_star, double max_eps = 1e-6, bool raise_error = false, bool remove = false);

    std::tuple<Eigen::VectorXd, double, vecd> firstOrderPerturbation(solver::SolverFull<SpMat> &solver, solver::SolverFull<SpMat> &solver_star, solver::SolverFull<SpMat> &solver_pert, std::string norm_method);

    std::tuple<Eigen::VectorXd, double, Tensor2D> highOrderPerturbation(int order, solver::SolverFull<SpMat> &solver, solver::SolverFull<SpMat> &solver_star, solver::SolverFull<SpMat> &solver_pert);
    std::tuple<Eigen::VectorXd, double, py::array_t<double>> highOrderPerturbationPython(int order, solver::SolverFull<SpMat> &solver, solver::SolverFull<SpMat> &solver_star, solver::SolverFull<SpMat> &solver_pert);

    // the response are not ratio
    std::tuple<double, Eigen::VectorXd, Eigen::VectorXd> firstOrderGPT(const solver::SolverFull<SpMat> &solver, const solver::SolverFull<SpMat> &solver_star,
                                                                       const solver::SolverFull<SpMat> &solver_pert,
                                                                       Eigen::VectorXd &response, Eigen::VectorXd &response_pert,
                                                                       Eigen::VectorXd &norm, Eigen::VectorXd &norm_pert,
                                                                       double tol, double tol_inner, int outer_max_iter, int inner_max_iter,
                                                                       std::string inner_solver, std::string inner_precond);

} // namespace perturbation

// using Tensor1D = Eigen::Tensor<double, 1, Eigen::RowMajor>;
// using Tensor2D = Eigen::Tensor<double, 2, Eigen::RowMajor>;
// template <>
// struct fmt::formatter<Tensor1D> : ostream_formatter
// {
// };
// template <>
// struct fmt::formatter<Tensor2D> : ostream_formatter
// {
// };

#endif // PERTURBATION_H