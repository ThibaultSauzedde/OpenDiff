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

    bool checkBiOrthogonality(solver::Solver<SpMat> &solver, solver::Solver<SpMat> &solver_star, double max_eps = 1e-6, bool raise_error = false);

    std::tuple<Eigen::VectorXd, double, vecd> firstOrderPerturbation(solver::Solver<SpMat> &solver, solver::Solver<SpMat> &solver_star, solver::Solver<SpMat> &solver_pert, std::string norm_method);

    std::tuple<Eigen::VectorXd, double, Tensor2D> highOrderPerturbation(int order, solver::Solver<SpMat> &solver, solver::Solver<SpMat> &solver_star, solver::Solver<SpMat> &solver_pert);
    std::tuple<Eigen::VectorXd, double, py::array_t<double>> highOrderPerturbationPython(int order, solver::Solver<SpMat> &solver, solver::Solver<SpMat> &solver_star, solver::Solver<SpMat> &solver_pert);

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