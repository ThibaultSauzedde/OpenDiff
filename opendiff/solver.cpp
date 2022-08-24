#include <vector>

#include "solver.h"

namespace solver
{
    using vecd = std::vector<double>;

    Eigen::Tensor<double, 1> delta_coord(vecd &coord)
    {
        int c_size = static_cast<int>(coord.size());
        auto c_map = Eigen::TensorMap<Eigen::Tensor<double, 1>>(&coord[0], c_size);
        Eigen::array<Eigen::Index, 1> offsets = {0};
        Eigen::array<Eigen::Index, 1> extents = {c_size};
        Eigen::array<Eigen::Index, 1> offsets_p = {1};
        Eigen::array<Eigen::Index, 1> extents_p = {c_size - 1};
        return c_map.slice(offsets_p, extents_p) - c_map.slice(offsets, extents);
    }

    void SolverEigen::solve(int nb_eigen_values, vecd v0, double tol, double tol_eigen_vectors)
    {
        int i = 0;
    }

} // namespace solver
