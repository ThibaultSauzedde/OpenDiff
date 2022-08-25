#ifndef SOLVER_H
#define SOLVER_H

#include <Eigen/Sparse>
#include <petscmat.h>

#include "macrolib.h"
#include "diff_operator.h"

namespace solver
{

    using SpMat = Eigen::SparseMatrix<double>; // declares a column-major sparse matrix type of double
    using vecd = std::vector<double>;
    using Tensor1D = Eigen::Tensor<double, 1>;

    Eigen::Tensor<double, 1> delta_coord(vecd &coord);

    template <class T>
    class Solver
    {
    protected:
        T m_K{};
        T m_M{};

        Tensor1D m_volumes{};

        Eigen::VectorXd m_eigen_values{};
        std::vector<Eigen::VectorXd> m_eigen_vectors{};

    public:
        Solver() = delete;
        Solver(const Solver &copy) = default;
        Solver(const T &K, const T &M)
        {
            m_K = K;
            m_M = M;
        };
        Solver(vecd &x, vecd &y, vecd &z, mat::Macrolib &macrolib,
               double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn)
        {
            // calculate dx, dy
            auto dx = delta_coord(x);
            auto dy = delta_coord(y);
            auto dz = delta_coord(z);
            Eigen::array<Eigen::IndexPair<long>, 0> empty_index_list = {};
            Eigen::Tensor<double, 3> vol = dx.contract(dy, empty_index_list).contract(dz, empty_index_list);
            Eigen::array<Eigen::DenseIndex, 1> one_dim({static_cast<int>(vol.size())});

            m_volumes = vol.reshape(one_dim);
            m_K = operators::diff_fission_op<T, Tensor1D>(m_volumes, macrolib);
            auto D = operators::diff_diffusion_op<T, Tensor1D>(dx, dy, dz, macrolib, albedo_x0, albedo_xn, albedo_y0, albedo_yn, albedo_z0, albedo_zn);
            m_M = operators::setup_m_operators<>(D, m_volumes, macrolib);
        };

        Solver(vecd &x, vecd &y, mat::Macrolib &macrolib,
               double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn)
        {
            // calculate dx, dy
            auto dx = delta_coord(x);
            auto dy = delta_coord(y);
            Eigen::array<Eigen::IndexPair<long>, 0> empty_index_list = {};
            Eigen::Tensor<double, 2> surf = dx.contract(dy, empty_index_list);
            Eigen::array<Eigen::DenseIndex, 1> one_dim({static_cast<int>(surf.size())});

            m_volumes = surf.reshape(one_dim);
            m_K = operators::diff_fission_op<T, Tensor1D>(m_volumes, macrolib);
            auto D = operators::diff_diffusion_op<T, Tensor1D>(dx, dy, macrolib, albedo_x0, albedo_xn, albedo_y0, albedo_yn);
            m_M = operators::setup_m_operators<T, Tensor1D>(D, m_volumes, macrolib);
        };

        Solver(vecd &x, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn)
        {
            // calculate dx
            auto dx = delta_coord(x);
            m_volumes = dx;

            m_K = operators::diff_fission_op<T, Tensor1D>(m_volumes, macrolib);
            auto D = operators::diff_diffusion_op<T, Tensor1D>(dx, macrolib, albedo_x0, albedo_xn);
            m_M = operators::setup_m_operators<T, Tensor1D>(D, m_volumes, macrolib);
        };

        const auto getEigenValues() const
        {
            return m_eigen_values;
        };
        const auto getEigenVectors() const
        {
            return m_eigen_vectors;
        };
        const auto getK() const
        {
            return m_K;
        };
        const auto getM() const
        {
            return m_M;
        };

        virtual void makeAdjoint()
        {
            m_M = m_M.adjoint();
            m_K = m_K.adjoint();
        }
        virtual void solve(double tol = 1e-6, double tol_eigen_vectors = 1e-4, int nb_eigen_values = 1, const Eigen::VectorXd &v0 = Eigen::VectorXd()) = 0;
        vecd get_power(mat::Macrolib &macrolib);
    };

    class SolverPowerIt : public Solver<SpMat>
    {
    public:
        SolverPowerIt(const Solver &copy) : Solver(copy){};
        SolverPowerIt(vecd &x, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn) : Solver(x, macrolib, albedo_x0, albedo_xn){};
        SolverPowerIt(vecd &x, vecd &y, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn) : Solver(x, y, macrolib, albedo_x0, albedo_xn, albedo_y0, albedo_yn){};
        SolverPowerIt(vecd &x, vecd &y, vecd &z, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn) : Solver(x, y, z, macrolib, albedo_x0, albedo_xn, albedo_y0, albedo_yn, albedo_z0, albedo_zn){};

        void solve(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0) override;
    };

    // class SolverSlepc : public Solver<Eigen::SparseMatrix<double, Eigen::RowMajor>> // row major for MatCreateSeqAIJWithArrays
    class SolverSlepc : public Solver<Eigen::SparseMatrix<double, Eigen::RowMajor>> // row major for MatCreateSeqAIJWithArrays
    {
    public:
        SolverSlepc(const Solver &copy) : Solver(copy){};
        SolverSlepc(vecd &x, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn) : Solver(x, macrolib, albedo_x0, albedo_xn){};
        SolverSlepc(vecd &x, vecd &y, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn) : Solver(x, y, macrolib, albedo_x0, albedo_xn, albedo_y0, albedo_yn){};
        SolverSlepc(vecd &x, vecd &y, vecd &z, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn) : Solver(x, y, z, macrolib, albedo_x0, albedo_xn, albedo_y0, albedo_yn, albedo_z0, albedo_zn){};

        void solve(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0) override;
    };

} // namespace solver

#endif // SOLVER_H