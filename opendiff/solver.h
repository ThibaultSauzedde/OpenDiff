#ifndef SOLVER_H
#define SOLVER_H

#include <Eigen/Sparse>
#include <petscmat.h>
#include <string>
#include "spdlog/spdlog.h"

#include "macrolib.h"
#include "diff_operator.h"

namespace solver
{

    using SpMat = Eigen::SparseMatrix<double>; // declares a column-major sparse matrix type of double
    using vecd = std::vector<double>;
    using Tensor1D = Eigen::Tensor<double, 1>;

    Eigen::Tensor<double, 1> delta_coord(vecd &coord);

    void init_slepc();

    template <class T>
    class Solver
    {
    protected:
        T m_K{};
        T m_M{};

        Tensor1D m_volumes{};

        std::vector<double> m_eigen_values{};
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

        virtual void solve(double tol = 1e-6, double tol_eigen_vectors = 1e-4, int nb_eigen_values = 1, const Eigen::VectorXd &v0 = Eigen::VectorXd(),
                           double tol_inner = 1e-6, int outer_max_iter = 500, int inner_max_iter = 200, std::string inner_solver = "BiCGSTAB", std::string inner_precond = "") = 0;
        vecd get_power(mat::Macrolib &macrolib);
    };

    class SolverPowerIt : public Solver<SpMat>
    {
    public:
        SolverPowerIt(const Solver &copy) : Solver(copy){};
        SolverPowerIt(vecd &x, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn) : Solver(x, macrolib, albedo_x0, albedo_xn){};
        SolverPowerIt(vecd &x, vecd &y, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn) : Solver(x, y, macrolib, albedo_x0, albedo_xn, albedo_y0, albedo_yn){};
        SolverPowerIt(vecd &x, vecd &y, vecd &z, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn) : Solver(x, y, z, macrolib, albedo_x0, albedo_xn, albedo_y0, albedo_yn, albedo_z0, albedo_zn){};

        void solve(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0,
                   double tol_inner, int outer_max_iter, int inner_max_iter, std::string inner_solver, std::string inner_precond) override;

        void solveLU(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0, int outer_max_iter);

        template <class T>
        void solveIterative(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0,
                            double tol_inner, int outer_max_iter, int inner_max_iter)
        {
            // todo: add parmter for choosing the solver for ax =b
            int pblm_dim = static_cast<int>(m_M.rows());
            int v0_size = static_cast<int>(v0.size());

            float r_tol = 1e5;
            float r_tol_ev = 1e5;

            Eigen::VectorXd v(v0);
            Eigen::VectorXd v_prec(v0);

            if (v0_size == 0)
            {
                v.setConstant(pblm_dim, 1.);
                v_prec.setConstant(pblm_dim, 1.);
            }
            else if (v0_size != pblm_dim)
                throw std::invalid_argument("The size of the initial vector must be identical to the matrix row or column size!");

            if (nb_eigen_values != 1)
                throw std::invalid_argument("Only one eigen value can be computed with PI!");

            double eigen_value = v.norm();
            double eigen_value_prec = eigen_value;

            T solver;
            solver.setMaxIterations(inner_max_iter);
            solver.setTolerance(tol_inner);
            solver.compute(m_M);

            // outer iteration
            int i = 0;
            while (r_tol > tol || r_tol_ev > tol_eigen_vectors || i > outer_max_iter)
            {
                spdlog::debug("----------------------------------------------------");
                spdlog::debug("Outer iteration {}", i);
                auto b = m_K * v;
                // inner iteration
                v = solver.solveWithGuess(b, v);
                spdlog::debug("Number of inner iteration: {}", solver.iterations());
                spdlog::debug("Estimated error in inner iteration: {:.2e}", solver.error());
                eigen_value = v.norm();
                v = v / eigen_value;
                // std::cout << v << std::endl;

                // precision computation
                r_tol = std::abs(eigen_value - eigen_value_prec);
                r_tol_ev = abs((v - v_prec).maxCoeff());
                eigen_value_prec = eigen_value;
                v_prec = v;
                spdlog::debug("Estimated error in outter iteration (eigen value): {:.2e}", r_tol);
                spdlog::debug("Estimated error in outter iteration (eigen vector): {:.2e}", r_tol_ev);
                i++;
            }
            spdlog::debug("----------------------------------------------------");
            m_eigen_values.clear();
            m_eigen_vectors.clear();
            m_eigen_values.push_back(eigen_value);
            m_eigen_vectors.push_back(v);
            spdlog::debug("Number of outter iteration: {}", i);
            spdlog::info("Eigen value = {:.5f}", eigen_value);
        }
    };

    // class SolverSlepc : public Solver<Eigen::SparseMatrix<double, Eigen::RowMajor>>
    class SolverSlepc : public Solver<Eigen::SparseMatrix<double, Eigen::RowMajor>> // row major for MatCreateSeqAIJWithArrays
    {
    public:
        SolverSlepc(const Solver &copy) : Solver(copy){};
        SolverSlepc(vecd &x, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn) : Solver(x, macrolib, albedo_x0, albedo_xn){};
        SolverSlepc(vecd &x, vecd &y, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn) : Solver(x, y, macrolib, albedo_x0, albedo_xn, albedo_y0, albedo_yn){};
        SolverSlepc(vecd &x, vecd &y, vecd &z, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn) : Solver(x, y, z, macrolib, albedo_x0, albedo_xn, albedo_y0, albedo_yn, albedo_z0, albedo_zn){};

        void solve(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0,
                   double tol_inner, int outer_max_iter, int inner_max_iter, std::string inner_solver, std::string inner_precond) override;

        void solveIterative(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0,
                            double tol_inner, int outer_max_iter, int inner_max_iter, std::string solver, std::string inner_solver, std::string inner_precond);
    };

} // namespace solver

#endif // SOLVER_H