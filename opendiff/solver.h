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

    Eigen::Tensor<double, 1> delta_coord(vecd &coord);

    template <class T>
    class Solver
    {
    protected:
        T m_K{};
        T m_M{};

        vecd m_eigen_values{};
        std::vector<vecd> m_eigen_vectors{};

    public:
        Solver() = delete;
        Solver(const Solver &copy) = default;
        Solver(const T &K, const T &M)
        {
            m_K = K;
            m_M = M;
        };
        Solver(vecd &x, vecd &y, vecd &z, mat::Macrolib &macrolib,
               double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn);
        Solver(vecd &x, vecd &y, mat::Macrolib &macrolib,
               double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn);
        Solver(vecd &x, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn)
        {
            // calculate dx
            auto dx = delta_coord(x);
            std::cout << dx[0] << std::endl;
            // cast to vect for operators functions
            // todo: use templates for operators functions
            // double *dx_data = dx.data();
            // int dx_size = static_cast<int>(dx.size());
            // vecd dx_v(std::move(dx_data), dx_data + dx_size);
            // for (auto data : dx_v)
            //     std::cout << data << std::endl;
            // auto D = operators::diff_diffusion_op<T>(dx, macrolib, albedo_x0, albedo_xn);
            // setup_operators(D, dx_v, macrolib);
        };

        void setup_operators(T &D, vecd volumes, mat::Macrolib &macrolib)
        {
            // for (auto data : volumes)
            //     std::cout << data << std::endl;
            auto R = operators::diff_removal_op<T>(volumes, macrolib);
        };

        void makeAdjoint();
        virtual void solve(int nb_eigen_values, vecd v0, double tol, double tol_eigen_vectors) = 0;
        vecd get_power(mat::Macrolib &macrolib);

        const auto getEigenValues() const { return m_eigen_values; };
        const auto getEigenVectors() const { return m_eigen_vectors; };
        const auto getK() const { return m_K; };
        const auto getM() const { return m_M; };
    };

    class SolverEigen : public Solver<SpMat>
    {
    public:
        SolverEigen(const Solver &copy) : Solver(copy){};
        SolverEigen(vecd &x, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn) : Solver(x, macrolib, albedo_x0, albedo_xn){};

        void solve(int nb_eigen_values, vecd v0, double tol, double tol_eigen_vectors) override;
    };

} // namespace solver

#endif // SOLVER_H