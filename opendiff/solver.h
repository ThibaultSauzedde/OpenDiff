#ifndef SOLVER_H
#define SOLVER_H

#include <vector>
#include <cmath>
#include <string>
#include "spdlog/spdlog.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

#include <petscmat.h>
#include <slepceps.h>

#include "macrolib.h"
#include "diff_operator.h"

namespace solver
{

    using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>; // declares a RowMajor sparse matrix type of double

    using vecd = std::vector<double>;
    using vecvec = std::vector<Eigen::VectorXd>;

    using Tensor0D = Eigen::Tensor<double, 0, Eigen::RowMajor>;
    using Tensor1D = Eigen::Tensor<double, 1, Eigen::RowMajor>;
    using Tensor2D = Eigen::Tensor<double, 2, Eigen::RowMajor>;
    using Tensor3D = Eigen::Tensor<double, 3, Eigen::RowMajor>;
    using Tensor4Dconst = Eigen::Tensor<const double, 4, Eigen::RowMajor>;

    Tensor1D delta_coord(vecd &coord);

    void init_slepc();

    class Solver
    {
    protected:
        mat::Macrolib m_macrolib;
        Tensor1D m_volumes{};

        bool m_is_normed{false};
        std::string m_norm_method{};

        vecd m_eigen_values{};
        vecvec m_eigen_vectors{};

    public:

        void setEigenValues(vecd eigen_values) //we want a copy 
        {
            m_eigen_values = eigen_values;
        };
        void setEigenVectors(vecvec eigen_vectors) //we want a copy 
        {
            m_eigen_vectors = eigen_vectors;
        };

        void clearEigenValues()
        {
            m_eigen_values.clear();
            m_eigen_vectors.clear();
        };

        void pushEigenValue(double eigen_value)
        {
            m_eigen_values.push_back(eigen_value);
        };

        void pushEigenVector(Eigen::VectorXd &eigen_vector)
        {
            m_eigen_vectors.push_back(eigen_vector);
        };

        const auto isNormed() const
        {
            return m_is_normed;
        };
        const auto getNormMethod() const
        {
            return m_norm_method;
        };

        const auto &getEigenValues() const
        {
            return m_eigen_values;
        };

    
        const auto getEigenValue(int i) const
        {
            return m_eigen_values[i];
        };

        const auto &getEigenVectors() const
        {
            return m_eigen_vectors;
        };

        const auto &getEigenVector(int i) const
        {
            return m_eigen_vectors[i];
        };

        const auto getEigenVector(int i, int i_grp) const
        {
            if (i_grp < 1 || i_grp > m_macrolib.getNbGroups())
                throw std::invalid_argument("The wanted nrj group (" + std::to_string(i_grp) + ") is not in the materials");

            auto dim = m_macrolib.getDim();
            auto dim_xyz = std::get<2>(dim)*std::get<1>(dim)* std::get<0>(dim);
            return m_eigen_vectors[i](Eigen::seqN(dim_xyz * (i_grp -1), dim_xyz));
        };

        const auto getEigenVector4D(int i, int dim_x, int dim_y, int dim_z, int nb_groups) const
        {
            Eigen::TensorMap<Tensor4Dconst> a(m_eigen_vectors[i].data(), nb_groups, dim_z, dim_y, dim_x);
            return a;
        };

        const auto getEigenVector4D(int i) const
        {
            auto nb_groups = m_macrolib.getNbGroups();
            auto dim = m_macrolib.getDim();
            auto dim_z = std::get<2>(dim), dim_y = std::get<1>(dim), dim_x = std::get<0>(dim);
            return getEigenVector4D(i, dim_x, dim_y, dim_z, nb_groups);
        };

        const py::array_t<double> getEigenVectorPython(int i) const
        {
            auto ev = getEigenVector4D(i);
            return py::array_t<double, py::array::c_style>({ev.dimension(0), ev.dimension(1), ev.dimension(2), ev.dimension(3)},
                                                           ev.data());
        };

        // void removeEigenVectors(std::vector<int> ids)
        // {
        //     for (auto id : ids)
        //     {
        //         m_eigen_values.erase(m_eigen_values.begin() + i);
        //         m_eigen_vectors.erase(m_eigen_vectors.begin() + i);
        //     }
        // }

        const auto &getVolumes() const
        {
            return m_volumes;
        };

        const auto getVolumesPython() const
        {
            return py::array_t<double, py::array::c_style>({m_volumes.dimension(0)},
                                                           m_volumes.data());
        };

        virtual void makeAdjoint() = 0;

        virtual void solve(double tol = 1e-6, double tol_eigen_vectors = 1e-4, int nb_eigen_values = 1, const Eigen::VectorXd &v0 = Eigen::VectorXd(), double ev0 = 1,
                           double tol_inner = 1e-6, int outer_max_iter = 500, int inner_max_iter = 200, std::string inner_solver = "BiCGSTAB", std::string inner_precond = "") = 0;

        void handleDenegeratedEigenvalues(double max_eps = 1e-6);

        // todo: use getPower(Tensor4Dconst
        // todo: use matrix muktiplication
        Tensor3D getPower(int i = 0);

        const py::array_t<double> getPowerPython();

        const Tensor3D normPower(double power_W = 1);

        const py::array_t<double> normPowerPython(double power_W = 1);

        void normPhiStarMPhi(solver::Solver &solver_star);

        virtual double getPhiStarMPhi(solver::Solver &solver_star, int i) = 0;

        void norm(std::string method, solver::Solver &solver_star, double power_W = 1);
    };

    template <class T>
    class SolverFull : public Solver
    {
    protected:
        T m_K{};
        T m_M{};

    public:
        using Solver::Solver;

        SolverFull(const SolverFull &copy) = default;

        SolverFull(vecd &x, vecd &y, vecd &z, mat::Macrolib &macrolib,
               double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn);

        SolverFull(vecd &x, vecd &y, mat::Macrolib &macrolib,
               double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn);

        SolverFull(vecd &x, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn);

        SolverFull(const T &K, const T &M)
        {
            m_K = K;
            m_M = M;
        };

        const auto &getK() const
        {
            return m_K;
        };
        const auto &getM() const
        {
            return m_M;
        };

        void makeAdjoint()
        {
            m_M = m_M.adjoint();
            m_K = m_K.adjoint();
        };

        double getPhiStarMPhi(solver::Solver &solver_star, int i)
        {
            return solver_star.getEigenVectors()[i].dot(m_M * m_eigen_vectors[i]);
        };
    };
    

    class SolverFullPowerIt : public SolverFull<SpMat>
    {
    public:
        using SolverFull<SpMat>::SolverFull;

        // SolverFullPowerIt(const SolverFullPowerIt &copy) = default;

        void solve(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0, double ev0,
                   double tol_inner, int outer_max_iter, int inner_max_iter, std::string inner_solver, std::string inner_precond) override;

        void solveLU(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0, double ev0, int outer_max_iter);

        template <class T>
        void solveIterative(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0, double ev0,
                            double tol_inner, int outer_max_iter, int inner_max_iter);
    };

    class SolverFullSlepc : public SolverFull<SpMat> // row major for MatCreateSeqAIJWithArrays
    {
    public:
        using SolverFull<SpMat>::SolverFull;

        // SolverFullSlepc(const SolverFullSlepc &copy) = default;

        void solve(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0, double ev0,
                   double tol_inner, int outer_max_iter, int inner_max_iter, std::string inner_solver, std::string inner_precond) override;

        void solveIterative(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0, double ev0,
                            double tol_inner, int outer_max_iter, int inner_max_iter, std::string solver, std::string inner_solver, std::string inner_precond);
    };

#include "solver.inl"

} // namespace solver

#endif // SOLVER_H