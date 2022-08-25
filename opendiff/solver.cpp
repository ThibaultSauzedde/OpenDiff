#include <vector>
#include <cmath>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include <slepceps.h>

#include "solver.h"

namespace solver
{
    using vecd = std::vector<double>;
    using SpMat = Eigen::SparseMatrix<double>;

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

    void SolverPowerIt::makeAdjoint()
    {
        m_M = m_M.adjoint();
        m_K = m_K.adjoint();
    }

    void SolverPowerIt::solve(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0)
    {
        //todo: add parmter for choosing the solver for ax =b
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

        // Eigen::LeastSquaresConjugateGradient<SpMat> solver;
        // Eigen::GMRES<SpMat> solver(m_M);
        Eigen::BiCGSTAB<SpMat> solver;
        solver.compute(m_M);
        // Eigen::SparseLU<SpMat> solver;
        // solver.analyzePattern(m_M);
        // // Compute the numerical factorization
        // solver.factorize(m_M);
        // std::cout << m_M << std::endl;

        // outer iteration
        while (r_tol > tol || r_tol_ev > tol_eigen_vectors)
        {
            auto b = m_K * v;
            // inner iteration
            v = solver.solveWithGuess(b, v);
            std::cout << "#iterations:     " << solver.iterations() << std::endl;
            std::cout << "estimated error: " << solver.error() << std::endl;
            eigen_value = v.norm();
            v = v / eigen_value;
            // std::cout << v << std::endl;

            // precision computation
            r_tol = std::abs(eigen_value - eigen_value_prec);
            r_tol_ev = abs((v - v_prec).maxCoeff());
            eigen_value_prec = eigen_value;
            v_prec = v;
            std::cout << r_tol << " " << r_tol_ev << std::endl;
        }

        std::cout << "vp = " << eigen_value << std::endl;
    }

    void SolverSlepc::makeAdjoint()
    {
        MatTranspose(m_M, MAT_INPLACE_MATRIX, &m_M);
        MatTranspose(m_K, MAT_INPLACE_MATRIX, &m_K);
    }

    void SolverSlepc::solve(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0)
    {
        //todo: add parameter for choosing the solver for ax =b
        PetscInt nrows, ncols;
        MatGetSize(m_M, &nrows, &ncols);
        int pblm_dim = static_cast<int>(nrows);
        int v0_size = static_cast<int>(v0.size());

        float r_tol = 1e5;
        float r_tol_ev = 1e5;

        Eigen::VectorXd v(v0);

        if (v0_size == 0)
        {
            v.setConstant(pblm_dim, 1.);
        }
        else if (v0_size != pblm_dim)
            throw std::invalid_argument("The size of the initial vector must be identical to the matrix row or column size!");

        EPS eps; /* eigenproblem solver context */

        EPSCreate(PETSC_COMM_WORLD, &eps);
        EPSSetOperators(eps, m_K, m_M);
        EPSSetProblemType(eps, EPS_GNHEP);
        EPSSetFromOptions(eps);

        Vec v0_petsc; /* initial vector */
        //copy from eigen vector
        VecPlaceArray(v0_petsc, v0.data());
        EPSSetInitialSpace(eps, 1, &v0_petsc);
    }

} // namespace solver
