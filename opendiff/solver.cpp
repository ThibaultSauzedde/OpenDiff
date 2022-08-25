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

    // void SolverSlepc::makeAdjoint()
    // {
    //     MatTranspose(m_M, MAT_INPLACE_MATRIX, &m_M);
    //     MatTranspose(m_K, MAT_INPLACE_MATRIX, &m_K);
    // }

    void SolverSlepc::solve(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0)
    {
        // MatGetSize(m_M, &nrows, &ncols);
        SlepcInitialize(NULL, NULL, NULL, NULL);

        int pblm_dim = static_cast<int>(m_M.rows());
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

        //convert eigen matrix to slepc one

        // //add 0 to diagonal (mandatory with MatCreateSeqAIJWithArrays)
        // for (int i{0}; i < pblm_dim; ++i)
        // {
        //     m_M.coeffRef(i, i) += 0.;
        //     m_K.coeffRef(i, i) += 0.;
        // }

        m_M.makeCompressed();
        m_K.makeCompressed();

        Mat M;
        Mat K;

        MatCreateSeqAIJWithArrays(PETSC_COMM_WORLD, pblm_dim, pblm_dim, m_M.outerIndexPtr(), m_M.innerIndexPtr(), m_M.valuePtr(), &M);
        MatCreateSeqAIJWithArrays(PETSC_COMM_WORLD, pblm_dim, pblm_dim, m_K.outerIndexPtr(), m_K.innerIndexPtr(), m_K.valuePtr(), &K);

        std::cout << m_M.rows() << std::endl;
        std::cout << m_M.cols() << std::endl;

        std::cout << m_M << std::endl;
        MatView(M, PETSC_VIEWER_STDOUT_WORLD);

        std::cout << m_K << std::endl;
        MatView(K, PETSC_VIEWER_STDOUT_WORLD);

        EPS eps; /* eigenproblem solver context */

        EPSCreate(PETSC_COMM_WORLD, &eps);
        EPSSetOperators(eps, K, M);
        EPSSetProblemType(eps, EPS_GNHEP);
        EPSSetType(eps, EPSKRYLOVSCHUR);
        EPSSetConvergenceTest(eps, EPS_CONV_ABS);
        EPSSetTolerances(eps, tol, 500); //todo: add max iteration for outer and inner
        EPSSetWhichEigenpairs(eps, EPS_LARGEST_MAGNITUDE);

        Vec v0_petsc; /* initial vector */
        MatCreateVecs(M, NULL, &v0_petsc);
        //copy from eigen vector
        VecPlaceArray(v0_petsc, v.data());

        std::cout << v << std::endl;
        VecView(v0_petsc, PETSC_VIEWER_STDOUT_WORLD);

        EPSSetInitialSpace(eps, 1, &v0_petsc);

        std::cout << "test1" << std::endl;

        EPSSolve(eps);

        std::cout << "test2" << std::endl;

        PetscInt nconv;
        PetscScalar kr, ki;
        PetscReal error, re, im;
        Vec xr, xi;
        MatCreateVecs(M, NULL, &xr);
        MatCreateVecs(M, NULL, &xi);

        EPSGetConverged(eps, &nconv);
        PetscPrintf(PETSC_COMM_WORLD, " Number of converged eigenpairs: %" PetscInt_FMT "\n\n", nconv);

        for (int i = 0; i < nconv; i++)
        {
            /*
         Get converged eigenpairs: i-th eigenvalue is stored in kr (real part) and
         ki (imaginary part)
       */
            EPSGetEigenpair(eps, i, &kr, &ki, xr, xi);
            /*
          Compute the relative error associated to each eigenpair
       */
            EPSComputeError(eps, i, EPS_ERROR_RELATIVE, &error);
#if defined(PETSC_USE_COMPLEX)
            re = PetscRealPart(kr);
            im = PetscImaginaryPart(kr);
#else
            re = kr;
            im = ki;
#endif
            if (im != 0.0)
                PetscPrintf(PETSC_COMM_WORLD, " %9f%+9fi %12g\n", (double)re, (double)im, (double)error);
            else
                PetscPrintf(PETSC_COMM_WORLD, "   %12f       %12g\n", (double)re, (double)error);
        }
        PetscPrintf(PETSC_COMM_WORLD, "\n");
        EPSDestroy(&eps);
        SlepcFinalize();
    }

} // namespace solver
