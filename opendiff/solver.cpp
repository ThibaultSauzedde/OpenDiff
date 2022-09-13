#include <vector>
#include <cmath>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include <slepceps.h>
#include "spdlog/spdlog.h"
#include <string>

#include "solver.h"

namespace solver
{
    using vecd = std::vector<double>;
    using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    using Tensor1D = Eigen::Tensor<double, 1, Eigen::RowMajor>;

    Tensor1D delta_coord(vecd &coord)
    {
        int c_size = static_cast<int>(coord.size());
        auto c_map = Eigen::TensorMap<Tensor1D>(&coord[0], c_size);
        Eigen::array<Eigen::Index, 1> offsets = {0};
        Eigen::array<Eigen::Index, 1> extents = {c_size};
        Eigen::array<Eigen::Index, 1> offsets_p = {1};
        Eigen::array<Eigen::Index, 1> extents_p = {c_size - 1};
        return c_map.slice(offsets_p, extents_p) - c_map.slice(offsets, extents);
    }

    void init_slepc()
    {
        SlepcInitialize(NULL, NULL, NULL, NULL);
    }

    void SolverPowerIt::solve(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0,
                              double tol_inner, int outer_max_iter, int inner_max_iter, std::string inner_solver, std::string inner_precond)
    {
        spdlog::debug("Inner solver : {}", inner_solver);
        spdlog::debug("Inner precond : {}", inner_precond);
        if (inner_solver == "SparseLU")
            SolverPowerIt::solveLU(tol, tol_eigen_vectors, nb_eigen_values, v0, outer_max_iter);
        else if (inner_solver == "LeastSquaresConjugateGradient" && inner_precond.empty())
            SolverPowerIt::solveIterative<Eigen::LeastSquaresConjugateGradient<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0,
                                                                                       tol_inner, outer_max_iter, inner_max_iter);
        else if (inner_solver == "BiCGSTAB" && inner_precond.empty())
            SolverPowerIt::solveIterative<Eigen::GMRES<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0,
                                                               tol_inner, outer_max_iter, inner_max_iter);
        else if (inner_solver == "GMRES" && inner_precond.empty())
            SolverPowerIt::solveIterative<Eigen::BiCGSTAB<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0,
                                                                  tol_inner, outer_max_iter, inner_max_iter);
        else if (inner_solver == "BiCGSTAB" && inner_precond == "IncompleteLUT")
            SolverPowerIt::solveIterative<Eigen::BiCGSTAB<SpMat, Eigen::IncompleteLUT<double>>>(tol, tol_eigen_vectors, nb_eigen_values, v0,
                                                                                                tol_inner, outer_max_iter, inner_max_iter);
        else
            throw std::invalid_argument("The combinaison of inner_solver and inner_precond is not known");
    }

    // todo: find a way to do not repeat with solveIterative
    void SolverPowerIt::solveLU(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0, int outer_max_iter)
    {
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

        Eigen::SparseLU<SpMat> solver; // can't be used with this template because of solveWithGuess
        solver.compute(m_M);

        // outer iteration
        int i = 0;
        while ((r_tol > tol || r_tol_ev > tol_eigen_vectors) && i < outer_max_iter)
        {
            spdlog::debug("----------------------------------------------------");
            spdlog::debug("Outer iteration {}", i);
            auto b = m_K * v;
            // inner iteration
            v = solver.solve(b);
            eigen_value = v.norm();
            v = v / eigen_value;
            // std::cout << v << std::endl;

            // precision computation
            r_tol = std::abs(eigen_value - eigen_value_prec);
            r_tol_ev = abs((v - v_prec).maxCoeff());
            eigen_value_prec = eigen_value;
            v_prec = v;
            spdlog::debug("Eigen value = {:.5f}", eigen_value);
            spdlog::debug("Estimated error in outter iteration (eigen value): {:.2e}", r_tol);
            spdlog::debug("Estimated error in outter iteration (eigen vector): {:.2e}", r_tol_ev);
            i++;
        }
        spdlog::debug("----------------------------------------------------");
        clearEigenValues();
        m_eigen_values.push_back(eigen_value);
        m_eigen_vectors.push_back(v);
        spdlog::debug("Number of outter iteration: {}", i);
        spdlog::info("Eigen value = {:.5f}", eigen_value);
    }

    // void SolverSlepc::makeAdjoint()
    // {
    //     MatTranspose(m_M, MAT_INPLACE_MATRIX, &m_M);
    //     MatTranspose(m_K, MAT_INPLACE_MATRIX, &m_K);
    // }
    void SolverSlepc::solve(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0,
                            double tol_inner, int outer_max_iter, int inner_max_iter, std::string inner_solver, std::string inner_precond)
    {
        SolverSlepc::solveIterative(tol, tol_eigen_vectors, nb_eigen_values, v0,
                                    tol_inner, outer_max_iter, inner_max_iter, "krylovschur", inner_solver, inner_precond);
    }

    //todo add which eigen values (smallest, largest)
    void SolverSlepc::solveIterative(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0,
                                     double tol_inner, int outer_max_iter, int inner_max_iter, std::string solver, std::string inner_solver, std::string inner_precond)
    {
        int pblm_dim = static_cast<int>(m_M.rows());
        int v0_size = static_cast<int>(v0.size());

        Eigen::VectorXd v(v0);

        if (v0_size == 0)
        {
            v.setConstant(pblm_dim, 1.);
        }
        else if (v0_size != pblm_dim)
            throw std::invalid_argument("The size of the initial vector must be identical to the matrix row or column size!");

        spdlog::debug("Solver : {}", solver);
        spdlog::debug("Inner solver : {}", inner_solver);
        spdlog::debug("Inner precond : {}", inner_precond);

        spdlog::debug("Tolerance in outter iteration (eigen value): {:.2e}", tol);
        // spdlog::debug("Tolerance in outter iteration (eigen vector): {:.2e}", tol_eigen_vectors);
        spdlog::debug("Tolerance in inner iteration : {:.2e}", tol_inner);
        spdlog::debug("Max. outer iteration : {}", outer_max_iter);
        spdlog::debug("Max. inner iteration : {}", inner_max_iter);

        // convert eigen matrix to slepc one
        m_M.makeCompressed();
        m_K.makeCompressed();

        Mat M;
        Mat K;
        MatCreateSeqAIJWithArrays(PETSC_COMM_WORLD, pblm_dim, pblm_dim, m_M.outerIndexPtr(), m_M.innerIndexPtr(), m_M.valuePtr(), &M);
        MatCreateSeqAIJWithArrays(PETSC_COMM_WORLD, pblm_dim, pblm_dim, m_K.outerIndexPtr(), m_K.innerIndexPtr(), m_K.valuePtr(), &K);

        // solver creation
        EPS eps; /* eigenproblem solver context */
        ST st;
        KSP ksp;
        PC pc;

        EPSCreate(PETSC_COMM_WORLD, &eps);
        EPSSetOperators(eps, K, M);
        EPSSetProblemType(eps, EPS_GNHEP);
        EPSSetDimensions(eps, nb_eigen_values, PETSC_DECIDE, PETSC_DECIDE);
        EPSSetFromOptions(eps);

        EPSGetST(eps, &st);
        STGetKSP(st, &ksp);
        KSPGetPC(ksp, &pc);

        if (solver == "power")
            EPSSetType(eps, EPSPOWER);
        else if (solver == "arnoldi")
            EPSSetType(eps, EPSARNOLDI);
        else if (solver == "arpack")
            EPSSetType(eps, EPSARPACK);
        else
            EPSSetType(eps, EPSKRYLOVSCHUR);

        if (inner_solver == "cgls")
            KSPSetType(ksp, KSPCGLS);
        else if (inner_solver == "ibcgs")
            KSPSetType(ksp, KSPIBCGS);
        else if (inner_solver == "bcgs")
            KSPSetType(ksp, KSPBCGS);
        else if (inner_solver == "gmres")
            KSPSetType(ksp, KSPGMRES); 	

        if (inner_precond == "jacobi")
            PCSetType(pc, PCJACOBI);
        else if (inner_precond == "sor")
            PCSetType(pc, PCSOR);
        else if (inner_precond == "cholesky")
            PCSetType(pc, PCCHOLESKY);
        else if (inner_precond == "ilu")
            PCSetType(pc, PCILU);
        else if (inner_precond == "asm")
            PCSetType(pc, PCASM);

        EPSSetConvergenceTest(eps, EPS_CONV_REL);
        EPSSetTolerances(eps, tol, outer_max_iter);
        EPSSetWhichEigenpairs(eps, EPS_LARGEST_MAGNITUDE);

        KSPSetTolerances(ksp, tol_inner, PETSC_DEFAULT, PETSC_DEFAULT, inner_max_iter);

        Vec v0_petsc; /* initial vector */
        MatCreateVecs(M, NULL, &v0_petsc);
        // copy from eigen vector
        VecPlaceArray(v0_petsc, v.data());

        EPSSetInitialSpace(eps, 1, &v0_petsc);

        // EPSMonitorSet(eps, EPSMonitorAll, NULL, NULL);
        // PetscViewerAndFormat *monviewer;
        // PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_DEFAULT, &monviewer);
        // // EPSMonitorSet(eps, EPSMonitorAll, monviewer, (PetscErrorCode(*)(void **))PetscViewerDestroy);
        // // EPSMonitorSet(eps, (PetscErrorCode(*)(EPS, PetscInt, PetscInt, PetscScalar *, PetscScalar *, PetscReal *, PetscInt, void *))EPSMonitorAllDrawLG, monviewer,
        // //               (PetscErrorCode(*)(void **))PetscViewerAndFormatDestroy);
        // EPSMonitorSet(eps, (PetscErrorCode(*)(EPS, PetscInt, PetscInt, PetscScalar *, PetscScalar *, PetscReal *, PetscInt, void *))EPSMonitorAll, monviewer,
        //               (PetscErrorCode(*)(void **))PetscViewerAndFormatDestroy);

        // EPSSetTrackAll(eps, PETSC_TRUE);

        // EPSSetTrueResidual(eps, PETSC_TRUE);

        //balancing
        // EPSSetBalance(eps, EPS_BALANCE_ONESIDE, 100, 1e-5);

        EPSSolve(eps);

        PetscInt nconv, its;
        PetscScalar kr, ki;
        PetscReal error, re, im;
        Vec xr, xi;
        PetscScalar *xrdata;
        EPSConvergedReason reason;
        MatCreateVecs(M, NULL, &xr);
        MatCreateVecs(M, NULL, &xi);


        clearEigenValues();

        EPSGetConverged(eps, &nconv);
        EPSGetIterationNumber(eps, &its);
        EPSGetConvergedReason(eps, &reason);
        spdlog::info("Number of converged eigenpairs: {}", nconv);
        spdlog::debug("Number of outter iteration: {}", its);

        std::string str_reason{};

        switch (reason)
        {
        case EPS_CONVERGED_TOL:
            str_reason = "tolerance";
            break;

        case EPS_CONVERGED_USER:
            str_reason = "tolerance (user)";
            break;

        case EPS_DIVERGED_ITS:
            str_reason = "failure";
            break;

        case EPS_DIVERGED_BREAKDOWN:
            str_reason = "failure";
            break;

        case EPS_DIVERGED_SYMMETRY_LOST:
            str_reason = "failure";
            break;
        case EPS_CONVERGED_ITERATING:
            str_reason = "failure";
            break;

        default:
            str_reason = "unknown";
        }
        spdlog::debug("Slepc converged reason: {}", str_reason);

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

            if (im != 0.0 && i < 20)
                spdlog::info("Eigen value {} = {:.5f} i {:.5f} +-  {:.2e}", i, (double)re, (double)im, (double)error);
            else if (im != 0.0)
                spdlog::debug("Eigen value {} = {:.5f} i {:.5f} +-  {:.2e}", i, (double)re, (double)im, (double)error);
            else if (i < 20)
                spdlog::info("Eigen value {} = {:.5f} +- {:.2e}", i, (double)re, (double)error);
            else
                spdlog::debug("Eigen value {} = {:.5f} +- {:.2e}", i, (double)re, (double)error);

            if (i <= nb_eigen_values)
            {
                m_eigen_values.push_back((double)re);
                VecGetArray(xr, &xrdata);
                auto xr_eigen = Eigen::Map<Eigen::VectorXd>(xrdata, pblm_dim);
                VecRestoreArray(xr, &xrdata);
                m_eigen_vectors.push_back(xr_eigen);
            }
        }
        EPSDestroy(&eps);
    }

} // namespace solver
