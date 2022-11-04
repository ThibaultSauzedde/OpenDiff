inline void init_slepc()
{
    SlepcInitialize(NULL, NULL, NULL, NULL);
}

inline Tensor1D delta_coord(vecd &coord)
{
    int c_size = static_cast<int>(coord.size());
    auto c_map = Eigen::TensorMap<Tensor1D>(&coord[0], c_size);
    Eigen::array<Eigen::Index, 1> offsets = {0};
    Eigen::array<Eigen::Index, 1> extents = {c_size};
    Eigen::array<Eigen::Index, 1> offsets_p = {1};
    Eigen::array<Eigen::Index, 1> extents_p = {c_size - 1};
    return c_map.slice(offsets_p, extents_p) - c_map.slice(offsets, extents);
}

//-------------------------------------------------------------------------
// Solver
//-------------------------------------------------------------------------

// void handleDenegeratedEigenvalues(double max_eps)
// {
//     // get degenerated eigen values

//     // handle each group of degenerated eigen values (if the eigenvectors are not orthogonal)
// }

// todo: use getPower(Tensor4Dconst
// todo: use matrix muktiplication
inline Tensor3D Solver::getPower(int i)
{
    auto nb_groups = m_macrolib.getNbGroups();

    auto dim = m_macrolib.getDim();
    auto dim_z = std::get<2>(dim), dim_y = std::get<1>(dim), dim_x = std::get<0>(dim);
    Tensor3D power(dim_z, dim_y, dim_x); // z, y, x
    power.setZero();
    auto eigenvectori = getEigenVector4D(i, dim_x, dim_y, dim_z, nb_groups);

    for (int i{0}; i < nb_groups; ++i)
    {
        power = power.eval() + m_macrolib.getValues(i + 1, "SIGF") * eigenvectori.chip(i, 0) * m_macrolib.getValues(i + 1, "EFISS");
    }

    return power;
}

// Tensor3D getPower(Tensor4Dconst eigenvectori)
// {
//     auto nb_groups = m_macrolib.getNbGroups();

//     auto dim = m_macrolib.getDim();
//     auto dim_z = std::get<2>(dim), dim_y = std::get<1>(dim), dim_x = std::get<0>(dim);
//     Tensor3D power(dim_z, dim_y, dim_x); // z, y, x
//     power.setZero();

//     for (int i{0}; i < nb_groups; ++i)
//     {
//         if (!m_macrolib.isIn(i + 1, "EFISS"))
//             m_macrolib.addReaction(i + 1, "EFISS", e_fiss_J);

//         if (!m_macrolib.isIn(i + 1, "SIGF"))
//             m_macrolib.addReaction(i + 1, "SIGF", m_macrolib.getValues(i + 1, "NU_SIGF") / nu);

//         power = power.eval() + m_macrolib.getValues(i + 1, "SIGF") * eigenvectori.chip(i, 3) * m_macrolib.getValues(i + 1, "EFISS");
//     }

//     return power;
// }

// Tensor3D getPower(int i = 0)
// {
//     auto eigenvectori = getEigenVector4D(i);
//     auto power = getPower(eigenvectori);
//     return power;
// }

inline const py::array_t<double> Solver::getPowerPython()
{
    auto power = getPower(0);
    return py::array_t<double, py::array::c_style>({power.dimension(0), power.dimension(1), power.dimension(2)},
                                                   power.data());
}

inline const Tensor3D Solver::normPower(double power_W)
{
    auto power = getPower(0);
    Tensor0D power_sum = power.sum();
    double factor = power_W * 1 / power_sum(0);
    for (auto &ev : m_eigen_vectors)
    {
        ev = ev * factor;
    }
    m_is_normed = true;
    m_norm_method = "power";
    return power * factor;
}

inline const py::array_t<double> Solver::normPowerPython(double power_W)
{
    auto power = normPower(power_W);
    return py::array_t<double, py::array::c_style>({power.dimension(0), power.dimension(1), power.dimension(2)},
                                                   power.data());
}

inline void Solver::normPhiStarMPhi(solver::Solver &solver_star)
{
    auto eigen_vectors_star = solver_star.getEigenVectors();
    // if (eigen_vectors_star.size() != m_eigen_vectors.size())
    //     throw std::invalid_argument("The number of eigen vectors has the be identical in this and solver_star!");

    int nb_ev = static_cast<int>(std::min(m_eigen_vectors.size(), eigen_vectors_star.size()));
    for (int i{0}; i < nb_ev; ++i)
    {
        double factor = getPhiStarMPhi(solver_star, i);
        m_eigen_vectors[i] = m_eigen_vectors[i] / factor;
    }
    m_is_normed = true;
    m_norm_method = "PhiStarMPhi";
}

inline void Solver::norm(std::string method, solver::Solver &solver_star, double power_W)
{
    if (method == "power")
        normPower(power_W);
    else if (method == "PhiStarMPhi")
        normPhiStarMPhi(solver_star);
    else
        throw std::invalid_argument("Invalid method name!");
}


template <class T>
void Solver::initInnerSolver(T &inner_solver, double tol_inner, int inner_max_iter)
{
    inner_solver.setMaxIterations(inner_max_iter);
    inner_solver.setTolerance(tol_inner);
}

//specialization
template <>
inline void Solver::initInnerSolver(Eigen::SparseLU<SpMat> &inner_solver, double tol_inner, int inner_max_iter){};

template <class T>
Eigen::VectorXd Solver::solveInner(T &inner_solver, Eigen::VectorXd &b, Eigen::VectorXd &x)
{
    x = inner_solver.solveWithGuess(b, x);
    if (inner_solver.iterations() == 0) // add a small perturbation because of conv issues
    {
        Eigen::VectorXd x0 = x * 1.001;
        x = inner_solver.solveWithGuess(b, x0);
    }
    spdlog::debug("Number of inner iteration: {}", inner_solver.iterations());
    spdlog::debug("Estimated error in inner iteration: {:.2e}", inner_solver.error());
    return x ;
}

//specialization
template <>
inline Eigen::VectorXd Solver::solveInner(Eigen::SparseLU<SpMat> &inner_solver, Eigen::VectorXd &b, Eigen::VectorXd &x)
{
    x = inner_solver.solve(b);
    return x ;
}

template <class T>
Eigen::VectorXd Solver::solveInner(T &inner_solver, Eigen::VectorXd &b, Eigen::VectorBlock<Eigen::VectorXd> &x)
{
    x = inner_solver.solveWithGuess(b, x);
    if (inner_solver.iterations() == 0) // add a small perturbation because of conv issues
    {
        Eigen::VectorXd x0 = x * 1.001;
        x = inner_solver.solveWithGuess(b, x0);
    }
    spdlog::debug("Number of inner iteration: {}", inner_solver.iterations());
    spdlog::debug("Estimated error in inner iteration: {:.2e}", inner_solver.error());
    return x ;
}

//specialization
template <>
inline Eigen::VectorXd Solver::solveInner(Eigen::SparseLU<SpMat> &inner_solver, Eigen::VectorXd &b, Eigen::VectorBlock<Eigen::VectorXd> &x)
{
    x = inner_solver.solve(b);
    return x ;
}


//-------------------------------------------------------------------------
// SolverFull
//-------------------------------------------------------------------------

template <class T>
SolverFull<T>::SolverFull(vecd &x, vecd &y, vecd &z, mat::Macrolib &macrolib,
                  double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn)
{
    // calculate dx, dy
    auto dx = delta_coord(x);
    auto dy = delta_coord(y);
    auto dz = delta_coord(z);
    Eigen::array<Eigen::IndexPair<long>, 0> empty_index_list = {};
    Tensor3D vol = dx.contract(dy, empty_index_list).contract(dz, empty_index_list);
    Eigen::array<Eigen::DenseIndex, 1> one_dim({static_cast<int>(vol.size())});

    m_volumes = vol.reshape(one_dim);
    m_K = operators::diff_fission_op<T, Tensor1D>(m_volumes, macrolib);
    auto D = operators::diff_diffusion_op<T, Tensor1D>(dx, dy, dz, macrolib, albedo_x0, albedo_xn, albedo_y0, albedo_yn, albedo_z0, albedo_zn);
    m_M = operators::setup_m_operators<>(D, m_volumes, macrolib);

    m_macrolib = macrolib;
}

template <class T>
SolverFull<T>::SolverFull(vecd &x, vecd &y, mat::Macrolib &macrolib,
                  double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn)
{
    // calculate dx, dy
    auto dx = delta_coord(x);
    auto dy = delta_coord(y);
    Eigen::array<Eigen::IndexPair<long>, 0> empty_index_list = {};
    Tensor2D surf = dx.contract(dy, empty_index_list);
    Eigen::array<Eigen::DenseIndex, 1> one_dim({static_cast<int>(surf.size())});

    m_volumes = surf.reshape(one_dim);
    m_K = operators::diff_fission_op<T, Tensor1D>(m_volumes, macrolib);
    auto D = operators::diff_diffusion_op<T, Tensor1D>(dx, dy, macrolib, albedo_x0, albedo_xn, albedo_y0, albedo_yn);
    m_M = operators::setup_m_operators<T, Tensor1D>(D, m_volumes, macrolib);

    m_macrolib = macrolib;
}

template <class T>
SolverFull<T>::SolverFull(vecd &x, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn)
{
    // calculate dx
    auto dx = delta_coord(x);
    m_volumes = dx;

    m_K = operators::diff_fission_op<T, Tensor1D>(m_volumes, macrolib);
    auto D = operators::diff_diffusion_op<T, Tensor1D>(dx, macrolib, albedo_x0, albedo_xn);
    m_M = operators::setup_m_operators<T, Tensor1D>(D, m_volumes, macrolib);

    m_macrolib = macrolib;
}

//-------------------------------------------------------------------------
// SolverFullPowerIt
//-------------------------------------------------------------------------


inline void SolverFullPowerIt::solve(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0, double ev0,
                                     double tol_inner, int outer_max_iter, int inner_max_iter, std::string inner_solver, std::string inner_precond)
{
    spdlog::debug("Inner solver : {}", inner_solver);
    spdlog::debug("Inner precond : {}", inner_precond);
    if (inner_solver == "SparseLU")
        SolverFullPowerIt::solveIterative<Eigen::SparseLU<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                           tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "LeastSquaresConjugateGradient" && inner_precond.empty())
        SolverFullPowerIt::solveIterative<Eigen::LeastSquaresConjugateGradient<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                                   tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "BiCGSTAB" && inner_precond.empty())
        SolverFullPowerIt::solveIterative<Eigen::BiCGSTAB<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                              tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "GMRES" && inner_precond.empty())
        SolverFullPowerIt::solveIterative<Eigen::GMRES<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                           tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "BiCGSTAB" && inner_precond == "IncompleteLUT")
        SolverFullPowerIt::solveIterative<Eigen::BiCGSTAB<SpMat, Eigen::IncompleteLUT<double>>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                                            tol_inner, outer_max_iter, inner_max_iter);
    else
        throw std::invalid_argument("The combinaison of inner_solver and inner_precond is not known");
}

template <class T>
void SolverFullPowerIt::solveIterative(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0, double ev0,
                                   double tol_inner, int outer_max_iter, int inner_max_iter)
{

    int pblm_dim = static_cast<int>(m_M.rows());
    int v0_size = static_cast<int>(v0.size());

    float r_tol = 1e5;
    float r_tol_ev = 1e5;
    float r_tol_ev2 = 1e5;

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

    spdlog::debug("Tolerance in outter iteration (eigen value): {:.2e}", tol);
    spdlog::debug("Tolerance in outter iteration (eigen vector): {:.2e}", tol_eigen_vectors);
    spdlog::debug("Tolerance in inner iteration : {:.2e}", tol_inner);
    spdlog::debug("Max. outer iteration : {}", outer_max_iter);
    spdlog::debug("Max. inner iteration : {}", inner_max_iter);

    double eigen_value = ev0;
    double eigen_value_prec = eigen_value;

    T solver;
    initInnerSolver<T>(solver, tol_inner, inner_max_iter);
    solver.compute(m_M);

    // outer iteration
    int i = 0;
    while ((r_tol > tol || r_tol_ev > tol_eigen_vectors || r_tol_ev2 > tol_eigen_vectors) && i < outer_max_iter)
    {
        spdlog::debug("----------------------------------------------------");
        spdlog::debug("Outer iteration {}", i);
        Eigen::VectorXd b = m_K * v;
        // inner iteration
        v = solveInner<T>(solver, b, v);

        eigen_value = v_prec.dot(v) ; //  v_norm_prec == 1 

        r_tol_ev = ((v.array() / v_prec.array()).maxCoeff() - (v.array() / v_prec.array()).minCoeff()) / (2 * eigen_value);

        v /= v.norm() ; 

        // convergence computation
        r_tol = std::abs(eigen_value - eigen_value_prec);
        r_tol_ev2 = (v - v_prec).norm() / eigen_value;
        // r_tol_ev2 = abs((v - v_prec).maxCoeff()) / eigen_value ;

        eigen_value_prec = eigen_value;
        v_prec = v;
        spdlog::debug("Eigen value = {:.5f}", eigen_value);
        spdlog::debug("Estimated error in outter iteration (eigen value): {:.2e}", r_tol);
        spdlog::debug("Estimated error in outter iteration (eigen vector): {:.2e}", r_tol_ev);
        spdlog::debug("Estimated error in outter iteration (eigen vector 2): {:.2e}", r_tol_ev2);
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

//-------------------------------------------------------------------------
// SolverFullSlepc
//-------------------------------------------------------------------------



// void SolverFullSlepc::makeAdjoint()
// {
//     MatTranspose(m_M, MAT_INPLACE_MATRIX, &m_M);
//     MatTranspose(m_K, MAT_INPLACE_MATRIX, &m_K);
// }

inline void SolverFullSlepc::solve(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0, double ev0,
                               double tol_inner, int outer_max_iter, int inner_max_iter, std::string inner_solver, std::string inner_precond)
{
    SolverFullSlepc::solveIterative(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                tol_inner, outer_max_iter, inner_max_iter, "krylovschur", inner_solver, inner_precond);
}

// todo add which eigen values (smallest, largest)
inline void SolverFullSlepc::solveIterative(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0, double ev0,
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

    // balancing
    //  EPSSetBalance(eps, EPS_BALANCE_ONESIDE, 100, 1e-5);

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

//-------------------------------------------------------------------------
// SolverCond
//-------------------------------------------------------------------------

template <class T>
SolverCond<T>::SolverCond(vecd &x, vecd &y, vecd &z, mat::Macrolib &macrolib,
                  double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn)
{
    m_macrolib = macrolib;

    // calculate dx, dy
    auto dx = delta_coord(x);
    auto dy = delta_coord(y);
    auto dz = delta_coord(z);
    Eigen::array<Eigen::IndexPair<long>, 0> empty_index_list = {};
    Tensor3D vol = dx.contract(dy, empty_index_list).contract(dz, empty_index_list);
    Eigen::array<Eigen::DenseIndex, 1> one_dim({static_cast<int>(vol.size())});

    m_volumes = vol.reshape(one_dim);
    
    std::vector<T> D{};
    for (int g{0}; g < m_macrolib.getNbGroups(); ++g)
        D.push_back(operators::diff_diffusion_op<T, Tensor1D>(g+1, dx, dy, dz, macrolib,
         albedo_x0, albedo_xn, albedo_y0, albedo_yn, albedo_z0, albedo_zn));

    operators::setup_cond_operators<T, Tensor1D>(m_F, m_chi, m_A, m_S,
                           D, m_volumes, m_macrolib);


}

template <class T>
SolverCond<T>::SolverCond(vecd &x, vecd &y, mat::Macrolib &macrolib,
                  double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn)
{
    m_macrolib = macrolib;

    // calculate dx, dy
    auto dx = delta_coord(x);
    auto dy = delta_coord(y);
    Eigen::array<Eigen::IndexPair<long>, 0> empty_index_list = {};
    Tensor2D surf = dx.contract(dy, empty_index_list);
    Eigen::array<Eigen::DenseIndex, 1> one_dim({static_cast<int>(surf.size())});

    m_volumes = surf.reshape(one_dim);

    std::vector<T> D{};
    for (int g{0}; g < m_macrolib.getNbGroups(); ++g)
        D.push_back(operators::diff_diffusion_op<T, Tensor1D>(g+1, dx, dy, macrolib,
             albedo_x0, albedo_xn, albedo_y0, albedo_yn));

    operators::setup_cond_operators<T, Tensor1D>(m_F, m_chi, m_A, m_S,
                           D, m_volumes, m_macrolib);
}

template <class T>
SolverCond<T>::SolverCond(vecd &x, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn)
{
    m_macrolib = macrolib;

    // calculate dx
    auto dx = delta_coord(x);
    m_volumes = dx;

    std::vector<T> D{};
    for (int g{0}; g < m_macrolib.getNbGroups(); ++g)
        D.push_back(operators::diff_diffusion_op<T, Tensor1D>(g+1, dx, macrolib, albedo_x0, albedo_xn));

    operators::setup_cond_operators<T, Tensor1D>(m_F, m_chi, m_A, m_S,
                           D, m_volumes, m_macrolib);
}

template <class T>
const auto SolverCond<T>::getFissionSource(Eigen::VectorXd &eigen_vector)
{   

    Eigen::VectorXd psi = Eigen::VectorXd::Zero(m_volumes.size());

    for (int g{0}; g < m_macrolib.getNbGroups(); ++g)
    {
        psi += m_F[g] * getEigenVector(g+1, eigen_vector) ; 
    }
    return psi ;     
}

inline void SolverCondPowerIt::solve(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0, double ev0,
                                 double tol_inner, int outer_max_iter, int inner_max_iter, std::string inner_solver, std::string inner_precond)
{
    spdlog::debug("Inner solver : {}", inner_solver);
    spdlog::debug("Inner precond : {}", inner_precond);

    if (inner_solver == "SparseLU")
        SolverCondPowerIt::solveIterative<Eigen::SparseLU<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                                   tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "ConjugateGradient" && inner_precond.empty())
        SolverCondPowerIt::solveIterative<Eigen::ConjugateGradient<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                                   tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "LeastSquaresConjugateGradient" && inner_precond.empty())
        SolverCondPowerIt::solveIterative<Eigen::LeastSquaresConjugateGradient<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                                   tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "BiCGSTAB" && inner_precond.empty())
        SolverCondPowerIt::solveIterative<Eigen::BiCGSTAB<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                              tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "GMRES" && inner_precond.empty())
        SolverCondPowerIt::solveIterative<Eigen::GMRES<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                           tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "BiCGSTAB" && inner_precond == "IncompleteLUT")
        SolverCondPowerIt::solveIterative<Eigen::BiCGSTAB<SpMat, Eigen::IncompleteLUT<double>>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                                            tol_inner, outer_max_iter, inner_max_iter);
    else
        throw std::invalid_argument("The combinaison of inner_solver and inner_precond is not known");
}


template <class T>
void SolverCondPowerIt::solveIterative(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0, double ev0,
                                       double tol_inner, int outer_max_iter, int inner_max_iter)
{

    int pblm_dim = m_macrolib.getNbGroups() * static_cast<int>(m_volumes.size());
    int v0_size = static_cast<int>(v0.size());
    auto dim = m_macrolib.getDim();
    auto dim_xyz = std::get<2>(dim)*std::get<1>(dim)* std::get<0>(dim);

    float r_tol = 1e5;
    float r_tol_ev = 1e5;
    float r_tol_ev2 = 1e5;

    Eigen::VectorXd v(v0);

    if (v0_size == 0)
    {
        v.setConstant(pblm_dim, 1.);
    }
    else if (v0_size != pblm_dim)
        throw std::invalid_argument("The size of the initial vector must be identical to the matrix row or column size!");

    if (nb_eigen_values != 1)
        throw std::invalid_argument("Only one eigen value can be computed with PI!");

    spdlog::debug("Tolerance in outter iteration (eigen value): {:.2e}", tol);
    spdlog::debug("Tolerance in outter iteration (eigen vector): {:.2e}", tol_eigen_vectors);
    spdlog::debug("Tolerance in inner iteration : {:.2e}", tol_inner);
    spdlog::debug("Max. outer iteration : {}", outer_max_iter);
    spdlog::debug("Max. inner iteration : {}", inner_max_iter);

    double eigen_value = ev0;
    double eigen_value_prec = eigen_value;

    Eigen::VectorXd psi = getFissionSource(v);
    Eigen::VectorXd psi_prec = Eigen::VectorXd{psi};
  
    std::vector<T> solvers(m_macrolib.getNbGroups());

    // init fore the inner iterations
    for (int g{0}; g < m_macrolib.getNbGroups(); ++g)
    {
        initInnerSolver<T>(solvers[g], tol_inner, inner_max_iter);
        solvers[g].compute(m_A[g]);
    }
    
    // outer iteration
    int i = 0;
    while ((r_tol > tol || r_tol_ev > tol_eigen_vectors || r_tol_ev2 > tol_eigen_vectors) && i < outer_max_iter)
    {
        spdlog::debug("----------------------------------------------------");
        spdlog::debug("Outer iteration {}", i);

        // inner iteration
        for (int g{0}; g < m_macrolib.getNbGroups(); ++g)
        {
            // auto phi_g = getEigenVector(g+1, v) ;             
            Eigen::VectorBlock<Eigen::VectorXd> phi_g =  v(Eigen::seqN(dim_xyz * g, dim_xyz));
            Eigen::VectorXd b = m_chi[g] * psi / eigen_value ; 
            for (int gp{0}; gp < m_macrolib.getNbGroups(); ++gp)
            {
                if (g <= gp)
                    continue ; 
                b += m_S[gp][g] * getEigenVector(gp+1, v);
            }   
            spdlog::debug("Inner iteration for group {}", g+1);
            phi_g = solveInner<T>(solvers[g], b, phi_g);
        }
        psi = eigen_value * getFissionSource(v);

        eigen_value = psi_prec.dot(psi) ; //  v_norm_prec == 1 

        Eigen::ArrayXd psi_div = psi.array() / psi_prec.array() ;
        r_tol_ev = (psi_div.maxCoeff<Eigen::NaNPropagationOptions::PropagateNumbers>() -
                    psi_div.minCoeff<Eigen::NaNPropagationOptions::PropagateNumbers>()) / (2 * eigen_value);

        psi /= psi.norm() ; 

        // convergence computation
        r_tol = std::abs(eigen_value - eigen_value_prec);
        r_tol_ev2 = (psi - psi_prec).norm() / eigen_value ;

        eigen_value_prec = eigen_value;
        psi_prec = psi;
        spdlog::debug("Eigen value = {:.5f}", eigen_value);
        spdlog::debug("Estimated error in outter iteration (eigen value): {:.2e}", r_tol);
        spdlog::debug("Estimated error in outter iteration (fission source): {:.2e}", r_tol_ev);
        spdlog::debug("Estimated error in outter iteration (fission source 2): {:.2e}", r_tol_ev2);
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
