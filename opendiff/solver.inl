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

inline void Solver::handleDenegeratedEigenvalues(double max_eps)
{
    // get degenerated eigen values
    int vsize = static_cast<int>(m_eigen_values.size());
    std::vector<int> ids_deg{};
    std::vector<std::vector<int>> ids_group{};
    ids_group.push_back({});
    for (auto i = 1; i < vsize; ++i)
    {
        if (std::abs(m_eigen_values[i] - m_eigen_values[i - 1]) < max_eps)
        {
            // first insertion
            if (ids_group.back().empty())
            {
                ids_group.back().push_back(i - 1);
                ids_group.back().push_back(i);
            }
            // last element of last group == i - 1
            else if (ids_group.back().back() == (i - 1))
                ids_group.back().push_back(i);
            else
            {
                ids_group.push_back({});
                ids_group.back().push_back(i - 1);
                ids_group.back().push_back(i);
            }
        }
    }

    // handle each group of degenerated eigen values (if the eigenvectors are not orthogonal)
    for (auto ids : ids_group)
    {
        if (ids.empty())
            continue;

        int ids_size = static_cast<int>(ids.size());

        // calc the coeff
        std::vector<double> a_ii{};
        a_ii.push_back(m_eigen_vectors[ids[0]].dot(m_eigen_vectors[ids[0]])); // a00

        spdlog::info("New orthogonalisation with a group of {} eigenvectors from {} to {}", ids_size, ids[0], ids.back());
        spdlog::debug("We keep the vector {} with eigenvalue: {:.5f}", ids[0], m_eigen_values[ids[0]]);

        // get the new eigenvectors
        for (auto ev_i = 1; ev_i < ids_size; ++ev_i) // all ev except the first one (we keep it without modif)
        {
            spdlog::debug("Orthogonalisation of eigenvector {} with eigenvalue: {:.5f}", ids[ev_i], m_eigen_values[ids[ev_i]]);
            for (auto k_i = 0; k_i < ev_i; ++k_i) // substract the unwanted part of the vector
            {
                auto ain = m_eigen_vectors[ids[k_i]].dot(m_eigen_vectors[ids.back()]);
                auto coeff = -ain / a_ii[k_i];
                spdlog::debug("Coeff for k_{} = {:.5f} = - {:.5e} / {:.5e}", k_i, coeff, ain, a_ii[k_i]);
                if (std::abs(coeff) < 1e-8)
                    continue;
                m_eigen_vectors[ids[ev_i]] += coeff * m_eigen_vectors[ids[k_i]];
            }
            // append the new norm coeff
            a_ii.push_back(m_eigen_vectors[ids[ev_i]].dot(m_eigen_vectors[ids[ev_i]]));
        }
    }
}

inline bool Solver::isOrthogonal(double max_eps, bool raise_error)
{
    int vsize = static_cast<int>(m_eigen_values.size());

    std::vector<double> all_test{};

    for (auto i = 0; i < vsize; ++i)
    {
        for (auto j = 0; j < vsize; ++j)
        {
            if (j <= i) // i = j or already done
                continue;

            double test = m_eigen_vectors[j].dot(m_eigen_vectors[i]);
            all_test.push_back(std::abs(test));
            if (std::abs(test) > max_eps)
            {
                spdlog::debug("Orthogonality test failed for {}, {}: {:.2e}", i, j, test);
            }
        }
    }

    double max_test = *std::max_element(all_test.begin(), all_test.end());

    spdlog::info("Orthogonality max test : {:.2e}", max_test);
    if (max_test > max_eps && raise_error)
        throw std::invalid_argument("The eigen vector are not bi-orthogonals!");
    else if (max_test > max_eps)
        return false;
    else
        return true;
}

// todo: use getPower(Tensor4Dconst
// todo: use matrix multiplication
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

inline Eigen::VectorXd Solver::getPowerNormVector()
{
    auto nb_groups = m_macrolib.getNbGroups();
    auto dim = m_macrolib.getDim();
    auto dim_z = std::get<2>(dim), dim_y = std::get<1>(dim), dim_x = std::get<0>(dim);
    auto dim_xyz = dim_z * dim_y * dim_x;
    Eigen::VectorXd norm_vector = Eigen::VectorXd::Zero(nb_groups * dim_xyz);

    for (int i{0}; i < nb_groups; ++i)
    {
        norm_vector.segment(i * dim_xyz, dim_xyz) = m_macrolib.getValuesArray(i + 1, "SIGF") * m_macrolib.getValuesArray(i + 1, "EFISS");
    }

    return norm_vector;
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

inline void Solver::normPhi()
{
    int nb_ev = static_cast<int>(m_eigen_vectors.size());
    for (int i{0}; i < nb_ev; ++i)
    {
        double factor = m_eigen_vectors[i].dot(m_eigen_vectors[i]);
        m_eigen_vectors[i] = m_eigen_vectors[i] / factor;
    }
    m_is_normed = true;
    m_norm_method = "Phi";
}

inline void Solver::normVector(Eigen::VectorXd vector, double value)
{
    int nb_ev = static_cast<int>(m_eigen_vectors.size());
    for (int i{0}; i < nb_ev; ++i)
    {
        double factor = vector.dot(m_eigen_vectors[i]) / value;
        m_eigen_vectors[i] = m_eigen_vectors[i] / factor;
    }
    m_is_normed = true;
    m_norm_method = "Vector";
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

// specialization
template <>
inline void Solver::initInnerSolver(Eigen::SparseLU<SpMat> &inner_solver, double tol_inner, int inner_max_iter){};

template <>
inline void Solver::initInnerSolver(Eigen::SimplicialLLT<SpMat> &inner_solver, double tol_inner, int inner_max_iter){};

template <>
inline void Solver::initInnerSolver(Eigen::SimplicialLDLT<SpMat> &inner_solver, double tol_inner, int inner_max_iter){};

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
    return x;
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
    return x;
}

// specialization
template <>
inline Eigen::VectorXd Solver::solveInner(Eigen::SparseLU<SpMat> &inner_solver, Eigen::VectorXd &b, Eigen::VectorBlock<Eigen::VectorXd> &x)
{
    x = inner_solver.solve(b);
    return x;
}

// specialization
template <>
inline Eigen::VectorXd Solver::solveInner(Eigen::SparseLU<SpMat> &inner_solver, Eigen::VectorXd &b, Eigen::VectorXd &x)
{
    x = inner_solver.solve(b);
    return x;
}

// specialization
template <>
inline Eigen::VectorXd Solver::solveInner(Eigen::SimplicialLLT<SpMat> &inner_solver, Eigen::VectorXd &b, Eigen::VectorBlock<Eigen::VectorXd> &x)
{
    x = inner_solver.solve(b);
    return x;
}

// specialization
template <>
inline Eigen::VectorXd Solver::solveInner(Eigen::SimplicialLLT<SpMat> &inner_solver, Eigen::VectorXd &b, Eigen::VectorXd &x)
{
    x = inner_solver.solve(b);
    return x;
}

// specialization
template <>
inline Eigen::VectorXd Solver::solveInner(Eigen::SimplicialLDLT<SpMat> &inner_solver, Eigen::VectorXd &b, Eigen::VectorBlock<Eigen::VectorXd> &x)
{
    x = inner_solver.solve(b);
    return x;
}

// specialization
template <>
inline Eigen::VectorXd Solver::solveInner(Eigen::SimplicialLDLT<SpMat> &inner_solver, Eigen::VectorXd &b, Eigen::VectorXd &x)
{
    x = inner_solver.solve(b);
    return x;
}

inline void Solver::dump(std::string file_name, std::string suffix)
{
    H5Easy::File file(file_name, H5Easy::File::OpenOrCreate);

    H5Easy::dump(file, "/eigenvectors" + suffix, m_eigen_vectors, H5Easy::DumpMode::Overwrite);
    H5Easy::dump(file, "/eigenvalues" + suffix, m_eigen_values, H5Easy::DumpMode::Overwrite);
}

inline void Solver::load(std::string file_name, std::string suffix)
{
    H5Easy::File file(file_name, H5Easy::File::ReadOnly);

    m_eigen_vectors = H5Easy::load<vecvec>(file, "/eigenvectors" + suffix);
    m_eigen_values = H5Easy::load<vecd>(file, "/eigenvalues" + suffix);
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
                                     double tol_inner, int outer_max_iter, int inner_max_iter, std::string inner_solver, std::string inner_precond,
                                     std::string acceleration)
{
    spdlog::debug("Inner solver : {}", inner_solver);
    spdlog::debug("Inner precond : {}", inner_precond);
    if (acceleration == "chebyshev" && inner_solver == "SparseLU")
        SolverFullPowerIt::solveChebyshev<Eigen::SparseLU<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                  tol_inner, outer_max_iter, inner_max_iter);
    else if (acceleration == "chebyshev" && inner_solver == "LeastSquaresConjugateGradient" && inner_precond.empty())
        SolverFullPowerIt::solveChebyshev<Eigen::LeastSquaresConjugateGradient<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                                       tol_inner, outer_max_iter, inner_max_iter);
    else if (acceleration == "chebyshev" && inner_solver == "BiCGSTAB" && inner_precond.empty())
        SolverFullPowerIt::solveChebyshev<Eigen::BiCGSTAB<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                  tol_inner, outer_max_iter, inner_max_iter);
    else if (acceleration == "chebyshev" && inner_solver == "GMRES" && inner_precond.empty())
        SolverFullPowerIt::solveChebyshev<Eigen::GMRES<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                               tol_inner, outer_max_iter, inner_max_iter);
    else if (acceleration == "chebyshev" && inner_solver == "GMRES" && inner_precond == "IncompleteLUT")
        SolverFullPowerIt::solveChebyshev<Eigen::GMRES<SpMat, Eigen::IncompleteLUT<double>>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                                             tol_inner, outer_max_iter, inner_max_iter);
    else if (acceleration == "chebyshev" && inner_solver == "DGMRES" && inner_precond.empty())
        SolverFullPowerIt::solveChebyshev<Eigen::DGMRES<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                tol_inner, outer_max_iter, inner_max_iter);
    else if (acceleration == "chebyshev" && inner_solver == "DGMRES" && inner_precond == "IncompleteLUT")
        SolverFullPowerIt::solveChebyshev<Eigen::DGMRES<SpMat, Eigen::IncompleteLUT<double>>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                                              tol_inner, outer_max_iter, inner_max_iter);
    else if (acceleration == "chebyshev" && inner_solver == "BiCGSTAB" && inner_precond == "IncompleteLUT")
        SolverFullPowerIt::solveChebyshev<Eigen::BiCGSTAB<SpMat, Eigen::IncompleteLUT<double>>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                                                tol_inner, outer_max_iter, inner_max_iter);
    else if (acceleration == "chebyshev" && inner_solver == "IDRS" && inner_precond.empty())
        SolverFullPowerIt::solveChebyshev<Eigen::IDRS<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                              tol_inner, outer_max_iter, inner_max_iter);
    else if (acceleration == "chebyshev" && inner_solver == "IDRS" && inner_precond == "IncompleteLUT")
        SolverFullPowerIt::solveChebyshev<Eigen::IDRS<SpMat, Eigen::IncompleteLUT<double>>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                                            tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "SparseLU")
        SolverFullPowerIt::solveUnaccelerated<Eigen::SparseLU<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                      tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "LeastSquaresConjugateGradient" && inner_precond.empty())
        SolverFullPowerIt::solveUnaccelerated<Eigen::LeastSquaresConjugateGradient<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                                           tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "BiCGSTAB" && inner_precond.empty())
        SolverFullPowerIt::solveUnaccelerated<Eigen::BiCGSTAB<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                      tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "GMRES" && inner_precond.empty())
        SolverFullPowerIt::solveUnaccelerated<Eigen::GMRES<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                   tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "BiCGSTAB" && inner_precond == "IncompleteLUT")
        SolverFullPowerIt::solveUnaccelerated<Eigen::BiCGSTAB<SpMat, Eigen::IncompleteLUT<double>>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                                                    tol_inner, outer_max_iter, inner_max_iter);
    else
        throw std::invalid_argument("The combinaison of acceleration, inner_solver and inner_precond is not known");
}

template <class T>
void SolverFullPowerIt::solveUnaccelerated(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0, double ev0,
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

        eigen_value = v_prec.dot(v); //  v_norm_prec == 1 // may be we can just use the norm and only the raleyh coeff at the end

        r_tol_ev = ((v.array() / v_prec.array()).maxCoeff() - (v.array() / v_prec.array()).minCoeff()) / (2 * eigen_value);

        v /= v.norm();

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

template <class T>
void SolverFullPowerIt::solveChebyshev(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0, double ev0,
                                       double tol_inner, int outer_max_iter, int inner_max_iter)
{

    int pblm_dim = static_cast<int>(m_M.rows());
    int v0_size = static_cast<int>(v0.size());

    float r_tol = 1e5;
    float r_tol_ev = 1e5;
    float r_tol_ev_prec = 1e5;
    float r_tol_ev_prec_acc = 1e5;
    float r_tol_ev2 = 1e5;

    int nb_free_iter = 4;
    int nb_chebyshev_cycle_min_iter = 4;

    Eigen::VectorXd v(v0);

    if (v0_size == 0)
        v.setConstant(pblm_dim, 1.);
    else if (v0_size != pblm_dim)
        throw std::invalid_argument("The size of the initial vector must be identical to the matrix row or column size!");
    else
        v /= v.norm();

    Eigen::VectorXd v_prec(v);
    Eigen::VectorXd v_prec_prec(v);

    if (nb_eigen_values != 1)
        throw std::invalid_argument("Only one eigen value can be computed with PI!");

    spdlog::debug("Tolerance in outter iteration (eigen value): {:.2e}", tol);
    spdlog::debug("Tolerance in outter iteration (eigen vector): {:.2e}", tol_eigen_vectors);
    spdlog::debug("Tolerance in inner iteration : {:.2e}", tol_inner);
    spdlog::debug("Max. outer iteration : {}", outer_max_iter);
    spdlog::debug("Max. inner iteration : {}", inner_max_iter);

    double eigen_value = ev0;
    double eigen_value_prec = eigen_value;
    double dominance_ratio = 0.4;

    double alpha;
    double beta;

    T solver;
    initInnerSolver<T>(solver, tol_inner, inner_max_iter);
    solver.compute(m_M);

    // outer iteration
    int i = 1; // total
    int p = 1; // acc.
    int n = 0; // outer
    while ((r_tol > tol || r_tol_ev > tol_eigen_vectors || r_tol_ev2 > tol_eigen_vectors) && i < outer_max_iter)
    {
        spdlog::debug("----------------------------------------------------");
        Eigen::VectorXd b = m_K * v / eigen_value;
        // inner iteration
        v = solveInner<T>(solver, b, v);

        // eigen_value *= v.lpNorm<1>() / v_prec.lpNorm<1>();
        eigen_value /= v.dot(m_K * v_prec) / v.dot(m_K * v);

        // convergence computation
        r_tol_ev = ((v.array() / v_prec.array()).maxCoeff() - (v.array() / v_prec.array()).minCoeff());
        r_tol_ev2 = (v - v_prec).norm() / std::sqrt(v.dot(v_prec));
        r_tol = std::abs(eigen_value - eigen_value_prec);

        // r_tol_ev2 = abs((v - v_prec).maxCoeff()) / eigen_value ;

        // free iteration
        if (i <= nb_free_iter || dominance_ratio < 0.5 || dominance_ratio > 1 || std::isnan(dominance_ratio))
        {
            spdlog::debug("Free outer iteration {}", i);
            r_tol_ev_prec_acc = r_tol_ev;
            dominance_ratio = r_tol_ev / r_tol_ev_prec;
            spdlog::debug("Dominance ratio estimation = {}", dominance_ratio);
            n++;
        }
        // chebyshev acc.
        else
        {
            spdlog::debug("Cheb outer iteration {} (cycle {}/{})", i, p, nb_chebyshev_cycle_min_iter);
            if (p == 1) // first cycle
            {
                alpha = 2 / (2 - dominance_ratio);
                beta = 0;
            }
            else
            {
                double gamma = std::acosh(2 / dominance_ratio - 1);
                spdlog::debug("gamma = {}", gamma);
                alpha = (4 / dominance_ratio) * std::cosh((p - 1) * gamma) / std::cosh(p * gamma);
                beta = std::cosh((p - 2) * gamma) / std::cosh(p * gamma);
                // beta = (1 - dominance_ratio / 2.) - 1 / alpha; //(1 - dominance_ratio / 2.) * alpha - 1;
            }
            spdlog::debug("alpha = {}", alpha);
            spdlog::debug("beta = {}", beta);

            v = v_prec + alpha * (v - v_prec) + beta * (v_prec - v_prec_prec);
            p++;
        }

        if (p >= (nb_chebyshev_cycle_min_iter + 1)) // possible end of the acceleration cycle
        {
            double error_ratio = r_tol_ev / r_tol_ev_prec_acc;
            double cp = std::cosh(p * std::acosh(2 / dominance_ratio - 1));

            if (error_ratio < 1 / cp)
            {
                spdlog::debug("The dominance ratio does not have to be modifed, we continue the acceleration cycle");
                nb_chebyshev_cycle_min_iter++;
            }
            else
            {
                spdlog::debug("We end the acceleration cycle");
                if (error_ratio < 1)
                    dominance_ratio *= (std::cosh(std::acosh(cp * error_ratio) / p) + 1) / 2.;
                else
                {
                    spdlog::warn("error_ratio > 1, there is probably an error, we multiply the dominance ratio by 0.95!");
                    dominance_ratio *= 0.95;
                }
                p = 1;
                n++;
                r_tol_ev_prec_acc = r_tol_ev;
            }

            // limitation of the rate of growth of the dominance ratio
            if (n <= 6)
                dominance_ratio = std::min(0.9, dominance_ratio);
            else if (n <= 9)
                dominance_ratio = std::min(0.95, dominance_ratio);
            else if (n <= 12)
                dominance_ratio = std::min(0.985, dominance_ratio);
            else if (n > 12)
                dominance_ratio = std::min(0.99, dominance_ratio);

            spdlog::debug("Dominance ratio estimation = {}", dominance_ratio);
        }

        // convergence computation
        r_tol_ev_prec = r_tol_ev;

        eigen_value_prec = eigen_value;
        v_prec_prec = v_prec;
        v_prec = v;
        spdlog::debug("Eigen value = {:.5f}", eigen_value);
        spdlog::debug("Estimated error in outter iteration (eigen value): {:.2e}", r_tol);
        spdlog::debug("Estimated error in outter iteration (eigen vector): {:.2e}", r_tol_ev);
        spdlog::debug("Estimated error in outter iteration (eigen vector 2): {:.2e}", r_tol_ev2);
        i++;
    }
    // eigen_value = // to re estimate

    spdlog::debug("----------------------------------------------------");
    m_eigen_values.clear();
    m_eigen_vectors.clear();
    m_eigen_values.push_back(eigen_value);
    m_eigen_vectors.push_back(v);
    m_dominance_ratio = dominance_ratio;
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
                                   double tol_inner, int outer_max_iter, int inner_max_iter, std::string inner_solver, std::string inner_precond,
                                   std::string acceleration)
{
    SolverFullSlepc::solve(tol, nb_eigen_values, v0, ev0,
                           tol_inner, outer_max_iter, inner_max_iter, "krylovschur", inner_solver, inner_precond);
}

// todo add which eigen values (smallest, largest)
inline void SolverFullSlepc::solve(double tol, int nb_eigen_values, const Eigen::VectorXd &v0, double ev0,
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
        D.push_back(operators::diff_diffusion_op<T, Tensor1D>(g + 1, dx, dy, dz, macrolib,
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
        D.push_back(operators::diff_diffusion_op<T, Tensor1D>(g + 1, dx, dy, macrolib,
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
        D.push_back(operators::diff_diffusion_op<T, Tensor1D>(g + 1, dx, macrolib, albedo_x0, albedo_xn));

    operators::setup_cond_operators<T, Tensor1D>(m_F, m_chi, m_A, m_S,
                                                 D, m_volumes, m_macrolib);
}

template <class T>
const auto SolverCond<T>::getFissionSource(Eigen::VectorXd &eigen_vector)
{

    Eigen::VectorXd psi = Eigen::VectorXd::Zero(m_volumes.size());

    for (int g{0}; g < m_macrolib.getNbGroups(); ++g)
    {
        psi += m_F[g] * getEigenVector(g + 1, eigen_vector);
    }
    return psi;
}

inline void SolverCondPowerIt::solve(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0, double ev0,
                                     double tol_inner, int outer_max_iter, int inner_max_iter, std::string inner_solver, std::string inner_precond,
                                     std::string acceleration)
{
    spdlog::debug("Inner solver : {}", inner_solver);
    spdlog::debug("Inner precond : {}", inner_precond);

    if (inner_solver == "SparseLU")
        SolverCondPowerIt::solveUnaccelerated<Eigen::SparseLU<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                      tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "SimplicialLLT")
        SolverCondPowerIt::solveUnaccelerated<Eigen::SimplicialLLT<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                           tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "SimplicialLDLT")
        SolverCondPowerIt::solveUnaccelerated<Eigen::SimplicialLDLT<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                            tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "ConjugateGradient" && inner_precond.empty())
        SolverCondPowerIt::solveUnaccelerated<Eigen::ConjugateGradient<SpMat, Eigen::Lower|Eigen::Upper>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                               tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "ConjugateGradient" && inner_precond == "IncompleteCholesky")
        SolverCondPowerIt::solveUnaccelerated<Eigen::ConjugateGradient<SpMat, Eigen::Lower|Eigen::Upper, Eigen::IncompleteCholesky<double>>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                                                    tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "LeastSquaresConjugateGradient" && inner_precond.empty())
        SolverCondPowerIt::solveUnaccelerated<Eigen::LeastSquaresConjugateGradient<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                                           tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "BiCGSTAB" && inner_precond.empty())
        SolverCondPowerIt::solveUnaccelerated<Eigen::BiCGSTAB<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                      tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "GMRES" && inner_precond.empty())
        SolverCondPowerIt::solveUnaccelerated<Eigen::GMRES<SpMat>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                   tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "BiCGSTAB" && inner_precond == "IncompleteLUT")
        SolverCondPowerIt::solveUnaccelerated<Eigen::BiCGSTAB<SpMat, Eigen::IncompleteLUT<double>>>(tol, tol_eigen_vectors, nb_eigen_values, v0, ev0,
                                                                                                    tol_inner, outer_max_iter, inner_max_iter);
    else
        throw std::invalid_argument("The combinaison of inner_solver and inner_precond is not known");
}

template <class T>
void SolverCondPowerIt::solveUnaccelerated(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0, double ev0,
                                           double tol_inner, int outer_max_iter, int inner_max_iter)
{

    int pblm_dim = m_macrolib.getNbGroups() * static_cast<int>(m_volumes.size());
    int v0_size = static_cast<int>(v0.size());
    auto dim = m_macrolib.getDim();
    auto dim_xyz = std::get<2>(dim) * std::get<1>(dim) * std::get<0>(dim);

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
            Eigen::VectorBlock<Eigen::VectorXd> phi_g = v(Eigen::seqN(dim_xyz * g, dim_xyz));
            Eigen::VectorXd b = m_chi[g] * psi / eigen_value;
            for (int gp{0}; gp < m_macrolib.getNbGroups(); ++gp)
            {
                if (g <= gp)
                    continue;
                b += m_S[gp][g] * getEigenVector(gp + 1, v);
            }
            spdlog::debug("Inner iteration for group {}", g + 1);
            phi_g = solveInner<T>(solvers[g], b, phi_g);
        }
        psi = eigen_value * getFissionSource(v);

        eigen_value = psi_prec.dot(psi); //  v_norm_prec == 1

        Eigen::ArrayXd psi_div = psi.array() / psi_prec.array();
        r_tol_ev = (psi_div.maxCoeff<Eigen::NaNPropagationOptions::PropagateNumbers>() -
                    psi_div.minCoeff<Eigen::NaNPropagationOptions::PropagateNumbers>()) /
                   (2 * eigen_value);

        psi /= psi.norm();

        // convergence computation
        r_tol = std::abs(eigen_value - eigen_value_prec);
        r_tol_ev2 = (psi - psi_prec).norm() / eigen_value;

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

//-------------------------------------------------------------------------
// SolverFull
//-------------------------------------------------------------------------

inline SolverFullFixedSource::SolverFullFixedSource(const SolverFull<SpMat> &solver, const SolverFull<SpMat> &solver_star, const Eigen::VectorXd &source)
{
    m_volumes = solver.getVolumes();
    m_K = solver.getK();
    m_M = solver.getM();
    m_macrolib = solver.getMacrolib();
    m_eigen_values = solver.getEigenValues();
    m_eigen_vectors = solver.getEigenVectors();
    m_eigen_vectors_star = solver_star.getEigenVectors();
    m_dominance_ratio = solver.getDominanceRatio();
    m_source = source;
}

inline void SolverFullFixedSource::solve(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0, double ev0,
                                         double tol_inner, int outer_max_iter, int inner_max_iter, std::string inner_solver, std::string inner_precond,
                                         std::string acceleration)
{
    spdlog::debug("Inner solver : {}", inner_solver);
    spdlog::debug("Inner precond : {}", inner_precond);
    if (acceleration == "chebyshev" && inner_solver == "SparseLU")
        SolverFullFixedSource::solveChebyshev<Eigen::SparseLU<SpMat>>(tol, v0,
                                                                      tol_inner, outer_max_iter, inner_max_iter);
    else if (acceleration == "chebyshev" && inner_solver == "LeastSquaresConjugateGradient" && inner_precond.empty())
        SolverFullFixedSource::solveChebyshev<Eigen::LeastSquaresConjugateGradient<SpMat>>(tol, v0,
                                                                                           tol_inner, outer_max_iter, inner_max_iter);
    else if (acceleration == "chebyshev" && inner_solver == "BiCGSTAB" && inner_precond.empty())
        SolverFullFixedSource::solveChebyshev<Eigen::BiCGSTAB<SpMat>>(tol, v0,
                                                                      tol_inner, outer_max_iter, inner_max_iter);
    else if (acceleration == "chebyshev" && inner_solver == "GMRES" && inner_precond.empty())
        SolverFullFixedSource::solveChebyshev<Eigen::GMRES<SpMat>>(tol, v0,
                                                                   tol_inner, outer_max_iter, inner_max_iter);
    else if (acceleration == "chebyshev" && inner_solver == "BiCGSTAB" && inner_precond == "IncompleteLUT")
        SolverFullFixedSource::solveChebyshev<Eigen::BiCGSTAB<SpMat, Eigen::IncompleteLUT<double>>>(tol, v0,
                                                                                                    tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "SparseLU")
        SolverFullFixedSource::solveUnaccelerated<Eigen::SparseLU<SpMat>>(tol, v0,
                                                                          tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "LeastSquaresConjugateGradient" && inner_precond.empty())
        SolverFullFixedSource::solveUnaccelerated<Eigen::LeastSquaresConjugateGradient<SpMat>>(tol, v0,
                                                                                               tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "BiCGSTAB" && inner_precond.empty())
        SolverFullFixedSource::solveUnaccelerated<Eigen::BiCGSTAB<SpMat>>(tol, v0,
                                                                          tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "GMRES" && inner_precond.empty())
        SolverFullFixedSource::solveUnaccelerated<Eigen::GMRES<SpMat>>(tol, v0,
                                                                       tol_inner, outer_max_iter, inner_max_iter);
    else if (inner_solver == "BiCGSTAB" && inner_precond == "IncompleteLUT")
        SolverFullFixedSource::solveUnaccelerated<Eigen::BiCGSTAB<SpMat, Eigen::IncompleteLUT<double>>>(tol, v0,
                                                                                                        tol_inner, outer_max_iter, inner_max_iter);
    else
        throw std::invalid_argument("The combinaison of inner_solver and inner_precond is not known");
}

template <class T>
void SolverFullFixedSource::solveUnaccelerated(double tol, const Eigen::VectorXd &v0,
                                               double tol_inner, int outer_max_iter, int inner_max_iter)
{

    int pblm_dim = static_cast<int>(m_M.rows());
    int v0_size = static_cast<int>(v0.size());

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

    spdlog::debug("Tolerance in outter iteration (gamma): {:.2e}", tol);
    spdlog::debug("Tolerance in inner iteration : {:.2e}", tol_inner);
    spdlog::debug("Max. outer iteration : {}", outer_max_iter);
    spdlog::debug("Max. inner iteration : {}", inner_max_iter);

    T solver;
    initInnerSolver<T>(solver, tol_inner, inner_max_iter);
    solver.compute(m_M);

    auto K = m_K;
    if (isAdjoint())
        K = m_K.adjoint();

    double s_star_K_s = (m_eigen_vectors_star[0].dot(K * m_eigen_vectors[0]));
    // double s_star_K_s = (m_eigen_vectors[0].dot(m_eigen_vectors[0])); // also works

    // outer iteration
    int i = 0;
    while ((r_tol_ev > tol || r_tol_ev2 > tol) && i < outer_max_iter)
    {
        spdlog::debug("----------------------------------------------------");
        spdlog::debug("Outer iteration {}", i);
        Eigen::VectorXd b = m_K * v / m_eigen_values[0] + m_source;
        // inner iteration
        v = solveInner<T>(solver, b, v);

        // decontamiation of gamma
        // maybe we can do it only every n step (6 for example in Variational Principles and Convergence Acceleration Strategies for the Neutron Diffusion Equation)
        if (isAdjoint())
            v -= (v.dot(K * m_eigen_vectors[0])) / s_star_K_s * m_eigen_vectors_star[0];
        else
            v -= (m_eigen_vectors_star[0].dot(K * v)) / s_star_K_s * m_eigen_vectors[0];

        // v -= (v.dot(m_eigen_vectors[0])) / s_star_K_s * m_eigen_vectors[0]; //  also works

        // convergence computation
        r_tol_ev = ((v.array() / v_prec.array()).maxCoeff() - (v.array() / v_prec.array()).minCoeff());
        r_tol_ev2 = (v - v_prec).norm();

        v_prec = v;
        spdlog::debug("Estimated error in outter iteration (eigen vector): {:.2e}", r_tol_ev);
        spdlog::debug("Estimated error in outter iteration (eigen vector 2): {:.2e}", r_tol_ev2);
        i++;
    }
    spdlog::debug("----------------------------------------------------");
    m_gamma = v;
    spdlog::debug("Number of outter iteration: {}", i);

    spdlog::debug("Orthogonlity tests: ");
    if (isAdjoint())
    {
        spdlog::debug("Gamma* K Phi : {}", v.dot(K * m_eigen_vectors[0]));
        spdlog::debug("Gamma* Phi* : {}", v.dot(m_eigen_vectors_star[0]));
    }
    else
    {
        spdlog::debug("Gamma K* Phi* : {}", v.dot(m_K * m_eigen_vectors_star[0]));
        spdlog::debug("Gamma Phi : {}", v.dot(m_eigen_vectors[0]));
    }
}

template <class T>
void SolverFullFixedSource::solveChebyshev(double tol, const Eigen::VectorXd &v0,
                                           double tol_inner, int outer_max_iter, int inner_max_iter)
{

    int pblm_dim = static_cast<int>(m_M.rows());
    int v0_size = static_cast<int>(v0.size());

    float r_tol_ev = 1e5;
    float r_tol_ev2_prec = 1e5;
    float r_tol_ev2_prec_acc = 1e5;
    float r_tol_ev2 = 1e5;

    int nb_free_iter = 4;
    int nb_chebyshev_cycle_min_iter = 4;

    Eigen::VectorXd v(v0);

    if (v0_size == 0)
        v.setConstant(pblm_dim, 1.);
    else if (v0_size != pblm_dim)
        throw std::invalid_argument("The size of the initial vector must be identical to the matrix row or column size!");
    else
        v /= v.norm();

    Eigen::VectorXd v_prec(v);
    Eigen::VectorXd v_prec_prec(v);

    spdlog::debug("Tolerance in outter iteration (gamma): {:.2e}", tol);
    spdlog::debug("Tolerance in inner iteration : {:.2e}", tol_inner);
    spdlog::debug("Max. outer iteration : {}", outer_max_iter);
    spdlog::debug("Max. inner iteration : {}", inner_max_iter);

    double alpha;
    double beta;
    double dominance_ratio = 0.4;

    T solver;
    initInnerSolver<T>(solver, tol_inner, inner_max_iter);
    solver.compute(m_M);

    auto K = m_K;
    if (isAdjoint())
        K = m_K.adjoint();

    double s_star_K_s = (m_eigen_vectors_star[0].dot(K * m_eigen_vectors[0]));
    // double s_star_K_s = (m_eigen_vectors[0].dot(m_eigen_vectors[0])); // also works

    // outer iteration
    int i = 1; // total
    int p = 1; // acc.
    int n = 0; // outer

    while (r_tol_ev2 > tol && i < outer_max_iter)
    // while ((r_tol_ev > tol || r_tol_ev2 > tol) && i < outer_max_iter) // in pratice r_tol_ev almost never converge...
    {
        spdlog::debug("----------------------------------------------------");
        spdlog::debug("Outer iteration {}", i);
        Eigen::VectorXd b = m_K * v / m_eigen_values[0] + m_source;
        // inner iteration
        v = solveInner<T>(solver, b, v);

        // convergence computation
        r_tol_ev = ((v.array() / v_prec.array()).maxCoeff() - (v.array() / v_prec.array()).minCoeff());
        r_tol_ev2 = (v - v_prec).norm() / std::sqrt(v.dot(v_prec));

        // free iteration
        if (i <= nb_free_iter || dominance_ratio < 0.5 || dominance_ratio > 1 || std::isnan(dominance_ratio))
        {
            spdlog::debug("Free outer iteration {}", i);
            r_tol_ev2_prec_acc = r_tol_ev2;
            dominance_ratio = r_tol_ev2 / r_tol_ev2_prec;
            spdlog::debug("Dominance ratio estimation = {}", dominance_ratio);
            n++;
        }
        // chebyshev acc.
        else
        {
            spdlog::debug("Cheb outer iteration {} (cycle {}/{})", i, p, nb_chebyshev_cycle_min_iter);
            if (p == 1) // first cycle
            {
                alpha = 2 / (2 - dominance_ratio);
                beta = 0;
            }
            else
            {
                double gamma = std::acosh(2 / dominance_ratio - 1);
                spdlog::debug("gamma = {}", gamma);
                alpha = (4 / dominance_ratio) * std::cosh((p - 1) * gamma) / std::cosh(p * gamma);
                beta = std::cosh((p - 2) * gamma) / std::cosh(p * gamma);
                // beta = (1 - dominance_ratio / 2.) - 1 / alpha; //(1 - dominance_ratio / 2.) * alpha - 1;
            }
            spdlog::debug("alpha = {}", alpha);
            spdlog::debug("beta = {}", beta);

            v = v_prec + alpha * (v - v_prec) + beta * (v_prec - v_prec_prec);
            p++;
        }

        if (p >= (nb_chebyshev_cycle_min_iter + 1)) // possible end of the acceleration cycle
        {
            double error_ratio = r_tol_ev2 / r_tol_ev2_prec_acc;
            double cp = std::cosh(p * std::acosh(2 / dominance_ratio - 1));

            if (error_ratio < 1 / cp)
            {
                spdlog::debug("The dominance ratio does not have to be modifed, we continue the acceleration cycle");
                nb_chebyshev_cycle_min_iter++;
            }
            else
            {
                spdlog::debug("We end the acceleration cycle");
                if (error_ratio < 1)
                    dominance_ratio *= (std::cosh(std::acosh(cp * error_ratio) / p) + 1) / 2.;
                else
                {
                    spdlog::warn("error_ratio >= 1, there is probably an error, we multiply the dominance ratio by 0.95!");
                    dominance_ratio *= 0.95;
                }
                p = 1;
                n++;
                r_tol_ev2_prec_acc = r_tol_ev2;
            }

            // limitation of the rate of growth of the dominance ratio
            if (n <= 6)
                dominance_ratio = std::min(0.9, dominance_ratio);
            else if (n <= 9)
                dominance_ratio = std::min(0.95, dominance_ratio);
            else if (n <= 12)
                dominance_ratio = std::min(0.985, dominance_ratio);
            else if (n > 12)
                dominance_ratio = std::min(0.99, dominance_ratio);

            spdlog::debug("Dominance ratio estimation = {}", dominance_ratio);
        }

        // decontamiation of gamma
        // maybe we can do it only every n step (6 for example in Variational Principles and Convergence Acceleration Strategies for the Neutron Diffusion Equation)
        if (isAdjoint() && (i % 6))
            v -= (v.dot(K * m_eigen_vectors[0])) / s_star_K_s * m_eigen_vectors_star[0];
        else if (i % 6)
            v -= (m_eigen_vectors_star[0].dot(K * v)) / s_star_K_s * m_eigen_vectors[0];
        // v -= (v.dot(m_eigen_vectors[0])) / s_star_K_s * m_eigen_vectors[0]; //  also works

        r_tol_ev2_prec = r_tol_ev2;
        v_prec_prec = v_prec;
        v_prec = v;

        spdlog::debug("Estimated error in outter iteration (eigen vector): {:.2e}", r_tol_ev);
        spdlog::debug("Estimated error in outter iteration (eigen vector 2): {:.2e}", r_tol_ev2);
        i++;
    }
    spdlog::debug("----------------------------------------------------");
    m_gamma = v;
    spdlog::debug("Number of outter iteration: {}", i);

    // decontamiation of gamma
    // maybe we can do it only every n step (6 for example in Variational Principles and Convergence Acceleration Strategies for the Neutron Diffusion Equation)
    if (isAdjoint())
        v -= (v.dot(K * m_eigen_vectors[0])) / s_star_K_s * m_eigen_vectors_star[0];
    else
        v -= (m_eigen_vectors_star[0].dot(K * v)) / s_star_K_s * m_eigen_vectors[0];

    spdlog::debug("Orthogonlity tests: ");
    if (isAdjoint())
    {
        spdlog::debug("Gamma* K Phi : {}", v.dot(K * m_eigen_vectors[0]) / v.dot(v));
        spdlog::debug("Gamma* Phi* : {}", v.dot(m_eigen_vectors_star[0]) / v.dot(v));
    }
    else
    {
        spdlog::debug("Gamma K* Phi* : {}", v.dot(m_K * m_eigen_vectors_star[0]) / v.dot(v));
        spdlog::debug("Gamma Phi : {}", v.dot(m_eigen_vectors[0]) / v.dot(v));
    }
}