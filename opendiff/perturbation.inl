template <typename T>
bool checkBiOrthogonality(T &solver, T &solver_star, double max_eps, bool raise_error, bool remove)
{
    auto eigen_vectors = solver.getEigenVectors();
    auto eigen_vectors_star = solver_star.getEigenVectors();

    // if (eigen_vectors_star.size() != eigen_vectors.size())
    //     throw std::invalid_argument("The number of eigen vectors has the be identical in solver and solver_star!");
    int vsize = static_cast<int>(std::min(eigen_vectors.size(), eigen_vectors_star.size()));

    auto M = solver.getM();

    std::vector<double> all_test{};
    std::vector<int> ids_remove{};

    for (auto i = 0; i < vsize; ++i)
    {
        for (auto j = 0; j < vsize; ++j)
        {
            if (i == j)
                continue;

            double test = eigen_vectors_star[j].dot(M * eigen_vectors[i]);
            all_test.push_back(std::abs(test));
            if (std::abs(test) > max_eps)
            {
                spdlog::debug("Biorthogonality test failed for {}, {}: {:.2e}", i, j, test);
                ids_remove.push_back(std::max(i, j)); // remove max of both
            }
        }
    }

    if (remove)
    {
        solver.removeEigenVectors(ids_remove);
        solver_star.removeEigenVectors(ids_remove); // the same in order to keep the same order
    }

    double max_test = *std::max_element(all_test.begin(), all_test.end());

    spdlog::info("Biorthogonality max test : {:.2e}", max_test);
    if (max_test > max_eps && raise_error)
        throw std::invalid_argument("The eigen vector are not bi-orthogonals!");
    else if (max_test > max_eps)
        return false;
    else
        return true;
}

template <typename T>
void handleDegeneratedEigenvalues(T &solver, T &solver_star, double max_eps)
{
    solver.handleDenegeratedEigenvalues(max_eps);
    solver_star.handleDenegeratedEigenvalues(max_eps);
}

template <typename T>
std::tuple<Eigen::VectorXd, double, vecd> firstOrderPerturbation(T &solver, T &solver_star,
                                                                 T &solver_pert, std::string norm_method)
{
    if (norm_method != "power" && norm_method != "PhiStarMPhi")
        throw std::invalid_argument("Invalid method name!");

    if (!(solver.isNormed() && solver.getNormMethod() == norm_method))
        solver.norm(norm_method, solver_star);

    auto K = solver.getK();
    auto M = solver.getM();
    auto K_pert = solver_pert.getK();
    auto M_pert = solver_pert.getM();
    auto eigen_values = solver.getEigenValues();
    auto eigen_vectors = solver.getEigenVectors();
    auto eigen_vectors_star = solver_star.getEigenVectors();
    auto delta_M = (M_pert - M);
    auto delta_L_ev = ((K_pert - K) - delta_M * eigen_values[0]) * eigen_vectors[0];
    int nb_ev = static_cast<int>(std::min(eigen_vectors.size(), eigen_vectors_star.size()));
    Eigen::VectorXd ev_recons(eigen_vectors[0].size());
    ev_recons.setZero();
    auto eval_recons = eigen_values[0];

    std::vector<double> a{};
    if (norm_method == "PhiStarMPhi")
    {
        // spdlog::debug("a0/2. + 1  = {:.2e}", eigen_vectors_star[0].dot(delta_M * eigen_vectors[0]));
        auto a0 = 1 - eigen_vectors_star[0].dot(delta_M * eigen_vectors[0]) / 2.;
        a.push_back(a0);
        for (int i{1}; i < nb_ev; ++i)
        {
            auto a_i = eigen_vectors_star[i].dot(delta_L_ev) * 1 / (eigen_values[0] - eigen_values[i]);
            a.push_back(a_i);
            ev_recons += a_i * eigen_vectors[i];
        }
        eval_recons += eigen_vectors_star[0].dot(delta_L_ev);
    }
    else if (norm_method == "power") // todo fix it, not working
    {
        // add unpert eigen vector (temporary)
        solver_pert.setEigenVectors(eigen_vectors);

        a.push_back(0);
        auto power = solver.getPower(0);
        Tensor0D power_sum = power.sum();
        auto power_pert = solver_pert.getPower(0);
        Tensor0D power_pert_sum = power_pert.sum();
        double a0 = power_sum(0);

        for (int i{1}; i < nb_ev; ++i)
        {
            auto coeff = eigen_vectors_star[i].dot(M * eigen_vectors[i]);
            auto a_i = eigen_vectors_star[i].dot(delta_L_ev) * 1 / ((eigen_values[0] - eigen_values[i]) * coeff);
            a.push_back(a_i);
            ev_recons += a_i * eigen_vectors[i];
            auto power_pert_i = solver_pert.getPower(i);
            Tensor0D power_pert_i_sum = power_pert_i.sum();
            a0 -= a_i * power_pert_i_sum(0);
        }
        a0 /= power_pert_sum(0);
        a[0] = a0;

        eval_recons += eigen_vectors_star[0].dot(delta_L_ev) / (eigen_vectors_star[0].dot(M * eigen_vectors[0]));
    }
    ev_recons += a[0] * eigen_vectors[0];

    solver_pert.clearEigenValues();
    solver_pert.pushEigenValue(eval_recons);
    solver_pert.pushEigenVector(ev_recons);
    return std::make_tuple(ev_recons, eval_recons, a);
}

template <typename T>
std::tuple<Eigen::VectorXd, double, Tensor2D> highOrderPerturbation(int order, T &solver,
                                                                    T &solver_star, T &solver_pert)
{
    order += 1;
    solver.normPower();

    auto K = solver.getK();
    auto M = solver.getM();
    auto K_pert = solver_pert.getK();
    auto M_pert = solver_pert.getM();
    auto eigen_values = solver.getEigenValues();
    auto eigen_vectors = solver.getEigenVectors();
    auto eigen_vectors_star = solver_star.getEigenVectors();
    auto delta_M = (M_pert - M);
    auto delta_K = (K_pert - K);
    auto delta_L = (delta_K - delta_M * eigen_values[0]);
    int nb_ev = static_cast<int>(std::min(eigen_vectors.size(), eigen_vectors_star.size()));

    Tensor1D eval_recons_list(order);
    eval_recons_list.setZero();

    Tensor2D a(order, nb_ev);
    a.setZero();

    Tensor1D norm(nb_ev);
    norm.setZero();

    Tensor2D ev_star_dm_ev(nb_ev, nb_ev);
    ev_star_dm_ev.setZero();

    Tensor2D ev_star_dl_ev(nb_ev, nb_ev);
    ev_star_dl_ev.setZero();

    // add unpert eigen vector (temporary)
    solver_pert.setEigenVectors(eigen_vectors);

    auto power = solver.getPower(0);
    Tensor0D power_sum = power.sum();
    auto power_pert = solver_pert.getPower(0);
    Tensor0D power_pert_sum = power_pert.sum();
    a(0, 0) = power_sum(0) / power_pert_sum(0);
    eval_recons_list(0) = eigen_values[0];

    // precompute values
    for (int i{0}; i < nb_ev; ++i)
    {
        norm(i) = eigen_vectors_star[i].dot(M * eigen_vectors[i]);
        for (int j{0}; j < nb_ev; ++j)
        {
            ev_star_dm_ev(i, j) = eigen_vectors_star[i].dot(delta_M * eigen_vectors[j]);
            ev_star_dl_ev(i, j) = eigen_vectors_star[i].dot(delta_L * eigen_vectors[j]);
        }
    }

    for (int i{1}; i < order; ++i)
    {
        // eigenvalue calculation
        auto tmp0 = 0.;
        for (int j{0}; j < nb_ev; ++j)
            tmp0 += ev_star_dl_ev(0, j) * a(i - 1, j);

        auto tmp1 = 0.;
        for (int j{0}; j < nb_ev; ++j)
            for (int k{0}; k <= i - 2; ++k)
                tmp1 += eval_recons_list(i - k - 1) * a(k, j) * ev_star_dm_ev(0, j);

        auto tmp2 = 0.;
        for (int k{1}; k <= i - 1; ++k)
            tmp2 += eval_recons_list(i - k) * a(k, 0);
        eval_recons_list(i) = (tmp0 - tmp1 - tmp2 * norm(0)) / (a(0, 0) * norm(0));

        // a coeff calculation
        auto power_pert_high_order = 0.;
        for (int n{1}; n < nb_ev; ++n)
        {
            auto tmpa_0 = 0.;
            for (int j{0}; j < nb_ev; ++j)
                tmpa_0 += ev_star_dl_ev(n, j) * a(i - 1, j);

            auto tmpa_1 = 0.;
            for (int j{0}; j < nb_ev; ++j)
                for (int k{0}; k <= i - 2; ++k)
                    tmpa_1 += eval_recons_list(i - k - 1) * ev_star_dm_ev(n, j) * a(k, j);

            auto tmpa_2 = 0.;
            for (int k{0}; k <= i - 1; ++k)
                tmpa_2 = eval_recons_list(i - k) * a(k, n);

            a(i, n) = (-tmpa_0 + tmpa_1 + tmpa_2 * norm(n)) / ((eigen_values[n] - eigen_values[0]) * norm(n));

            auto power_pert_n = solver_pert.getPower(n);
            Tensor0D power_pert_n_sum = power_pert_n.sum();
            power_pert_high_order += a(i, n) * power_pert_n_sum(0);
        }

        a(i, 0) = -power_pert_high_order / power_pert_sum(0);
    }

    Tensor0D eval_recons_sum = eval_recons_list.sum();
    double eval_recons = eval_recons_sum(0);
    Eigen::VectorXd ev_recons(eigen_vectors[0].size());
    ev_recons.setZero();

    Eigen::array<int, 1> dims({0});
    Tensor1D a_sum = a.sum(dims);
    for (int n{0}; n < nb_ev; ++n)
        ev_recons += a_sum(n) * eigen_vectors[n];

    solver_pert.clearEigenValues();
    solver_pert.pushEigenValue(eval_recons);
    solver_pert.pushEigenVector(ev_recons);

    return std::make_tuple(ev_recons, eval_recons, a);
}

template <typename T>
std::tuple<Eigen::VectorXd, double, py::array_t<double>> highOrderPerturbationPython(int order, T &solver,
                                                                                     T &solver_star, T &solver_pert)
{
    auto [ev_recons, eval_recons, a] = highOrderPerturbation(order, solver, solver_star, solver_pert);
    auto a_python = py::array_t<double, py::array::c_style>({a.dimension(0), a.dimension(1)},
                                                            a.data());
    return std::make_tuple(ev_recons, eval_recons, a_python);
}

template <typename T>
std::tuple<double, Eigen::VectorXd> GPTAdjointImportance(const T &solver, const T &solver_star, Eigen::VectorXd &response, Eigen::VectorXd &norm,
                                                         double tol, double tol_inner, int outer_max_iter, int inner_max_iter, std::string inner_solver, std::string inner_precond)
{
    auto eigen_vector = solver.getEigenVectors()[0];
    double N_star = response.dot(eigen_vector) / norm.dot(eigen_vector);
    Eigen::VectorXd source = response - N_star * norm;
    auto solver_fixed_source_star = solver::SolverFullFixedSource(solver, solver_star, source);
    solver_fixed_source_star.makeAdjoint();
    auto v0 = Eigen::VectorXd();
    solver_fixed_source_star.solve(tol, tol, 1, v0, 1.,
                                   tol_inner, outer_max_iter, inner_max_iter, inner_solver, inner_precond);

    Eigen::VectorXd gamma_star = solver_fixed_source_star.getGamma();

    return std::make_tuple(N_star, gamma_star);
}

template <typename T>
double firstOrderGPT(const T &solver, const T &solver_star, const T &solver_pert,
                     Eigen::VectorXd &response, Eigen::VectorXd &response_pert,
                     Eigen::VectorXd &norm, Eigen::VectorXd &norm_pert,
                     double &N_star, Eigen::VectorXd &gamma_star)
{
    auto K = solver.getK();
    auto M = solver.getM();
    auto K_pert = solver_pert.getK();
    auto M_pert = solver_pert.getM();

    auto eigen_vector = solver.getEigenVectors()[0];
    auto eigen_vector_star = solver_star.getEigenVectors()[0];
    auto eigen_value = solver.getEigenValues()[0];
    auto delta_norm = norm_pert - norm;

    double pert = 0;

    // only if response != response_pert (we use the adress)
    if (std::addressof(response_pert) != std::addressof(response))
        pert += (response_pert - response).dot(eigen_vector);

    spdlog::debug("Delta response (only direct)  = {}", pert);
    pert -= gamma_star.dot(((K_pert - K) - (M_pert - M) * eigen_value) * eigen_vector);
    spdlog::debug("Delta response (direct + indirect)  = {}", pert);
    pert -= N_star * delta_norm.dot(eigen_vector);
    spdlog::debug("Delta response (direct + indirect + norm)  = {}", pert);
    return pert;
}

template <typename T>
std::tuple<double, Eigen::VectorXd> firstOrderGPT(const T &solver, const T &solver_star,
                                                  const T &solver_pert,
                                                  Eigen::VectorXd &response, Eigen::VectorXd &response_pert,
                                                  Eigen::VectorXd &norm, Eigen::VectorXd &norm_pert,
                                                  double tol, double tol_inner, int outer_max_iter, int inner_max_iter, std::string inner_solver, std::string inner_precond)
{

    auto [N_star, gamma_star] = GPTAdjointImportance(solver, solver_star, response, norm,
                                                     tol, tol_inner, outer_max_iter, inner_max_iter, inner_solver, inner_precond);
    auto pert = firstOrderGPT(solver, solver_star, solver_pert,
                              response, response_pert,
                              norm, norm_pert,
                              N_star, gamma_star);

    return std::make_tuple(pert, gamma_star);
}

//
// EpGPT
//
template <class T>
EpGPT<T>::EpGPT(vecd &x, vecd &y, vecd &z, mat::Middles &middles, const std::vector<std::vector<std::vector<std::string>>> &geometry,
                double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn)
{
    m_x = x;
    m_y = y;
    m_z = z;
    m_albedos[0] = albedo_x0;
    m_albedos[1] = albedo_xn;
    m_albedos[2] = albedo_y0;
    m_albedos[3] = albedo_yn;
    m_albedos[4] = albedo_z0;
    m_albedos[5] = albedo_zn;
    m_middles = middles;
    m_geometry = geometry;
    mat::Macrolib macrolib = mat::Macrolib(middles, geometry);
    m_solver = T(x, y, z, macrolib,
                 albedo_x0, albedo_xn,
                 albedo_y0, albedo_yn,
                 albedo_z0, albedo_zn);
}

template <class T>
EpGPT<T>::EpGPT(vecd &x, vecd &y, mat::Middles &middles, const std::vector<std::vector<std::vector<std::string>>> &geometry,
                double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn)
{
    m_x = x;
    m_y = y;
    m_albedos[0] = albedo_x0;
    m_albedos[1] = albedo_xn;
    m_albedos[2] = albedo_y0;
    m_albedos[3] = albedo_yn;
    m_middles = middles;
    m_geometry = geometry;
    mat::Macrolib macrolib = mat::Macrolib(middles, geometry);
    m_solver = T(x, y, macrolib,
                 albedo_x0, albedo_xn,
                 albedo_y0, albedo_yn);
}

template <class T>
EpGPT<T>::EpGPT(vecd &x, mat::Middles &middles, const std::vector<std::vector<std::vector<std::string>>> &geometry,
                double albedo_x0, double albedo_xn)
{
    m_x = x;
    m_albedos[0] = albedo_x0;
    m_albedos[1] = albedo_xn;
    m_middles = middles;
    m_geometry = geometry;
    mat::Macrolib macrolib = mat::Macrolib(middles, geometry);
    m_solver = T(x, macrolib,
                 albedo_x0, albedo_xn);
}

template <class T>
void EpGPT<T>::clearBasis()
{
    m_basis.clear();
}

template <class T>
void EpGPT<T>::createBasis(double precision, std::vector<std::string> reactions, double pert_value_max, double power_W,
                           double tol, double tol_eigen_vectors, const Eigen::VectorXd &v0, double ev0,
                           double tol_inner, int outer_max_iter, int inner_max_iter, std::string inner_solver, std::string inner_precond)
{
    int nb_trial = 10;
    int nb_trial_succed = 0;
    double r_precision = 1e5;
    double r_precision_max_trial = 0.;
    double r_precision_theory = 1e5;
    clearBasis();

    // solve the unperturbeb problem todo: move it in the constructor? or give the solvers and get info from it !!! it is probably better !
    m_solver.solve(tol, tol_eigen_vectors, 1, v0, ev0,
                   tol_inner, outer_max_iter, inner_max_iter, inner_solver, inner_precond);
    m_solver.normPower(power_W);
    m_norm_vector = m_solver.getPowerNormVector();
    // m_solver.normVector(m_norm_vector, power_W);

    // todo :move this elsewhere???
    m_solver_star = T(m_solver);
    m_solver_star.makeAdjoint();
    m_solver_star.solve(tol, tol_eigen_vectors, 1, v0, ev0,
                        tol_inner, outer_max_iter, inner_max_iter, inner_solver, inner_precond);

    // complete the basis
    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::geometric_distribution<int> middles_distribution(0.5);
    std::uniform_int_distribution<int> grp_distribution(0, m_middles.getNbGroups() - 1);
    std::uniform_real_distribution<double> pert_value_distribution(-pert_value_max, +pert_value_max);
    while ((r_precision > precision) || (r_precision_theory > precision))
    {
        // perturbeb problem
        mat::Middles middles_pert = mat::Middles(m_middles);
        middles_pert.randomPerturbation(reactions, generator, middles_distribution,
                                        grp_distribution, pert_value_distribution);
        mat::Macrolib macrolib_pert = mat::Macrolib(middles_pert, m_geometry);
        T solver_i{};
        if (m_y.empty())
            solver_i = T(m_x, macrolib_pert,
                         m_albedos[0], m_albedos[1]);
        else if (m_z.empty())
            solver_i = T(m_x, m_y, macrolib_pert,
                         m_albedos[0], m_albedos[1],
                         m_albedos[2], m_albedos[3]);
        else
            solver_i = T(m_x, m_y, m_z, macrolib_pert,
                         m_albedos[0], m_albedos[1],
                         m_albedos[2], m_albedos[3],
                         m_albedos[4], m_albedos[5]);
        Eigen::VectorXd v0 = m_solver.getEigenVectors()[0];
        solver_i.solve(tol, tol_eigen_vectors, 1, v0, m_solver.getEigenValues()[0],
                       tol_inner, outer_max_iter, inner_max_iter, inner_solver, inner_precond);
        solver_i.normPower(power_W);

        Eigen::VectorXd delta_ev = solver_i.getEigenVectors()[0] - m_solver.getEigenVectors()[0];
        spdlog::info("eigenvalue {},", 1e5 * (m_solver.getEigenValues()[0] - solver_i.getEigenValues()[0]) / (m_solver.getEigenValues()[0] * solver_i.getEigenValues()[0]));

        // basis tests
        Eigen::VectorXd delta_ev_recons(m_solver.getEigenVectors()[0].size());
        delta_ev_recons.setZero();
        for (auto k{0}; k < static_cast<int>(m_basis.size()); ++k)
        {
            double coeff = m_basis[k].dot(delta_ev);
            delta_ev_recons += coeff * m_basis[k];
        }
        r_precision = (delta_ev - delta_ev_recons).norm() / delta_ev.norm();

        spdlog::warn("The reconstruction precision is {:.2e} with a basis size {}", r_precision, m_basis.size());

        // Gram–Schmidt_process
        if (r_precision > precision)
        {
            Eigen::VectorXd u_i = delta_ev;
            for (int k{0}; k < static_cast<int>(m_basis.size()); ++k)
                u_i -= m_basis[k].dot(u_i) * m_basis[k];

            u_i /= u_i.norm();
            m_basis.push_back(u_i);

            // handle theory precision
            nb_trial_succed = 0.;
            r_precision_max_trial = 0.;
        }
        // handle theory precision
        else
        {
            nb_trial_succed += 1;
            r_precision_max_trial = std::max(r_precision_max_trial, r_precision);
        }

        if (nb_trial_succed >= nb_trial)
        {
            r_precision_theory = 10 * std::sqrt(2 / M_PI) * r_precision_max_trial;
            nb_trial_succed = 0.;
            r_precision_max_trial = 0.;
        }
    }
}

template <class T>
void EpGPT<T>::calcImportances(double tol, const Eigen::VectorXd &v0, double tol_inner,
                               int outer_max_iter, int inner_max_iter, std::string inner_solver)
{
    for (auto k{0}; k < static_cast<int>(m_basis.size()); ++k)
    {
        // importance calc
        auto [N_star, gamma_star] = GPTAdjointImportance(m_solver, m_solver_star, m_basis[k], m_norm_vector,
                                                         tol, tol_inner, outer_max_iter, inner_max_iter, inner_solver, "");
        m_gamma_star.push_back(gamma_star);
        m_N_star.push_back(N_star);
        spdlog::warn("Importance {} / {} calculated", k + 1, m_basis.size());
    }
}

template <class T>
std::tuple<Eigen::VectorXd, double, vecd> EpGPT<T>::firstOrderPerturbation(T &solver_pert)
{
    auto K = m_solver.getK();
    auto M = m_solver.getM();
    auto K_pert = solver_pert.getK();
    auto M_pert = solver_pert.getM();
    auto eigen_values = m_solver.getEigenValues();
    auto eigen_vectors = m_solver.getEigenVectors();
    auto eigen_vectors_star = m_solver_star.getEigenVectors();
    auto delta_M = (M_pert - M);
    auto delta_L_ev = ((K_pert - K) - delta_M * eigen_values[0]) * eigen_vectors[0];

    Eigen::VectorXd ev_recons = eigen_vectors[0]; // copy
    auto eval_recons = eigen_values[0];
    std::vector<double> a{};

    auto norm_vector_pert = solver_pert.getPowerNormVector();
    for (auto k{0}; k < static_cast<int>(m_basis.size()); ++k)
    {
        auto a_k = firstOrderGPT(m_solver, m_solver_star, solver_pert,
                                 m_basis[k], m_basis[k],
                                 m_norm_vector, norm_vector_pert,
                                 m_N_star[k], m_gamma_star[k]);

        a.push_back(a_k);
        ev_recons -= a_k * m_basis[k];
    }

    eval_recons += eigen_vectors_star[0].dot(delta_L_ev) / (eigen_vectors_star[0].dot(M * eigen_vectors[0]));

    solver_pert.clearEigenValues();
    solver_pert.pushEigenValue(eval_recons);
    solver_pert.pushEigenVector(ev_recons);

    return std::make_tuple(ev_recons, eval_recons, a);
}

template <class T>
void EpGPT<T>::dump(std::string file_name)
{
    H5Easy::File file(file_name, H5Easy::File::Overwrite);

    H5Easy::dump(file, "/basis", m_basis);
    H5Easy::dump(file, "/gamma_star", m_gamma_star);
    H5Easy::dump(file, "/N_star", m_N_star);
}

template <class T>
void EpGPT<T>::load(std::string file_name)
{
    H5Easy::File file(file_name, H5Easy::File::ReadOnly);

    m_basis = H5Easy::load<vecvec>(file, "/basis");
    m_gamma_star = H5Easy::load<vecvec>(file, "/gamma_star");
    m_N_star = H5Easy::load<std::vector<double>>(file, "/N_star");
}
