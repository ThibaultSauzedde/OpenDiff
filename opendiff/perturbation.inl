inline vecd randomCoordPerturbations(vecd coord, std::default_random_engine &generator,
                                     std::normal_distribution<double> &pert_value_distribution)
{
    if (coord.empty())
        return coord;

    for (int i{1}; i < static_cast<int>(coord.size()) - 1; ++i)
    {
        coord[i] *= pert_value_distribution(generator);
    }

    return coord;
}

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
                spdlog::debug("M-orthogonality test failed for {}, {}: {:.2e}", i, j, test);
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
        throw std::invalid_argument("The eigenvectors direct and adjoint basis are not M-orthogonals!");
    else if (max_test > max_eps)
        return false;
    else
        return true;
}

template <typename T>
void handleDegeneratedEigenvalues(T &solver, T &solver_star, double max_eps)
{
    auto K = solver.getK();
    auto M = solver.getM();
    auto eigen_values = solver.getEigenValues();
    auto eigen_values_star = solver_star.getEigenValues();
    auto eigen_vectors = solver.getEigenVectors();
    auto eigen_vectors_star = solver_star.getEigenVectors();

    // assert that the eigenvalues are identicals for direct and ajoint problems
    int vsize = static_cast<int>(std::min(eigen_values.size(), eigen_values_star.size()));
    for (auto i = 0; i < vsize; ++i)
    {
        if (std::abs(eigen_values[i] - eigen_values_star[i]) > max_eps)
        {
            spdlog::debug("Eigenvalues test failed for {}, {}: {:.2e}", i, eigen_values[i], eigen_values_star[i]);
            throw std::invalid_argument("The eigenv are not identicals for the direct and ajoint problems!");
        }
    }

    //
    // get degenerated eigenvalues
    //
    std::vector<int> ids_deg{};
    std::vector<std::vector<int>> ids_group{};
    ids_group.push_back({});
    for (auto i = 1; i < vsize; ++i)
    {
        if (std::abs(eigen_values[i] - eigen_values[i - 1]) < max_eps)
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

    //
    // handle each group of degenerated eigen values (if the eigenvectors are not orthogonal)
    //
    for (auto ids : ids_group)
    {
        if (ids.empty())
            continue;

        int ids_size = static_cast<int>(ids.size());
        
        //
        // Make the eigenvector basis orthogonal 
        //

        // calc the coeff
        std::vector<double> a_ii{};
        a_ii.push_back(eigen_vectors[ids[0]].dot(eigen_vectors[ids[0]])); // a00

        spdlog::info("New orthogonalisation with a group of {} eigenvectors from {} to {}", ids_size, ids[0], ids.back());
        spdlog::debug("We keep the vector {} with eigenvalue: {:.5f}", ids[0], eigen_values[ids[0]]);

        // get the new eigenvectors
        for (auto ev_i = 1; ev_i < ids_size; ++ev_i) // all ev except the first one (we keep it without modif)
        {
            spdlog::debug("Orthogonalisation of eigenvector {} with eigenvalue: {:.5f}", ids[ev_i], eigen_values[ids[ev_i]]);
            for (auto k_i = 0; k_i < ev_i; ++k_i) // substract the unwanted part of the vector
            {
                auto ain = eigen_vectors[ids[k_i]].dot(eigen_vectors[ids.back()]);
                auto coeff = -ain / a_ii[k_i];
                spdlog::debug("Coeff for k_{} = {:.5f} = - {:.5e} / {:.5e}", k_i, coeff, ain, a_ii[k_i]);
                eigen_vectors[ids[ev_i]] += coeff * eigen_vectors[ids[k_i]]; // why does it works??? 
            }
            // append the new norm coeff
            a_ii.push_back(eigen_vectors[ids[ev_i]].dot(eigen_vectors[ids[ev_i]]));
        }

        //
        // Make the eigenvector basis and adjoint eigenvector basis M-orthogonal 
        //

        // calc the coeff
        a_ii.clear();
        a_ii.push_back(eigen_vectors_star[ids[0]].dot(M * eigen_vectors[ids[0]])); // a00

        spdlog::info("New M orthogonalisation with a group of {} eigenvectors from {} to {}", ids_size, ids[0], ids.back());
        spdlog::debug("We keep the adjoint eigenvector {} with eigenvalue: {:.5f}", ids[0], eigen_values[ids[0]]);

        // get the new eigenvectors
        for (auto ev_i = 1; ev_i < ids_size; ++ev_i) // all ev except the first one (we keep it without modif)
        {
            spdlog::debug("M orthogonalisation of adjoint eigenvector {} with eigenvalue: {:.5f}", ids[ev_i], eigen_values[ids[ev_i]]);
            for (auto k_i = 0; k_i < ev_i; ++k_i) // substract the unwanted part of the vector
            {
                auto ain = eigen_vectors_star[ids[k_i]].dot(M * eigen_vectors[ids.back()]);
                auto coeff = -ain / a_ii[k_i];
                spdlog::debug("Coeff for k_{} = {:.5f} = - {:.5e} / {:.5e}", k_i, coeff, ain, a_ii[k_i]);
                eigen_vectors_star[ids[ev_i]] += coeff * eigen_vectors_star[ids[k_i]]; // why does it works??? 
            }
            // append the new norm coeff
            a_ii.push_back(eigen_vectors_star[ids[ev_i]].dot(M * eigen_vectors[ids[ev_i]]));
        }
    }
}


//
// Modal Expansion
//


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
    else if (norm_method == "power") 
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
            // auto coeff = ((eigen_values[0] - eigen_values[i]) * eigen_vectors_star[i].dot(M * eigen_vectors[i]));
            auto coeff = (eigen_vectors_star[i].dot((eigen_values[0] * M - K) * eigen_vectors[i])); // more general when we complete the basis
            auto a_i = eigen_vectors_star[i].dot(delta_L_ev) / coeff;
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

template <class T>
std::tuple<Eigen::VectorXd, double, Eigen::VectorXd> exactPerturbation(T &solver, T &solver_star, T &solver_pert, double tol_eigen_value, int max_iter, int basis_size)
{
    if (!(solver.isNormed() && solver.getNormMethod() == "power"))
        solver.norm("power", solver_star);

    auto K = solver.getK();
    auto M = solver.getM();
    auto K_pert = solver_pert.getK();
    auto M_pert = solver_pert.getM();
    auto eigen_values = solver.getEigenValues();
    auto eigen_vectors = solver.getEigenVectors();
    auto eigen_vectors_star = solver_star.getEigenVectors();
    Eigen::VectorXd norm_ref = solver.getPowerNormVector();
    Eigen::VectorXd norm_pert = solver_pert.getPowerNormVector();
    double power = norm_ref.dot(eigen_vectors[0]);

    auto delta_M = (M_pert - M);
    auto delta_L = ((K_pert - K) - delta_M * eigen_values[0]);

    auto A = eigen_vectors_star[0].dot(delta_L * eigen_vectors[0]);
    auto B = eigen_vectors_star[0].dot(M_pert * eigen_vectors[0]);

    auto delta_eval = A / B;
    auto delta_eval_prec = delta_eval;
    spdlog::debug("Estimated error in iteration 0 (delta eigen value = {:.5e})  ({} {})", delta_eval, max_iter, basis_size);
    int basis_size_real = static_cast<int>(eigen_vectors_star.size());
    if (basis_size <= 0 || basis_size > basis_size_real)
        basis_size = basis_size_real;

    Eigen::VectorXd c1(basis_size);
    Eigen::VectorXd c2(basis_size);
    Eigen::VectorXd d1(basis_size);
    Eigen::VectorXd d2(basis_size);
    Eigen::VectorXd power_pert(basis_size);

    Eigen::MatrixXd C1(basis_size, basis_size);
    Eigen::MatrixXd C2(basis_size, basis_size);

    Eigen::MatrixXd I_minus_c_inv(basis_size, basis_size); // I


    // eigen value calculation
    for (auto i{0}; i < basis_size; ++i)
    {
        // test if i = 0
        double norm = (eigen_values[0] - eigen_values[i]) * eigen_vectors_star[i].dot(M * eigen_vectors[i]);
        if (i ==0)
        {
            d1(i) = 0; // a_0
            d2(i) = 0; // a_0
        }
        else
        {
            d1(i) = eigen_vectors_star[i].dot(delta_L * eigen_vectors[0]) / norm;
            d2(i) = eigen_vectors_star[i].dot(M_pert * eigen_vectors[0]) / norm;
        }

        c1(i) = eigen_vectors_star[0].dot(delta_L * eigen_vectors[i]);
        c2(i) = eigen_vectors_star[0].dot(M_pert * eigen_vectors[i]);
        power_pert(i) = norm_pert.dot(eigen_vectors[i]);

        for (auto j{0}; j < basis_size; ++j)
        {
            if (i ==0)
            {
                C1(i, j) = 0;
                C2(i, j) = 0;
            }
            else
            {
                C1(i, j) = eigen_vectors_star[i].dot(delta_L * eigen_vectors[j]) / norm;
                C2(i, j) = eigen_vectors_star[i].dot(M_pert * eigen_vectors[j]) / norm;
            }
        }
    }

    float r_tol_ev = 1e5;
    int k = 0;
    Eigen::VectorXd beta = d1;
    while (r_tol_ev > tol_eigen_value && k < max_iter)
    {
        // a_0 calculattion
        d1(0) = power - power_pert(0);
        for (auto i{1}; i < basis_size; ++i)
            d1(0) -= power_pert(i) * beta[i];
        d1(0) /= power_pert(0);

        I_minus_c_inv = (Eigen::MatrixXd::Identity(basis_size, basis_size) - (C1 - delta_eval * C2)).inverse();
        // beta
        beta = I_minus_c_inv * (d1 - delta_eval * d2);
        double C_k = (c1 - delta_eval * c2).transpose() * beta;
        delta_eval = (A + C_k) / B;
        r_tol_ev = std::abs(delta_eval - delta_eval_prec);
        delta_eval_prec = delta_eval;

        spdlog::debug("Estimated error in iteration {} (delta eigen value = {:.5e}): {:.2e}", k, delta_eval, r_tol_ev);
        spdlog::debug("Ck = {:.5e}", C_k);
        k++;
    }

    // a_0 calculattion
    // d1(0) = power - power_pert(0);
    // for (auto i{1}; i < basis_size; ++i)
    //     d1(0) -= power_pert(i) * beta[i];
    // d1(0) /= power_pert(0);
    
    // flux recons
    Eigen::VectorXd ev_recons = eigen_vectors[0]; // copy
    for (auto k{0}; k < basis_size; ++k)
        ev_recons += beta(k) * eigen_vectors[k];

    auto eval_recons = eigen_values[0] + delta_eval;

    solver_pert.clearEigenValues();
    solver_pert.pushEigenValue(eval_recons);
    solver_pert.pushEigenVector(ev_recons);

    // beta(0) -= 1. ; 
    return std::make_tuple(ev_recons, eval_recons, beta);
}

//
// Modal expansion * range finding
//

// template <typename T, typename F>
// vecvec createAdjointBasis(T &solver, int ev_basis_size, vecvec basis, std::vector<F> basis_solvers,
//                           double tol, double tol_eigen_vectors, double tol_inner, int outer_max_iter, int inner_max_iter,
//                           std::string inner_solver, std::string inner_precond, std::string acceleration)
// {
//     auto M = solver.getM();

//     // get the direct basis
//     auto eigen_vectors = solver.getEigenVectors();
//     int ev_basis_size_real = static_cast<int>(eigen_vectors.size());
//     if (ev_basis_size <= 0 || ev_basis_size > ev_basis_size_real)
//         ev_basis_size = ev_basis_size_real;

//     vecvec basis_full{};
//     solver.normPhi(); // norm to one every eigenvector
//     for (int i{0}; i < ev_basis_size; ++i)
//         basis_full.push_back(eigen_vectors[i]);
//     int rf_basis_size = static_cast<int>(basis.size());
//     for (int i{0}; i < rf_basis_size; ++i)
//     {
//         basis[i] /= basis[i].norm(); // probably not necesssary
//         basis_full.push_back(basis[i]);
//     }
//     spdlog::debug("Direct basis created");


//     vecvec adjoint_basis{};
//     for (auto solver_i : basis_solvers)
//     {
//         F solver_i_star = F(solver_i);
//         solver_i_star.makeAdjoint();
//         Eigen::VectorXd v0{};
//         solver_i_star.solve(tol, tol_eigen_vectors, 1, v0, solver.getEigenValues()[0],
//                             tol_inner, outer_max_iter, inner_max_iter,
//                             inner_solver, inner_precond, acceleration);
//         Eigen::VectorXd eigen_vector_star_i = solver_i_star.getEigenVectors()[0];
//         spdlog::debug("Adjoint problem solved");

//         // Gram–Schmidt_process
//         for (int k{0}; k < static_cast<int>(basis_full.size()); ++k)
//             eigen_vector_star_i -= eigen_vector_star_i.dot(M * basis_full[k]) * basis_full[k];

//         // 2nd Gram–Schmidt_process
//         for (int k{0}; k < static_cast<int>(basis_full.size()); ++k)
//             eigen_vector_star_i -= eigen_vector_star_i.dot(M * basis_full[k]) * basis_full[k];

//         eigen_vector_star_i /= eigen_vector_star_i.norm();
//         adjoint_basis.push_back(eigen_vector_star_i);
//     }

//     return adjoint_basis;
// }

template <typename T>
std::tuple<Eigen::VectorXd, double, Eigen::VectorXd> firstOrderPerturbationEpGPT(T &solver, T &solver_star,
                                                                                 T &solver_pert, int ev_basis_size,
                                                                                 vecvec basis, vecvec gamma_star, std::vector<double> N_star)
{
    if (!(solver.isNormed() && solver.getNormMethod() == "power"))
        solver.norm("power", solver_star);

    auto K = solver.getK();
    auto M = solver.getM();
    auto K_pert = solver_pert.getK();
    auto M_pert = solver_pert.getM();
    auto eigen_values = solver.getEigenValues();
    auto eigen_vectors = solver.getEigenVectors();
    auto eigen_vectors_star = solver_star.getEigenVectors();

    Eigen::VectorXd norm_vector = solver.getPowerNormVector();
    Eigen::VectorXd norm_vector_pert = solver_pert.getPowerNormVector();
    double power = norm_vector.dot(eigen_vectors[0]);

    auto delta_M = (M_pert - M);
    auto delta_L = ((K_pert - K) - delta_M * eigen_values[0]);

    int ev_basis_size_real = static_cast<int>(eigen_vectors_star.size());
    if (ev_basis_size <= 0 || ev_basis_size > ev_basis_size_real)
        ev_basis_size = ev_basis_size_real;

    int rf_basis_size = static_cast<int>(basis.size());
    int basis_size = ev_basis_size + rf_basis_size;

    Eigen::VectorXd beta(basis_size);
    Eigen::VectorXd power_pert(basis_size);

   
    for (auto k{0}; k < rf_basis_size; ++k)
    {
        auto a_k = firstOrderGPT(solver, solver_star, solver_pert,
                                 basis[k], basis[k],
                                 norm_vector, norm_vector_pert,
                                 N_star[k], gamma_star[k]);

        beta(ev_basis_size + k) = a_k;
        power_pert(ev_basis_size + k) = norm_vector_pert.dot(basis[k]);
    }

    for (auto i{0}; i < ev_basis_size; ++i)
    {
        double norm = (eigen_values[0] - eigen_values[i]) * eigen_vectors_star[i].dot(M * eigen_vectors[i]);
        if (i == 0)
            beta(i) = 0; // a_0
        else
        {
            beta(i) = eigen_vectors_star[i].dot(delta_L * eigen_vectors[0]) / norm;
            for (auto k{0}; k < rf_basis_size; ++k)
                beta(i) -= beta(ev_basis_size + k) * eigen_vectors_star[i].dot(K * basis[k]) / norm;
        }

        power_pert(i) = norm_vector_pert.dot(eigen_vectors[i]);
    }

    // a_0 calculation
    beta(0) = power - power_pert(0);
    for (auto i{1}; i < basis_size; ++i)
        beta(0) -= power_pert(i) * beta[i];
    beta(0) /= power_pert(0);

    // flux recons
    Eigen::VectorXd ev_recons = eigen_vectors[0]; // copy
    for (auto k{0}; k < ev_basis_size; ++k)
        ev_recons += beta(k) * eigen_vectors[k];
    for (auto k{0}; k < rf_basis_size; ++k)
        ev_recons += beta(ev_basis_size + k) * basis[k];

    auto A = eigen_vectors_star[0].dot(delta_L * eigen_vectors[0]);
    auto B = eigen_vectors_star[0].dot(M * eigen_vectors[0]);

    auto delta_eval = A / B;
    for (auto k{0}; k < rf_basis_size; ++k)
        delta_eval -= beta(ev_basis_size + k) * eigen_vectors_star[0].dot(K * basis[k]) / B;

    auto eval_recons = eigen_values[0] + delta_eval;

    solver_pert.clearEigenValues();
    solver_pert.pushEigenValue(eval_recons);
    solver_pert.pushEigenVector(ev_recons);

    beta(0) -= 1.;
    return std::make_tuple(ev_recons, eval_recons, beta);
}

template <typename T>
std::tuple<Eigen::VectorXd, double, Eigen::VectorXd> firstOrderPerturbationRangeFinding(T &solver, T &solver_pert, vecvec basis, vecvec adjoint_basis)
/*This algorithm is not stable ! 
*/
{

    auto K = solver.getK();
    auto M = solver.getM();
    auto K_pert = solver_pert.getK();
    auto M_pert = solver_pert.getM();
    auto eigen_values = solver.getEigenValues();

    int basis_size = static_cast<int>(basis.size());

    Eigen::VectorXd norm = solver.getPowerNormVector();
    Eigen::VectorXd norm_pert = solver_pert.getPowerNormVector();
    double power = norm.dot(basis[0]);

    auto delta_M = (M_pert - M);
    auto delta_L = ((K_pert - K) - delta_M * eigen_values[0]);

    Eigen::MatrixXd A(basis_size, basis_size);
    Eigen::VectorXd b(basis_size);

    for (auto j{0}; j < basis_size; ++j)
        A(0, j) = norm_pert.dot(basis[j]);

    b(0) = power - A(0, 0);
    for (auto i{1}; i < basis_size; ++i)
        b(i) = adjoint_basis[i].dot(delta_L * basis[0]);

    for (auto i{0}; i < basis_size; ++i)
    {
        for (auto j{0}; j < basis_size; ++j)
        {
            if (i == j)
                A(i, j) = adjoint_basis[i].dot((eigen_values[0] * M - K) * basis[i]);
            else
                A(i, j) = -adjoint_basis[i].dot(K * basis[j]);
        }     
    }

    Eigen::VectorXd beta = A.fullPivLu().solve(b);

    // eigenvalue update
    double C_k = 0;
    for (auto i{1}; i < basis_size; ++i)
        C_k += beta(i) * adjoint_basis[0].dot(K * basis[i]);

    spdlog::debug("Ck = {:.5e}) ", C_k);

    double delta_eval = (adjoint_basis[0].dot(delta_L * basis[0]) + C_k) / (adjoint_basis[0].dot(M * basis[0]));
    spdlog::debug("delta_eval = {:.5e}) ", delta_eval);

    // flux recons
    Eigen::VectorXd ev_recons = basis[0]; // copy
    for (auto k{0}; k < basis_size; ++k)
        ev_recons += beta(k) * basis[k];


    auto eval_recons = eigen_values[0] + delta_eval;

    solver_pert.clearEigenValues();
    solver_pert.pushEigenValue(eval_recons);
    solver_pert.pushEigenVector(ev_recons);

    // beta(0) -= 1.;
    return std::make_tuple(ev_recons, eval_recons, beta);
}

//
// Generalized Perturbation Theory
//

template <typename T, typename F>
std::tuple<double, Eigen::VectorXd> GPTAdjointImportance(const T &solver, const T &solver_star, Eigen::VectorXd &response, Eigen::VectorXd &norm,
                                                         double tol, double tol_inner, int outer_max_iter, int inner_max_iter,
                                                         std::string inner_solver, std::string inner_precond, std::string acceleration)
{
    auto eigen_vector = solver.getEigenVectors()[0];
    double N_star = response.dot(eigen_vector) / norm.dot(eigen_vector);
    Eigen::VectorXd source = response - N_star * norm;
    auto solver_fixed_source_star = F(solver, solver_star, source);
    solver_fixed_source_star.makeAdjoint();
    auto v0 = Eigen::VectorXd();
    solver_fixed_source_star.solve(tol, tol, 1, v0, 1.,
                                   tol_inner, outer_max_iter, inner_max_iter,
                                   inner_solver, inner_precond, acceleration);

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
    Eigen::VectorXd delta_norm = norm_pert - norm;

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

template <typename T, typename F>
std::tuple<double, Eigen::VectorXd> firstOrderGPT(const T &solver, const T &solver_star,
                                                  const T &solver_pert,
                                                  Eigen::VectorXd &response, Eigen::VectorXd &response_pert,
                                                  Eigen::VectorXd &norm, Eigen::VectorXd &norm_pert,
                                                  double tol, double tol_inner, int outer_max_iter, int inner_max_iter,
                                                  std::string inner_solver, std::string inner_precond, std::string acceleration)
{

    auto [N_star, gamma_star] = GPTAdjointImportance<T, F>(solver, solver_star, response, norm,
                                                     tol, tol_inner, outer_max_iter, inner_max_iter,
                                                     inner_solver, inner_precond, acceleration);
    auto pert = firstOrderGPT(solver, solver_star, solver_pert,
                              response, response_pert,
                              norm, norm_pert,
                              N_star, gamma_star);

    return std::make_tuple(pert, gamma_star);
}

//
// EpGPT
//


template <class T, typename F>
EpGPT<T, F>::EpGPT(vecd &x, vecd &y, vecd &z, mat::Middles &middles, const std::vector<std::vector<std::vector<std::string>>> &geometry,
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

    m_solver_star = T(m_solver);
    m_solver_star.makeAdjoint();
}

template <class T, typename F>
EpGPT<T, F>::EpGPT(vecd &x, vecd &y, mat::Middles &middles, const std::vector<std::vector<std::vector<std::string>>> &geometry,
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

    m_solver_star = T(m_solver);
    m_solver_star.makeAdjoint();
}

template <class T, typename F>
EpGPT<T, F>::EpGPT(vecd &x, mat::Middles &middles, const std::vector<std::vector<std::vector<std::string>>> &geometry,
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

    m_solver_star = T(m_solver);
    m_solver_star.makeAdjoint();
}

template <class T, typename F>
void EpGPT<T, F>::clearBasis()
{
    m_basis.clear();
    m_gamma_star.clear();
    m_N_star.clear();
}

template <class T, typename F>
void EpGPT<T, F>::solveReference(double tol, double tol_eigen_vectors, const Eigen::VectorXd &v0, double ev0,
                              double tol_inner, int outer_max_iter, int inner_max_iter, std::string inner_solver,
                              std::string inner_precond, std::string acceleration)
{
    m_solver.solve(tol, tol_eigen_vectors, 1, v0, ev0,
                   tol_inner, outer_max_iter, inner_max_iter,
                   inner_solver, inner_precond, acceleration);

    m_solver_star.solve(tol, tol_eigen_vectors, 1, v0, m_solver.getEigenValues()[0],
                        tol_inner, outer_max_iter, inner_max_iter,
                        inner_solver, inner_precond, acceleration);
}

// template <class T>
// void EpGPT<T, F>::createBasis(double precision, std::vector<std::string> reactions, double pert_value_max, double middles_distribution_p,
//                            double power_W, double tol, double tol_eigen_vectors, const Eigen::VectorXd &v0, double ev0,
//                            double tol_inner, int outer_max_iter, int inner_max_iter, std::string inner_solver,
//                            std::string inner_precond, std::string acceleration)
// {
//     int nb_trial = 10;
//     int nb_trial_succed = 0;
//     double r_precision = 1e5;
//     double r_precision_max_trial = 0.;
//     double r_precision_theory = 1e5;

//     m_solver.normPower(power_W);
//     m_norm_vector = m_solver.getPowerNormVector();
//     // m_solver.normVector(m_norm_vector, power_W);

//     // complete the basis
//     std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
//     std::geometric_distribution<int> middles_distribution(middles_distribution_p); // 0.3-0.6 is ok
//     std::uniform_int_distribution<int> grp_distribution(0, m_middles.getNbGroups() - 1);
//     std::uniform_real_distribution<double> pert_value_distribution(-pert_value_max, +pert_value_max);
//     while ((r_precision > precision) || (r_precision_theory > precision))
//     {
//         // perturbeb problem
//         mat::Middles middles_pert = mat::Middles(m_middles);
//         middles_pert.randomPerturbation(reactions, generator, middles_distribution,
//                                         grp_distribution, pert_value_distribution);
//         mat::Macrolib macrolib_pert = mat::Macrolib(middles_pert, m_geometry);
//         T solver_i{};
//         if (m_y.empty())
//             solver_i = T(m_x, macrolib_pert,
//                          m_albedos[0], m_albedos[1]);
//         else if (m_z.empty())
//             solver_i = T(m_x, m_y, macrolib_pert,
//                          m_albedos[0], m_albedos[1],
//                          m_albedos[2], m_albedos[3]);
//         else
//             solver_i = T(m_x, m_y, m_z, macrolib_pert,
//                          m_albedos[0], m_albedos[1],
//                          m_albedos[2], m_albedos[3],
//                          m_albedos[4], m_albedos[5]);
//         Eigen::VectorXd v0 = m_solver.getEigenVectors()[0];
//         solver_i.solve(tol, tol_eigen_vectors, 1, v0, m_solver.getEigenValues()[0],
//                        tol_inner, outer_max_iter, inner_max_iter,
//                        inner_solver, inner_precond, acceleration);
//         solver_i.normPower(power_W);

//         Eigen::VectorXd delta_ev = solver_i.getEigenVectors()[0] - m_solver.getEigenVectors()[0];
//         spdlog::info("Delta eigenvalue {},", 1e5 * (m_solver.getEigenValues()[0] - solver_i.getEigenValues()[0]) / (m_solver.getEigenValues()[0] * solver_i.getEigenValues()[0]));

//         // basis tests
//         Eigen::VectorXd delta_ev_recons(m_solver.getEigenVectors()[0].size());
//         delta_ev_recons.setZero();
//         for (auto k{0}; k < static_cast<int>(m_basis.size()); ++k)
//         {
//             double coeff = m_basis[k].dot(delta_ev);
//             delta_ev_recons += coeff * m_basis[k];
//         }
//         r_precision = (delta_ev - delta_ev_recons).norm() / delta_ev.norm();

//         spdlog::info("The reconstruction precision is {:.2e} with a basis size {}", r_precision, m_basis.size());

//         // Gram–Schmidt_process
//         if (10 * std::sqrt(2 / M_PI) * r_precision > precision)
//         {
//             Eigen::VectorXd u_i = delta_ev;
//             for (int k{0}; k < static_cast<int>(m_basis.size()); ++k)
//                 u_i -= m_basis[k].dot(u_i) * m_basis[k];

//             u_i /= u_i.norm();
//             m_basis.push_back(u_i);

//             // handle theory precision
//             nb_trial_succed = 0.;
//             r_precision_max_trial = 0.;
//         }
//         // handle theory precision
//         else
//         {
//             nb_trial_succed += 1;
//             r_precision_max_trial = std::max(r_precision_max_trial, r_precision);
//         }

//         if (nb_trial_succed >= nb_trial)
//         {
//             r_precision_theory = 10 * std::sqrt(2 / M_PI) * r_precision_max_trial;
//             nb_trial_succed = 0.;
//             r_precision_max_trial = 0.;
//         }
//     }
// }

template <class T, typename F>
Eigen::VectorXd EpGPT<T, F>::calcSnapshot(std::default_random_engine &generator,
                                          std::normal_distribution<double> &pert_xs_distribution,
                                          std::normal_distribution<double> &pert_x_distribution,
                                          std::normal_distribution<double> &pert_y_distribution,
                                          std::normal_distribution<double> &pert_z_distribution,
                                          vector_tuple control_rod_pos, std::string rod_middle, std::string unroded_middle,
                                          double power_W, double tol, double tol_eigen_vectors, double ev0,
                                          double tol_inner, int outer_max_iter, int inner_max_iter, std::string inner_solver, std::string inner_precond,
                                          std::string acceleration, bool store_solvers)
{
    // perturbeb problem
    mat::Middles middles_pert = mat::Middles(m_middles);
    middles_pert.randomPerturbation(generator, pert_xs_distribution);
    
    geometry_vector geometry_pert{};

    if (control_rod_pos.empty())
        geometry_pert = m_geometry;
    else
       { 
        std::vector<double> control_rod_zpos{};
        for (geometry_tuple rod_description : control_rod_pos)
        {
            std::uniform_real_distribution<double> rod_distribution(std::get<4>(rod_description), std::get<5>(rod_description));
            control_rod_zpos.push_back(rod_distribution(generator));
            spdlog::debug("We modify the rod position at z =  {} for  the rod x ({}, {}) y ({}, {}) z ({}, {})", control_rod_zpos.back(),
             std::get<0>(rod_description), std::get<1>(rod_description), std::get<2>(rod_description), std::get<3>(rod_description),
             std::get<4>(rod_description), std::get<5>(rod_description));
        }
        geometry_pert = mat::get_geometry_roded(m_geometry, m_x, m_y, m_z,
                                           control_rod_pos, rod_middle, unroded_middle,
                                           control_rod_zpos);
        }


    mat::Macrolib macrolib_pert = mat::Macrolib(middles_pert, geometry_pert);
    T solver_i{};

    vecd x = randomCoordPerturbations(m_x, generator, pert_x_distribution);
    vecd y = randomCoordPerturbations(m_y, generator, pert_y_distribution);
    vecd z = randomCoordPerturbations(m_z, generator, pert_z_distribution);

    // for (double i : x)
    //     std::cout << i << ",";
    // std::cout << std::endl;

    if (m_y.empty())
        solver_i = T(x,
                     macrolib_pert,
                     m_albedos[0], m_albedos[1]);
    else if (m_z.empty())
        solver_i = T(x,
                     y,
                     macrolib_pert,
                     m_albedos[0], m_albedos[1],
                     m_albedos[2], m_albedos[3]);
    else
        solver_i = T(x,
                     y,
                     z,
                     macrolib_pert,
                     m_albedos[0], m_albedos[1],
                     m_albedos[2], m_albedos[3],
                     m_albedos[4], m_albedos[5]);

    Eigen::VectorXd v0 = m_solver.getEigenVectors()[0];
    solver_i.solve(tol, tol_eigen_vectors, 1, v0, m_solver.getEigenValues()[0],
                   tol_inner, outer_max_iter, inner_max_iter,
                   inner_solver, inner_precond, acceleration);
    solver_i.normPower(power_W);

    if (store_solvers)
        m_basis_solvers.push_back(solver_i);

    Eigen::VectorXd delta_ev = solver_i.getEigenVectors()[0] - m_solver.getEigenVectors()[0];
    spdlog::info("Snapshot delta eigenvalue {},", 1e5 * (m_solver.getEigenValues()[0] - solver_i.getEigenValues()[0]) / (m_solver.getEigenValues()[0] * solver_i.getEigenValues()[0]));
    return delta_ev;
}

template <class T, typename F>
void EpGPT<T, F>::createBasis(double precision, double pert_xs_sigma,
                              double pert_x_sigma, double pert_y_sigma, double pert_z_sigma,
                              vector_tuple control_rod_pos, std::string rod_middle, std::string unroded_middle,
                              double power_W, double tol, double tol_eigen_vectors, double ev0,
                              double tol_inner, int outer_max_iter, int inner_max_iter, std::string inner_solver,
                              std::string inner_precond, std::string acceleration, bool store_solvers)
{
    const int nb_trial = 10;

    m_solver.normPower(power_W);
    m_norm_vector = m_solver.getPowerNormVector();
    double ev_norm = m_solver.getEigenVectors()[0].norm();
    // m_solver.normVector(m_norm_vector, power_W);

    //if the basis already exits, we norm it
    for (auto k{0}; k < static_cast<int>(m_basis.size()); ++k)
    {
        m_basis[k] = m_basis[k] / m_basis[k].norm();
    }

    // complete the basis
    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::normal_distribution<double> pert_xs_distribution(1.0, pert_xs_sigma / 100.);
    std::normal_distribution<double> pert_x_distribution(1.0, pert_x_sigma / 100.);
    std::normal_distribution<double> pert_y_distribution(1.0, pert_y_sigma / 100.);
    std::normal_distribution<double> pert_z_distribution(1.0, pert_z_sigma / 100.);

    // create the first nb_trial vectors from simulation
    std::array<Eigen::VectorXd, nb_trial> trials;
    std::array<double, nb_trial> trials_norm;
    for (auto i{0}; i < nb_trial; ++i)
    {
        trials[i] = calcSnapshot(generator, pert_xs_distribution,
                                 pert_x_distribution, pert_y_distribution, pert_z_distribution,
                                 control_rod_pos, rod_middle, unroded_middle,
                                 power_W,
                                 tol, tol_eigen_vectors, ev0,
                                 tol_inner, outer_max_iter, inner_max_iter, inner_solver, inner_precond,
                                 acceleration, store_solvers);

        // if the basis is not empty, we remove it in the trials
        for (int k{0}; k < static_cast<int>(m_basis.size()); ++k)
            trials[i] -= m_basis[k].dot(trials[i]) * m_basis[k];

        trials_norm[i] = trials[i].norm() / ev_norm;
    }

    int j = static_cast<int>(m_basis.size()) - 1;
    while (*std::max_element(trials_norm.begin(), trials_norm.end()) > precision / (10 * std::sqrt(2 / M_PI)))
    {
        j++;

        // Gram–Schmidt_process with the current element in trials (2nd time if j > nb_trial)
        int i = j % nb_trial;
        spdlog::info("Iteration {}, we modify the index {} in trials array", j, i);
        for (int k{0}; k < static_cast<int>(m_basis.size()); ++k)
            trials[i] -= m_basis[k].dot(trials[i]) * m_basis[k];

        Eigen::VectorXd u_i = trials[i] / trials[i].norm();
        m_basis.push_back(u_i);

        // Replace the current element by a new snapshot
        trials[i] = calcSnapshot(generator, pert_xs_distribution,
                                 pert_x_distribution, pert_y_distribution, pert_z_distribution,
                                 control_rod_pos, rod_middle, unroded_middle,
                                 power_W,
                                 tol, tol_eigen_vectors, ev0,
                                 tol_inner, outer_max_iter, inner_max_iter, inner_solver, inner_precond,
                                 acceleration, store_solvers);
        
        // Gram–Schmidt_process with the new element
        for (int k{0}; k < static_cast<int>(m_basis.size()); ++k)
            trials[i] -= m_basis[k].dot(trials[i]) * m_basis[k];

        // remove the part of the new vector in the trials
        for (auto k{0}; k < nb_trial; ++k)
        {
            if (k != i)
                trials[k] -= m_basis.back().dot(trials[k]) * m_basis.back();

            trials_norm[k] = trials[k].norm() / ev_norm;
        }

        spdlog::info("Max trials norm = {} must be shorter than {} with a basis of size {}",
                     *std::max_element(trials_norm.begin(), trials_norm.end()),
                     precision / (10 * std::sqrt(2 / M_PI)), m_basis.size());
    }
}

template <class T, typename F>
void EpGPT<T, F>::calcImportances(double tol, const Eigen::VectorXd &v0, double tol_inner,
                                  int outer_max_iter, int inner_max_iter,
                                  std::string inner_solver, std::string inner_precond, std::string acceleration)
{
    for (auto k{0}; k < static_cast<int>(m_basis.size()); ++k)
    {
        // importance calc
        auto [N_star, gamma_star] = GPTAdjointImportance<T, F>(m_solver, m_solver_star, m_basis[k], m_norm_vector,
                                                               tol, tol_inner, outer_max_iter, inner_max_iter,
                                                               inner_solver, inner_precond, acceleration);
        m_gamma_star.push_back(gamma_star);
        m_N_star.push_back(N_star);
        spdlog::warn("Importance {} / {} calculated", k + 1, m_basis.size());
    }
}

template <class T, typename F>
std::tuple<Eigen::VectorXd, double, vecd> EpGPT<T, F>::firstOrderPerturbation(T &solver_pert, int basis_size)
{
    auto K = m_solver.getK();
    auto M = m_solver.getM();
    auto K_pert = solver_pert.getK();
    auto M_pert = solver_pert.getM();
    auto eigen_values = m_solver.getEigenValues();
    auto eigen_vectors = m_solver.getEigenVectors();
    auto eigen_vectors_star = m_solver_star.getEigenVectors();
    // TODO test the size of the vectors
    auto delta_M = (M_pert - M);
    auto delta_L_ev = ((K_pert - K) - delta_M * eigen_values[0]) * eigen_vectors[0];

    Eigen::VectorXd ev_recons = eigen_vectors[0]; // copy
    auto eval_recons = eigen_values[0];
    std::vector<double> a{};
    auto norm_vector_pert = solver_pert.getPowerNormVector();
    Eigen::VectorXd delta_norm = norm_vector_pert - m_norm_vector;

    int basis_size_real = static_cast<int>(m_basis.size());
    if (basis_size <= 0 || basis_size > basis_size_real)
        basis_size = basis_size_real;

    for (auto k{0}; k < basis_size; ++k)
    {
        auto a_k = -m_gamma_star[k].dot(delta_L_ev) - m_N_star[k] * delta_norm.dot(eigen_vectors[0]) ; 
                
        // firstOrderGPT(m_solver, m_solver_star, solver_pert,
        //                          m_basis[k], m_basis[k],
        //                          m_norm_vector, norm_vector_pert,
        //                          m_N_star[k], m_gamma_star[k]);

        a.push_back(a_k);
        ev_recons -= a_k * m_basis[k];
        spdlog::debug("Coefficient {} = {:.5e}", k, a_k);
    }

    eval_recons += eigen_vectors_star[0].dot(delta_L_ev) / (eigen_vectors_star[0].dot(M * eigen_vectors[0]));

    solver_pert.clearEigenValues();
    solver_pert.pushEigenValue(eval_recons);
    solver_pert.pushEigenVector(ev_recons);

    return std::make_tuple(ev_recons, eval_recons, a);
}

template <class T, typename F>
std::tuple<Eigen::VectorXd, double, Eigen::VectorXd> EpGPT<T, F>::highOrderPerturbation(T &solver_pert, double tol_eigen_value, int max_iter, int basis_size)
{
    auto K = m_solver.getK();
    auto M = m_solver.getM();
    auto K_pert = solver_pert.getK();
    auto M_pert = solver_pert.getM();
    auto eigen_values = m_solver.getEigenValues();
    auto eigen_vectors = m_solver.getEigenVectors();
    auto eigen_vectors_star = m_solver_star.getEigenVectors();
    // TODO test the size of the vectors

    auto delta_L = ((K_pert - K) - (M_pert - M) * eigen_values[0]);


    Eigen::VectorXd dL_ev(basis_size);
    Eigen::VectorXd Mp_ev(basis_size);
    dL_ev = delta_L * eigen_vectors[0] ; 
    Mp_ev = M_pert * eigen_vectors[0] ; 

    auto A = eigen_vectors_star[0].dot(dL_ev);
    auto B = eigen_vectors_star[0].dot(Mp_ev);

    auto delta_eval = A / B;
    auto delta_eval_prec = delta_eval;

    auto norm_vector_pert = solver_pert.getPowerNormVector();

    int basis_size_real = static_cast<int>(m_basis.size());
    if (basis_size <= 0 || basis_size > basis_size_real)
        basis_size = basis_size_real;

    Eigen::VectorXd c1(basis_size);
    Eigen::VectorXd c2(basis_size);
    Eigen::VectorXd d1(basis_size);
    Eigen::VectorXd d2(basis_size);

    Eigen::MatrixXd C1(basis_size, basis_size);
    Eigen::MatrixXd C2(basis_size, basis_size);

    // Eigen::MatrixXd I_minus_c_inv(basis_size, basis_size); // I 
    Eigen::VectorXd tmp1(basis_size);
    Eigen::VectorXd tmp2(basis_size);
    // eigen value calculation

    for (auto i{0}; i < basis_size; ++i)
    {
        spdlog::debug("Calculating the matrices {}/{}", i, basis_size);
        tmp1 =  delta_L * m_basis[i] ;
        tmp2 =  M_pert * m_basis[i] ;
        c1(i) = eigen_vectors_star[0].dot(tmp1);
        c2(i) = eigen_vectors_star[0].dot(tmp2);
        d1(i) = m_gamma_star[i].dot(dL_ev) ;//+ m_N_star[i] * (norm_vector_pert-m_norm_vector).dot(eigen_vectors[0]);
        d2(i) = m_gamma_star[i].dot(Mp_ev);

        for (auto j{0}; j < basis_size; ++j)
        {
            C1(j, i) = -m_gamma_star[j].dot(tmp1);
            C2(j, i) = -m_gamma_star[j].dot(tmp2);
        }
    }


    spdlog::debug("The matrices are calculated!");

    float r_tol_ev = 1e5;
    int k = 0;
    Eigen::MatrixXd Identity = Eigen::MatrixXd::Identity(basis_size, basis_size); 
    Eigen::VectorXd beta(basis_size);
    while (r_tol_ev > tol_eigen_value && k < max_iter)
    {
        // todo: résoudre un problème ax=b pour beta ! 
        // I_minus_c_inv = -(Eigen::MatrixXd::Identity(basis_size, basis_size) + (C1 - delta_eval * C2)).inverse();
        // I_minus_c_inv = (-(Identity + (C1 - delta_eval * C2))).partialPivLu().inverse();
        beta = (-(Identity + (C1 - delta_eval * C2))).partialPivLu().solve((d1 - delta_eval * d2));
        // double C_k = -(c1 - delta_eval * c2).transpose() * I_minus_c_inv * (d1 - delta_eval * d2);
        double C_k = -(c1 - delta_eval * c2).dot(beta);
        delta_eval = (A + C_k) / B;
        r_tol_ev = std::abs(delta_eval - delta_eval_prec);
        delta_eval_prec = delta_eval;
        spdlog::debug("Estimated error in iteration {} (delta eigen value = {:.5e}): {:.2e}", k, delta_eval, r_tol_ev);
        spdlog::debug("Ck = {:.5e}", C_k);
        k++;
    }

    // // beta lin calculation 
    // // ça correspond à d1 en fait ! 
    // Eigen::VectorXd beta_lin(basis_size);
    // for (auto k{0}; k < basis_size; ++k)
    // {
    //     beta_lin(k) = firstOrderGPT(m_solver, m_solver_star, solver_pert,
    //                                 m_basis[k], m_basis[k],
    //                                 m_norm_vector, norm_vector_pert,
    //                                 m_N_star[k], m_gamma_star[k]);
    // }

    // beta
    // beta = I_minus_c_inv * (d1 - delta_eval * d2);

    // flux recons
    Eigen::VectorXd ev_recons = eigen_vectors[0]; // copy
    for (auto k{0}; k < basis_size; ++k)
        ev_recons -= beta(k) * m_basis[k];

    auto eval_recons = eigen_values[0] + delta_eval;

    solver_pert.clearEigenValues();
    solver_pert.pushEigenValue(eval_recons);
    solver_pert.pushEigenVector(ev_recons);

    return std::make_tuple(ev_recons, eval_recons, beta);
}

template <class T, typename F>
void EpGPT<T, F>::dump(std::string file_name)
{
    H5Easy::File file(file_name, H5Easy::File::OpenOrCreate);

    H5Easy::dump(file, "/basis", m_basis, H5Easy::DumpMode::Overwrite);
    H5Easy::dump(file, "/gamma_star", m_gamma_star, H5Easy::DumpMode::Overwrite);
    H5Easy::dump(file, "/N_star", m_N_star, H5Easy::DumpMode::Overwrite);
    H5Easy::dump(file, "/norm_vector", m_norm_vector, H5Easy::DumpMode::Overwrite);

    m_solver.dump(file_name);
    m_solver_star.dump(file_name, "_star");
}

template <class T, typename F>
void EpGPT<T, F>::load(std::string file_name)
{
    H5Easy::File file(file_name, H5Easy::File::ReadOnly);

    m_basis = H5Easy::load<vecvec>(file, "/basis");
    m_gamma_star = H5Easy::load<vecvec>(file, "/gamma_star");
    m_N_star = H5Easy::load<std::vector<double>>(file, "/N_star");
    m_norm_vector = H5Easy::load<Eigen::VectorXd>(file, "/norm_vector");

    m_solver.load(file_name);
    m_solver_star.load(file_name, "_star");
}
