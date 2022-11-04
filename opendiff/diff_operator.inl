//
// template for creating the matrix content in triplet for one nrj group
//

// removal op
template <typename V>
void diff_removal_op_triplet(std::vector<Triplet> &coefficients, const int i_grp, V &volumes_1d, mat::Macrolib &macrolib,
                             int offset_i, int offset_j)
{
    if (i_grp < 1 || i_grp > macrolib.getNbGroups())
        throw std::invalid_argument("The wanted nrj group (" + std::to_string(i_grp) + ") is not in the materials");

    int nb_cells = static_cast<int>(volumes_1d.size());

    for (int i{0}; i < nb_cells; ++i)
    {
        coefficients.push_back(Triplet(offset_i + i, offset_j + i, macrolib.getValues1D(i_grp, "SIGR")[i] * volumes_1d[i]));
    }
}

template <typename V>
std::vector<Triplet> diff_removal_op_triplet(const int i_grp, V &volumes_1d, mat::Macrolib &macrolib,
                                             int offset_i, int offset_j)
{
    int nb_cells = static_cast<int>(volumes_1d.size());
    std::vector<Triplet> coefficients{};
    coefficients.reserve(nb_cells);
    diff_removal_op_triplet(coefficients, i_grp, volumes_1d, macrolib, offset_i, offset_j);
    return coefficients;
}

template <typename V>
std::vector<Triplet> diff_removal_op_triplet(V &volumes_1d, mat::Macrolib &macrolib)
{
    auto nb_groups = macrolib.getNbGroups();
    int nb_cells = static_cast<int>(volumes_1d.size());
    std::vector<Triplet> coefficients{};
    coefficients.reserve(nb_cells * nb_groups);
    for (int g{0}; g < nb_groups; ++g)
        diff_removal_op_triplet(coefficients, g + 1, volumes_1d, macrolib, g * nb_cells, g * nb_cells);
    return coefficients;
}

// fission op
template <typename V>
void diff_fission_op_triplet(std::vector<Triplet> &coefficients, const int i_grp, const int i_grp_p, V &volumes_1d, mat::Macrolib &macrolib,
                             int offset_i, int offset_j)
{
    if (i_grp < 1 || i_grp > macrolib.getNbGroups())
        throw std::invalid_argument("The wanted nrj group (" + std::to_string(i_grp) + ") is not in the materials");
    if (i_grp_p < 1 || i_grp_p > macrolib.getNbGroups())
        throw std::invalid_argument("The wanted nrj group (" + std::to_string(i_grp_p) + ") is not in the materials");

    int nb_cells = static_cast<int>(volumes_1d.size());
    
    for (int i{0}; i < nb_cells; ++i)
    {
        if (macrolib.getValues1D(i_grp_p, "CHI")[i] < 10 * std::numeric_limits<double>::lowest())
            continue ;
        
        if (macrolib.getValues1D(i_grp, "NU_SIGF")[i] < 10 * std::numeric_limits<double>::lowest())
            continue ;

        auto t = Triplet(offset_i + i, offset_j + i, macrolib.getValues1D(i_grp, "NU_SIGF")[i] * macrolib.getValues1D(i_grp_p, "CHI")[i] * volumes_1d[i]);
        coefficients.push_back(t);
    }
}

template <typename V>
std::vector<Triplet> diff_fission_op_triplet(const int i_grp, const int i_grp_p, V &volumes_1d, mat::Macrolib &macrolib,
                                             int offset_i, int offset_j)
{
    int nb_cells = static_cast<int>(volumes_1d.size());
    std::vector<Triplet> coefficients{};
    coefficients.reserve(nb_cells);
    diff_fission_op_triplet(coefficients, i_grp, i_grp_p, volumes_1d, macrolib, offset_i, offset_j);
    return coefficients;
}

template <typename V>
std::vector<Triplet> diff_fission_op_triplet(V &volumes_1d, mat::Macrolib &macrolib)
{
    auto nb_groups = macrolib.getNbGroups();
    int nb_cells = static_cast<int>(volumes_1d.size());
    std::vector<Triplet> coefficients{};
    coefficients.reserve(nb_groups * nb_groups * nb_cells);
    for (int g{0}; g < nb_groups; ++g)
    {
        for (int gp{0}; gp < nb_groups; ++gp)
            diff_fission_op_triplet(coefficients, g + 1, gp + 1, volumes_1d, macrolib, gp * nb_cells, g * nb_cells);
    }
    return coefficients;
}

template <typename V>
std::vector<Triplet> diff_fission_op_triplet(const int i_grp, V &volumes_1d, mat::Macrolib &macrolib)
{
    if (i_grp < 1 || i_grp > macrolib.getNbGroups())
        throw std::invalid_argument("The wanted nrj group (" + std::to_string(i_grp) + ") is not in the materials");

    int nb_cells = static_cast<int>(volumes_1d.size());
    std::vector<Triplet> coefficients{};
    coefficients.reserve(nb_cells);
    for (int i{0}; i < nb_cells; ++i)
    {
        if (macrolib.getValues1D(i_grp, "NU_SIGF")[i] < 10 * std::numeric_limits<double>::lowest())
            continue ;

        auto t = Triplet(i, i, macrolib.getValues1D(i_grp, "NU_SIGF")[i] * volumes_1d[i]);
        coefficients.push_back(t);
    }
    return coefficients ; 
}

template <typename V>
std::vector<Triplet> diff_fission_spectrum_op_triplet(const int i_grp, V &volumes_1d, mat::Macrolib &macrolib)
{
    if (i_grp < 1 || i_grp > macrolib.getNbGroups())
        throw std::invalid_argument("The wanted nrj group (" + std::to_string(i_grp) + ") is not in the materials");

    int nb_cells = static_cast<int>(volumes_1d.size());
    std::vector<Triplet> coefficients{};
    coefficients.reserve(nb_cells);
    for (int i{0}; i < nb_cells; ++i)
    {
        if (macrolib.getValues1D(i_grp, "CHI")[i] < 10 * std::numeric_limits<double>::lowest())
            continue ;

        auto t = Triplet(i, i, macrolib.getValues1D(i_grp, "CHI")[i]);
        coefficients.push_back(t);
    }
    return coefficients ; 
}

// scatering op
template <typename V>
void diff_scatering_op_triplet(std::vector<Triplet> &coefficients, const int i_grp, const int i_grp_p, V &volumes_1d, mat::Macrolib &macrolib,
                               int offset_i, int offset_j)
{
    if (i_grp < 1 || i_grp > macrolib.getNbGroups())
        throw std::invalid_argument("The wanted nrj group (" + std::to_string(i_grp) + ") is not in the materials");
    if (i_grp_p < 1 || i_grp_p > macrolib.getNbGroups())
        throw std::invalid_argument("The wanted nrj group (" + std::to_string(i_grp_p) + ") is not in the materials");

    int nb_cells = static_cast<int>(volumes_1d.size());

    auto xs = macrolib.getValues1D(i_grp, std::to_string(i_grp_p)) ;

    for (int i{0}; i < nb_cells; ++i)
    {
        if (xs[i] < 10 * std::numeric_limits<double>::lowest())
            continue ;

        auto t = Triplet(offset_i + i, offset_j + i, xs[i] * volumes_1d[i]);
        coefficients.push_back(t);
    }
}

template <typename V>
std::vector<Triplet> diff_scatering_op_triplet(const int i_grp, const int i_grp_p, V &volumes_1d, mat::Macrolib &macrolib,
                                               int offset_i, int offset_j)
{
    int nb_cells = static_cast<int>(volumes_1d.size());
    std::vector<Triplet> coefficients{};
    coefficients.reserve(nb_cells);
    diff_scatering_op_triplet(coefficients, i_grp, i_grp_p, volumes_1d, macrolib, offset_i, offset_j);
    return coefficients;
}

template <typename V>
std::vector<Triplet> diff_scatering_op_triplet(V &volumes_1d, mat::Macrolib &macrolib)
{
    auto nb_groups = macrolib.getNbGroups();
    int nb_cells = static_cast<int>(volumes_1d.size());
    std::vector<Triplet> coefficients{};
    coefficients.reserve(nb_groups * nb_groups * nb_cells);
    for (int g{0}; g < nb_groups; ++g)
    {
        for (int gp{0}; gp < nb_groups; ++gp)
            diff_scatering_op_triplet(coefficients, g + 1, gp + 1, volumes_1d, macrolib, gp * nb_cells, g * nb_cells);
    }
    return coefficients;
}

// diffusion op 1d
template <typename V>
void diff_diffusion_op_triplet(std::vector<Triplet> &coefficients, const int i_grp, V &dx, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn,
                               int offset_i, int offset_j)
{
    if (i_grp < 1 || i_grp > macrolib.getNbGroups())
        throw std::invalid_argument("The wanted nrj group (" + std::to_string(i_grp) + ") is not in the materials");

    int nb_cells = static_cast<int>(dx.size());

    // left and right C
    auto diff_coeff_1d = macrolib.getValues1D(i_grp, "D");

    auto C_x0 = 2 * (diff_coeff_1d(0) * (1 - albedo_x0)) / (4 * diff_coeff_1d(0) * (1 + albedo_x0) + dx[0] * (1 - albedo_x0));
    auto C_xn = 2 * (diff_coeff_1d(nb_cells - 1) * (1 - albedo_xn)) / (4 * diff_coeff_1d(nb_cells - 1) * (1 + albedo_xn) + dx[nb_cells - 1] * (1 - albedo_xn));

    // midle C
    Tensor1D C(nb_cells + 1);
    for (int i{0}; i < nb_cells - 1; ++i)
    {
        C(i + 1) = 2 * (diff_coeff_1d(i) * diff_coeff_1d(i + 1)) /
                   (dx[i + 1] * diff_coeff_1d(i) + dx[i] * diff_coeff_1d(i + 1));
    }
    C(0) = C_x0;
    C(nb_cells) = C_xn;

    // create the coefficients (only diagonals and sub diagonals)
    for (int i{0}; i < nb_cells; ++i)
    {
        // diagonal term
        coefficients.push_back(Triplet(offset_i + i, offset_j + i, -(C(i) + C(i + 1))));

        // sub diagonal terms
        if (i != nb_cells - 1)
        {
            coefficients.push_back(Triplet(offset_i + i, offset_j + i + 1, C(i + 1)));
            coefficients.push_back(Triplet(offset_i + i + 1, offset_j + i, C(i + 1)));
        }
    }
}

template <typename V>
std::vector<Triplet> diff_diffusion_op_triplet(const int i_grp, V &dx, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn,
                                               int offset_i, int offset_j)
{
    int nb_cells = static_cast<int>(dx.size());
    std::vector<Triplet> coefficients{};
    coefficients.reserve(nb_cells + 2 * (nb_cells - 1));
    diff_diffusion_op_triplet(coefficients, i_grp, dx, macrolib, albedo_x0, albedo_xn, offset_i, offset_j);
    return coefficients;
}

template <typename V>
std::vector<Triplet> diff_diffusion_op_triplet(V &dx, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn)
{
    auto nb_groups = macrolib.getNbGroups();
    int nb_cells = static_cast<int>(dx.size());
    std::vector<Triplet> coefficients{};
    coefficients.reserve(nb_groups * (nb_cells + 2 * (nb_cells - 1)));
    for (int g{0}; g < nb_groups; ++g)
        diff_diffusion_op_triplet(coefficients, g + 1, dx, macrolib, albedo_x0, albedo_xn, g * nb_cells, g * nb_cells);
    return coefficients;
}

// diffusion op 2d
template <typename V>
void diff_diffusion_op_triplet(std::vector<Triplet> &coefficients, const int i_grp, V &dx, V &dy, mat::Macrolib &macrolib,
                               double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn,
                               int offset_i, int offset_j)
{
    if (i_grp < 1 || i_grp > macrolib.getNbGroups())
        throw std::invalid_argument("The wanted nrj group (" + std::to_string(i_grp) + ") is not in the materials");

    int dx_size = static_cast<int>(dx.size());
    int dy_size = static_cast<int>(dy.size());

    // left and right C
    auto diff_coeff_2d = macrolib.getValues(i_grp, "D").chip(0, 0); // chip(0,0) remove the null z axis

    // left and right C
    auto C_x0 = 2 * (diff_coeff_2d.chip(0, 1) * (1 - albedo_x0)) /
                (4 * diff_coeff_2d.chip(0, 1) * (1 + albedo_x0) + dx[0] * (1 - albedo_x0));
    auto C_xn = 2 * (diff_coeff_2d.chip(dx_size - 1, 1) * (1 - albedo_xn)) /
                (4 * diff_coeff_2d.chip(dx_size - 1, 1) * (1 + albedo_xn) + dx[dx_size - 1] * (1 - albedo_xn));

    // down and up C
    auto C_y0 = 2 * (diff_coeff_2d.chip(0, 0) * (1 - albedo_y0)) /
                (4 * diff_coeff_2d.chip(0, 0) * (1 + albedo_y0) + dy[0] * (1 - albedo_y0));
    auto C_yn = 2 * (diff_coeff_2d.chip(dy_size - 1, 0) * (1 - albedo_yn)) /
                (4 * diff_coeff_2d.chip(dy_size - 1, 0) * (1 + albedo_yn) + dy[dy_size - 1] * (1 - albedo_yn));

    // midle C
    Tensor2D C_x(dy_size, dx_size + 1);
    for (int i{0}; i < dx_size + 1; ++i)
    {
        for (int j{0}; j < dy_size; ++j)
        {
            Eigen::array<Eigen::Index, 2> offsets = {j, i};
            Eigen::array<Eigen::Index, 2> offsets_m = {j, i - 1};
            Eigen::array<Eigen::Index, 2> extents = {1, 1};

            if (i == 0)
            {
                C_x.chip(j, 0).chip(i, 0) = C_x0.chip(j, 0); // C_x.chip(j, 1).chip(i, 1) == C_x.slice(offsets, extents) but it is lvalue !
            }
            else if (i == dx_size)
            {
                C_x.chip(j, 0).chip(i, 0) = C_xn.chip(j, 0);
            }
            else
            {
                C_x.chip(j, 0).chip(i, 0) = 2 * (diff_coeff_2d.slice(offsets, extents) * diff_coeff_2d.slice(offsets_m, extents)) /
                                            (dx[i] * diff_coeff_2d.slice(offsets_m, extents) + dx[i - 1] * diff_coeff_2d.slice(offsets, extents));
            }
        }
    }

    Tensor2D C_y(dy_size + 1, dx_size);
    for (int i{0}; i < dx_size; ++i)
    {
        for (int j{0}; j < dy_size + 1; ++j)
        {
            Eigen::array<Eigen::Index, 2> offsets = {j, i};
            Eigen::array<Eigen::Index, 2> offsets_m = {j - 1, i};
            Eigen::array<Eigen::Index, 2> extents = {1, 1};
            if (j == 0)
            {
                C_y.chip(j, 0).chip(i, 0) = C_y0.chip(i, 0); // C_y.chip(j, 1).chip(i, 1) == C_y.slice(offsets, extents) bit it is lvalue !
            }
            else if (j == dy_size)
            {
                C_y.chip(j, 0).chip(i, 0) = C_yn.chip(i, 0);
            }
            else
            {
                C_y.chip(j, 0).chip(i, 0) = 2 * (diff_coeff_2d.slice(offsets, extents) * diff_coeff_2d.slice(offsets_m, extents)) /
                                            (dy[j] * diff_coeff_2d.slice(offsets_m, extents) + dy[j - 1] * diff_coeff_2d.slice(offsets, extents));
            }
        }
    }

    // create the coefficients (only diagonals and sub diagonals)
    for (int i{0}; i < dx_size; ++i)
    {
        for (int j{0}; j < dy_size; ++j)
        {
            // diagonal term
            int id = i + j * dx_size;
            coefficients.push_back(Triplet(offset_i + id, offset_j + id,
                                           -dy[j] * (C_x(j, i) + C_x(j, i + 1)) - dx[i] * (C_y(j, i) + C_y(j + 1, i))));

            // +1 sub diagonal terms
            if (i != dx_size - 1)
            {
                coefficients.push_back(Triplet(offset_i + id, offset_j + id + 1,
                                               dy[j] * C_x(j, i + 1)));
                coefficients.push_back(Triplet(offset_i + id + 1, offset_j + id,
                                               dy[j] * C_x(j, i + 1)));
            }

            // dx_size sub diagonal terms
            if (j != dy_size - 1)
            {
                coefficients.push_back(Triplet(offset_i + id, offset_j + id + dx_size,
                                               dx[j] * C_y(j + 1, i)));
                coefficients.push_back(Triplet(offset_i + id + dx_size, offset_j + id,
                                               dx[j] * C_y(j + 1, i)));
            }
        }
    }
}

template <typename V>
std::vector<Triplet> diff_diffusion_op_triplet(const int i_grp, V &dx, V &dy, mat::Macrolib &macrolib,
                                               double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn,
                                               int offset_i, int offset_j)
{
    int dx_size = static_cast<int>(dx.size());
    int dy_size = static_cast<int>(dy.size());
    int nb_cells = dx_size * dy_size;
    std::vector<Triplet> coefficients{};
    coefficients.reserve(nb_cells + 2 * (dx_size - 1) * dy_size + 2 * dx_size * (dy_size - 1));
    diff_diffusion_op_triplet(coefficients, i_grp, dx, dy, macrolib,
                              albedo_x0, albedo_xn, albedo_y0, albedo_yn,
                              offset_i, offset_j);
    return coefficients;
}

template <typename V>
std::vector<Triplet> diff_diffusion_op_triplet(V &dx, V &dy, mat::Macrolib &macrolib,
                                               double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn)
{
    auto nb_groups = macrolib.getNbGroups();
    int dx_size = static_cast<int>(dx.size());
    int dy_size = static_cast<int>(dy.size());
    int nb_cells = dx_size * dy_size;
    std::vector<Triplet> coefficients{};
    coefficients.reserve(nb_groups * (nb_cells + 2 * (dx_size - 1) * dy_size + 2 * dx_size * (dy_size - 1)));
    for (int g{0}; g < nb_groups; ++g)
        diff_diffusion_op_triplet(coefficients, g + 1, dx, dy, macrolib,
                                  albedo_x0, albedo_xn, albedo_y0, albedo_yn,
                                  g * nb_cells, g * nb_cells);
    return coefficients;
}

// diffusion op 3d
template <typename V>
void diff_diffusion_op_triplet(std::vector<Triplet> &coefficients, const int i_grp, V &dx, V &dy, V &dz, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn,
                               double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn,
                               int offset_i, int offset_j)
{
    if (i_grp < 1 || i_grp > macrolib.getNbGroups())
        throw std::invalid_argument("The wanted nrj group (" + std::to_string(i_grp) + ") is not in the materials");

    int dx_size = static_cast<int>(dx.size());
    int dy_size = static_cast<int>(dy.size());
    int dz_size = static_cast<int>(dz.size());

    // left and right C
    auto diff_coeff_3d = macrolib.getValues(i_grp, "D");

    // left and right C
    auto C_x0 = 2 * (diff_coeff_3d.chip(0, 2) * (1 - albedo_x0)) /
                (4 * diff_coeff_3d.chip(0, 2) * (1 + albedo_x0) + dx[0] * (1 - albedo_x0));
    auto C_xn = 2 * (diff_coeff_3d.chip(dx_size - 1, 2) * (1 - albedo_xn)) /
                (4 * diff_coeff_3d.chip(dx_size - 1, 2) * (1 + albedo_xn) + dx[dx_size - 1] * (1 - albedo_xn));

    // down and up C
    auto C_y0 = 2 * (diff_coeff_3d.chip(0, 1) * (1 - albedo_y0)) /
                (4 * diff_coeff_3d.chip(0, 1) * (1 + albedo_y0) + dy[0] * (1 - albedo_y0));
    auto C_yn = 2 * (diff_coeff_3d.chip(dy_size - 1, 1) * (1 - albedo_yn)) /
                (4 * diff_coeff_3d.chip(dy_size - 1, 1) * (1 + albedo_yn) + dy[dy_size - 1] * (1 - albedo_yn));

    // down and up C
    auto C_z0 = 2 * (diff_coeff_3d.chip(0, 0) * (1 - albedo_z0)) /
                (4 * diff_coeff_3d.chip(0, 0) * (1 + albedo_z0) + dz[0] * (1 - albedo_z0));
    auto C_zn = 2 * (diff_coeff_3d.chip(dz_size - 1, 0) * (1 - albedo_zn)) /
                (4 * diff_coeff_3d.chip(dz_size - 1, 0) * (1 + albedo_zn) + dz[dz_size - 1] * (1 - albedo_zn));

    // midle C
    Tensor3D C_x(dz_size, dy_size, dx_size + 1);
    for (int i{0}; i < dx_size + 1; ++i)
    {
        for (int j{0}; j < dy_size; ++j)
        {
            for (int k{0}; k < dz_size; ++k)
            {
                Eigen::array<Eigen::Index, 3> offsets = {k, j, i};
                Eigen::array<Eigen::Index, 3> offsets_m = {k, j, i - 1};
                Eigen::array<Eigen::Index, 3> extents = {1, 1, 1};

                Eigen::array<Eigen::Index, 2> offsets_c = {k, j};
                Eigen::array<Eigen::Index, 2> extents_c = {1, 1};

                if (i == 0)
                {
                    C_x.chip(k, 0).chip(j, 0).chip(i, 0) = C_x0.slice(offsets_c, extents_c); // C_x.chip(j, 1).chip(i, 1) == C_x.slice(offsets, extents) bit it is lvalue !
                }
                else if (i == dx_size)
                {
                    C_x.chip(k, 0).chip(j, 0).chip(i, 0) = C_xn.slice(offsets_c, extents_c);
                }
                else
                {
                    C_x.chip(k, 0).chip(j, 0).chip(i, 0) = 2 * (diff_coeff_3d.slice(offsets, extents) * diff_coeff_3d.slice(offsets_m, extents)) /
                                                           (dx[i] * diff_coeff_3d.slice(offsets_m, extents) + dx[i - 1] * diff_coeff_3d.slice(offsets, extents));
                }
            }
        }
    }

    Tensor3D C_y(dz_size, dy_size + 1, dx_size);
    for (int i{0}; i < dx_size; ++i)
    {
        for (int j{0}; j < dy_size + 1; ++j)
        {
            for (int k{0}; k < dz_size; ++k)
            {
                Eigen::array<Eigen::Index, 3> offsets = {k, j, i};
                Eigen::array<Eigen::Index, 3> offsets_m = {k, j - 1, i};
                Eigen::array<Eigen::Index, 3> extents = {1, 1, 1};

                Eigen::array<Eigen::Index, 2> offsets_c = {k, i};
                Eigen::array<Eigen::Index, 2> extents_c = {1, 1};

                if (j == 0)
                {
                    C_y.chip(k, 0).chip(j, 0).chip(i, 0) = C_y0.slice(offsets_c, extents_c); // C_y.chip(j, 1).chip(i, 1) == C_y.slice(offsets, extents) bit it is lvalue !
                }
                else if (j == dy_size)
                {
                    C_y.chip(k, 0).chip(j, 0).chip(i, 0) = C_yn.slice(offsets_c, extents_c);
                }
                else
                {
                    C_y.chip(k, 0).chip(j, 0).chip(i, 0) = 2 * (diff_coeff_3d.slice(offsets, extents) * diff_coeff_3d.slice(offsets_m, extents)) /
                                                           (dy[j] * diff_coeff_3d.slice(offsets_m, extents) + dy[j - 1] * diff_coeff_3d.slice(offsets, extents));
                }
            }
        }
    }

    Tensor3D C_z(dz_size + 1, dy_size, dx_size);
    for (int i{0}; i < dx_size; ++i)
    {
        for (int j{0}; j < dy_size; ++j)
        {
            for (int k{0}; k < dz_size + 1; ++k)
            {
                Eigen::array<Eigen::Index, 3> offsets = {k, j, i};
                Eigen::array<Eigen::Index, 3> offsets_m = {k - 1, j, i};
                Eigen::array<Eigen::Index, 3> extents = {1, 1, 1};

                Eigen::array<Eigen::Index, 2> offsets_c = {j, i};
                Eigen::array<Eigen::Index, 2> extents_c = {1, 1};

                if (k == 0)
                {
                    C_z.chip(k, 0).chip(j, 0).chip(i, 0) = C_z0.slice(offsets_c, extents_c); // C_z.chip(j, 1).chip(i, 1) == C_z.slice(offsets, extents) bit it is lvalue !
                }
                else if (k == dz_size)
                {
                    C_z.chip(k, 0).chip(j, 0).chip(i, 0) = C_zn.slice(offsets_c, extents_c);
                }
                else
                {
                    C_z.chip(k, 0).chip(j, 0).chip(i, 0) = 2 * (diff_coeff_3d.slice(offsets, extents) * diff_coeff_3d.slice(offsets_m, extents)) /
                                                           (dz[k] * diff_coeff_3d.slice(offsets_m, extents) + dz[k - 1] * diff_coeff_3d.slice(offsets, extents));
                }
            }
        }
    }

    // std::cout << C_x.format(Eigen::TensorIOFormat::Numpy()) << std::endl;
    // std::cout << C_y.format(Eigen::TensorIOFormat::Numpy()) << std::endl;
    // std::cout << C_z.format(Eigen::TensorIOFormat::Numpy()) << std::endl;

    // create the coefficients (only diagonals and sub diagonals)
    for (int i{0}; i < dx_size; ++i)
    {
        for (int j{0}; j < dy_size; ++j)
        {
            for (int k{0}; k < dz_size; ++k)
            {
                // diagonal term
                int id = i + j * dx_size + k * dx_size * dy_size;
                coefficients.push_back(Triplet(offset_i + id, offset_j + id,
                                               -dy[j] * dz[k] * (C_x(k, j, i) + C_x(k, j, i + 1)) -
                                                   dx[i] * dz[k] * (C_y(k, j, i) + C_y(k, j + 1, i)) -
                                                   dx[i] * dy[j] * (C_z(k, j, i) + C_z(k + 1, j, i))));

                // sub diagonal terms
                if (i != dx_size - 1)
                {
                    coefficients.push_back(Triplet(offset_i + id, offset_j + id + 1,
                                                   dy[j] * dz[k] * C_x(k, j, i + 1)));
                    coefficients.push_back(Triplet(offset_i + id + 1, offset_j + id,
                                                   dy[j] * dz[k] * C_x(k, j, i + 1)));
                }

                // dx_size sub diagonal terms
                if (j != dy_size - 1)
                {
                    coefficients.push_back(Triplet(offset_i + id, offset_j + id + dx_size,
                                                   dx[i] * dz[k] * C_y(k, j + 1, i)));
                    coefficients.push_back(Triplet(offset_i + id + dx_size, offset_j + id,
                                                   dx[i] * dz[k] * C_y(k, j + 1, i)));
                }

                // dx_size * dy_size sub diagonal terms
                if (k != dz_size - 1)
                {
                    coefficients.push_back(Triplet(offset_i + id, offset_j + id + dx_size * dy_size,
                                                   dx[i] * dy[j] * C_z(k + 1, j, i)));
                    coefficients.push_back(Triplet(offset_i + id + dx_size * dy_size, offset_j + id,
                                                   dx[i] * dy[j] * C_z(k + 1, j, i)));
                }
            }
        }
    }
}

template <typename V>
std::vector<Triplet> diff_diffusion_op_triplet(const int i_grp, V &dx, V &dy, V &dz, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn,
                                               double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn,
                                               int offset_i, int offset_j)
{
    int dx_size = static_cast<int>(dx.size());
    int dy_size = static_cast<int>(dy.size());
    int dz_size = static_cast<int>(dz.size());
    int nb_cells = dx_size * dy_size * dz_size;
    std::vector<Triplet> coefficients{};
    coefficients.reserve(nb_cells + 2 * (dx_size - 1) * dy_size * dz_size +
                         2 * dx_size * (dy_size - 1) * dz_size + 2 * dx_size * dy_size * (dz_size - 1));
    diff_diffusion_op_triplet(coefficients, i_grp, dx, dy, dz, macrolib, albedo_x0, albedo_xn,
                              albedo_y0, albedo_yn, albedo_z0, albedo_zn,
                              offset_i, offset_j);
    return coefficients;
}

template <typename V>
std::vector<Triplet> diff_diffusion_op_triplet(V &dx, V &dy, V &dz, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn,
                                               double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn)
{
    auto nb_groups = macrolib.getNbGroups();
    int dx_size = static_cast<int>(dx.size());
    int dy_size = static_cast<int>(dy.size());
    int dz_size = static_cast<int>(dz.size());
    int nb_cells = dx_size * dy_size * dz_size;
    std::vector<Triplet> coefficients{};
    coefficients.reserve(nb_groups * (nb_cells + 2 * (dx_size - 1) * dy_size * dz_size +
                                      2 * dx_size * (dy_size - 1) * dz_size + 2 * dx_size * dy_size * (dz_size - 1)));
    for (int g{0}; g < nb_groups; ++g)
        diff_diffusion_op_triplet(coefficients, g + 1, dx, dy, dz, macrolib, albedo_x0, albedo_xn,
                                  albedo_y0, albedo_yn, albedo_z0, albedo_zn,
                                  g * nb_cells, g * nb_cells);
    return coefficients;
}

//
// template operators for one group (for slepc and eigen matrix)
//

template <typename T, typename V>
T diff_removal_op(const int i_grp, V &volumes_1d, mat::Macrolib &macrolib)
{
    auto coefficients = diff_removal_op_triplet(i_grp, volumes_1d, macrolib);
    int matrix_size = static_cast<int>(volumes_1d.size());
    return matrix_from_coeff<T>(coefficients, matrix_size);
}

template <typename T, typename V>
T diff_fission_op(const int i_grp, const int i_grp_p, V &volumes_1d, mat::Macrolib &macrolib)
{
    auto coefficients = diff_fission_op_triplet(i_grp, i_grp_p, volumes_1d, macrolib);
    int matrix_size = static_cast<int>(volumes_1d.size());
    return matrix_from_coeff<T>(coefficients, matrix_size);
}

template <typename T, typename V>
T diff_fission_op(const int i_grp, V &volumes_1d, mat::Macrolib &macrolib)
{
    auto coefficients = diff_fission_op_triplet(i_grp, volumes_1d, macrolib);
    int matrix_size = static_cast<int>(volumes_1d.size());
    return matrix_from_coeff<T>(coefficients, matrix_size);
}

template <typename T, typename V>
T diff_fission_spectrum_op(const int i_grp, V &volumes_1d, mat::Macrolib &macrolib)
{
    auto coefficients = diff_fission_spectrum_op_triplet(i_grp, volumes_1d, macrolib);
    int matrix_size = static_cast<int>(volumes_1d.size());
    return matrix_from_coeff<T>(coefficients, matrix_size);
}

template <typename T, typename V>
T diff_scatering_op(const int i_grp, const int i_grp_p, V &volumes_1d, mat::Macrolib &macrolib)
{
    auto coefficients = diff_scatering_op_triplet(i_grp, i_grp_p, volumes_1d, macrolib);
    int matrix_size = static_cast<int>(volumes_1d.size());
    return matrix_from_coeff<T>(coefficients, matrix_size);
}

template <typename T, typename V>
T diff_diffusion_op(const int i_grp, V &dx, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn)
{
    auto coefficients = diff_diffusion_op_triplet(i_grp, dx, macrolib, albedo_x0, albedo_xn);
    int matrix_size = static_cast<int>(dx.size());
    return matrix_from_coeff<T>(coefficients, matrix_size);
}

template <typename T, typename V>
T diff_diffusion_op(const int i_grp, V &dx, V &dy, mat::Macrolib &macrolib,
                    double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn)
{
    auto coefficients = diff_diffusion_op_triplet(i_grp, dx, dy, macrolib, albedo_x0, albedo_xn, albedo_y0, albedo_yn);
    int matrix_size = static_cast<int>(dx.size() * dy.size());
    return matrix_from_coeff<T>(coefficients, matrix_size);
}

template <typename T, typename V>
T diff_diffusion_op(const int i_grp, V &dx, V &dy, V &dz, mat::Macrolib &macrolib,
                    double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn)
{
    auto coefficients = diff_diffusion_op_triplet(i_grp, dx, dy, dz, macrolib, albedo_x0, albedo_xn,
                                                  albedo_y0, albedo_yn, albedo_z0, albedo_zn);
    int matrix_size = static_cast<int>(dx.size() * dy.size() * dz.size());
    return matrix_from_coeff<T>(coefficients, matrix_size);
}

//
// template full operators (for slepc and eigen matrix)
//

template <typename T, typename V>
T diff_removal_op(V &volumes_1d, mat::Macrolib &macrolib)
{
    auto coefficients = diff_removal_op_triplet(volumes_1d, macrolib);
    int matrix_size = macrolib.getNbGroups() * static_cast<int>(volumes_1d.size());
    return matrix_from_coeff<T>(coefficients, matrix_size);
}

template <typename T, typename V>
T diff_fission_op(V &volumes_1d, mat::Macrolib &macrolib)
{
    auto coefficients = diff_fission_op_triplet(volumes_1d, macrolib);
    int matrix_size = macrolib.getNbGroups() * static_cast<int>(volumes_1d.size());
    return matrix_from_coeff<T>(coefficients, matrix_size);
}

template <typename T, typename V>
T diff_scatering_op(V &volumes_1d, mat::Macrolib &macrolib)
{
    auto coefficients = diff_scatering_op_triplet(volumes_1d, macrolib);
    int matrix_size = macrolib.getNbGroups() * static_cast<int>(volumes_1d.size());
    return matrix_from_coeff<T>(coefficients, matrix_size);
}

template <typename T, typename V>
T diff_diffusion_op(V &dx, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn)
{
    auto coefficients = diff_diffusion_op_triplet(dx, macrolib, albedo_x0, albedo_xn);
    int matrix_size = macrolib.getNbGroups() * static_cast<int>(dx.size());
    return matrix_from_coeff<T>(coefficients, matrix_size);
}

template <typename T, typename V>
T diff_diffusion_op(V &dx, V &dy, mat::Macrolib &macrolib,
                    double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn)
{
    auto coefficients = diff_diffusion_op_triplet(dx, dy, macrolib, albedo_x0, albedo_xn, albedo_y0, albedo_yn);
    int matrix_size = macrolib.getNbGroups() * static_cast<int>(dx.size() * dy.size());
    return matrix_from_coeff<T>(coefficients, matrix_size);
}

template <typename T, typename V>
T diff_diffusion_op(V &dx, V &dy, V &dz, mat::Macrolib &macrolib,
                    double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn)
{
    auto coefficients = diff_diffusion_op_triplet(dx, dy, dz, macrolib, albedo_x0, albedo_xn,
                                                  albedo_y0, albedo_yn, albedo_z0, albedo_zn);
    int matrix_size = macrolib.getNbGroups() * static_cast<int>(dx.size() * dy.size() * dz.size());
    return matrix_from_coeff<T>(coefficients, matrix_size);
}

//
// template for filling the matrix (for slepc and eigen matrix)
//

template <typename T>
T matrix_from_coeff(const std::vector<Triplet> &coefficients, int matrix_size)
{
    T A(matrix_size, matrix_size);
    A.setFromTriplets(coefficients.begin(), coefficients.end());
    return A;
}
template <typename T>
void matrix_from_coeff(T &A, const std::vector<Triplet> &coefficients)
{
    A.setFromTriplets(coefficients.begin(), coefficients.end());
}

// slepc specialisation
template <>
inline void matrix_from_coeff(Mat &A, const std::vector<Triplet> &coefficients)
{
    PetscInt Istart, Iend;
    MatSetFromOptions(A);
    MatSetUp(A);
    MatGetOwnershipRange(A, &Istart, &Iend);
    for (auto t : coefficients)
    {
        MatSetValue(A, t.row(), t.col(), t.value(), INSERT_VALUES);
    }

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

// REALLY SLOW, do not use it !!!! we convert eigen matrix to petsc one in the solve...
template <>
inline Mat matrix_from_coeff(const std::vector<Triplet> &coefficients, int matrix_size)
{
    enum
    {
        IsRowMajor = SpMat::IsRowMajor
    };

    typename SpMat::IndexVector wi(matrix_size);

    // pass 1: count the nnz per inner-vector
    wi.setZero();
    for (auto it(coefficients.begin()); it != coefficients.end(); ++it)
    {
        eigen_assert(it->row() >= 0 && it->row() < mat.rows() && it->col() >= 0 && it->col() < mat.cols());
        wi(IsRowMajor ? it->col() : it->row())++;
    }

    Mat A;

    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, matrix_size, matrix_size);
    MatSeqAIJSetPreallocation(A, 0, wi.data());

    // MatCreateSeqAIJ(PETSC_COMM_WORLD, matrix_size, matrix_size, 0, wi.data(), &A);
    // matrix_from_coeff<Mat>(A, coefficients);
    PetscInt Istart, Iend;
    MatSetFromOptions(A);
    MatSetUp(A);
    MatGetOwnershipRange(A, &Istart, &Iend);

    int i = 0;
    for (auto t : coefficients)
    {
        MatSetValue(A, t.row(), t.col(), t.value(), INSERT_VALUES);
        i++;
    }

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    // MatView(A, PETSC_VIEWER_STDOUT_WORLD);

    return A;
}

//
// template for creating M matrix (K is the fission operator)
//

template <typename T, typename V>
T setup_m_operators(T &D, V volumes, mat::Macrolib &macrolib)
{
    auto R = operators::diff_removal_op<T, V>(volumes, macrolib);
    auto S = operators::diff_scatering_op<T, V>(volumes, macrolib);
    auto M = R - S - D;
    return M;
}

template <>
inline Mat setup_m_operators(Mat &D, Tensor1D volumes, mat::Macrolib &macrolib)
{
    auto M = operators::diff_removal_op<Mat, Tensor1D>(volumes, macrolib);

    auto S = operators::diff_scatering_op<Mat, Tensor1D>(volumes, macrolib);

    MatAXPY(M, -1.0, S, SAME_NONZERO_PATTERN);
    MatAXPY(M, -1.0, D, SAME_NONZERO_PATTERN);
    return M;
}

//
//template for creating the cond operators 
//
template <typename T, typename V>
void setup_cond_operators(std::vector<T> &F, std::vector<T> &chi, std::vector<T> &A, std::vector<std::vector<T>> &S,
                       std::vector<T> &D, V volumes, mat::Macrolib &macrolib)
{
    for (int g{0}; g < macrolib.getNbGroups(); ++g)
    {
        F.push_back(operators::diff_fission_op<T, Tensor1D>(g+1, volumes, macrolib));
        chi.push_back(operators::diff_fission_spectrum_op<T, Tensor1D>(g+1, volumes, macrolib));

        auto ag = operators::diff_removal_op<T, Tensor1D>(g+1, volumes, macrolib) ;
        ag -= D[g] ;
        A.push_back(ag);

        std::vector<T> sg {};
        for (int gp{0}; gp < macrolib.getNbGroups(); ++gp)
        {
            if (gp <= g)
            {
                auto null = T{};
                sg.push_back(null);
            }
            else
                sg.push_back(operators::diff_scatering_op<T, Tensor1D>(g+1, gp+1, volumes, macrolib));
        }
        S.push_back(sg);
    }
}

