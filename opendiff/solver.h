#ifndef SOLVER_H
#define SOLVER_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Eigen/Sparse>
#include <petscmat.h>
#include <string>
#include "spdlog/spdlog.h"

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

    template <class T>
    class Solver
    {
    protected:
        T m_K{};
        T m_M{};

        mat::Macrolib m_macrolib;
        Tensor1D m_volumes{};

        bool m_is_normed{false};
        std::string m_norm_method{};

        vecd m_eigen_values{};
        vecvec m_eigen_vectors{};

    public:
        Solver() = delete;
        Solver(const Solver &copy) = default;
        Solver(const T &K, const T &M)
        {
            m_K = K;
            m_M = M;
        };
        Solver(vecd &x, vecd &y, vecd &z, mat::Macrolib &macrolib,
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
        };

        Solver(vecd &x, vecd &y, mat::Macrolib &macrolib,
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
        };

        Solver(vecd &x, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn)
        {
            // calculate dx
            auto dx = delta_coord(x);
            m_volumes = dx;

            m_K = operators::diff_fission_op<T, Tensor1D>(m_volumes, macrolib);
            auto D = operators::diff_diffusion_op<T, Tensor1D>(dx, macrolib, albedo_x0, albedo_xn);
            m_M = operators::setup_m_operators<T, Tensor1D>(D, m_volumes, macrolib);

            m_macrolib = macrolib;
        };

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
        const auto &getEigenVectors() const
        {
            return m_eigen_vectors;
        };

        const auto &getEigenVector(int i) const
        {
            return m_eigen_vectors[i];
        };

        const auto getEigenVector4D(int i, int dim_x, int dim_y, int dim_z, int nb_groups) const
        {
            Eigen::TensorMap<Tensor4Dconst> a(m_eigen_vectors[i].data(), dim_z, dim_y, dim_x, nb_groups);
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

        const auto &getK() const
        {
            return m_K;
        };
        const auto &getM() const
        {
            return m_M;
        };

        const auto &getVolumes() const
        {
            return m_volumes;
        };

        const auto getVolumesPython() const
        {
            return py::array_t<double, py::array::c_style>({m_volumes.dimension(0)},
                                                           m_volumes.data());
        };

        virtual void makeAdjoint()
        {
            m_M = m_M.adjoint();
            m_K = m_K.adjoint();
        };

        virtual void solve(double tol = 1e-6, double tol_eigen_vectors = 1e-4, int nb_eigen_values = 1, const Eigen::VectorXd &v0 = Eigen::VectorXd(),
                           double tol_inner = 1e-6, int outer_max_iter = 500, int inner_max_iter = 200, std::string inner_solver = "BiCGSTAB", std::string inner_precond = "") = 0;

        void handleDenegeratedEigenvalues();

        // todo: use getPower(Tensor4Dconst
        // todo: use matrix muktiplication
        Tensor3D getPower(int i = 0)
        {
            auto nb_groups = m_macrolib.getNbGroups();

            auto dim = m_macrolib.getDim();
            auto dim_z = std::get<2>(dim), dim_y = std::get<1>(dim), dim_x = std::get<0>(dim);
            Tensor3D power(dim_z, dim_y, dim_x); // z, y, x
            power.setZero();
            auto eigenvectori = getEigenVector4D(i, dim_x, dim_y, dim_z, nb_groups);

            for (int i{0}; i < nb_groups; ++i)
            {
                power = power.eval() + m_macrolib.getValues(i + 1, "SIGF") * eigenvectori.chip(i, 3) * m_macrolib.getValues(i + 1, "EFISS");
            }

            return power;
        };

        // // todo: add EFISS and SIGF in materials and macrolib
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
        // };

        // Tensor3D getPower(int i = 0)
        // {
        //     auto eigenvectori = getEigenVector4D(i);
        //     auto power = getPower(eigenvectori);
        //     return power;
        // };

        const py::array_t<double> getPowerPython()
        {
            auto power = getPower(0);
            return py::array_t<double, py::array::c_style>({power.dimension(0), power.dimension(1), power.dimension(2)},
                                                           power.data());
        };

        const Tensor3D normPower(double power_W = 1)
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

        const py::array_t<double> normPowerPython(double power_W = 1)
        {
            auto power = normPower(power_W);
            return py::array_t<double, py::array::c_style>({power.dimension(0), power.dimension(1), power.dimension(2)},
                                                           power.data());
        };

        void normPhiStarMPhi(solver::Solver<SpMat> &solver_star)
        {
            auto eigen_vectors_star = solver_star.getEigenVectors();
            // if (eigen_vectors_star.size() != m_eigen_vectors.size())
            //     throw std::invalid_argument("The number of eigen vectors has the be identical in this and solver_star!");

            int nb_ev = static_cast<int>(std::min(m_eigen_vectors.size(), eigen_vectors_star.size()));
            for (int i{0}; i < nb_ev; ++i)
            {
                double factor = eigen_vectors_star[i].dot(m_M * m_eigen_vectors[i]);
                m_eigen_vectors[i] = m_eigen_vectors[i] / factor;
            }
            m_is_normed = true;
            m_norm_method = "PhiStarMPhi";
        }

        void norm(std::string method, solver::Solver<SpMat> &solver_star, double power_W = 1)
        {
            if (method == "power")
                normPower(power_W);
            else if (method == "PhiStarMPhi")
                normPhiStarMPhi(solver_star);
            else
                throw std::invalid_argument("Invalid method name!");
        }
    };

    class SolverPowerIt : public Solver<SpMat>
    {
    public:
        SolverPowerIt(const Solver &copy) : Solver(copy){};
        SolverPowerIt(vecd &x, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn) : Solver(x, macrolib, albedo_x0, albedo_xn){};
        SolverPowerIt(vecd &x, vecd &y, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn) : Solver(x, y, macrolib, albedo_x0, albedo_xn, albedo_y0, albedo_yn){};
        SolverPowerIt(vecd &x, vecd &y, vecd &z, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn) : Solver(x, y, z, macrolib, albedo_x0, albedo_xn, albedo_y0, albedo_yn, albedo_z0, albedo_zn){};

        void solve(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0,
                   double tol_inner, int outer_max_iter, int inner_max_iter, std::string inner_solver, std::string inner_precond) override;

        void solveLU(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0, int outer_max_iter);

        template <class T>
        void solveIterative(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0,
                            double tol_inner, int outer_max_iter, int inner_max_iter)
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

            spdlog::debug("Tolerance in outter iteration (eigen value): {:.2e}", tol);
            spdlog::debug("Tolerance in outter iteration (eigen vector): {:.2e}", tol_eigen_vectors);
            spdlog::debug("Tolerance in inner iteration : {:.2e}", tol_inner);
            spdlog::debug("Max. outer iteration : {}", outer_max_iter);
            spdlog::debug("Max. inner iteration : {}", inner_max_iter);

            double eigen_value = v.norm();
            double eigen_value_prec = eigen_value;

            T solver;
            solver.setMaxIterations(inner_max_iter);
            solver.setTolerance(tol_inner);
            solver.compute(m_M);

            // outer iteration
            int i = 0;
            while ((r_tol > tol || r_tol_ev > tol_eigen_vectors) && i < outer_max_iter)
            {
                spdlog::debug("----------------------------------------------------");
                spdlog::debug("Outer iteration {}", i);
                auto b = m_K * v;
                // inner iteration
                v = solver.solveWithGuess(b, v);
                spdlog::debug("Number of inner iteration: {}", solver.iterations());
                spdlog::debug("Estimated error in inner iteration: {:.2e}", solver.error());
                eigen_value = v.norm();
                v = v / eigen_value;

                // precision computation
                r_tol = std::abs(eigen_value - eigen_value_prec); // todo: use relative eps (use the norm for the eigen vectors)
                r_tol_ev = abs((v - v_prec).maxCoeff());
                eigen_value_prec = eigen_value;
                v_prec = v;
                spdlog::debug("Eigen value = {:.5f}", eigen_value);
                spdlog::debug("Estimated error in outter iteration (eigen value): {:.2e}", r_tol);
                spdlog::debug("Estimated error in outter iteration (eigen vector): {:.2e}", r_tol_ev);
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
    };

    class SolverSlepc : public Solver<SpMat> // row major for MatCreateSeqAIJWithArrays
    {
    public:
        SolverSlepc(const Solver &copy) : Solver(copy){};
        SolverSlepc(vecd &x, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn) : Solver(x, macrolib, albedo_x0, albedo_xn){};
        SolverSlepc(vecd &x, vecd &y, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn) : Solver(x, y, macrolib, albedo_x0, albedo_xn, albedo_y0, albedo_yn){};
        SolverSlepc(vecd &x, vecd &y, vecd &z, mat::Macrolib &macrolib, double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn) : Solver(x, y, z, macrolib, albedo_x0, albedo_xn, albedo_y0, albedo_yn, albedo_z0, albedo_zn){};

        void solve(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0,
                   double tol_inner, int outer_max_iter, int inner_max_iter, std::string inner_solver, std::string inner_precond) override;

        void solveIterative(double tol, double tol_eigen_vectors, int nb_eigen_values, const Eigen::VectorXd &v0,
                            double tol_inner, int outer_max_iter, int inner_max_iter, std::string solver, std::string inner_solver, std::string inner_precond);
    };

} // namespace solver

#endif // SOLVER_H