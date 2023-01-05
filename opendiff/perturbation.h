#ifndef PERTURBATION_H
#define PERTURBATION_H

#include <tuple>
#include <string>
#include <vector>
#include <map>
#include <random>
#include <chrono>
#include "spdlog/fmt/ostr.h"
#include <math.h>
#include <memory> //std::addressof

#include <highfive/H5Easy.hpp> // serialization

#include "solver.h"
#include "materials.h"

namespace perturbation
{
    using vecd = std::vector<double>;
    using vecvec = std::vector<Eigen::VectorXd>;
    using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    using Tensor0D = Eigen::Tensor<double, 0, Eigen::RowMajor>;
    using Tensor1D = Eigen::Tensor<double, 1, Eigen::RowMajor>;
    using Tensor2D = Eigen::Tensor<double, 2, Eigen::RowMajor>;

    template <typename T>
    void handleDegeneratedEigenvalues(T &solver, T &solver_star, double max_eps = 1e-6);

    template <typename T>
    bool checkBiOrthogonality(T &solver, T &solver_star, double max_eps = 1e-6, bool raise_error = false, bool remove = false);

    template <typename T>
    std::tuple<Eigen::VectorXd, double, vecd> firstOrderPerturbation(T &solver, T &solver_star, T &solver_pert, std::string norm_method);

    template <typename T>
    std::tuple<Eigen::VectorXd, double, Tensor2D> highOrderPerturbation(int order, T &solver, T &solver_star, T &solver_pert);
    template <typename T>
    std::tuple<Eigen::VectorXd, double, py::array_t<double>> highOrderPerturbationPython(int order, T &solver, T &solver_star, T &solver_pert);

    template <typename T>
    std::tuple<double, Eigen::VectorXd> GPTAdjointImportance(const T &solver, const T &solver_star,
                                                             Eigen::VectorXd &response, Eigen::VectorXd &norm,
                                                             double tol, double tol_inner, int outer_max_iter, int inner_max_iter,
                                                             std::string inner_solver, std::string inner_precond, std::string acceleration);

    template <typename T>
    double firstOrderGPT(const T &solver, const T &solver_star, const T &solver_pert,
                         Eigen::VectorXd &response, Eigen::VectorXd &response_pert,
                         Eigen::VectorXd &norm, Eigen::VectorXd &norm_pert,
                         double &N_star, Eigen::VectorXd &gamma_star);

    // the response are not ratio
    template <typename T>
    std::tuple<double, Eigen::VectorXd> firstOrderGPT(const T &solver, const T &solver_star,
                                                      const T &solver_pert,
                                                      Eigen::VectorXd &response, Eigen::VectorXd &response_pert,
                                                      Eigen::VectorXd &norm, Eigen::VectorXd &norm_pert,
                                                      double tol, double tol_inner, int outer_max_iter, int inner_max_iter,
                                                      std::string inner_solver, std::string inner_precond, std::string acceleration);
    template <class T>
    class EpGPT
    {
    protected:
        vecd m_x{};
        vecd m_y{};
        vecd m_z{};
        std::array<double, 6> m_albedos{{0., 0., 0., 0., 0., 0.}};
        std::vector<std::vector<std::vector<std::string>>> m_geometry{};

        mat::Middles m_middles{};

        T m_solver{};
        T m_solver_star{};
        Eigen::VectorXd m_norm_vector{};

        double m_precision{1e-5};
        vecvec m_basis{};
        vecvec m_gamma_star{};
        std::vector<double> m_N_star{};

        T &createPerturbedSolver(std::map<std::string, double> reactions);

        void clearBasis();

    public:
        EpGPT(const EpGPT &copy) = default;

        EpGPT(vecd &x, vecd &y, vecd &z, mat::Middles &middles, const std::vector<std::vector<std::vector<std::string>>> &geometry,
              double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn, double albedo_z0, double albedo_zn);

        EpGPT(vecd &x, vecd &y, mat::Middles &middles, const std::vector<std::vector<std::vector<std::string>>> &geometry,
              double albedo_x0, double albedo_xn, double albedo_y0, double albedo_yn);

        EpGPT(vecd &x, mat::Middles &middles, const std::vector<std::vector<std::vector<std::string>>> &geometry,
              double albedo_x0, double albedo_xn);

        void createBasis(double precision, std::vector<std::string> reactions, double pert_value_max, double power_W,
                         double tol, double tol_eigen_vectors, const Eigen::VectorXd &v0, double ev0,
                         double tol_inner, int outer_max_iter, int inner_max_iter, std::string inner_solver, std::string inner_precond,
                         std::string acceleration);

        void calcImportances(double tol, const Eigen::VectorXd &v0, double tol_inner,
                             int outer_max_iter, int inner_max_iter, std::string inner_solver, std::string acceleration);

        std::tuple<Eigen::VectorXd, double, vecd> firstOrderPerturbation(T &solver_pert);

        auto &getBasis() { return m_basis; };
        auto &getImportances() { return m_gamma_star; };
        auto &getN_star() { return m_N_star; };
        auto &getSolver() { return m_solver; };
        auto &getSolverStar() { return m_solver_star; };

        void dump(std::string file_name);

        void load(std::string file_name);
    };

#include "perturbation.inl"

} // namespace perturbation

#endif // PERTURBATION_H