#define BOOST_TEST_MODULE Gpufit

#include "Cpufit/cpufit.h"

#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace
{
std::vector<REAL> make_patlak_curve(
    std::vector<REAL> const& time_x,
    std::vector<REAL> const& cp,
    REAL const ktrans,
    REAL const vp)
{
    std::size_t const n_points = time_x.size();
    std::vector<REAL> data(n_points, 0.f);
    for (std::size_t point_index = 0; point_index < n_points; point_index++)
    {
        REAL conv_cp = 0.f;
        for (std::size_t i = 1; i <= point_index; i++)
        {
            REAL const spacing = time_x[i] - time_x[i - 1];
            conv_cp += (cp[i - 1] + cp[i]) * spacing / 2.f;
        }
        data[point_index] = ktrans * conv_cp + vp * cp[point_index];
    }
    return data;
}

void build_time_and_cp(std::size_t const n_points, std::vector<REAL>& time_x, std::vector<REAL>& cp)
{
    time_x.resize(n_points);
    cp.resize(n_points);
    REAL const dt = (15.f - 0.25f) / REAL(n_points - 1);

    for (std::size_t i = 0; i < n_points; i++)
    {
        time_x[i] = 0.25f + REAL(i) * dt;
        cp[i] = time_x[i] >= 1.f ? 5.5f * std::exp(-0.6f * time_x[i]) : 0.f;
    }
}
}

BOOST_AUTO_TEST_CASE(PatlakCpu)
{
    std::size_t const n_fits = 1;
    std::size_t const n_points = 60;

    std::vector<REAL> time_x;
    std::vector<REAL> cp;
    build_time_and_cp(n_points, time_x, cp);

    REAL const ktrans_true = 0.05f;
    REAL const vp_true = 0.03f;

    std::vector<REAL> data = make_patlak_curve(time_x, cp, ktrans_true, vp_true);

    std::vector<REAL> initial_parameters{ 0.02f, 0.01f };
    std::vector<int> parameters_to_fit{ 1, 1 };

    std::vector<REAL> user_info;
    user_info.reserve(2 * n_points);
    user_info.insert(user_info.end(), time_x.begin(), time_x.end());
    user_info.insert(user_info.end(), cp.begin(), cp.end());

    std::vector<REAL> output_parameters(2, 0.f);
    int output_state = -1;
    REAL output_chi_square = -1.f;
    int output_iterations = -1;

    int const status = cpufit(
        n_fits,
        n_points,
        data.data(),
        nullptr,
        PATLAK,
        initial_parameters.data(),
        1e-7f,
        200,
        parameters_to_fit.data(),
        LSE,
        user_info.size() * sizeof(REAL),
        reinterpret_cast<char*>(user_info.data()),
        output_parameters.data(),
        &output_state,
        &output_chi_square,
        &output_iterations);

    BOOST_CHECK(status == 0);
    BOOST_CHECK(output_state == 0);
    BOOST_CHECK(std::abs(output_parameters[0] - ktrans_true) < 1e-6f);
    BOOST_CHECK(std::abs(output_parameters[1] - vp_true) < 1e-6f);
}

BOOST_AUTO_TEST_CASE(PatlakCpuConstrainedGlobalAndLinear)
{
    std::size_t const n_fits = 8;
    std::size_t const n_points = 60;

    std::vector<REAL> time_x;
    std::vector<REAL> cp;
    build_time_and_cp(n_points, time_x, cp);

    std::vector<REAL> user_info;
    user_info.reserve(2 * n_points);
    user_info.insert(user_info.end(), time_x.begin(), time_x.end());
    user_info.insert(user_info.end(), cp.begin(), cp.end());

    std::vector<REAL> data(n_fits * n_points, 0.f);
    std::vector<REAL> true_parameters{
        0.00f, 0.03f,
        0.00f, 0.05f,
        0.02f, 0.03f,
        0.06f, 0.03f,
        0.10f, 0.03f,
        0.10f, 0.01f,
        0.04f, 0.06f,
        0.08f, 0.04f,
    };

    for (std::size_t fit_index = 0; fit_index < n_fits; fit_index++)
    {
        std::vector<REAL> curve = make_patlak_curve(
            time_x,
            cp,
            true_parameters[fit_index * 2 + 0],
            true_parameters[fit_index * 2 + 1]);
        for (std::size_t point_index = 0; point_index < n_points; point_index++)
        {
            data[fit_index * n_points + point_index] = curve[point_index];
        }
    }

    std::vector<REAL> initial_parameters(n_fits * 2, 0.f);
    for (std::size_t fit_index = 0; fit_index < n_fits; fit_index++)
    {
        initial_parameters[fit_index * 2 + 0] = 0.18f;
        initial_parameters[fit_index * 2 + 1] = 0.18f;
    }

    std::vector<REAL> constraints(n_fits * 4, 0.f);
    for (std::size_t fit_index = 0; fit_index < n_fits; fit_index++)
    {
        constraints[fit_index * 4 + 0] = 0.0f;
        constraints[fit_index * 4 + 1] = 0.10f;
        constraints[fit_index * 4 + 2] = 0.0f;
        constraints[fit_index * 4 + 3] = 0.20f;
    }

    std::vector<int> constraint_types{ ConstraintType::LOWER_UPPER, ConstraintType::LOWER_UPPER };
    std::vector<int> parameters_to_fit{ 1, 1 };

    std::vector<REAL> lm_parameters(n_fits * 2, 0.f);
    std::vector<int> lm_states(n_fits, -1);
    std::vector<REAL> lm_chi_square(n_fits, -1.f);
    std::vector<int> lm_iterations(n_fits, -1);

    int const lm_status = cpufit_constrained(
        n_fits,
        n_points,
        data.data(),
        nullptr,
        PATLAK,
        initial_parameters.data(),
        constraints.data(),
        constraint_types.data(),
        1e-7f,
        200,
        parameters_to_fit.data(),
        LSE,
        user_info.size() * sizeof(REAL),
        reinterpret_cast<char*>(user_info.data()),
        lm_parameters.data(),
        lm_states.data(),
        lm_chi_square.data(),
        lm_iterations.data());

    BOOST_CHECK(lm_status == 0);

    std::vector<REAL> linear_parameters(n_fits * 2, 0.f);
    std::vector<int> linear_states(n_fits, -1);
    std::vector<REAL> linear_chi_square(n_fits, -1.f);
    std::vector<int> linear_iterations(n_fits, -1);

    int const linear_status = cpufit_patlak_bounded_linear(
        n_fits,
        n_points,
        data.data(),
        nullptr,
        initial_parameters.data(),
        constraints.data(),
        constraint_types.data(),
        parameters_to_fit.data(),
        user_info.size() * sizeof(REAL),
        reinterpret_cast<char*>(user_info.data()),
        linear_parameters.data(),
        linear_states.data(),
        linear_chi_square.data(),
        linear_iterations.data());

    BOOST_CHECK(linear_status == 0);

    for (std::size_t fit_index = 0; fit_index < n_fits; fit_index++)
    {
        BOOST_CHECK(lm_states[fit_index] == FitState::CONVERGED);
        BOOST_CHECK(linear_states[fit_index] == FitState::CONVERGED);
        REAL const chi_square_delta = std::abs(lm_chi_square[fit_index] - linear_chi_square[fit_index]);
        REAL const chi_square_tol = (std::max)(
            static_cast<REAL>(1e-6f),
            static_cast<REAL>(1e-3f) * (std::max)(std::abs(linear_chi_square[fit_index]), static_cast<REAL>(1.f)));
        BOOST_CHECK(chi_square_delta <= chi_square_tol);
        BOOST_CHECK(std::abs(lm_parameters[fit_index * 2 + 0] - linear_parameters[fit_index * 2 + 0]) < 2e-4f);
        BOOST_CHECK(std::abs(lm_parameters[fit_index * 2 + 1] - linear_parameters[fit_index * 2 + 1]) < 2e-4f);
    }
}
