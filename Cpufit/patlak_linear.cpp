#include "patlak_linear.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
constexpr REAL EPS = static_cast<REAL>(1e-20f);

REAL clamp_value(REAL const value, REAL const lower, REAL const upper)
{
    return std::min(std::max(value, lower), upper);
}

struct Candidate
{
    REAL ktrans;
    REAL vp;
    REAL objective;
};

REAL objective_value(
    REAL const c,
    REAL const g0,
    REAL const g1,
    REAL const a00,
    REAL const a01,
    REAL const a11,
    REAL const ktrans,
    REAL const vp)
{
    return c
        - static_cast<REAL>(2.f) * (g0 * ktrans + g1 * vp)
        + (a00 * ktrans * ktrans + static_cast<REAL>(2.f) * a01 * ktrans * vp + a11 * vp * vp);
}

void solve_single_fit(
    std::size_t const n_points,
    REAL const * y,
    REAL const * w,
    REAL const * integral_cp,
    REAL const * cp,
    REAL const * initial_parameters,
    REAL const * constraints,
    int const * constraint_types,
    int const * parameters_to_fit,
    REAL & out_ktrans,
    REAL & out_vp,
    REAL & out_chi_square,
    int & out_state,
    int & out_iterations)
{
    REAL lower_k = -std::numeric_limits<REAL>::max();
    REAL upper_k = std::numeric_limits<REAL>::max();
    REAL lower_v = -std::numeric_limits<REAL>::max();
    REAL upper_v = std::numeric_limits<REAL>::max();

    if (constraint_types && constraints)
    {
        if (constraint_types[0] == ConstraintType::LOWER || constraint_types[0] == ConstraintType::LOWER_UPPER)
        {
            lower_k = constraints[0 * 2 + LOWER_BOUND];
        }
        if (constraint_types[0] == ConstraintType::UPPER || constraint_types[0] == ConstraintType::LOWER_UPPER)
        {
            upper_k = constraints[0 * 2 + UPPER_BOUND];
        }

        if (constraint_types[1] == ConstraintType::LOWER || constraint_types[1] == ConstraintType::LOWER_UPPER)
        {
            lower_v = constraints[1 * 2 + LOWER_BOUND];
        }
        if (constraint_types[1] == ConstraintType::UPPER || constraint_types[1] == ConstraintType::LOWER_UPPER)
        {
            upper_v = constraints[1 * 2 + UPPER_BOUND];
        }
    }

    if (lower_k > upper_k || lower_v > upper_v)
    {
        throw std::runtime_error("Invalid Patlak bounds: lower bound exceeds upper bound.");
    }

    if (parameters_to_fit && !parameters_to_fit[0])
    {
        lower_k = initial_parameters[0];
        upper_k = initial_parameters[0];
    }
    if (parameters_to_fit && !parameters_to_fit[1])
    {
        lower_v = initial_parameters[1];
        upper_v = initial_parameters[1];
    }

    REAL a00 = 0.f;
    REAL a01 = 0.f;
    REAL a11 = 0.f;
    REAL g0 = 0.f;
    REAL g1 = 0.f;
    REAL c = 0.f;

    for (std::size_t i = 0; i < n_points; i++)
    {
        REAL const wi = w ? w[i] : static_cast<REAL>(1.f);
        REAL const xi0 = integral_cp[i];
        REAL const xi1 = cp[i];
        REAL const yi = y[i];

        a00 += wi * xi0 * xi0;
        a01 += wi * xi0 * xi1;
        a11 += wi * xi1 * xi1;
        g0 += wi * xi0 * yi;
        g1 += wi * xi1 * yi;
        c += wi * yi * yi;
    }

    std::vector<Candidate> candidates;
    candidates.reserve(9);

    REAL const det = a00 * a11 - a01 * a01;
    if (std::abs(det) > EPS)
    {
        REAL const k_unc = (g0 * a11 - g1 * a01) / det;
        REAL const v_unc = (g1 * a00 - g0 * a01) / det;
        if (k_unc >= lower_k && k_unc <= upper_k && v_unc >= lower_v && v_unc <= upper_v)
        {
            candidates.push_back({
                k_unc,
                v_unc,
                objective_value(c, g0, g1, a00, a01, a11, k_unc, v_unc)
            });
        }
    }

    auto add_edge_k = [&](REAL const k_fixed)
    {
        REAL v_opt = 0.f;
        if (a11 > EPS)
        {
            v_opt = (g1 - a01 * k_fixed) / a11;
        }
        else
        {
            REAL const slope = g1 - a01 * k_fixed;
            v_opt = slope >= 0.f ? upper_v : lower_v;
        }
        v_opt = clamp_value(v_opt, lower_v, upper_v);
        candidates.push_back({
            k_fixed,
            v_opt,
            objective_value(c, g0, g1, a00, a01, a11, k_fixed, v_opt)
        });
    };

    auto add_edge_v = [&](REAL const v_fixed)
    {
        REAL k_opt = 0.f;
        if (a00 > EPS)
        {
            k_opt = (g0 - a01 * v_fixed) / a00;
        }
        else
        {
            REAL const slope = g0 - a01 * v_fixed;
            k_opt = slope >= 0.f ? upper_k : lower_k;
        }
        k_opt = clamp_value(k_opt, lower_k, upper_k);
        candidates.push_back({
            k_opt,
            v_fixed,
            objective_value(c, g0, g1, a00, a01, a11, k_opt, v_fixed)
        });
    };

    add_edge_k(lower_k);
    add_edge_k(upper_k);
    add_edge_v(lower_v);
    add_edge_v(upper_v);

    candidates.push_back({ lower_k, lower_v, objective_value(c, g0, g1, a00, a01, a11, lower_k, lower_v) });
    candidates.push_back({ lower_k, upper_v, objective_value(c, g0, g1, a00, a01, a11, lower_k, upper_v) });
    candidates.push_back({ upper_k, lower_v, objective_value(c, g0, g1, a00, a01, a11, upper_k, lower_v) });
    candidates.push_back({ upper_k, upper_v, objective_value(c, g0, g1, a00, a01, a11, upper_k, upper_v) });

    auto const best_it = std::min_element(
        candidates.begin(),
        candidates.end(),
        [](Candidate const & a, Candidate const & b)
        {
            return a.objective < b.objective;
        });

    if (best_it == candidates.end() || !std::isfinite(best_it->objective))
    {
        out_state = FitState::SINGULAR_HESSIAN;
        out_iterations = 0;
        out_chi_square = 0.f;
        out_ktrans = clamp_value(initial_parameters[0], lower_k, upper_k);
        out_vp = clamp_value(initial_parameters[1], lower_v, upper_v);
        return;
    }

    out_state = FitState::CONVERGED;
    out_iterations = 1;
    out_ktrans = best_it->ktrans;
    out_vp = best_it->vp;
    out_chi_square = std::max(static_cast<REAL>(0.f), best_it->objective);
}
}

void solve_patlak_bounded_linear(
    std::size_t n_fits,
    std::size_t n_points,
    REAL const * data,
    REAL const * weights,
    REAL const * initial_parameters,
    REAL const * constraints,
    int const * constraint_types,
    int const * parameters_to_fit,
    std::size_t user_info_size,
    char const * user_info,
    REAL * output_parameters,
    int * output_states,
    REAL * output_chi_squares,
    int * output_n_iterations)
{
    if (!data || !initial_parameters || !output_parameters || !output_states || !output_chi_squares || !output_n_iterations)
    {
        throw std::runtime_error("Patlak bounded linear solver received null required pointer.");
    }

    if (!user_info)
    {
        throw std::runtime_error("PATLAK bounded linear solver requires user info containing time and Cp arrays.");
    }

    std::size_t const minimum_user_info_size = 2 * n_points * sizeof(REAL);
    if (user_info_size < minimum_user_info_size)
    {
        throw std::runtime_error("PATLAK bounded linear solver requires user_info_size >= 2 * n_points * sizeof(REAL).");
    }

    REAL const * const user_info_float = reinterpret_cast<REAL const *>(user_info);
    REAL const * const time = user_info_float;
    REAL const * const cp = user_info_float + n_points;

    std::vector<REAL> integral_cp(n_points, 0.f);
    for (std::size_t i = 1; i < n_points; i++)
    {
        REAL const spacing = time[i] - time[i - 1];
        integral_cp[i] = integral_cp[i - 1] + (cp[i - 1] + cp[i]) * spacing / 2.f;
    }

    for (std::size_t fit_index = 0; fit_index < n_fits; fit_index++)
    {
        REAL const * const y = data + fit_index * n_points;
        REAL const * const w = weights ? (weights + fit_index * n_points) : nullptr;
        REAL const * const fit_initial = initial_parameters + fit_index * 2;
        REAL const * const fit_constraints = constraints ? (constraints + fit_index * 4) : nullptr;

        REAL & out_k = output_parameters[fit_index * 2 + 0];
        REAL & out_v = output_parameters[fit_index * 2 + 1];
        REAL & out_chi = output_chi_squares[fit_index];
        int & out_state = output_states[fit_index];
        int & out_iter = output_n_iterations[fit_index];

        solve_single_fit(
            n_points,
            y,
            w,
            integral_cp.data(),
            cp,
            fit_initial,
            fit_constraints,
            constraint_types,
            parameters_to_fit,
            out_k,
            out_v,
            out_chi,
            out_state,
            out_iter);
    }
}
