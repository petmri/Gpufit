#include "Cpufit/cpufit.h"
#include "Gpufit/gpufit.h"
#include "tests/utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr REAL PI = static_cast<REAL>(3.14159265359);

struct ModelCase
{
    std::string name;
    int model_id;
    std::size_t n_fits;
    std::size_t n_points;
    std::size_t n_parameters;
    REAL tolerance;
    int max_n_iterations;
    // CPU/GPU agreement bars; TISSUE_UPTAKE and TWO_COMPARTMENT_EXCHANGE are stiff,
    // weakly-identifiable fits that need looser parameter bars than the rest.
    REAL param_rel_diff_threshold;
    REAL chi_square_abs_diff_threshold;
};

struct ComparisonResult
{
    bool ok;
    int cpufit_status;
    int gpufit_status;
    std::size_t state_mismatches;
    std::size_t joint_converged;
    REAL max_parameter_rel_diff;
    REAL max_chi_square_abs_diff;
};

std::vector<REAL> make_time_points(std::size_t const n_points)
{
    std::vector<REAL> time_points(n_points);
    REAL const start = 0.25f;
    REAL const end = 15.0f;
    REAL const dt = (end - start) / static_cast<REAL>(n_points - 1);
    for (std::size_t i = 0; i < n_points; i++)
    {
        time_points[i] = start + static_cast<REAL>(i) * dt;
    }
    return time_points;
}

std::vector<REAL> make_cp(std::vector<REAL> const & time_points)
{
    std::vector<REAL> cp(time_points.size(), 0.f);
    for (std::size_t i = 0; i < time_points.size(); i++)
    {
        cp[i] = time_points[i] >= 1.f ? 5.5f * std::exp(-0.6f * time_points[i]) : 0.f;
    }
    return cp;
}

std::vector<REAL> make_user_info_time_cp(std::vector<REAL> const & time_points, std::vector<REAL> const & cp)
{
    std::vector<REAL> user_info(2 * time_points.size(), 0.f);
    std::copy(time_points.begin(), time_points.end(), user_info.begin());
    std::copy(cp.begin(), cp.end(), user_info.begin() + time_points.size());
    return user_info;
}

std::vector<REAL> make_user_info_t1()
{
    std::array<REAL, 5> const theta_degrees{ { 2.f, 5.f, 10.f, 12.f, 15.f } };
    std::array<REAL, 5> const tr{ { 21.572f, 21.572f, 21.572f, 21.572f, 21.572f } };

    std::vector<REAL> user_info(10, 0.f);
    std::copy(theta_degrees.begin(), theta_degrees.end(), user_info.begin());
    std::copy(tr.begin(), tr.end(), user_info.begin() + theta_degrees.size());
    return user_info;
}

REAL patlak_value(
    REAL const ktrans,
    REAL const vp,
    std::size_t const point_index,
    std::vector<REAL> const & time_points,
    std::vector<REAL> const & cp)
{
    REAL convolution = 0.f;
    for (std::size_t i = 1; i < point_index; i++)
    {
        REAL const spacing = time_points[i] - time_points[i - 1];
        convolution += (cp[i - 1] + cp[i]) * spacing / 2.f;
    }
    return ktrans * convolution + vp * cp[point_index];
}

REAL tofts_value(
    REAL const ktrans,
    REAL const ve,
    std::size_t const point_index,
    std::vector<REAL> const & time_points,
    std::vector<REAL> const & cp)
{
    REAL convolution = 0.f;
    for (std::size_t i = 1; i <= point_index; i++)
    {
        REAL const spacing = time_points[i] - time_points[i - 1];
        REAL const ct = cp[i] * std::exp(-ktrans * (time_points[point_index] - time_points[i]) / ve);
        REAL const ct_prev = cp[i - 1] * std::exp(-ktrans * (time_points[point_index] - time_points[i - 1]) / ve);
        convolution += (ct + ct_prev) * spacing / 2.f;
    }
    return ktrans * convolution;
}

REAL tofts_extended_value(
    REAL const ktrans,
    REAL const ve,
    REAL const vp,
    std::size_t const point_index,
    std::vector<REAL> const & time_points,
    std::vector<REAL> const & cp)
{
    return tofts_value(ktrans, ve, point_index, time_points, cp) + vp * cp[point_index];
}

REAL tissue_uptake_value(
    REAL const ktrans,
    REAL const vp,
    REAL const fp,
    std::size_t const point_index,
    std::vector<REAL> const & time_points,
    std::vector<REAL> const & cp)
{
    REAL convolution = 0.f;
    REAL const tp = vp / (fp / ((fp / ktrans) - 1.f) + fp);
    for (std::size_t i = 1; i < point_index; i++)
    {
        REAL const spacing = time_points[i] - time_points[i - 1];
        REAL const dt = time_points[point_index] - time_points[i];
        REAL const dt_prev = time_points[point_index] - time_points[i - 1];
        REAL const ct = cp[i] * (fp * std::exp(-dt / tp) + ktrans * (1.f - std::exp(-dt / tp)));
        REAL const ct_prev = cp[i - 1] * (fp * std::exp(-dt_prev / tp) + ktrans * (1.f - std::exp(-dt_prev / tp)));
        convolution += (ct + ct_prev) * spacing / 2.f;
    }
    return convolution;
}

REAL two_compartment_exchange_value(
    REAL const ktrans,
    REAL const ve,
    REAL const vp,
    REAL const fp,
    std::size_t const point_index,
    std::vector<REAL> const & time_points,
    std::vector<REAL> const & cp)
{
    REAL ps = 0.f;
    if (ktrans >= fp)
    {
        ps = 10e8;
    }
    else
    {
        ps = fp / ((fp / ktrans) - 1.f);
    }

    REAL const tp = vp / (ps + fp);
    REAL const te = ve / ps;
    REAL const tb = vp / fp;
    REAL const ksum = 1.f / tp + 1.f / te;
    REAL const square_root_term = std::sqrt(std::pow(ksum, 2) - 4.f * (1.f / te) * (1.f / tb));
    REAL const kpos = 0.5f * (ksum + square_root_term);
    REAL const kneg = 0.5f * (ksum - square_root_term);
    REAL const eneg = (kpos - 1.f / tb) / (kpos - kneg);

    REAL convolution = 0.f;
    for (std::size_t i = 1; i < point_index; i++)
    {
        REAL const spacing = time_points[i] - time_points[i - 1];
        REAL const dt = time_points[point_index] - time_points[i];
        REAL const dt_prev = time_points[point_index] - time_points[i - 1];
        REAL const ct = cp[i] * (std::exp(-dt * kpos) + eneg * (std::exp(-dt * kneg) - std::exp(-kpos)));
        REAL const ct_prev = cp[i - 1] * (std::exp(-dt_prev * kpos) + eneg * (std::exp(-dt_prev * kneg) - std::exp(-kpos)));
        convolution += (ct + ct_prev) * spacing / 2.f;
    }
    return fp * convolution;
}

REAL t1_fa_exponential_value(
    REAL const a,
    REAL const t1,
    std::size_t const point_index,
    std::vector<REAL> const & theta_degrees,
    std::vector<REAL> const & tr)
{
    REAL const theta = theta_degrees[point_index] * PI / 180.f;
    REAL const ex = std::exp(-tr[point_index] / t1);
    return a * ((1.f - ex) * std::sin(theta)) / (1.f - ex * std::cos(theta));
}

std::vector<REAL> jitter_parameters(
    std::vector<REAL> const & base_parameters,
    std::size_t const n_fits,
    REAL const spread)
{
    std::uniform_real_distribution<REAL> uniform_dist(0.f, 1.f);
    std::vector<REAL> parameters(n_fits * base_parameters.size(), 0.f);
    for (std::size_t fit_index = 0; fit_index < n_fits; fit_index++)
    {
        for (std::size_t parameter_index = 0; parameter_index < base_parameters.size(); parameter_index++)
        {
            REAL const scale = 1.f - spread + 2.f * spread * uniform_dist(rng);
            parameters[fit_index * base_parameters.size() + parameter_index]
                = base_parameters[parameter_index] * scale;
        }
    }
    return parameters;
}

std::vector<int> constant_parameters_to_fit(std::size_t const n_parameters)
{
    return std::vector<int>(n_parameters, 1);
}

ComparisonResult run_comparison(
    ModelCase const & model,
    std::vector<REAL> const & data,
    std::vector<REAL> const & initial_parameters,
    std::vector<int> const & parameters_to_fit,
    std::vector<REAL> const & user_info)
{
    std::vector<REAL> cpu_data = data;
    std::vector<REAL> gpu_data = data;
    std::vector<REAL> cpu_initial_parameters = initial_parameters;
    std::vector<REAL> gpu_initial_parameters = initial_parameters;
    std::vector<int> cpu_parameters_to_fit = parameters_to_fit;
    std::vector<int> gpu_parameters_to_fit = parameters_to_fit;
    std::vector<REAL> cpu_user_info = user_info;
    std::vector<REAL> gpu_user_info = user_info;

    std::vector<REAL> cpu_parameters(model.n_fits * model.n_parameters, 0.f);
    std::vector<int> cpu_states(model.n_fits, -1);
    std::vector<REAL> cpu_chi_squares(model.n_fits, 0.f);
    std::vector<int> cpu_iterations(model.n_fits, 0);

    std::vector<REAL> gpu_parameters(model.n_fits * model.n_parameters, 0.f);
    std::vector<int> gpu_states(model.n_fits, -1);
    std::vector<REAL> gpu_chi_squares(model.n_fits, 0.f);
    std::vector<int> gpu_iterations(model.n_fits, 0);

    int const cpufit_status = cpufit(
        model.n_fits,
        model.n_points,
        cpu_data.data(),
        nullptr,
        model.model_id,
        cpu_initial_parameters.data(),
        model.tolerance,
        model.max_n_iterations,
        cpu_parameters_to_fit.data(),
        LSE,
        cpu_user_info.size() * sizeof(REAL),
        reinterpret_cast<char *>(cpu_user_info.data()),
        cpu_parameters.data(),
        cpu_states.data(),
        cpu_chi_squares.data(),
        cpu_iterations.data());

    int const gpufit_status = gpufit(
        model.n_fits,
        model.n_points,
        gpu_data.data(),
        nullptr,
        model.model_id,
        gpu_initial_parameters.data(),
        model.tolerance,
        model.max_n_iterations,
        gpu_parameters_to_fit.data(),
        LSE,
        gpu_user_info.size() * sizeof(REAL),
        reinterpret_cast<char *>(gpu_user_info.data()),
        gpu_parameters.data(),
        gpu_states.data(),
        gpu_chi_squares.data(),
        gpu_iterations.data());

    std::size_t state_mismatches = 0;
    std::size_t joint_converged = 0;
    REAL max_parameter_rel_diff = 0.f;
    REAL max_chi_square_abs_diff = 0.f;

    for (std::size_t fit_index = 0; fit_index < model.n_fits; fit_index++)
    {
        if (cpu_states[fit_index] != gpu_states[fit_index])
        {
            state_mismatches++;
        }

        if (cpu_states[fit_index] == CONVERGED && gpu_states[fit_index] == CONVERGED)
        {
            joint_converged++;

            REAL const chi_abs_diff = std::abs(cpu_chi_squares[fit_index] - gpu_chi_squares[fit_index]);
            max_chi_square_abs_diff = std::max(max_chi_square_abs_diff, chi_abs_diff);

            for (std::size_t parameter_index = 0; parameter_index < model.n_parameters; parameter_index++)
            {
                std::size_t const flat_index = fit_index * model.n_parameters + parameter_index;
                REAL const denom = std::max(std::abs(cpu_parameters[flat_index]), std::abs(gpu_parameters[flat_index]));
                REAL const rel_diff = denom > 0.f
                    ? std::abs(cpu_parameters[flat_index] - gpu_parameters[flat_index]) / denom
                    : std::abs(cpu_parameters[flat_index] - gpu_parameters[flat_index]);
                max_parameter_rel_diff = std::max(max_parameter_rel_diff, rel_diff);
            }
        }
    }

    bool const status_ok = cpufit_status == ReturnState::OK && gpufit_status == ReturnState::OK;
    bool const convergence_ok = joint_converged > 0;
    bool const parity_ok = state_mismatches == 0
        && max_parameter_rel_diff < model.param_rel_diff_threshold
        && max_chi_square_abs_diff < model.chi_square_abs_diff_threshold;

    return {
        status_ok && convergence_ok && parity_ok,
        cpufit_status,
        gpufit_status,
        state_mismatches,
        joint_converged,
        max_parameter_rel_diff,
        max_chi_square_abs_diff
    };
}

ComparisonResult compare_patlak(ModelCase const & model, std::vector<REAL> const & time_points, std::vector<REAL> const & cp)
{
    std::vector<REAL> const base_parameters{ 0.05f, 0.03f };
    std::vector<REAL> const truth = jitter_parameters(base_parameters, model.n_fits, 0.1f);
    std::vector<REAL> const initial = jitter_parameters(base_parameters, model.n_fits, 0.02f);
    std::vector<REAL> data(model.n_fits * model.n_points, 0.f);

    for (std::size_t fit_index = 0; fit_index < model.n_fits; fit_index++)
    {
        REAL const ktrans = truth[fit_index * model.n_parameters + 0];
        REAL const vp = truth[fit_index * model.n_parameters + 1];
        for (std::size_t point_index = 0; point_index < model.n_points; point_index++)
        {
            data[fit_index * model.n_points + point_index] = patlak_value(ktrans, vp, point_index, time_points, cp);
        }
    }

    return run_comparison(model, data, initial, constant_parameters_to_fit(model.n_parameters), make_user_info_time_cp(time_points, cp));
}

ComparisonResult compare_tofts(ModelCase const & model, std::vector<REAL> const & time_points, std::vector<REAL> const & cp)
{
    std::vector<REAL> const base_parameters{ 0.18f, 0.45f };
    std::vector<REAL> const truth = jitter_parameters(base_parameters, model.n_fits, 0.1f);
    std::vector<REAL> const initial = jitter_parameters(base_parameters, model.n_fits, 0.02f);
    std::vector<REAL> data(model.n_fits * model.n_points, 0.f);

    for (std::size_t fit_index = 0; fit_index < model.n_fits; fit_index++)
    {
        REAL const ktrans = truth[fit_index * model.n_parameters + 0];
        REAL const ve = truth[fit_index * model.n_parameters + 1];
        for (std::size_t point_index = 0; point_index < model.n_points; point_index++)
        {
            data[fit_index * model.n_points + point_index] = tofts_value(ktrans, ve, point_index, time_points, cp);
        }
    }

    return run_comparison(model, data, initial, constant_parameters_to_fit(model.n_parameters), make_user_info_time_cp(time_points, cp));
}

ComparisonResult compare_tofts_extended(ModelCase const & model, std::vector<REAL> const & time_points, std::vector<REAL> const & cp)
{
    std::vector<REAL> const base_parameters{ 0.18f, 0.45f, 0.03f };
    std::vector<REAL> const truth = jitter_parameters(base_parameters, model.n_fits, 0.1f);
    std::vector<REAL> const initial = jitter_parameters(base_parameters, model.n_fits, 0.02f);
    std::vector<REAL> data(model.n_fits * model.n_points, 0.f);

    for (std::size_t fit_index = 0; fit_index < model.n_fits; fit_index++)
    {
        REAL const ktrans = truth[fit_index * model.n_parameters + 0];
        REAL const ve = truth[fit_index * model.n_parameters + 1];
        REAL const vp = truth[fit_index * model.n_parameters + 2];
        for (std::size_t point_index = 0; point_index < model.n_points; point_index++)
        {
            data[fit_index * model.n_points + point_index] = tofts_extended_value(ktrans, ve, vp, point_index, time_points, cp);
        }
    }

    return run_comparison(model, data, initial, constant_parameters_to_fit(model.n_parameters), make_user_info_time_cp(time_points, cp));
}

ComparisonResult compare_tissue_uptake(ModelCase const & model, std::vector<REAL> const & time_points, std::vector<REAL> const & cp)
{
    std::vector<REAL> const base_parameters{ 0.2f, 0.2f, 0.6f };
    std::vector<REAL> const truth = jitter_parameters(base_parameters, model.n_fits, 0.1f);
    std::vector<REAL> const initial = jitter_parameters(base_parameters, model.n_fits, 0.02f);
    std::vector<REAL> data(model.n_fits * model.n_points, 0.f);

    for (std::size_t fit_index = 0; fit_index < model.n_fits; fit_index++)
    {
        REAL const ktrans = truth[fit_index * model.n_parameters + 0];
        REAL const vp = truth[fit_index * model.n_parameters + 1];
        REAL const fp = truth[fit_index * model.n_parameters + 2];
        for (std::size_t point_index = 0; point_index < model.n_points; point_index++)
        {
            data[fit_index * model.n_points + point_index] = tissue_uptake_value(ktrans, vp, fp, point_index, time_points, cp);
        }
    }

    return run_comparison(model, data, initial, constant_parameters_to_fit(model.n_parameters), make_user_info_time_cp(time_points, cp));
}

ComparisonResult compare_two_compartment_exchange(ModelCase const & model, std::vector<REAL> const & time_points, std::vector<REAL> const & cp)
{
    std::vector<REAL> const base_parameters{ 0.2f, 0.6f, 0.2f, 0.6f };
    std::vector<REAL> const truth = jitter_parameters(base_parameters, model.n_fits, 0.05f);
    std::vector<REAL> const initial = jitter_parameters(base_parameters, model.n_fits, 0.01f);
    std::vector<REAL> data(model.n_fits * model.n_points, 0.f);

    for (std::size_t fit_index = 0; fit_index < model.n_fits; fit_index++)
    {
        REAL const ktrans = truth[fit_index * model.n_parameters + 0];
        REAL const ve = truth[fit_index * model.n_parameters + 1];
        REAL const vp = truth[fit_index * model.n_parameters + 2];
        REAL const fp = truth[fit_index * model.n_parameters + 3];
        for (std::size_t point_index = 0; point_index < model.n_points; point_index++)
        {
            data[fit_index * model.n_points + point_index] = two_compartment_exchange_value(ktrans, ve, vp, fp, point_index, time_points, cp);
        }
    }

    return run_comparison(model, data, initial, constant_parameters_to_fit(model.n_parameters), make_user_info_time_cp(time_points, cp));
}

ComparisonResult compare_t1_fa_exponential(ModelCase const & model)
{
    std::vector<REAL> const user_info = make_user_info_t1();
    std::vector<REAL> const theta_degrees(user_info.begin(), user_info.begin() + model.n_points);
    std::vector<REAL> const tr(user_info.begin() + model.n_points, user_info.end());

    std::vector<REAL> const base_parameters{ 1.f, 1800.f };
    std::vector<REAL> const truth = jitter_parameters(base_parameters, model.n_fits, 0.1f);
    std::vector<REAL> const initial = jitter_parameters(base_parameters, model.n_fits, 0.02f);
    std::vector<REAL> data(model.n_fits * model.n_points, 0.f);

    for (std::size_t fit_index = 0; fit_index < model.n_fits; fit_index++)
    {
        REAL const a = truth[fit_index * model.n_parameters + 0];
        REAL const t1 = truth[fit_index * model.n_parameters + 1];
        for (std::size_t point_index = 0; point_index < model.n_points; point_index++)
        {
            data[fit_index * model.n_points + point_index]
                = t1_fa_exponential_value(a, t1, point_index, theta_degrees, tr);
        }
    }

    return run_comparison(model, data, initial, constant_parameters_to_fit(model.n_parameters), user_info);
}

void print_result(ModelCase const & model, ComparisonResult const & result)
{
    std::cout << model.name << std::endl;
    std::cout << "  cpufit status:         " << result.cpufit_status << std::endl;
    std::cout << "  gpufit status:         " << result.gpufit_status << std::endl;
    std::cout << "  state mismatches:      " << result.state_mismatches << std::endl;
    std::cout << "  jointly converged:     " << result.joint_converged << " / " << model.n_fits << std::endl;
    std::cout << "  max |param rel diff|:  " << result.max_parameter_rel_diff
        << " (threshold " << model.param_rel_diff_threshold << ")" << std::endl;
    std::cout << "  max |chi-square diff|: " << result.max_chi_square_abs_diff
        << " (threshold " << model.chi_square_abs_diff_threshold << ")" << std::endl;
    std::cout << "  parity result:         " << (result.ok ? "PASS" : "FAIL") << std::endl;
    std::cout << std::endl;
}
}

int main()
{
    if (!gpufit_cuda_available())
    {
        std::cout << "CUDA not available: " << gpufit_get_last_error() << std::endl;
        return 1;
    }

    std::vector<REAL> const time_points = make_time_points(60);
    std::vector<REAL> const cp = make_cp(time_points);

    // 2e-3 default; TISSUE_UPTAKE/TWO_COMPARTMENT_EXCHANGE get looser param bars
    std::vector<ModelCase> const models{
        { "PATLAK", PATLAK, 128, 60, 2, 1e-8f, 200, 2e-3f, 2e-3f },
        { "TOFTS", TOFTS, 128, 60, 2, 1e-8f, 200, 2e-3f, 2e-3f },
        { "TOFTS_EXTENDED", TOFTS_EXTENDED, 128, 60, 3, 1e-8f, 200, 2e-3f, 2e-3f },
        { "TISSUE_UPTAKE", TISSUE_UPTAKE, 128, 60, 3, 1e-8f, 200, 5e-3f, 2e-3f },
        { "TWO_COMPARTMENT_EXCHANGE", TWO_COMPARTMENT_EXCHANGE, 128, 60, 4, 1e-8f, 200, 1e-2f, 2e-3f },
        { "T1_FA_EXPONENTIAL", T1_FA_EXPONENTIAL, 128, 5, 2, 1e-10f, 200, 2e-3f, 2e-3f }
    };

    bool all_ok = true;

    for (ModelCase const & model : models)
    {
        ComparisonResult result{};
        if (model.model_id == PATLAK)
        {
            result = compare_patlak(model, time_points, cp);
        }
        else if (model.model_id == TOFTS)
        {
            result = compare_tofts(model, time_points, cp);
        }
        else if (model.model_id == TOFTS_EXTENDED)
        {
            result = compare_tofts_extended(model, time_points, cp);
        }
        else if (model.model_id == TISSUE_UPTAKE)
        {
            result = compare_tissue_uptake(model, time_points, cp);
        }
        else if (model.model_id == TWO_COMPARTMENT_EXCHANGE)
        {
            result = compare_two_compartment_exchange(model, time_points, cp);
        }
        else if (model.model_id == T1_FA_EXPONENTIAL)
        {
            result = compare_t1_fa_exponential(model);
        }
        else
        {
            throw std::runtime_error("Unknown model ID in parity driver.");
        }

        print_result(model, result);
        all_ok = all_ok && result.ok;
    }

    if (!all_ok)
    {
        std::cout << "MRI GPU/CPU parity check failed for one or more models." << std::endl;
        return 1;
    }

    std::cout << "MRI GPU/CPU parity check passed." << std::endl;
    return 0;
}
