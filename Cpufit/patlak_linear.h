#ifndef CPUFIT_PATLAK_LINEAR_H_INCLUDED
#define CPUFIT_PATLAK_LINEAR_H_INCLUDED

#include "cpufit.h"

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
    int * output_n_iterations);

#endif
