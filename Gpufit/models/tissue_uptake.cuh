#ifdef USE_TISSUE_UPTAKE
#define GPUFIT_TISSUE_UPTAKE_CUH_INCLUDED

// Reparameterized 2CUM (tissue uptake): parameters [E = Ktrans/Fp, vp, Fp].
//   PS = Fp*E/(1-E);  rate rp = Fp/(vp*(1-E)) = 1/Tp.
//   C = E*Fp*U + Fp*(1-E)*G(rp),  U = integral of Cp, G = exp-weighted integral of Cp.
// Analytic Jacobian from the exp-weighted convolution and its rate-derivative (no PS pole,
// no numerical differences). The extraction-fraction bound E -> 1 replaces the old Ktrans=Fp
// singularity. The verified reference implementation is the CPU backend
// (Cpufit/lm_fit_cpp.cpp); this CUDA kernel mirrors that math but is not built/tested on the
// current (CPU-only) platform. See
// docs/project-management/projects/osipi-verification/STATUS.md.

__device__ void calculate_tissue_uptake (               // function name
	REAL const * parameters,
	int const n_fits,
	int const n_points,
	REAL * value,
	REAL * derivative,
	int const point_index,
	int const fit_index,
	int const chunk_index,
	char * user_info,							 // contains time and Cp values in 1 dimensional array
	std::size_t const user_info_size)
{
	// indices
	REAL* user_info_float = (REAL*)user_info;

	// split user_info array into time and Cp
	REAL* T = user_info_float;
	REAL* Cp = user_info_float + n_points;

	REAL const E = parameters[0];
	REAL const vp = parameters[1];
	REAL const Fp = parameters[2];
	REAL const eps = 1e-12;
	REAL const oneME = 1 - E;

	REAL * current_derivative = derivative + point_index;

	// Degenerate guard: keep value/derivatives finite (bounds normally keep E in (0,1)).
	if (!(E > eps) || !(E < 1 - eps) || !(vp > eps) || !(Fp > eps))
	{
		value[point_index] = 0;
		current_derivative[0 * n_points] = 0;
		current_derivative[1 * n_points] = 0;
		current_derivative[2 * n_points] = 0;
		return;
	}

	REAL const rp = Fp / (vp * oneME);

	// G = conv(Cp, exp(-rp*lag)); Gp = dG/drp; U = integral of Cp -- all evaluated at point_index.
	REAL G = 0;
	REAL Gp = 0;
	REAL U = 0;
	for (int i = 1; i <= point_index; i++)
	{
		REAL const spacing = T[i] - T[i - 1];
		REAL const lag = T[point_index] - T[i];
		REAL const lag_prev = T[point_index] - T[i - 1];
		REAL const e_i = exp(-rp * lag);
		REAL const e_p = exp(-rp * lag_prev);
		G  += 0.5 * spacing * (Cp[i] * e_i + Cp[i - 1] * e_p);
		Gp += 0.5 * spacing * (Cp[i] * (-lag) * e_i + Cp[i - 1] * (-lag_prev) * e_p);
		U  += 0.5 * spacing * (Cp[i] + Cp[i - 1]);
	}

	value[point_index] = E * Fp * U + Fp * oneME * G;
	current_derivative[0 * n_points] = Fp * U - Fp * G + Fp * rp * Gp;        // dC/dE
	current_derivative[1 * n_points] = -Fp * oneME * rp * Gp / vp;            // dC/dvp
	current_derivative[2 * n_points] = E * U + oneME * G + oneME * rp * Gp;   // dC/dFp
}
#endif
