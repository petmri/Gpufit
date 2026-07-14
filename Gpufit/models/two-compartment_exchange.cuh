#ifdef USE_2CXM
#define GPUFIT_TWO_COMPARTMENT_EXCHANGE_CUH_INCLUDED

// Reparameterized 2CXM: parameters [E = Ktrans/Fp, ve, vp, Fp].
//   PS = Fp*E/(1-E) (no pole / no `if (Ktrans>=Fp) PS=1e9` sentinel).
//   rp = (PS+Fp)/vp = 1/Tp,  re = PS/ve = 1/Te,  rb = Fp/vp = 1/Tb
//   a = rp+re,  Dr = sqrt(a^2 - 4*re*rb),  Kpos/Kneg = (a +/- Dr)/2,  Eneg = (Kpos-rb)/Dr
//   C = Fp*[ (1-Eneg)*Gpos + Eneg*Gneg ]   (Gpos,Gneg = exp-weighted integrals of Cp).
// Analytic Jacobian via the convolution rate-derivatives plus O(1) scalar partials of the
// internal rates (no numerical differences). The verified reference implementation is the
// CPU backend (Cpufit/lm_fit_cpp.cpp); this CUDA kernel mirrors that math but is not
// built/tested on the current (CPU-only) platform. See
// docs/project-management/projects/osipi-verification/STATUS.md.

__device__ void calculate_two_compartment_exchange (               // function name
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
	REAL const ve = parameters[1];
	REAL const vp = parameters[2];
	REAL const Fp = parameters[3];
	REAL const eps = 1e-12;
	REAL const oneME = 1 - E;

	REAL * current_derivative = derivative + point_index;

	// Internal rates and the two exponential poles (Kpos, Kneg) with the mixing weight Eneg.
	bool ok = (E > eps) && (E < 1 - eps) && (ve > eps) && (vp > eps) && (Fp > eps);
	REAL PS = 0, rp = 0, re = 0, rb = 0, a = 0, dr = 0, kpos = 0, kneg = 0, eneg = 0, dr_safe = eps;
	if (ok)
	{
		PS = Fp * E / oneME;
		rp = (PS + Fp) / vp;
		re = PS / ve;
		rb = Fp / vp;
		a = rp + re;
		REAL c = re * rb;
		REAL disc = a * a - 4 * c;
		if (!(disc > 0)) disc = 0;
		dr = sqrt(disc);
		kpos = 0.5 * (a + dr);
		kneg = 0.5 * (a - dr);
		dr_safe = (dr < eps) ? eps : dr;
		eneg = (kpos - rb) / dr_safe;
	}
	if (!ok)
	{
		value[point_index] = 0;
		current_derivative[0 * n_points] = 0;
		current_derivative[1 * n_points] = 0;
		current_derivative[2 * n_points] = 0;
		current_derivative[3 * n_points] = 0;
		return;
	}

	// Convolutions and their rate-derivatives at point_index (direct O(point_index) sums).
	REAL Gpos = 0, Gppos = 0, Gneg = 0, Gpneg = 0;
	for (int i = 1; i <= point_index; i++)
	{
		REAL const spacing = T[i] - T[i - 1];
		REAL const lag = T[point_index] - T[i];
		REAL const lag_prev = T[point_index] - T[i - 1];
		REAL const ep_i = exp(-kpos * lag);
		REAL const ep_p = exp(-kpos * lag_prev);
		REAL const en_i = exp(-kneg * lag);
		REAL const en_p = exp(-kneg * lag_prev);
		Gpos  += 0.5 * spacing * (Cp[i] * ep_i + Cp[i - 1] * ep_p);
		Gppos += 0.5 * spacing * (Cp[i] * (-lag) * ep_i + Cp[i - 1] * (-lag_prev) * ep_p);
		Gneg  += 0.5 * spacing * (Cp[i] * en_i + Cp[i - 1] * en_p);
		Gpneg += 0.5 * spacing * (Cp[i] * (-lag) * en_i + Cp[i - 1] * (-lag_prev) * en_p);
	}

	value[point_index] = Fp * ((1 - eneg) * Gpos + eneg * Gneg);

	// O(1) scalar partials of {rp,re,rb} -> {a,c,Dr,Kpos,Kneg,Eneg} for each parameter.
	// dPS/dparam is nonzero only for E and Fp.
	REAL const dps[4] = { Fp / (oneME * oneME), 0, 0, E / oneME };
	for (int p = 0; p < 4; p++)
	{
		REAL const is_ve = (p == 1) ? 1 : 0;
		REAL const is_vp = (p == 2) ? 1 : 0;
		REAL const is_fp = (p == 3) ? 1 : 0;
		REAL const drp = dps[p] / vp + is_fp / vp - is_vp * (PS + Fp) / (vp * vp);
		REAL const dre = dps[p] / ve - is_ve * PS / (ve * ve);
		REAL const drb = is_fp / vp - is_vp * Fp / (vp * vp);
		REAL const da = drp + dre;
		REAL const dc = dre * rb + re * drb;
		REAL const ddr = (a * da - 2 * dc) / dr_safe;
		REAL const dkpos = 0.5 * (da + ddr);
		REAL const dkneg = 0.5 * (da - ddr);
		REAL const deneg = ((dkpos - drb) * dr_safe - (kpos - rb) * ddr) / (dr_safe * dr_safe);
		REAL col = Fp * (deneg * (Gneg - Gpos) + (1 - eneg) * Gppos * dkpos + eneg * Gpneg * dkneg);
		if (p == 3)     // extra explicit-Fp term
			col += (1 - eneg) * Gpos + eneg * Gneg;
		current_derivative[p * n_points] = col;
	}
}
#endif
