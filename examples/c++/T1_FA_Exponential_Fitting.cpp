#include "../../Gpufit/gpufit.h"

#include <time.h>
#include <vector>
#include <random>
#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

void t1_fa_exponential_two()
{
	
	/*
	This example generates test data in form of 10000 one dimensional linear
	curves with the size of 20 data points per curve. It is noised by normal
	distributed noise. The initial guesses were randomized, within a specified
	range of the true value. The T1_FA_EXPONENTIAL model is fitted to the test data sets
	using the LSE estimator. The optional parameter user_info is used to pass
	custom x positions of the data sets. The same x position values are used for
	every fit.

	The console output shows
	- the ratio of converged fits including ratios of not converged fits for
	  different reasons,
	- the values of the true parameters and the mean values of the fitted
	  parameters including their standard deviation,
	- the mean chi square value
	- and the mean number of iterations needed.
	*/

	// start timer
	clock_t time_start, time_end;
	time_start = clock();

	// number of fits, fit points and parameters
	size_t const n_fits = 10000;
	size_t const n_points_per_fit = 5;
	size_t const n_model_parameters = 2;
	REAL snr = 10;

	// variable flip angle, in degrees. The model converts to radians itself
	// (T1_fa_fit.cuh), so passing pre-converted radians here double-converts them.
	REAL theta_degrees[] = { 2, 5, 10, 12, 15 };

	// Time resolution (constant)
	REAL tr[] =   {	21.572f, 21.572f, 21.572f, 21.572f, 21.572f };


	std::vector< REAL > user_info(2 * n_points_per_fit);
	for (size_t i = 0; i < n_points_per_fit; i++)
	{
		user_info[i] = static_cast<REAL>(theta_degrees[i]);
	}

	for (size_t i = n_points_per_fit; i < 2 * n_points_per_fit; i++)
	{
		user_info[i] = static_cast<REAL>(tr[i - n_points_per_fit]);
	}

	// size of user info in bytes
	size_t const user_info_size = n_model_parameters * n_points_per_fit * sizeof(REAL);

	// initialize random number generator
	std::mt19937 rng;
	rng.seed(time(NULL));
	std::uniform_real_distribution< REAL > uniform_dist(0, 1);
	std::normal_distribution< REAL > normal_dist(0, 1);

	// true parameters
	std::vector< REAL > true_parameters{ 1, 1800 };		// a, t1

	// initial parameters (randomized)
	std::vector< REAL > initial_parameters(n_fits * n_model_parameters);
	for (size_t i = 0; i != n_fits; i++)
	{
		// random a
		initial_parameters[i * n_model_parameters + 0] = true_parameters[0] * (0.1f + 1.8f * uniform_dist(rng));
		// random t1
		initial_parameters[i * n_model_parameters + 1] = true_parameters[1] * (0.1f + 1.8f * uniform_dist(rng));
	}

	// generate data
	std::vector< REAL > data(n_points_per_fit * n_fits);
	REAL mean_y = 0;
	for (size_t i = 0; i != data.size(); i++)
	{
		size_t j = i / n_points_per_fit; // the fit
		size_t k = i % n_points_per_fit; // the position within a fit
//		REAL x = 0;
//		for (int n = 1; n < k; n++) {
//
//			REAL spacing = theta[n] - theta[n - 1];
//			x += (tr[n - 1] + tr[n]) / 2 * spacing;
//		}
//		REAL y = true_parameters[0] * x + true_parameters[1] * tr[k];
		REAL const theta_rad = theta_degrees[k] * static_cast<REAL>(M_PI) / 180.0f;
		REAL y = true_parameters[0] * ( (1 - exp(-tr[k]/true_parameters[1])) * sin(theta_rad)) / (1 - exp(-tr[k]/true_parameters[1]) * cos(theta_rad) );
		//data[i] = y + normal_dist(rng);
		//data[i] = y * (0.2f + 1.6f * uniform_dist(rng));
		data[i] = y;
		mean_y += y;
	}
	mean_y = mean_y / data.size();
	std::normal_distribution<REAL> norm_snr(0,mean_y/snr);
	for (size_t i = 0; i != data.size(); i++)
	{
		data[i] = data[i] + norm_snr(rng);
	}

	// tolerance
	REAL const tolerance = 10e-12f;

	// maximum number of iterations
	int const max_number_iterations = 200;

	// estimator ID
	int const estimator_id = LSE;

	// model ID
	int const model_id = T1_FA_EXPONENTIAL;

	// parameters to fit (all of them)
	std::vector< int > parameters_to_fit(n_model_parameters, 1);

	// output parameters
	std::vector< REAL > output_parameters(n_fits * n_model_parameters);
	std::vector< int > output_states(n_fits);
	std::vector< REAL > output_chi_square(n_fits);
	std::vector< int > output_number_iterations(n_fits);

	// call to gpufit (C interface)
	int const status = gpufit
	(
		n_fits,
		n_points_per_fit,
		data.data(),
		0,
		model_id,
		initial_parameters.data(),
		tolerance,
		max_number_iterations,
		parameters_to_fit.data(),
		estimator_id,
		user_info_size,
		reinterpret_cast< char* >( user_info.data() ),
		output_parameters.data(),
		output_states.data(),
		output_chi_square.data(),
		output_number_iterations.data()
	);


	// check status
	if (status != ReturnState::OK)
	{
		throw std::runtime_error(gpufit_get_last_error());
	}


	// get fit states
	std::vector< int > output_states_histogram(5, 0);
	for (std::vector< int >::iterator it = output_states.begin(); it != output_states.end(); ++it)
	{
		output_states_histogram[*it]++;
	}

	std::cout << "ratio converged              " << (REAL)output_states_histogram[0] / n_fits << "\n";
	std::cout << "ratio max iteration exceeded " << (REAL)output_states_histogram[1] / n_fits << "\n";
	std::cout << "ratio singular hessian       " << (REAL)output_states_histogram[2] / n_fits << "\n";
	std::cout << "ratio neg curvature MLE      " << (REAL)output_states_histogram[3] / n_fits << "\n";
	std::cout << "ratio gpu not read           " << (REAL)output_states_histogram[4] / n_fits << "\n";

	// compute mean fitted parameters for converged fits
	std::vector< REAL > output_parameters_mean(n_model_parameters, 0);
	std::vector< REAL > output_parameters_mean_error(n_model_parameters, 0);
	for (size_t i = 0; i != n_fits; i++)
	{
		if (output_states[i] == FitState::CONVERGED)
		{
			// add a
			output_parameters_mean[0] += output_parameters[i * n_model_parameters + 0];
			// add T1
			output_parameters_mean[1] += output_parameters[i * n_model_parameters + 1];
			// add a
			output_parameters_mean_error[0] += abs(output_parameters[i * n_model_parameters + 0]-true_parameters[0]);
			// add T1
			output_parameters_mean_error[1] += abs(output_parameters[i * n_model_parameters + 1]-true_parameters[1]);
		}
	}
	output_parameters_mean[0] /= output_states_histogram[0];
	output_parameters_mean[1] /= output_states_histogram[0];

	// compute std of fitted parameters for converged fits
	std::vector< REAL > output_parameters_std(n_model_parameters, 0);
	for (size_t i = 0; i != n_fits; i++)
	{
		if (output_states[i] == FitState::CONVERGED)
		{
			// add squared deviation for a
			output_parameters_std[0] += (output_parameters[i * n_model_parameters + 0] - output_parameters_mean[0]) * (output_parameters[i * n_model_parameters + 0] - output_parameters_mean[0]);
			// add squared deviation for T1
			output_parameters_std[1] += (output_parameters[i * n_model_parameters + 1] - output_parameters_mean[1]) * (output_parameters[i * n_model_parameters + 1] - output_parameters_mean[1]);
		}
	}
	// divide and take square root
	output_parameters_std[0] = sqrt(output_parameters_std[0] / output_states_histogram[0]);
	output_parameters_std[1] = sqrt(output_parameters_std[1] / output_states_histogram[0]);

	// print mean and std
	std::cout << "a  true " << true_parameters[0] << " mean " << output_parameters_mean[0] << " std " << output_parameters_std[0] << "\n";
	std::cout << "T1	true " << true_parameters[1] << " mean " << output_parameters_mean[1] << " std " << output_parameters_std[1] << "\n";

	// compute mean chi-square for those converged
	REAL  output_chi_square_mean = 0;
	for (size_t i = 0; i != n_fits; i++)
	{
		if (output_states[i] == FitState::CONVERGED)
		{
			output_chi_square_mean += output_chi_square[i];
		}
	}
	output_chi_square_mean /= static_cast<REAL>(output_states_histogram[0]);
	std::cout << "mean chi square " << output_chi_square_mean << "\n";

	// compute mean number of iterations for those converged
	REAL  output_number_iterations_mean = 0;
	for (size_t i = 0; i != n_fits; i++)
	{
		if (output_states[i] == FitState::CONVERGED)
		{
			output_number_iterations_mean += static_cast<REAL>(output_number_iterations[i]);
		}
	}

	// normalize
	output_number_iterations_mean /= static_cast<REAL>(output_states_histogram[0]);
	std::cout << "mean number of iterations " << output_number_iterations_mean << "\n";

	// time
	//time(&time_end);
	time_end = clock();
	double time_taken_sec = double(time_end-time_start)/double(CLOCKS_PER_SEC);
	std::cout << "execution time for " << n_fits << " fits was " << time_taken_sec << " seconds\n";
}


int main(int argc, char* argv[])
{
	std::cout << std::endl << "Beginning T1 FA Exponential fit..." << std::endl;
	t1_fa_exponential_two();

	std::cout << std::endl << "T1 exponential fit completed!" << std::endl;

	return 0;
}
