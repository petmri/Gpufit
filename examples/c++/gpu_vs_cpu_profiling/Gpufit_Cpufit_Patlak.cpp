#include "Cpufit/cpufit.h"
#include "Gpufit/gpufit.h"
#include "tests/utils.h"

#include <stdexcept>
#include <array>
#include <vector>
#include <iostream>
#include <iomanip>
#include <random>
#include <numeric>
#include <chrono>
#include <string>

#define _USE_MATH_DEFINES
#include <math.h>

#ifdef USE_CUBLAS
#define PRINT_IF_USE_CUBLAS() \
        std::cout << "CUBLAS enabled: Yes" << std::endl << std::endl
#else
#define PRINT_IF_USE_CUBLAS() \
        std::cout << "CUBLAS enabled: No" << std::endl << std::endl
#endif // USE_CUBLAS

// consider condensing main into functions

/*
    Names of parameters for the Patlak model
*/
struct Parameters
{
    REAL Ktrans;
    REAL vp;
};

/*
    Randomize parameters, slightly differently
*/
void generate_initial_parameters(std::vector<REAL> & parameters_set, std::vector<Parameters> const & parameters)
{
    std::uniform_real_distribution< REAL> uniform_dist(0, 1);

    REAL const a = 0.9f;
    REAL const b = 0.2f;

    int const n_parameters = sizeof(Parameters) / sizeof(REAL);
    for (std::size_t i = 0; i < parameters_set.size() / n_parameters; i++)
    {
        parameters_set[0 + i * n_parameters] = parameters[i].Ktrans * (a + b * uniform_dist(rng));
        parameters_set[1 + i * n_parameters] = parameters[i].vp * (a + b * uniform_dist(rng));
    }
}

/*
    Randomize parameters
*/
void generate_test_parameters(std::vector<Parameters> & target, Parameters const source)
{
    std::size_t const n_fits = target.size();

    std::uniform_real_distribution< REAL> uniform_dist(0, 1);

    REAL const a = 0.9f;
    REAL const b = 0.2f;

    int const text_width = 30;
    int const progress_width = 25;

    std::cout << std::setw(text_width) << " ";
    for (int i = 0; i < progress_width; i++)
        std::cout << "-";
    std::cout << std::endl;
    std::cout << std::setw(text_width) << std::left << "Generating test parameters";

    std::size_t progress = 0;

    for (std::size_t i = 0; i < n_fits; i++)
    {
        target[i].Ktrans = source.Ktrans * (a + b * uniform_dist(rng));
        target[i].vp = source.vp * (a + b * uniform_dist(rng));

        progress += 1;
        if (progress >= n_fits / progress_width)
        {
            progress = 0;
            std::cout << "|";
        }
    }

    std::cout << std::endl;
    std::cout << std::setw(text_width) << " ";
    for (int i = 0; i < progress_width; i++)
        std::cout << "-";
    std::cout << std::endl;
}

/*
Runs Gpufit vs. Cpufit for various number of fits and compares the speed

No weights, Model: Gauss_2D, Estimator: LSE
Model: Patlak, Estimator: LSE
*/
int main(int argc, char * argv[])
{
    // title 
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Patlak Model Performance Comparison" << std::endl;
    std::cout << "----------------------------------------" << std::endl << std::endl;

    std::cout << "Please note that execution speed test results depend on" << std::endl;
    std::cout << "the details of the CPU and GPU hardware." << std::endl;
    std::cout << std::endl;


    // check for CUDA availability
	bool const cuda_available = gpufit_cuda_available() != 0;
	if (!gpufit_cuda_available())
	{
		std::cout << "CUDA not available" << std::endl;
	}

	// check for CUDA runtime and driver
    int cuda_runtime_version = 0;
    int cuda_driver_version = 0;
    bool const version_available = gpufit_get_cuda_version(&cuda_runtime_version, &cuda_driver_version) == ReturnState::OK;
    int const cuda_runtime_major = cuda_runtime_version / 1000;
    int const cuda_runtime_minor = cuda_runtime_version % 1000 / 10;
    int const cuda_driver_major = cuda_driver_version / 1000;
    int const cuda_driver_minor = cuda_driver_version % 1000 / 10;

    bool do_gpufits = false;
    if (cuda_available & version_available)
    {
        std::cout << "CUDA runtime version: ";
        std::cout << cuda_runtime_major << "." << cuda_runtime_minor << std::endl;
        std::cout << "CUDA driver version:  ";
        std::cout << cuda_driver_major << "." << cuda_driver_minor << std::endl;
        std::cout << std::endl;

        bool const cuda_available = cuda_driver_version > 0;
        if (cuda_available)
        {
            bool const version_compatible
                = cuda_driver_version >= cuda_runtime_version
                && cuda_runtime_version > 0;
            if (version_compatible)
            {
                do_gpufits = true;
            }
            else
            {
                std::cout << "The CUDA runtime version is not compatible with the" << std::endl;
                std::cout << "current graphics driver. Please update the driver, or" << std::endl;
                std::cout << "re - build Gpufit from source using a compatible version" << std::endl;
                std::cout << "of the CUDA toolkit." << std::endl;
                std::cout << std::endl;
            }
        }
        else
        {
            std::cout << "No CUDA enabled graphics card detected." << std::endl;
            std::cout << std::endl;
        }
    }
    else
    {
        std::cout << "CUDA error detected. Error string: ";
        std::cout << gpufit_get_last_error() << std::endl;
        std::cout << std::endl;
    }
    if (!do_gpufits)
    {
        std::cout << "Skipping Gpufit computations." << std::endl << std::endl;
    }
    else
    {
        PRINT_IF_USE_CUBLAS();
    }

    // all numbers of fits
    std::vector<std::size_t> n_fits_all;
    if (sizeof(void*) < 8)
    {
        n_fits_all = { 10, 100, 1000, 10000, 100000, 1000000};
    }
    else
    {
        n_fits_all = { 10, 100, 1000, 10000, 100000, 1000000, 10000000 };
    }

    std::size_t const max_n_fits = n_fits_all.back();

    // fit parameters constant for every run
    // std::size_t const size_x = 5;
    std::size_t const n_points = 60;
    std::size_t const n_parameters = 2;
    std::vector<int> parameters_to_fit(n_parameters, 1);
    REAL const tolerance = 0.0001f;
    int const max_n_iterations = 200;
    const REAL snr = 0.8;

    // initial parameters
    Parameters true_parameters;
    true_parameters.Ktrans = 0.05;
    true_parameters.vp = .03;


    // test parameters
    std::vector<Parameters> test_parameters(max_n_fits);
    generate_test_parameters(test_parameters, true_parameters);

    //  test data
    std::vector<REAL> user_info(n_parameters * n_points);
    std::vector<REAL> data(max_n_fits * n_points);
    // generate_patlak(max_n_fits, n_points, data, test_parameters, user_info);
    // custom x positions for the data points of every fit, stored in user info
	// time independent variable, given in minutes
	REAL timeX[] = { 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5,
					5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9, 9.25, 9.5, 9.75, 10,
					10.25, 10.5, 10.75, 11, 11.25, 11.5, 11.75, 12, 12.25, 12.5, 12.75, 13, 13.25, 13.5, 13.75, 14, 14.25, 14.5, 14.75, 15 };

	// Concentration of plasma (independent), at 1 min based on equation: Cp(t) = 5.5e^(-.6t)
	REAL Cp[] =   {	0.0f, 0.0f, 0.0f, 3.01846399851715f, 2.59801604007558f, 2.2361331285733f, 1.92465762011135f, 1.65656816551711f, 1.4258214335524f,
					1.22721588081636f, 1.05627449741415f, 0.909143885218726f, 0.782507393725825f, 0.6735103553914f, 0.579695735090254f,
					0.498948743091769f, 0.429449163006342f, 0.369630320068624f, 0.318143764811612f, 0.273828876023252f, 0.235686697768721f,
					0.20285742070682f, 0.174601000079374f, 0.150280473460109f, 0.12934760220805f, 0.111330512951924f, 0.0958230605172143f,
					0.0824756725126274f, 0.0709874691926393f, 0.0610994809603327f, 0.0525888106179893f, 0.0452636087696102f, 0.0389587491097867f,
					0.033532106110336f, 0.0288613511954976f, 0.0248411951843697f, 0.0213810148391187f, 0.018402810016092f, 0.0158394453694853f,
					0.013633136971665f, 0.0117341497352074f, 0.010099676273659f, 0.00869287192804919f, 0.00748202420651342f, 0.00643983791435146f,
					0.00554281985976681f, 0.00477074926518851f, 0.00410622194607174f, 0.00353425798195557f, 0.00304196403581309f, 0.00261824270962248f, 
					0.00225354238438883f, 0.00193964190545541f, 0.00166946525943377f, 0.00143692206515917f, 0.00123677028298367f, 0.00106449804756952f,
					0.000916221960431984f, 0.000788599549519612f, 0.000678753922476738f };

	for (size_t i = 0; i < n_points; i++)
	{
		user_info[i] = static_cast<REAL>(timeX[i]);
	}

	for (size_t i = n_points; i < 2 * n_points; i++)
	{
		user_info[i] = static_cast<REAL>(Cp[i - n_points]);
	}
    // size of user info in bytes
	size_t const user_info_size = 2 * n_points * sizeof(REAL);

	// generate data
	REAL mean_y = 0;
	for (size_t i = 0; i != data.size(); i++)
	{
		size_t j = i / n_points; // the fit
		size_t k = i % n_points; // the position within a fit
		REAL x = 0;
		// n <= k to match the model's own loop bound (patlak.cuh)
		for (int n = 1; n <= (int)k; n++) {

			REAL spacing = timeX[n] - timeX[n - 1];
			x += (Cp[n - 1] + Cp[n]) / 2 * spacing;
		}
		REAL y = true_parameters.Ktrans * x + true_parameters.vp * Cp[k];
		//data[i] = y + normal_dist(rng);
		//data[i] = y * (0.2f + 1.6f * uniform_dist(rng));
		data[i] = y;
		mean_y += y;
		//std::cout << data[i] << std::endl;
	}
	mean_y = mean_y / data.size();
	std::normal_distribution<REAL> norm_snr(0,mean_y/snr);
	for (size_t i = 0; i != data.size(); i++)
	{
		data[i] = data[i] + norm_snr(rng);
	}
    // add_gauss_noise(data, true_parameters, 10);

    // initial parameter set
    std::vector<REAL> initial_parameters(n_parameters * max_n_fits);
    generate_initial_parameters(initial_parameters, test_parameters);

    // print column identifiers
    std::cout << "Delta values are GPU mean - CPU mean.\n";
    std::cout << std::endl << std::right;
    std::cout << std::setw(8) << "Number" << std::setw(3) << "|";
    std::cout << std::setw(9) << "Cpufit" << std::setw(3) << "|";
    std::cout << std::setw(9) << "Gpufit" << std::setw(3) << "|";
    std::cout << std::setw(9) << "Perf gain" << std::setw(2) << "|";
    for (int i = 0; i < n_parameters; i++)
        std::cout << std::setw(12) << "Delta Param" << i << std::setw(3) << "|";
    std::cout << std::setw(13) << "Delta Chi Sq" << std::setw(3) << "|";
    std::cout << std::setw(13) << "Delta n Iter" << std::setw(3) << "|";
    std::cout << std::setw(13) << "Converge";
    std::cout << std::endl;
    std::cout << std::setw(8) << "of fits" << std::setw(3) << "|";
    std::cout << std::setw(10) << "(fits/s)" << std::setw(2) << "|";
    std::cout << std::setw(10) << "(fits/s)" << std::setw(2) << "|";
    std::cout << std::setw(9) << "factor" << std::setw(2) << "|";
    // for (int i = 0; i < n_parameters; i++)
    //     std::cout << std::setw(16) << "|";
    std::cout << std::setw(10) << "Ktrans" << std::setw(6) << "|";
    std::cout << std::setw(9) << "vp" << std::setw(7) << "|";
    std::cout << std::setw(16) << "|";
    std::cout << std::setw(16) << "|";
    std::cout << std::setw(13) << " Ratio (CPU GPU)";
    std::cout << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------------------------------------------------------------------";
    std::cout << std::endl;

    // loop over number of fits
    for (std::size_t fit_index = 0; fit_index < n_fits_all.size(); fit_index++)
    {
        // number of fits
        std::size_t n_fits = n_fits_all[fit_index];
        std::cout << std::setw(8) << n_fits << std::setw(3) << "|";

        // Cpufit output
        std::vector<REAL> cpufit_parameters(n_fits * n_parameters);
        std::vector<int> cpufit_states(n_fits);
        std::vector<REAL> cpufit_chi_squares(n_fits);
        std::vector<int> cpufit_n_iterations(n_fits);

        // run Cpufit and measure time
        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
        int const cpu_status
            = cpufit
            (
                n_fits,
                n_points,
                data.data(),
                0,
                PATLAK,
                initial_parameters.data(),
                tolerance,
                max_n_iterations,
                parameters_to_fit.data(),
                LSE,
                user_info_size,
                reinterpret_cast< char* >( user_info.data() ),
                cpufit_parameters.data(),
                cpufit_states.data(),
                cpufit_chi_squares.data(),
                cpufit_n_iterations.data()
            );
        std::chrono::milliseconds::rep const dt_cpufit = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t0).count();

        if (cpu_status != 0)
        {
            // error in cpufit, should actually not happen
            std::cout << "Error in cpufit: " << cpufit_get_last_error() << std::endl;
        }

        std::chrono::milliseconds::rep dt_gpufit = 0;

        // Gpufit output parameters
        std::vector<REAL> gpufit_parameters(n_fits * n_parameters);
        std::vector<int> gpufit_states(n_fits);
        std::vector<REAL> gpufit_chi_squares(n_fits);
        std::vector<int> gpufit_n_iterations(n_fits);

        // if we do not do gpufit, we skip the rest of the loop
        if (do_gpufits)
        {
            // run Gpufit and measure time
            t0 = std::chrono::high_resolution_clock::now();
            int const gpu_status
                = gpufit
                (
                n_fits,
                n_points,
                data.data(),
                0,
                PATLAK,
                initial_parameters.data(),
                tolerance,
                max_n_iterations,
                parameters_to_fit.data(),
                LSE,
                user_info_size,
                reinterpret_cast< char* >( user_info.data() ),
                gpufit_parameters.data(),
                gpufit_states.data(),
                gpufit_chi_squares.data(),
                gpufit_n_iterations.data()
                );
            dt_gpufit = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t0).count();

            if (gpu_status != 0)
            {
                // error in gpufit
                std::cout << "Error in gpufit: " << gpufit_get_last_error() << std::endl;
                do_gpufits = false;
            }
        }
        // get fit states
        std::vector< int > cpu_output_states_histogram(5, 0);
        std::vector< int > gpu_output_states_histogram(5, 0);
        for (std::vector< int >::iterator it = cpufit_states.begin(); it != cpufit_states.end(); ++it)
        {
            cpu_output_states_histogram[*it]++;
        }
        for (std::vector< int >::iterator it = gpufit_states.begin(); it != gpufit_states.end(); ++it)
        {
            gpu_output_states_histogram[*it]++;
        }
        // std::cout << "CPU ratio converged          " << (REAL)cpu_output_states_histogram[0] / n_fits << "\n";
        // std::cout << "ratio max iteration exceeded " << (REAL)cpu_output_states_histogram[1] / n_fits << "\n";
        // std::cout << "ratio singular hessian       " << (REAL)cpu_output_states_histogram[2] / n_fits << "\n";
        // std::cout << "ratio neg curvature MLE      " << (REAL)cpu_output_states_histogram[3] / n_fits << "\n";
        // std::cout << "ratio cpu not read           " << (REAL)cpu_output_states_histogram[4] / n_fits << "\n";
        
        // std::cout << "GPU ratio converged          " << (REAL)gpu_output_states_histogram[0] / n_fits << "\n";
        // std::cout << "ratio max iteration exceeded " << (REAL)gpu_output_states_histogram[1] / n_fits << "\n";
        // std::cout << "ratio singular hessian       " << (REAL)gpu_output_states_histogram[2] / n_fits << "\n";
        // std::cout << "ratio neg curvature MLE      " << (REAL)gpu_output_states_histogram[3] / n_fits << "\n";
        // std::cout << "ratio gpu not read           " << (REAL)gpu_output_states_histogram[4] / n_fits << "\n";

        // compute mean fitted parameters for converged fits
        std::vector< double > cpu_output_parameters_mean(n_parameters, 0);
        std::vector< double > cpu_output_parameters_mean_error(n_parameters, 0);
        std::vector< double > gpu_output_parameters_mean(n_parameters, 0);
        std::vector< double > gpu_output_parameters_mean_error(n_parameters, 0);
        for (size_t i = 0; i != n_fits; i++)
        {
            if (cpufit_states[i] == FitState::CONVERGED)
            {
                // add Ktrans
                cpu_output_parameters_mean[0] += cpufit_parameters[i * n_parameters + 0];
                // add vp
                cpu_output_parameters_mean[1] += cpufit_parameters[i * n_parameters + 1];
                // add Ktrans
                cpu_output_parameters_mean_error[0] += abs(cpufit_parameters[i * n_parameters + 0]-true_parameters.Ktrans);
                // add vp
                cpu_output_parameters_mean_error[1] += abs(cpufit_parameters[i * n_parameters + 1]-true_parameters.vp);
            }

            if (gpufit_states[i] == FitState::CONVERGED)
            {
                // add Ktrans
                gpu_output_parameters_mean[0] += gpufit_parameters[i * n_parameters + 0];
                // add vp
                gpu_output_parameters_mean[1] += gpufit_parameters[i * n_parameters + 1];
                // add Ktrans
                gpu_output_parameters_mean_error[0] += abs(gpufit_parameters[i * n_parameters + 0]-true_parameters.Ktrans);
                // add vp
                gpu_output_parameters_mean_error[1] += abs(gpufit_parameters[i * n_parameters + 1]-true_parameters.vp);
            }   
        }
        cpu_output_parameters_mean[0] /= cpu_output_states_histogram[0];
        cpu_output_parameters_mean[1] /= cpu_output_states_histogram[0];
        gpu_output_parameters_mean[0] /= gpu_output_states_histogram[0];
        gpu_output_parameters_mean[1] /= gpu_output_states_histogram[0];

        // compute std of fitted parameters for converged fits
        std::vector< double > cpu_output_parameters_std(n_parameters, 0);
        std::vector< double > gpu_output_parameters_std(n_parameters, 0);
        for (size_t i = 0; i != n_fits; i++)
        {
            if (cpufit_states[i] == FitState::CONVERGED)
            {
                // add squared deviation for Ktrans
                cpu_output_parameters_std[0] += (cpufit_parameters[i * n_parameters + 0] - cpu_output_parameters_mean[0]) * (cpufit_parameters[i * n_parameters + 0] - cpu_output_parameters_mean[0]);
                // add squared deviation for vp
                cpu_output_parameters_std[1] += (cpufit_parameters[i * n_parameters + 1] - cpu_output_parameters_mean[1]) * (cpufit_parameters[i * n_parameters + 1] - cpu_output_parameters_mean[1]);
            }
            if (gpufit_states[i] == FitState::CONVERGED)
            {
                // add squared deviation for Ktrans
                gpu_output_parameters_std[0] += (gpufit_parameters[i * n_parameters + 0] - gpu_output_parameters_mean[0]) * (gpufit_parameters[i * n_parameters + 0] - gpu_output_parameters_mean[0]);
                // add squared deviation for vp
                gpu_output_parameters_std[1] += (gpufit_parameters[i * n_parameters + 1] - gpu_output_parameters_mean[1]) * (gpufit_parameters[i * n_parameters + 1] - gpu_output_parameters_mean[1]);
            }

        }
        // divide and take square root
        cpu_output_parameters_std[0] = sqrt(cpu_output_parameters_std[0] / cpu_output_states_histogram[0]);
        cpu_output_parameters_std[1] = sqrt(cpu_output_parameters_std[1] / cpu_output_states_histogram[0]);
        gpu_output_parameters_std[0] = sqrt(gpu_output_parameters_std[0] / gpu_output_states_histogram[0]);
        gpu_output_parameters_std[1] = sqrt(gpu_output_parameters_std[1] / gpu_output_states_histogram[0]);

        // print mean and std
        // std::cout << "Ktrans  true " << true_parameters[0] << " mean " << output_parameters_mean[0] << " std " << output_parameters_std[0] << "\n";
        // std::cout << "vp	true " << true_parameters[1] << " mean " << output_parameters_mean[1] << " std " << output_parameters_std[1] << "\n";

        // compute mean chi-square for those converged
        double cpu_output_chi_square_mean = 0;
        double gpu_output_chi_square_mean = 0;
        for (size_t i = 0; i != n_fits; i++)
        {
            if (cpufit_states[i] == FitState::CONVERGED)
            {
                cpu_output_chi_square_mean += cpufit_chi_squares[i];
            }

            if (gpufit_states[i] == FitState::CONVERGED)
                gpu_output_chi_square_mean += gpufit_chi_squares[i];
        }
        cpu_output_chi_square_mean /= static_cast<REAL>(cpu_output_states_histogram[0]);
        gpu_output_chi_square_mean /= static_cast<REAL>(gpu_output_states_histogram[0]);
        // std::cout << "mean chi square " << cpu_output_chi_square_mean << "\n";

        // compute mean number of iterations for those converged
        double cpu_output_number_iterations_mean = 0;
        double gpu_output_number_iterations_mean = 0;
        for (size_t i = 0; i != n_fits; i++)
        {
            if (cpufit_states[i] == FitState::CONVERGED)
            {
                cpu_output_number_iterations_mean += static_cast<REAL>(cpufit_n_iterations[i]);
            }

            if (gpufit_states[i] == FitState::CONVERGED)
            {
                gpu_output_number_iterations_mean += static_cast<REAL>(gpufit_n_iterations[i]);
            }
        }

        // normalize
        cpu_output_number_iterations_mean /= static_cast<REAL>(cpu_output_states_histogram[0]);
        gpu_output_number_iterations_mean /= static_cast<REAL>(gpu_output_states_histogram[0]);
        // std::cout << "mean number of iterations " << output_number_iterations_mean << "\n";

        // print the calculation speed in fits/s
        std::cout << std::fixed << std::setprecision(0);
        if (dt_cpufit)
        {
            std::cout << std::setw(9) << static_cast<double>(n_fits) / static_cast<double>(dt_cpufit)* 1000.0 << std::setw(3) << "|";
        }
        else
        {
            std::cout << std::setw(9) << "inf" << std::setw(3) << "|";
        }
        if (dt_gpufit)
        {
            std::cout << std::setw(9) << static_cast<double>(n_fits) / static_cast<double>(dt_gpufit)* 1000.0 << std::setw(3) << "|";
            std::cout << std::fixed << std::setprecision(2);
            std::cout << std::setw(8) << static_cast<double>(dt_cpufit) / static_cast<double>(dt_gpufit) << std::setw(3) << "|";
            std::cout << std::fixed << std::setprecision(9);
            // for (int i = 0; i < n_parameters; i++)
            //     std::cout << std::setw(13) << *(gpufit_parameters.data()+fit_index*n_parameters+i)-*(cpufit_parameters.data()+fit_index*n_parameters+i) << std::setw(3) << "|";
            for (int i = 0; i < n_parameters; i++)
                std::cout << std::setw(13) << gpu_output_parameters_mean[i]-cpu_output_parameters_mean[i] << std::setw(3) << "|";
            // std::cout << std::setw(13) << *(gpufit_chi_squares.data()+fit_index)-*(cpufit_chi_squares.data()+fit_index) << std::setw(3) << "|";
            // std::cout << std::setw(7) << *(gpufit_n_iterations.data()+fit_index)-*(cpufit_n_iterations.data()+fit_index);
            std::cout << std::setw(13) << gpu_output_chi_square_mean-cpu_output_chi_square_mean << std::setw(3) << "|";
            std::cout << std::setw(13) << gpu_output_number_iterations_mean-cpu_output_number_iterations_mean << std::setw(3) << "|";
            std::cout << std::setw(12) << (REAL)cpu_output_states_histogram[0] / n_fits << std::setw(3);
            std::cout << std::setw(12) << (REAL)gpu_output_states_histogram[0] / n_fits;
        }
        else if (!do_gpufits)
        {
            std::cout << std::setw(13) << "--" << std::setw(3) << "|";
            std::cout << std::setw(12) << "--";
        }
        else
        {
            std::cout << std::setw(9) << "inf" << std::setw(3) << "|";
            std::cout << std::setw(8) << "inf" << std::setw(3) << "|";
            std::cout << std::fixed << std::setprecision(9);
            for (int i = 0; i < n_parameters; i++)
                std::cout << std::setw(13) << gpu_output_parameters_mean[i]-cpu_output_parameters_mean[i] << std::setw(3) << "|";
            std::cout << std::setw(13) << gpu_output_chi_square_mean-cpu_output_chi_square_mean << std::setw(3) << "|";
            std::cout << std::setw(13) << gpu_output_number_iterations_mean-cpu_output_number_iterations_mean << std::setw(3) << "|";
            std::cout << std::setw(12) << (REAL)cpu_output_states_histogram[0] / n_fits << std::setw(3);
            std::cout << std::setw(12) << (REAL)gpu_output_states_histogram[0] / n_fits;
        }
        
        std::cout << std::endl;        
    }
    std::cout << std::endl << "Test completed!" << std::endl;
    std::cout << "Press ENTER to exit" << std::endl;
    std::getchar();

    return 0;
}
