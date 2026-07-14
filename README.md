# Gpufit

Levenberg Marquardt curve fitting in CUDA with several MRI relevant models.

This fork has been updated to support newer compilers and CUDA versions, and includes additional MRI-relevant models (Patlak, Tofts, Extended Tofts, Tissue Uptake, 2CXM, T1 FA Exponential). It also includes binaries for windows, linux and mac. The original Gpufit repository is referenced below along with the relevant citation.

Homepage for original project: [github.com/gpufit/Gpufit](https://github.com/gpufit/Gpufit)

The manuscript describing Gpufit is now published in [Scientific Reports](https://www.nature.com/articles/s41598-017-15313-9).

## Changelog

### v1.5.0

- **Fixed silent wrong-but-"converged" fits in the constrained solver** — the constrained Levenberg-Marquardt loop could report `CONVERGED` on a rejected backtracking step, freezing stiff models (2CXM/2CUM) at implausible parameter values with no error flag. The CPU backtracking line search was also ported to the CUDA solver so constrained fits behave the same on both backends.
- **2CXM/2CUM reparameterized to fit `E = Ktrans/Fp`** — removes the `Ktrans -> Fp` pole improving convergence. Replaces the numerical Jacobian with analytic derivatives, faster and improved convergence. Fixes the residual low-flow convergence failures,
- **O(N) convolution for MRI Models** — the MRI models now use a custom O(N) convolution for the AIF instead of a naive O(N^2) implementation. This is especially important for 2CXM/2CUM, and is roughly 3000x faster in worst case scenarios when fitting 2CXM.
- **Fixed a MLE (Poisson) fitting bug** — the gradient/Hessian calculations divided data by the model value without guarding the case where both are legitimately zero (e.g. an AIF's pre-bolus samples), corrupting every fit with `NaN`. Fixed in both `Cpufit` and `Gpufit` for parity.
- **New closed-form linear Patlak solver** — `Cpufit`'s Patlak fit now supports `patlak_algorithm='bounded_linear'` (LSE only) as a fast alternative to the iterative LM solver.
- **Improved MRI example and test robustness** — fixed parameter-recovery bugs across the C++ MRI fitting examples, and parity test, and fixed/enabled the Patlak Boost test.
- **CI/release coverage expanded** — added Windows CUDA 13.2 builds, and pinned explicit macOS deployment targets at OS11.
- **Release artifacts are now execution-smoke-tested** — `pycpufit` is exercised with a real fit (not just an import check) against both the build tree and the final packaged artifact before publishing.

### v1.4.1

- **Complete Cpufit MRI model coverage** — all six MRI pharmacokinetic models (Patlak, Tofts, Extended Tofts, Tissue Uptake, 2CXM, T1 FA Exponential) are now available in both GPU (`Gpufit`) and CPU (`Cpufit`) libraries with full parity.
- **Fixed Patlak CPU derivative bug** — the trapezoidal integral in `calc_derivatives_patlak` accumulated across data points instead of resetting per point, producing incorrect gradients and convergence failures.
- **GPU/CPU MRI parity test** — new `Gpufit_Cpufit_MRI_Parity` example validates that all six MRI models produce matching results between GPU and CPU.
- **Updated CUDA architecture support** — added Hopper (sm_90) and Blackwell (sm_100, sm_120) architectures; always emits PTX for the highest architecture to guarantee forward compatibility with future GPUs.
- **Automated CI/CD releases** — GitHub Actions workflows for continuous integration and automated binary releases (tag-triggered and manual dev builds) across Windows (CUDA 12.4, 12.8), Linux (CUDA 11.8, 12.5, 13.0), and macOS (CPU-only x64 and arm64).

## Quick start instructions

The release artifacts provide library and wrapper payloads. To run GPU-vs-CPU performance checks, build and run the profiling examples from source (for example `examples/c++/gpu_vs_cpu_profiling/Gpufit_Cpufit_Performance_Comparison.cpp`).

## Binary distribution

The latest Gpufit binary release can be found on the [release page](https://github.com/ironictoo/Gpufit/releases).

Current automated release artifacts and runtime requirements:

| Artifact | Includes | Built on CI runner | Runtime requirements |
|---|---|---|---|
| `windows-x64-cuda12.4` | `Gpufit` + `Cpufit` | `windows-2022` | Windows x64; NVIDIA display driver `>= 551.61` (CUDA 12.4 GA minimum) |
| `windows-x64-cuda13.2` | `Gpufit` + `Cpufit` | `windows-2025` | Windows x64; NVIDIA display driver from the [CUDA 13.x compatible driver series](https://docs.nvidia.com/deploy/cuda-compatibility/) (Windows driver is no longer bundled with the toolkit as of CUDA 13.1) |
| `linux-x64-cuda11.8` | `Gpufit` + `Cpufit` | `ubuntu-22.04` | Linux x86_64; NVIDIA driver `>= 520.61.05` (CUDA 11.8 GA minimum); userspace compatibility with Ubuntu 22.04 build baseline |
| `linux-x64-cuda12.5` | `Gpufit` + `Cpufit` | `ubuntu-24.04` | Linux x86_64; NVIDIA driver `>= 555.42.02` (CUDA 12.5 GA minimum); userspace compatibility with Ubuntu 24.04 build baseline |
| `linux-x64-cuda13.0` | `Gpufit` + `Cpufit` | `ubuntu-24.04` | Linux x86_64; NVIDIA driver `>= 575.51.03` (CUDA 13.0 GA minimum); userspace compatibility with Ubuntu 24.04 build baseline |
| `macos-x64-cpu` | `Cpufit` only | `macos-15-intel` | macOS x64; no NVIDIA/CUDA requirement |
| `macos-arm64-cpu` | `Cpufit` only | `macos-14` | macOS arm64; no NVIDIA/CUDA requirement |

Notes:

- macOS artifacts are CPU-only and do not include the CUDA `Gpufit` shared library.
- CUDA driver minimums above come from NVIDIA CUDA Toolkit release notes: [11.8](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/), [12.4](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-toolkit-release-notes/index.html), [12.5](https://docs.nvidia.com/cuda/archive/12.5.1/cuda-toolkit-release-notes/index.html), [13.0](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-toolkit-release-notes/index.html), [13.2](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html) (Linux driver minimum for 13.2 GA is `>= 595.45.04`; NVIDIA stopped publishing a bundled Windows minimum starting with CUDA 13.1, see the [CUDA Compatibility Guide](https://docs.nvidia.com/deploy/cuda-compatibility/)).
- If your target machine is older/newer than the CI baseline OS, build from source for best compatibility.

## Documentation

[![Documentation Status](https://readthedocs.org/projects/gpufit/badge/?version=latest)](http://gpufit.readthedocs.io/en/latest/?badge=latest)

Documentation for the Gpufit library may be found online ([latest documentation](http://gpufit.readthedocs.io/en/latest/?badge=latest)), and also
as a PDF file in the binary distribution of Gpufit.

## Building Gpufit from source code

Instructions for building Gpufit are found in the documentation: [Building from source code](https://github.com/gpufit/Gpufit/blob/master/docs/installation.rst).

## Using the Gpufit binary distribution

Instructions for using the binary distribution may be found in the documentation. The binary package contains:

- `Cpufit` shared library in all artifacts.
- `Gpufit` shared library in CUDA-enabled Linux/Windows artifacts.
- C/C++ headers under `include/` (`Cpufit` in all artifacts, `Gpufit` in CUDA-enabled artifacts).
- Python wrapper payloads (`Cpufit` and `Gpufit` source wrapper trees, plus wheels if generated).
- MATLAB wrapper payload when generated by the CI build.
- Java wrapper payload when generated by the CI build.
- `BUILD_INFO.txt` with build metadata.

## MRI model support (release CUDA binaries)

Release model notes:

- Linux/Windows CUDA artifacts include these models in Gpufit.
- macOS artifacts are CPU-only (Cpufit only; no CUDA/Gpufit shared library).
- The `Gpufit_Cpufit_MRI_Parity` executable is included in the `bin` folder of the release package to verify model implementation.

| Model | Included in CUDA release binaries (Linux/Windows) | CPU version (Cpufit) | macOS release support |
|---|---|---|---|
| PATLAK | Yes | Yes | CPU-only (Cpufit) |
| TOFTS | Yes | Yes | CPU-only (Cpufit) |
| TOFTS_EXTENDED | Yes | Yes | CPU-only (Cpufit) |
| TISSUE_UPTAKE | Yes | Yes | CPU-only (Cpufit) |
| TWO_COMPARTMENT_EXCHANGE (2CXM) | Yes | Yes | CPU-only (Cpufit) |
| T1_FA_EXPONENTIAL | Yes | Yes | CPU-only (Cpufit) |

## Examples

There are various examples that demonstrate the capabilities and usage of Gpufit. They can be found at the following locations:

- /examples/c++ - C++ examples for Gpufit
- /examples/c++/gpu_vs_cpu_profiling - C++ examples that use Gpufit and Cpufit
- /examples/matlab - Matlab examples for Gpufit including spline fit examples (also requires [Gpuspline](https://github.com/gpufit/Gpuspline))
- /examples/python - Python examples for Gpufit including spline fit examples (also requires [Gpuspline](https://github.com/gpufit/Gpuspline))
- /Cpufit/matlab/examples - Matlab examples that only uses Cpufit
- /Gpufit/java/gpufit/src/test/java/com/github/gpufit/examples - Java examples for Gpufit

## Authors

Gpufit was created by Mark Bates, Adrian Przybylski, Björn Thiel, and Jan Keller-Findeisen at the Max Planck Institute for Biophysical Chemistry, in Göttingen, Germany.

## How to cite Gpufit

If you use Gpufit in your research, please cite our publication describing the software.  A paper describing the software was published in Scientific Reports.  The open-access manuscript is available from the Scientific Reports website, [here](https://www.nature.com/articles/s41598-017-15313-9).

  *  Gpufit: An open-source toolkit for GPU-accelerated curve fitting  
     Adrian Przybylski, Björn Thiel, Jan Keller-Findeisen, Bernd Stock, and Mark Bates  
     Scientific Reports, vol. 7, 15722 (2017); doi: https://doi.org/10.1038/s41598-017-15313-9 

## License

MIT License

Copyright (c) 2017 Mark Bates, Adrian Przybylski, Björn Thiel, and Jan Keller-Findeisen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
