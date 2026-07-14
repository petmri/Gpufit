#!/usr/bin/env python3
"""Execution smoke test for a built/packaged pyCpufit: fits a real PATLAK
curve (not just an import check) so a broken or mis-packaged shared library
fails the release build instead of shipping silently. PATLAK is one of the
DCE-MRI models this project actually ships, so it exercises the user_info
(time/AIF) code path that a generic model like GAUSS_1D would not."""

import sys

import numpy as np
import pycpufit.cpufit as cf

print("pycpufit:", cf.__file__)

# time (min) and a simple decaying arterial input function (AIF)
timer = np.linspace(0.0, 5.0, 20, dtype=np.float32)
cp = (6.0 * np.exp(-2.0 * timer) + 0.5 * np.exp(-0.2 * timer)).astype(np.float32)

# trapezoidal cumulative integral of Cp, matching the PATLAK model's convolution
spacing = np.diff(timer)
increments = (cp[:-1] + cp[1:]) / 2.0 * spacing
conv_cp = np.concatenate([[0.0], np.cumsum(increments)]).astype(np.float32)

true_parameters = np.array([0.1, 0.05], dtype=np.float32)  # Ktrans, vp
data = (true_parameters[0] * conv_cp + true_parameters[1] * cp).reshape(1, -1).astype(np.float32)
initial_parameters = np.array([[0.05, 0.02]], dtype=np.float32)
user_info = np.ascontiguousarray(np.concatenate([timer, cp]), dtype=np.float32)

parameters, states, chi_squares, n_iterations, elapsed_sec = cf.fit(
    data, None, cf.ModelID.PATLAK, initial_parameters, user_info=user_info
)

state = int(np.asarray(states, dtype=np.int32).reshape(-1)[0])
recovered = np.asarray(parameters, dtype=np.float64).reshape(-1)

if state != cf.FitState.CONVERGED:
    sys.exit(f"pycpufit smoke test did not converge: state={cf.fit_state_name(state)}")
if not np.all(np.isfinite(recovered)):
    sys.exit(f"pycpufit smoke test produced non-finite parameters: {recovered}")
if not np.allclose(recovered, true_parameters, atol=0.01):
    sys.exit(f"pycpufit smoke test parameters off: got {recovered}, expected {true_parameters}")

print("pycpufit smoke test OK:", recovered)
