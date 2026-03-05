"""
Python binding for Cpufit, the CPU implementation of Gpufit-style fitting.

The binding is based on ctypes.
"""

import os
import sys
import time
from ctypes import cdll, POINTER, c_int, c_float, c_char, c_char_p, c_size_t

import numpy as np


package_dir = os.path.dirname(os.path.realpath(__file__))

if os.name == 'nt':
    lib_path = os.path.join(package_dir, 'Cpufit.dll')
elif os.name == 'posix':
    if sys.platform == 'darwin':
        lib_path = os.path.join(package_dir, 'libCpufit.dylib')
    else:
        lib_path = os.path.join(package_dir, 'libCpufit.so')
else:
    raise RuntimeError('OS {} not supported by pyCpufit.'.format(os.name))

lib = cdll.LoadLibrary(lib_path)

# cpufit_constrained function in the library
try:
    cpufit_func = lib.cpufit_constrained
except AttributeError as exc:
    raise RuntimeError(
        "Cpufit library does not export 'cpufit_constrained'. "
        "This usually indicates an outdated or mispackaged Cpufit binary."
    ) from exc
cpufit_func.restype = c_int
cpufit_func.argtypes = [
    c_size_t,
    c_size_t,
    POINTER(c_float),
    POINTER(c_float),
    c_int,
    POINTER(c_float),
    POINTER(c_float),
    POINTER(c_int),
    c_float,
    c_int,
    POINTER(c_int),
    c_int,
    c_size_t,
    POINTER(c_char),
    POINTER(c_float),
    POINTER(c_int),
    POINTER(c_float),
    POINTER(c_int),
]

try:
    cpufit_patlak_bounded_linear_func = lib.cpufit_patlak_bounded_linear
except AttributeError:
    cpufit_patlak_bounded_linear_func = None

if cpufit_patlak_bounded_linear_func is not None:
    cpufit_patlak_bounded_linear_func.restype = c_int
    cpufit_patlak_bounded_linear_func.argtypes = [
        c_size_t,
        c_size_t,
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_int),
        POINTER(c_int),
        c_size_t,
        POINTER(c_char),
        POINTER(c_float),
        POINTER(c_int),
        POINTER(c_float),
        POINTER(c_int),
    ]

# cpufit_get_last_error function in the library
error_func = lib.cpufit_get_last_error
error_func.restype = c_char_p
error_func.argtypes = None


class ModelID:
    GAUSS_1D = 0
    GAUSS_2D = 1
    GAUSS_2D_ELLIPTIC = 2
    GAUSS_2D_ROTATED = 3
    CAUCHY_2D_ELLIPTIC = 4
    LINEAR_1D = 5
    FLETCHER_POWELL = 6
    BROWN_DENNIS = 7
    SPLINE_1D = 8
    SPLINE_2D = 9
    SPLINE_3D = 10
    SPLINE_3D_MULTICHANNEL = 11
    SPLINE_3D_PHASE_MULTICHANNEL = 12
    LIVER_FAT_TWO = 13
    LIVER_FAT_THREE = 14
    LIVER_FAT_FOUR = 15
    EXPONENTIAL = 16
    PATLAK = 17
    TOFTS = 18
    TOFTS_EXTENDED = 19
    TISSUE_UPTAKE = 20
    TWO_COMPARTMENT_EXCHANGE = 21
    T1_FA_EXPONENTIAL = 22


class EstimatorID:
    LSE = 0
    MLE = 1


class ConstraintType:
    FREE = 0
    LOWER = 1
    UPPER = 2
    LOWER_UPPER = 3


class Status:
    Ok = 0
    Error = -1


class FitState:
    CONVERGED = 0
    MAX_ITERATION = 1
    SINGULAR_HESSIAN = 2
    NEG_CURVATURE_MLE = 3
    GPU_NOT_READY = 4


_FIT_STATE_NAMES = {
    FitState.CONVERGED: "CONVERGED",
    FitState.MAX_ITERATION: "MAX_ITERATION",
    FitState.SINGULAR_HESSIAN: "SINGULAR_HESSIAN",
    FitState.NEG_CURVATURE_MLE: "NEG_CURVATURE_MLE",
    FitState.GPU_NOT_READY: "GPU_NOT_READY",
}


def _valid_id(cls, id_value):
    properties = [key for key in cls.__dict__.keys() if not key.startswith('__')]
    values = [cls.__dict__[key] for key in properties]
    return id_value in values


def _decode_error(error_message):
    if isinstance(error_message, bytes):
        return error_message.decode('utf-8', errors='replace')
    return str(error_message)


def fit_state_name(state):
    """Return symbolic name for a Cpufit fit state."""
    try:
        state_int = int(state)
    except Exception:
        return 'UNKNOWN'
    return _FIT_STATE_NAMES.get(state_int, 'UNKNOWN')


def summarize_fit_states(states, number_iterations=None):
    """Summarize fit-state counts and example indices for diagnostics."""
    states_arr = np.asarray(states, dtype=np.int32).reshape(-1)
    summary = {'total_fits': int(states_arr.size), 'counts': {}, 'failed_examples': []}
    if number_iterations is not None:
        iterations_arr = np.asarray(number_iterations, dtype=np.int32).reshape(-1)
    else:
        iterations_arr = None

    unique, counts = np.unique(states_arr, return_counts=True)
    for state_value, count in zip(unique.tolist(), counts.tolist()):
        summary['counts'][fit_state_name(state_value)] = int(count)

    failed_idx = np.nonzero(states_arr != FitState.CONVERGED)[0]
    for idx in failed_idx[:8]:
        item = {'index': int(idx), 'state': int(states_arr[idx]), 'state_name': fit_state_name(states_arr[idx])}
        if iterations_arr is not None and idx < iterations_arr.size:
            item['iterations'] = int(iterations_arr[idx])
        summary['failed_examples'].append(item)
    return summary


def fit(
    data,
    weights,
    model_id,
    initial_parameters,
    tolerance=None,
    max_number_iterations=None,
    parameters_to_fit=None,
    estimator_id=None,
    user_info=None,
    patlak_algorithm='lm',
):
    """Calls fit_constrained without constraints."""

    return fit_constrained(
        data,
        weights,
        model_id,
        initial_parameters,
        constraints=None,
        constraint_types=None,
        tolerance=tolerance,
        max_number_iterations=max_number_iterations,
        parameters_to_fit=parameters_to_fit,
        estimator_id=estimator_id,
        user_info=user_info,
        patlak_algorithm=patlak_algorithm,
    )


def fit_constrained(
    data,
    weights,
    model_id,
    initial_parameters,
    constraints=None,
    constraint_types=None,
    tolerance=None,
    max_number_iterations=None,
    parameters_to_fit=None,
    estimator_id=None,
    user_info=None,
    patlak_algorithm='lm',
):
    """
    Calls the C interface fit function in the Cpufit library.
    """

    # check all 2D NumPy arrays for row-major memory layout
    if not data.flags.c_contiguous:
        raise RuntimeError('Memory layout of data array mismatch.')
    if weights is not None and not weights.flags.c_contiguous:
        raise RuntimeError('Memory layout of weights array mismatch.')
    if not initial_parameters.flags.c_contiguous:
        raise RuntimeError('Memory layout of initial_parameters array mismatch.')

    # size check: data is 2D and read number of points and fits
    if data.ndim != 2:
        raise RuntimeError('data is not two-dimensional')
    number_points = data.shape[1]
    number_fits = data.shape[0]

    # size check: consistency with weights (if given)
    if weights is not None and data.shape != weights.shape:
        raise RuntimeError('dimension mismatch between data and weights')

    # size check: initial parameters is 2D and read number of parameters
    if initial_parameters.ndim != 2:
        raise RuntimeError('initial_parameters is not two-dimensional')
    number_parameters = initial_parameters.shape[1]
    if initial_parameters.shape[0] != number_fits:
        raise RuntimeError('dimension mismatch in number of fits between data and initial_parameters')

    # size check: constraints is 2D and number of fits, 2x number of parameters if given
    if constraints is not None:
        if constraints.ndim != 2:
            raise RuntimeError('constraints not two-dimensional')
        if constraints.shape != (number_fits, 2 * number_parameters):
            raise RuntimeError('constraints array has invalid shape')

    # size check: constraint_types has certain length (if given)
    if constraint_types is not None and constraint_types.shape[0] != number_parameters:
        raise RuntimeError('constraint_types should have length of number of parameters')

    # size check: consistency with parameters_to_fit (if given)
    if parameters_to_fit is not None and parameters_to_fit.shape[0] != number_parameters:
        raise RuntimeError(
            'dimension mismatch in number of parameters between initial_parameters and parameters_to_fit'
        )

    # default values
    if constraint_types is None:
        constraint_types = np.full(number_parameters, ConstraintType.FREE, dtype=np.int32)
    if tolerance is None:
        tolerance = 1e-4
    if max_number_iterations is None:
        max_number_iterations = 25
    if estimator_id is None:
        estimator_id = EstimatorID.LSE
    if parameters_to_fit is None:
        parameters_to_fit = np.ones(number_parameters, dtype=np.int32)

    # type checks
    if data.dtype != np.float32:
        raise RuntimeError('type of data is not np.float32')
    if weights is not None and weights.dtype != np.float32:
        raise RuntimeError('type of weights is not np.float32')
    if initial_parameters.dtype != np.float32:
        raise RuntimeError('type of initial_parameters is not np.float32')
    if constraints is not None and constraints.dtype != np.float32:
        raise RuntimeError('type of constraints is not np.float32')
    if parameters_to_fit.dtype != np.int32:
        raise RuntimeError('type of parameters_to_fit is not np.int32')
    if constraint_types.dtype != np.int32:
        raise RuntimeError('type of constraint_types is not np.int32')

    if not _valid_id(ModelID, model_id):
        raise RuntimeError('Invalid model ID, use an attribute of ModelID')
    if not _valid_id(EstimatorID, estimator_id):
        raise RuntimeError('Invalid estimator ID, use an attribute of EstimatorID')
    if not all(_valid_id(ConstraintType, item) for item in constraint_types):
        raise RuntimeError('Invalid constraint type, use an attribute of ConstraintType')

    patlak_algorithm_normalized = str(patlak_algorithm).strip().lower()
    if patlak_algorithm_normalized not in {'lm', 'bounded_linear'}:
        raise RuntimeError("patlak_algorithm must be either 'lm' or 'bounded_linear'")

    if user_info is not None:
        user_info_size = user_info.nbytes
    else:
        user_info_size = 0

    # pre-allocate output variables
    parameters = np.zeros((number_fits, number_parameters), dtype=np.float32)
    states = np.zeros(number_fits, dtype=np.int32)
    chi_squares = np.zeros(number_fits, dtype=np.float32)
    number_iterations = np.zeros(number_fits, dtype=np.int32)

    # conversion to ctypes types for optional C interface parameters
    if weights is not None:
        weights_p = weights.ctypes.data_as(cpufit_func.argtypes[3])
    else:
        weights_p = None
    if constraints is not None:
        constraints_p = constraints.ctypes.data_as(cpufit_func.argtypes[6])
    else:
        constraints_p = None
    if user_info is not None:
        user_info_p = user_info.ctypes.data_as(cpufit_func.argtypes[13])
    else:
        user_info_p = None

    use_patlak_bounded_linear = (
        model_id == ModelID.PATLAK and patlak_algorithm_normalized == 'bounded_linear'
    )
    if use_patlak_bounded_linear:
        if cpufit_patlak_bounded_linear_func is None:
            raise RuntimeError(
                "Cpufit library does not export 'cpufit_patlak_bounded_linear'. "
                "Rebuild/install the updated Cpufit binary to use patlak_algorithm='bounded_linear'."
            )
        if estimator_id != EstimatorID.LSE:
            raise RuntimeError("patlak_algorithm='bounded_linear' supports only estimator_id=LSE")

    # call into the library (measure time)
    t0 = time.perf_counter()
    if use_patlak_bounded_linear:
        status = cpufit_patlak_bounded_linear_func(
            cpufit_patlak_bounded_linear_func.argtypes[0](number_fits),
            cpufit_patlak_bounded_linear_func.argtypes[1](number_points),
            data.ctypes.data_as(cpufit_patlak_bounded_linear_func.argtypes[2]),
            weights_p,
            initial_parameters.ctypes.data_as(cpufit_patlak_bounded_linear_func.argtypes[4]),
            constraints_p,
            constraint_types.ctypes.data_as(cpufit_patlak_bounded_linear_func.argtypes[6]),
            parameters_to_fit.ctypes.data_as(cpufit_patlak_bounded_linear_func.argtypes[7]),
            cpufit_patlak_bounded_linear_func.argtypes[8](user_info_size),
            user_info_p,
            parameters.ctypes.data_as(cpufit_patlak_bounded_linear_func.argtypes[10]),
            states.ctypes.data_as(cpufit_patlak_bounded_linear_func.argtypes[11]),
            chi_squares.ctypes.data_as(cpufit_patlak_bounded_linear_func.argtypes[12]),
            number_iterations.ctypes.data_as(cpufit_patlak_bounded_linear_func.argtypes[13]),
        )
    else:
        status = cpufit_func(
            cpufit_func.argtypes[0](number_fits),
            cpufit_func.argtypes[1](number_points),
            data.ctypes.data_as(cpufit_func.argtypes[2]),
            weights_p,
            cpufit_func.argtypes[4](model_id),
            initial_parameters.ctypes.data_as(cpufit_func.argtypes[5]),
            constraints_p,
            constraint_types.ctypes.data_as(cpufit_func.argtypes[7]),
            cpufit_func.argtypes[8](tolerance),
            cpufit_func.argtypes[9](max_number_iterations),
            parameters_to_fit.ctypes.data_as(cpufit_func.argtypes[10]),
            cpufit_func.argtypes[11](estimator_id),
            cpufit_func.argtypes[12](user_info_size),
            user_info_p,
            parameters.ctypes.data_as(cpufit_func.argtypes[14]),
            states.ctypes.data_as(cpufit_func.argtypes[15]),
            chi_squares.ctypes.data_as(cpufit_func.argtypes[16]),
            number_iterations.ctypes.data_as(cpufit_func.argtypes[17]),
        )
    t1 = time.perf_counter()

    if status != Status.Ok:
        error_message = _decode_error(error_func())
        raise RuntimeError('status = {}, message = {}'.format(status, error_message))

    return parameters, states, chi_squares, number_iterations, t1 - t0


def get_last_error():
    """Returns the message of the last Cpufit error."""
    return _decode_error(error_func())
