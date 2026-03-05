========
Appendix
========

Levenberg-Marquardt algorithm
-----------------------------

A flowchart of the implementation of the Levenberg-Marquardt algorithm is given in :numref:`appendix-gpufit-flowchart`.

Constrained globalized LM details
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For constrained fits, the implementation uses projected trial steps and backtracking:

- The raw LM step is computed in parameter space.
- Trial parameters are projected to box constraints.
- If a projected trial does not decrease :math:`\chi^2`, the step is reduced (backtracking) and retried.

The convergence test for constrained fits includes an additional guard for active-bound solutions.
Besides the standard :math:`\chi^2`-change criterion, a fit is accepted as converged when both conditions are true:

- the projected step infinity norm is tiny relative to tolerance, and
- :math:`\chi^2` is non-increasing with a tiny absolute change.

This avoids reporting :code:`MAX_ITERATION` for fits that are already stationary after projection near box boundaries.
The same constrained convergence behavior is implemented for both CPU and GPU LM paths.

.. _appendix-gpufit-flowchart:

.. figure:: /images/gpufit_program_flow_skeleton_v3.png
   :width: 14 cm
   :align: center

   Levenberg-Marquardt algorithm flow as implemented in the Gpufit library.
   
   
Performance comparison to other GPU benchmarks
----------------------------------------------

Using the bundled application (initial release created with CUDA runtime 8.0) to estimate the fitting speed per second of 10 million fits for various CUDA capable
graphics cards of various architectures (on different computers with different versions of graphics drivers) we can
compare to the results of Passmark G3D. By and large, the results seem to correlate, i.e. a high Passmark G3D score
also relates to a high Gpufit fitting speed.

.. figure:: /images/GPUfit_PassmarkG3D_relative_performance.png
   :width: 14 cm
   :align: center

   Performance of Gpufit vs Passmark G3D