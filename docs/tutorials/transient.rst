Transient problems
==================

Time-dependent problems add a time-derivative kernel and use the
:math:`\theta` method.

.. literalinclude:: ../../examples/transient_slab.py
   :language: python

Points to note:

* ``HeatConductionTimeDerivative`` (or the generic ``TimeDerivative``) is the
  only kernel that changes; everything else is as in the steady problem.
* ``theta=1`` is the backward Euler method, ``theta=0.5`` Crank-Nicolson, and
  ``theta=0`` forward Euler.
* ``quadrature="nodal"`` on the time-derivative kernel lumps the capacity term,
  which in the dual mesh method means the measure of the control domain times
  the nodal rate of change.
* ``output_interval`` and ``output_file_base`` write a VTK file every few
  steps; ``set_time_step_callback`` runs arbitrary Python after each converged
  step, which is convenient for recording a time history.
