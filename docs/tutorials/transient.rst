Transient problems
==================

A time-dependent heat conduction problem adds the heat capacity term to the
``heat_transfer`` physics and is advanced in time with the :math:`\theta`
method.

.. literalinclude:: ../../examples/transient_slab.py
   :language: python

Points to note:

* The ``density`` and ``specific_heat`` of the ``heat_transfer`` physics add the
  heat capacity term, and everything else is as in the steady problem.
* ``implicitness=1`` is the backward Euler method, ``implicitness=0.5`` the
  Crank-Nicolson method, and ``implicitness=0`` the forward Euler method.
* At the object level, ``quadrature="nodal"`` on the kernel
  ``heat_conduction_time_derivative`` lumps the capacity term,
  which in the dual mesh method means the measure of the control domain times
  the nodal rate of change.
* An :class:`~dualmesh.Output` group in the ``output`` parameter writes the
  fields at the output times, for example every 0.1 s with
  ``dm.Output(interval=0.1)``.  ``set_time_step_callback`` runs arbitrary Python code after each
  converged step, which is convenient for recording a time history.
