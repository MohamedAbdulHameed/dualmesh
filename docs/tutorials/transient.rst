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
* ``time_scheme`` is ``"backward_Euler"`` (the default), ``"Crank-Nicolson"``
  or ``"forward_Euler"``.
* At the object level, ``quadrature="nodal"`` on the kernel
  ``heat_conduction_time_derivative`` lumps the capacity term,
  which in the dual mesh method means the measure of the control domain times
  the nodal rate of change.
* An :class:`~dualmesh.Output` group in the ``output`` parameter writes the
  fields at the output times, for example every 0.1 s with
  ``dualmesh.Output(interval=0.1)``.  ``set_time_step_callback`` runs arbitrary Python code after each
  converged step, which is convenient for recording a time history.
