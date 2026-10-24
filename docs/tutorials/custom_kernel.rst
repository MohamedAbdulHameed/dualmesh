User-defined kernels
====================

Physics that the modules do not cover can be added in Python, with exact
derivatives.  The example below is a bar in large deformation, whose axial
stiffness depends on the strain.

.. literalinclude:: ../../examples/custom_kernel.py
   :language: python

A Python kernel follows four rules:

* ``compute_flux`` returns the components of :math:`\mathbf{F}`, and
  ``compute_source`` returns :math:`S`, for the canonical form
  :math:`-\nabla\cdot\mathbf{F} + S = 0`.  Define only the one that the term requires.
* Quantities taken from ``ctx`` are AD numbers.  Arithmetic works as usual.
  For elementary functions use ``dm.exp``, ``dm.sqrt``, and so on, which accept
  both floats and AD numbers.
* Names are resolved in ``setup(problem)``, for example
  ``problem.variable_index("temperature")`` for a coupled variable or
  ``problem.property_id("stress")`` for a property.
* Every keyword argument passed to the constructor becomes an attribute, so
  that a parameter is read as, e.g., ``self.axial_stiffness``.

The context ``ctx`` of an integration point gives the position ``ctx.x``, the
time ``ctx.time`` and the time step ``ctx.dt``, the values and gradients of the
variables at the current iterate (``ctx.value(v)``, ``ctx.gradient(v)``) and at
the end of the last time step (``ctx.old_value(v)``, ``ctx.old_gradient(v)``),
the values of the properties (``ctx.property(id)``), whether the point is on a
boundary (``ctx.on_boundary``), and whether the nonlinear solver is the Picard
iteration (``ctx.is_picard``), in which a kernel may lag a nonlinear
coefficient.  A time derivative written in Python reads, for instance,

.. code-block:: python

   class TimeDerivative(dm.PythonKernel):
       def compute_source(self, ctx):
           u = self.variable
           return (ctx.value(u) - ctx.old_value(u)) / ctx.dt

Boundary conditions and property objects follow the same pattern with
:class:`dualmesh.PythonBoundaryCondition`,
:class:`dualmesh.PythonNodalBoundaryCondition` and
:class:`dualmesh.PythonProperty`.  A nonlinear flux condition, for
instance :math:`n\cdot F = -u^2`, takes four lines:

.. code-block:: python

   class NonlinearFlux(dm.PythonBoundaryCondition):
       def compute_boundary_flux(self, ctx):
           u = ctx.value(self.variable)
           return -(u * u)

A Python model that is to be kept can be translated into a C++ object almost
line for line (see :doc:`../developing`).
