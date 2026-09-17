A kernel of your own
====================

Physics that the modules do not cover can be added in Python, with exact
derivatives.  The example below is the large-deformation bar of Example 6.2.4,
whose axial stiffness depends on the strain.

.. literalinclude:: ../../examples/custom_kernel.py
   :language: python

The rules are short:

* ``compute_flux`` returns the components of :math:`\mathbf{F}`, and
  ``compute_source`` returns :math:`S`, for the canonical form
  :math:`-\nabla\cdot\mathbf{F} + S = 0`.  Define only the one you need.
* Quantities taken from ``ctx`` are AD numbers.  Arithmetic works as usual;
  for elementary functions use ``dm.exp``, ``dm.sqrt``, and so on, which accept
  both floats and AD numbers.
* ``setup(problem)`` is the place to resolve names, for example
  ``problem.variable_index("temperature")`` for a coupled variable or
  ``problem.property_id("stress")`` for a material property.
* Anything passed as a keyword argument to the constructor becomes an
  attribute, so parameters are simply ``self.axial_stiffness``.

Boundary conditions and materials follow the same pattern with
:class:`dualmesh.PythonBoundaryCondition`, :class:`dualmesh.PythonNodalBoundaryCondition`,
and :class:`dualmesh.PythonMaterial`.  A nonlinear flux condition, for
instance :math:`n\cdot F = -u^2`, is four lines:

.. code-block:: python

   class NonlinearFlux(dm.PythonBoundaryCondition):
       def compute_boundary_flux(self, ctx):
           u = ctx.value(self.variable)
           return -(u * u)

When a model turns out to be worth keeping, the same code moves to C++ almost
unchanged; see :doc:`../developing`.
