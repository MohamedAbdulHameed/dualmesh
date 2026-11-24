A kernel written in Python
==========================

This tutorial adds a term written in Python to a problem and solves a bar in
large deformation, whose axial stiffness depends on the strain.  It follows
the script ``examples/custom_kernel.py``.  The reference of the Python
objects is :ref:`problem-setup-python-objects`.

The problem
-----------

A bar of length :math:`L = 1\ \mathrm{m}`, with the axial position :math:`x`
in :math:`\mathrm{m}`, is fixed at :math:`x = 0` and carries the axial load
:math:`Q` in :math:`\mathrm{N}` at :math:`x = L`.  Its
axial displacement :math:`u` in :math:`\mathrm{m}` satisfies Example 6.2.4 of
[Reddy2024]_:

.. math::

   -\frac{d}{dx} \left( a(u') \frac{du}{dx} \right) = 0 \quad (0 < x < L) ,
   \qquad u(0) = 0 , \qquad a(u') \frac{du}{dx} \Big|_{x = L} = Q ,

with the dimensionless strain :math:`u' = du/dx` and the axial stiffness

.. math::

   a(u') = E A \left( 1 + 1.5\, u' + 0.5\, u'^2 \right) ,

where :math:`E` is Young's modulus in :math:`\mathrm{Pa}` and :math:`A` the
cross-sectional area in :math:`\mathrm{m^2}`.  The script takes
:math:`EA = 1\ \mathrm{N}`.

As a balance law :math:`dF/dx = 0`, the flux is :math:`F = -a(u')\, u'`.  A
kernel returns the kernel flux :math:`\hat{F} = -F = a(u')\, u'` (see
:ref:`problem-setup-kernels`).

The script
----------

.. literalinclude:: ../../examples/custom_kernel.py
   :language: python

The script works in these steps:

1. The class ``large_deformation_bar`` subclasses
   :class:`dualmesh.PythonKernel` and defines ``compute_flux`` only, so the
   kernel has a flux and no source.
2. ``compute_flux`` reads the strain :math:`u'` as the :math:`x` component of
   ``ctx.gradient(self.variable)`` at the integration point, and returns the
   one component of :math:`\hat{F} = a(u')\, u'`.
3. The keyword ``axial_stiffness=1.0`` of the constructor becomes the attribute
   ``self.axial_stiffness``, which is :math:`EA`.
4. ``problem.add_kernel`` takes the instance of the kernel.
5. The ``Neumann`` condition at the right end prescribes
   :math:`n \hat{F} = a(u')\, u'`, with the outward normal :math:`n = 1` there,
   so it sets the load :math:`Q`.
6. ``problem.solve`` runs Newton's method.  The strain and the stiffness are
   :class:`dualmesh.ADReal` numbers, which carry their derivatives, so the
   Jacobian is exact without a derivative written by hand.

The result
----------

For each load :math:`Q = 0.2`, :math:`1.0` and :math:`5.0\ \mathrm{N}`, the
script prints the computed tip displacement :math:`u(L)` beside the value that
[Reddy2024]_ reports for eight elements.  The strain is uniform, so the exact
tip displacement is :math:`u(L) = L\, u'`, with :math:`u'` the root of

.. math::

   E A \left( u' + 1.5\, u'^2 + 0.5\, u'^3 \right) = Q .

The book converges its iterations to a tolerance of :math:`10^{-3}`.
``tests/python/test_heat_transfer.py`` solves the same bar with the dual mesh
control domain method (DMCDM) and checks the tip displacement against these
values.

A time derivative
-----------------

A kernel of a time derivative returns the kernel source
:math:`\hat{S} = \partial u / \partial t`, the rate of the stored quantity
:math:`m = u`, computed from the value at the end of the last time step.  It
defines ``is_time_kernel`` to return true, so that a steady solve skips it and
the time scheme does not weight it:

.. code-block:: python

   class TimeDerivative(dualmesh.PythonKernel):
       """Kernel source (u - u_old) / dt."""

       def is_time_kernel(self):
           return True

       def compute_source(self, ctx):
           u = self.variable
           return (ctx.value(u) - ctx.old_value(u)) / ctx.dt

A boundary condition
--------------------

:class:`dualmesh.PythonBoundaryCondition` defines ``compute_boundary_flux``,
which returns :math:`n \hat{F}`, the flux entering the body.  Example 6.2.2 of
[Reddy2024]_ solves :math:`-d^2 u / dx^2 + 2 u^3 = 0` on :math:`1 < x < 2`,
with :math:`u(1) = 1` and the nonlinear condition
:math:`du/dx + u^2 = 0` at :math:`x = 2`, for a dimensionless :math:`u`.  The
kernel flux of the ``diffusion`` kernel is :math:`du/dx`, so the condition is
:math:`n \hat{F} = -u^2`:

.. code-block:: python

   import dualmesh


   class NonlinearFlux(dualmesh.PythonBoundaryCondition):
       """The flux entering the body, -u^2."""

       def compute_boundary_flux(self, ctx):
           u = ctx.value(self.variable)
           return -(u * u)


   mesh = dualmesh.generate_line_mesh(start=1.0, end=2.0, num_elements=8)
   problem = dualmesh.Problem(mesh, method="DMCDM")
   problem.add_variable("u", initial_condition=0.5)
   problem.add_kernel("diffusion", variable="u")
   problem.add_kernel("reaction", variable="u", coefficient=2.0, exponent=3.0)
   problem.add_boundary_condition("Dirichlet", "left", variable="u", value=1.0)
   problem.add_boundary_condition(NonlinearFlux(variable="u", boundary="right"))
   problem.solve()

``tests/python/test_heat_transfer.py`` runs this problem and checks the nodal
values against the book.

Next steps
----------

* :class:`dualmesh.PythonNodalBoundaryCondition` and
  :class:`dualmesh.PythonProperty` follow the same pattern for a prescribed
  value and for a property (see :ref:`problem-setup-python-objects`).
* An object written in Python puts the assembly on one thread.  Once the form
  of a term is settled, write it as a C++ object (see :doc:`/developing`).
