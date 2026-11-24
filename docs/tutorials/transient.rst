Transient problems
==================

In this tutorial, you follow the quench of a steel slab in time with the
input, :class:`dualmesh.problem`, and you compare the temperature with the
Fourier series of the exact solution.

The equations
-------------

A body at rest stores heat in its heat capacity, and the heat balance is

.. math::

   \rho c_p \frac{\partial T}{\partial t} + \nabla\cdot\mathbf{q} = s,
   \qquad \mathbf{q} = -k\nabla T,

where :math:`T` is the temperature in :math:`\mathrm{K}`, :math:`t` the
time in :math:`\mathrm{s}`, :math:`\rho` the density in
:math:`\mathrm{kg/m^3}`, :math:`c_p` the specific heat capacity in
:math:`\mathrm{J/(kg\,K)}`, :math:`\mathbf{q}` the heat flux in
:math:`\mathrm{W/m^2}`, :math:`k` the thermal conductivity in
:math:`\mathrm{W/(m\,K)}` and :math:`s` the heat source in
:math:`\mathrm{W/m^3}`.

A slab of thickness :math:`L = 0.1\ \mathrm{m}` of steel, with :math:`k =
45\ \mathrm{W/(m\,K)}`, :math:`\rho = 7850\ \mathrm{kg/m^3}` and :math:`c_p
= 490\ \mathrm{J/(kg\,K)}`, starts at :math:`T_i = 573\ \mathrm{K}`.  From
:math:`t = 0`, both faces are held at :math:`T_s = 293\ \mathrm{K}`.  Without
a source, the temperature is the Fourier series

.. math::

   T(x, t) = T_s + (T_i - T_s) \sum_{j\ \mathrm{odd}} \frac{4}{j\pi}
   \sin\frac{j\pi x}{L}\, \exp\left(-\frac{j^2\pi^2 \kappa t}{L^2}\right),

where :math:`x` is the position through the slab in :math:`\mathrm{m}`,
:math:`j` the index of a term and :math:`\kappa = k/(\rho c_p) = 1.17 \times
10^{-5}\ \mathrm{m^2/s}` the thermal diffusivity.  At :math:`t = 40\
\mathrm{s}` the series gives :math:`515.8\ \mathrm{K}` at the middle of the
slab, which has then cooled by a fifth of the drop :math:`T_i - T_s`.

Build and solve the slab
------------------------

The script is ``examples/transient_slab.py``.

1. Create the problem on a segment of 40 cells.  Without a method, heat
   conduction takes the dual mesh control domain method (DMCDM):

   .. literalinclude:: ../../examples/transient_slab.py
      :language: python
      :start-at: quench = dualmesh.problem(
      :end-at: quench = dualmesh.problem(
      :dedent: 4

2. Add the physics and its properties.  A transient problem needs the
   density and the specific heat capacity besides the conductivity:

   .. literalinclude:: ../../examples/transient_slab.py
      :language: python
      :start-at: quench.physics("heat_conduction")
      :end-before: quench.initial_conditions(
      :dedent: 4

3. Give the temperature at the start, and hold both faces:

   .. literalinclude:: ../../examples/transient_slab.py
      :language: python
      :start-at: quench.initial_conditions(
      :end-at: quench.boundary_conditions("right"
      :dedent: 4

4. Make the analysis transient.  The time section takes the end time, the
   first time step and the control of the step:

   .. literalinclude:: ../../examples/transient_slab.py
      :language: python
      :start-at: quench.time(
      :end-at: quench.time(
      :dedent: 4

   In the script the step is :math:`\Delta t = 0.1\ \mathrm{s}`, and
   ``{"by": "fixed"}`` keeps it fixed.  The other controls choose the step as
   the solve goes:

   * ``{"by": "error"}``, the default, keeps the estimated error of each step
     below ``relative_tolerance``, :math:`10^{-3}` by default.
   * ``{"by": "iterations"}`` aims at ``target_iterations`` Newton iterations
     per step, 5 by default.
   * ``{"by": "Courant"}`` keeps the Courant number :math:`C` of a flow,
     the distance that the flow travels in a step divided by the element
     size, at ``Courant_number``, 0.5 by default.

   Each step is a backward Euler step.

5. Solve, and compare the temperature at the nodes with the series:

   .. literalinclude:: ../../examples/transient_slab.py
      :language: python
      :start-at: solution = slab().solve()
      :end-at: reference = exact(x, end_time)
      :dedent: 4

   ``exact`` is the function of the script that sums the first 200 terms of
   the series.

The script prints the largest computed temperature at :math:`40\ \mathrm{s}`
next to that of the series, and the largest difference between the two.

Write the fields in time
------------------------

A post-processor is a scalar result of the solve, and the output section
writes the fields to files.  The test
``test_a_transient_run_writes_its_fields_and_its_postprocessors_with_their_units``
in ``tests/python/test_front_end.py`` adds both to a heated slab:

.. code-block:: python

   slab.postprocessor("peak", maximum="temperature")
   slab.output(file="slab", interval=2, fields=["temperature"])

The solve then writes the temperature every second step and at the end, as
``slab_0000.vtu``, ``slab_0001.vtu`` and so on, with the collection
``slab.pvd`` that ParaView opens as one time series.  The file ``slab.csv``
holds the time and the post-processor ``peak`` in columns headed by their
units, ``time [s]`` and ``peak [K]``.  ``solution.times`` lists the times of the accepted
steps.
