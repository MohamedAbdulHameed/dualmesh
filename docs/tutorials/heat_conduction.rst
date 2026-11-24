Heat conduction
===============

In this tutorial, you solve two steady heat conduction problems with the
input, :class:`dualmesh.problem`: a cooling fin in one dimension and a bus bar
in two.  Both are examples of J. N. Reddy's book on the dual mesh control
domain method (DMCDM) [Reddy2024]_, and you check each result against the
values that the book publishes.

The equations
-------------

A body at rest conducts heat by Fourier's law, :math:`\mathbf{q} =
-k\nabla T`, where :math:`T` is the temperature in :math:`\mathrm{K}`,
:math:`\mathbf{q}` the heat flux in :math:`\mathrm{W/m^2}` and :math:`k` the
thermal conductivity in :math:`\mathrm{W/(m\,K)}`.  In a steady state the
heat balance is

.. math::

   \nabla \cdot \mathbf{q} = s \quad \text{in } \Omega,

where :math:`s` is the heat generated per unit volume in
:math:`\mathrm{W/m^3}` and :math:`\Omega` is the body.  Each part of the
boundary :math:`\partial\Omega` takes one condition: a temperature, or the
heat flux :math:`\mathbf{q}\cdot\mathbf{n}` that leaves the body along the
outward unit normal :math:`\mathbf{n}`.  Convection to a fluid at the ambient
temperature :math:`T_a` gives :math:`\mathbf{q}\cdot\mathbf{n} = h_c (T -
T_a)`, where :math:`h_c` is the heat transfer coefficient in
:math:`\mathrm{W/(m^2\,K)}`.  A boundary without a condition is insulated:
no heat crosses it.

A cooling fin in one dimension
------------------------------

An aluminium fin of length :math:`L = 50\ \mathrm{mm}` and conductivity
:math:`k = 200\ \mathrm{W/(m\,K)}` is held at :math:`593\ \mathrm{K}` at its
root and loses heat along its length to air at :math:`T_a = 293\
\mathrm{K}`.  Its excess temperature :math:`\theta = T - T_a` solves the
equation of Example 5.3.1 of [Reddy2024]_,

.. math::

   -\frac{\mathrm{d}^2\theta}{\mathrm{d}x^2} + c\,\theta = 0, \qquad
   \theta(0) = 300\ \mathrm{K}, \qquad
   \frac{\mathrm{d}\theta}{\mathrm{d}x}(L) + \frac{h_c}{k}\,\theta(L) = 0,

where :math:`x` is the position along the fin in :math:`\mathrm{m}` and
:math:`c = 400\ \mathrm{1/m^2}` is the ratio of the convection over the
perimeter of the fin to the conduction along it.  The heat that the fin loses
along its length is the volumetric source :math:`s = -c\,k\,(T - T_a)`.  At
the tip, the fin loses heat by convection with :math:`h_c = 400\
\mathrm{W/(m^2\,K)}`, so that :math:`h_c/k = 2\ \mathrm{1/m}`.

The script is ``examples/cooling_fin.py``.

1. Create the problem on a segment, solved by the DMCDM.  The function
   ``fin`` of the script takes the number of cells, five by default:

   .. literalinclude:: ../../examples/cooling_fin.py
      :language: python
      :start-at: cooling_fin = dualmesh.problem(
      :end-before: cooling_fin.physics(
      :dedent: 4

   Each dimensional quantity is a text with its unit.  The segment has the
   boundaries ``left`` and ``right``.

2. Add the physics and its properties.  The heat source is a formula of the
   temperature, with each constant in square brackets with its unit:

   .. literalinclude:: ../../examples/cooling_fin.py
      :language: python
      :start-at: cooling_fin.physics(
      :end-before: cooling_fin.boundary_conditions("left"
      :dedent: 4

3. Hold the root at its temperature and let the tip convect:

   .. literalinclude:: ../../examples/cooling_fin.py
      :language: python
      :start-at: cooling_fin.boundary_conditions("left"
      :end-before: return cooling_fin
      :dedent: 4

4. Solve, and read :math:`\theta` at the nodes in the order of :math:`x`.
   The script defines ``ambient_temperature = 293.0``:

   .. literalinclude:: ../../examples/cooling_fin.py
      :language: python
      :start-at: solution = fin().solve()
      :end-at: excess =
      :dedent: 4

The script prints :math:`\theta` at the six nodes next to the values of the
book, 300.00, 257.62, 225.59, 202.64, 187.83 and 180.57 K.  The test
``test_the_cooling_fin_example_gives_the_temperatures_of_Reddy`` in
``tests/python/test_front_end.py`` runs the script and checks that every value
agrees with the book to :math:`0.005\ \mathrm{K}`.

A bus bar in two dimensions
---------------------------

A bus bar of :math:`0.1 \times 0.05\ \mathrm{m}` and conductivity :math:`k =
20\ \mathrm{W/(m\,K)}` generates :math:`s = 1\ \mathrm{MW/m^3}`.  Its left
and right sides are held at 40 and :math:`10\ \mathrm{^\circ C}`, its bottom
is insulated, and its top loses heat by convection with :math:`h_c = 75\
\mathrm{W/(m^2\,K)}` to surroundings at :math:`0\ \mathrm{^\circ C}`.  This
is Example 5.4.3 of [Reddy2024]_.

The script is ``examples/bus_bar.py``.

1. Create the problem on a rectangle of :math:`10 \times 5` cells, with the
   physics, the properties and the conditions.  The bottom takes no
   condition, so it is insulated:

   .. literalinclude:: ../../examples/bus_bar.py
      :language: python
      :start-at: bar = dualmesh.problem(
      :end-before: bar.postprocessor("heat_through_left"
      :dedent: 4

   The rectangle has the boundaries ``left``, ``right``, ``bottom`` and
   ``top``.  A temperature in :math:`\mathrm{^\circ C}` converts to
   :math:`\mathrm{K}`.

2. Add the post-processors, the scalar results of the solve: the heat that
   leaves through the left and the right sides, the temperatures at the middle
   of the bottom and the top edges, and the largest temperature:

   .. literalinclude:: ../../examples/bus_bar.py
      :language: python
      :start-at: bar.postprocessor("heat_through_left"
      :end-at: bar.postprocessor("hottest"
      :dedent: 4

   A ``boundary_flux`` of ``heat_flux`` is the heat that leaves the body
   through the boundary.

3. Write the temperature to files:

   .. literalinclude:: ../../examples/bus_bar.py
      :language: python
      :start-at: bar.output(
      :end-at: bar.output(
      :dedent: 4

   The solve writes ``bus_bar_0000.vtu``, a ``bus_bar.pvd`` collection that
   ParaView opens, and ``bus_bar.csv`` with the post-processors, each column
   headed by its name and its unit.

4. Solve, and print each post-processor with its unit:

   .. literalinclude:: ../../examples/bus_bar.py
      :language: python
      :start-at: solution = bus_bar().solve()
      :end-at: print(f"{name}
      :dedent: 4

The script prints the temperature at the middle of the bottom edge in
:math:`\mathrm{^\circ C}` next to the value of the book, 83.142.  The test
``test_the_bus_bar_example_gives_the_temperature_of_Reddy`` in
``tests/python/test_front_end.py`` runs the script and checks this value to
:math:`5 \times 10^{-4}\ \mathrm{^\circ C}`.

Save the input as a file
------------------------

A problem built in Python writes its input as a YAML file with
:meth:`~dualmesh.problem.to_YAML`, or a TOML file with
:meth:`~dualmesh.problem.to_TOML`, and the command line solves the file.  The
test ``test_the_same_slab_read_from_a_YAML_file_gives_the_same_temperatures``
in ``tests/python/test_front_end.py`` checks that a problem read back from its
YAML file gives the same temperatures:

.. code-block:: python

   bar = bus_bar()
   bar.to_YAML("bus_bar.yaml")

.. code-block:: console

   dualmesh run bus_bar.yaml

:doc:`/user_guide/input` describes every section and key of the input.
