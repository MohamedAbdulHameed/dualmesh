Adaptive refinement
===================

This page shows how to refine a mesh where the error of the solution is
large, and coarsen it where the error is small.  An adaptive run repeats four
steps: it solves on the current mesh, estimates the error of each cell, marks
the cells to refine and to coarsen, and changes the mesh.  You run this loop
from the ``mesh_adaptivity`` section of the input, for a steady or a transient
analysis, or from the Python functions of ``dualmesh.adaptivity`` at the
object level.  :doc:`/theory/adaptivity` describes the error indicator, the
marking rules and the refinement of each cell type.

.. contents::
   :local:
   :depth: 2

Meshes that adapt
-----------------

Local refinement and coarsening take a mesh of one element type:

- ``Tri3``, ``Tri6``, ``Tet4`` and ``Tet10``, refined by bisection, which
  keeps the mesh conforming.
- ``Quad4`` and ``Hex8``, refined by subdivision into :math:`2^d` children,
  with :math:`d` the dimension of the mesh.  Nodes hang on the sides of the
  neighbours that stay whole, and two cells that share a side differ by at
  most one level of refinement.

Any other mesh raises an error that names the element types above.

A refinement level halves the size of a cell: one subdivision of a
quadrilateral or a hexahedron, or :math:`d` bisections of a triangle or a
tetrahedron.  Coarsening undoes earlier refinement, so a cell never becomes
coarser than its cell of the initial mesh.

Adapt the mesh from the input
-----------------------------

The ``mesh_adaptivity`` section takes four keys, all required, and in a
transient run a fifth, ``interval``.  The error indicator is that of
Zienkiewicz and Zhu [ZienkiewiczZhu1987]_.

``field``
   The field whose indicator drives the marking, the name of a field that
   the physics solve for, such as ``temperature``, ``displacement`` or
   ``velocity``.  The indicators of the components of a vector field add as
   squares.

``refine_fraction``
   A number above 0 and at most 1.  The run refines the smallest set of cells whose
   squared indicators sum to this fraction of the total, the cells with the
   largest indicators first.  This is bulk marking [Dorfler1996]_.

``coarsen_fraction``
   A number from 0 to 1.  The run marks for coarsening the refined cells with
   the smallest indicators, outside those marked to refine, as many as keep
   the sum of their squared indicators at most this fraction of the total.  A
   refinement is undone where every cell it made is marked.  With 0, the run
   does not coarsen.

``maximum_refinement_level``
   An integer of at least 0: the most refinement levels a cell may have
   above its cell of the initial mesh.

``interval``
   An integer of at least 1: the number of accepted time steps between two
   changes of the mesh in a transient run.  A steady run changes the mesh
   after each solve, so the input refuses ``interval`` there.

This plate of quadrilaterals is held at :math:`400\ \mathrm{K}` on its left
side and at :math:`300\ \mathrm{K}` on its bottom side, so the temperature
jumps at the corner between them, where the run refines (from
``tests/python/test_adaptivity.py``):

.. code-block:: python

   import dualmesh

   plate = dualmesh.problem(
       mesh={"rectangle": {"size": ["1 m", "1 m"], "cells": [8, 8], "cell_type": "quadrilateral"}}
   )
   plate.physics("heat_conduction")
   plate.properties(thermal_conductivity="45 W/(m*K)")
   plate.boundary_conditions("left", temperature="400 K")
   plate.boundary_conditions("bottom", temperature="300 K")
   plate.mesh_adaptivity(
       field="temperature",
       refine_fraction=0.5,
       coarsen_fraction=0.05,
       maximum_refinement_level=3,
   )
   solution = plate.solve()

In a case file, the same section reads:

.. code-block:: yaml

   mesh_adaptivity:
     field: temperature
     refine_fraction: 0.5
     coarsen_fraction: 0.05
     maximum_refinement_level: 3

The input refines the sides of the mesh as straight lines.  To follow a
curved side, use :func:`~dualmesh.solve_with_adaptive_refinement` with its
``boundaries`` argument (see `Follow a curved side`_).

The steady loop
~~~~~~~~~~~~~~~

Without a ``time`` section, the run calls
:func:`~dualmesh.solve_with_adaptive_refinement` (see `Run the loop`_) with
the problem of the input on each mesh, and repeats these steps:

#. It solves the problem on the current mesh, with the settings of the
   ``solver`` section, starting from the solution of the last mesh carried to
   this one.
#. It computes the indicator of each cell.
#. It marks the cells to refine by ``refine_fraction``.  The loop ends when
   no marked cell is below ``maximum_refinement_level``.
#. It marks the cells to coarsen by ``coarsen_fraction``.
#. It coarsens and refines the mesh in one change.  The loop ends when the
   mesh does not change.

Refinement and coarsening can undo each other, so the loop runs at most
:math:`4 d L + 1` cycles, with :math:`d` the dimension of the mesh and
:math:`L` the ``maximum_refinement_level``.  With the report ``"full"``, each
cycle prints its number of cells and the estimated error, the square root of
the sum of the squared indicators, and the loop then prints the table of its
cycles, which the report ``"summary"`` prints alone.

The transient loop
~~~~~~~~~~~~~~~~~~

With a ``time`` section, the run repeats these steps until it reaches the
end time:

#. It takes ``interval`` accepted time steps.
#. It computes the indicators of the state it reached.
#. It marks the cells below ``maximum_refinement_level`` to refine by
   ``refine_fraction``, and the cells to coarsen by ``coarsen_fraction``.
#. It changes the mesh, carries the state to the new mesh, and builds the
   problem again on it from the input.
#. It goes on from the time it reached, with the time step that the step
   control chose before the change.

The history of the post-processors and the output files run on across the
meshes, in one table, one numbering of the files and one collection.  The runs
on each mesh print nothing.  The loop prints one line at its start and one at
its end, and with the report ``"full"`` a line at each change of the mesh.  This steel plate at :math:`300\ \mathrm{K}` has its left side held
at :math:`400\ \mathrm{K}`, and the mesh follows the heat that enters from
that side (from ``tests/python/test_adaptivity.py``):

.. code-block:: python

   import dualmesh

   plate = dualmesh.problem(
       mesh={"rectangle": {"size": ["1 m", "1 m"], "cells": [8, 8], "cell_type": "triangle"}}
   )
   plate.physics("heat_conduction")
   plate.properties(
       thermal_conductivity="45 W/(m*K)",
       density="7850 kg/m^3",
       specific_heat_capacity="490 J/(kg*K)",
   )
   plate.initial_conditions(temperature="300 K")
   plate.boundary_conditions("left", temperature="400 K")
   plate.time(end_time="2000 s", initial_time_step="100 s", step_control={"by": "fixed"})
   plate.postprocessor("peak", maximum="temperature")
   plate.mesh_adaptivity(
       field="temperature",
       refine_fraction=0.5,
       coarsen_fraction=0.05,
       maximum_refinement_level=3,
       interval=2,
   )
   solution = plate.solve()

What a change of mesh carries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A change of mesh in a transient run carries two parts of the state:

- The value of every field at the nodes.
- The velocity of an equation with a second time derivative, such as the
  balance of momentum of a solid with a ``density``.  The run goes on from
  the velocity it reached, and the initial condition applies at the start of
  the run only.

Nothing else is carried: the problem on the new mesh is built again from the
input.  The method of the problem decides how the values are carried:

- Under FEM, a node that the old mesh has keeps its value, and a new node
  takes the value of the old solution at its position, interpolated in the
  old cell that holds it.  Refinement alone carries the fields as the same
  functions.  Coarsening removes nodes, and their values with them.
- Under DMCDM, each field keeps its content: the sum over the nodes of its
  value times the measure of the control domain of the node.  A node that
  the coarsening removes gives its content to the corners of the coarse cell
  that holds it, and each corner takes the mean of its own value and the
  values it receives, weighted by their measures.  A new node takes the
  interpolated value, as under FEM.  Take a constant density :math:`\rho` in
  :math:`\mathrm{kg/m^3}` and specific heat capacity :math:`c_p` in
  :math:`\mathrm{J/(kg\,K)}`.  On triangles and tetrahedra in Cartesian
  coordinates, the content of the temperature times :math:`\rho c_p` is then
  the heat that the lumped heat capacity of conduction stores.  An insulated
  body without a heat source therefore keeps its heat through every change
  of mesh.  DMCDM carries values this way on linear elements only.  On
  ``Tri6`` and ``Tet10``, whose shape functions are negative at some removed
  nodes, it interpolates them as FEM does.

The steady loop of the input and of Python carries the solution in the same
way, as the start of the next solve.

Adapt the mesh from Python
--------------------------

Run the loop
~~~~~~~~~~~~

:func:`~dualmesh.solve_with_adaptive_refinement` runs the steady loop at the
object level.  It takes a function that builds the problem on a given mesh,
and calls it on every new mesh:

.. code-block:: python

   import dualmesh


   def build(mesh):
       problem = dualmesh.Problem(mesh)
       problem.add_variable("u")
       problem.add_kernel("diffusion", variable="u")
       problem.add_kernel("body_force", variable="u", value=1.0)
       problem.add_boundary_condition("Dirichlet", mesh.sideset_names(), variable="u", value=0.0)
       return problem


   mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 4, 4, element_type="Tri3")
   result = dualmesh.solve_with_adaptive_refinement(build, mesh, variable="u", num_cycles=4)
   print(result.summary())
   problem = result.problem

Each cycle builds the problem on the current mesh, sets the solution of the
last mesh carried to this one as the start, solves, and computes the
indicators of ``variable``.  ``variable`` is the name of a variable, or a
list of names, such as the components of a velocity, whose indicators add as
squares.  The loop then marks the cells with ``marker``, and with
``coarsen_fraction`` the cells to coarsen, and changes the mesh.  The steady
loop of the input is this function.

The parameters are:

``build_problem``
   A function that takes a mesh and returns a :class:`~dualmesh.Problem`
   defined on it, ready to solve.  Building the problem again on each mesh
   keeps the conditions and the properties as you wrote them.

``mesh``
   The initial mesh.

``num_cycles`` (default 3)
   The most solves the loop takes, at least 1.

``marker`` (default :func:`~dualmesh.mark_by_error_fraction`)
   A function that takes the array of indicators, one per cell, and returns
   a boolean array of the cells to refine.

``max_elements`` (default ``None``)
   The loop stops once the mesh has this many cells or more.

``callback`` (default ``None``)
   ``callback(cycle, problem, indicators)`` runs after every solve, for
   example to write output or record a convergence history.

``report`` (default ``"full"``)
   ``"full"`` prints a line per cycle and a summary, ``"summary"`` one line
   and the summary, and ``"none"`` nothing.  The solves inside the loop print
   nothing.

``boundaries`` (default ``None``)
   The curved sides to follow (see `Follow a curved side`_).

``coarsen_fraction`` (default 0)
   The share of the total squared error that the coarsened cells may carry,
   as the key of the input sets it.  With 0, the loop refines only.

``maximum_refinement_level`` (default ``None``)
   The most refinement levels a cell may have above its cell of the initial
   mesh.  With ``None``, the levels are unbounded.

The other keyword arguments are the solver options of
:meth:`~dualmesh.Problem.solve`, for every solve of the loop.

The loop stops after ``num_cycles`` solves, when the mesh has reached
``max_elements``, or when the marker marks no cell below the maximum level.  It returns an
:class:`~dualmesh.AdaptivityResult` with ``problem`` and ``mesh``, the
problem and the mesh of the last cycle, and one entry per cycle in
``num_elements``, ``num_dofs`` and ``estimated_error``.
:meth:`~dualmesh.AdaptivityResult.summary` formats them as a table, and
:meth:`~dualmesh.AdaptivityResult.write_json` writes them to a file.

Choose the marking rule
~~~~~~~~~~~~~~~~~~~~~~~

Three functions turn the indicators :math:`\eta_K` of the cells :math:`K`
into a boolean array of the cells to refine:

:func:`~dualmesh.mark_by_error_fraction` ``(indicators, fraction=0.5)``
   Bulk marking [Dorfler1996]_: the smallest set of cells whose
   :math:`\eta_K^2` sum to ``fraction`` of the total.

:func:`~dualmesh.mark_by_fraction` ``(indicators, fraction=0.3)``
   The ``fraction`` of the cells with the largest indicators, rounded up.

:func:`~dualmesh.mark_by_threshold` ``(indicators, threshold)``
   Every cell whose indicator exceeds ``threshold``, in the unit of the
   indicator.

The first two take ``fraction`` from above 0 to 1, and mark no cell when
every indicator is zero.  To pass a parameter, wrap the function:

.. code-block:: python

   result = dualmesh.solve_with_adaptive_refinement(
       build,
       mesh,
       variable="u",
       num_cycles=12,
       marker=lambda indicators: dualmesh.mark_by_error_fraction(indicators, 0.4),
   )

Follow a curved side
~~~~~~~~~~~~~~~~~~~~

``boundaries`` maps the names of side sets to projections.  A projection
takes a point ``(x, y, z)`` and returns the point of the curve that the node
belongs on.  After each refinement, the loop moves the nodes of each side set
onto its curve, so that refining there also refines the geometry.  A node
that hangs on the side of a larger neighbour stays on the straight side of
that neighbour until the side is split.  This quarter annulus follows the
unit circle on its outer side (from ``tests/python/test_adaptivity.py``):

.. code-block:: python

   import dualmesh
   import numpy

   mesh = dualmesh.generate_annulus_mesh(0.5, 1.0, 2, 8, 0.0, 90.0, element_type="Quad4")


   def build(mesh):
       problem = dualmesh.Problem(mesh)
       problem.add_variable("u")
       problem.add_kernel("diffusion", variable="u")
       problem.add_boundary_condition("Dirichlet", "outer", variable="u", value=0.0)
       problem.add_boundary_condition("Dirichlet", "inner", variable="u", value=1.0)
       return problem


   def onto_circle(point):
       radius = numpy.hypot(point[0], point[1])
       return (point[0] / radius, point[1] / radius, 0.0)


   result = dualmesh.solve_with_adaptive_refinement(
       build, mesh, "u", num_cycles=4, boundaries={"outer": onto_circle}
   )

Refine once
~~~~~~~~~~~

:func:`~dualmesh.refine_marked` refines the marked cells of a mesh once, with
the neighbours that the refinement needs, and returns
``(refined_mesh, parents)``.  ``marked`` holds one flag per cell, and
``parents[i]`` is the index of the cell of ``mesh`` that cell ``i`` of the
refined mesh comes from, which carries a field stored per cell to the new
mesh.  Each call starts a new refinement history.  For repeated refinement,
use :func:`~dualmesh.solve_with_adaptive_refinement`, which keeps one history
for all its cycles, as newest-vertex bisection of triangles needs.

:meth:`~dualmesh.Problem.error_indicator` returns the indicator of each cell
for one variable, to mark the cells by a rule of your own.

Adaptive runs in parallel
-------------------------

The adaptive loops run on several processes (see :doc:`solving`).  Every process holds
the whole mesh and its refinement history, and marks it from the indicators
of the whole mesh, which :meth:`~dualmesh.Problem.gathered_error_indicator`
returns.  Every process therefore adapts the mesh in the same way.

A transient adaptive run is available from the input only.
