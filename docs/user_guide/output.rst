Results and output
==================

A solve leaves the values of the unknowns at the nodes of the mesh. This page shows how you turn
them into the quantities you need: fields, values at points, gradients, integrals, flows through
boundaries, post-processors and files for ParaView, a spreadsheet or another program.

The first section covers the input, which you write as a YAML or TOML case file or build with
:func:`dualmesh.problem`. The other sections cover the object-level API, :class:`dualmesh.Problem`.

.. contents::
   :local:
   :depth: 2

Results of the input
--------------------

The ``postprocessors`` and ``output`` sections of the input set what a run computes and writes.
This example from ``examples/bus_bar.py`` adds five post-processors and an output section:

.. code-block:: python

   bar = dualmesh.problem(
       mesh={"rectangle": {"size": ["0.1 m", "0.05 m"], "cells": [10, 5]}}, method="DMCDM"
   )
   # The physics, properties and boundary conditions of examples/bus_bar.py go here.
   bar.postprocessor("heat_through_left", boundary_flux="heat_flux", boundary="left")
   bar.postprocessor("heat_through_right", boundary_flux="heat_flux", boundary="right")
   bar.postprocessor("temperature_bottom", point_value="temperature", point=["0.05 m", "0 m"])
   bar.postprocessor("temperature_top", point_value="temperature", point=["0.05 m", "0.05 m"])
   bar.postprocessor("hottest", maximum="temperature")
   bar.output(file="bus_bar", interval=1, fields=["temperature"])

   solution = bar.solve()
   for name in solution.postprocessor_names():
       print(f"{name}: {solution.postprocessor(name)[-1]:.6g} {solution.unit(name)}")

In a case file the same post-processors read as in ``examples/heated_plate.yaml``:

.. code-block:: yaml

   postprocessors:
     heat_through_left: {boundary_flux: heat_flux, boundary: left}
     peak_temperature: {maximum: temperature}

:doc:`input` lists every key of both sections.

Post-processors in the input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A post-processor reduces a field to one number, which the run computes after a steady solve, and
at the start and after every accepted time step of a transient run. Each post-processor has a
name of your choice and exactly one of these keys:

``integral``, ``average``, ``maximum``, ``minimum``
   The integral over the domain, the average over the domain, or the largest or smallest nodal
   value of a field. The integral has the unit of the field times
   :math:`\mathrm{m}^d`, with :math:`d` the dimension of the mesh, or :math:`\mathrm{m}^3` in
   axisymmetric coordinates, where the integral covers the body of revolution. The input refuses a
   post-processor over one ``region``.

``point_value``
   The value of a field at ``point``, a list of one length for each direction of the mesh. A
   component such as ``displacement_x`` or ``stress_yy`` names one component of a field. A stress
   component is available as a ``point_value`` of a steady solid at small strain in Cartesian
   coordinates, in Pa.

``boundary_flux``
   The flow of a flux out of the body through ``boundary``. ``heat_flux`` is the flux available in
   this version. Its flow is in W on a three-dimensional or axisymmetric mesh, in
   :math:`\mathrm{W/m}` on a two-dimensional one, and in :math:`\mathrm{W/m^2}` on a segment.

The results of a run
~~~~~~~~~~~~~~~~~~~~

:meth:`problem.solve <dualmesh.problem.solve>` returns a :class:`dualmesh.results` object. Its
``report`` argument sets what the run prints: ``"summary"`` (the default), ``"full"`` or
``"none"``, as described in :ref:`output-study-report`.

``field(name)``
   The values of a field such as ``temperature`` or ``displacement``, an array with one row for
   each node and one column for each component.

``coordinates(name)``
   The coordinates of the nodes of a field in m, one row for each node.

``field_names()`` and ``postprocessor_names()``
   The names of the fields and of the post-processors.

``postprocessor(name)`` and ``unit(name)``
   The values of a post-processor in the units of the International System of Units (SI), one
   for each row of the post-processor table, and
   the unit, for example ``"K"``.

``times`` and ``output_times``
   The times of the accepted time steps, and the times of the rows of the post-processor table. A
   steady run has one row, at 0 s.

``table()``
   The post-processor table as a dictionary of columns. The keys are the headers of the
   comma-separated values (CSV) file that the next section describes: ``time [s]``, then
   ``<name> [<unit>]`` for each post-processor.

``examples/transient_slab.py`` compares a field with an exact solution:

.. code-block:: python

   solution = slab().solve()
   x = solution.coordinates("temperature")[:, 0]
   computed = solution.field("temperature")[:, 0]

The files of the input
~~~~~~~~~~~~~~~~~~~~~~

An ``output`` section has three keys:

``file``
   The start of every file name. A relative path starts from the folder of the case file, or
   from the working directory in :func:`dualmesh.problem`.

``interval``
   A positive integer. A transient run writes the fields at its first accepted time step, at
   every ``interval``-th accepted step after it, and at its end. A steady run writes them once.

``fields``
   The fields that the files hold, among the fields that the physics solve for:
   ``temperature``, ``displacement``, ``velocity`` and ``pressure``.

The run writes these files:

* ``<file>_0000.vtu``, ``<file>_0001.vtu`` and so on: the fields at each output time, in the VTK
  unstructured grid (VTU) format of the Visualization Toolkit (VTK).
* ``<file>.pvd``: the ParaView data (PVD) collection that lists the time of each VTU file.
  ParaView opens it as one time series.
* ``<file>.csv``: the post-processor table, when the input has post-processors, with the unit of
  each column in brackets in its header.

Results of the object-level API
-------------------------------

.. _output-study-report:

What a study prints
~~~~~~~~~~~~~~~~~~~

Every study takes a ``report`` argument with three levels:

``"full"``
   The default of the object-level API. The study prints the problem with every parameter and
   the defaults marked, then its progress (the Newton iterations, or a line for each time step or
   mesh), then the result.

``"summary"``
   The study prints one line that names it, then the result.

``"none"``
   The study prints nothing.

The studies are :meth:`~dualmesh.Problem.solve`, :meth:`~dualmesh.Problem.solve_transient`,
:meth:`~dualmesh.Problem.solve_eigenvalue`, :meth:`~dualmesh.Problem.solve_buckling`,
:meth:`~dualmesh.Problem.sensitivities`,
:meth:`ManufacturedSolution.convergence_study <dualmesh.mms.ManufacturedSolution.convergence_study>`
and :func:`~dualmesh.solve_with_adaptive_refinement`. A study that runs other studies prints its
own report alone, so the solves of a convergence study print nothing. In a run on several
processes, the first process prints. When your script runs a study in a loop and prints its own
table, pass ``report="none"``:

.. code-block:: python

   for method in ("FEM", "DMCDM"):
       problem = build(method)
       problem.solve(report="none")
       print(method, problem.sample("temperature", [[0.05, 0.0]])[0])

Here ``build`` is your function that sets up the problem with the finite element method (FEM) or
the dual mesh control domain method (DMCDM).

Result objects
~~~~~~~~~~~~~~

The studies return result objects that share these methods:

``summary()``
   The result as text: a title, the answer, then the tables.

``tables``
   A dictionary of the named tables of the result, each a :class:`~dualmesh.Table`.

``write_csv(path, table=None)``
   Writes the main table, or the table that ``table`` names, to a CSV file. The header names each
   column with its unit in parentheses, for example ``temperature (K)``.

``write_json(path)`` and ``to_dict()``
   All the numbers of the result, as a JavaScript Object Notation (JSON) file or a dictionary.

The result objects are the :class:`~dualmesh.SolveResult` of a solve, whose table ``newton`` has
one row for each Newton iteration, the :class:`~dualmesh.AdaptivityResult` of adaptive refinement,
with one row for each cycle, and the :class:`~dualmesh.mms.ConvergenceResult` of a convergence
study, with one row for each mesh. A :class:`~dualmesh.SolveResult` also holds ``converged``,
``total_iterations`` and ``linear_iterations``, and after a transient solve ``time_steps``,
``rejected_steps`` and ``next_time_step``.

.. code-block:: python

   result = problem.solve(report="none")
   print(result.summary())
   result.write_csv("newton_history.csv")

Values at nodes and points
~~~~~~~~~~~~~~~~~~~~~~~~~~

Both methods place one unknown of each variable at each node, so ``problem.values("u")[i]`` is
the value of ``u`` at node ``i``, whose coordinates are ``problem.node_coordinates()[i]``.

.. code-block:: python

   problem.values("temperature")  # The whole field
   problem.set_values("temperature", array)  # Overwrite it
   problem.values("temperature")[[0, 5, 9]]  # At chosen nodes
   problem.sample("temperature", [[0.05, 0.0], [0.08, 0.0]])  # At points
   problem.sample_gradient("temperature", [[0.05, 0.0]])  # The gradient at points

:meth:`~dualmesh.Problem.sample` interpolates a variable at any points, for a comparison with a
published table or a profile along a line. It finds the element that holds each point and
evaluates the interpolation there, and it returns NaN (not a number) for a point outside the
mesh.
:meth:`~dualmesh.Problem.sample_gradient` returns one row of three components for each point.

:meth:`~dualmesh.Problem.node_at` returns the node nearest to a point, within ``tolerance``
(:math:`10^{-9}` by default), and :meth:`~dualmesh.Problem.nodes_where` returns the nodes whose
coordinates satisfy a predicate of ``(x, y, z)``. In a distributed problem,
:meth:`~dualmesh.Problem.values` holds the nodes of its process, and :meth:`~dualmesh.Problem.gathered_values` returns the values at
every node of the whole mesh on every process.

Gradients, fluxes and properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These methods evaluate at the element centroids and return one row for each element:

.. code-block:: python

   problem.gradient_at_centroids("temperature")
   problem.kernel_flux_at_centroids(conduction)  # A kernel that add_kernel returned
   problem.property_at_centroids("stress")

:meth:`~dualmesh.Problem.property_at_centroids` returns any property that a property object
declares. ``small_strain_stress`` declares ``stress``, ``strain``, ``elastic_strain``,
``volumetric_strain`` and ``von_mises_stress``. Stress and strain have six components in the
Voigt order :math:`(xx, yy, zz, yz, xz, xy)`, in which :math:`zz` is the hoop component of an
axisymmetric problem.

Integrals
~~~~~~~~~

.. code-block:: python

   problem.integrate("temperature")
   problem.element_integrals("temperature")  # One value for each element
   problem.domain_volume()

:meth:`~dualmesh.Problem.integrate` integrates a variable over the domain with the coordinate
factor of the problem. In axisymmetric coordinates the factor is :math:`2\pi r`, with :math:`r`
the radial coordinate in m, so the integral covers the body of revolution. In spherical
coordinates the factor is :math:`4\pi r^2`. :meth:`~dualmesh.Problem.domain_volume`,
:meth:`~dualmesh.Problem.block_volume` and :meth:`~dualmesh.Problem.boundary_area` measure the
domain, a list of blocks and a list of side sets with the same factor.
:meth:`~dualmesh.Problem.maximum` and :meth:`~dualmesh.Problem.minimum` return the largest and
smallest nodal values of a variable.

When you know the exact solution, :meth:`~dualmesh.Problem.error_norms` returns the
:math:`L^2` norm of the error of a variable and the :math:`H^1` seminorm, the :math:`L^2` norm of
the error of its gradient.

Flows through a boundary
~~~~~~~~~~~~~~~~~~~~~~~~

Each physics names the flow of its flux through a boundary, the quantity that leaves the body,
with the sign of the flux of the boundary conditions:

.. code-block:: python

   heat.heat_flow("left")  # W: the heat leaving through left
   solid.reaction_force("bottom")  # N: {"x": ..., "y": ...}, the force of the support on the body
   flow.force("cylinder")  # N: {"x": drag, "y": lift}, the force of the fluid on the wall
   flow.flow_rate("outlet")  # m^3/s: the volume of fluid leaving through outlet
   pde.boundary_flux("left")  # The flow of the flux of a general_form_PDE
   problem.boundary_flux("temperature", "left", per_node=True)  # {node: flow}

Here ``heat``, ``solid``, ``flow`` and ``pde`` are the physics that
:meth:`~dualmesh.Problem.add_physics` returned. In a plane problem the flows are per unit
thickness: :math:`\mathrm{W/m}`, :math:`\mathrm{N/m}` and :math:`\mathrm{m^2/s}`.

On a boundary that prescribes the value of the unknown, the flow is the reaction there, the
residual of the equation at the nodes of the boundary. In DMCDM [Reddy2024]_ this residual is the
balance of the control domain :math:`V_i` of each boundary node :math:`i`, and the part of the
surface of :math:`V_i` on the boundary :math:`\partial\Omega` closes it. The reaction at node
:math:`i` is the flow of the flux :math:`\mathbf{F}` along the outward normal :math:`\mathbf{n}`
through that part,

.. math::

   Q_i = \int_{\partial V_i \cap \partial\Omega} \mathbf{F} \cdot \mathbf{n} \,\mathrm{d}S ,

in W for a heat flux. On any other boundary the flow is the integral of the flux that the
boundary conditions prescribe, so a boundary without a condition carries none.

:meth:`~dualmesh.Problem.boundary_flux` with ``per_node=True`` returns the share of each node of
the boundary, and refuses a distributed problem. A node on two boundaries of prescribed values,
such as a corner, carries the reaction of both and appears in both.
:meth:`~dualmesh.Problem.kernel_boundary_flux` integrates the normal flux of one kernel over a
side set, which is the part of the flux that one term of the equation carries.

Post-processors
~~~~~~~~~~~~~~~

A post-processor is a named number that the problem computes after every steady solve, and at
the start and after every accepted time step of a transient solve:

.. code-block:: python

   problem.add_postprocessor("average", "mean_temperature", variable="temperature")
   problem.add_postprocessor("heat_flow", "heat_out", boundary="left")
   problem.solve()
   print(problem.postprocessor_table())
   problem.write_postprocessor_csv("history.csv")

:meth:`~dualmesh.Problem.add_postprocessor` takes the type, the name and the parameters of the
type:

========================  ===================================  ===========================================
Type                      Parameters                           Value
========================  ===================================  ===========================================
``integral``              ``variable``                         The integral over the domain
``average``               ``variable``                         The integral divided by the volume
``point_value``           ``variable``, ``point``              The interpolated value, NaN outside the mesh
``maximum``               ``variable``                         The largest nodal value
``minimum``               ``variable``                         The smallest nodal value
``boundary_flux``         ``variable``, ``boundary``           :meth:`~dualmesh.Problem.boundary_flux`
``heat_flow``             ``boundary``, ``physics``            The heat leaving, in W
``reaction_force``        ``boundary``, ``direction``,         One component of the reaction, in N
                          ``physics``
``force``                 ``boundary``, ``direction``,         One component of the force of a fluid, in N
                          ``physics``
``flow_rate``             ``boundary``, ``physics``            The volume flow leaving, in
                                                               :math:`\mathrm{m^3/s}`
``kernel_boundary_flux``  ``kernel``, ``boundary``             :meth:`~dualmesh.Problem.kernel_boundary_flux`
========================  ===================================  ===========================================

``direction`` is ``"x"`` (the default), ``"y"`` or ``"z"``. ``physics`` names the physics, and
you can leave it out when the problem has one physics of that type.
:meth:`~dualmesh.Problem.postprocessor_values` returns the history as
``{"time": array, name: array, ...}``, :meth:`~dualmesh.Problem.postprocessor_table` returns it as
aligned text, and :meth:`~dualmesh.Problem.write_postprocessor_csv` writes it as CSV, with one row
for each evaluation.

The error indicator
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   indicators = problem.error_indicator("temperature")  # One value for each element

:meth:`~dualmesh.Problem.error_indicator` follows Zienkiewicz and Zhu [ZienkiewiczZhu1987]_. It
averages the element gradients of the computed field :math:`u_h` onto the nodes, weighted by the
share of each element in the control domain of each node, and interpolates the nodal averages
over each element as the recovered gradient :math:`\mathbf{G}(u_h)`. The indicator of element
:math:`e` is

.. math::

   \eta_e = \left( \int_e \left| \nabla u_h - \mathbf{G}(u_h) \right|^2 \mathrm{d}V \right)^{1/2} ,

in the unit of :math:`u_h` times :math:`\mathrm{m}^{d/2-1}`, with :math:`d` the dimension. It
shows which elements carry most of the error, which the marking rules of :doc:`adaptivity` use,
and it gives no bound on the size of the error.

Files
~~~~~

A study writes the files of its fields when you give it an :class:`~dualmesh.Output` group as
``output``. The group sets where, when, what and in which format:

.. code-block:: python

   output = dualmesh.Output(
       directory="results",
       file_base="slab",
       interval=10.0,
       fields=["temperature"],
       formats=["vtu", "csv"],
   )
   problem.solve_transient(end_time=100.0, time_step=1.0, output=output)

``directory`` and ``file_base``
   The directory, which the study makes when it doesn't exist, and the start of each file name.
   The defaults are ``"."`` and ``"output"``.

``times`` or ``interval``
   The output times of a transient study in s: a list of increasing times, or the time between
   two outputs from the start time. Give at most one of the two. The time steps land on the
   output times. With ``interval``, the study also writes its end time. Without either, a
   transient study writes its start and end times. A steady solve and an eigenvalue study write once, at the end, and
   refuse both keys.

``fields``
   The variables that the files hold. The default is every variable.

``formats``
   ``"vtu"``, the default, for a VTU file, and ``"csv"`` for a CSV file with one row for each
   node. A distributed problem writes VTU alone.

``checkpoint_interval``
   The number of accepted steps between two checkpoints of a transient study, as
   :ref:`output-checkpoints` describes.

A steady solve writes ``slab.vtu``, and an eigenvalue study writes its first mode in the same
way. A transient study numbers its files ``slab_00000.vtu``, ``slab_00001.vtu`` and so on, and
writes the PVD collection ``slab.pvd``. A distributed run writes one VTU file for each process and
a parallel VTU (PVTU) index, which ParaView opens as one data set.

You can also write files at any point of a script:

.. code-block:: python

   problem.write_vtu("result.vtu")
   problem.write_vtu("stress.vtu", cell_properties=["stress"])
   problem.write_csv("nodes.csv", variables=["temperature"])
   problem.write_mesh_file("result.msh")

:meth:`~dualmesh.Problem.write_vtu` writes the mesh, the variables of ``fields`` (every variable
by default) as point data, and the properties of ``cell_properties`` as cell data. Each data array of a variable carries its unit in the attribute ``units``. Quadratic
elements become the quadratic VTK cells of the same nodes, so ParaView can draw their curved
sides. :meth:`~dualmesh.Problem.write_csv` writes one row for each node, with its coordinates and
the values of ``variables`` (every variable by default), and a header with the unit of each
column. :meth:`~dualmesh.Problem.write_mesh_file` writes the mesh and every nodal field through
:func:`dualmesh.write_mesh` in any format of the meshio library [meshio]_, such as Gmsh
[Gmsh2009]_ or Exodus II.

Each physics gives the SI unit of each variable that it makes: ``K`` for a temperature, ``m`` for
a displacement, ``rad`` for a rotation, ``m/s`` for a velocity and ``Pa`` for a pressure. A
variable that you add yourself has a unit when you give one:

.. code-block:: python

   problem.add_variable("temperature", unit="K")

.. _output-checkpoints:

Checkpoints and restarts
~~~~~~~~~~~~~~~~~~~~~~~~

A checkpoint holds what a transient run needs to continue: the time, the next step size, the
number of steps taken, the solution, the history of the material and the history of the
post-processors. With ``checkpoint_interval``, a
transient study writes ``<file_base>.chk`` every that many accepted steps and after its last step,
and replaces the file each time. When the problem has post-processors, their history goes beside
it, in ``<file_base>.chk.postprocessors.json``. ``restart`` continues a run from a checkpoint, on any
number of processes:

.. code-block:: python

   output = dualmesh.Output(directory="results", file_base="plate", checkpoint_interval=50)
   problem.solve_transient(end_time=1e6, time_step=1e3, output=output)

   # Later, in a new process that builds the same problem:
   problem.solve_transient(end_time=2e6, time_step=1e3, restart="results/plate.chk")

``start_time`` stays the start of the original run, so the output times and the file numbers
continue those of the original run. Checkpoints need a build with the Portable, Extensible Toolkit
for Scientific Computation (PETSc), which every build with the Message Passing Interface (MPI)
has.

Helpers for comparisons
~~~~~~~~~~~~~~~~~~~~~~~

``dualmesh.postprocess`` holds small functions for comparisons with references:

* ``sample_line(problem, variable, start, end, num_points=21)`` samples a variable along a
  segment and returns the arc length and the values.
* ``values_on_line(problem, variable, coordinates, axis=0, other=0.0)`` samples a variable at
  given coordinates along one axis.
* ``relative_error(computed, reference)`` and ``max_relative_error(computed, reference)`` compare
  values with a reference.
* ``convergence_rates(mesh_sizes, errors)`` returns the observed order between successive
  meshes.
* ``comparison_table(labels, computed, reference, title="")`` formats computed values against
  reference values, with the relative error of each.

.. code-block:: python

   from dualmesh import postprocess

   distance, values = postprocess.sample_line(problem, "temperature", [0.0, 0.0], [1.0, 0.0], 101)

Looking at the results
~~~~~~~~~~~~~~~~~~~~~~

Open the VTU file, or the PVD collection of a transient run, in ParaView. The point data carry
the variable names and the cell data the property names. The *Warp By Vector* filter on a
displacement shows the deformed shape, and *Plot Over Line* draws a profile for a comparison with
a published curve.
