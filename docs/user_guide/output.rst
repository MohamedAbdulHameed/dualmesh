Results and output
==================

A solved problem holds a vector of numbers.  This chapter describes how that
vector is converted into the quantities of interest: field values at known
positions, gradients, stresses, integrals, reactions, and files that another
program can read.

.. contents::
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

Output design
-------------

Every module of dualmesh gives its output in the same four ways, so that you
learn them once:

* The ``report`` parameter of a study sets what the study prints.
* Every table is a :class:`~dualmesh.Table`, whose CSV file names each column
  with its unit.
* Every result has the same methods: ``summary()``, ``tables``,
  ``write_csv()``, ``write_json()`` and ``to_dict()``.
* Every study that calculates fields writes its files through one
  :class:`~dualmesh.Output` group.

The sections that follow describe each of these.

What a study prints
~~~~~~~~~~~~~~~~~~~

Every study takes a ``report`` parameter with three levels:

``"full"``
   The default. The study prints its inputs, with the defaults marked, then
   its progress (the Newton iterations, a line for each time step, run or
   mesh), then the result.

``"summary"``
   The study prints one line that names it, then the result.

``"none"``
   The study prints nothing.

The studies are :meth:`~dualmesh.Problem.solve`,
:meth:`~dualmesh.Problem.solve_transient`,
:meth:`~dualmesh.Problem.solve_eigenvalue`,
:meth:`~dualmesh.fuel.FuelRod.run`,
:meth:`~dualmesh.fuel.TrisoParticleModel.run`, the uncertainty studies
(:func:`~dualmesh.uq.propagate`, :func:`~dualmesh.uq.sobol` and
:func:`~dualmesh.uq.calibrate`), the convergence study
(:meth:`~dualmesh.mms.ManufacturedSolution.convergence_study`) and
:func:`~dualmesh.solve_with_adaptive_refinement`.

A study that runs other studies prints only its own report. For example, an
uncertainty study of a fuel rod prints a line for each run and its summary,
and the 64 rods that it runs print nothing. When your script runs a study in
a loop and prints its own table, give ``report="none"`` to the study in the
loop.

.. code-block:: python

   for method in ("fem", "dmcdm"):
       problem = build(method)
       problem.solve(report="none")
       print(method, problem.sample("temperature", [[0.05, 0.0]])[0])

Tables and results
~~~~~~~~~~~~~~~~~~

Every study returns a result with the same methods:

``summary()``
   The report of the result as text: a title, the answer first, then the
   tables.

``tables``
   A dictionary of the named tables of the result, each a
   :class:`~dualmesh.Table`.

``write_csv(path, table=None)``
   Writes one table to a CSV file: the main table, or the table that
   ``table`` names.

``write_json(path)`` and ``to_dict()``
   Give all the numbers of the result, for a program.

A CSV file has one header line, in which each column has its unit, for
example ``temperature (K)``. A spreadsheet, pandas and gnuplot read the file
without changes.

.. code-block:: python

   result = problem.solve(report="none")
   print(result.summary())
   result.write_csv("newton_history.csv")

The results are the :class:`~dualmesh.SolveResult` of the solves (one row for
each Newton iteration), the :class:`~dualmesh.AdaptivityResult` of adaptive
refinement (one row for each cycle), the
:class:`~dualmesh.mms.ConvergenceResult` of a convergence study (one row for
each mesh), the results of the uncertainty studies (one row for each run,
index or parameter), and the results of the fuel rod and the TRISO particle.

The files of the fields
~~~~~~~~~~~~~~~~~~~~~~~

A study writes field files only when you give it an :class:`~dualmesh.Output`
group in its ``output`` parameter. The group sets where, when, what and in
which format:

.. code-block:: python

   output = dm.Output(directory="results", file_base="slab", interval=10.0, fields=["temperature"], formats=["vtu", "csv"])
   problem.solve_transient(end_time=100.0, time_step=1.0, output=output)

``directory`` and ``file_base``
   The directory, which the study makes when it doesn't exist, and the start
   of each file name.

``times`` or ``interval``
   The output times of a transient study: a list of times, or an interval
   from the start time. Give only one of the two. The time steps land on the
   output times. Without either, a transient study writes the fields at its
   start and end times. A steady solve and an eigenvalue study write the
   fields once, at the end.

``fields``
   The variables that the files contain. The default is all the variables.

``formats``
   ``vtu`` (a VTK unstructured grid for ParaView and VisIt, the default) or
   ``csv`` (one row for each node, or for each cell and boundary face of the
   cell-centred method), or both.

A transient study numbers its files ``slab_00000.vtu``, ``slab_00001.vtu``
and so on, and writes the ParaView collection ``slab.pvd``, which lists the
time of each file. ParaView opens the collection as one time series.

The fuel rod has the same parameters in :class:`~dualmesh.fuel.RodOutput`,
with the unit of burnup. Its files are the tables of the rod, in the formats
``csv`` and ``json``.

Units
~~~~~

Each physics gives the SI unit of each variable that it makes: ``K`` for a
temperature, ``m`` for a displacement, ``rad`` for a rotation, ``m/s`` for a
velocity, ``Pa`` for a pressure and ``1/(m^2 s)`` for a neutron flux. The CSV
files name each column with its unit, and the VTU files give the unit in the
``units`` attribute of each data array. A variable that you add with
:meth:`~dualmesh.Problem.add_variable` has a unit only when you give one:

.. code-block:: python

   problem.add_variable("temperature", unit="K")

Location of the unknowns
------------------------

Three of the four methods place their unknowns at the mesh nodes, and the fourth
places them at element and boundary face centroids.  The rest of this chapter
depends on this distinction.

For ``dmcdm``, ``fem`` and ``hfvm`` there is one unknown per mesh node per
variable, and ``problem.values("u")[i]`` is the value at ``mesh.points()[i]``.

For ``zfvm`` the unknowns are the *entities* of the cell-centred mesh: first one
per element, at its centroid, and then one per boundary face, at its centroid.
There are therefore more unknowns than elements, and their numbering is
unrelated to the node numbering.

So that post-processing code need not branch on the method, the library exposes
the positions of the degrees of freedom directly:

.. code-block:: python

   values = problem.values("temperature")  # one number per degree of freedom
   points = problem.entity_points()  # (num_dofs, 3), the same order

``points[i]`` is the position of ``values[i]`` for every method.  Code written
against :meth:`~dualmesh.Problem.entity_points` works unchanged for all four
discretisations, and code written against ``mesh.points()`` silently gives the
wrong answer for ``zfvm``.  Use ``entity_points``.

:attr:`~dualmesh.Problem.is_cell_centered` reports whether the method is
cell-centred, for code that must distinguish the two cases.

Reading values out
------------------

.. code-block:: python

   problem.values("temperature")  # the whole field
   problem.set_values("temperature", array)  # overwrite it
   problem.values_at_nodes("temperature", [0, 5, 9])  # at chosen nodes
   problem.sample("temperature", [[0.05, 0.0], [0.08, 0.0]])

:meth:`~dualmesh.Problem.sample` interpolates at arbitrary points, as required
for a comparison with a published table or for a plot along a line.  It
locates the element containing each point and evaluates the interpolation there,
returning ``nan`` for a point outside the mesh.  For the cell-centred method it
reconstructs linearly from the cell value and the cell gradient, so that the
sampled field is second-order accurate, while the cell values alone are
piecewise constant.

:meth:`~dualmesh.Problem.node_at` and :meth:`~dualmesh.Problem.nodes_where` find
node numbers by position and by predicate, so that a boundary condition or a
probe can be attached to a position in space independently of the node
numbering.

``dualmesh.postprocess`` collects the operations that are used most often:
``sample_line`` and ``values_on_line`` sample along a segment,
``relative_error`` and ``max_relative_error`` compare against a reference,
``convergence_rates`` fits the slope of an error sequence, and
``comparison_table`` formats a table of computed against published values of the
kind used throughout :doc:`/verification`.

Gradients, fluxes and properties
-----------------------------------------

.. code-block:: python

   problem.gradient_at_centroids("temperature")  # (num_elements, 3)
   problem.kernel_flux_at_centroids("conduction")  # (num_elements, 3)
   problem.property_at_centroids("stress")  # (num_elements, 6)

These are evaluated at element centroids, one row per element, because that is
where a piecewise quantity is best represented and where Gauss-point quantities
are most accurate.  ``property_at_centroids`` returns any property that a
property object declared, so ``"stress"``, ``"strain"`` and ``"volumetric_strain"`` are
available from ``linear_elastic_stress`` without further setup.  Stress and
strain use the Voigt ordering :math:`(xx, yy, zz, yz, xz, xy)`, and the ``zz``
component is filled in for plane strain and axisymmetric problems as well as
three-dimensional ones, because it is nonzero there.

Integrals
---------

.. code-block:: python

   problem.integrate("temperature")
   problem.boundary_flux_integral("conduction", "top")

:meth:`~dualmesh.Problem.integrate` integrates a variable over the whole domain,
with the coordinate factor included, so that for an axisymmetric problem the
result is the volume integral over the body of revolution, with the factor
:math:`2\pi r` applied to the integral over the :math:`(r, z)` mesh.  Setting a
variable to one everywhere and integrating it gives a simple and direct measure
of how well the mesh represents the domain.  The fourth-order convergence of a
curved quadratic boundary is measured in this way in :doc:`/theory/elements`.

:meth:`~dualmesh.Problem.boundary_flux_integral` integrates the normal flux of
one named kernel over one side set, which is the total heat flow through a face,
or the total force on it.

Reactions
---------

Reactions are treated in a separate section, because their direct computation is
one of the main reasons to use the dual mesh control domain method.

.. code-block:: python

   problem.reactions("temperature", "left")  # [(node, value), ...]
   problem.total_reaction("temperature", "left")  # their sum

In a finite element code a reaction is recovered after the solution, by
multiplying the assembled stiffness matrix by the solution and reading off the
rows that were constrained.  The result is correct, but it is an algebraic
by-product of the linear system, and the method itself computes no
corresponding physical quantity.

In the dual mesh control domain method the reaction is a physical quantity that
the method computes directly.  The control domain of a boundary node is a real
region of space, and part of its surface lies on the boundary of the domain.
The balance law over that control domain already contains the integral of the
normal flux over that part of the surface, which is the term that the
discretisation supplies to close the balance.  The reaction at a node is
therefore

.. math::

   Q_I = \int_{\partial CD_I \cap \partial \Omega} \mathbf{F} \cdot \mathbf{n} \, dS ,

i.e., a flux through a known area, computed as part of the method.  This is the
property that Reddy emphasises in [Reddy2024]_: the secondary variables (heat
flows, forces and moments) appear naturally on the control domain interfaces,
which is the physical appeal of the finite volume method, while the element
interpolation removes the ad hoc gradient reconstructions that the finite volume
method otherwise needs.

Because the control domains partition the domain exactly, the reactions sum to
the net flux through the boundary, and the test suite checks this identity.
``total_reaction`` returns that sum.

The error indicator
-------------------

.. code-block:: python

   indicators = problem.error_indicator("temperature")  # one per element

The indicator is one number per element: the square root of the integral over
that element of the squared difference between the computed gradient and a
smoother gradient recovered from it [ZienkiewiczZhu1987]_.  It identifies the
elements that carry most of the error, which is the information that the marking
rules of :doc:`/theory/adaptivity` require.  It indicates the distribution of the
error and gives no certified bound on its size.  It is unavailable for the
cell-centred method, whose unknowns are cell values, and a request for it with
that method raises an error that states this.

Writing files
-------------

.. code-block:: python

   problem.write_vtu("result.vtu")
   problem.write_vtu("result.vtu", cell_properties=["stress"])
   problem.write_csv("line.csv", variables=["temperature"])
   problem.write_mesh_file("mesh.msh")

:meth:`~dualmesh.Problem.write_vtu` writes a VTK unstructured grid file
containing the mesh, every variable as point data, and any property
named in ``cell_properties`` as cell data.  For the cell-centred method the
variables are written as cell data, because they are cell values.  Quadratic
elements are written as the corresponding quadratic VTK cell types (e.g.,
``VTK_TRIANGLE`` becomes ``VTK_QUADRATIC_TRIANGLE``), with the node permutation
that VTK's triquadratic hexahedron needs, so that a ``Hex27`` mesh is rendered
with its curved faces.

A transient study writes its files at the output times of its
:class:`~dualmesh.Output` group (see `The files of the fields`_). A
distributed run writes one ``.vtu`` file for each process and a ``.pvtu``
index, which ParaView opens as one data set. A distributed run writes the
``vtu`` format only.

:func:`dualmesh.write_mesh` writes the mesh alone, with optional point data,
through meshio [meshio]_, so that every one of the several dozen formats that
meshio supports (e.g., Gmsh [Gmsh2009]_, Exodus, XDMF and Abaqus) is available.

Looking at the results
----------------------

Open the ``.vtu`` file in ParaView.  Point data appear under the variable names,
and cell data under the property names.  Useful first steps are a
*Warp By Vector* filter on a displacement field to see the deformed shape, a
*Plot Over Line* filter to compare a profile against a published curve, and the
*Calculator* to form a derived quantity such as a von Mises stress from the
stress components.

For a line plot it is usually faster to sample in Python and plot there:

.. code-block:: python

   import numpy as np
   from dualmesh import postprocess

   x = np.linspace(0.0, 1.0, 101)
   values = problem.sample("temperature", np.column_stack([x, np.zeros_like(x)]))

which avoids writing and reading a file and gives the same numbers, because
``sample`` and the VTU writer read the same interpolation.
