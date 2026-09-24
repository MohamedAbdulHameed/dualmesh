Results and output
==================

A solved problem holds a vector of numbers.  This chapter is about turning that
vector into the quantities you actually wanted: field values at known positions,
gradients, stresses, integrals, reactions, and files another program can read.

.. contents::
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

Where the unknowns live
-----------------------

Three of the four methods put their unknowns at the mesh nodes, and the fourth
does not.  Everything in this chapter follows from that, so it is worth stating
plainly.

For ``dmcdm``, ``fem`` and ``hfvm`` there is one unknown per mesh node per
variable, and ``problem.values("u")[i]`` is the value at ``mesh.points()[i]``.

For ``zfvm`` the unknowns are the *entities* of the cell-centred mesh: first one
per element, at its centroid, and then one per boundary face, at its centroid.
There are therefore more unknowns than elements and no correspondence at all
with the node numbering.

Rather than have every piece of post-processing branch on the method, the
library exposes the positions of the degrees of freedom directly:

.. code-block:: python

   values = problem.values("temperature")     # one number per degree of freedom
   points = problem.entity_points()           # (num_dofs, 3), the same order

``points[i]`` is where ``values[i]`` lives, whatever the method.  Code written
against :meth:`~dualmesh.Problem.entity_points` works unchanged for all four
discretisations, and code written against ``mesh.points()`` silently gives the
wrong answer for ``zfvm``.  Use ``entity_points``.

:attr:`~dualmesh.Problem.is_cell_centered` reports which case you are in, when
you genuinely need to know.

Reading values out
------------------

.. code-block:: python

   problem.values("temperature")                     # the whole field
   problem.set_values("temperature", array)          # overwrite it
   problem.values_at_nodes("temperature", [0, 5, 9]) # at chosen nodes
   problem.sample("temperature", [[0.05, 0.0], [0.08, 0.0]])

:meth:`~dualmesh.Problem.sample` interpolates at arbitrary points, which is what
you want for comparing against a published table or plotting along a line.  It
locates the element containing each point and evaluates the interpolation there,
returning ``nan`` for a point outside the mesh.  For the cell-centred method it
reconstructs linearly from the cell value and the cell gradient, so it is
second-order accurate rather than piecewise constant.

:meth:`~dualmesh.Problem.node_at` and :meth:`~dualmesh.Problem.nodes_where` find
node numbers by position and by predicate, which is how a boundary condition or
a probe is attached to a place rather than to an index.

``dualmesh.postprocess`` collects the operations that come up repeatedly:
``sample_line`` and ``values_on_line`` sample along a segment,
``relative_error`` and ``max_relative_error`` compare against a reference,
``convergence_rates`` fits the slope of an error sequence, and
``comparison_table`` formats a table of computed against published values of the
kind used throughout :doc:`/verification`.

Gradients, fluxes and material properties
-----------------------------------------

.. code-block:: python

   problem.gradient_at_centroids("temperature")   # (num_elements, 3)
   problem.kernel_flux_at_centroids("conduction") # (num_elements, 3)
   problem.property_at_centroids("stress")        # (num_elements, 6)

These are evaluated at element centroids, one row per element, because that is
where a piecewise quantity is best represented and where Gauss-point quantities
are most accurate.  ``property_at_centroids`` reaches any property a material
declared, so ``"stress"``, ``"strain"`` and ``"volumetric_strain"`` come out of
``LinearElasticStress`` with no extra setup.  Stress and strain use the Voigt
ordering :math:`(xx, yy, zz, yz, xz, xy)`, and the ``zz`` component is filled in
for plane strain and axisymmetric problems as well as three-dimensional ones,
because it is not zero there.

Integrals
---------

.. code-block:: python

   problem.integrate("temperature")
   problem.boundary_flux_integral("conduction", "top")

:meth:`~dualmesh.Problem.integrate` integrates a variable over the whole domain,
with the coordinate factor included, so an axisymmetric problem gives the true
volume integral rather than the integral over the :math:`(r, z)` rectangle.
Setting a variable to one everywhere and integrating it is a quick and honest
measure of how well the mesh represents the domain; that is how the fourth-order
convergence of a curved quadratic boundary is measured in
:doc:`/theory/elements`.

:meth:`~dualmesh.Problem.boundary_flux_integral` integrates the normal flux of
one named kernel over one side set, which is the total heat flow through a face,
or the total force on it.

Reactions
---------

Reactions deserve a section of their own, because they are one of the reasons to
use this method at all.

.. code-block:: python

   problem.reactions("temperature", "left")       # [(node, value), ...]
   problem.total_reaction("temperature", "left")  # their sum

In a finite element code a reaction is recovered after the fact, by multiplying
the assembled stiffness matrix by the solution and reading off the rows that
were constrained.  It is correct, but it is an algebraic by-product of the
linear system rather than a physical quantity the method computed.

In the dual mesh control domain method it is neither.  The control domain of a
boundary node is a real region of space, and part of its surface lies on the
boundary of the domain.  The balance law over that control domain already
contains the integral of the normal flux over that piece of surface: it is the
term the discretisation had to supply in order to close the balance.  So the
reaction at a node is

.. math::

   Q_I = \int_{\partial CD_I \cap \partial \Omega} \mathbf{F} \cdot \mathbf{n} \, dS ,

a flux through a known area, computed by the method rather than extracted from
it.  This is the property Reddy emphasises in [Reddy2024]_: the secondary
variables — heat flows, forces, moments — appear naturally on the control domain
interfaces, which is the physical appeal of the finite volume method, while the
element interpolation removes the ad hoc gradient reconstructions the finite
volume method otherwise needs.

Because the control domains tile the domain exactly, the reactions sum to the
net flux through the boundary, and that identity is checked in the test suite
rather than assumed.  ``total_reaction`` is that sum.

The error indicator
-------------------

.. code-block:: python

   indicators = problem.error_indicator("temperature")   # one per element

One number per element, the square root of the integral over that element of the
squared difference between the computed gradient and a smoother gradient
recovered from it [ZienkiewiczZhu1987]_.  It says which elements carry most of
the error, which is what the marking rules of :doc:`/theory/adaptivity` need.
It is an indicator and not a bound: it does not certify the size of the error.
It is unavailable for the cell-centred method, whose unknowns are cell values,
and asking for it there raises an error that says so.

Writing files
-------------

.. code-block:: python

   problem.write_vtu("result.vtu")
   problem.write_vtu("result.vtu", cell_properties=["stress"])
   problem.write_csv("line.csv", variables=["temperature"])
   problem.write_mesh_file("mesh.msh")

:meth:`~dualmesh.Problem.write_vtu` writes a VTK unstructured grid file
containing the mesh, every variable as point data, and any material property
named in ``cell_properties`` as cell data.  For the cell-centred method the
variables are written as cell data instead, because that is what they are.
Quadratic elements are written as their proper VTK cell types — ``VTK_TRIANGLE``
becomes ``VTK_QUADRATIC_TRIANGLE`` and so on — with the node permutation that
VTK's triquadratic hexahedron needs, so a ``Hex27`` mesh renders with its curved
faces rather than as a straight-sided approximation.

A transient run writes a file per output interval when ``output_file_base`` is
given to :meth:`~dualmesh.Problem.solve_transient`, numbered so that ParaView
groups them into a single time series.  A distributed run writes one ``.vtu``
per rank plus a ``.pvtu`` index, which ParaView opens as one dataset.

:func:`dualmesh.write_mesh` writes the mesh alone, with optional point data,
through meshio [meshio]_, so any of the several dozen formats meshio supports —
Gmsh [Gmsh2009]_, Exodus, XDMF, Abaqus — is available.

Looking at the results
----------------------

Open the ``.vtu`` file in ParaView.  Point data appears under the variable names
you chose, and cell data under the property names.  The useful first steps are a
*Warp By Vector* filter on a displacement field to see the deformed shape, a
*Plot Over Line* filter to compare a profile against a published curve, and the
*Calculator* to form a derived quantity such as a von Mises stress from the
stress components.

For a line plot it is usually quicker to sample in Python and plot there:

.. code-block:: python

   import numpy as np
   from dualmesh import postprocess

   x = np.linspace(0.0, 1.0, 101)
   values = problem.sample("temperature", np.column_stack([x, np.zeros_like(x)]))

which avoids the round trip through a file and gives the same numbers, because
``sample`` and the VTU writer read the same interpolation.
