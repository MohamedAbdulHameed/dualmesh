Meshes
======

Every problem in ``dualmesh`` begins with a mesh.  This chapter explains what a
mesh is in this library, how to generate one, how to read one that another tool
wrote, and how to name, bend and refine it before handing it to a
:class:`~dualmesh.Problem`.

A mesh is independent of the discretisation.  The same mesh can be given to the
dual mesh control domain method, to the Galerkin finite element method and to
either finite volume method, and none of them modifies it; that is what makes
the comparisons of :doc:`/verification` meaningful.  The dual mesh of control
domains is never stored: it is reconstructed element by element from the
reference element definitions whenever a residual is assembled, so there is
nothing for you to build or keep consistent.

.. contents::
   :local:
   :depth: 2

What a mesh contains
--------------------

A :class:`dualmesh.Mesh` holds five things.

**Nodes** are points in space.  Every node carries three coordinates whatever
the dimension of the mesh, with the unused ones set to zero, so a
one-dimensional mesh still reports :math:`y = z = 0`.  Nodes are numbered from
zero, and that numbering is the numbering of the degrees of freedom for every
method except the cell-centred finite volume method.

**Elements** are the cells of the primal mesh.  Each names its type, lists the
global numbers of its nodes in the local order the reference element defines,
and carries a block number.  The element provides the interpolation of the
unknowns and the geometry from which the control domains are cut.

**Blocks**, called subdomains in some codes, are integer labels on elements.
They are how a problem is given more than one material: an object added with
``block="steel"`` acts only on that block's elements, and an object added
without a ``block`` acts everywhere.  ``set_block_name`` attaches a name, and
both the name and the integer written as text are accepted wherever a block is
named.

**Side sets** are lists of boundary *faces*, each stored as a pair
``(element, local side)`` rather than as a list of nodes, so a side set knows
which element it belongs to and therefore which way its outward normal points.
Every boundary condition that prescribes a *flux* — ``NeumannBC``,
``RobinBC``, ``ConvectiveHeatFluxBC``, ``TractionBC``, ``PressureBC`` — needs a
side set, because a flux must be integrated over an area and that integral
needs the face, its Jacobian and its normal.

**Node sets** are plain lists of node numbers and carry no geometry.  A
condition that prescribes a *value*, such as ``DirichletBC``, needs nothing
more and so accepts either kind.  Every place that asks for a boundary name
looks first for a node set of that name and then for a side set, taking the
nodes of its faces; that is what ``mesh.boundary_nodes(name)`` does.  The rule
is therefore short: use a side set unless you have a reason not to, because a
side set can always act as a node set but never the reverse.

.. code-block:: python

   import dualmesh as dm

   mesh = dm.generate_rectangle_mesh(
       x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
       num_x_elements=2, num_y_elements=2)

   print(mesh.dimension, mesh.num_nodes, mesh.num_elements)    # 2 9 4
   print(mesh.sideset_names())    # ['bottom', 'left', 'right', 'top']
   print(mesh.sideset("left"))    # [(0, 3), (2, 3)] — two faces
   print(mesh.boundary_nodes("left"))                          # [0, 3, 6]
   print(mesh.nodeset_names())    # [] — no node set was stored

The node list above is derived from the side set on demand.  The
one-dimensional generator is the one exception: it stores ``left`` and
``right`` as both a side set and a node set, because in one dimension a face is
a single node and the distinction has no content.

Element types
-------------

Fourteen types are supported, and a mesh may mix them.

=============  =====  ====  =============  ============================================
Type           Nodes  Dim.  Order          Node layout
=============  =====  ====  =============  ============================================
``Edge2``          2     1  linear         the two end points
``Edge3``          3     1  quadratic      two ends and the midpoint
``Tri3``           3     2  linear         three corners
``Tri6``           6     2  quadratic      three corners, three edge midpoints
``Quad4``          4     2  bilinear       four corners
``Quad8``          8     2  serendipity    four corners, four edge midpoints
``Quad9``          9     2  biquadratic    four corners, four edge midpoints, centre
``Tet4``           4     3  linear         four corners
``Tet10``         10     3  quadratic      four corners, six edge midpoints
``Hex8``           8     3  trilinear      eight corners
``Hex20``         20     3  serendipity    eight corners, twelve edge midpoints
``Hex27``         27     3  triquadratic   eight corners, twelve edge midpoints, six
                                           face centres, centre
``Wedge6``         6     3  linear         six corners of a triangular prism
``Pyramid5``       5     3  linear         four base corners and the apex
=============  =====  ====  =============  ============================================

Nodes are numbered as VTK numbers them.  The finite element and cell-centred
finite volume methods accept every type.  The dual mesh and vertex-centred
finite volume methods need the element to be divided into node-centred control
domains, which is not defined for the serendipity elements (they have no
interior node to own the centre of the element) or for the pyramid (its apex
is shared by four edges), so they refuse ``Quad8``, ``Hex20`` and ``Pyramid5``
with the reason; :doc:`/theory/elements` gives the details.

The generators
--------------

``generate_line_mesh``
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   mesh = dm.generate_line_mesh(start=0.0, end=0.05, num_elements=5)
   print(mesh.num_nodes, mesh.num_elements)                    # 6 5

   quadratic = dm.generate_line_mesh(start=0.0, end=0.05, num_elements=5,
                                     element_type="Edge3")
   print(quadratic.num_nodes, quadratic.num_elements)          # 11 5

``start`` and ``end`` bound the interval and ``num_elements`` counts elements,
not nodes.  ``bias`` defaults to ``1.0``, which is uniform spacing; any other
positive value applies the geometric grading described under :ref:`grading`.
``coordinates`` replaces all three and takes the node positions directly, in
increasing order.  ``element_type`` is ``"Edge2"`` by default and may be
``"Edge3"``; ``num_elements`` counts elements whichever you choose, so the
quadratic mesh above has the same five elements and eleven rather than six
nodes.  Side sets and node sets ``left`` and ``right`` are always created.

``generate_rectangle_mesh``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   quads = dm.generate_rectangle_mesh(
       x_min=0.0, x_max=2.0, y_min=0.0, y_max=1.0,
       num_x_elements=20, num_y_elements=10)

   triangles = dm.generate_rectangle_mesh(
       x_min=0.0, x_max=2.0, y_min=0.0, y_max=1.0,
       num_x_elements=20, num_y_elements=10,
       element_type="Tri3", diagonal="alternate")

   print(quads.num_elements, triangles.num_elements)           # 200 400

The four bounds and two element counts define a tensor grid.  ``element_type``
is ``"Quad4"`` by default and may be ``"Tri3"``, ``"Quad8"``, ``"Quad9"`` or
``"Tri6"``; a quadratic type is built by generating the linear mesh and
promoting it, so the
element count is unchanged and only the node count grows.  ``diagonal`` applies
to the triangular types and decides how each quadrilateral is cut:
``"right"``, the default, draws every diagonal the same way, and
``"alternate"`` flips it from cell to cell, which removes the directional bias
a single diagonal direction introduces.  Prefer ``"alternate"`` unless you are
reproducing a published result that used a fixed diagonal.  Side sets ``left``,
``right``, ``bottom`` and ``top`` are created.

.. _grading:

Grading: ``x_bias`` and ``x_coordinates``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each axis of the rectangle and box generators takes a bias.  ``x_bias`` is the
ratio between the lengths of successive elements along that axis: the second
element is ``x_bias`` times as long as the first, the third ``x_bias`` times as
long as the second, and the whole set is scaled to span the interval exactly.
A bias of ``1.0``, the default, is uniform.  A bias greater than one makes the
elements grow towards the upper bound, so the mesh is fine near the lower
bound; a bias smaller than one does the opposite.  Use it to resolve a boundary
layer, a re-entrant corner or a stress concentration without paying for a
uniformly fine mesh.

.. code-block:: python

   import numpy as np

   graded = dm.generate_rectangle_mesh(
       x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
       num_x_elements=8, num_y_elements=8, y_bias=0.7)

   y = np.unique(np.asarray(graded.points())[:, 1])
   print(np.round(np.diff(y), 4))
   # [0.3184 0.2228 0.156  0.1092 0.0764 0.0535 0.0375 0.0262]

A bias grades monotonically across the whole axis, which is blunt.  When you
want two boundary layers, a fine band in the interior, or a jump at a material
interface, give the coordinates explicitly.  ``x_coordinates`` and
``y_coordinates`` replace the bounds, the count and the bias for their axis and
are simply the node positions in increasing order.  Two helpers build such
lists.  :func:`dualmesh.graded_coordinates` (``start``, ``end``,
``num_elements``, ``bias``) returns exactly the list the bias arguments would
have produced, so it is the way to grade one axis while leaving another
explicit.  ``coordinates_from_spacings`` accumulates element sizes from a
starting point, which is the natural form when a drawing gives you the
thicknesses of a stack of layers; the interval is whatever the spacings add up
to, since nothing is normalised.

.. code-block:: python

   from dualmesh.meshing import coordinates_from_spacings

   mesh = dm.generate_rectangle_mesh(
       x_coordinates=[0.0, 0.5, 1.5, 3.0],
       y_coordinates=dm.graded_coordinates(0.0, 1.0, 10, bias=0.8))
   print(mesh.num_elements)                                    # 30

   print(coordinates_from_spacings(0.0, [0.1, 0.1, 0.2, 0.4, 0.8]))
   # [0.0, 0.1, 0.2, 0.4, 0.8, 1.6]

``generate_box_mesh``
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   box = dm.generate_box_mesh(
       x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, z_min=0.0, z_max=1.0,
       num_x_elements=4, num_y_elements=4, num_z_elements=4,
       element_type="Tet4")
   print(box.num_elements)                                     # 384
   print(box.sideset_names())
   # ['back', 'bottom', 'front', 'left', 'right', 'top']

This is the rectangle generator with a third axis.  ``element_type`` is
``"Hex8"`` by default and may be ``"Tet4"``, ``"Wedge6"``, ``"Pyramid5"``,
``"Hex20"``, ``"Hex27"`` or ``"Tet10"``.  Each hexahedral cell of the grid is
split into six tetrahedra for the tetrahedral types, which is why the 64-cell
grid above yields 384 elements, into two prisms along a face diagonal for
``"Wedge6"``, and into six pyramids, one on each face with its apex at a new
node at the cell centre, for ``"Pyramid5"``.  There are three biases,
``x_bias``, ``y_bias`` and ``z_bias``, and three coordinate lists, all behaving
exactly as in two dimensions.  The six side sets are ``left`` and ``right``
(the :math:`x` faces), ``bottom`` and ``top`` (the :math:`y` faces), and
``back`` and ``front`` (the :math:`z` faces).

``generate_annulus_mesh``
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   annulus = dm.generate_annulus_mesh(
       inner_radius=0.05, outer_radius=0.10,
       num_radial_elements=8, num_angular_elements=12,
       start_angle=0.0, end_angle=90.0, radial_bias=0.85)

   print(annulus.sideset_names())
   # ['bottom', 'end', 'inner', 'left', 'outer', 'right', 'start', 'top']

This generates an annular sector: the mesh for a thick pressurised cylinder and
for the quarter plate with a central hole.  ``inner_radius`` and
``outer_radius`` bound it radially and ``num_radial_elements`` divides that
span.  ``start_angle`` and ``end_angle`` are in **degrees** measured from the
:math:`x` axis and default to a quarter turn; ``num_angular_elements`` divides
the arc uniformly, and there is no angular bias.  ``radial_bias`` grades the
radial direction exactly as ``x_bias`` does, and ``radial_coordinates``
replaces the bias and the count with an explicit list of radii.
``element_type`` accepts the same four two-dimensional types as the rectangle
generator.

The four geometric side sets are ``inner``, ``outer``, ``start`` (the straight
edge at ``start_angle``) and ``end``.  Because the generator builds a rectangle
in the :math:`(r, \theta)` plane and bends it, the rectangle's own names
``left``, ``right``, ``bottom`` and ``top`` survive as aliases for the same
faces; they are harmless, but use the geometric names.  The generator promotes
to second order *before* bending, which is why a ``Quad9`` annulus is genuinely
isoparametric: the mid-side nodes land on the true circle.  The same care is
needed when you do this yourself; see :ref:`transforming`.

Building a mesh from arrays
---------------------------

When the geometry comes from elsewhere — an analytical construction, a NumPy
computation, another library — hand the arrays to
:func:`dualmesh.mesh_from_arrays`.

.. code-block:: python

   mesh = dm.mesh_from_arrays(
       [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        [2.0, 0.0], [2.0, 1.0]],
       [[0, 1, 2, 3], [1, 4, 5, 2]],
       element_type="Quad4", blocks=[1, 2])
   print(mesh.num_nodes, mesh.num_elements, mesh.block_ids())  # 6 2 [1, 2]

``points`` has one row per node and one to three columns, missing columns being
filled with zeros.  ``cells`` is either a plain connectivity array of shape
``(num_elements, nodes_per_element)``, in which case ``element_type`` is
required, or a list of ``(element_type, connectivity)`` pairs, which is how a
mesh of mixed element types is built:

.. code-block:: python

   mixed = dm.mesh_from_arrays(
       [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
       [("Tri3", [[0, 1, 2], [0, 2, 3]])])
   print(mixed.num_elements)                                   # 2

``blocks`` gives one block number per element in the order the elements are
added; omitted, every element goes into block 0.  ``dimension`` overrides the
spatial dimension, which is otherwise the largest among the element types
given — set it when a mesh of ``Edge2`` elements is meant to be a plane truss
rather than a one-dimensional bar.  :meth:`~dualmesh.Mesh.fix_orientation` is
called for you, so the winding order of your connectivity does not matter.

A mesh built this way has **no** side sets and no node sets, so nothing can be
prescribed on it yet.  The quickest repair, when the domain is a box, is
:meth:`~dualmesh.Mesh.add_bounding_box_sidesets`, which computes the bounding
box and creates the standard ``left``/``right``/``bottom``/``top`` (and, in
three dimensions, ``back``/``front``) side sets from the exterior faces whose
centroids lie on each face of that box, within a ``tolerance`` defaulting to
:math:`10^{-10}`.  It helps only when the domain is aligned with the axes; on a
curved or inclined boundary, use the predicates below.

.. code-block:: python

   mesh.add_bounding_box_sidesets()
   print(mesh.sideset_names())   # ['bottom', 'left', 'right', 'top']

Reading and writing files
-------------------------

:func:`dualmesh.read_mesh` and :func:`dualmesh.write_mesh` go through
`meshio <https://github.com/nschloe/meshio>`_ [meshio]_, so every format meshio
handles is available: Gmsh ``.msh`` [Gmsh2009]_, Exodus II ``.e`` and ``.exo``,
VTK and VTU, Abaqus ``.inp``, MED and the rest.  meshio picks the format from
the extension; ``file_format`` overrides that when the extension is unhelpful.
Some formats need an optional meshio dependency that is not installed by
default — Exodus needs ``netCDF4``, MED needs ``h5py`` — and a missing one
raises a plain :class:`ModuleNotFoundError` naming the package.

.. code-block:: python

   dm.write_mesh(mesh, "two_blocks.vtu")
   dm.write_mesh(mesh, "two_blocks.msh", file_format="gmsh")

   temperature = np.linspace(0.0, 1.0, mesh.num_nodes)
   dm.write_mesh(mesh, "with_field.vtu", temperature=temperature)

Any further keyword argument to :func:`~dualmesh.write_mesh` is written as a
nodal field of that name.  In practice you will more often use
:meth:`Problem.write_vtu <dualmesh.Problem.write_vtu>`, which does this for
every variable of a solved problem; see :doc:`output`.

Only the ten supported cell types are read.  Cells of the mesh dimension become
elements, and cells one dimension lower become **side sets**, which is how a
Gmsh model transfers its boundary names.  Reading applies these rules:

* The physical or material tag of a volume cell, taken from the
  ``gmsh:physical``, ``medit:ref``, ``cell_tags`` or ``material`` cell data,
  becomes the element's block number.  Untagged cells go into block 0.
* Each group of lower-dimensional cells sharing a tag becomes one side set and
  one node set, named after the physical name the file gives that tag, or
  ``"boundary_<tag>"`` when the file names it not at all.
* ``boundary_names`` renames them on the way in, as a mapping from the name the
  file used to the name you want.
* If the file carried no boundary cells, or if you pass
  ``add_bounding_box_sidesets=True``, the bounding-box side sets are added, so
  a bare volume mesh is still usable.

For a Gmsh model whose physical groups are called ``fluid``, ``inlet`` and
``outlet``, that gives:

.. code-block:: python

   channel = dm.read_mesh("channel.msh")
   print(channel.sideset_names(), channel.block_ids())
   # ['inlet', 'outlet'] [1]

   renamed = dm.read_mesh("channel.msh", boundary_names={"inlet": "hot_wall"})
   print(renamed.sideset_names())               # ['hot_wall', 'outlet']

.. warning::

   Writing does not round-trip block numbers.
   :func:`~dualmesh.write_mesh` stores them as cell data named ``block``, but
   :func:`~dualmesh.read_mesh` looks only at the four tag names listed above,
   so a multi-block mesh written by ``dualmesh`` and read back has every
   element in block 0.  Side sets are not written at all.  Treat file output as
   a way of getting results to a visualiser, and keep the generating script, or
   the original Gmsh file, as the definition of a mesh you intend to reuse.

Naming boundaries after the fact
--------------------------------

When a mesh arrives without the boundary names you need, add them
geometrically.  Both predicates take a Python function of ``(x, y, z)`` and are
evaluated once, when you call them, not during the solve.

.. code-block:: python

   mesh = dm.generate_rectangle_mesh(
       x_min=0.0, x_max=2.0, y_min=0.0, y_max=1.0,
       num_x_elements=20, num_y_elements=10)

   mesh.add_sideset_by_predicate(
       "outlet_half", lambda x, y, z: x > 2.0 - 1e-9 and y > 0.5)
   mesh.add_nodeset_by_predicate(
       "centre_line", lambda x, y, z: abs(y - 0.5) < 1e-9)

   print(len(mesh.sideset("outlet_half")))                     # 5
   print(len(mesh.boundary_nodes("centre_line")))              # 21

:meth:`~dualmesh.Mesh.add_sideset_by_predicate` tests the **centroid of each
exterior face** and collects the faces that pass.  Only exterior faces are
considered, so a predicate true in the interior selects nothing — which is what
you want, since a flux condition on an interior face is meaningless.
:meth:`~dualmesh.Mesh.add_nodeset_by_predicate` tests **every node**, interior
nodes included, which is how ``centre_line`` picks up a line through the middle
of the domain.  That is the other reason to reach for a node set: prescribing a
symmetry condition on an interior plane.

Write the tolerance into the predicate yourself, loosely enough for
floating-point arithmetic.  ``x > 2.0 - 1e-9`` is safe; ``x == 2.0`` is not,
because the coordinate was computed by ``linspace``.
:meth:`~dualmesh.Mesh.alias_sideset` gives an existing side set a second name,
which is useful when a file calls a boundary something your input does not.

.. _transforming:

Bending a generated mesh
------------------------

:meth:`~dualmesh.Mesh.transform_nodes` applies a map to every node in place.
The function receives ``(x, y, z)`` and returns up to three coordinates.
Elements, blocks, side sets and node sets are untouched, so a boundary named
before the transformation keeps its name after it.  This is how a curved domain
is built from a structured grid: generate a rectangle in the parameter plane,
name its edges, and bend it.

.. code-block:: python

   import math

   rectangle = dm.generate_rectangle_mesh(
       x_min=1.0, x_max=2.0, y_min=0.0, y_max=0.5 * math.pi,
       num_x_elements=4, num_y_elements=6)

   sector = rectangle.second_order()          # promote FIRST
   sector.transform_nodes(
       lambda r, theta, z: [r * math.cos(theta), r * math.sin(theta), 0.0])
   sector.fix_orientation()
   sector.alias_sideset("left", "inner")
   sector.alias_sideset("right", "outer")

   radii = np.hypot(*np.asarray(sector.points())[:, :2].T)
   print(round(radii.min(), 12), round(radii.max(), 12))       # 1.0 2.0

.. warning::

   **Promote to second order before transforming, never after.**
   :meth:`~dualmesh.Mesh.second_order` places each added node at the midpoint
   of the edge, face or cell it belongs to.  If the mesh has already been bent,
   that midpoint is the midpoint of the straight *chord* between two corners
   which now lie on a circle, and the mid-side node sits inside the true
   boundary instead of on it.  The element is then subparametric, and the
   geometric error introduced is of the same order as the discretisation error
   you promoted the mesh to remove.

   Moving the ``second_order()`` call in the example above to *after* the
   ``transform_nodes()`` call changes the smallest radius in the mesh from
   1.0 to 0.991445: the inner boundary should sit at radius 1, and half its
   nodes have ended up nearly one per cent inside the domain.  No refinement in
   the angular direction removes that error faster than first order.

Refining and promoting
----------------------

:meth:`~dualmesh.Mesh.refined` returns a new mesh in which every linear
element has been split into :math:`2^{d}` children, :math:`d` being the
dimension; a prism becomes eight prisms, and a pyramid becomes six pyramids and
four tetrahedra, because a pyramid cannot be split into pyramids alone.  Side
sets and node sets carry over, so boundary conditions written against the
coarse mesh work unchanged on the fine one, and the original is not modified,
which makes a convergence study a one-line loop.

.. code-block:: python

   coarse = dm.generate_rectangle_mesh(
       x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
       num_x_elements=2, num_y_elements=2)

   fine = coarse.refined()
   print(fine.num_elements, fine.num_nodes)                    # 16 25
   print(coarse.refined().refined().num_elements)              # 64

   quadratic = fine.second_order()
   print(quadratic.element_type(0), quadratic.num_elements,
         quadratic.num_nodes)                          # Quad9 16 81

Only linear meshes can be refined; refining a quadratic mesh raises an error
telling you to refine the linear mesh first and promote the result.  For
*adaptive* refinement, where only the elements an error indicator marks are
split, see :func:`dualmesh.refine_marked` and the marking strategies described
in :doc:`/theory/adaptivity`.

:meth:`~dualmesh.Mesh.second_order` returns a copy with quadratic elements:
``Edge2`` becomes ``Edge3``, ``Tri3`` becomes ``Tri6``, ``Quad4`` becomes
``Quad9``, ``Tet4`` becomes ``Tet10`` and ``Hex8`` becomes ``Hex27``; with
``serendipity=True``, ``Quad4`` becomes ``Quad8`` and ``Hex8`` becomes
``Hex20`` instead.  There is no quadratic prism or pyramid, and a mesh
containing either cannot be promoted.  The
corner nodes keep their numbers and positions, so the domain does not change
shape and any node index recorded earlier is still valid.  Side sets carry over
unchanged, since they refer to element sides, which still exist, and node sets
gain the added nodes lying between two members.

Orientation
-----------

:meth:`~dualmesh.Mesh.fix_orientation` renumbers the nodes of any element whose
Jacobian determinant is negative, so that every element is wound consistently
and every volume is positive.  A negative Jacobian is not merely untidy: the
control domain volumes would come out negative, the assembled matrix would be
wrong, and the failure would be silent.  You rarely call it yourself, because
the generators, :func:`~dualmesh.mesh_from_arrays` and
:func:`~dualmesh.read_mesh` all call it already, which is why a deliberately
reversed quadrilateral comes back in the conventional order:

.. code-block:: python

   flipped = dm.mesh_from_arrays(
       [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
       [[0, 3, 2, 1]], element_type="Quad4")
   print(flipped.element_nodes(0))                             # [0, 1, 2, 3]

Call it after :meth:`~dualmesh.Mesh.transform_nodes`, and after any
``add_element`` calls on a mesh you are assembling by hand.  It cannot repair
an element that is genuinely tangled — one whose Jacobian changes sign inside
it — and no reordering of nodes can; that is a meshing error to fix upstream.

With a mesh in hand, :doc:`problem_setup` explains how to attach variables and
physics to it, :doc:`solving` covers the solvers, and :doc:`output` covers
getting the answers back out.
