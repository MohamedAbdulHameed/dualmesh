Meshes
======

Every problem starts from a mesh of finite elements. This page shows how to make a mesh in the
input, generate one in Python, build one from arrays, read one from a file, and name, transform
and refine it before you give it to a :class:`~dualmesh.Problem`.

The finite element method (FEM) and the dual mesh control domain method (DMCDM) take the same
mesh. DMCDM builds the control domain :math:`V_i` of each node :math:`i` from the elements around
the node, so you provide the elements alone.

.. contents::
   :local:
   :depth: 2

The mesh section of the input
-----------------------------

In a YAML or TOML case file, or in :func:`dualmesh.problem`, the ``mesh`` section takes exactly
one of four forms. A generated mesh starts at the origin and takes its sizes with their units:

.. code-block:: yaml

   mesh:
     rectangle: {size: [0.1 m, 0.02 m], cells: [40, 8]}

This section comes from ``examples/heated_plate.yaml``. The four forms are:

``segment``
   A line of ``length`` cut into ``cells`` elements. Its boundaries are ``left`` and ``right``.

``rectangle``
   A ``size`` of two lengths cut into ``cells`` of two counts. ``cell_type`` is
   ``quadrilateral`` (the default) or ``triangle``. Its boundaries are ``left``, ``right``,
   ``bottom`` and ``top``.

``box``
   A ``size`` of three lengths cut into ``cells`` of three counts. ``cell_type`` is
   ``hexahedron`` (the default) or ``tetrahedron``. Its boundaries are those of the rectangle and
   ``back`` and ``front``, the faces normal to :math:`z`.

``file``
   A mesh file. A relative path starts from the folder of the case file, or from the working
   directory in :func:`dualmesh.problem`. A mesh file doesn't state
   the unit of its coordinates, so you give it as ``length_unit``, for example ``length_unit:
   mm``. The boundaries are the side sets of the file, and the regions are its named blocks.

A generated mesh has one region, ``domain``. The key ``coordinate_system`` is ``cartesian`` (the
default) or ``axisymmetric``, in which the first coordinate is the radius. A segment or a box
can't be axisymmetric. :doc:`input` lists every key of the section.

The rest of this page describes the meshes of the Python API, which the input builds on.

What a mesh contains
--------------------

A :class:`dualmesh.Mesh` holds five kinds of data.

Nodes
   Points in space, numbered from zero. Every node has three coordinates, and the unused ones are
   zero. The node numbering is the numbering of the unknowns.

Elements
   The cells of the mesh. Each element has a type, the global numbers of its nodes in the local
   order of its type, and a block number.

Blocks
   Integer labels on the elements, one for each material or region. An object that you add with
   ``block=...`` acts on the elements of those blocks, and an object without ``block`` acts on
   every element. :meth:`~dualmesh.Mesh.set_block_name` names a block, and you can then refer to
   the block by its name or by its number written as text.

Side sets
   Named lists of boundary faces, each stored as a pair ``(element, local side)``. The element
   gives the outward normal of the face. A condition that prescribes a flux, such as
   ``Neumann``, ``Robin``, ``radiative_heat_flux``, ``traction`` or ``pressure``, needs a side
   set, because its integral needs the faces and their normals.

Node sets
   Named lists of node numbers, without faces. A condition that prescribes a value, such as
   ``Dirichlet``, takes a node set or a side set. A boundary name means the node set of that name
   when there is one, and otherwise the nodes of the side set of that name.

A side set can stand in for a node set, and a node set can't stand in for a side set, so name a
boundary with a side set unless it lies inside the domain.

.. code-block:: python

   import dualmesh

   mesh = dualmesh.generate_rectangle_mesh(
       x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=2, num_y_elements=2
   )

   print(mesh.dimension, mesh.num_nodes, mesh.num_elements)  # 2 9 4
   print(mesh.sideset_names())  # ['bottom', 'left', 'right', 'top']
   print(mesh.boundary_nodes("left"))  # [0, 3, 6]
   print(mesh.nodeset_names())  # []

:meth:`~dualmesh.Mesh.boundary_nodes` takes the nodes of the faces of ``left``, because the
rectangle has no node sets. :meth:`~dualmesh.Mesh.summary` prints the counts, the element types,
the blocks, the side sets and the node sets of a mesh.

Element types
-------------

A mesh may mix the fourteen element types below. Each type numbers its nodes as the
Visualization Toolkit (VTK) numbers them.

=============  =====  =========  ============  ============================================
Type           Nodes  Dimension  Order         Node positions
=============  =====  =========  ============  ============================================
``Edge2``          2          1  linear        two ends
``Edge3``          3          1  quadratic     two ends and the midpoint
``Tri3``           3          2  linear        three corners
``Tri6``           6          2  quadratic     three corners and three edge midpoints
``Quad4``          4          2  bilinear      four corners
``Quad8``          8          2  serendipity   four corners and four edge midpoints
``Quad9``          9          2  biquadratic   four corners, four edge midpoints and the centre
``Tet4``           4          3  linear        four corners
``Tet10``         10          3  quadratic     four corners and six edge midpoints
``Hex8``           8          3  trilinear     eight corners
``Hex20``         20          3  serendipity   eight corners and twelve edge midpoints
``Hex27``         27          3  triquadratic  eight corners, twelve edge midpoints, six face
                                               centres and the centre
``Wedge6``         6          3  linear        six corners of a triangular prism
``Pyramid5``       5          3  linear        four base corners and the apex
=============  =====  =========  ============  ============================================

FEM takes every type. DMCDM refuses ``Quad8``, ``Hex20`` and ``Pyramid5``, and its error message
gives the reason. A serendipity element has no interior node to own the centre of the element.
At the apex of a pyramid four edges meet, and the control domain there needs a decomposition
that DMCDM doesn't implement.
:doc:`/theory/elements` describes each element.

The generators
--------------

Each generator returns a new mesh with named side sets. A quadratic element type is built as the
linear mesh and then promoted, so the element counts are the same for both orders and the
quadratic mesh has more nodes.

``generate_line_mesh``
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   mesh = dualmesh.generate_line_mesh(start=0.0, end=0.05, num_elements=5)
   print(mesh.num_nodes, mesh.num_elements)  # 6 5

   quadratic = dualmesh.generate_line_mesh(
       start=0.0, end=0.05, num_elements=5, element_type="Edge3"
   )
   print(quadratic.num_nodes, quadratic.num_elements)  # 11 5

``start`` and ``end`` bound the interval and ``num_elements`` counts the elements. ``bias``
grades the spacing as :ref:`grading` describes, with ``1.0``, the default, for uniform elements.
``coordinates`` gives the node positions in increasing order instead of the three.
``element_type`` is ``"Edge2"`` (the default) or ``"Edge3"``. The ends ``left`` and ``right`` are
both side sets and node sets.

``generate_rectangle_mesh``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   quads = dualmesh.generate_rectangle_mesh(
       x_min=0.0, x_max=2.0, y_min=0.0, y_max=1.0, num_x_elements=20, num_y_elements=10
   )

   triangles = dualmesh.generate_rectangle_mesh(
       x=[0.0, 2.0], y=[0.0, 1.0], num_elements=[20, 10], element_type="Tri3"
   )

   print(quads.num_elements, triangles.num_elements)  # 200 400

You give the bounds as ``x_min``, ``x_max``, ``y_min`` and ``y_max``, or as the pairs ``x`` and
``y``. You give the element counts as ``num_x_elements`` and ``num_y_elements``, or as the list
``num_elements``. A call that gives both forms of the same input is refused. ``element_type`` is
``"Quad4"`` (the default), ``"Tri3"``, ``"Quad8"``, ``"Quad9"`` or ``"Tri6"``.

``diagonal`` sets how a triangular mesh cuts each rectangle into two triangles. With
``"alternate"``, the default, the direction of the diagonal changes from each cell to its
neighbours, which avoids the bias of one diagonal direction. ``"right"`` and ``"left"`` cut every
cell the same way, as a published case with a fixed diagonal needs.

The side sets are ``left`` and ``right`` (the faces normal to :math:`x`) and ``bottom`` and
``top`` (the faces normal to :math:`y`).

.. _grading:

Graded spacing
~~~~~~~~~~~~~~

Each axis of the rectangle and box generators takes a bias, ``x_bias`` for the :math:`x` axis,
and the line generator takes ``bias``. The bias is the ratio between the lengths of two successive elements along that
axis, and the lengths are scaled so that the elements span the interval. A bias greater than one
makes the elements grow towards the upper bound, and a bias smaller than one makes them shrink
towards it. Use a bias to resolve a boundary layer or a stress concentration without a uniformly
fine mesh.

.. code-block:: python

   import numpy

   graded = dualmesh.generate_rectangle_mesh(
       x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=8, num_y_elements=8, y_bias=0.7
   )

   y = numpy.unique(numpy.asarray(graded.points())[:, 1])
   print(numpy.round(numpy.diff(y), 4))
   # [0.3184 0.2228 0.156  0.1092 0.0764 0.0535 0.0375 0.0262]

A bias grades a whole axis in one direction. For two boundary layers, a fine band inside the
domain or a change of size at a material interface, give the node positions of the axis as
``x_coordinates``, in increasing order, in place of its bounds, count and bias. Two helpers build
such lists:

* :func:`dualmesh.graded_coordinates` (``start``, ``end``, ``num_elements``, ``bias``) returns the
  positions that the bias of a generator gives.
* ``dualmesh.meshing.coordinates_from_spacings`` (``start``, ``spacings``) adds up the element
  sizes from a starting point, for a stack of layers of given thicknesses. The interval ends at
  ``start`` plus the sum of the sizes.

.. code-block:: python

   from dualmesh.meshing import coordinates_from_spacings

   mesh = dualmesh.generate_rectangle_mesh(
       x_coordinates=[0.0, 0.5, 1.5, 3.0],
       y_coordinates=dualmesh.graded_coordinates(0.0, 1.0, 10, bias=0.8),
   )
   print(mesh.num_elements)  # 30

   layers = coordinates_from_spacings(0.0, [0.1, 0.1, 0.2, 0.4, 0.8])  # Nodes from 0 to 1.6

``generate_box_mesh``
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   box = dualmesh.generate_box_mesh(
       x=[0.0, 1.0], y=[0.0, 1.0], z=[0.0, 1.0], num_elements=[4, 4, 4], element_type="Tet4"
   )
   print(box.num_elements)  # 384
   print(box.sideset_names())  # ['back', 'bottom', 'front', 'left', 'right', 'top']

The box generator adds the :math:`z` axis to the arguments of the rectangle generator, with
``z_bias`` and ``z_coordinates``. ``element_type`` is ``"Hex8"`` (the default), ``"Tet4"``,
``"Wedge6"``, ``"Pyramid5"``, ``"Hex20"``, ``"Hex27"`` or ``"Tet10"``. The generator cuts each
hexahedral cell of the grid as follows:

* ``Tet4`` and ``Tet10``: six tetrahedra around the main diagonal of the cell, so the
  :math:`4 \times 4 \times 4` grid above has 384 elements.
* ``Wedge6``: two prisms, split along the same diagonal of the base in every cell.
* ``Pyramid5``: six pyramids, one on each face, with their apex at a node added at the centre of
  the cell.

The side sets are ``left`` and ``right`` (normal to :math:`x`), ``bottom`` and ``top`` (normal to
:math:`y`) and ``back`` and ``front`` (normal to :math:`z`).

``generate_annulus_mesh``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   annulus = dualmesh.generate_annulus_mesh(
       inner_radius=0.05,
       outer_radius=0.10,
       num_radial_elements=8,
       num_angular_elements=12,
       start_angle=0.0,
       end_angle=90.0,
       radial_bias=0.85,
   )
   print(annulus.sideset_names())
   # ['bottom', 'end', 'inner', 'left', 'outer', 'right', 'start', 'top']

The annulus generator meshes a sector of an annulus, such as a thick cylinder under pressure or a
quarter plate with a central hole. ``inner_radius`` and ``outer_radius`` bound the sector, and
``num_radial_elements`` cuts the radial span. ``start_angle`` and ``end_angle`` are in degrees
from the :math:`x` axis, with defaults of 0 and 90. ``num_angular_elements`` cuts the arc into
equal angles. ``radial_bias`` grades the radial direction, and ``radial_coordinates`` gives the
radii of the nodes instead. ``element_type`` takes the five types of the rectangle generator.

The side sets ``inner``, ``outer``, ``start`` (the straight edge at ``start_angle``) and ``end``
name the edges of the sector. The generator maps a rectangle in the plane of the radius and the
angle onto the sector, and the rectangle's names ``left``, ``right``, ``bottom`` and ``top`` stay
as second names of the same edges. The generator promotes a quadratic mesh before the mapping, so
the mid-side nodes of a ``Quad9`` annulus lie on the circles. :ref:`transforming` explains why
the order matters.

.. _surface-meshes:

Surface meshes
--------------

A mesh of two-dimensional elements whose nodes lie in three-dimensional space is a surface. Its
:attr:`~dualmesh.Mesh.dimension` is 2, the dimension of the elements, and its
:attr:`~dualmesh.Mesh.space_dimension` is 3, the dimension of the space. ``solid_mechanics`` with
``formulation="shell"`` solves a shell on a surface mesh of ``Quad4`` elements. A mesh of
``Edge2`` elements whose nodes span a plane or space is a curve, and ``formulation="truss"`` takes
a mesh of ``Edge2`` elements in one, two or three dimensions.

A surface element maps the reference coordinates :math:`\boldsymbol{\xi} = (\xi_1, \xi_2)` to
space through its :math:`3 \times 2` Jacobian matrix :math:`\mathbf{J}`, whose columns are the two
tangents :math:`\partial \mathbf{x} / \partial \xi_1` and
:math:`\partial \mathbf{x} / \partial \xi_2`. Its area element is
:math:`\sqrt{\det(\mathbf{J}^T \mathbf{J})}`, and the gradient of a shape function :math:`N_i` is
the tangential vector

.. math::

   \nabla N_i = \mathbf{J} \left(\mathbf{J}^T \mathbf{J}\right)^{-1}
   \frac{\partial N_i}{\partial \boldsymbol{\xi}} .

An equation on a surface is therefore solved in the surface, as for conduction in a thin curved
sheet.

Orientation of a surface
~~~~~~~~~~~~~~~~~~~~~~~~

The elements of a surface are oriented so that the normals of neighbours agree.
:meth:`~dualmesh.Mesh.fix_orientation` orients each connected part of a surface. On a closed
part the normals point outward. On an open part the largest component of the sum of the normals
is positive, which is :math:`+z` for a plate in the :math:`xy` plane. To choose the side yourself,
pass ``preferred``, an array of shape ``(num_elements, 3)`` with one normal for each element, and
each part takes the side that agrees with most of these normals. An edge shared by more than two
elements, as at a stiffener, doesn't tie the orientations of those elements. A surface with one
side, such as a Moebius strip, is refused. Uniform refinement keeps the orientation of the coarse
surface.

The surface generators below orient their normals as each one states. Their angles are in
degrees, while the trigonometric functions inside an equation take radians. An edge that closes
up, at a pole or along a seam of 360 degrees, has no name.

``generate_cylinder_surface_mesh``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # The Scordelis-Lo roof: 80 degrees of a cylinder of radius 25, 50 long.
   roof = dualmesh.generate_cylinder_surface_mesh(
       radius=25.0, length=50.0, arc_angle=80.0, num_elements=[16, 16]
   )
   print(roof.dimension, roof.space_dimension, roof.num_elements)  # 2 3 256
   print(roof.sideset_names())
   # ['far_curved_edge', 'left_straight_edge', 'near_curved_edge', 'right_straight_edge']

The axis of the cylinder runs along :math:`y`, from :math:`y = 0` to the ``length``. The arc of
``arc_angle`` degrees, 360 by default, is centred on the :math:`+z` direction, the crown.
``num_elements`` is ``[around the arc, along the axis]``. ``near_curved_edge`` lies at
:math:`y = 0` and ``far_curved_edge`` at the far end. An arc shorter than 360 degrees also has
``left_straight_edge``, on the :math:`-x` side, and ``right_straight_edge``, on the :math:`+x`
side. The normals point away from the axis.

``generate_spherical_shell_mesh``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # A quadrant of a hemisphere with an 18-degree hole at the top.
   shell = dualmesh.generate_spherical_shell_mesh(
       radius=10.0, polar_angle=[18, 90], azimuth=[0, 90], num_elements=[8, 8]
   )
   print(shell.sideset_names())
   # ['end_meridian_edge', 'lower_circular_edge', 'start_meridian_edge', 'upper_circular_edge']

   # A closed hemisphere, whose cap at the pole needs no collapsed element.
   cap = dualmesh.generate_spherical_shell_mesh(radius=10.0, polar_angle=[0, 90], num_elements=[8, 16])

``polar_angle=[from, to]`` is measured from the :math:`+z` axis, the pole, so 90 is the equator.
``azimuth=[from, to]`` is measured from :math:`+x` towards :math:`+y`. The default is the whole
sphere. ``num_elements`` is ``[along a meridian, around the circle]``, and ``center`` moves the
sphere. ``upper_circular_edge`` and ``lower_circular_edge`` are the edges of constant polar angle,
with the upper one nearer the pole. ``start_meridian_edge`` and ``end_meridian_edge`` are the
edges of constant azimuth.

Where the zone closes at a pole, each sector of at most 90 degrees of the cap is a rhombus at the
pole and two ring patches, so no element collapses there. The number of elements around the
circle must then be a multiple of twice the number of sectors. The normals point away from the
centre.

``generate_surface_mesh``
~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~dualmesh.generate_surface_mesh` meshes a surface that you write as equations. You give
the definitions that the equations use as keywords, each a number or a text in terms of earlier
definitions, and a definition that the equations don't use is refused. The other keywords choose
one of four constructions:

.. code-block:: python

   # A height surface z = f(x, y) over a rectangle.
   saddle = dualmesh.generate_surface_mesh(
       surface="z = c*(x^2 - y^2)", c=0.1,
       x_min=-5, x_max=5, y_min=-5, y_max=5, num_elements=[10, 10],
   )
   print(saddle.sideset_names())  # ['x_max_edge', 'x_min_edge', 'y_max_edge', 'y_min_edge']

   # A parametric surface x(u, v), y(u, v), z(u, v): a torus, whose seams close.
   torus = dualmesh.generate_surface_mesh(
       surface=["x = (R + a*cos(v))*cos(u)", "y = (R + a*cos(v))*sin(u)", "z = a*sin(v)"],
       R=2.0, a=0.5, u_min=0, u_max="2*pi", v_min=0, v_max="2*pi", num_elements=[32, 12],
   )
   print(torus.num_elements, torus.sideset_names())  # 384 []

   # A surface f(x, y, z) = c seen from a centre.
   ellipsoid = dualmesh.generate_surface_mesh(
       surface="x^2/a^2 + y^2/b^2 + z^2/c^2 = 1", a=2.0, b=1.5, c=1.0,
       center=[0, 0, 0], num_elements=[8, 16],
   )

   # Any surface f(x, y, z) = c inside a box: here half a torus.
   half_torus = dualmesh.generate_surface_mesh(
       surface="(sqrt(x^2 + y^2) - R)^2 + z^2 = a^2", R=2.0, a=0.5,
       box=[[-3, 0, -1], [3, 3, 1]], element_size=0.2,
   )

Height surface
   One coordinate as a function of the other two, with the bounds of those two and
   ``num_elements``. The edges are ``x_min_edge``, ``x_max_edge`` and so on, after the bounds.
   The normals point towards increasing height.

Parametric surface
   Three equations in the parameters ``u`` and ``v``, with ``u_min``, ``u_max``, ``v_min`` and
   ``v_max`` (numbers or text) and ``num_elements``. Seams that close are merged. The edges are
   ``u_min_edge``, ``u_max_edge``, ``v_min_edge`` and ``v_max_edge``. The normals point along
   :math:`\mathbf{r}_u \times \mathbf{r}_v`, with :math:`\mathbf{r}_u` and :math:`\mathbf{r}_v`
   the derivatives of the position with respect to the parameters.

Surface seen from a centre
   ``center`` and ``num_elements`` give the mesh of a sphere, moved along the rays from the
   centre onto the surface, so the quadrilaterals keep the shapes of those of the sphere. Every
   ray from the centre must meet the surface once, as on an ellipsoid or a dented sphere, and the
   generator refuses a surface that a ray meets more than once, such as a torus. ``polar_angle``
   and ``azimuth`` take a part of the surface, with the edge names of the sphere. The normals
   point away from the centre.

Surface in a box
   ``box=[[x_min, y_min, z_min], [x_max, y_max, z_max]]`` and ``element_size`` mesh the part of
   any surface inside the box with quadrilaterals of about that size, by surface nets on a regular
   grid, smoothed along the surface. The edge where the surface meets the box is ``box_edge``.
   ``element_type="Tri3"`` splits each quadrilateral into two triangles. The normals point towards
   :math:`f > c`.

The gradient of :math:`f` must not vanish on :math:`f = c`, so a surface with an apex or a
self-crossing is refused. Every node of a generated surface lies on the surface, and a quadratic
element type places its mid-side nodes on it too.

Meshes from arrays
------------------

When the geometry comes from another program or from your own computation, pass the arrays to
:func:`dualmesh.mesh_from_arrays`:

.. code-block:: python

   mesh = dualmesh.mesh_from_arrays(
       [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]],
       [[0, 1, 2, 3], [1, 4, 5, 2]],
       element_type="Quad4",
       blocks=[1, 2],
   )
   print(mesh.num_nodes, mesh.num_elements, mesh.block_ids())  # 6 2 [1, 2]

``points`` has one row for each node and one to three columns, and the missing columns are zero.
``cells`` takes one of three forms:

* A connectivity array of shape ``(num_elements, nodes_per_element)``, with ``element_type``.
* A list of ``(element_type, connectivity)`` pairs, for a mesh of mixed types.
* The ``(element_type, nodes, block)`` triples that :meth:`~dualmesh.Mesh.cells` returns, which
  carry the blocks themselves.

.. code-block:: python

   mixed = dualmesh.mesh_from_arrays(
       [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], [("Tri3", [[0, 1, 2], [0, 2, 3]])]
   )
   print(mixed.num_elements)  # 2

``blocks`` gives one block number for each element, in the order of the elements, and every
element is in block 0 without it. :func:`~dualmesh.mesh_from_arrays` calls
:meth:`~dualmesh.Mesh.fix_orientation`, so the winding of your connectivity doesn't matter.

The mesh lies in the space that its coordinates span. A coordinate spans an axis when its range
is larger than :math:`10^{-12}` times the size of the mesh. A mesh of ``Edge2`` elements whose
:math:`y` varies is a curve in the plane, and a mesh of two-dimensional elements whose :math:`z`
varies is a surface in space. A constant coordinate keeps the mesh in its plane, so a plate at
:math:`z = 0.5` is a plane mesh. ``dimension=3`` places a flat plate in space, as a shell needs.

A mesh from arrays has no side sets and no node sets. For a domain whose sides are normal to the
axes, :meth:`~dualmesh.Mesh.add_bounding_box_sidesets` adds the side sets ``left``, ``right``,
``bottom``, ``top`` and, in three dimensions, ``back`` and ``front``. Each holds the exterior
faces whose nodes all lie on that face of the bounding box. A node lies on a face when its
distance to the face is at most ``tolerance``, :math:`10^{-10}` by default, multiplied by the
extent of the box along that axis when the extent is larger than 1. On a surface or a curve the method skips each
axis along which the mesh is flat. For a curved or inclined boundary, use the predicates of
:ref:`mesh-naming-boundaries`.

.. code-block:: python

   mesh.add_bounding_box_sidesets()
   print(mesh.sideset_names())  # ['bottom', 'left', 'right', 'top']

You can also build a mesh node by node: ``dualmesh.Mesh(dimension, space_dimension)`` makes an
empty mesh, :meth:`~dualmesh.Mesh.add_node` adds a node at ``(x, y, z)`` and
:meth:`~dualmesh.Mesh.add_element` adds an element of a named type in a block. Call
:meth:`~dualmesh.Mesh.fix_orientation` when you finish.

Reading and writing files
-------------------------

:func:`dualmesh.read_mesh` and :func:`dualmesh.write_mesh` use the meshio library [meshio]_, which
reads and writes Gmsh ``.msh`` [Gmsh2009]_, Exodus II ``.e`` and ``.exo``, VTK and VTK
unstructured grid (VTU), Abaqus ``.inp`` and the other formats that meshio supports. meshio takes
the format from the extension of the file, and ``file_format`` sets it when the extension is
ambiguous.

.. code-block:: python

   wrench = dualmesh.read_mesh("examples/wrench/wrench.msh")
   print(dualmesh.sideset_summary(wrench))

``length_unit`` is the length of one unit of the file's coordinates in metres, for example
``length_unit=1e-3`` for a file in millimetres. The default is 1.0.

:func:`~dualmesh.read_mesh` reads the cells of the fourteen supported types. The cells of the
highest dimension become elements, and the cells one dimension lower become side sets:

* The tag of an element, from the cell data ``gmsh:physical``, ``medit:ref``, ``cell_tags`` or
  ``material``, becomes its block number. When the file names the tag, as a Gmsh physical group
  does, the name becomes the block name. Untagged elements go into block 0.
* The boundary cells that share a tag become one side set and one node set of the same name. The
  name is the physical name of the tag, or ``boundary_<tag>`` when the file gives the tag no name.
  Gmsh numbers its physical groups separately in each dimension, so a group of elements and a
  group of boundary cells with the same tag keep their own names.
* ``boundary_names`` renames these side sets and node sets as you read them, as a mapping from the name in the file
  to the new name, for example ``{"boundary_1": "inlet"}``.
* When the file has no boundary cells, or when you pass ``add_bounding_box_sidesets=True``, the
  reader adds the bounding box side sets.

``examples/wrench/wrench.msh`` has the physical groups ``jaws`` and ``grip``, which become side
sets of those names. :doc:`/tutorials/wrench` uses them.

Check the boundaries of a file before you set up a problem on it.
:func:`dualmesh.sideset_summary` returns a table of the side sets with the number of faces and
nodes of each, its length or area, and the range of the coordinates of its nodes.
:func:`dualmesh.write_sidesets` writes the boundary faces to a ``.vtu`` file for ParaView, with
the cell field ``side_set`` and one field for each side set, which is 1 on its faces and 0
elsewhere. The command line runs both on a file without a solve:

.. code-block:: console

   dualmesh mesh examples/wrench/wrench.msh --sidesets wrench_sidesets.vtu

:func:`~dualmesh.write_mesh` writes a mesh and, as further keyword arguments, nodal fields of
those names:

.. code-block:: python

   dualmesh.write_mesh(mesh, "two_blocks.vtu")
   dualmesh.write_mesh(mesh, "two_blocks.msh", file_format="gmsh")
   dualmesh.write_mesh(mesh, "with_field.vtu", temperature=numpy.linspace(0.0, 1.0, mesh.num_nodes))

To write the fields of a solved problem, use the methods of :doc:`output`.

.. warning::

   A mesh that you write and read back loses its blocks and its side sets.
   :func:`~dualmesh.write_mesh` stores the block numbers as the cell data ``block``, which
   :func:`~dualmesh.read_mesh` doesn't read, and it doesn't write side sets. Keep the script that
   generates a mesh, or the original mesh file, as the definition of a mesh that you reuse.

.. _mesh-naming-boundaries:

Naming boundaries
-----------------

To add a boundary that a mesh lacks, select it by position. Each method below takes a predicate,
a Python function of ``(x, y, z)`` that returns ``True`` or ``False``. The method evaluates the
predicate once, when you call it.

.. code-block:: python

   mesh = dualmesh.generate_rectangle_mesh(
       x_min=0.0, x_max=2.0, y_min=0.0, y_max=1.0, num_x_elements=20, num_y_elements=10
   )

   mesh.add_sideset_by_predicate("outlet_half", lambda x, y, z: x > 2.0 - 1e-9 and y > 0.5)
   mesh.add_nodeset_by_predicate("centre_line", lambda x, y, z: abs(y - 0.5) < 1e-9)

   print(len(mesh.sideset("outlet_half")))  # 5
   print(len(mesh.boundary_nodes("centre_line")))  # 21

:meth:`~dualmesh.Mesh.add_sideset_by_predicate` tests the centroid of each exterior face, so a
predicate that holds only inside the domain selects no face.
:meth:`~dualmesh.Mesh.add_nodeset_by_predicate` tests every node, interior nodes included, so
``centre_line`` holds the nodes along the middle of the domain. A node set is how you prescribe a
value on a line or a plane inside the domain.

Give each predicate a tolerance for round-off. ``x > 2.0 - 1e-9`` selects the right edge, while
``x == 2.0`` can miss nodes whose coordinates carry a round-off error.
:meth:`~dualmesh.Mesh.alias_sideset` gives an existing side set a second name, for a mesh file and
a script that name the same boundary differently. :doc:`/tutorials/wrench` names the same
boundaries of a Gmsh file by physical groups and by predicates.

.. _transforming:

Transforming a mesh
-------------------

:meth:`~dualmesh.Mesh.transform_nodes` moves every node in place. Its function receives
``(x, y, z)`` and returns up to three coordinates. The elements, blocks, side sets and node sets
don't change, so a boundary keeps its name. To mesh a curved domain, generate a rectangle in the
parameter plane, name its edges and map it onto the domain:

.. code-block:: python

   import math

   rectangle = dualmesh.generate_rectangle_mesh(
       x_min=1.0, x_max=2.0, y_min=0.0, y_max=0.5 * math.pi, num_x_elements=4, num_y_elements=6
   )

   sector = rectangle.second_order()  # Promote before the map
   sector.transform_nodes(lambda r, theta, z: [r * math.cos(theta), r * math.sin(theta), 0.0])
   sector.fix_orientation()
   sector.alias_sideset("left", "inner")
   sector.alias_sideset("right", "outer")

   radii = numpy.hypot(*numpy.asarray(sector.points())[:, :2].T)
   print(round(radii.min(), 12), round(radii.max(), 12))  # 1.0 2.0

.. warning::

   Promote a mesh to second order before you map it onto a curved domain.
   :meth:`~dualmesh.Mesh.second_order` places each added node at the midpoint of its edge, face
   or cell. After the map, that midpoint lies on the straight chord between two corners on the
   circle, inside the curved boundary. In the example above, each element spans
   :math:`15^\circ` of the inner circle of radius 1, so a mid-side node added after the map lies at
   radius :math:`\cos(7.5^\circ) \approx 0.9914`. The quadratic elements then carry a geometric
   error that refinement along the arc reduces at first order only.

Call :meth:`~dualmesh.Mesh.fix_orientation` after :meth:`~dualmesh.Mesh.transform_nodes`, because
a map can reverse the winding of the elements.

Refining and promoting
----------------------

Uniform refinement
~~~~~~~~~~~~~~~~~~

:meth:`~dualmesh.Mesh.refined` returns a new mesh in which every element is split into
:math:`2^d` children, with :math:`d` the dimension of the elements. A prism becomes eight prisms,
and a pyramid becomes six pyramids and four tetrahedra. The side sets and node sets carry over,
so the conditions of the coarse mesh apply to the fine one, and the original mesh doesn't change.

.. code-block:: python

   coarse = dualmesh.generate_rectangle_mesh(
       x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=2, num_y_elements=2
   )

   fine = coarse.refined()
   print(fine.num_elements, fine.num_nodes)  # 16 25
   print(coarse.refined().refined().num_elements)  # 64

   quadratic = fine.second_order()
   print(quadratic.element_type(0), quadratic.num_elements, quadratic.num_nodes)  # Quad9 16 81

:meth:`~dualmesh.Mesh.refined` refuses a quadratic mesh. Refine the linear mesh, and then promote
the result.

Local refinement
~~~~~~~~~~~~~~~~

:func:`dualmesh.refine_marked` refines only the elements that you mark, with one flag for each
element. It takes a mesh of one element type:

* ``Tri3``, ``Tri6``, ``Tet4`` and ``Tet10`` are refined by bisection. The neighbours that the
  conformity of the mesh needs are bisected too.
* ``Quad4`` and ``Hex8`` are refined by subdivision into :math:`2^d` children. Nodes hang on the
  sides of the neighbours that stay whole, and the neighbours that keep a balance of 2:1 across
  every side are split too.

.. code-block:: python

   import numpy

   centroids = numpy.array([coarse.element_centroid(e) for e in range(coarse.num_elements)])
   marked = centroids[:, 0] < 0.5
   refined, parents = dualmesh.refine_marked(coarse, marked)

``parents[i]`` is the element of the original mesh that element ``i`` of the refined mesh comes
from, so you can carry a field stored per element to the refined mesh. :doc:`adaptivity`
describes the adaptive loop, which marks the elements from an error indicator, and
:doc:`/theory/adaptivity` describes the methods.

Promotion to second order
~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~dualmesh.Mesh.second_order` returns a copy with quadratic elements. ``Edge2`` becomes
``Edge3``, ``Tri3`` becomes ``Tri6``, ``Quad4`` becomes ``Quad9``, ``Tet4`` becomes ``Tet10`` and
``Hex8`` becomes ``Hex27``. With ``serendipity=True``, ``Quad4`` becomes ``Quad8`` and ``Hex8``
becomes ``Hex20``. A mesh with prisms or pyramids can't be promoted, because these types have no
quadratic form here.

The corner nodes keep their numbers and positions, so the domain keeps its shape and a node number
that you recorded stays valid. The side sets carry over unchanged. A node set gains each added node
whose parent nodes all belong to the set. When every member of a node set lies on the boundary,
the set gains only the added nodes on the boundary, so the midpoint of a diagonal between two
boundary nodes stays out of it. Uniform refinement grows node sets by the same rule.

Orientation
-----------

:meth:`~dualmesh.Mesh.fix_orientation` renumbers the nodes of each element whose Jacobian
determinant is negative, so that every element has a positive volume. An element with a negative
volume would give control domains of negative volume and a wrong matrix without an error. The
generators, :func:`~dualmesh.mesh_from_arrays` and :func:`~dualmesh.read_mesh` call it, so a
reversed quadrilateral comes back in the usual order:

.. code-block:: python

   flipped = dualmesh.mesh_from_arrays(
       [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], [[0, 3, 2, 1]], element_type="Quad4"
   )
   print(flipped.element_nodes(0))  # [0, 1, 2, 3]

The method tests the sign of the Jacobian determinant at the centroid of each element. It can't
repair a tangled element, whose Jacobian determinant changes sign inside it, because no order of
the nodes repairs it. Correct a tangled element where you generate the mesh.

With a mesh in hand, :doc:`problem_setup` attaches variables and physics to it, :doc:`solving`
solves the problem and :doc:`output` reads the results.
