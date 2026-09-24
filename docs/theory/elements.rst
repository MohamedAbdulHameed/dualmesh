Elements and the dual mesh they generate
========================================

The primal mesh of ``dualmesh`` is a mesh of Lagrange finite elements, and the
dual mesh of control domains is generated from it, element by element, from a
description held once per reference element
(``include/dualmesh/fe/ReferenceElement.h``, ``src/fe/ReferenceElement.cpp``).
This chapter states which elements exist, which discretisations accept them,
what their shape functions are, how the control domains inside them are
constructed, which quadrature rules are available, and — at some length,
because it is the result a reader is least likely to expect — what accuracy the
quadratic elements actually buy the dual mesh control domain method.  The notation of :doc:`foundations` is used
throughout: :math:`\boldsymbol{\xi}` denotes reference coordinates,
:math:`\psi_a` the shape function of local node :math:`a`, and the
isoparametric map :math:`\mathbf{x}(\boldsymbol{\xi}) = \sum_a \mathbf{x}_a
\psi_a(\boldsymbol{\xi})` [Irons1966]_ carries every reference construction
into physical space.

.. contents::
   :local:
   :depth: 2

The element families
--------------------

Fourteen element types are supported.  The **linear** elements interpolate
with polynomials of degree one in each reference direction (the pyramid with a
rational function, below).  The **quadratic** elements add a node at the
midpoint of every edge and interpolate with polynomials of degree two; the
*Lagrange* quadratic elements ``Quad9`` and ``Hex27`` also have nodes at the
face and element centres, and the *serendipity* elements ``Quad8`` and
``Hex20`` do without them.

=============  ===  =====  =====  ================================================
Type           Dim  Nodes  Order  Node set
=============  ===  =====  =====  ================================================
``Edge2``      1    2      1      the two ends
``Edge3``      1    3      2      the two ends and the midpoint
``Tri3``       2    3      1      the three vertices
``Tri6``       2    6      2      vertices and the three edge midpoints
``Quad4``      2    4      1      the four corners
``Quad8``      2    8      2      corners and the four edge midpoints
``Quad9``      2    9      2      corners, four edge midpoints, centre
``Tet4``       3    4      1      the four vertices
``Tet10``      3    10     2      vertices and the six edge midpoints
``Hex8``       3    8      1      the eight corners
``Hex20``      3    20     2      corners and the 12 edge midpoints
``Hex27``      3    27     2      corners, 12 edge midpoints, 6 face centres,
                                  centre
``Wedge6``     3    6      1      the six corners of a triangular prism
``Pyramid5``   3    5      1      the four base corners and the apex
=============  ===  =====  =====  ================================================

Every type can be used by the finite element method and by the cell-centred
finite volume method, which need only the element's geometry and shape
functions.  The dual mesh control domain method and the vertex-centred finite
volume method also need the element to be divided into node-centred control
domains, and three types cannot be: the two serendipity elements and the
pyramid (see `Which methods accept which elements`_).  A mesh may mix types
freely -- hexahedra, prisms, pyramids and tetrahedra in one mesh, for instance
-- and the methods apply to it element by element.

The tensor elements
^^^^^^^^^^^^^^^^^^^

``Edge2``, ``Edge3``, ``Quad4``, ``Quad9``, ``Hex8`` and ``Hex27`` have their
nodes on a tensor-product grid of one-dimensional coordinates: the set
:math:`c = \{-1, +1\}` for the linear members and
:math:`c = \{-1, 0, +1\}` for the quadratic ones, in every direction.  Let
:math:`L_k` be the one-dimensional Lagrange polynomial that is one at
:math:`c_k` and zero at the other grid coordinates.  For the linear grid,

.. math::

   L_{-1}(t) = \tfrac{1}{2}(1 - t) , \qquad
   L_{+1}(t) = \tfrac{1}{2}(1 + t) ,

and for the quadratic grid,

.. math::
   :label: lagrange_quadratic

   L_{-1}(t) = \tfrac{1}{2}\,t\,(t - 1) , \qquad
   L_{0}(t) = 1 - t^{2} , \qquad
   L_{+1}(t) = \tfrac{1}{2}\,t\,(t + 1) .

If node :math:`a` sits at the grid position :math:`(k_1, k_2, k_3)`, its shape
function is the product of the one-dimensional polynomials of its own
coordinates,

.. math::
   :label: tensor_shape

   \psi_a(\boldsymbol{\xi}) = \prod_{d=1}^{n_{\mathrm{dim}}} L_{k_d}(\xi_d) ,

and its gradient is obtained by differentiating one factor at a time.  This is
literally how ``ReferenceElement::shape`` computes them: one routine serves the
linear and the quadratic elements, the only difference being the grid :math:`c`
it is given.

The reference nodes are ordered as VTK orders them, for every element type,
so that a mesh read from or written to a file needs no renumbering.  Corners
come first.  ``Quad4`` has its corners at :math:`(-1,-1)`, :math:`(1,-1)`,
:math:`(1,1)`, :math:`(-1,1)`; ``Quad8`` adds the edge midpoints
:math:`(0,-1)`, :math:`(1,0)`, :math:`(0,1)`, :math:`(-1,0)`, and ``Quad9``
adds after them the centre :math:`(0,0)`.  ``Hex8`` takes the four corners of
the face :math:`\xi_3 = -1` and then the four of :math:`\xi_3 = +1`;
``Hex20`` adds the twelve edge midpoints -- the four of the bottom face, the
four of the top face, then the four vertical edges -- and ``Hex27`` adds after
them the six face centres in the order :math:`\xi_1 = -1`,
:math:`\xi_1 = +1`, :math:`\xi_2 = -1`, :math:`\xi_2 = +1`,
:math:`\xi_3 = -1`, :math:`\xi_3 = +1`, and finally the centre
:math:`(0,0,0)`.  ``Edge2`` numbers its nodes :math:`\xi = -1` then
:math:`+1`, and ``Edge3`` puts the midpoint last, so its nodes are at
:math:`\xi = -1, +1, 0`.

The simplices
^^^^^^^^^^^^^

``Tri3``, ``Tri6``, ``Tet4`` and ``Tet10`` are built from the barycentric
coordinates of the reference simplex,

.. math::

   \lambda_0 = 1 - \sum_{d=1}^{n_{\mathrm{dim}}} \xi_d ,
   \qquad
   \lambda_i = \xi_i \quad (i = 1, \dots, n_{\mathrm{dim}}) ,

whose gradients are constant: :math:`\nabla \lambda_i = \mathbf{e}_i` and
:math:`\nabla \lambda_0 = -\sum_i \mathbf{e}_i`.  The reference triangle has
vertices :math:`(0,0)`, :math:`(1,0)`, :math:`(0,1)` and the reference
tetrahedron adds :math:`(0,0,1)`.  For the linear elements the shape functions
are the barycentric coordinates themselves,

.. math::

   \psi_i = \lambda_i ,

and for the quadratic ones they are

.. math::
   :label: simplex_quadratic

   \psi_i = \lambda_i \, (2 \lambda_i - 1) \quad \text{at a vertex} ,
   \qquad
   \psi_{ij} = 4 \, \lambda_i \, \lambda_j
   \quad \text{at the midpoint of edge } (i,j) .

``Tri6`` takes its edges as :math:`(0,1)`, :math:`(1,2)`, :math:`(2,0)`, so
nodes 3, 4, 5 sit at :math:`(\tfrac12, 0)`, :math:`(\tfrac12, \tfrac12)`,
:math:`(0, \tfrac12)`; ``Tet10`` takes its six edges as :math:`(0,1)`,
:math:`(1,2)`, :math:`(0,2)`, :math:`(0,3)`, :math:`(1,3)`, :math:`(2,3)`, so
nodes 4 to 9 sit at :math:`(\tfrac12,0,0)`, :math:`(\tfrac12,\tfrac12,0)`,
:math:`(0,\tfrac12,0)`, :math:`(0,0,\tfrac12)`, :math:`(\tfrac12,0,\tfrac12)`,
:math:`(0,\tfrac12,\tfrac12)`.

A unit test checks the three properties every Lagrange basis must have: each
shape function is one at its own node and zero at the others to
:math:`10^{-13}`, the basis sums to one to :math:`10^{-12}`, and the gradients
sum to zero to :math:`10^{-12}` (``quadratic_shape_functions`` in
``tests/cpp/unit_tests.cpp``).

The serendipity elements, the prism and the pyramid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **serendipity** elements drop the interior nodes of the Lagrange quadratic
elements and keep a complete quadratic polynomial, plus the terms needed for
the element to be conforming.  The shape function of ``Quad8`` at a corner
:math:`(\xi_a, \eta_a)` and at the midpoint of an edge with
:math:`\xi_a = 0` are

.. math::

   \psi_a = \tfrac14 (1 + \xi \xi_a)(1 + \eta \eta_a)(\xi \xi_a + \eta \eta_a - 1) ,
   \qquad
   \psi_a = \tfrac12 (1 - \xi^2)(1 + \eta \eta_a) ,

and those of ``Hex20`` are the three-dimensional analogues,
:math:`\tfrac18 (1 + \xi\xi_a)(1 + \eta\eta_a)(1 + \zeta\zeta_a)
(\xi\xi_a + \eta\eta_a + \zeta\zeta_a - 2)` at a corner and
:math:`\tfrac14 (1 - \xi^2)(1 + \eta\eta_a)(1 + \zeta\zeta_a)` at an edge
midpoint [Reddy2019b]_.

The **triangular prism** ``Wedge6`` is the product of a triangle and a
segment: its reference element is the triangle with vertices :math:`(0,0)`,
:math:`(0,1)`, :math:`(1,0)` (in the VTK order) extruded over
:math:`-1 \le \zeta \le 1`, and its shape functions are the triangle's
barycentric coordinates times :math:`\tfrac12 (1 \mp \zeta)`.  It
conforms to a ``Tet4`` across a triangular face and to a ``Hex8`` across a
quadrilateral one, which is what makes it the usual element for a layer of
cells along a wall.

The **pyramid** ``Pyramid5`` joins a quadrilateral face to four triangles.  No
polynomial basis on five nodes is linear on the triangular faces and bilinear
on the base at the same time, so it uses the rational basis of Bedrosian
[Bedrosian1992]_:

.. math::

   \psi_a = \frac{(\zeta - s_a \xi - 1)(\zeta - t_a \eta - 1)}{4 (1 - \zeta)}
   \quad \text{at the base corner with signs } (s_a, t_a) ,
   \qquad
   \psi_4 = \zeta \quad \text{at the apex} ,

on the reference pyramid with base :math:`[-1,1]^2` at :math:`\zeta = 0` and
apex :math:`(0,0,1)`.  It is linear on every edge and triangular face and
bilinear on the base, so it conforms to both tetrahedra and hexahedra.  The
denominator vanishes only at the apex, which no quadrature point reaches.

Which methods accept which elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

=============  ======  ======  ======  ======
Type           fem     zfvm    dmcdm   hfvm
=============  ======  ======  ======  ======
``Quad8``      yes     yes     no      no
``Hex20``      yes     yes     no      no
``Pyramid5``   yes     yes     no      no
every other    yes     yes     yes     yes
=============  ======  ======  ======  ======

The dual mesh control domain method and the vertex-centred finite volume
method both integrate the balance law over node-centred control domains, and
those must *partition* the element: every point belongs to exactly one control
domain, and every node owns one.  The median dual makes that assignment from
the node set, and it is not defined for three of the elements.

* A **serendipity** element has mid-edge nodes and nothing in the interior.
  The region around the element centre then belongs to no node: there is no
  centre node to give it to, and handing it to a corner or a mid-edge node
  instead would destroy the symmetry of the partition, make the control domains
  depend on the node numbering, and break the property on which the method
  rests, that the control domains tile the element and their surfaces close.

* The **pyramid**'s apex is shared by four edges rather than three.  The median
  sub-cell of the apex is then not a hexahedral patch like every other
  sub-cell in the library, and the non-tensor decomposition that would match
  the control domain interfaces exactly is not implemented.

The finite element and cell-centred finite volume methods need no partition --
the first integrates over the element and the second over the cell -- so they
accept all fourteen types.  The check is made when a problem is created: a
dual-mesh method on a mesh containing one of the three types raises an error
that names the type, gives the reason above, and says which methods do accept
it (``test_element_types.py``).  Promoting a linear mesh to the serendipity
elements is ``mesh.second_order(serendipity=True)``, and the generators accept
``Quad8`` and ``Hex20`` directly.

How the dual mesh is built inside an element
--------------------------------------------

The control domains and their interfaces are never meshed explicitly.  Each is
described once, in reference coordinates of the parent element, as a
tensor-product *patch* of dimension zero to three, and is carried into physical
space by the element's own isoparametric map; the same description therefore
serves a straight element, a distorted one and a quadratic one whose edges are
curved.  Two constructions are needed, because the nodes of a tensor element
lie on a grid and admit an exact box construction while the nodes of a simplex
do not.

Tensor elements: boxes
^^^^^^^^^^^^^^^^^^^^^^

For a tensor element, the control domain of a node is the box bounded, in each
direction, by the planes half way to its neighbours on the grid; at the edge of
the element the box stops at the element boundary.  If node :math:`a` has grid
index :math:`k` in direction :math:`d`, its box spans

.. math::

   \Bigl[ \tfrac{1}{2}(c_{k-1} + c_{k}) , \; \tfrac{1}{2}(c_{k} + c_{k+1}) \Bigr]
   \quad \text{in } \xi_d ,

with the lower bound replaced by :math:`c_0` when :math:`k = 0` and the upper
bound by :math:`c_{n-1}` when :math:`k = n-1`.  Two nodes share an interface
when their grid indices differ by one in exactly one direction, and the
interface is the face of their boxes on the plane half way between them.

For the linear elements this is the familiar bisecting construction: a
``Quad4`` is cut into four equal quadrilaterals by the two lines through the
element centroid, a ``Hex8`` into eight boxes.  For the quadratic ones the grid
is :math:`\{-1, 0, 1\}`, so the cuts fall at :math:`\xi = \pm \tfrac12` and
the nine control domains of a ``Quad9`` take unequal shares: each corner node
owns a box such as :math:`[-1,-\tfrac12]^2`, a sixteenth of the reference area,
each mid-edge node an eighth, and the centre node
:math:`[-\tfrac12,\tfrac12]^2`, a quarter.
The shares sum to one, as they must.  The position :math:`\xi = \pm\tfrac12`
of the interfaces is not incidental; it is the subject of the accuracy section
below.

Simplices: sub-simplices and median sub-cells
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A simplex has no grid, so the box construction does not apply.  Instead the
element is first written as a set of **linear sub-simplices**, and the median
dual of each sub-simplex is accumulated onto its vertices.  A linear element is
its own single sub-simplex; a quadratic one is subdivided.

Inside one linear sub-simplex with vertices :math:`X_0, \dots, X_{n}` and
centroid :math:`X_c`, the median sub-cell of vertex :math:`X_i` is the region
bounded by the vertex, the midpoints of the edges that meet at it, the
centroids of the faces that meet at it, and the centroid of the sub-simplex.
In two dimensions it is the quadrilateral with corners

.. math::

   X_i , \quad \tfrac{1}{2}(X_i + X_j) , \quad X_c , \quad
   \tfrac{1}{2}(X_i + X_k) ,

and in three dimensions it is the hexahedron whose eight corners are the
vertex, the three edge midpoints, the three face centroids and :math:`X_c`.
The interface between the sub-cells of :math:`X_i` and :math:`X_j` passes
through the midpoint of their common edge, the centroids of the faces that
share that edge, and :math:`X_c`: a segment in two dimensions, a quadrilateral
in three.

For ``Tri3`` and ``Tet4`` there is one sub-simplex and this is the classical
median dual.  For the quadratic simplices the element is first split by **red
refinement**, which is the uniform subdivision that connects the mid-edge
nodes.  A ``Tri6`` becomes four sub-triangles, given in local node numbers as

.. math::

   (0,3,5) , \qquad (3,1,4) , \qquad (5,4,2) , \qquad (3,4,5) ,

that is, three corner triangles and the central one.  A ``Tet10`` becomes eight
sub-tetrahedra: the four corner tetrahedra

.. math::

   (0,4,6,7) , \qquad (4,1,5,8) , \qquad (6,5,2,9) , \qquad (7,8,9,3) ,

and the four that tile the inner octahedron, split along the diagonal that
joins nodes 4 and 9,

.. math::

   (4,9,5,6) , \qquad (4,9,6,7) , \qquad (4,9,7,8) , \qquad (4,9,8,5) .

The control domain of a node is then the **union of its median sub-cells over
every sub-simplex that touches it**, which is why a control domain is stored as
a list of patches rather than as one.  The arithmetic is worth doing once,
because it shows how unequal the partition is.  On a ``Tri6`` of area :math:`A`
each sub-triangle has area :math:`A/4` and each median sub-cell within it
:math:`A/12`; a corner node belongs to one sub-triangle and owns :math:`A/12`,
a mid-edge node belongs to three and owns :math:`A/4`, so the corner nodes
share a quarter of the element and the mid-edge nodes three quarters.  On a
``Tet10`` of volume :math:`V` every sub-tetrahedron has volume :math:`V/8` and
every median sub-cell :math:`V/32`; a corner node owns :math:`V/32`, nodes 4
and 9 belong to two corner tetrahedra and to all four inner ones and own
:math:`6V/32 = 3V/16` each, and the remaining four mid-edge nodes own
:math:`4V/32 = V/8` each.  The shares sum to :math:`V`.

The position of nodes 4 and 9 in that list is a genuine, if minor, defect: the
inner octahedron can be split along any of its three diagonals, the code always
chooses :math:`4`–:math:`9`, and the two nodes on it receive larger control
domains than the other four.  The partition is therefore not invariant under a
renumbering of the element's nodes.  It remains a valid partition — it tiles
and it closes — and the discretisation remains consistent, as the patch test
confirms, but two differently numbered ``Tet10`` meshes of the same geometry
will give slightly different numbers.

The prism: a product of duals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A triangular prism is the product of a triangle and a segment, and its median
dual is the product of theirs.  The control domain of a node is the median
sub-cell of its triangle vertex (the quadrilateral bounded by the vertex, the
two edge midpoints and the triangle centroid) times the half of the segment on
the node's side of the mid-plane :math:`\zeta = 0`, a hexahedral patch.  The
interfaces are of two kinds: the median segments of the triangle times a half
segment, between nodes on the same end face, and the median quadrilateral of a
vertex in the mid-plane, between a node and the node above it.  The
construction is exactly the general median construction -- bounded by edge
midpoints, face centroids and the element centroid -- so on a triangular face
it matches the dual of a neighbouring tetrahedron and on a quadrilateral face
that of a neighbouring hexahedron, and a mesh of hexahedra, prisms and
tetrahedra has a conforming dual.

Verification rather than citation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The median dual of a linear simplex and the box dual of a tensor element are
standard; the construction just described for a *quadratic simplex* is not.  It
is not a published construction, as far as is known here: it was devised for
this library because the method needed one, the literature on the dual mesh
control domain method having treated linear and tensor-product elements.  It is
therefore offered as verified rather than as cited, and the verification is the
pair of properties a partition must have:

* **Tiling.** The measures of an element's control domains, each integrated
  with a four-point Gauss rule on every patch, sum to the measure of the
  element.
* **Closure.** For every node, the signed area vectors of all the surfaces
  bounding its control domain — the interfaces it shares with the other nodes,
  plus the parts of the element's sides that belong to it — sum to the zero
  vector.  This is the discrete divergence theorem applied to a constant field,
  and a discretisation whose control domains do not close cannot reproduce a
  constant flux.

Both are checked for every element type that has a dual, on meshes distorted
so that no element is a parallelogram and no quadratic edge is straight
(``dual_mesh_partitions_element``, ``quadratic_dual_mesh_partitions_element``
and ``prism_dual_mesh_partitions_element`` in ``tests/cpp/unit_tests.cpp``).
The tolerance is :math:`10^{-12}` for the linear elements and :math:`10^{-11}`
for the quadratic ones, which is quadrature noise rather than a geometric
discrepancy.  The patch test in ``tests/python/test_quadratic_elements.py``
adds the analytic counterpart: a linear field is reproduced exactly on
distorted ``Tri6``, ``Quad9``, ``Tet10`` and ``Hex27`` meshes, to
:math:`10^{-11}` or better.

Quadrature
----------

Every kernel names its own rule with the ``quadrature`` parameter, because in a
control volume method the rule is part of the model and not only a numerical
detail: a lumped source and a Gauss-integrated source are different
discretisations of the same equation, not two approximations of the same
discretisation.  The rules are one-dimensional rules applied as tensor products
on each patch of a control domain, of an interface or of an element, except
where a symmetric rule for the whole element is cheaper (below).

``gauss1`` … ``gauss10``
    Gauss–Legendre with that many points per direction, exact for polynomials
    of degree :math:`2n - 1`.  The points and weights are computed once, by
    Newton iteration on the Legendre polynomial, and then looked up.  Asking
    for more than ten is an error, and ``gauss`` without a number means
    ``gauss2``.

``midpoint`` (alias ``centroid``)
    One point at the centre of the patch, which is ``gauss1``.  Combined with
    ``reduced_integration=True`` it gives the selective reduced integration used
    for the transverse shear terms of beams and plates and for the penalty term
    of incompressible flow [ZienkiewiczTaylorToo1971]_ [HughesCohenHaroun1978]_
    [MalkusHughes1978]_.

``trapezoid`` (alias ``trapezoidal``), ``simpson``
    The corner rule :math:`\{-1, +1\}` with weights :math:`\{1, 1\}`, and
    Simpson's rule :math:`\{-1, 0, +1\}` with weights
    :math:`\{\tfrac13, \tfrac43, \tfrac13\}`, per direction; these reproduce
    classical finite volume and finite difference source treatments.

``nodal`` (alias ``lumped``)
    A single point at the owning node, weighted by the measure of the node's
    control domain for the dual mesh and finite volume methods and by
    :math:`\int \psi_a \,\mathrm{d}V` for the finite element method, which is
    row-sum lumping.  This is how a lumped capacity or mass term is obtained.

``interface``
    A single point at the corner of the control domain lying deepest inside the
    element — the control domain interface in one dimension, the element
    centroid in two and three — with the measure of the control domain as the
    weight, which reproduces the trapezoidal source rule of the finite volume
    literature.

``control_domain_trapezoid`` (alias ``cd_trapezoid``)
    The trapezoidal rule over the whole control domain: for an interior node in
    one dimension :math:`F_I = \tfrac{\Delta x}{2}\,[f(x_A) + f(x_B)]` with
    :math:`x_A` and :math:`x_B` the two interfaces, while a boundary node, whose
    control domain is bounded by the domain boundary as well, puts half the
    weight at its single interface and half at the node itself.  This is the
    source rule of Chapter 3 of [Reddy2024]_ and reproduces Example 3.3.1 of the
    book exactly.

``automatic`` (alias ``auto``, the default)
    Gauss–Legendre with one more point per direction than the polynomial order
    of the mesh, which integrates a product of two shape functions exactly.  The
    choice is made once, when the object is set up and the mesh is known: **two
    points per direction on a linear mesh and three on a quadratic one**, a mesh
    containing any quadratic element counting as quadratic.

**Simplices, prisms and pyramids.**  A patch that is a triangle, a
tetrahedron, a prism or a pyramid is integrated as a square or a cube with some
of its corners merged, the collapsed-coordinate map of Duffy [Duffy1982]_.  The
collapse puts a factor of degree :math:`n_{\mathrm{dim}} - 1` into the
Jacobian of the map, so a Gauss rule exact to a given degree on the element
needs one more point per direction on the collapsed patch, and the library adds
it.  For the finite element method, which integrates over the whole element,
there is a cheaper choice on triangles, tetrahedra and prisms: a *symmetric*
rule whose points form orbits of the symmetry group of the simplex.  ``gaussn``
is then replaced by the symmetric rule exact to degree :math:`2n - 1`: 1, 6 and
7 points on a triangle for :math:`n = 1, 2, 3` (the last two those of
Dunavant [Dunavant1985]_), 1, 8 and 15 points on a tetrahedron, and the product of the triangle
rule with a Gauss rule on a prism.  The default rule on a ``Tet4`` mesh uses 8
points instead of the 27 of the collapsed rule, and the assembly is three times
faster.  All the rules have positive weights and interior points, and a unit
test (``simplex_quadrature_is_exact``) integrates every monomial up to the
stated degree exactly.

The last three rules place their points by reference to a node-centred control
domain, so they are meaningful only for volume terms.  On a control domain
interface they are replaced by ``gauss2`` and on an element side by ``gauss3``;
for a side, a ``nodal`` rule additionally collapses its points onto the node
while preserving the total vector area, so that the boundary flux is still
integrated exactly for a constant flux.

The accuracy of the dual mesh method on quadratic elements
-----------------------------------------------------------

The finite element method gains a great deal from quadratic elements.  On a
smooth problem its nodal values are fourth-order accurate — two orders better
than the energy norm — which is the classical superconvergence of the Galerkin
method and is what makes the extra unknowns worth paying for.

That the nodal values are better than the global rate is the result of
Douglas and Dupont [DouglasDupont1974]_, and that the derivative is most
accurate at the Gauss points is the result of Barlow [Barlow1976]_.  Both are
needed below.

**The dual mesh
control domain method does not gain an order from quadratic elements.  It stays
second order.**  This section explains why, proves it in one dimension, and
gives the measured numbers in two.

Why: where the interfaces sit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The flux through a control domain interface is the flux function evaluated at
the gradient of the interpolant *there*: for a diffusion problem it is
:math:`k \, \nabla u` with :math:`\nabla u = \sum_a U_a \nabla \psi_a`.  The
truncation error of the method is therefore governed by the error the
interpolant's gradient makes at the interface points, and by nothing else.

For a quadratic interpolant of a smooth function on an element of size
:math:`h`, that derivative error is :math:`O(h^{2})` at a generic point but
:math:`O(h^{3})` at the two Gauss points :math:`\xi = \pm 1/\sqrt{3}`, which
are the *superconvergent points* of the derivative.  A method that samples the
gradient there — as the Galerkin method effectively does, its element integrals
being dominated by the Gauss points — inherits the extra order.

The median dual does not sample there.  Its interfaces lie half way between the
nodes, at :math:`\xi = \pm \tfrac12`, and
:math:`\tfrac12 \neq 1/\sqrt{3} \approx 0.5774`.  At :math:`\xi = \pm\tfrac12`
the gradient carries its generic :math:`O(h^{2})` error, the errors at the two
interfaces bounding a control domain do not cancel beyond leading order, and
the balance is left with a truncation error one order lower than the Galerkin
method's.  Moving the interfaces to the Gauss points is not an option: their
position at :math:`\xi = \pm\tfrac12` is what makes the control domains tile
the element and close, and shifting them would destroy the partition.

The one-dimensional proof
^^^^^^^^^^^^^^^^^^^^^^^^^

In one dimension the statement can be made exactly, with no asymptotics at all.
Take an ``Edge3`` element with nodes at :math:`\xi = -1`, :math:`\xi = 0` and
:math:`\xi = +1`, carrying values :math:`u_0`, :math:`u_m` and :math:`u_1`,
where :math:`u_m` is the value at the midpoint node.  From
:eq:`lagrange_quadratic` the interpolant is

.. math::

   p(\xi) = N_0 u_0 + N_1 u_1 + N_2 u_m ,
   \qquad
   N_0 = \tfrac{1}{2}\xi(\xi - 1) , \quad
   N_1 = \tfrac{1}{2}\xi(\xi + 1) , \quad
   N_2 = 1 - \xi^{2} ,

and its derivative with respect to :math:`\xi` is

.. math::
   :label: edge3_derivative

   p'(\xi) = \bigl(\xi - \tfrac12\bigr) u_0
           + \bigl(\xi + \tfrac12\bigr) u_1
           - 2\xi \, u_m .

Evaluate :eq:`edge3_derivative` at the two control domain interfaces.  At
:math:`\xi = -\tfrac12`,

.. math::

   p'\bigl(-\tfrac12\bigr) = (-1)\,u_0 + 0 \cdot u_1 + u_m = u_m - u_0 ,

and at :math:`\xi = +\tfrac12`,

.. math::

   p'\bigl(+\tfrac12\bigr) = 0 \cdot u_0 + 1 \cdot u_1 - u_m = u_1 - u_m .

Both collapse to a **two-point difference across a half element**.  On a
physical element of length :math:`h` the map gives
:math:`\mathrm{d}u/\mathrm{d}x = (2/h)\,p'(\xi)`, so the gradient the method
uses at the left interface is exactly :math:`(u_m - u_0)/(h/2)` and at the
right interface :math:`(u_1 - u_m)/(h/2)`.  The quadratic term has vanished
identically: at the interfaces the interpolant's derivative knows nothing about
the curvature the third node contributes, and the node on the far side of the
element has no influence at all.

The consequence is exact, not asymptotic: the dual mesh discretisation of a
quadratic one-dimensional mesh produces, term by term, the same flux
expressions as the dual mesh discretisation of the linear mesh on the same
points, that is, of twice as many ``Edge2`` elements.  With the quadrature of
the source held fixed, so that the ``automatic`` rule does not silently change
with the element order, the two computations agree to machine precision:
solving :math:`-u'' = \pi^2 \sin \pi x` on :math:`(0,1)` with four ``Edge3``
elements and with eight ``Edge2`` elements gives nodal values differing by
:math:`2 \times 10^{-16}`
(``test_one_dimensional_dual_mesh_on_quadratic_elements_equals_the_refined_linear_mesh``).
In one dimension a quadratic element buys the method nothing beyond the
refinement its extra nodes represent.

The two-dimensional numbers
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In two dimensions the algebra does not collapse so cleanly, but the order does
not improve.  The measurements below solve

.. math::

   -\nabla^2 u = 2\pi^{2} \sin \pi x \, \sin \pi y
   \quad \text{on } (0,1)^{2} , \qquad u = 0 \text{ on the boundary} ,

whose exact solution is :math:`u = \sin \pi x \, \sin \pi y`, on uniform meshes
of :math:`4 \times 4`, :math:`8 \times 8` and :math:`16 \times 16` elements,
and report the maximum nodal error and the rate observed between successive
meshes.

=========  =======  ====================  ====================  ====================  ==========
Element    Method   :math:`4\times4`      :math:`8\times8`      :math:`16\times16`    Rates
=========  =======  ====================  ====================  ====================  ==========
``Quad9``  dmcdm    6.92e-3               1.64e-3               4.03e-4               2.08, 2.02
``Quad9``  fem      5.56e-4               3.35e-5               2.07e-6               4.05, 4.01
``Tri6``   dmcdm    1.64e-2               3.78e-3               9.22e-4               2.12, 2.03
``Tri6``   fem      3.52e-3               2.29e-4               1.44e-5               3.95, 3.99
=========  =======  ====================  ====================  ====================  ==========

The rates are unambiguous: two for the dual mesh control domain method and four
for the finite element method, on both element shapes.

What the quadratic element does change for the dual mesh method is the
constant, and the sign of the change depends on the element.  Compared at
**equal node count** — a quadratic mesh of :math:`n \times n` elements has the
nodes of a linear mesh of :math:`2n \times 2n` — the measured error ratios
are

========================  ==============  ==============  ===============
Ratio                     :math:`n = 4`   :math:`n = 8`   :math:`n = 16`
========================  ==============  ==============  ===============
``Quad9`` over ``Quad4``  0.357           0.339           0.335
``Tri6`` over ``Tri3``    8.84            8.01            7.79
========================  ==============  ==============  ===============

Promoting ``Quad4`` to ``Quad9`` divides the error by about three; promoting
``Tri3`` to ``Tri6`` multiplies it by about eight.  Both ratios are steady
under refinement, which is the statement that the order has not changed in
either case.  The triangular result is the one to keep in mind: the median dual
of a linear triangle is an unusually favourable configuration for this
discretisation — it is also the case in which the method coincides with the
Galerkin finite element method for constant coefficients — and the sub-cells of
the red-refined triangle are not.  For the dual mesh control domain method,
``Tri6`` is worse than ``Tri3`` at the same cost, and the honest advice is not
to use it for accuracy.

None of this applies to the finite element method, which gains its two extra
orders on both element shapes, nor to the two finite volume methods, which are
second-order schemes on any mesh.

What quadratic elements do buy: geometry
-----------------------------------------

The reason to use a quadratic element with the dual mesh control domain method
is not the interpolation of the solution; it is the interpolation of the
*domain*.

An element edge on a curved boundary is a chord when the element is linear.
When the element is quadratic and its mid-side node has been placed on the true
boundary, the edge is the parabola through three points of the arc, and the
isoparametric map carries the control domains and their interfaces onto that
curved region; the error in the represented geometry drops from
:math:`O(h^{2})` to :math:`O(h^{4})` and every integral over the domain
inherits the improvement.  The test measures this by integrating the constant
:math:`1` over a quarter annulus of inner radius :math:`1` and outer radius
:math:`2`, whose exact area is :math:`3\pi/4`; on meshes of
:math:`2\times2`, :math:`4\times4` and :math:`8\times8` elements the absolute
error is

=====================  ================  ================  ================  ==========
Element                :math:`2\times2`  :math:`4\times4`  :math:`8\times8`  Rates
=====================  ================  ================  ================  ==========
``Quad4`` / ``Tri3``   2.35e-1           6.01e-2           1.51e-2           1.97, 1.99
``Quad9`` / ``Tri6``   1.83e-3           1.16e-4           7.29e-6           3.98, 4.00
=====================  ================  ================  ================  ==========

The linear and the quadratic families give identical numbers within each row
because the triangular mesh is the quadrilateral mesh cut along a diagonal and
the two cover the same region.  The rate is two for the linear elements and
four for the quadratic ones, and on the coarsest mesh the quadratic element is
already more than a hundred times more accurate
(``test_curved_boundaries_are_resolved_to_fourth_order``).  For a problem on a
cylinder, an annulus or a plate with a hole, this is usually the dominant error
of a coarse linear mesh, and it is the reason the quadratic family exists in
this library even though it does not raise the order of the discretisation.

Promotion, generation and file formats
---------------------------------------

A linear mesh is promoted with :meth:`dualmesh.Mesh.second_order`
(``Mesh::secondOrder``): ``Edge2`` becomes ``Edge3``, ``Tri3`` becomes
``Tri6``, ``Quad4`` becomes ``Quad9``, ``Tet4`` becomes ``Tet10`` and ``Hex8``
becomes ``Hex27``.  The corner nodes keep their numbers and positions; each
added node is placed at the image, under the *linear* shape functions of the
original element, of the point the quadratic reference element gives it, and is
identified by the sorted set of corner nodes it interpolates, so elements that
share an edge or a face share the node on it and the promoted mesh is
conforming.  Side sets and node sets are carried over and now cover the new
mid-edge nodes.  Promoting a mesh that is already quadratic does nothing.  The
generators ``generate_line_mesh``, ``generate_rectangle_mesh``,
``generate_box_mesh`` and ``generate_annulus_mesh`` accept a quadratic element
type directly: they build the linear mesh and promote it, so the element count
is unchanged and only the node count grows.

**A curved domain must be promoted before its coordinates are transformed.**
Getting this ordering wrong silently costs the fourth-order geometry measured
above.  A rectangular mesh in :math:`(r, \theta)` that is mapped to an annulus
and only then promoted has its mid-side nodes at the midpoints of the straight
chords, and its edges stay chords whatever element type it claims; promoted
first, the mid-side nodes are created in :math:`(r,\theta)` space and the
mapping carries them onto the true circle.  ``generate_annulus_mesh`` does it
in the correct order, and a user who transforms nodes by hand must do the
same.

Uniform refinement and promotion do not commute and refinement is not offered
on a quadratic mesh: :meth:`dualmesh.Mesh.refined` raises an error telling the
caller to refine the linear mesh first and promote the result.

All fourteen element types are read and written through ``meshio`` [meshio]_
under their VTK cell types (``VTK_WEDGE``, ``VTK_PYRAMID``,
``VTK_QUADRATIC_QUAD`` and ``VTK_QUADRATIC_HEXAHEDRON`` among them), and
because the library numbers the nodes of every type as VTK does, no
permutation is applied in either direction: a mesh written and read back is
the mesh that was written, node for node
(``test_quadratic_meshes_survive_a_file_round_trip`` and the round-trip test of
``test_element_types.py``).  A mesh whose elements are inverted -- numbered
clockwise where the convention is anticlockwise, as some generators produce --
is repaired on reading by renumbering those elements.

One practical limit belongs here.  The automatic differentiation that builds
the Jacobian [Wengert1964]_ seeds one derivative slot per element node and
variable, within a default budget of 48 slots, so a ``Hex27`` mesh spends 27
slots on one variable and cannot hold a second.  The library detects this when
the second variable is added and raises an error naming the element type, the
number of variables that do fit and the build setting that raises the limit
(``-DDUALMESH_MAX_AD_DERIVATIVES=<n>``), rather than failing obscurely during
assembly; a multi-field problem on ``Hex27`` needs a rebuild.
