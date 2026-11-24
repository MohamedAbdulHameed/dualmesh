Elements and the dual mesh they generate
========================================

The primal mesh is a mesh of Lagrange finite elements, and the code builds the
dual mesh of control domains from it, element by element.  This page gives the
element types, their shape functions, the construction of the control domains
inside each element, the quadrature rules and the accuracy that quadratic
elements give the two methods: the finite element method (FEM) and the dual
mesh control domain method (DMCDM).

Throughout the page, :math:`\boldsymbol{\xi} = (\xi_1, \xi_2, \xi_3)` are the
reference coordinates, also written :math:`(\xi, \eta, \zeta)`, :math:`N_a` is
the shape function of the local node :math:`a`, and :math:`\mathbf{x}_a` is the
position of that node (m).  The isoparametric map
:math:`\mathbf{x}(\boldsymbol{\xi}) = \sum_a \mathbf{x}_a N_a(\boldsymbol{\xi})`
[Irons1966]_ carries every construction from the reference element into the
mesh.  The element size :math:`h` (m) is the length of an element edge.

.. contents::
   :local:
   :depth: 1

The element types
-----------------

The code has fourteen element types.  The **linear** elements interpolate with
polynomials of degree one in each reference direction, and the pyramid with a
rational function.  The **quadratic** elements add a node at the midpoint of
every edge and interpolate with polynomials of degree two.  The Lagrange
quadratic elements ``Quad9`` and ``Hex27`` also have nodes at the centres of the
faces and of the element, and the serendipity elements ``Quad8`` and ``Hex20``
omit them.

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

FEM accepts every type, because it needs only the geometry and the shape
functions of an element.  DMCDM also needs the element divided into
node-centred control domains, which ``Quad8``, ``Hex20`` and ``Pyramid5`` do not
have (:ref:`elements-without-dual`).  DMCDM on a mesh that contains one of these
three types stops with an error that names the type and the reason.

The nodes of every side are wound so that their right-hand normal points out of
the element.  At a point of a side, the outward unit normal :math:`\mathbf{n}`
follows from the outward normal :math:`\mathbf{n}_0` of the side in the
reference element by the relation of Nanson,

.. math::

   \mathbf{n} \, \mathrm{d}A
   = \det\Bigl(\frac{\partial \mathbf{x}}{\partial \boldsymbol{\xi}}\Bigr)
     \Bigl(\frac{\partial \mathbf{x}}{\partial \boldsymbol{\xi}}\Bigr)^{-T}
     \mathbf{n}_0 \, \mathrm{d}A_0 ,

with :math:`\mathrm{d}A` the area element of the side in the mesh and
:math:`\mathrm{d}A_0` that in the reference element.  On a surface or a curve in
space, the inverse is the pseudo-inverse
:math:`\bigl( (\partial \mathbf{x} / \partial \boldsymbol{\xi})^T
(\partial \mathbf{x} / \partial \boldsymbol{\xi}) \bigr)^{-1}
(\partial \mathbf{x} / \partial \boldsymbol{\xi})^T`, and the relation gives the conormal of an edge, which lies in the surface.

Shape functions
---------------

The tensor elements
~~~~~~~~~~~~~~~~~~~

``Edge2``, ``Edge3``, ``Quad4``, ``Quad9``, ``Hex8`` and ``Hex27`` have their
nodes on a tensor-product grid of one-dimensional coordinates :math:`c_k`: the
set :math:`\{-1, +1\}` for the linear elements and :math:`\{-1, 0, +1\}` for the
quadratic ones, in every direction.  Let :math:`L_k(t)` be the one-dimensional
Lagrange polynomial that is one at :math:`c_k` and zero at the other grid
coordinates.  For the linear grid,

.. math::

   L_{-1}(t) = \tfrac{1}{2}(1 - t) , \qquad
   L_{+1}(t) = \tfrac{1}{2}(1 + t) ,

and for the quadratic grid,

.. math::
   :label: lagrange_quadratic

   L_{-1}(t) = \tfrac{1}{2}\,t\,(t - 1) , \qquad
   L_{0}(t) = 1 - t^{2} , \qquad
   L_{+1}(t) = \tfrac{1}{2}\,t\,(t + 1) .

The shape function of the node at the grid position :math:`(k_1, k_2, k_3)` is
the product of the one-dimensional polynomials of its coordinates,

.. math::

   N_a(\boldsymbol{\xi}) = \prod_{d=1}^{n_\mathrm{d}} L_{k_d}(\xi_d) ,

with :math:`n_\mathrm{d}` the dimension of the element.

The simplices
~~~~~~~~~~~~~

``Tri3``, ``Tri6``, ``Tet4`` and ``Tet10`` are built from the barycentric
coordinates of the reference simplex,

.. math::

   b_0 = 1 - \sum_{d=1}^{n_\mathrm{d}} \xi_d , \qquad
   b_i = \xi_i \quad (i = 1, \dots, n_\mathrm{d}) .

The reference triangle has the vertices :math:`(0,0)`, :math:`(1,0)` and
:math:`(0,1)`, and the reference tetrahedron adds :math:`(0,0,1)`.  The linear
elements take the barycentric coordinates as shape functions,
:math:`N_i = b_i`, and the quadratic ones take

.. math::

   N_i = b_i \, (2 b_i - 1) \quad \text{at vertex } i , \qquad
   N_{ij} = 4 \, b_i \, b_j \quad \text{at the midpoint of edge } (i, j) .

The serendipity elements, the prism and the pyramid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **serendipity** elements drop the interior nodes of the Lagrange quadratic
elements and keep a complete quadratic polynomial [Reddy2019b]_.  With
:math:`(\xi_a, \eta_a, \zeta_a)` the reference position of node :math:`a`, the
shape functions of ``Quad8`` are

.. math::

   N_a = \tfrac14 (1 + \xi \xi_a)(1 + \eta \eta_a)(\xi \xi_a + \eta \eta_a - 1)
   \quad \text{at a corner} , \qquad
   N_a = \tfrac12 (1 - \xi^2)(1 + \eta \eta_a)
   \quad \text{at a midpoint with } \xi_a = 0 ,

and those of ``Hex20`` are

.. math::

   N_a = \tfrac18 (1 + \xi\xi_a)(1 + \eta\eta_a)(1 + \zeta\zeta_a)
         (\xi\xi_a + \eta\eta_a + \zeta\zeta_a - 2)
   \quad \text{at a corner} , \qquad
   N_a = \tfrac14 (1 - \xi^2)(1 + \eta\eta_a)(1 + \zeta\zeta_a)
   \quad \text{at a midpoint with } \xi_a = 0 ,

with the roles of the coordinates exchanged on the edges along :math:`\eta` and
:math:`\zeta`.

The **triangular prism** ``Wedge6`` is the product of a triangle and a segment.
Its reference element is the triangle with the vertices :math:`(0,0)`,
:math:`(0,1)` and :math:`(1,0)`, in this order, extruded over
:math:`-1 \le \zeta \le 1`.  Its shape functions are the barycentric coordinates
of the triangle times :math:`\tfrac12 (1 - \zeta)` on the base and
:math:`\tfrac12 (1 + \zeta)` on the top.

The **pyramid** ``Pyramid5`` joins a quadrilateral face to four triangles.  No
polynomial basis on five nodes is linear on the triangular faces and bilinear on
the base, so the pyramid takes the rational basis of [Bedrosian1992]_,

.. math::

   N_a = \frac{(\zeta - s_a \xi - 1)(\zeta - t_a \eta - 1)}{4 (1 - \zeta)}
   \quad \text{at the base corner with the signs } (s_a, t_a) , \qquad
   N_4 = \zeta \quad \text{at the apex} ,

on the reference pyramid with the base :math:`[-1,1]^2` at :math:`\zeta = 0` and
the apex :math:`(0,0,1)`.  The signs :math:`s_a` and :math:`t_a` are those of the
corner coordinates.  The basis is linear on every edge and triangular face and
bilinear on the base.  The prism and the pyramid therefore conform to
tetrahedra across a triangular face and to hexahedra across a quadrilateral
one, and connect the two in one mesh.  The denominator vanishes only at the
apex, which no quadrature point reaches.

The order of the nodes
~~~~~~~~~~~~~~~~~~~~~~

The reference nodes of every type are in the order of the Visualization Toolkit
(VTK), so a mesh read from or written to a file needs no renumbering.  Corners
come first.

* ``Edge2`` has its nodes at :math:`\xi = -1, +1`, and ``Edge3`` adds the
  midpoint :math:`\xi = 0` last.
* ``Quad4`` has its corners at :math:`(-1,-1)`, :math:`(1,-1)`, :math:`(1,1)`
  and :math:`(-1,1)`.  ``Quad8`` adds the edge midpoints :math:`(0,-1)`,
  :math:`(1,0)`, :math:`(0,1)` and :math:`(-1,0)`, and ``Quad9`` adds the
  centre :math:`(0,0)` after them.
* ``Hex8`` takes the four corners of the face :math:`\zeta = -1`, then the four
  of :math:`\zeta = +1`.  ``Hex20`` adds the twelve edge midpoints: the four of
  the bottom face, the four of the top face, then the four vertical edges.
  ``Hex27`` adds the six face centres in the order :math:`\xi = -1`,
  :math:`\xi = +1`, :math:`\eta = -1`, :math:`\eta = +1`, :math:`\zeta = -1`,
  :math:`\zeta = +1`, and last the centre :math:`(0,0,0)`.
* ``Tri6`` takes its edges as :math:`(0,1)`, :math:`(1,2)` and :math:`(2,0)`,
  so its nodes 3, 4 and 5 sit at :math:`(\tfrac12, 0)`,
  :math:`(\tfrac12, \tfrac12)` and :math:`(0, \tfrac12)`.
* ``Tet10`` takes its edges as :math:`(0,1)`, :math:`(1,2)`, :math:`(0,2)`,
  :math:`(0,3)`, :math:`(1,3)` and :math:`(2,3)`, so its nodes 4 to 9 sit at the
  midpoints of these edges in this order.

The dual mesh inside an element
-------------------------------

The code never meshes the control domains or their interfaces.  It describes
each once, in the reference coordinates of the element, as a tensor-product
patch of dimension zero to three, and the isoparametric map carries it into the
mesh.  The same description serves a straight element, a distorted one and a
quadratic one with curved edges.  The tensor elements have a box construction,
because their nodes lie on a grid, and the simplices and the prism have a median
construction.

Tensor elements: boxes
~~~~~~~~~~~~~~~~~~~~~~

The control domain of a node of a tensor element is the box bounded, in each
direction, by the planes half way to its neighbours on the grid.  At the edge of
the element the box stops at the element boundary.  If the node has the grid
index :math:`k` of :math:`n` in direction :math:`d`, its box spans

.. math::

   \Bigl[ \tfrac{1}{2}(c_{k-1} + c_{k}) , \quad \tfrac{1}{2}(c_{k} + c_{k+1}) \Bigr]
   \quad \text{in } \xi_d ,

with the lower bound replaced by :math:`c_0` for the first index and the upper
bound by :math:`c_{n-1}` for the last.  Two nodes share an interface when their
grid indices differ by one in exactly one direction, and the interface is the
face of their boxes on the plane half way between them.

On the linear elements this is the bisecting construction: the two lines through
the centroid cut a ``Quad4`` into four boxes, and three planes cut a ``Hex8``
into eight.  On the quadratic grid the cuts fall at :math:`\xi = \pm\tfrac12`,
and the nine control domains of a ``Quad9`` take unequal shares of the reference
square.  A corner node owns a box such as :math:`[-1,-\tfrac12]^2`, a sixteenth
of the square, each mid-edge node an eighth, and the centre node the box
:math:`[-\tfrac12,\tfrac12]^2`, a quarter.

Simplices: sub-simplices and median sub-cells
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A simplex has no grid.  The code first writes the element as a set of linear
sub-simplices and gives each vertex of each sub-simplex its median sub-cell.  A
linear element is its own single sub-simplex, and a quadratic one is subdivided.

Inside a linear sub-simplex with the vertices :math:`\mathbf{X}_0, \dots,
\mathbf{X}_{n_\mathrm{d}}` and the centroid :math:`\mathbf{X}_c`, the median
sub-cell of vertex :math:`\mathbf{X}_i` is the region bounded by the vertex, the
midpoints of the edges that meet at it, the centroids of the faces that meet at
it and the centroid of the sub-simplex.  In two dimensions it is the
quadrilateral with the corners

.. math::

   \mathbf{X}_i , \quad \tfrac{1}{2}(\mathbf{X}_i + \mathbf{X}_j) , \quad
   \mathbf{X}_c , \quad \tfrac{1}{2}(\mathbf{X}_i + \mathbf{X}_k) ,

with :math:`j` and :math:`k` the other two vertices.  In three dimensions it is
the hexahedron whose eight corners are the vertex, the three edge midpoints, the
three face centroids and :math:`\mathbf{X}_c`.  The interface between the
sub-cells of :math:`\mathbf{X}_i` and :math:`\mathbf{X}_j` passes through the
midpoint of their common edge, the centroids of the faces that share that edge
and :math:`\mathbf{X}_c`: a segment in two dimensions and a quadrilateral in
three.

``Tri3`` and ``Tet4`` are one sub-simplex each, which gives the classical median
dual.  A quadratic simplex is first split by red refinement, the uniform
subdivision through the mid-edge nodes.  A ``Tri6`` becomes four sub-triangles,
in local node numbers

.. math::

   (0,3,5) , \qquad (3,1,4) , \qquad (5,4,2) , \qquad (3,4,5) ,

three at the corners and one in the centre.  A ``Tet10`` becomes eight
sub-tetrahedra: the four at the corners,

.. math::

   (0,4,6,7) , \qquad (4,1,5,8) , \qquad (6,5,2,9) , \qquad (7,8,9,3) ,

and four that tile the inner octahedron, split along the diagonal that joins
nodes 4 and 9,

.. math::

   (4,9,5,6) , \qquad (4,9,6,7) , \qquad (4,9,7,8) , \qquad (4,9,8,5) .

The control domain of a node is the union of its median sub-cells over every
sub-simplex that touches it.  On a ``Tri6`` of area :math:`A_e`
(:math:`\mathrm{m^2}`), each sub-triangle has the area :math:`A_e/4` and each
median sub-cell :math:`A_e/12`.  A corner node belongs to one sub-triangle and
owns :math:`A_e/12`, and a mid-edge node belongs to three and owns :math:`A_e/4`.
On a ``Tet10`` of volume :math:`V_e` (:math:`\mathrm{m^3}`), each sub-tetrahedron
has the volume :math:`V_e/8` and each median sub-cell :math:`V_e/32`.  A corner node
owns :math:`V_e/32`, nodes 4 and 9 belong to six sub-tetrahedra and own
:math:`3V_e/16` each, and the four other mid-edge nodes own :math:`V_e/8` each.

The choice of the diagonal makes the partition of a ``Tet10`` depend on the
numbering of its nodes, because nodes 4 and 9 receive larger control domains
than the other four mid-edge nodes.  The partition still tiles the element and
its surfaces still close, so the discretization stays consistent.  Two meshes
of the same geometry whose ``Tet10`` elements are numbered differently give
slightly different results.

The prism: a product of duals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A triangular prism is the product of a triangle and a segment, and its median
dual is the product of theirs.  The control domain of a node is the median
sub-cell of its triangle vertex, the quadrilateral bounded by the vertex, the
two edge midpoints and the centroid of the triangle, times the half of the
segment on the node's side of the mid-plane :math:`\zeta = 0`.  Two nodes on the
same end face share the median segment of the triangle times a half segment,
and a node and the node above it share the median quadrilateral in the
mid-plane.  The construction is the general median construction, bounded by edge
midpoints, face centroids and the element centroid.  On a triangular face it
therefore matches the dual of a neighbouring tetrahedron, and on a
quadrilateral face that of a neighbouring hexahedron, so a mesh of hexahedra,
prisms and tetrahedra has a conforming dual.

.. _elements-without-dual:

Elements without a dual
~~~~~~~~~~~~~~~~~~~~~~~

The control domains must partition the element: every point belongs to one
control domain, and every node owns one.  The median dual does not give such a
partition for three element types.

* A serendipity element, ``Quad8`` or ``Hex20``, has mid-edge nodes and no
  interior node.  The region around the element centre belongs to no node.
  Giving it to some of the other nodes would make the control domains depend on
  the numbering and break the symmetry of the partition.
* The apex of ``Pyramid5`` is shared by four edges, where each vertex of a
  tetrahedron, a prism or a hexahedron is shared by three.  The median sub-cell
  of the apex is then no hexahedral patch, and the code has no decomposition of
  it that matches the interfaces of its neighbours.

FEM integrates over the whole element and needs no partition, so it accepts all
fourteen types.

Quadrature
----------

Every kernel names its quadrature rule with its ``quadrature`` parameter.  In a
control volume method the rule is part of the discretization: a lumped source
and a Gauss-integrated source are two discretizations of the same equation.
The rules are one-dimensional rules applied as tensor products on each patch of
a control domain, an interface or an element, except where a symmetric rule for
the whole element is cheaper (below).  The names are matched without regard to
case.

``gauss1`` to ``gauss20``
    Gauss-Legendre with that many points per direction, exact for polynomials
    of degree :math:`2n - 1` with :math:`n` the number of points.  ``gauss``
    alone means ``gauss2``.

``midpoint`` (alias ``centroid``)
    One point at the centre of the patch, the same rule as ``gauss1``.

``trapezoid`` (alias ``trapezoidal``) and ``simpson``
    The corner rule with the points :math:`\{-1, +1\}` and the weights
    :math:`\{1, 1\}`, and Simpson's rule with the points :math:`\{-1, 0, +1\}`
    and the weights :math:`\{\tfrac13, \tfrac43, \tfrac13\}`, per direction.

``nodal`` (alias ``lumped``)
    One point at the owning node.  Its weight is the measure of the node's
    control domain in the element for DMCDM, and
    :math:`\int_e N_a \, \mathrm{d}V` over the element :math:`e` for FEM, which
    is row-sum lumping.  With it, a capacity or mass term is lumped.

``interface``
    One point at the corner of the control domain nearest the element centroid,
    weighted by the measure of the control domain.  In one dimension the point
    is the interface of the control domain, and on a linear element in two and
    three dimensions it is the element centroid.  On a quadratic simplex the
    point is the mean of these corners over the sub-cells, weighted by their
    measures.

``control_domain_trapezoid`` (alias ``cd_trapezoid``)
    The trapezoidal rule over the whole control domain.  For an interior node in
    one dimension it gives
    :math:`\tfrac{\Delta x}{2}\,[f(x_A) + f(x_B)]`, with :math:`f` the
    integrand, :math:`\Delta x` (m) the element length and :math:`x_A` and
    :math:`x_B` the two interfaces of the control domain.  A boundary node puts
    half the weight at its interface and half at the node.  This is the
    classical source rule of the finite volume method.

``automatic`` (alias ``auto``, the default)
    Gauss-Legendre with one more point per direction than the polynomial order
    of the mesh, which integrates a product of two shape functions exactly on
    an affine element: two points per direction on a linear mesh and three on a
    mesh that contains any quadratic element.

The parameter ``reduced_integration`` evaluates the field and its gradient at
the element centroid while the rule that ``quadrature`` names integrates the
term.  This selective reduced integration removes the locking of the transverse
shear terms of beams and plates and of the penalty term of an incompressible
flow [ZienkiewiczTaylorToo1971]_ [HughesCohenHaroun1978]_ [MalkusHughes1978]_.

**Simplices, prisms and pyramids.**  The code integrates a triangle, a
tetrahedron, a prism or a pyramid as a square or a cube with some corners
merged, the collapsed-coordinate map of [Duffy1982]_.  The collapse puts a
factor of degree :math:`n_\mathrm{d} - 1` into the Jacobian of the map, so a
Gauss rule exact to a given degree on the element needs one more point per
direction on the collapsed patch, and the code adds it.  For FEM, which
integrates over the whole element, a cheaper rule exists on triangles,
tetrahedra and prisms: a symmetric rule whose points form orbits of the
symmetry group of the simplex.  For :math:`n` from 1 to 3, ``gauss`` with
:math:`n` points takes the symmetric rule exact to degree :math:`2n - 1`: 1, 6
and 7 points on a triangle, the last two those of [Dunavant1985]_, 1, 8 and 15
points on a tetrahedron, and on a prism the triangle rule times the Gauss rule
along the prism axis.  These rules have positive weights and interior points.

**Interfaces and sides.**  The rules ``nodal``, ``interface`` and
``control_domain_trapezoid`` place their points by reference to a node-centred
control domain, so they apply to volume terms.  On an interface between control
domains they are replaced by ``gauss2``, and on an element side by ``gauss3``.
On a side, ``nodal`` then gathers the points onto the node with their total area
vector, so a constant flux through the side stays exact.

The accuracy on quadratic elements
----------------------------------

On a smooth problem, the nodal values of FEM on quadratic elements are accurate
to fourth order in :math:`h` [DouglasDupont1974]_, two orders above its
convergence in the energy norm, and the derivative of the quadratic interpolant
is most accurate at the Gauss points [Barlow1976]_.  DMCDM on quadratic
elements stays second-order accurate, the order it has on linear elements.  This
section gives the reason and the proof in one dimension.

The position of the interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The flux through an interface uses the gradient of the interpolant there.  For a
diffusion problem it is :math:`-k \nabla u` with :math:`k` the diffusivity and
:math:`\nabla u = \sum_a U_a \nabla N_a`, :math:`U_a` the value at node
:math:`a`.  The truncation error of the method is therefore the error of the
gradient of the interpolant at the interface points.

For a quadratic interpolant of a smooth function, that error is
:math:`O(h^{2})` at a general point and :math:`O(h^{3})` at the two Gauss points
:math:`\xi = \pm 1/\sqrt{3}`, the superconvergent points of the derivative.  The
interfaces of the dual mesh lie half way between the nodes, at
:math:`\xi = \pm \tfrac12`.  There the gradient carries the error
:math:`O(h^{2})`, and the balance keeps a truncation error one order below that
of FEM.  The interfaces cannot move to the Gauss points, because their position
at :math:`\xi = \pm\tfrac12` is what makes the control domains tile the element.

The proof in one dimension
~~~~~~~~~~~~~~~~~~~~~~~~~~

Take an ``Edge3`` element with the nodes at :math:`\xi = -1`, :math:`\xi = +1`
and :math:`\xi = 0` and the values :math:`U_0`, :math:`U_1` and :math:`U_m`
there.  From :eq:`lagrange_quadratic`, the interpolant is

.. math::

   p(\xi) = N_0 U_0 + N_1 U_1 + N_2 U_m ,
   \qquad
   N_0 = \tfrac{1}{2}\xi(\xi - 1) , \quad
   N_1 = \tfrac{1}{2}\xi(\xi + 1) , \quad
   N_2 = 1 - \xi^{2} ,

and its derivative with respect to :math:`\xi` is

.. math::
   :label: edge3_derivative

   p'(\xi) = \bigl(\xi - \tfrac12\bigr) U_0
           + \bigl(\xi + \tfrac12\bigr) U_1
           - 2\xi \, U_m .

At the two interfaces, :eq:`edge3_derivative` gives

.. math::

   p'\bigl(-\tfrac12\bigr) = U_m - U_0 , \qquad
   p'\bigl(+\tfrac12\bigr) = U_1 - U_m .

On an element of length :math:`h` the map gives
:math:`\mathrm{d}u/\mathrm{d}x = (2/h)\,p'(\xi)`, so the gradient at the left
interface is :math:`(U_m - U_0)/(h/2)` and at the right one
:math:`(U_1 - U_m)/(h/2)`.  Each is a two-point difference across half the
element, independent of the node on the far side.

DMCDM on a mesh of ``Edge3`` elements therefore gives the same flux terms as
DMCDM on the mesh of ``Edge2`` elements with the same nodes, twice as many
elements, for a diffusivity that does not depend on :math:`u`.  With the same
quadrature of the source, the two give the same nodal values, and a quadratic element gives DMCDM in one dimension the accuracy of the
linear mesh on its nodes.

The benefit of quadratic elements: geometry
-------------------------------------------

A quadratic element improves the representation of the domain.  On a curved
boundary, a linear element edge is a chord.  A quadratic edge whose mid-side
node lies on the true boundary is the parabola through three points of the arc,
and the isoparametric map carries the control domains and their interfaces onto
that curved region.  The distance between the represented boundary and the true
one falls from :math:`O(h^{2})` to :math:`O(h^{3})`.  The leading term of the
gap of the parabola is odd about the mid-side node, so it integrates to zero
over the element, and the error of the area of the domain falls from
:math:`O(h^{2})` to :math:`O(h^{4})`.  On a cylinder, an annulus or a plate with
a hole, the geometry is often the largest error of a coarse linear mesh.

Promotion and curved domains
----------------------------

:meth:`dualmesh.Mesh.second_order` promotes a linear mesh: ``Edge2`` becomes
``Edge3``, ``Tri3`` becomes ``Tri6``, ``Quad4`` becomes ``Quad9``, ``Tet4``
becomes ``Tet10`` and ``Hex8`` becomes ``Hex27``.  With ``serendipity=True``,
``Quad4`` becomes ``Quad8`` and ``Hex8`` becomes ``Hex20``.  The corner nodes
keep their numbers and positions.  Each added node lies at the image, under the
linear shape functions of the original element, of its position in the
quadratic reference element.  The code identifies it by the set of corner nodes
it interpolates, so elements that share an edge or a face share the node on it
and the promoted mesh conforms.  Side sets carry over, and node sets gain the
added nodes between their members.  Promoting a quadratic mesh returns it
unchanged.  A mesh with ``Wedge6`` or ``Pyramid5`` elements cannot be promoted,
because these types have no quadratic form.

A curved domain must be promoted before its coordinates are transformed.  A
rectangle in :math:`(r, \theta)`, with :math:`r` the radius (m) and
:math:`\theta` the angle, that is mapped to an annulus and only then promoted
has its mid-side nodes at the midpoints of the straight chords, and its edges
stay chords.  When the mesh is promoted first, the mid-side nodes are created
in :math:`(r, \theta)` and the map carries them onto the circle.
:func:`dualmesh.generate_annulus_mesh` takes the two steps in this order, and so must a
transformation of the nodes that you write.

:meth:`dualmesh.Mesh.refined` refuses a quadratic mesh: refine the linear mesh
first and promote the result.  The Jacobian of the code holds one derivative per
element node and variable, up to 96 by default, so a ``Hex27`` mesh carries up
to three variables.  The build setting ``-DDUALMESH_MAX_AD_DERIVATIVES=<n>``
raises the limit.

:doc:`/verification` gives the cases that check the elements, their dual
meshes, the quadrature rules and the convergence on quadratic elements.
