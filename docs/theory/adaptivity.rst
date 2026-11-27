Adaptive mesh refinement
========================

This page states how the code adapts a mesh to the error of a solution: the
error indicator, the marking rules, the refinement and coarsening of each
element type, the treatment of curved boundaries, the transfer of the fields to
the new mesh and the adaptivity of a distributed problem.  To run an adaptive
calculation from the input or from Python, see :doc:`/user_guide/adaptivity`.

.. contents::
   :local:
   :depth: 1

The adaptive loop
-----------------

A discrete method replaces the unknown field :math:`u` by an approximation
:math:`u_h` on a mesh :math:`\mathcal{T}_h` of elements :math:`K`.  The error
:math:`u - u_h` is uneven over the domain.  It is large where the solution
varies on the scale of the element size :math:`h`, near a re-entrant corner, a
crack tip, a boundary layer, a point load or a change of material, and small
where the solution is nearly linear over an element.  Near a re-entrant corner
of interior angle :math:`\Phi > \pi` (rad), the solution of Laplace's equation
holds a term :math:`r^{\pi/\Phi}`, with :math:`r` (m) the distance to the
corner, whose gradient is unbounded.  Under uniform refinement the error of
linear elements in the energy norm then falls as :math:`h^{\pi/\Phi}` instead of
:math:`h`.  A mesh graded towards the corner recovers the rate of a smooth
solution in terms of the number of unknowns [BinevDahmenDeVore2004]_
[Stevenson2007]_.

An adaptive calculation repeats four steps:

1. Solve on the current mesh.
2. Estimate the error, with one indicator :math:`\eta_K \ge 0` per element.
3. Mark the elements to refine and, optionally, the elements to coarsen.
4. Refine and coarsen them, and carry the fields to the new mesh.

Each step is a separate function, so you can replace any one of them:
:meth:`~dualmesh.Problem.error_indicator` estimates,
:func:`~dualmesh.mark_by_fraction`, :func:`~dualmesh.mark_by_error_fraction`
and :func:`~dualmesh.mark_by_threshold` mark, :func:`~dualmesh.refine_marked`
refines, and :func:`~dualmesh.solve_with_adaptive_refinement` runs the loop.

The error indicator
-------------------

The indicator is the gradient-recovery estimator of Zienkiewicz and Zhu
[ZienkiewiczZhu1987]_.  The computed solution :math:`u_h` is continuous across
the faces of the elements, but its gradient jumps from one element to the next.
The gradient of a smooth exact solution has no jumps, so the jumps measure the
error.

The estimator first recovers a continuous gradient by averaging the element
gradients at the nodes.  Each element :math:`K` that holds node :math:`i`
contributes with the weight of its part :math:`K \cap V_i` in the control domain
:math:`V_i` of the node.  The recovered gradient of node :math:`i` is

.. math::
   :label: adaptivity-recovery

   \mathbf{g}^*_i =
   \frac{\displaystyle \sum_{K \ni i} \int_{K \cap V_i} \nabla u_h \,\mathrm{d}V}
        {\displaystyle \sum_{K \ni i} |K \cap V_i|} ,

with :math:`|K \cap V_i|` the measure of that part and the sums over the
elements that hold node :math:`i`.  These parts partition each element, so the
weight is the volume for which the node is responsible.  An element type without
control domains shares each integration point equally among its nodes, which
keeps the weights positive.

The recovered field interpolates the nodal values with the shape functions
:math:`N_i` of the solution,
:math:`\mathbf{g}^*_h = \sum_i N_i\,\mathbf{g}^*_i`.  The indicator of an element
is the :math:`L^2` norm over the element of the difference between the
recovered and the computed gradients,

.. math::
   :label: adaptivity-indicator

   \eta_K = \left( \int_K \left| \mathbf{g}^*_h - \nabla u_h \right|^2 \mathrm{d}V
   \right)^{1/2} ,

with the volume element of the coordinate system, :math:`2\pi r\,\mathrm{d}r\,\mathrm{d}z`
in axisymmetric coordinates.  The estimated error of the mesh is
:math:`\eta = (\sum_K \eta_K^2)^{1/2}`.  For a field with several components,
such as a velocity, the indicators of the components add as squares.  A field
that the elements represent exactly, such as a linear field on linear elements,
has a constant exact gradient that :eq:`adaptivity-recovery` reproduces, so its
indicators vanish and the mesh is left alone.

The indicator gives no guaranteed bound on the error.  Its purpose is to rank
the elements by their share of the error, which is what a marking rule needs.
The argument for it assumes a solution smooth enough for the averaging to cancel
the leading error term, which fails at the singular points that adaptivity
resolves, and the indicator still ranks those elements highest.  The code uses
the averaging recovery of [ZienkiewiczZhu1987]_ and leaves out the superconvergent
patch recovery of [ZienkiewiczZhu1992a]_ and [ZienkiewiczZhu1992b]_.  At a
boundary node fewer elements enter the average, so the recovery is one-sided
there and the indicator is less sharp on the layer of elements at the boundary.

Marking
-------

A marking rule turns the indicators into the set :math:`\mathcal{M}` of elements
to refine.  :math:`N_e` is the number of elements.

- **Fixed fraction.** :func:`~dualmesh.mark_by_fraction` with the fraction
  :math:`f_r \in (0, 1]` (default 0.3) marks the :math:`\lceil f_r N_e \rceil`
  elements of largest indicator.  The mesh grows at a rate that you fix in
  advance, whatever the shape of the error, so the rule suits a fixed budget of
  memory or time.
- **Bulk (Dörfler) marking.** :func:`~dualmesh.mark_by_error_fraction` with
  :math:`f_r \in (0, 1]` (default 0.5) marks the smallest set for which

  .. math::
     :label: adaptivity-dorfler

     \sum_{K \in \mathcal{M}} \eta_K^2 \ge f_r \sum_{K \in \mathcal{T}_h} \eta_K^2 .

  The code sorts the elements by :math:`\eta_K^2` in decreasing order and adds
  them until the fraction is reached.  A concentrated error marks few elements
  and a spread error marks many.  The elements left unmarked carry at most the
  fraction :math:`1 - f_r` of the squared error, which is the property that
  Dörfler's proof of a fixed error reduction per cycle uses [Dorfler1996]_.
  Bulk marking is the default of :func:`~dualmesh.solve_with_adaptive_refinement`
  and the rule of the ``refine_fraction`` of the input.
- **Threshold.** :func:`~dualmesh.mark_by_threshold` marks every element with
  :math:`\eta_K > \eta_{\max}`, for a threshold :math:`\eta_{\max}` in the units
  of the indicator.  Once every indicator is below it, nothing is marked and the
  loop stops.

The fixed-fraction and bulk rules mark at least one element whenever an
indicator is positive, and none when every indicator is zero.

The input also coarsens.  Its ``coarsen_fraction`` :math:`f_c` marks for
coarsening the largest set of refined elements outside :math:`\mathcal{M}` whose
squared indicators sum to at most :math:`f_c` of the total, the smallest
indicators first.  Its ``maximum_refinement_level`` bounds the refinement.  A
level halves the size of an element: :math:`d` bisections of a triangle or a
tetrahedron, with :math:`d` the space dimension, or one subdivision of a
quadrilateral or a hexahedron.  An element marked for both is refined.

Conforming meshes and the dual mesh
-----------------------------------

A node that is a corner of the elements on one side of a face and lies inside
that face for the element on the other side is a hanging node.  The two methods
need different things from a mesh with hanging nodes.

The finite element method (FEM) needs the discrete functions to be continuous
across element faces.  A hanging node :math:`m` on the edge from node :math:`a`
to node :math:`b` breaks continuity, because the small elements interpolate its
own value while the large element interpolates between :math:`u_a` and
:math:`u_b`.  The constraint :math:`u_m = \tfrac{1}{2}(u_a + u_b)` restores
continuity along the whole edge.

The dual mesh control domain method (DMCDM) integrates the balance law over the
control domain :math:`V_i` of each node.  With the flux :math:`\mathbf{F}` and
the source :math:`s` of a steady balance law :math:`\nabla\cdot\mathbf{F} = s`,
the equation of node :math:`i` is

.. math::
   :label: adaptivity-balance

   \oint_{\partial V_i} \mathbf{F}\cdot\mathbf{n}\,\mathrm{d}S
   = \int_{V_i} s\,\mathrm{d}V ,

with :math:`\mathbf{n}` the outward unit normal of :math:`\partial V_i`.  Every
interior face of the dual mesh bounds two control domains, with opposite
normals, so the sum of :eq:`adaptivity-balance` over the nodes leaves the global
balance

.. math::

   \oint_{\partial\Omega} \mathbf{F}\cdot\mathbf{n}\,\mathrm{d}S
   = \int_{\Omega} s\,\mathrm{d}V ,

which the discrete solution satisfies on any mesh.  The cancellation needs the
control domains to cover the domain :math:`\Omega` without overlap.  A naive
treatment of a hanging node breaks this partition: the node gets a control
domain from the small elements but none from the large one, and a strip along
the face belongs to no single pair of neighbouring control domains.  The flux
through that strip then cancels in no pair, and the global balance fails.  A
constraint on values says nothing about which control domain the material next
to the face belongs to, so DMCDM needs a geometric treatment.

Triangles and tetrahedra are refined by bisection, which leaves no hanging node.
Quadrilaterals and hexahedra are refined by subdivision, which leaves hanging
nodes that FEM constrains and DMCDM gives control domains of their own.

Newest-vertex bisection of triangles
------------------------------------

To bisect a triangle is to insert a node at the midpoint of one of its edges
and split the triangle along the segment from that midpoint to the opposite
vertex.  Each triangle :math:`[a, b, c]` lists its refinement edge
:math:`a`-:math:`b` first.  Its bisection inserts the midpoint :math:`m` of that
edge and gives the children :math:`[c, a, m]` and :math:`[b, c, m]`.  The new
node :math:`m` is the newest vertex of both children, and each child's
refinement edge is the edge opposite it.  A triangle of the initial mesh takes
its longest edge as its refinement edge, with ties to the larger node numbers.

A refinement follows the edge marking of [FunkenPraetoriusWissgott2011]_:

1. Each marked triangle marks its refinement edge.
2. Every triangle with a marked edge marks its refinement edge, and this step
   repeats until no new edge is marked.
3. Each triangle splits by the pattern of its marked edges.  It bisects its
   refinement edge, and each child bisects its own refinement edge again when
   that edge is marked.

Every marked edge then has one midpoint, shared by the triangles on both of its
sides, so no node hangs.  Step 2 marks a finite set of edges, so the refinement
ends.  The children of a triangle keep its orientation and its block.
Newest-vertex bisection is the refinement for which adaptive finite element
methods converge at the optimal rate [BinevDahmenDeVore2004]_ [Stevenson2007]_.
However often it repeats, the triangles it makes fall into finitely many classes
of similar triangles, so their smallest angle stays bounded below.

The hierarchy of bisections is kept, so a refinement can be undone.  A node made
by bisection is removed when it is the newest vertex of every triangle around
it, there are two of them (on the boundary) or four (inside), and all of them are
marked for coarsening.  Each pair :math:`[c, a, m]`, :math:`[b, c, m]` then
merges back into :math:`[a, b, c]`.

Quadratic triangles and tetrahedra (``Tri6``, ``Tet10``) are bisected by their
corners.  The middle node of an edge is the node that the bisection of that edge
makes, so it becomes a corner of the children.

Longest-edge bisection of tetrahedra
------------------------------------

A tetrahedral mesh is refined by longest-edge bisection [RivaraLevin1992]_.  A
marked tetrahedron bisects its longest edge, and every tetrahedron around that
edge is bisected through the same midpoint, so no node hangs.  Each tetrahedron
of that ring splits into two: one with the second end of the edge replaced by
the midpoint, and one with the first end replaced, which keeps its orientation.
A tetrahedron of the ring whose longest edge is another is bisected first,
across its own longest edge.  Each such detour bisects a strictly longer edge,
so the recursion ends for any initial mesh.  Ties of length go to the larger
node numbers, so that every tetrahedron agrees on which edge is the longest.

Coarsening undoes one bisection.  A node made by bisection is removed when every
tetrahedron around it was made by that bisection and all of them are marked.
The two children of each parent share the face through the node and merge back
into the parent.

Subdivision of quadrilaterals and hexahedra
-------------------------------------------

A quadrilateral or a hexahedron splits into :math:`2^d` children of half its
size, through the middles of its edges, the centres of its faces and its
centre.  A neighbour that is not split keeps its side whole, and the node in the
middle of that side hangs.  The mesh is kept 2:1 balanced across edges and
faces: a cell is split when a neighbour across one of its edges or faces is more
than one level finer, so a node hangs on one level only.  Coarsening merges the
:math:`2^d` children of a parent back into it when all of them are marked and
the parent keeps the balance.  A new node is placed by the bilinear (trilinear)
map of the corners of its parent as they are.

The two methods treat a hanging node differently.

FEM constrains it.  The value at a hanging node is the mean of the two ends of
its edge, or of the four corners of its face.  Each hanging unknown is therefore
a weighted sum of independent ones, and its equation is added to theirs with the
same weights.  One map holds every such dependent unknown: a hanging node with
its weights, a periodic copy with the weight one on its primary, and the value of
a first-order variable at a node of a quadratic element that is not a corner,
the linear interpolant of the corners.  Chains resolve to independent unknowns,
so a node that hangs on a periodic boundary makes every copy of it hang alike.
The residual :math:`\mathbf{R}` becomes :math:`\mathbf{C}^{\mathsf{T}}\mathbf{R}`
and the Jacobian :math:`\mathbf{J}` becomes
:math:`\mathbf{C}^{\mathsf{T}}\mathbf{J}\,\mathbf{C}`, with :math:`\mathbf{C}`
the matrix that maps the independent unknowns to all of them.  The adjoint of
the sensitivities solves with the transpose of the same matrix.  A distributed
run joins periodic copies in its gather-scatter, and the process of one copy
need not hold the side that another copy hangs on, so a distributed run under
FEM refuses a node that hangs on a periodic boundary.

DMCDM keeps the hanging node as an unknown of its own, with a control domain of
its own.  The interfaces of the large cell would end at the middle of its side,
and those of the small cells at the middles of their halves, so the segment
between them would carry no flux.  The code therefore splits the large cell as a
polygon whose sides are the leaf edges: the two halves of a side on which a node
hangs, and every other side whole.  Each flag, the triangle of the centre of the cell, the middle of a
leaf edge and an end of that edge, belongs to the control domain of that end,
and the segment from the centre to the middle of a leaf edge is the interface
between its ends.  These interfaces meet those of the small cells at the same
middles, so the control domains tile the domain and the balance stays
conservative.  A hexahedron splits the same way over its leaf faces: four for a
side whose centre hangs and one for another, with the hanging middles of its
edges among its corners.  Each flag, the tetrahedron of the centre of the cell,
the centre of a leaf face, the middle of an edge of that face and an end of the
edge, belongs to the control domain of that end, and the triangle of its first
three points is the interface between the ends.  The fields inside the large
cell come from its own corners, so a linear field stays exact.  The cell is
integrated as a conforming cell is, with the same quadrature rules on patches of
its reference element.  In 2D the two flags of a node make one quadrilateral, the
shape of a sub-cell of a conforming quadrilateral.  In 3D the two flags of a node
on a leaf face make a pyramid, a cube collapsed onto the centre of the cell, and
each interface triangle is a square collapsed onto the centre.  A collapsed patch
takes one Gauss point more per direction, as the sub-cells of a simplex do.

Curved boundaries
-----------------

A node that a refinement adds on a boundary lies on the straight edge between
its neighbours.  To follow a curved boundary, the ``boundaries`` argument of
:func:`~dualmesh.solve_with_adaptive_refinement` maps a side set to a projection
of its curve.  After each refinement, the nodes of the side set move onto the
curve, and the later splits start from the moved nodes.  A hanging node on such
a side stays on the straight side of its large neighbour, where its constraint
puts it, and moves onto the curve once that side is split.  On a quadratic mesh
of triangles or tetrahedra, every new node is placed by the quadratic map of the
element of the initial mesh it lies in, so the new nodes of a curved side lie on
its quadratic curve.

Transfer of the fields
----------------------

Every element records the element of the initial mesh it comes from, which
carries a field stored per element to the new mesh.  Side sets and node sets
follow the refinement: the halves of a bisected edge belong to the side sets of
the edge and give them back when they merge, and a midpoint whose two ends
belong to a node set joins that set.  After subdivision, a new node joins a node
set on the boundary only where it lies on the boundary too.

Under FEM, a nodal field keeps its values at the nodes that stay.  A new node
takes the value of the interpolant of the old element it lies in, found in the
reference coordinates of its initial element.  A field that the elements
represent, linear or quadratic, therefore transfers exactly under refinement.

Under DMCDM, a change of mesh keeps the content of each field.  Let
:math:`u_i` be the value of a field at node :math:`i`, in the unit of the field.
Let :math:`M_i` be the measure of the control domain of the node, an area in
:math:`\mathrm{m^2}` in two dimensions and a volume in :math:`\mathrm{m^3}` in
three, with the coordinate factor of the problem integrated over it, such as
:math:`2 \pi r` in axisymmetric coordinates with :math:`r` the radius in m.  The
content of the field is :math:`\sum_i M_i u_i`.  A coarsening removes the middle
of a bisected edge, or the middles of the edges, the centres of the faces and
the centre of a cell whose children merge.  Let :math:`P_{ij}`, a dimensionless
weight, be the value at old node :math:`i` of the shape function of node
:math:`j` of the coarsened mesh.  It is 1 for :math:`i = j`, :math:`1/2` at each
end of the edge of a removed edge middle, :math:`1/4` at each corner of the face
of a removed face centre, and :math:`1/2^d` at each corner of the cell of a
removed cell centre, with :math:`d` the dimension.  Each node :math:`j` of the
coarsened mesh takes the weighted mean over the old nodes :math:`i`

.. math::
   :label: adaptivity-restriction

   u_j = \frac{\sum_i P_{ij} M_i u_i}{\sum_i P_{ij} M_i}.

The weights are positive, so the mean lies between the old values, and a
constant field stays the same constant.  Every old node has
:math:`\sum_j P_{ij} = 1`, so the content stays the same wherever the measure of
each new control domain is the sum :math:`\sum_i P_{ij} M_i` of the pieces it
receives.  In Cartesian coordinates, the control domain of a node takes a third
of each linear triangle around it, and a quarter of each linear tetrahedron, as
much as the integral of the shape function of the node.  The coarsening of such
a mesh meets the condition exactly.  Finite-volume codes with adaptive meshes,
such as AMReX and Chombo, restrict a field to a coarse cell by a volume-weighted
mean of the same kind.  A refinement in the same change then interpolates the
coarsened field, which keeps it as the same function.  The shape functions of a
quadratic element are negative at some of the removed nodes, which would give
the mean :eq:`adaptivity-restriction` negative weights.  DMCDM therefore carries
values this way on linear elements only, and interpolates them on ``Tri6`` and
``Tet10``, as FEM does.

A steady adaptive loop starts each solve from the solution of the last mesh,
carried to the new one.  A transient adaptive run, of the input or of
:func:`~dualmesh.solve_transient_with_adaptive_refinement`, adapts the mesh every
``interval`` accepted time steps and goes on from the time it reached, with the
time step that the step control chose before the change.  At each change, the
fields are carried to the new mesh, and so is the velocity of a second-order
equation, such as the balance of momentum of a solid with a density.  Under
refinement alone both stay the same functions, so the momentum
:math:`\int_\Omega \rho\,\mathbf{v}\,\mathrm{d}V` is unchanged, with :math:`\rho`
the density and :math:`\mathbf{v}` the velocity.  Under DMCDM, the coarsening of
linear triangles or tetrahedra in Cartesian coordinates keeps the content
:math:`\sum_i M_i u_i` of each of them as well.

Element types
-------------

Local refinement and coarsening take a mesh of one element type: ``Tri3``,
``Tri6``, ``Tet4`` and ``Tet10`` by bisection, and ``Quad4`` and ``Hex8`` by
subdivision.  :func:`~dualmesh.refine_marked` and the adaptive loops refuse any
other mesh, with the reason.

:meth:`~dualmesh.Mesh.refined` refines a mesh uniformly.  It splits an ``Edge2``
into two, a ``Tri3`` or a ``Quad4`` into four, a ``Tet4`` or a ``Hex8`` into
eight, a ``Wedge6`` into eight prisms and a ``Pyramid5`` into six pyramids and
four tetrahedra.  Every element is split, so no node hangs.  Quadratic meshes
are refused: refine the linear mesh and promote the result with
:meth:`~dualmesh.Mesh.second_order`.  A mesh of prisms or pyramids is therefore
refined uniformly, and each round multiplies the number of elements by eight.

Adaptivity of a distributed problem
-----------------------------------

In a distributed problem every process holds the same hierarchy of the whole
mesh.  The recovery :eq:`adaptivity-recovery` sums the contributions of every
process at the nodes they share, so the indicators are those of the whole mesh.
Every process gathers them, marks the same elements and adapts its copy of the
hierarchy the same way.  The problem on the new mesh is built and partitioned
again (see :doc:`parallel`).  The distribution keeps the hanging nodes whole:
the process that holds a hanging node also holds the nodes it depends on, and the
process that holds a large cell also holds the nodes that hang on its sides.

The verification cases of adaptive refinement, with their sources, computed
values and errors, are on the :doc:`/verification` page.
