Adaptive mesh refinement
========================

.. contents::
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

Why adapt at all
----------------

A discrete method replaces the unknown field :math:`u` by an approximation
:math:`u_h` drawn from a finite-dimensional space built on a mesh
:math:`\mathcal{T}_h` of elements :math:`K`.  The error :math:`u - u_h` is not
spread evenly over the domain: it is large where the solution varies on a length
scale comparable with the element size -- near a re-entrant corner, a crack tip,
a boundary layer, a point load, an abrupt change of material -- and small
wherever the solution is nearly linear over an element.  Halving the element
size everywhere multiplies the number of unknowns by four in two dimensions and
by eight in three, and most of those new unknowns are spent on parts of the
solution that were already resolved, so placing elements where the error is buys
the same accuracy far more cheaply.  The benchmark at the end of this chapter
makes the difference concrete: a uniform sequence of meshes reaches a maximum
nodal error of :math:`8.30 \times 10^{-3}` using 3201 unknowns, while an
adaptive sequence reaches :math:`5.16 \times 10^{-3}` using 122.

An adaptive calculation is a loop over three steps.  It **solves** on the
current mesh; it **estimates**, producing one number :math:`\eta_K \ge 0` per
element, the *error indicator*, meant to be large on the elements carrying most
of the error; and it **marks** a subset of elements and **refines** them, after
which the loop repeats on the new mesh.  The three steps are separate functions
in ``dualmesh``, so a calculation can replace any one of them:
:meth:`~dualmesh.Problem.error_indicator` estimates,
:func:`~dualmesh.mark_by_fraction` and its relatives mark,
:func:`~dualmesh.refine_marked` refines, and
:func:`~dualmesh.solve_with_adaptive_refinement` runs the whole loop.

The error indicator
-------------------

The indicator implemented here is the gradient-recovery estimator of Zienkiewicz
and Zhu [ZienkiewiczZhu1987]_.  Its starting point is a property of the computed
solution that is at first sight a defect.  With linear elements :math:`u_h` is
continuous across element faces but its gradient is not: on each element
:math:`\nabla u_h` is a constant (for a triangle) or a low-order polynomial, and
it jumps from one element to the next.  The exact gradient of a smooth solution
has no such jumps, so the discontinuity is a visible symptom of the
discretisation error.

The estimator turns this into a number in two steps.  First a *recovered*
gradient :math:`\mathbf{G}_h` is built: a continuous field obtained by averaging
the element gradients onto the nodes and interpolating back.  For a smooth
solution the averaging cancels the leading error term of the element gradients,
so :math:`\mathbf{G}_h` approximates :math:`\nabla u` better than
:math:`\nabla u_h` does.  Second, the difference between the two is measured
element by element.  Because :math:`\mathbf{G}_h` is the better of the two,
:math:`\| \mathbf{G}_h - \nabla u_h \|_{L^2(K)} \approx
\| \nabla u - \nabla u_h \|_{L^2(K)}`, so the computable left-hand side
estimates the uncomputable right-hand side.

What distinguishes one recovery from another is the weight with which each
element contributes to the nodal average.  ``dualmesh`` weights by control
domain measure, the natural choice here because the control domains are already
part of the discretisation.  Let :math:`\Omega_a` be the control domain of node
:math:`a`, the region of the dual mesh over which the balance law for node
:math:`a` is enforced.  The dual mesh subdivides every element :math:`K` among
the nodes of :math:`K`, so write :math:`K_a = K \cap \Omega_a` for the part of
:math:`K` belonging to node :math:`a` and :math:`|K_a| = \int_{K_a}\mathrm{d}V`
for its measure; these pieces tile the element,
:math:`\sum_{a \in K} |K_a| = |K|`.  The recovered nodal gradient is

.. math::
   :label: recovery

   \mathbf{G}_a \;=\;
   \frac{\displaystyle \sum_{K \ni a} \int_{K_a} \nabla u_h \, \mathrm{d}V}
        {\displaystyle \sum_{K \ni a} |K_a|} ,

the sums running over the elements that have :math:`a` as a node.  Writing
:math:`\mathbf{g}_{K,a} = |K_a|^{-1} \int_{K_a} \nabla u_h \,\mathrm{d}V` for
the mean gradient of :math:`K` over that node's share of it, :eq:`recovery`
reads :math:`\mathbf{G}_a = \sum_K |K_a| \mathbf{g}_{K,a} / \sum_K |K_a|`: the
nodal gradient is the average of the element gradients at that node, weighted by
the measure of the part of each element belonging to that node's control domain.
In the code the integrals use a three-point rule on the dual point set of each
element; every integration point carries the index of the node that owns it, and
:math:`|K_a|` is accumulated as the sum of the weights of the points owned by
:math:`a`.

The recovered field is interpolated with the same shape functions :math:`N_a` as
the solution, :math:`\mathbf{G}_h(\mathbf{x}) = \sum_a N_a(\mathbf{x})
\mathbf{G}_a`, and the indicator of an element is the :math:`L^2` norm over that
element of the difference between the recovered and the computed gradient:

.. math::
   :label: indicator

   \eta_K \;=\;
   \left( \int_K \bigl| \mathbf{G}_h - \nabla u_h \bigr|^2 \, \mathrm{d}V
   \right)^{1/2} .

This integral again uses a three-point rule, on the ordinary element point set,
and carries the coordinate factor of the coordinate system, so in an
axisymmetric problem the volume element is
:math:`2\pi r \,\mathrm{d}r\,\mathrm{d}z`.  The global quantity
:math:`\eta = ( \sum_K \eta_K^2 )^{1/2}` is the norm of the same difference over
the whole domain.  For a solution the element space represents exactly -- a
linear field on linear elements -- the computed gradient is already constant and
exact, the recovery :eq:`recovery` reproduces it, and :math:`\eta_K` vanishes;
the test suite measures an indicator below :math:`10^{-12}` in that case, the
behaviour an estimator must have if it is not to refine a perfect mesh.

What the indicator does not do
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*It is an indicator, not a bound.*  Expression :eq:`indicator` comes with no
proof that :math:`c_1 \eta \le \| \nabla u - \nabla u_h \| \le c_2 \eta` for
known constants, and the library makes no such claim.  What it is good for is
*ranking*: it says which elements carry more error than others, which is all a
marking rule needs.  It must not be used to certify that a solution is within a
stated tolerance.  The argument behind it also assumes the solution is smooth
enough for the averaging to cancel the leading error term, and that assumption
fails precisely at the singular points an adaptive run is aimed at -- though in
practice the indicator still ranks those elements highest, which is what matters.

*It uses simple averaging recovery, not superconvergent patch recovery.*
Zienkiewicz and Zhu later replaced the nodal average by a least-squares
polynomial fitted over a patch of elements around each node, sampled at the
superconvergent points of those elements [ZienkiewiczZhu1992a]_
[ZienkiewiczZhu1992b]_.  That construction recovers a gradient of higher order
and gives a sharper estimator.  ``dualmesh`` implements only the earlier,
simpler averaging recovery of [ZienkiewiczZhu1987]_.  One consequence is that
the recovery is one-sided at boundary nodes, where fewer elements contribute to
the average, so the indicator is less reliable on the layer of elements touching
the boundary.

*It is not available for the cell-centred finite volume method.*  Calling
:meth:`~dualmesh.Problem.error_indicator` on a problem built with
``method="zfvm"`` raises an error rather than returning a misleading number.
The reason is structural: the unknowns of that method are cell averages, not
nodal values, so there is no nodal field to differentiate and no element
gradient of the kind :eq:`recovery` averages.  Recovering a nodal gradient from
cell averages needs a different construction -- a least-squares or Green-Gauss
reconstruction over the cells around each node -- which the library does not
provide.  The dual mesh control domain method, the finite element method and the
vertex-centred finite volume method all carry nodal unknowns and all support the
indicator.

Marking
-------

A marking rule turns the indicators :math:`\{\eta_K\}` into a set
:math:`\mathcal{M}` of elements to refine.  The three rules differ in how the
size of :math:`\mathcal{M}` responds to the shape of the error distribution.

**Fixed fraction.**  :func:`~dualmesh.mark_by_fraction` with
parameter :math:`\theta \in (0,1]` marks the :math:`\lceil \theta N \rceil`
elements with the largest indicators, :math:`N` being the element count.  The
size of :math:`\mathcal{M}` depends only on :math:`\theta`, never on the
indicators, so the mesh grows geometrically at a rate the user fixes in advance.
Choose it when the budget matters more than optimality -- when a calculation
must fit in a known amount of memory, or a fixed number of cycles must land near
a target size.  Its weakness is that it refines :math:`\theta N` elements
whether or not they hold a meaningful share of the error: on a problem whose
error sits in three elements it refines far more than three, and on one whose
error is spread evenly it refines far fewer than it should.

**Bulk, or Dörfler, marking.**
:func:`~dualmesh.mark_by_error_fraction` with parameter
:math:`\theta \in (0,1]` marks the smallest set :math:`\mathcal{M}` for which

.. math::
   :label: dorfler

   \sum_{K \in \mathcal{M}} \eta_K^2 \;\ge\; \theta \sum_{K \in \mathcal{T}_h} \eta_K^2 .

The code sorts the elements by :math:`\eta_K^2` in decreasing order, accumulates
the sum, and stops as soon as the fraction :math:`\theta` is reached.  The size
of :math:`\mathcal{M}` is now set by the error distribution: where the error is
concentrated a handful of elements meets :eq:`dorfler` and the mesh grows
slowly, and where it is spread out most elements are marked and the cycle
behaves almost like uniform refinement.  That adaptive response is why
:eq:`dorfler` is the rule convergence proofs are built on.  Dörfler
[Dorfler1996]_ showed for Poisson's equation that marking this way guarantees a
fixed reduction of the error at every cycle, so the loop converges at a definite
rate rather than merely refining forever.  The proof needs this rule and not the
fixed fraction because :eq:`dorfler` bounds *the error left unmarked*: the
elements outside :math:`\mathcal{M}` carry at most a fraction
:math:`1 - \theta` of the total, so refining :math:`\mathcal{M}` necessarily
attacks the bulk of the error, whereas the :math:`\theta N` largest indicators
of a fixed-fraction rule may together account for an arbitrarily small share.
Bulk marking is the default of
:func:`~dualmesh.solve_with_adaptive_refinement`, with
:math:`\theta = 0.5`.

**Absolute threshold.**  :func:`~dualmesh.mark_by_threshold` marks
every element with :math:`\eta_K > \tau` for a fixed :math:`\tau`.  Unlike the
other two it has a fixed point: once every element is below the threshold
nothing is marked, the driver loop stops, and the mesh stops growing.  It is the
right rule when the user knows what a tolerable local error looks like in the
units of the problem, and the wrong rule otherwise, because a threshold set too
low refines everything and one set too high refines nothing.  A common use is to
run a few cycles of bulk marking first and set :math:`\tau` from the indicators
that result.

Both :func:`~dualmesh.mark_by_fraction` and
:func:`~dualmesh.mark_by_error_fraction` mark at least one element
whenever the indicators are not all zero, so a loop driven by either never
stalls on a mesh that still has error in it.

Why the refinement must be conforming
-------------------------------------

Suppose one element of a quadrilateral mesh is split into four and its
neighbours are left alone.  The midpoint of the shared face is now a node of the
four children but not of the neighbour: it lies in the *interior* of one of the
neighbour's faces.  Such a node is a *hanging node*, and a mesh that has one is
*non-conforming*.  Refinement schemes that subdivide only the marked elements
produce hanging nodes as a matter of course, and they are the normal state of
affairs in adaptive finite element codes.

A conforming finite element method needs one thing from its mesh: that the
functions in the discrete space be continuous across element faces, so the space
is a subspace of :math:`H^1(\Omega)` and the weak form is meaningful.  A hanging
node :math:`h` in the interior of the face joining nodes :math:`p` and :math:`q`
breaks continuity, because the children interpolate towards the independent
value :math:`u_h` while the neighbour interpolates linearly between :math:`u_p`
and :math:`u_q`.  The remedy is to remove the offending freedom by
*constraining* it, :math:`u_h = \frac{1}{2}(u_p + u_q)`, and eliminating the
constrained degree of freedom from the system.  With that one line the two sides
agree along the whole face, the interpolant is continuous again, and the method
proceeds unchanged.  This is why finite element frameworks such as MOOSE
[MOOSE2025]_ and libMesh [libMesh2006]_ can refine purely locally.

The dual mesh control domain method does not derive its equations from a weak
form and does not rest on the continuity of an interpolant.  It integrates the
balance law over the control domain of each node.  Starting from the canonical
form :math:`-\nabla \cdot \mathbf{F} + S = 0`, integrating over
:math:`\Omega_a` and applying the divergence theorem gives one equation per
node,

.. math::
   :label: balance

   \oint_{\partial \Omega_a} \mathbf{F} \cdot \mathbf{n} \, \mathrm{d}S
   \;=\; \int_{\Omega_a} S \, \mathrm{d}V ,

with :math:`\mathbf{n}` the outward unit normal of :math:`\partial \Omega_a`.
The method's headline property follows from summing :eq:`balance` over all
nodes.  Every interior face of the dual mesh is shared by exactly two control
domains and appears in the two corresponding equations with opposite normals, so
those two contributions cancel identically.  Everything cancels except the terms
on the outer boundary, leaving

.. math::

   \oint_{\partial \Omega} \mathbf{F} \cdot \mathbf{n} \, \mathrm{d}S
   \;=\; \int_{\Omega} S \, \mathrm{d}V ,

the global balance, satisfied exactly by the discrete solution on any mesh,
however coarse.  That is the property the method exists to guarantee.

The cancellation requires two things of the dual mesh, and they are geometric
rather than functional.  The control domains must **cover** the domain,
:math:`\bigcup_a \Omega_a = \Omega`, and they must **not overlap**, so every
point of :math:`\Omega` belongs to exactly one of them.  The dual mesh is not an
interpolation device; it is a partition of space.

A hanging node destroys the partition.  Let :math:`h` hang in the interior of a
face :math:`f` shared between the refined side and the unrefined element
:math:`K^-`.  On the refined side :math:`h` is a vertex, so it is assigned a
control domain :math:`\Omega_h` assembled from pieces of the children meeting
there.  On the :math:`K^-` side :math:`h` is not a vertex at all, so it gets no
share of :math:`K^-`; the material of :math:`K^-` next to :math:`f` is divided
between the control domains of the *end nodes* of :math:`f`.  The result is a
strip along :math:`f` where :math:`\partial \Omega_h` is not matched by the
boundary of any single neighbouring control domain.  Flux computed as leaving
:math:`\Omega_h` through that strip is not credited as entering anything, and on
the other side the control domains of the end nodes of :math:`f` receive flux
through a piece of boundary that is not part of :math:`\partial \Omega_h`.  The
contributions no longer cancel in pairs, the telescoping sum leaves a spurious
interior residual, and the discrete global balance fails.  The method loses
precisely the property that motivates it, and it does so silently: the solve
still returns numbers, and they are simply not conservative.

Constraining :math:`u_h = \frac{1}{2}(u_p + u_q)` does not repair this.  The
constraint is a statement about *values*; the defect is a statement about
*regions*.  Making the hanging value a combination of its neighbours' values
says nothing about which control domain the material next to the face belongs
to.  There is no analogue of the finite element fix, so ``dualmesh`` refines
conformingly: the refinement is not allowed to leave a hanging node anywhere.
The test suite checks this through a consequence.  After four rounds of
refinement concentrated in one corner of a square -- the pattern that would
produce hanging nodes if the propagation were incomplete -- a linear field is
reproduced to :math:`10^{-11}` by all three nodal discretisations, a patch test
a mesh with a hanging node would fail because its control domains would no
longer tile the domain.

Longest-edge bisection
----------------------

Conformity is maintained by Rivara's longest-edge bisection [Rivara1984]_
[Rivara1991]_.  To bisect a triangle is to insert a node at the midpoint of one
of its edges and split it into two triangles along the segment joining that
midpoint to the opposite vertex.  If the bisected edge is interior, the triangle
on the other side must be bisected across the *same* edge using the *same* new
node, or that node hangs.

The rule that makes this terminate is to bisect only the longest edge.  The
library keeps a stack of triangles that still have to be bisected, initialised
with the marked ones, and repeats the following until the stack is empty.  Take
the triangle :math:`t` on top of the stack, discarding it if an earlier step has
already bisected it, and find its longest edge :math:`e`.  Ties are broken by
comparing the sorted pair of node indices, so two triangles sharing an edge
always agree on whether it is the longest of either; without a deterministic
tie-break, two triangles of a mesh with equal edges can send each other back and
forth forever.  Then look for the neighbour :math:`t'` across :math:`e`.

- If there is no neighbour -- :math:`e` lies on the exterior boundary -- simply
  bisect :math:`t` and remove it from the stack.
- If :math:`e` is also the longest edge of :math:`t'`, both may be bisected
  across :math:`e` at once: create the midpoint (or reuse it, if an earlier step
  already created it), split :math:`t`, split :math:`t'` across the same node,
  and remove :math:`t` from the stack.
- If :math:`e` is *not* the longest edge of :math:`t'`, then :math:`t` cannot be
  bisected yet: push :math:`t'` on the stack, leaving :math:`t` beneath it, and
  start again, so the neighbour is refined first.

The children inherit their parent's block, and the refined mesh records for
every element the index of the element of the original mesh it came from; that
map is what a field stored per element is transferred with.  Side sets and node
sets are carried across: a midpoint whose two end nodes both belong to a set
joins that set, and a child side joins a side set when it lies on the exterior
and all of its nodes belong to that set.

Each detour -- each time a neighbour is pushed instead of the triangle being
bisected -- moves to a strictly longer edge, since :math:`t'` must have an edge
strictly longer than :math:`e` for the detour to happen at all.  A chain of
detours therefore visits edges of strictly increasing length; since a mesh has
finitely many edges and each can occur at most once in such a chain, the chain
has bounded length and the algorithm terminates.  This is Rivara's argument
[Rivara1984]_.  The code nonetheless caps the number of steps at
:math:`1000\,(N + 1)` and raises an error if the cap is hit: the theorem applies
to a valid triangulation, and a degenerate or duplicated element could otherwise
produce a program that never returns.

Repeated subdivision could in principle produce slivers, and a mesh of slivers
gives an ill-conditioned system and a poor approximation.  Longest-edge
bisection does not.  Rivara proved that the smallest angle of any mesh obtained
by repeated longest-edge bisection of an initial mesh :math:`\mathcal{T}_0`
satisfies

.. math::

   \min_{K \in \mathcal{T}_h} \alpha_K \;\ge\;
   \tfrac{1}{2} \min_{K \in \mathcal{T}_0} \alpha_K ,

where :math:`\alpha_K` is the smallest angle of element :math:`K`
[Rivara1984]_.  The bound is uniform in the number of refinement rounds: the
shapes cannot deteriorate without limit however many cycles are run, and the
quality of the adaptive mesh is settled by the quality of the mesh the user
started from.  The test suite exercises this over twelve rounds of random
marking and confirms that every triangle stays positively oriented and that the
ratio of smallest to largest element area never collapses.

What is refined, and what is not
--------------------------------

**Local refinement is available for triangular meshes only.**
:func:`~dualmesh.refine_marked` accepts a mesh of ``Tri3`` elements
and raises an error, naming the reason, for any other element type.  This is a
real restriction and not an oversight of the interface: longest-edge bisection
is a triangle algorithm.  Its termination argument and its angle bound both rest
on a triangle having one longest edge whose bisection produces two triangles,
and there is no equally simple statement for a quadrilateral, a tetrahedron or a
hexahedron.  Conforming local refinement of those types needs a different
algorithm, which the library does not implement.

Uniform refinement is available for all of them.
:meth:`~dualmesh.Mesh.refined` splits every element at once: an ``Edge2`` into
two, a ``Tri3`` into four, a ``Quad4`` into four, a ``Tet4`` into eight, a
``Hex8`` into eight, a ``Wedge6`` into eight prisms and a ``Pyramid5`` into six
pyramids and four tetrahedra.  Because every
element is subdivided, no hanging node can arise and the result is conforming.
Quadratic meshes are refused; refine the linear mesh first and promote the
result with :meth:`~dualmesh.Mesh.second_order`.

A user with a quadrilateral, tetrahedral or hexahedral mesh therefore has two
options.  The first is to refine uniformly with :meth:`~dualmesh.Mesh.refined`, which is
the right answer when the solution has no localised feature but multiplies the
unknowns by four in two dimensions and eight in three, so only a few rounds are
affordable; the error indicator remains useful here as a diagnostic, showing
where the error is even when nothing can be done about it locally.  The second
is to work with triangles instead: build the mesh as ``Tri3`` from the start --
:func:`~dualmesh.generate_rectangle_mesh` takes ``element_type="Tri3"``, and
:func:`~dualmesh.mesh_from_arrays` accepts an arbitrary triangulation -- and the
full adaptive loop becomes available.  The library provides no function that
converts an existing quadrilateral mesh into a triangular one, so that
conversion, if needed, must be done by the user or the mesh generator.  Local
refinement of a three-dimensional mesh is not available by any route.

A measured benchmark: the L-shaped domain
-----------------------------------------

The standard test of an adaptive scheme is Laplace's equation on a domain with a
re-entrant corner.  Take :math:`\Omega` to be the square
:math:`[-1,1] \times [-1,1]` with the quadrant :math:`x > 0,\; y < 0` removed,
an L-shaped region whose interior angle at the origin is :math:`3\pi/2`.  Solve
:math:`-\nabla^2 u = 0` in :math:`\Omega` with Dirichlet data on the whole
boundary taken from the exact solution

.. math::
   :label: corner

   u(r, \theta) = r^{2/3} \sin\!\left( \tfrac{2}{3}\theta \right) ,

in polar coordinates centred on the corner, with :math:`\theta` measured from
the positive :math:`x` axis in :math:`[0, 3\pi/2]`.  A direct calculation
confirms that :eq:`corner` is harmonic and vanishes on the two faces of the
corner, :math:`\theta = 0` and :math:`\theta = 3\pi/2`.

The solution is smooth everywhere except at the origin, where its gradient
behaves like :math:`r^{-1/3}` and is unbounded: :math:`u` lies in
:math:`H^1(\Omega)`, so the problem is well posed, but not in
:math:`H^2(\Omega)`.  This is not an artefact of the data.  Any solution of
Laplace's equation on a domain with a re-entrant corner of interior angle
:math:`\omega > \pi` contains a term :math:`r^{\pi/\omega}` with
:math:`\pi/\omega < 1`, and :eq:`corner` is that term for
:math:`\omega = 3\pi/2`; corner singularities of this kind are a property of the
geometry and are therefore unavoidable in practice.

For a smooth solution, linear elements give an error falling like :math:`h^2`,
and in two dimensions the number of unknowns :math:`N` grows like
:math:`h^{-2}`, so the error falls like :math:`N^{-1}` -- the best a sequence of
linear-element meshes can do.  At a re-entrant corner the argument fails,
because the :math:`h^2` estimate needs a bounded second derivative and
:eq:`corner` has none at the origin.  The few elements nearest the corner then
dominate the total error, and uniform refinement improves them at the same slow
rate as everything else while spending three quarters of each new element budget
far from the corner, where the solution was already well resolved.  Adaptive
refinement escapes this by grading the mesh towards the corner: the elements
there shrink much faster than :math:`h`, the corner contribution falls at the
same rate as the rest, and the :math:`N^{-1}` rate is recovered even though the
solution is not smooth.

Both sequences below start from the same coarse mesh of 24 triangles and 21
nodes, and the adaptive one uses the gradient-recovery indicator with bulk
marking at :math:`\theta = 0.4`.  The quantity reported is the maximum nodal
error :math:`\max_a |u_h(\mathbf{x}_a) - u(\mathbf{x}_a)|` against the number of
unknowns :math:`N`, which here is the number of nodes; comparing against
:math:`N` rather than against the cycle number is essential, because adaptive
and uniform meshes have quite different sizes at the same cycle.

.. table:: Uniform refinement (5 cycles) and adaptive refinement (12 cycles)

   =========  ============================  ==========  ============================
   Uniform N  Uniform error                 Adaptive N  Adaptive error
   =========  ============================  ==========  ============================
   21         :math:`3.62 \times 10^{-2}`   21          :math:`3.62 \times 10^{-2}`
   65         :math:`2.96 \times 10^{-2}`   30          :math:`2.91 \times 10^{-2}`
   225        :math:`2.01 \times 10^{-2}`   37          :math:`1.95 \times 10^{-2}`
   833        :math:`1.30 \times 10^{-2}`   57          :math:`1.11 \times 10^{-2}`
   3201       :math:`8.30 \times 10^{-3}`   93          :math:`7.30 \times 10^{-3}`
   \-         \-                            122         :math:`5.16 \times 10^{-3}`
   =========  ============================  ==========  ============================

Fitting a straight line to :math:`\log(\text{error})` against :math:`\log N`
gives the observed rates :math:`\text{error} \sim N^{-0.30}` for the uniform
sequence and :math:`\text{error} \sim N^{-1.10}` for the adaptive one.  The
uniform rate is close to the :math:`-1/3` predicted for this interior angle and
far short of the :math:`-1` a smooth problem would give; the adaptive rate,
fitted over the last seven cycles once the mesh has begun to grade properly,
recovers the optimal rate for linear elements.  The test suite asserts these as
bands rather than exact figures -- the uniform slope between :math:`-0.45` and
:math:`-0.2`, the adaptive slope below :math:`-0.8` -- so the test records the
qualitative fact without being brittle.  The end points make the same point
without any fitting: twelve adaptive cycles give 122 unknowns at a maximum error
of :math:`5.16 \times 10^{-3}`, while five uniform cycles give 3201 unknowns at
:math:`8.30 \times 10^{-3}`.  The adaptive mesh is a factor of 26 smaller and
its answer is better by a factor of 1.6.

A second run exercises
:func:`~dualmesh.solve_with_adaptive_refinement` on the same problem
with bulk marking at :math:`\theta = 0.6`.  Over eight cycles the element count
rises through 24, 30, 35, 42, 60, 80, 116 to 176, and the maximum nodal error
falls from :math:`3.62 \times 10^{-2}` to :math:`1.17 \times 10^{-2}`.  For
comparison, a uniform mesh of 1536 elements is still at
:math:`1.30 \times 10^{-2}` and only the 6144-element mesh does better, at
:math:`8.30 \times 10^{-3}`: matching the adaptive result uniformly takes a mesh
of a few thousand elements in place of 176.

Running the loop
----------------

:func:`~dualmesh.solve_with_adaptive_refinement` performs the whole
cycle.  It takes a callable that builds a problem on a given mesh, and calls it
afresh on every mesh rather than transferring the old problem across:

.. code-block:: python

   def build(mesh):
       problem = dualmesh.Problem(mesh)
       problem.add_variable("u")
       problem.add_kernel("Diffusion", "diffusion", variable="u")
       problem.add_kernel("BodyForce", "source", variable="u", value=1.0)
       problem.add_boundary_condition(
           "DirichletBC", "walls", variable="u",
           boundary=mesh.sideset_names(), value=0.0)
       return problem

   problem, mesh = dualmesh.solve_with_adaptive_refinement(
       build, initial_mesh, variable="u", num_cycles=4)

Rebuilding rather than transferring is deliberate: it keeps the boundary
conditions, the material definitions and the initial state exactly as the user
wrote them on every mesh, at the cost of discarding the previous solution as a
starting guess.  For a nonlinear problem in which that guess is valuable, drive
the three steps directly instead of using this driver.  The loop stops early if
the marker selects nothing, or if ``max_elements`` is given and the mesh has
grown past it, and a ``callback(cycle, problem, indicators)`` is invoked after
every solve, which is where a calculation writes output files or records a
convergence history like the table above.
