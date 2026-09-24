The finite volume methods
=========================

Besides the dual mesh control domain method and the Galerkin finite element
method, ``dualmesh`` implements the two finite volume discretisations that
Chapter 3 of [Reddy2024]_ develops.  The code calls them ``"hfvm"`` and
``"zfvm"``, after the book's names for them: the *half control volume*
formulation, whose unknowns sit at the mesh nodes, and the *zero-thickness
control volume* formulation, whose unknowns sit at the cell centroids.  They
are available through the ``method`` argument of :class:`dualmesh.Problem` and
they consume exactly the same problem definition — the same kernels, the same
materials, the same boundary conditions and the same solvers — as the other
two methods, which is what makes a comparison between them meaningful.

This chapter states what each of the two computes.  It assumes the canonical
form of a problem and the construction of the dual mesh, both of which are
given in :doc:`foundations`.

.. contents::
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

What a finite volume method is
------------------------------

Every equation this library solves is written as a balance law: for the
unknown field :math:`u` there is a flux :math:`\mathbf{F}` and a source
:math:`S`, both of which may depend on :math:`u`, on :math:`\nabla u`, on the
position :math:`\mathbf{x}` and on the time :math:`t`, such that

.. math::
   :label: fv_canonical

   -\nabla \cdot \mathbf{F} + S = 0 \quad \text{in } \Omega .

A **finite volume method** partitions :math:`\Omega` into non-overlapping
*control volumes* :math:`\omega_I`, integrates :eq:`fv_canonical` over each one
and converts the divergence into a surface integral with the divergence
theorem.  The discrete equation of control volume :math:`I` is therefore

.. math::
   :label: fv_balance

   R_I = -\oint_{\partial \omega_I} \mathbf{F} \cdot \mathbf{n} \, \mathrm{d}S
         + \int_{\omega_I} S \, \mathrm{d}V = 0 ,

with :math:`\mathbf{n}` the outward unit normal of :math:`\partial\omega_I`.
No weight function appears.  Because each interior face of the partition is
shared by exactly two control volumes, and the flux through it is computed once
and added to one balance with each sign, the sum of all the discrete equations
retains only the fluxes through :math:`\partial\Omega`.  This is *local
conservation*, and it holds for every union of control volumes, not only for
the whole domain.  It is the property that makes the family attractive for
transport problems [Patankar1980]_.

Equation :eq:`fv_balance` is also, word for word, the equation of the dual mesh
control domain method.  The three discretisations of this library are not
distinguished by the balance they enforce, which is the same, but by two
choices:

Where the unknowns sit.
   The dual mesh control domain method and the vertex-centred finite volume
   method carry one unknown per mesh node and use the median dual control
   domains described in :doc:`foundations`.  The cell-centred finite volume
   method carries one unknown per element, placed at the element centroid, plus
   one per boundary face.  The Galerkin finite element method carries one
   unknown per node and uses no control volumes at all.

How the flux at a face is obtained.
   The dual mesh control domain method evaluates :math:`\mathbf{F}` from the
   element interpolation: the gradient at an interface is the gradient of the
   interpolant there, so no reconstruction is needed.  The vertex-centred
   finite volume method replaces the component of that gradient along the edge
   by a two-point difference between the two nodal values.  The cell-centred
   method has no interpolation to differentiate at all and must reconstruct a
   gradient from the surrounding cell values.  The finite element method never
   forms a face flux; it multiplies :eq:`fv_canonical` by a shape function and
   integrates by parts, which spreads :math:`\mathbf{F}` over the element
   interior.

The finite element method is therefore the outlier of the four: it is the only
one that is not locally conservative in the sense above.  The other three
differ only in the second choice, the one that decides how a gradient is found
at a face.

The vertex-centred method (``"hfvm"``)
--------------------------------------

Control volumes
^^^^^^^^^^^^^^^

The control volumes of the vertex-centred method *are* the control domains of
the dual mesh.  Nothing is recomputed: the same reference-element description
of the sub-cells and of the interfaces, and the same assembly loop, serve both
methods (``src/base/ProblemAssembly.cpp``).  In the book's language the control
volume of an interior node in one dimension is the union of the two half
elements on either side of it, which is where the name *half control volume*
comes from; at a boundary node the control volume is truncated by
:math:`\partial\Omega` and only one half element remains.

The method belongs to the older family of control volume finite element
schemes, which integrate a balance over node-centred control volumes while
interpolating with finite element shape functions [BaligaPatankar1980]_
[BaligaPatankar1983]_.  On a median dual mesh the vertex-centred scheme is also
known as the *box method*, and its convergence theory is given in [BankRose1987]_
and [Hackbusch1989]_.

The edge gradient and its correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The single difference from the dual mesh control domain method is the gradient
used at a control domain interface.  Let the interface separate the control
domains of nodes :math:`O` and :math:`N`, which are the two end nodes of an
element edge, with nodal values :math:`U_O` and :math:`U_N` and position
vectors :math:`\mathbf{x}_O` and :math:`\mathbf{x}_N`.  Write

.. math::

   \mathbf{d} = \mathbf{x}_N - \mathbf{x}_O

for the vector along the edge.  At an integration point of the interface, the
code first evaluates the gradient of the element interpolant,

.. math::

   \nabla u = \sum_{a} U_a \, \nabla \psi_a ,

exactly as the dual mesh control domain method does, and then applies the
correction

.. math::
   :label: hfvm_correction

   \nabla u \;\longleftarrow\; \nabla u
   + \bigl[ (U_N - U_O) - \nabla u \cdot \mathbf{d} \bigr]
     \frac{\mathbf{d}}{\lvert \mathbf{d} \rvert^{2}} .

Read :eq:`hfvm_correction` component by component.  The quantity
:math:`\nabla u \cdot \mathbf{d}` is the change in the interpolant predicted
along the edge; the quantity :math:`U_N - U_O` is the change that the two nodal
values actually record.  Their difference is the error the interpolation makes
along the edge, and dividing it by :math:`\lvert\mathbf{d}\rvert^{2}` and
multiplying by :math:`\mathbf{d}` turns it into the gradient increment that
removes that error.  Taking the dot product of the corrected gradient with
:math:`\mathbf{d}` gives :math:`U_N - U_O` identically, so after the correction
the directional derivative along the edge is exactly the two-point difference
:math:`(U_N - U_O)/\lvert\mathbf{d}\rvert`, which is the classical finite
volume approximation.  The components of the gradient transverse to
:math:`\mathbf{d}` are left as the interpolation gave them.

That transverse part is the **non-orthogonal correction**.  It is needed
because the flux through the interface is :math:`\mathbf{F}\cdot\mathbf{n}`,
and on a general mesh the interface normal :math:`\mathbf{n}` is not parallel
to :math:`\mathbf{d}`.  A pure two-point difference gives only the derivative
along :math:`\mathbf{d}`; on a mesh whose edges are not orthogonal to the faces
they cross, the missing transverse derivative contributes to
:math:`\mathbf{n}\cdot\nabla u` at first order, and a scheme that omits it is
not even consistent.  Keeping the interpolated transverse part restores
consistency, and it costs nothing here because the interpolation was evaluated
anyway.  The treatment follows the standard practice of the finite volume
literature [Jasak1996]_ [DemirdzicMuzaferija1995]_, with one simplification:
because the transverse part comes from a finite element interpolation rather
than from a reconstruction on the cell stencil, it is available as an exact,
differentiated function of the element's nodal values, and the Jacobian of the
corrected flux is exact.

Two consequences follow immediately.  The correction vanishes whenever the
interpolant is linear along the edge, because then
:math:`\nabla u \cdot \mathbf{d} = U_N - U_O` exactly; the method therefore
passes the patch test on arbitrarily distorted meshes, which the test suite
checks for ``Quad4`` and ``Tri3`` in two dimensions and for ``Hex8`` in three,
to :math:`10^{-8}` and :math:`10^{-10}` respectively.  And on a mesh whose
edges are parallel to the interface normals — a one-dimensional mesh, or a
rectangular grid — the transverse term drops out of
:math:`\mathbf{n}\cdot\nabla u` and the scheme is exactly the two-point formula
of Eqs. (3.2.16) and (3.4.6) of [Reddy2024]_.

The correction is applied only at interfaces between two control domains.  At
the part of a boundary control domain's surface that lies on
:math:`\partial\Omega` there is no second node to difference against, so the
interpolated gradient is used unchanged, and the boundary treatment — natural
conditions integrated over the boundary patch, essential conditions replacing
the node's equation, reactions recovered from the discarded equation — is
exactly that of the dual mesh control domain method, described in
:doc:`foundations`.

One dimension: the two methods coincide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In one dimension the vertex-centred finite volume method and the dual mesh
control domain method are the same method.  The reason is that
:eq:`hfvm_correction` has nothing to do: on an ``Edge2`` element the
interpolant is linear, so its derivative at the interface already equals
:math:`(U_N - U_O)/h`, the correction term is identically zero, and the two
assemblies produce the same algebraic equations.  The same is true on an
``Edge3`` element, for the reason given in :doc:`elements`: the derivative of
the quadratic interpolant, evaluated at the control domain interface
:math:`\xi = \pm 1/2`, collapses to the two-point difference across the half
element.

This is not merely an asymptotic statement.  Solving
:math:`-u'' = 10 \cos x` on :math:`(0,1)` with six elements, once with
``method="hfvm"`` and once with ``method="dmcdm"``, gives nodal values whose
largest difference is exactly zero in double precision, and the same holds on a
quadratic ``Edge3`` mesh.  The test suite asserts the agreement to
:math:`10^{-12}`
(``test_half_control_volume_is_the_dual_mesh_method_in_one_dimension``).  It is
the computational statement of Section 5.1 of [Reddy2024]_, where the dual mesh
control domain method is said to reduce to the half control volume formulation
in one dimension.

In two dimensions the two separate, because the interpolated gradient along an
edge of a quadrilateral is not the two-point difference.  On the
:math:`3a \times 2a` conduction problem of Example 3.4.1 of [Reddy2024]_,
discretised with a :math:`3 \times 2` mesh, the largest difference between the
two nodal solutions is :math:`1.7 \times 10^{-2}` on a solution whose range is
one.  Both converge at second order: on that problem, refined through
:math:`6\times4`, :math:`12\times8` and :math:`24\times16` elements, the
observed rate in the maximum nodal error exceeds :math:`1.8` for ``Quad4`` and
for ``Tri3`` alike.

The cell-centred method (``"zfvm"``)
------------------------------------

The unknowns
^^^^^^^^^^^^

The cell-centred method places one unknown at the centroid of every element and
one more at the centroid of every boundary face.  The cell unknowns are the
usual finite volume degrees of freedom; the boundary unknowns are what the book
calls the *zero-thickness control volumes*.  A boundary control volume has no
interior — its measure is zero — so it can carry no source and no accumulation,
and its discrete equation reduces to the statement that the flux arriving from
the adjacent cell equals the flux specified by the boundary condition.  Its
purpose is to give the boundary value somewhere to live: it is a degree of
freedom like any other, so a Dirichlet condition fixes it, a Neumann or Robin
condition leaves it free and determines it from the balance, and the value it
takes is the method's estimate of :math:`u` on the boundary.

The data structure is the owner/neighbour face list that every production
finite volume code is built on, and which ``dualmesh`` shares with OpenFOAM
[OpenFOAM1998]_.  It is built once, in :class:`dualmesh::CellMesh`
(``src/fv/CellMesh.cpp``).  Each face records its owner cell, its neighbour
cell (or :math:`-1` on the boundary), its centroid :math:`\mathbf{x}_f`, its
measure and its outward area vector :math:`\mathbf{S}_f`, whose norm is the
face measure and whose direction points away from the owner.  Centroids,
volumes and area vectors are computed by three-point Gauss quadrature over the
element and its sides using the isoparametric map [Irons1966]_, so curved and
distorted elements are handled without special cases.  The degrees of freedom
are numbered cells first, then boundary faces, which is the order in which
:meth:`dualmesh.Problem.values` and :meth:`dualmesh.Problem.entity_points`
return them.

Why a reconstructed gradient is needed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A cell-centred method stores one number per cell and nothing else.  There is no
interpolation whose derivative could be taken, so the gradient that the flux
:math:`\mathbf{F}(u, \nabla u)` needs at a face must be *reconstructed* from
the surrounding cell values.  The obvious two-point difference between the two
cell centroids is not enough for the same reason as in the vertex-centred
method: the line joining the two centroids is not parallel to the face normal
unless the mesh is orthogonal, and on a triangular or a skewed mesh it never
is.  A transverse contribution is therefore missing, and it must come from a
reconstructed gradient.

Least-squares reconstruction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``dualmesh`` reconstructs the gradient of cell :math:`c` by a weighted
least-squares fit over the entities that share a face with it — the
neighbouring cells, and the zero-thickness boundary nodes of any faces of
:math:`c` that lie on :math:`\partial\Omega`.  Write :math:`\mathbf{x}_c` for
the centroid of :math:`c`, :math:`U_c` for its value, and, for each neighbour
:math:`i`,

.. math::

   \mathbf{d}_i = \mathbf{x}_i - \mathbf{x}_c , \qquad
   w_i = \frac{1}{\lvert \mathbf{d}_i \rvert^{2}} .

The reconstructed gradient :math:`\mathbf{g}` is the minimiser of

.. math::
   :label: lsq_objective

   \min_{\mathbf{g}} \; \sum_i w_i
   \bigl( \mathbf{g} \cdot \mathbf{d}_i - (U_i - U_c) \bigr)^{2} ,

that is, the vector whose directional derivative along each
:math:`\mathbf{d}_i` best matches the observed difference.  The inverse-square
weighting makes the fit local: a distant neighbour influences the result less
than a close one.  Setting the derivative of :eq:`lsq_objective` to zero gives
the normal equations

.. math::
   :label: lsq_normal

   \mathbf{A} \, \mathbf{g} = \sum_i w_i \, (U_i - U_c) \, \mathbf{d}_i ,
   \qquad
   \mathbf{A} = \sum_i w_i \, \mathbf{d}_i \otimes \mathbf{d}_i ,

in which :math:`\mathbf{A}` is a symmetric :math:`n_{\mathrm{dim}} \times
n_{\mathrm{dim}}` matrix that depends only on the geometry.  It is inverted
once, when the :class:`dualmesh::CellMesh` is built, and the result is stored
as one coefficient vector per neighbour, so that the gradient of any field is

.. math::
   :label: lsq_stencil

   \mathbf{g}_c = \sum_i \mathbf{c}_i \, (U_i - U_c) ,
   \qquad
   \mathbf{c}_i = w_i \, \mathbf{A}^{-1} \mathbf{d}_i .

The property that makes :eq:`lsq_stencil` trustworthy is that it is **exact for
a linear field on any mesh, however distorted**.  If :math:`u` is linear with
gradient :math:`\mathbf{G}`, then :math:`U_i - U_c = \mathbf{G} \cdot
\mathbf{d}_i` for every :math:`i`, every residual in :eq:`lsq_objective`
vanishes at :math:`\mathbf{g} = \mathbf{G}`, the minimum is zero, and the fit
returns :math:`\mathbf{G}` whatever the weights and whatever the arrangement of
the neighbours.  The construction is that of [BarthJespersen1989]_; the
higher-order generalisation is [Barth1993]_.  The Green–Gauss gradient, the
other common choice, does *not* have this property, because it needs a value at
the face centroid and the value obtained by interpolating between the two cell
centroids does not sit at the face centroid on a skewed mesh.

The consequence is checked directly.  On the patch test — a linear field
imposed on all four sides of a unit square, solved for the interior — the
cell-centred method reproduces the field at every cell centroid and every
boundary node to about :math:`10^{-10}`: on a :math:`4\times4` mesh the
measured maximum error is :math:`9 \times 10^{-16}` on a regular ``Quad4``
mesh, :math:`1.7 \times 10^{-10}` on the same mesh distorted so that no element
is a parallelogram, and :math:`4 \times 10^{-11}` to :math:`1.1 \times 10^{-10}`
on ``Tri3`` meshes.  The residual is the tolerance of the nonlinear iteration,
not a discretisation error.

The face flux
^^^^^^^^^^^^^

For every face the flux is evaluated once, at the face centroid, and added to
the balance of the owner and subtracted from the balance of the neighbour, so
the method is conservative by construction.  The value at the face is the
distance-weighted interpolation

.. math::

   u_f = w \, U_O + (1 - w) \, U_N , \qquad
   w = \frac{\lvert \mathbf{x}_f - \mathbf{x}_N \rvert}
            {\lvert \mathbf{x}_f - \mathbf{x}_O \rvert
             + \lvert \mathbf{x}_f - \mathbf{x}_N \rvert} ,

which gives the nearer centroid the larger weight, and the gradient at the face
is the corrected average

.. math::
   :label: zfvm_gradient

   \nabla u |_f = \bar{\mathbf{g}}
   + \bigl[ (U_N - U_O) - \bar{\mathbf{g}} \cdot \mathbf{d} \bigr]
     \frac{\mathbf{d}}{\lvert \mathbf{d} \rvert^{2}} ,
   \qquad
   \bar{\mathbf{g}} = w \, \mathbf{g}_O + (1 - w) \, \mathbf{g}_N ,
   \qquad
   \mathbf{d} = \mathbf{x}_N - \mathbf{x}_O ,

with :math:`\mathbf{g}_O` and :math:`\mathbf{g}_N` the least-squares gradients
:eq:`lsq_stencil` of the two cells.  Equation :eq:`zfvm_gradient` is the same
correction as :eq:`hfvm_correction`, with the reconstructed gradient in place
of the interpolated one: the component along :math:`\mathbf{d}` is the
two-point difference, and the transverse component is the reconstruction
[Jasak1996]_ [DemirdzicMuzaferija1995]_.  On a boundary face, :math:`N` is the
zero-thickness boundary node, :math:`\mathbf{d}` runs from the cell centroid to
the face centroid, and :math:`\bar{\mathbf{g}}` is the owner's gradient alone.

The term :math:`\bar{\mathbf{g}}` is **differentiated exactly**.  Each
least-squares gradient is a linear combination of the values of the cell and
of the cells in its stencil, so :math:`\bar{\mathbf{g}}` at a face depends on
the owner, the neighbour and all of their stencil neighbours, and the
automatic differentiation carries a derivative slot for every one of them.
The Jacobian is therefore exact on any mesh, and Newton's method solves a
linear problem in one iteration and converges quadratically on a nonlinear one,
skewed mesh or not (``test_the_cell_centred_jacobian_is_exact_on_a_skewed_mesh``).
The price is a wider matrix: a face couples its two cells to their stencil
neighbours, not only to each other.  Many codes instead *lag* the
non-orthogonal part -- evaluate it at the previous iterate and leave it out of
the Jacobian, the deferred correction of [Jasak1996]_ -- which keeps the matrix
as compact as a finite-difference matrix but makes the Jacobian inexact on a
non-orthogonal mesh; an earlier version of this library did so, and on a
hybrid hexahedron and pyramid mesh its Newton iteration contracted the error by
only 1.3 per cent per step.  The lagged form survives in one place: when a
face's stencil needs more derivative slots than the automatic differentiation
provides (``DUALMESH_MAX_AD_DERIVATIVES``, 48 by default), that face falls back
to it.  On a mesh whose centroid line is parallel to the face normal,
:math:`\bar{\mathbf{g}}` contributes nothing to
:math:`\mathbf{n} \cdot \nabla u`, and the scheme is exactly Eqs. (3.2.16)
and (3.4.26) of [Reddy2024]_.

Sources are integrated over the cell, with the quadrature rule the kernel asks
for; the rules that are meaningful only for a node-centred control domain
(``nodal``, ``interface``, ``control_domain_trapezoid``) collapse to the
one-point Gauss rule on each patch of the element, which is the cell-centred
meaning of lumping a term.

Boundary treatment for the cell-centred method
----------------------------------------------

The normal gradient at a boundary face needs its own approximation, because
there is only one cell on one side of it.  The ``boundary_gradient`` option of
:class:`dualmesh.Problem` selects between two, corresponding to the enumeration
:cpp:enum:`dualmesh::BoundaryGradient`.

``"first_order"`` (:cpp:enumerator:`BoundaryGradient::FirstOrder`) uses the
general face formula :eq:`zfvm_gradient` with :math:`N` the boundary node, so
that the normal derivative comes from the two-point difference between the cell
value and the boundary value over the distance from the cell centroid to the
face centroid, corrected transversely by the owner's reconstructed gradient.
It is the default and it is robust, but the difference is one-sided over a
distance of half a cell, so its truncation error is first order in the cell
size.

``"second_order"`` (:cpp:enumerator:`BoundaryGradient::SecondOrder`) fits a
one-sided quadratic through three values: the boundary node, the owner cell
:math:`O`, and a second cell :math:`2` beyond it.  The second cell is chosen
once, when the mesh is built, as the neighbour across the face of :math:`O`
whose normal is most nearly opposed to the boundary normal.  Let
:math:`\hat{\mathbf{n}}` be the outward unit normal of the boundary face and

.. math::

   \mathbf{r}_O = \mathbf{x}_O - \mathbf{x}_f , \qquad
   \mathbf{r}_2 = \mathbf{x}_2 - \mathbf{x}_f ,

be the offsets of the two cell centroids from the face centroid.  Their
distances *along the inward normal* are

.. math::

   s_1 = -\, \mathbf{r}_O \cdot \hat{\mathbf{n}} , \qquad
   s_2 = -\, \mathbf{r}_2 \cdot \hat{\mathbf{n}} .

Differentiating the Lagrange interpolant through the three points
:math:`s = 0, s_1, s_2` at :math:`s = 0`, and negating it so that the result is
the derivative along the outward normal, gives

.. math::
   :label: zfvm_second_order

   \frac{\partial u}{\partial n}\Big|_f
   = c_b \, U_b + c_O \, \tilde{U}_O + c_2 \, \tilde{U}_2 ,
   \qquad
   c_b = \frac{s_1 + s_2}{s_1 s_2} , \quad
   c_O = \frac{s_2}{s_1 (s_1 - s_2)} , \quad
   c_2 = \frac{s_1}{s_2 (s_2 - s_1)} ,

which on a uniform one-dimensional mesh of spacing :math:`h`, where
:math:`s_1 = h/2` and :math:`s_2 = 3h/2`, reduces to the familiar
:math:`(8 U_b - 9 U_O + U_2)/(3h)`, Eq. (3.2.14) of [Reddy2024]_.

The tildes in :eq:`zfvm_second_order` are the part that is *not* in the book,
and they matter.  Equation (3.2.14) is derived for a one-dimensional mesh, in
which the two cell centres lie on the normal through the face centre.  On an
unstructured mesh they do not: a cell centroid is displaced sideways as well as
inwards.  The code therefore moves each cell value onto the normal line before
using it, with the cell's own reconstructed gradient:

.. math::

   \tilde{U}_O = U_O - \mathbf{g}_O \cdot \mathbf{t}_O , \qquad
   \tilde{U}_2 = U_2 - \mathbf{g}_2 \cdot \mathbf{t}_2 ,
   \qquad
   \mathbf{t} = \mathbf{r} - (\mathbf{r}\cdot\hat{\mathbf{n}})\,\hat{\mathbf{n}} ,

where :math:`\mathbf{t}` is the tangential (skew) part of the offset.  This is
a first-order Taylor correction of the value from the centroid to its
projection on the normal, and without it the one-sided quadratic would not
reproduce even a linear field on a triangular mesh, so the patch test would
fail.  With it, :eq:`zfvm_second_order` is the generalisation of Eq. (3.2.14)
of [Reddy2024]_ to unstructured meshes; the generalisation is this library's,
not the book's.

The full face gradient is then assembled from the one-sided normal derivative
and the transverse part of the averaged reconstruction,

.. math::

   \nabla u|_f = \frac{\partial u}{\partial n}\Big|_f \, \hat{\mathbf{n}}
   + \bigl[ \bar{\mathbf{g}} - (\bar{\mathbf{g}} \cdot \hat{\mathbf{n}})
            \hat{\mathbf{n}} \bigr] .

Two honest qualifications belong here.  The second-order
formula is used only when the owner has an interior face opposed to the
boundary normal and the geometry is sane, that is, when :math:`s_1 > 0` and
:math:`s_2 > s_1`; when it is not — for a cell wedged in a corner of the
domain, or one whose second centroid is not farther from the face than its own
— the code falls back silently to the first-order difference on that face, so
the boundary treatment of a mesh may be mixed.
And the second-order formula widens the stencil of every boundary face from two
entities to three, which slightly increases the bandwidth of the matrix.

The gain is real.  On :math:`-u'' = 10 \cos x` over :math:`(0,1)` with eight
cells and a trapezoidal source rule, the maximum error against the exact
solution falls from :math:`1.9 \times 10^{-2}` with the first-order boundary
gradient to :math:`1.2 \times 10^{-3}` with the second-order one, a factor of
fifteen at no change in the number of unknowns.  The test suite asserts a
factor of at least :math:`2.5`
(``test_second_order_boundary_gradient_is_more_accurate``).  Both variants are
verified against Example 3.3.1 of [Reddy2024]_ node by node: with four cells
the book's Eq. (7) gives cell values :math:`0.5686, 1.0906, 1.0356, 0.4777` for
the first-order boundary gradient and its Eq. (8) gives
:math:`0.4951, 1.0240, 0.9757, 0.4246` for the second-order one, and the code
reproduces both to :math:`1.1 \times 10^{-4}`, which is the precision to which
the book prints them.

Choosing between them
---------------------

The three control volume methods solve the same problems and agree as the mesh
is refined.  What differs is cost, convenience and where the accuracy comes
from.

The **dual mesh control domain method** is the default and is usually the right
choice for a diffusion-dominated problem on a mesh of good quality.  It needs
no reconstruction at all, its Jacobian is exact because every quantity in the
residual is a differentiated function of the nodal values, and on simplicial
meshes with constant coefficients it produces the same algebraic equations as
the Galerkin finite element method.  Its limitation is stated in
:doc:`elements`: it does not gain an order of accuracy from quadratic elements.

The **vertex-centred finite volume method** costs the same as the dual mesh
control domain method — the same unknowns, the same sparsity, the same assembly
loop — and differs only in :eq:`hfvm_correction`.  The reason to choose it is
that it is the scheme of Chapter 3 of [Reddy2024]_ and of the classical finite
volume literature, so it is the right method to run when the object is to
reproduce those results or to compare against a code that uses them.  It is not
more accurate than the dual mesh control domain method: on the conduction
problem of Example 3.4.1, refined through three meshes, the maximum nodal error
of the vertex-centred method is about twice that of the dual mesh control
domain method on the same mesh (:math:`1.13\times10^{-2}` against
:math:`5.93\times10^{-3}`, then :math:`2.90\times10^{-3}` against
:math:`1.47\times10^{-3}`, then :math:`7.28\times10^{-4}` against
:math:`3.65\times10^{-4}`), and in one dimension it is not different from it at
all.

The **cell-centred finite volume method** is the layout of production
computational fluid dynamics, and it is the method to use when the point of
comparison is such a code, or when the problem is transport-dominated.  Its
face-based structure makes the flux through any surface directly available, and
it accepts every element type, including the serendipity elements and the
pyramid that the node-based methods refuse.  It costs more unknowns on a
simplicial mesh — a two-dimensional triangulation has about twice as many
triangles as nodes, and a tetrahedral mesh about five to six times as many
cells as nodes — and its exact Jacobian couples each cell to the stencils of
its neighbours, so its matrix is wider than those of the node-based methods;
the automatic linear solver's preconditioned iteration handles it well in
three dimensions.  It is
also the only one of the three whose accuracy depends on a reconstruction and
therefore on the quality of the cell stencils: a cell with few or badly placed
neighbours has a poor gradient, and the boundary is where that shows first,
which is the reason the second-order boundary gradient exists.

A cost that applies to all three: the matrices of the dual mesh and finite
volume methods are in general **not symmetric**, even for a self-adjoint
operator such as :math:`-\nabla\cdot(k\nabla u)`.  The equation of a control
volume is a balance, not a weighted residual with the shape function of its
node, so there is nothing to make the coefficient coupling :math:`I` to
:math:`J` equal the one coupling :math:`J` to :math:`I`; the Galerkin method,
whose bilinear form is symmetric, does give a symmetric matrix.  The conjugate
gradient method is therefore unavailable with the three control volume methods:
use ``linear_solver="automatic"`` (the default), ``"lu"``, ``"bicgstab"`` or
``"gmres"``, which is what the
error message says if the conjugate gradient solver is asked for and fails.

Verification
------------

The two finite volume methods are checked against Chapter 3 of [Reddy2024]_ in
``tests/python/test_finite_volume.py``.  The one-dimensional problem of Example
3.3.1 is reproduced node by node for both formulations and for both boundary
gradients, including the secondary variables at the two ends, which
Table 3.3.1 of the book gives as :math:`-q(0) = 4.5898` and :math:`q(1) =
3.7888` for the half control volume formulation with four subdivisions; the
two-dimensional conduction problem of Example 3.4.1 is reproduced for the
:math:`3 \times 2` mesh of the book's Eqs. (9) and (14).  Beyond the book, the
tests check the patch test on distorted meshes in two and three dimensions,
second-order convergence in the maximum nodal error, a nonlinear conduction
problem against its closed-form Kirchhoff solution with both Newton and Picard
iteration, a transient slab against its analytical decay, and a convecting fin,
which fixes the sign of the Robin condition.
