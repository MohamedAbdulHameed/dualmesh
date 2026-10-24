Control volumes and face fluxes
===============================

The dual mesh control domain method (DMCDM) of [Reddy2024]_ enforces each
equation as a balance over the control volume of every node, with the
unknowns interpolated by the finite element shape functions.  This chapter
states the balance, how it differs from the Galerkin finite element method,
and the face fluxes with which DMCDM writes an advective term.  It assumes the
canonical form of a problem and the construction of the dual mesh, both given
in :doc:`foundations`.

.. contents::
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

The balance over a control volume
---------------------------------

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
conservation*, and it holds for every union of control volumes as well as for
the whole domain.  It is the property that makes the family attractive for
transport problems [Patankar1980]_.

Equation :eq:`fv_balance` is the equation of DMCDM.  Its control volumes are the
median dual control domains of :doc:`foundations`, one per mesh node, and the
flux :math:`\mathbf{F}` on their faces is evaluated from the element
interpolation: the gradient on a face is the gradient of the interpolant there,
so no reconstruction is needed.  A textbook finite volume method takes the face
flux from finite differences between neighbouring values instead.  In
weighted-residual terms DMCDM is a Petrov-Galerkin method, with finite element
trial functions and the indicator function of each control volume as the test
function.  The same construction is the control volume finite element method
of [BaligaPatankar1980]_.

The Galerkin finite element method (FEM) never forms a face flux.  It multiplies
:eq:`fv_canonical` by a shape function and integrates by parts, which
distributes :math:`\mathbf{F}` over the element interior.  It is conservative
over the whole domain but not over each control volume.

The matrix of DMCDM is in general **not symmetric**, even for a self-adjoint
operator such as :math:`-\nabla\cdot(k\nabla u)`.  The equation of a control
volume is an unweighted balance over that volume, so no property of the
discretisation makes the coefficient coupling :math:`I` to :math:`J` equal the
one coupling :math:`J` to :math:`I`.  The Galerkin method, whose bilinear form
is symmetric, gives a symmetric matrix.  The conjugate gradient method is
therefore unavailable with DMCDM.  Use ``linear_solver="automatic"`` (the
default), ``"LU"``, ``"BiCGStab"`` or ``"GMRES"``.

Advection schemes
-----------------

An advective term (``advection``, the heat that a flow carries and the inertia
of a flow) has two statements in DMCDM.  With ``advection_scheme="central"``,
the default of ``advection``, it is the source :math:`\mathbf{v} \cdot \nabla
\phi` integrated over the control volume, with the interpolated field, as the
finite element method does.  The streamline stabilisation (SUPG), the default
of a flow and of the heat it carries, adds to this source (see
:doc:`heat_and_fluids`).  Every other scheme writes the term with the faces of
the control volume and a face value :math:`\phi_f` that the scheme chooses.
With :math:`m = \mathbf{v} \cdot \mathbf{A}` the normal velocity times the
face area, outward from the control volume :math:`V`, the term is

.. math::

   \int_V \mathbf{v} \cdot \nabla \phi \, dV \approx \sum_f m\, (\phi_f - \phi_V)

in the non-conservative form, and :math:`\sum_f m\, \phi_f` in the conservative
form :math:`\nabla \cdot (\mathbf{v} \phi)`.  The two differ by
:math:`\phi_V \sum_f m`, the discrete divergence of the velocity times the
value.  The interpolated velocity of a discrete flow is not exactly divergence
free, so its conservative form would add the source
:math:`T \nabla \cdot \mathbf{v}`, which depends on the temperature scale: on a
channel at :math:`\mathrm{Re} = 100` that develops from a uniform inflow
(16 by 16 elements), it moved the temperature by 13 K when every temperature
was raised by 300 K, and it took the temperature to 1.67 on a problem bounded
by :math:`[0, 1]`.

``heat_convection`` and ``convective_inertia`` therefore carry the velocity
whose flux the mass equation conserves, :math:`\mathbf{w} = \mathbf{v} - \tau
\mathbf{r}_m` through the faces between control volumes and :math:`\mathbf{v}`
through the boundary, as the mass equation takes them, with :math:`m =
\mathbf{w} \cdot \mathbf{A}`.  Its flux sums to zero over every control volume
whose mass equation is solved, so the two forms coincide: the heat and the
momentum are conserved exactly, the solution does not depend on the
temperature scale, and upwind is bounded.  In a closed cavity driven by its
lid, the heat entering at the hot wall leaves at the cold one to round-off.
Without a mass equation (the penalty formulation, or a velocity that
conditions prescribe), they take the non-conservative form with the
interpolated velocity.  ``advection`` takes the form of ``advection_form``, the
non-conservative one by default.

Let :math:`C` be the upstream node of the face, the owner when
:math:`\mathbf{v} \cdot \mathbf{n} \ge 0` and the neighbour otherwise, and
:math:`D` the downstream node.  The schemes are

* ``"upwind"``: :math:`\phi_f = \phi_C`.  It is bounded and first order.
* ``"linear_upwind"``: :math:`\phi_f = \phi_C + (\mathbf{x}_f - \mathbf{x}_C)
  \cdot \nabla \phi_C`, with the gradient :math:`\nabla \phi_C` at the upstream
  node [Jasak1996]_.  It is second order but not bounded.
* ``"limited"``: :math:`\phi_f = \phi_C + \psi(r)\,(\phi_f^{CD} - \phi_C)`, the
  upwind value blended towards the central value :math:`\phi_f^{CD} = w\,
  \phi_O + (1 - w)\,\phi_N` by a limiter, in the form of Eq. (3.66) of
  [Jasak1996]_.  The ratio of consecutive differences is
  :math:`r = 2\, \mathbf{d} \cdot \nabla \phi_C / (\phi_D - \phi_C) - 1`, with
  :math:`\mathbf{d}` the vector from the upstream to the downstream node, as
  the TVD schemes of OpenFOAM [OpenFOAM1998]_ compute it, and the limiter is
  that of van Albada, :math:`\psi(r) = r (r + 1) / (r^2 + 1)` for :math:`r > 0`
  and zero otherwise.  dualmesh evaluates it as :math:`\psi = 2a (2a - b) /
  ((2a - b)^2 + b^2)` with :math:`a = \mathbf{d} \cdot \nabla \phi_C` and
  :math:`b = \phi_D - \phi_C`, which has no division by :math:`b`.  It tends to
  1 when :math:`b` tends to zero with the sign of :math:`a`, and the face value
  is continuous where the limiter switches to upwind, because there
  :math:`\phi_f^{CD} - \phi_C` tends to zero as well.  It is second order where
  the solution is smooth and nearly bounded at a jump.

A node has no gradient of its own, so DMCDM reconstructs it by least squares
over the nodes of the elements around the node, :math:`\nabla \phi_a = \sum_b
\mathbf{w}_{ab} (\phi_b - \phi_a)` with :math:`\mathbf{w}_{ab} =
\mathbf{G}_a^{-1} (\mathbf{x}_b - \mathbf{x}_a)` and :math:`\mathbf{G}_a =
\sum_b (\mathbf{x}_b - \mathbf{x}_a)(\mathbf{x}_b - \mathbf{x}_a)^T`, which is
exact for a linear field.  The flux of a face then depends on the nodes around
its two nodes, and the Jacobian holds those derivatives exactly: linear
upwind, which is linear in :math:`\phi`, converges in one Newton step.  A part
of a distributed mesh does not hold the nodes around its interface nodes, so
linear upwind and limited are refused there for now.

The limiter is a differentiable function of the nodal values away from
:math:`r = 0`, and the automatic differentiation of the residual includes it,
so Newton's method keeps its fast convergence: four iterations on the boundary
layer below.  It is not differentiable at :math:`r = 0`, where it switches to
upwind, and Newton's method can cycle between the two sides of the kink.  The
default ``line_search="automatic"`` therefore turns the backtracking line
search on when a kernel uses the limited scheme, and ``line_search="none"``
turns it off.

**At the boundary.**  In the non-conservative form a face on the boundary of
the domain takes :math:`m (\phi_f - \phi_V)`, which is zero at an outflow,
where the upstream value is :math:`\phi_V`.  The conservative flux through the
boundary carries the value of the control volume that the face bounds: at an
outflow this value is upstream, and at an inflow without a condition it is the
only value there is.  A boundary without a condition therefore has a zero
conductive flux in both forms and passes the advected field out of the domain
unchanged.

**Measurements.**  On :math:`\phi' = \phi'' / \mathrm{Pe}` over
:math:`(0, 1)` with :math:`\phi(0) = 0` and :math:`\phi(1) = 1`, with
:math:`\mathrm{Pe} = 100` and 20 elements (a cell Péclet number of 5), and on
the skew jump of [BrooksHughes1982]_ (a jump of the inflow value carried
across the unit square at 30 degrees to a :math:`24 \times 24` mesh, with
:math:`k = 10^{-6}`):

.. list-table::
   :header-rows: 1

   * - Scheme
     - Range of :math:`\phi`, boundary layer
     - Largest error
     - Order at :math:`\mathrm{Pe} = 5`
     - Range of :math:`\phi`, skew jump
   * - central
     - :math:`[-0.43, 1]`
     - 0.44
     - 2.0
     - :math:`[-0.29, 1.02]`
   * - upwind
     - :math:`[0, 1]`
     - 0.16
     - 0.92
     - :math:`[0, 1]`
   * - linear_upwind
     - :math:`[-0.041, 1]`
     - 0.048
     - 1.86
     - :math:`[-0.034, 1.05]`
   * - limited
     - :math:`[0, 1]`
     - 0.11
     - 2.05
     - :math:`[-0.018, 1.004]`

``tests/python/test_advection_schemes.py`` checks these properties.

Verification
------------

``tests/python/test_conduction_methods.py`` checks both methods against
closed-form solutions: a nonlinear conduction problem against its Kirchhoff
solution with Newton and Picard iteration, a transient slab against its
analytical decay, a convecting fin, which verifies the sign of the Robin
condition, and a jump of the conductivity between two blocks.  The
convergence orders of both methods on every element type are in
:doc:`/verification`.
