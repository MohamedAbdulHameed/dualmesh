Control volumes and face fluxes
===============================

The dual mesh control domain method (DMCDM) of [Reddy2024]_ imposes each
equation as a balance over the control volume of every node, with the unknowns
interpolated by the shape functions of the finite elements.  This page gives
the discrete equation, its boundary conditions, how it differs from the
Galerkin finite element method (FEM), and the face fluxes of an advective term.
It builds on the balance law and the dual mesh of :doc:`foundations`.

.. contents::
   :local:
   :depth: 1

The balance over a control volume
---------------------------------

An equation for the field :math:`u` is the balance law

.. math::

   \frac{\partial m}{\partial t} + \nabla \cdot \mathbf{F} = s
   \quad \text{in } \Omega ,

with :math:`m` the storage per unit volume, :math:`\mathbf{F}` the physical
flux and :math:`s` the source per unit volume, in the units of the balanced
quantity.  The rate of the storage is :math:`c \, \partial u / \partial t`,
with :math:`c = \partial m / \partial u` the capacity.

The dual mesh gives every node :math:`I` a control volume :math:`V_I`, and
these control volumes partition the domain :math:`\Omega`.  DMCDM integrates the
balance law over :math:`V_I` and turns the divergence into a surface integral.
The discrete equation of node :math:`I` is

.. math::
   :label: dmcdm-residual

   R_I = \int_{V_I} \Bigl( c \, \frac{\partial u}{\partial t} - s \Bigr)
         \mathrm{d}V
       + \oint_{\partial V_I} \mathbf{F} \cdot \mathbf{n} \, \mathrm{d}S = 0 ,

with :math:`\mathbf{n}` the outward unit normal of :math:`\partial V_I`.  The
field and its gradient in these integrals are those of the finite element
interpolant, :math:`u = \sum_a U_a N_a` with :math:`U_a` the value at node
:math:`a` and :math:`N_a` its shape function.  The flux through an interface
therefore uses the gradient of the interpolant there, and no gradient
reconstruction or finite difference enters a diffusive flux.

In weighted-residual terms, DMCDM is a Petrov-Galerkin method: its trial
functions are the shape functions, and the test function of node :math:`I` is
one in :math:`V_I` and zero outside.  The same construction is the control
volume finite element method of [BaligaPatankar1980]_.  A classical finite
volume method [Patankar1980]_ takes the face flux from differences between
neighbouring values instead.

Each interior interface bounds two control volumes, and its flux enters their
two equations with opposite signs.  The sum of the equations over any union of
control volumes leaves only the flux through the boundary of the union, so the
discrete balance holds over every control volume.

Assembly
--------

The code assembles :eq:`dmcdm-residual` element by element.  For each element,
it

1. maps the integration points of the sub-cells (the parts of the control
   volumes inside the element) and of the interfaces between them from the
   reference element,
2. evaluates the field, its gradient and the properties at those points from
   the nodal values of the element,
3. adds the volume integral of each sub-cell to the equation of the node that
   owns it, and
4. adds :math:`\mathbf{F} \cdot \mathbf{A}` to the equation of one node of each
   interface and :math:`-\mathbf{F} \cdot \mathbf{A}` to the equation of the
   other, with :math:`\mathbf{A}` the area vector of the interface (its area
   times its unit normal, :math:`\mathrm{m^2}`) pointing from the first node to
   the second.

The element loop only visits the sub-cells.  The method forms no element
stiffness matrix and no bilinear form.

Boundary conditions
-------------------

Part of the boundary :math:`\partial V_I` of a boundary node lies on the domain
boundary :math:`\partial\Omega`.

* On a boundary with a natural condition, the condition gives the outward normal
  flux :math:`\mathbf{F} \cdot \mathbf{n}`, and its integral over that part
  enters :eq:`dmcdm-residual`.  When the flux depends on :math:`u`, as in a
  Robin condition, the integral enters the Jacobian as well.
* On a boundary with an essential condition, the equation of the node becomes
  :math:`U_I = \hat{U}_I`, with :math:`\hat{U}_I` the prescribed value.

The equation that an essential condition replaces still holds the balance of
:math:`V_I`.  Assembled at the solution without the boundary part, it gives the
reaction, the flow out of the body through the part of :math:`\partial V_I` on
:math:`\partial\Omega`,

.. math::

   Q_I = \int_{\partial V_I \cap \partial\Omega} \mathbf{F} \cdot \mathbf{n}
         \, \mathrm{d}S = -R_I .

:meth:`dualmesh.Problem.boundary_flux` adds :math:`Q_I` over the nodes of a
boundary.  The physics name this flow:
:meth:`~dualmesh.physics.HeatTransfer.heat_flow` gives the heat that leaves
through a boundary, and :meth:`~dualmesh.physics.SolidMechanics.reaction_force`
the force of the body on its support.

Comparison with the finite element method
-----------------------------------------

With ``method="FEM"``, the same kernels, properties, boundary conditions and
solvers give the Galerkin form of the same balance law.  The equation of node
:math:`i` multiplies the balance law by :math:`N_i` and integrates the flux term
by parts,

.. math::

   R_i = \int_\Omega N_i \Bigl( c \, \frac{\partial u}{\partial t} - s \Bigr)
         \mathrm{d}V
       - \int_\Omega \nabla N_i \cdot \mathbf{F} \, \mathrm{d}V
       + \oint_{\partial\Omega} N_i \, \mathbf{F} \cdot \mathbf{n} \, \mathrm{d}S
       = 0 .

FEM forms no face flux.  The shape functions sum to one, so the sum of all its
equations is the balance of the whole domain, but a single equation is no
balance of a region.

On linear triangles and tetrahedra, with a diffusivity and a source that are
constant in each element, the steady equations of the two methods are the same.
On other elements they differ.

DMCDM gives a matrix that is in general not symmetric, even for a symmetric
operator such as :math:`-\nabla \cdot (k \nabla u)`, with :math:`k` the
diffusivity.  The equation of a control volume is an unweighted balance, and
nothing makes the coefficient that couples node :math:`I` to node :math:`J`
equal the one that couples :math:`J` to :math:`I`.  The conjugate gradient
method needs a symmetric matrix, so it does not apply to DMCDM, and the
automatic iterative solvers use the generalized minimal residual method
(GMRES).

Advection schemes
-----------------

An advective term :math:`\mathbf{v} \cdot \nabla u`, with :math:`\mathbf{v}`
the velocity (m/s), enters DMCDM in one of two ways.

* As a volume term,
  :math:`\int_{V_I} \mathbf{v} \cdot \nabla u \, \mathrm{d}V`, with the
  interpolated field.  The kernel ``advection`` takes this form with the scheme
  ``central`` and its default ``advection_form``.
* As a sum over the faces :math:`f` of the control volume, with a face value
  :math:`u_f` that the scheme chooses.  The schemes ``upwind``,
  ``linear_upwind`` and ``limited`` always take this form.  The kernels
  ``heat_convection`` and ``convective_inertia`` take it with ``central`` as
  well when the problem solves a mass equation of the order of the field.

FEM takes ``central`` only.  Its streamline upwind Petrov-Galerkin (SUPG)
stabilization is a kernel of its own, which :doc:`heat_transfer` and
:doc:`fluids` describe.

The face forms
~~~~~~~~~~~~~~

Let :math:`\dot{V}_f = \mathbf{v} \cdot \mathbf{A}_f` be the volume flow
(:math:`\mathrm{m^3/s}`) through face :math:`f`, with :math:`\mathbf{A}_f` its
area vector pointing out of :math:`V_I`.  Since
:math:`\mathbf{v} \cdot \nabla u = \nabla \cdot (\mathbf{v} u) - u \, \nabla
\cdot \mathbf{v}`, the advective term over :math:`V_I` is

.. math::

   \int_{V_I} \mathbf{v} \cdot \nabla u \, \mathrm{d}V
   \approx \sum_f \dot{V}_f \, (u_f - U_I)

in the non-conservative form, and
:math:`\sum_f \dot{V}_f \, u_f` in the conservative form
:math:`\nabla \cdot (\mathbf{v} u)`.  The two differ by
:math:`U_I \sum_f \dot{V}_f`, the discrete divergence of the velocity times the
nodal value.  The interpolated velocity of a discrete flow is not exactly
divergence free, so the conservative form adds a source
:math:`u \, \nabla \cdot \mathbf{v}` that grows with the level of :math:`u`.
The computed temperature then changes when a constant is added to every
temperature, as in a change from degrees Celsius to kelvin.

When the problem solves a mass equation of the order of the field, the kernels
``heat_convection`` and ``convective_inertia`` therefore take the
non-conservative face form with every scheme.  Through the interfaces they
carry the velocity :math:`\mathbf{w}` whose flux the mass equation conserves
(:ref:`the conserved velocity <conserved-advection>`), and through
:math:`\partial\Omega` the velocity :math:`\mathbf{v}`, as the mass equation
takes them, with :math:`\dot{V}_f = \mathbf{w} \cdot \mathbf{A}_f`.  The volume
flows then sum to zero over every control volume whose mass equation is solved,
and the two forms coincide there.  The heat and the momentum are conserved
exactly, the solution does not depend on the level of the temperature at any
iterate, and the upwind scheme is bounded.

Without such a mass equation, in the penalty and Taylor-Hood formulations or
with a velocity that conditions prescribe, the two kernels carry the
interpolated velocity: as a volume term with ``central``, and as a
non-conservative face sum with the other schemes.  The kernel ``advection``
takes the form that its ``advection_form`` names, ``non_conservative`` by
default.

The face values
~~~~~~~~~~~~~~~

Each interface joins two nodes.  The upstream node is the one that
:math:`\mathbf{v} \cdot \mathbf{A}_f` points away from, with the value
:math:`u_\mathrm{up}` and the position :math:`\mathbf{x}_\mathrm{up}`, and the
other is the downstream node, with :math:`u_\mathrm{dn}` and
:math:`\mathbf{x}_\mathrm{dn}`.  Let :math:`\mathbf{x}_f` be the integration
point on the face.  The schemes take these face values [Jasak1996]_.

``central``
   The interpolated field at :math:`\mathbf{x}_f`.

``upwind``
   :math:`u_f = u_\mathrm{up}`.  It is bounded and first order.

``linear_upwind``
   :math:`u_f = u_\mathrm{up} + (\mathbf{x}_f - \mathbf{x}_\mathrm{up}) \cdot
   \nabla u_\mathrm{up}`, with :math:`\nabla u_\mathrm{up}` the gradient at the
   upstream node.  It is second order and unbounded.

``limited``
   :math:`u_f = u_\mathrm{up} + \psi \, (\bar{u}_f - u_\mathrm{up})`, the upwind
   value blended towards the mean :math:`\bar{u}_f = (u_\mathrm{up} +
   u_\mathrm{dn}) / 2` by the van Albada limiter :math:`\psi`.  With
   :math:`a = (\mathbf{x}_\mathrm{dn} - \mathbf{x}_\mathrm{up}) \cdot \nabla
   u_\mathrm{up}` and :math:`b = u_\mathrm{dn} - u_\mathrm{up}`, the ratio of
   consecutive differences is :math:`\chi = 2a/b - 1` [Jasak1996]_, and

   .. math::

      \psi = \frac{\chi (\chi + 1)}{\chi^2 + 1}
           = \frac{2a \, (2a - b)}{(2a - b)^2 + b^2}
      \quad \text{for } \chi > 0 , \qquad \psi = 0 \text{ otherwise} .

   The code evaluates the second expression, which has no division by
   :math:`b`.  The face value is continuous where the limiter switches to
   upwind: :math:`\psi` tends to zero as :math:`\chi` tends to zero, and the
   blend :math:`\bar{u}_f - u_\mathrm{up} = b/2` vanishes with :math:`b`.  The
   scheme is second order where the solution is smooth and close to bounded at
   a jump.

A node has no gradient of its own, so the code reconstructs it by least squares
over the set :math:`\mathcal{N}_a` of the nodes that share an element with node
:math:`a`:

.. math::

   \nabla u_a = \sum_{b \in \mathcal{N}_a} \mathbf{W}_a^{-1}
      (\mathbf{x}_b - \mathbf{x}_a) \, (U_b - U_a) ,
   \qquad
   \mathbf{W}_a = \sum_{b \in \mathcal{N}_a}
      (\mathbf{x}_b - \mathbf{x}_a) (\mathbf{x}_b - \mathbf{x}_a)^T .

The reconstruction is exact for a linear field.  The face flux then depends on
the nodes around the two nodes of the face, and the Jacobian holds these
derivatives exactly.  On a linear problem with a prescribed velocity, Newton's
method therefore converges in one step with ``linear_upwind``, whose face value
is linear in the nodal values.  The limiter
is differentiable away from :math:`\chi = 0`, where it switches to upwind and
Newton's method can cycle.  The default ``line_search="automatic"`` therefore
turns on the backtracking line search of :doc:`nonlinear_and_time` when a
kernel uses ``limited``.

The schemes ``linear_upwind`` and ``limited`` have these limits:

* they need a mesh of the dimension of its space, so a surface or a line in
  space takes ``upwind``,
* they run on one process, because a part of a distributed mesh does not hold
  the nodes around its interface nodes.

At the boundary
~~~~~~~~~~~~~~~

On a face that lies on :math:`\partial\Omega`, both sides are the node whose
control volume the face bounds: at an outflow it is upstream, and at an inflow
without a condition it holds the only value there is.  In the non-conservative
form, the face term :math:`\dot{V}_f (u_f - U_I)` is zero with ``upwind``.  With
``central`` it is the difference between the interpolated value on the face
and the nodal value, which keeps an outlet consistent.  The conservative flux
carries the face value out of the domain.  A boundary without a condition
therefore has zero diffusive flux in both forms and lets the advected field
leave the domain.

:doc:`/verification` gives the cases that check the methods and the advection
schemes.
