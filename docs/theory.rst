Theory manual
=============

This chapter states exactly what the code computes.  The notation follows
J. N. Reddy, *Computational Methods in Engineering: Finite Difference, Finite
Volume, Finite Element, and Dual Mesh Control Domain Methods* (CRC Press,
2024), referred to below as "the book".

.. contents::
   :local:
   :depth: 2

The canonical form of a problem
-------------------------------

Every equation solved by this library is written as a balance law: for each
unknown field :math:`u_i` there is a flux :math:`\mathbf{F}_i` and a source
:math:`S_i` such that

.. math::
   :label: canonical

   \mathcal{R}_i(u) \equiv -\nabla \cdot \mathbf{F}_i(u, \nabla u, \mathbf{x}, t)
   + S_i(u, \nabla u, \mathbf{x}, t) = 0 \quad \text{in } \Omega .

Both :math:`\mathbf{F}_i` and :math:`S_i` may depend on every unknown of the
problem and on its gradient, which is what makes coupled and nonlinear systems
expressible.  Examples, all of which appear in the modules:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Equation
     - Flux :math:`\mathbf{F}`
     - Source :math:`S`
   * - Steady conduction :math:`-\nabla\cdot(k\nabla T) = q'''`
     - :math:`k \nabla T`
     - :math:`-q'''`
   * - Advection–diffusion
     - :math:`k \nabla u` (and :math:`-\mathbf{v}u` in conservative form)
     - :math:`\mathbf{v}\cdot\nabla u`
   * - Elasticity, component :math:`i`
     - :math:`h\,(\sigma_{i1}, \sigma_{i2}, \sigma_{i3})`
     - :math:`-h f_i`
   * - Penalty Stokes, component :math:`i`
     - :math:`\mu(\nabla v_i + (\nabla \mathbf{v})_i) + \gamma (\nabla\cdot\mathbf{v})\,\mathbf{e}_i`
     - :math:`-f_i`
   * - Mixed Euler–Bernoulli beam, :math:`w` equation
     - :math:`\mathrm{d}M/\mathrm{d}x + N\,\mathrm{d}w/\mathrm{d}x`
     - :math:`c_f w - q`

The *secondary variable* of an equation is the normal component of its flux,

.. math::

   q_n \equiv \mathbf{n}\cdot\mathbf{F} ,

which is the quantity that appears in the natural boundary condition: the
heat flow into the body, the traction component, the axial force of a bar, the
shear force of a beam.  The primary variable :math:`u` and its secondary
variable :math:`q_n` form the duality pair of the equation, and exactly one of
the two is specified at every boundary point.

The primal mesh and the dual mesh
---------------------------------

The **primal mesh** is a mesh of finite elements.  Supported element types are
``Edge2``, ``Tri3``, ``Quad4``, ``Tet4``, and ``Hex8``; all field variables use
the corresponding (multi-)linear Lagrange interpolation,

.. math::

   u(\mathbf{x}) \approx \sum_{a=1}^{n_e} U_a \, \psi_a(\mathbf{x}) ,

with the isoparametric map :math:`\mathbf{x}(\boldsymbol{\xi}) = \sum_a
\mathbf{x}_a \psi_a(\boldsymbol{\xi})`.

The **dual mesh** is the set of node-centred *control domains* (CDs).  It is
the median dual (also called the box or Donald dual): inside each element, the
part of the control domain belonging to node :math:`a` is the sub-cell bounded
by

* the node itself,
* the midpoints of the element edges that meet at the node,
* the centroids of the element faces that meet at the node (in three
  dimensions),
* the element centroid.

Consequently

* in one dimension the control domain of an interior node is the union of the
  two half-elements on either side of it;
* for a rectangle the control domains are the bisecting rectangles used
  throughout Chapter 5 of the book;
* for a general quadrilateral or a triangle it is the construction of Section
  9.9 of the book;
* at a boundary node the control domain is truncated by the domain boundary
  (a half control domain in one dimension, a quarter of one at a corner).

The interface between the control domains of the two end nodes :math:`(a, b)`
of an element edge passes through the edge midpoint, the centroids of the two
adjacent faces, and the element centroid.  Because every interface is shared by
exactly two control domains with opposite normals, the discrete equations are
*locally conservative*: summing the equations of all control domains leaves
only the boundary fluxes.

In this implementation the sub-cells and the interfaces are never meshed
explicitly.  They are described once per reference element as tensor-product
patches in reference coordinates and are mapped with the element's own
isoparametric map, so the same code handles straight and distorted elements in
one, two, and three dimensions (see :class:`Patch` in
``include/dualmesh/fe/ReferenceElement.h``).

The dual mesh control domain method
-----------------------------------

Integrating :eq:`canonical` over the control domain of node :math:`I` and
applying the divergence theorem gives the discrete equation of that node:

.. math::
   :label: dmcdm

   R_I = -\oint_{\partial CD_I} \mathbf{F}\cdot\mathbf{n} \, \mathrm{d}S
         + \int_{CD_I} S \, \mathrm{d}V = 0 .

There is no weight function: the balance law is enforced exactly as it is
written, which is the finite volume idea.  The fluxes and the sources are
evaluated from the primal interpolation, which is the finite element idea.  In
particular, the gradient needed on an interface is the gradient of the
interpolant, so no gradient reconstruction and no ad-hoc differencing appear
anywhere.

Assembly is element by element.  For an element :math:`e` the code

1. builds the integration points of the sub-cells (volume terms) and of the
   interfaces (surface terms) from the reference description of the dual mesh;
2. evaluates :math:`u`, :math:`\nabla u`, and the material properties at those
   points from the element's nodal values;
3. adds :math:`\int S \,\mathrm{d}V` to the owning node's equation and
   :math:`\mp \mathbf{F}\cdot\mathbf{n}\,\Delta S` to the two nodes that share
   each interface.

There are no element stiffness matrices and no element-by-element assembly of
a bilinear form; the element loop is only a device for visiting the sub-cells.

Boundary control domains and secondary variables
------------------------------------------------

Part of the boundary :math:`\partial CD_I` of a boundary node's control domain
lies on :math:`\partial\Omega`.  Two cases arise.

*Natural (Neumann, Robin) boundary:* the secondary variable is known, so
:math:`-\int q_n \,\mathrm{d}S` is added to the residual, with :math:`q_n`
given by the boundary condition.  For a Robin condition,
:math:`q_n = q_0 - h\,(u - u_\infty)`, the unknown :math:`u` enters that
integral, which contributes to the Jacobian as well.

*Essential (Dirichlet) boundary:* the primary variable is known, and the
node's equation is replaced by :math:`u_I = \hat{u}_I`.  The equation that was
replaced is not discarded: evaluated at the converged solution it returns the
secondary variable, i.e. the *reaction*

.. math::

   Q_I = \int_{\partial CD_I \cap \partial\Omega} q_n \, \mathrm{d}S
       = -\oint_{\partial CD_I \setminus \partial\Omega} \mathbf{F}\cdot\mathbf{n}\,\mathrm{d}S
         + \int_{CD_I} S\,\mathrm{d}V ,

which :meth:`dualmesh.Problem.reactions` returns node by node and
:meth:`dualmesh.Problem.total_reaction` returns summed over a boundary.  This
is how the heat flow :math:`Q(0)` of Example 5.3.1, the heat flow through the
surface of a cylinder in Example 5.3.2, and support reactions are obtained.

The finite element method for comparison
----------------------------------------

Setting ``method="fem"`` on the same problem replaces :eq:`dmcdm` by the
Galerkin weak form of the same canonical equation,

.. math::

   R_I = \int_{\Omega} \nabla \psi_I \cdot \mathbf{F} \, \mathrm{d}V
       + \int_{\Omega} \psi_I \, S \, \mathrm{d}V
       - \oint_{\partial\Omega} \psi_I \, q_n \, \mathrm{d}S = 0 ,

so the two methods are driven by the same kernels, the same materials, the same
boundary conditions, and the same solvers.  Two consequences are worth
remembering, and both are checked in the test suite:

* for linear triangles and tetrahedra with constant coefficients the two
  methods produce **identical** algebraic equations (Section 5.4 of the book;
  see ``test_triangles_make_the_dual_mesh_method_equal_the_finite_element_method``);
* for quadrilaterals the two differ slightly, and the dual mesh results are
  usually the more accurate of the two (Tables 5.4.1–5.4.3 of the book).

Quadrature
----------

Each kernel chooses its own rule with the ``quadrature`` parameter, because in
this method the rule is part of the model, not only a numerical detail:

``gauss1`` … ``gauss10``
    Gauss–Legendre rules on the sub-cell or interface patch (the default is
    ``gauss2``, which integrates the linear interpolant exactly).

``midpoint``
    One point at the centre of the patch.  Combined with
    ``reduced_integration=True`` this is the selective reduced integration used
    for the transverse shear terms of shear-deformable beams and plates and for
    the penalty term of incompressible flow.

``trapezoid``, ``simpson``
    The corner and Simpson rules on the patch, for reproducing classical finite
    volume source treatments.

``nodal``
    One point at the owning node, with the measure of the control domain as
    the weight; the row-sum (lumped) treatment of mass-like terms.

``interface``
    One point at the control domain interface, with the measure of the
    sub-cell as the weight.

``control_domain_trapezoid``
    The trapezoidal rule over the whole control domain, which for an interior
    node in one dimension is :math:`F_I = \tfrac{\Delta x}{2}\,[f(x_A) +
    f(x_B)]` with :math:`x_A` and :math:`x_B` the two interfaces, and for a
    boundary node uses the node and its single interface.  This is the source
    rule of Chapter 3 of the book and reproduces its Example 3.3.1 exactly.

Setting ``reduced_integration=True`` evaluates the *solution* (and hence the
material properties) at the element centroid while the geometry is still
integrated exactly.  This is the mechanism the book uses to avoid shear
locking in displacement models of shear-deformable theories, to avoid membrane
locking in von Kármán problems, and to make the penalty formulation of
incompressible flow work.

Coordinate systems
------------------

The integrals in :eq:`dmcdm` carry a factor that depends on the coordinate
system:

.. list-table::
   :header-rows: 1

   * - ``coordinates``
     - Factor
     - Meaning
   * - ``"cartesian"``
     - :math:`1`
     - plane or three-dimensional problems
   * - ``"axisymmetric"``
     - :math:`2\pi r`, :math:`r = x_1`
     - a one-dimensional radial mesh, or an :math:`(r, z)` mesh
   * - ``"spherical"``
     - :math:`4\pi r^2`
     - a one-dimensional radial mesh

With ``"axisymmetric"`` the reaction returned at a boundary node is therefore a
heat flow per unit length (or a force), not a flux density; Example 5.3.2 of
the book, where :math:`Q(R_0) = \pi R_0^2 g_0 = 2\pi\times 10^4` W/m, is
reproduced to machine precision.

Nonlinear problems
------------------

Two schemes are available, both driven by the same residual.

**Newton's method** uses the exact Jacobian.  Every kernel is written in terms
of the forward-mode automatic differentiation type :class:`ADReal`, whose
partial derivatives with respect to the local (element) degrees of freedom are
carried through every arithmetic operation.  The element contributions to
:math:`\partial R_I / \partial U_J` therefore need no hand coding and are exact
for any nonlinearity, including nonlinear boundary conditions.

**Direct (Picard) iteration** freezes the nonlinear coefficients at the
previous iterate.  A kernel asks for the lagged value with
``ctx.coefficient_value(variable)`` — in Newton mode the same call returns the
current AD value, so one kernel serves both schemes.  With
``nonlinear_solver="picard"`` the iteration is that of Section 6.2 of the book,
and the acceleration (relaxation) parameter of Eq. (6.2.15),

.. math::

   \bar{U} = (1-\gamma)\,U^{r} + \gamma\, U^{r-1} , \qquad 0 \le \gamma < 1 ,

is the ``relaxation`` option.  The nonlinear beam problems of Section 7.6 use
:math:`\gamma = 0.35`, and the lid-driven cavity at :math:`Re = 1000` converges
with :math:`\gamma = 0.5`.

**Load stepping** applies the loads in increments, taking the converged
solution of one step as the initial guess of the next.  Objects whose
contribution scales with the load (body forces, tractions, distributed loads,
point loads) are multiplied by the load factor; a Dirichlet condition can be
ramped too by setting ``scale_with_load=True``.  The lid-driven cavity at
:math:`Re = 1000` does not converge from rest with Newton's method but does
converge in a handful of load steps, and the nonlinear beams of Table 7.6.1 use
increments of :math:`\Delta q_0 = 1`.

Convergence is declared when the residual norm drops below
``absolute_tolerance``, or below ``relative_tolerance`` times its initial
value, or when the relative solution increment

.. math::

   \frac{\lVert U^{r+1} - U^{r}\rVert}{\lVert U^{r+1}\rVert} \le \varepsilon

falls below ``step_tolerance``, which is the criterion used in the book.

Time integration
----------------

Transient problems use the :math:`\theta` method.  With
:math:`R_{\text{time}}` the residual of the time-derivative kernels and
:math:`R_{\text{ss}}` the rest,

.. math::

   R_{\text{time}}(U^{n+1}) + \theta\, R_{\text{ss}}(U^{n+1}, t^{n+1})
   + (1-\theta)\, R_{\text{ss}}(U^{n}, t^{n}) = 0 ,

so :math:`\theta = 1` is the backward Euler method, :math:`\theta = 1/2` the
Crank–Nicolson method, and :math:`\theta = 0` the forward Euler method.  A
time-derivative kernel with ``quadrature="nodal"`` gives the lumped capacity
matrix, which in the dual mesh method is simply the measure of the control
domain times the nodal rate.

Solvers and complexity
----------------------

The linear systems are solved with Eigen: a sparse LU factorization by default
(``linear_solver="lu"``), or BiCGSTAB with an incomplete-LU preconditioner, or
the conjugate gradient method for symmetric problems.  The unknowns are
ordered node by node (``dof = node * num_variables + variable``), which keeps
the couplings of a multi-field model close to the diagonal.

Properties worth knowing
------------------------

* **Local conservation.** The discrete balance holds over every control domain
  and, by telescoping, over every union of control domains.
* **Patch test.** A linear field is reproduced exactly on arbitrary distorted
  meshes of all supported element types (``test_patch_test_linear_field``).
* **Exact nodal values in special cases.** For the radial conduction problem of
  Example 5.3.2 the method reproduces the exact solution at the nodes; for the
  mixed beam models the bending moment at the load centre is exact on any mesh
  (Table 7.5.2).
* **Equivalence with the finite element method** on simplicial meshes with
  constant coefficients, as described above.
