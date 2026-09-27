Foundations: the conservation form and the dual mesh
====================================================

This chapter introduces the form in which every equation in the library is
written, the two meshes the method keeps, and the discretisation itself.  It
also states, for comparison on identical input, what the Galerkin finite
element method does with the same problem.  The quadrature rules that evaluate
the integrals, and the elements that generate the dual mesh, are the subject of
:doc:`elements`.

.. contents::
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

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
