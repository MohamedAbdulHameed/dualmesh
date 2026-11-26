The conservation form and the dual mesh
=======================================

Every equation that the code solves is a balance law, and the dual mesh control
domain method (DMCDM) of [Reddy2024]_ imposes that balance on the control
domains of a dual mesh.  This page states the balance law, the two meshes, the
local conservation that follows from them and the coordinate systems.
:doc:`finite_volume` gives the discrete equations, and :doc:`elements` the
elements and the construction of the dual mesh inside them.

.. contents::
   :local:
   :depth: 1

The balance law
---------------

Let :math:`u` be the unknown field of an equation, a function of the position
:math:`\mathbf{x}` (m) and the time :math:`t` (s) in the domain :math:`\Omega`.
The equation is the balance law

.. math::
   :label: balance-law

   \frac{\partial m}{\partial t} + \nabla \cdot \mathbf{F} = s
   \quad \text{in } \Omega ,

with :math:`m` the storage, the amount of the balanced quantity per unit volume,
:math:`\mathbf{F}` its flux, the amount that crosses a unit area per unit time,
and :math:`s` its source, the amount produced per unit volume and unit time.
For heat, :math:`m` is in :math:`\mathrm{J/m^3}`, :math:`\mathbf{F}` in
:math:`\mathrm{W/m^2}` and :math:`s` in :math:`\mathrm{W/m^3}`.

The flux is the physical flux.  On a surface with the unit normal
:math:`\mathbf{n}`, the product :math:`\mathbf{F} \cdot \mathbf{n}` is the rate
per unit area at which the quantity crosses the surface in the direction of
:math:`\mathbf{n}`.  On the boundary :math:`\partial\Omega`, with
:math:`\mathbf{n}` the outward unit normal, it is the rate at which the quantity
leaves the body.

The storage, the flux and the source may depend on :math:`u`, on
:math:`\nabla u`, on :math:`\mathbf{x}`, on :math:`t` and on the other unknowns
of the problem, so one form covers nonlinear and coupled systems.  The code
writes the rate of the storage as

.. math::

   \frac{\partial m}{\partial t} = c \, \frac{\partial u}{\partial t} ,
   \qquad c = \frac{\partial m}{\partial u} ,

with :math:`c` the capacity of the equation.  For heat conduction with the
density :math:`\rho` (:math:`\mathrm{kg/m^3}`) and the specific heat capacity
:math:`c_p` (:math:`\mathrm{J/(kg\,K)}`), the capacity is :math:`\rho c_p`.

The table gives the balance law of three equations.  In it, :math:`T` is the
temperature (K), :math:`k` the thermal conductivity
(:math:`\mathrm{W/(m\,K)}`) or, for a general field, its diffusivity,
:math:`\mathbf{v}` a velocity (m/s), :math:`\mathbf{u}` the displacement (m)
with the components :math:`u_i`, :math:`\boldsymbol{\sigma}` the Cauchy stress
(Pa) and :math:`\mathbf{f}` a body force per unit volume
(:math:`\mathrm{N/m^3}`) with the components :math:`f_i`.

.. list-table::
   :header-rows: 1
   :widths: 28 22 28 22

   * - Equation
     - Storage :math:`m`
     - Flux :math:`\mathbf{F}`
     - Source :math:`s`
   * - Heat conduction, for :math:`T`
     - :math:`\rho c_p T`
     - :math:`-k \nabla T`
     - the heat source
   * - Advection and diffusion of :math:`u`, conservative form
     - :math:`u`
     - :math:`\mathbf{v} u - k \nabla u`
     - the source
   * - Linear momentum of a solid, component :math:`i`
     - :math:`\rho \, \partial u_i / \partial t`
     - :math:`-(\sigma_{i1}, \sigma_{i2}, \sigma_{i3})`
     - :math:`f_i`

Primary and secondary variables
-------------------------------

The normal flux :math:`\mathbf{F} \cdot \mathbf{n}` is the secondary variable of
the equation, and :math:`u` is its primary variable.  At every point of
:math:`\partial\Omega` a boundary condition gives one of the two.

* An essential (Dirichlet) condition gives the value of :math:`u`.
* A natural (Neumann) condition gives the normal flux: the heat flow out of the
  body, or the traction with the opposite sign.
* A Robin condition gives the normal flux as a function of :math:`u`, as a
  convective heat loss does.

The integral form
-----------------

Integrating :eq:`balance-law` over a region :math:`V` with the boundary
:math:`\partial V` and the outward unit normal :math:`\mathbf{n}`, and turning
the divergence into a surface integral with the divergence theorem, gives

.. math::
   :label: balance-integral

   \int_V \frac{\partial m}{\partial t} \, \mathrm{d}V
   + \oint_{\partial V} \mathbf{F} \cdot \mathbf{n} \, \mathrm{d}S
   = \int_V s \, \mathrm{d}V .

The rate of change of the amount in :math:`V`, plus what leaves through its
boundary, equals what its source produces.  DMCDM imposes :eq:`balance-integral`
on every control domain of the dual mesh, which the next section defines.

The primal mesh and the dual mesh
---------------------------------

The **primal mesh** is a mesh of Lagrange finite elements.  On it, the field is
interpolated from its values :math:`U_a` at the nodes :math:`a`,

.. math::

   u(\mathbf{x}) \approx \sum_a U_a \, N_a(\mathbf{x}) ,

with :math:`N_a` the shape function of node :math:`a`.  Each element is the
image of a reference element under its isoparametric map
:math:`\mathbf{x}(\boldsymbol{\xi}) = \sum_a \mathbf{x}_a N_a(\boldsymbol{\xi})`,
with :math:`\boldsymbol{\xi}` the reference coordinates and
:math:`\mathbf{x}_a` the position of node :math:`a`.

The **dual mesh** holds one control domain :math:`V_I` around every node
:math:`I`.  It is the median dual.  Inside one element, the part of
:math:`V_I` is the region bounded by

* the node :math:`I`,
* the midpoints of the element edges that meet at :math:`I`,
* the centroids of the element faces that meet at :math:`I`, in three
  dimensions,
* the centroid of the element.

This construction gives the following control domains.

* In one dimension, the control domain of an interior node joins the halves of
  the two elements beside it.
* On a mesh of rectangles, the control domains are the rectangles that the
  bisectors of the element sides bound.
* On triangles and general quadrilaterals, the control domain boundaries join
  the midpoints of the element sides to the element centroids.
* At a node on :math:`\partial\Omega`, the domain boundary truncates the control
  domain: in one dimension to a half, and on a rectangular mesh to a half on a
  side and to a quarter at a corner.

The interface between the control domains of the two end nodes of an element
edge passes through the midpoint of the edge, the centroids of the faces that
share the edge and the centroid of the element.  :doc:`elements` gives the
construction for every element type, including the quadratic ones.

Local conservation
------------------

Two control domains meet at each interior interface, and their outward normals
there are opposite.  The flux through the interface therefore leaves one
control domain and enters the other with the same value.  When you add
:eq:`balance-integral` over any union of control domains, the interior
interfaces cancel and only the flux through the boundary of the union remains.
The discrete balance therefore holds over every control domain, over every
union of control domains and over the whole domain.

.. _theory-user-models:

Models of your own
------------------

A model of the ``models`` section of the input (:ref:`input-models`) is the
balance law :eq:`balance-law` of one scalar unknown :math:`u`, with internal
states :math:`q_1, \dots, q_n`. Each state is a number at each integration point
with the rate law

.. math::
   :label: state-rate

   \frac{\mathrm{d}q_i}{\mathrm{d}t} = g_i(\mathbf{q}, u, \nabla u, \mathbf{x}, t) ,

with :math:`g_i` the rate of :math:`q_i` in the unit of :math:`q_i` per s and
:math:`\mathbf{q}` the vector of the states. The storage :math:`m`, the flux
:math:`\mathbf{F}` and the source :math:`s` may read the states. The input
writes the properties of each region into the terms, so each term is a
formula of the fields, their gradients, the states, the position and the time.
The solve differentiates the rates symbolically for the Newton method of the
states, and automatic differentiation gives the Jacobian of the whole problem.

The storage term
~~~~~~~~~~~~~~~~

A step from the time :math:`t^n` to :math:`t^{n+1} = t^n + \Delta t` takes
the rate of the storage as the change of the storage itself:

.. math::
   :label: storage-difference

   \frac{\partial m}{\partial t} \approx
   \frac{m(u^{n+1}, \mathbf{q}^{n+1}) - m(u^n, \mathbf{q}^n)}{\Delta t} ,

with the superscripts naming the time of each value and :math:`\Delta t` the
time step in s. The fluxes and the sources are taken at :math:`t^{n+1}`, as
the backward Euler method takes them. Summed over all control domains, or
over all test functions of the finite element method (FEM), the interior
fluxes cancel (see `Local conservation`_), and
:eq:`storage-difference` gives

.. math::

   \int_\Omega m^{n+1} \, \mathrm{d}V = \int_\Omega m^n \, \mathrm{d}V
   + \Delta t \left( \int_\Omega s^{n+1} \, \mathrm{d}V
   - \oint_{\partial\Omega} \mathbf{F}^{n+1} \cdot \mathbf{n} \, \mathrm{d}S \right) .

The amount stored therefore changes by what the sources and the boundary bring
in each step, for any storage, linear or not. The capacity form
:math:`c\,\partial u/\partial t` of `The balance law`_, with :math:`c` taken at
:math:`t^{n+1}`, gains or loses a part of the amount in each step where
:math:`c` changes, so a model takes the difference
:eq:`storage-difference`. For a storage :math:`m = c\,u` with a constant
capacity :math:`c` the two forms are equal, and a model of heat conduction
gives the temperatures of ``heat_conduction``.

On a mesh of linear elements, the storage term ``time_derivative`` is lumped at
the nodes, as the heat capacity of the ``heat_conduction`` physics is: each node takes the storage of its
control domain, or the integral of its shape function under FEM, at its own
value. The system of each step then keeps the maximum principle of diffusion
on a mesh without obtuse angles, so a concentration stays between its bounds.
A term that carries the unknown, a flux such as :math:`\mathbf{v} u` or a source
such as :math:`\mathbf{v} \cdot \nabla u` with a velocity :math:`\mathbf{v}` in
m/s, keeps the Gauss rule, since a lumped storage loses the phase accuracy of a
transport.  An eigenvalue study takes the Gauss rule on every mesh, since the
lumping serves time stepping.

The states
~~~~~~~~~~

Each integration point keeps its states from step to step. A step integrates :eq:`state-rate` by the
backward Euler method,

.. math::
   :label: state-step

   r_i(\mathbf{q}^{n+1}) = \frac{q_i^{n+1} - q_i^n}{\Delta t}
   - g_i(\mathbf{q}^{n+1}, u^{n+1}, \nabla u^{n+1}, \mathbf{x}, t^{n+1}) = 0 ,

with :math:`r_i` the residual of state :math:`i`. The point solves
:eq:`state-step` by the Newton method from :math:`\mathbf{q}^n`, with the
Jacobian :math:`J_{ij} = \delta_{ij}/\Delta t - \partial g_i/\partial q_j`,
where :math:`\delta_{ij}` is 1 for :math:`i = j` and 0 otherwise. A step that
does not lower the norm of :math:`\mathbf{r}` is halved. The derivative of the
states with respect to the unknowns of the problem follows from the implicit
function theorem,
:math:`\partial \mathbf{q}^{n+1}/\partial u = -J^{-1}\,\partial \mathbf{r}/\partial u`,
so the Newton method of the problem converges at its quadratic rate.

The backward Euler method is the method of the transient problems of the
input, so the unknown and the states advance with the same method. It is
stable for any step, and it holds a state at the root of a fast rate, such as
the local equilibrium of a trap whose rates exceed :math:`1/\Delta t` by far.
Then :math:`q_i^{n+1}` tends to the root of :math:`g_i`, and the storage
:eq:`storage-difference` of a linear trap :math:`m = c + q` with
:math:`q = K c` becomes :math:`(1 + K)\,c`. The concentration :math:`c` then
diffuses with the effective diffusivity :math:`D/(1 + K)` of
[Oriani1970]_, with :math:`D` the diffusivity in :math:`\mathrm{m^2/s}` and
:math:`K` the dimensionless ratio of the trapping and release rates.

A steady problem has no time derivative, so it holds each state at the root
of its rates, :math:`\mathbf{g}(\mathbf{q}) = \mathbf{0}`, the equilibrium of
each point. The Newton method of the point then has the Jacobian
:math:`-\partial g_i/\partial q_j`.

The storage at :math:`t^n` reads the states at :math:`t^n`, which the point
keeps, and the fields at :math:`t^n`. Under the lumped rule the states live at
the nodes, where the storage is evaluated.

Coordinate systems
------------------

The integrals of :eq:`balance-integral` carry a factor that depends on the
coordinate system.  In axisymmetric and spherical coordinates, the radius
:math:`r` (m) is the first coordinate :math:`x_1`, and the mesh lies in
:math:`x_1 \ge 0`.

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Coordinate system
     - Factor
     - Mesh
   * - ``cartesian``
     - :math:`1`
     - a mesh in one, two or three dimensions
   * - ``axisymmetric``
     - :math:`2\pi r`
     - a radial mesh in one dimension, or an :math:`(r, z)` mesh in two
   * - ``spherical``
     - :math:`4\pi r^2`
     - a radial mesh in one dimension

You choose the system with the ``coordinates`` argument of
:class:`dualmesh.Problem` or the ``coordinate_system`` key of the mesh section
of the input.  With the factor, an integral over an :math:`(r, z)` mesh is the
integral over the body of revolution, and an integral over a radial mesh in
axisymmetric coordinates is the integral per unit length of the cylinder.  The
heat that leaves a cylinder through its surface is then in W on an
:math:`(r, z)` mesh and in W/m on a radial mesh.

:doc:`/verification` gives the cases that check these properties.
