Equations written as text
=========================

The ``general_form_PDE`` physics solves partial differential equations that you
write as text, as they are printed, or give by their coefficients.  This page
defines the language of the equations and of their boundary conditions, and
then writes a series of equations, each beside its mathematical form.  It
assumes the object-level API of :doc:`/user_guide/problem_setup`.

The examples start from this mesh of the unit square, with lengths in
:math:`\mathrm{m}`, unless they build their own:

.. code-block:: python

   import dualmesh

   mesh = dualmesh.generate_rectangle_mesh(
       x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=20, num_y_elements=20
   )
   walls = ["left", "right", "bottom", "top"]

.. contents::
   :local:
   :depth: 1

Fields
------

A field is a function of position and time.  On a two-dimensional mesh a field
:math:`u` is :math:`u(x, y, t)`, with :math:`x` and :math:`y` the coordinates
in :math:`\mathrm{m}` and :math:`t` the time in :math:`\mathrm{s}`.  In
axisymmetric coordinates it is :math:`u(r, z, t)`, with :math:`r` the radius and
:math:`z` the axial position.  ``variables`` names the fields:

.. code-block:: python

   variables="u"                     # one scalar field
   variables=["u", "v"]              # two scalar fields
   variables=[{"U": "vector"}, "p"]  # a vector field and a scalar field

A field written without a kind is a scalar, so ``[{"U": "vector"}, "p"]`` and
``[{"U": "vector"}, {"p": "scalar"}]`` are the same.  A vector field
:math:`\mathbf{U}` has one component per dimension, named ``U_x``, ``U_y`` and
``U_z``, or ``U_r`` and ``U_z`` in axisymmetric coordinates.  The components
are the variables of the problem and of the output files.  A field name starts
with a letter and holds letters, digits and underscores.

The general form
----------------

Each equation is reduced, for each of its scalar components, to the form

.. math::

   d_2 \frac{\partial^2 u}{\partial t^2} + d_1 \frac{\partial u}{\partial t}
   + \nabla \cdot \mathbf{F} = s ,

where :math:`d_1` and :math:`d_2` are the factors of the first and second time
derivatives, :math:`\mathbf{F}` is the flux and :math:`s` the source.  With
:math:`d_2 = 0` this is the balance law
:math:`\partial m / \partial t + \nabla \cdot \mathbf{F} = s` of a stored
quantity :math:`m`, whose rate is :math:`d_1 \partial u / \partial t`.  The
text ``lhs = rhs`` stands for ``lhs - rhs = 0``.  The flux is the sum of the
arguments of the divergences of ``lhs - rhs`` that hold a derivative of a
field, each with its sign and factor, and the source :math:`s` is minus the sum
of the other terms that are not time derivatives.  For ``"-laplacian(u) = 1"`` the flux is
:math:`\mathbf{F} = -\nabla u` and the source is :math:`s = 1`.

The equation becomes a conservation law of the model language, the law that
a model of your own of the input also becomes, and the two lower onto the same
objects.  For each component, ``general_form_PDE`` adds a ``parsed_kernel``
with the flux and the source, and, when the equation has them, a
``time_derivative`` kernel with :math:`d_1` and a ``second_time_derivative``
kernel with :math:`d_2`.  Automatic differentiation gives Newton's method the
exact Jacobian.

The notation
------------

In this table :math:`\mathbf{a}` and :math:`\mathbf{b}` are vectors,
:math:`\mathbf{A}` and :math:`\mathbf{B}` tensors, and :math:`n_d` the
dimension of the mesh.

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Mathematics
     - Text
     - Notes
   * - :math:`\partial u / \partial t`
     - ``du/dt``
     - A derivative when ``u`` is a field.
   * - :math:`\partial^2 u / \partial t^2`
     - ``d2u/dt2``
     - A transient run integrates the equation by the generalized-alpha method
       [ChungHulbert1993]_.
   * - :math:`\partial u / \partial x`
     - ``du/dx`` or ``grad_x(u)``
     - Also ``dy`` and ``dz``, and ``dr`` and ``dz`` in axisymmetric
       coordinates.
   * - :math:`\mathbf{n} \cdot \nabla u`
     - ``du/dn``
     - In a boundary condition, with :math:`\mathbf{n}` the outward unit
       normal, which is ``n`` in the text.
   * - :math:`\nabla u`
     - ``grad(u)``
     - A vector.  Of a vector field, the tensor
       :math:`\partial U_i / \partial x_j`, with :math:`x_j` the
       :math:`j`-th coordinate.
   * - :math:`\nabla \cdot \mathbf{a}`, :math:`\nabla \cdot \mathbf{A}`
     - ``div(a)``, ``div(A)``
     - Of a vector, a scalar.  Of a tensor, the vector of the divergences of
       its rows.
   * - :math:`\nabla^2 u`
     - ``laplacian(u)``
     - The same as ``div(grad(u))``.
   * - :math:`\nabla \times \mathbf{a}`
     - ``curl(a)``
     - In two dimensions the curl of a vector is a scalar, and the curl of a
       scalar :math:`u` is the vector
       :math:`(\partial u / \partial y, -\partial u / \partial x)`.
   * - :math:`\mathbf{a} \cdot \mathbf{b}`, :math:`\mathbf{a} \times \mathbf{b}`, :math:`|\mathbf{a}|`
     - ``dot(a, b)``, ``cross(a, b)``, ``norm(a)``
     - ``dot`` also takes tensors: :math:`\mathbf{A} \mathbf{b}`,
       :math:`\mathbf{a} \mathbf{A}` and :math:`\mathbf{A} \mathbf{B}`.
   * - a vector, a tensor
     - ``[a, b]``, ``[[a, b], [c, d]]``
     - One entry per dimension.  A tensor is written row by row.
   * - :math:`\mathbf{I}`, :math:`\mathbf{A}^\mathsf{T}`, :math:`\operatorname{tr} \mathbf{A}`, :math:`\det \mathbf{A}`, :math:`\mathbf{A}^{-1}`
     - ``I``, ``transpose(A)``, ``trace(A)``, ``determinant(A)``, ``inverse(A)``
     -
   * - :math:`\tfrac12 (\mathbf{A} + \mathbf{A}^\mathsf{T})`, :math:`\tfrac12 (\mathbf{A} - \mathbf{A}^\mathsf{T})`
     - ``symmetric(A)``, ``skew(A)``
     -
   * - :math:`\mathbf{A} - \tfrac{1}{n_d} (\operatorname{tr} \mathbf{A}) \mathbf{I}`
     - ``deviatoric(A)``
     -
   * - :math:`\mathbf{A} : \mathbf{B}`, :math:`\mathbf{a} \otimes \mathbf{b}`
     - ``inner(A, B)``, ``outer(a, b)``
     -
   * - functions of scalars
     - ``exp(a)``, ``max(a, b)`` and the other functions of a formula
       (see :ref:`input-formulas`)
     - The arguments are scalars.
   * - a condition
     - ``conditional(c, a, b)``, with ``<``, ``>``, ``<=``,
       ``>=``, ``==``, ``!=``, ``and``, ``or``
     - A comparison is 1 when true and 0 when false.  A condition holds where it is not 0.
   * - :math:`\pi`, :math:`e`, the eigenvalue
     - ``pi``, ``e``, ``eigenvalue``
     - ``eigenvalue`` in an eigenvalue study.

The language is the formula language of the input (see
:doc:`/user_guide/input`).  The equations add the derivatives ``du/dt``,
``du/dx`` and ``du/dn``, and ``div``, ``laplacian``, ``curl``, ``cross`` and
``grad`` of any expression.  Each function has one name.  The text follows
these rules:

* The equations carry no units: the fields, the coordinates, the time and the
  definitions are numbers.  Write each value as its number in one consistent
  system of units.
* You write every multiplication: ``2*u``.  The text ``2u`` is refused.
* ``^`` or ``**`` is a power.  It binds tighter than a minus sign and groups
  from the right, so ``-u^2`` is :math:`-(u^2)` and ``a^b^c`` is
  :math:`a^{(b^c)}`.
* Names are case-sensitive.
* ``d<field>/d<coordinate>`` is a derivative when ``<field>`` is a field.
* Any other name is a definition (see :ref:`equations-definitions`), a
  registered function of :math:`(x, y, z, t)`, a property or an element field,
  used by its name alone.

How an equation is read
-----------------------

The terms of each side are sorted by these rules:

* A term ``div(...)`` or ``laplacian(...)`` whose argument holds a derivative
  of a field belongs to the flux :math:`\mathbf{F}`.  It is integrated by parts,
  so its second derivatives never appear, and its normal component
  :math:`\mathbf{n} \cdot \mathbf{F}` is the quantity that a flux condition
  prescribes.
* Such a term may be multiplied only by a factor that holds no field, no
  coordinate, and no function, property or element field.  The text
  ``k*laplacian(u)`` with a ``k`` that depends on :math:`x` is refused, with
  the advice to write the conservative form ``div(k*grad(u))``.
* A ``div`` of an argument without a derivative, such as ``div(U)`` of a
  velocity, is a first-order term, expanded in place.
* ``du/dt`` gives :math:`d_1`, the factor that multiplies it, which may depend on
  the fields.  ``d2u/dt2`` gives :math:`d_2` in the same way.  An equation holds
  the time derivatives of one field only, and for a vector field you write the
  derivative of the vector, ``dU/dt``.
* A time derivative, a ``div`` or a ``laplacian`` is a term of its own,
  multiplied or divided by coefficients.  Inside a function or another
  operator it is refused.
* Every other term moves to the right side and becomes part of the source
  :math:`s`.
* A term that needs a second derivative outside a divergence, such as
  ``laplacian(laplacian(u))``, is refused.  You write a fourth-order equation
  as two second-order ones, as the Cahn-Hilliard example does.
* A registered function, a property or an element field may not stand inside a
  derivative that is expanded.  Inside the flux of a divergence it may.

The reduction is symbolic.  A derivative of an expression follows the chain
rule, and the vector and tensor operators expand component by component.

Each equation then takes the field it governs.  You may name the field of each
equation with a dict, as ``equations={"u": "...", "v": "..."}``.  Otherwise:

1. An equation with a time derivative governs the field of that derivative.
2. An equation whose flux holds the gradient of one field without an equation
   governs that field.  This rule repeats while it assigns fields.
3. A single equation left over governs the single field left over.

``div(U) = 0`` holds no time derivative and no flux, so in a flow it governs
the pressure, the field left over.  When two equations could govern the same
field, the input is refused with a request to name the field of each
equation.

.. _equations-definitions:

Definitions
-----------

A keyword of ``add_physics`` that is not a parameter is a definition.  A
number is a named constant, and a text a named expression.  A definition may
use the fields and the definitions before it:

.. code-block:: python

   problem = dualmesh.Problem(mesh, method="FEM")
   pde = problem.add_physics(
       "general_form_PDE",
       variables="u",
       k0=2.0,
       grad_u_squared="dot(grad(u), grad(u))",
       k="k0*(1 + grad_u_squared)",
       equations="-div(k*grad(u)) = 1",
   )

A definition is refused when no equation, definition or boundary condition
uses it, so a misspelled parameter is not taken for a definition.  A keyword
too close to the name of a parameter is refused as a misspelling.  A
definition may not take the name of a field, a coordinate of the problem, or
one of the names ``t``, ``n``, ``I``, ``pi``, ``e`` and ``eigenvalue``.

The other parameters of ``general_form_PDE`` are ``units``, the unit of each
field for the output files, ``initial_condition``, the initial value of each
field, and ``block``, the blocks where the equations hold.  With several
fields, ``units`` and ``initial_condition`` are dicts keyed by the field.

Boundary conditions
-------------------

The ``add_boundary_condition`` method of the physics takes a condition written
as text, and ``boundary`` names the side sets or node sets where it holds.

* A field alone on the left prescribes its value: ``"u = 0"``,
  ``"u = 300 + 10*y"``, or ``"U = [0, 0]"`` for a vector field.  The value is
  an expression in the coordinates and :math:`t`, and it may not hold a
  derivative.
* A flux on the left prescribes the normal flux
  :math:`\mathbf{n} \cdot \mathbf{F}` of the equation of a field.  You write the
  flux out with ``d<field>/dn``, as ``"-k*du/dn = 0.5*(u - 1)"``, or name it,
  as ``"normal_flux = 5"``.  The flux written out must equal
  :math:`\mathbf{n} \cdot \mathbf{F}` of the equation, and a condition that
  differs, such as ``"k*du/dn = 1"`` for the equation
  ``"-div(k*grad(u)) = 0"``, is refused with the flux that it should be.
  ``normal_flux`` finds its field by the kind of its value: a vector, as
  ``"normal_flux = [0, -1e6]"``, is the flux of the vector field, and a scalar
  the flux of the scalar field.  When two scalar fields have a flux, you write
  the flux out with ``d<field>/dn``.
* The right side of a flux condition may depend on the fields, so convection,
  radiation and any other nonlinear condition are written as they are printed.

A list of texts adds several conditions on one boundary.  The condition sees
the definitions of ``add_physics``, and a further keyword of the call, a number
or a text, is a definition that the condition uses by its name.  A boundary
without a condition carries a zero flux,
:math:`\mathbf{n} \cdot \mathbf{F} = 0`.  For the ``pde`` of
:ref:`equations-definitions`, with :math:`\mathbf{F} = -k \nabla u`:

.. code-block:: python

   pde.add_boundary_condition("u = 0", boundary=["left", "right"])
   pde.add_boundary_condition("-k*du/dn = 0.5*(u - 1)", boundary="top")
   pde.add_boundary_condition("normal_flux = 0.5*(u - 1)", boundary="bottom")

The last two conditions state the same flux, written out and by its name.
``pde.boundary_flux(boundary)`` returns the integral of
:math:`\mathbf{n} \cdot \mathbf{F}` over a boundary, and takes ``variable``
when the physics has several.  Where the boundary prescribes the value, it
returns the reaction there.

The coefficient form
--------------------

You may give the equation of a scalar field by its coefficients instead.  The
keywords ``m``, ``d``, ``c``, ``alpha``, ``gamma``, ``beta``, ``a`` and ``f``,
written here as :math:`\mathtt{m}` to :math:`\mathtt{f}`, define the equation

.. math::

   \mathtt{m} \frac{\partial^2 u}{\partial t^2}
   + \mathtt{d} \frac{\partial u}{\partial t}
   + \nabla \cdot \left( -\mathtt{c} \, \nabla u - \mathtt{alpha} \, u + \mathtt{gamma} \right)
   + \mathtt{beta} \cdot \nabla u + \mathtt{a} \, u = \mathtt{f} .

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Keyword
     - Name
     - Value
   * - ``m``
     - mass coefficient
     - an expression
   * - ``d``
     - damping coefficient
     - an expression
   * - ``c``
     - diffusion coefficient
     - an expression, a list of one value per dimension (orthotropic), or a
       list of the dimension squared, row by row (a tensor)
   * - ``alpha``
     - conservative flux convection coefficient
     - one value per dimension
   * - ``gamma``
     - conservative flux source
     - one value per dimension
   * - ``beta``
     - convection coefficient
     - one value per dimension
   * - ``a``
     - absorption coefficient
     - an expression
   * - ``f``
     - source term
     - an expression

With several fields, each coefficient is a dict keyed by the field of the
equation.  A value of ``c``, ``alpha``, ``beta`` or ``a`` that is itself a dict
couples the equation to other fields: ``a={"u": {"u": 2.0, "v": -1.0}}`` is the
term :math:`2u - v` of the equation of :math:`u`.  A field takes an equation or
coefficients, and refuses both.

Examples
--------

Poisson's equation
~~~~~~~~~~~~~~~~~~

.. math::

   -\nabla^2 u = 1 \ \text{in } \Omega , \qquad u = 0 \ \text{on } \partial \Omega ,

where :math:`\Omega` is the domain and :math:`\partial \Omega` its boundary.

.. code-block:: python

   problem = dualmesh.Problem(mesh, method="FEM")
   pde = problem.add_physics("general_form_PDE", variables="u", equations="-laplacian(u) = 1")
   pde.add_boundary_condition("u = 0", boundary=walls)
   problem.solve()

By its coefficients the equation is ``c=1, f=1``.

Transient heat conduction with a source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A steel plate with the heat source :math:`s` in :math:`\mathrm{W/m^3}` loses
heat by convection through its top side:

.. math::

   \rho c_p \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + s ,
   \qquad -k \frac{\partial T}{\partial n} = h_\mathrm{c} (T - T_\infty) \ \text{on the top}.

:math:`T` is the temperature in :math:`\mathrm{K}`, :math:`\rho` the density
in :math:`\mathrm{kg/m^3}`, :math:`c_p` the specific heat capacity in
:math:`\mathrm{J/(kg\,K)}`, :math:`k` the thermal conductivity in
:math:`\mathrm{W/(m\,K)}`, :math:`h_\mathrm{c}` the heat transfer coefficient
in :math:`\mathrm{W/(m^2\,K)}` and :math:`T_\infty` the ambient temperature in
:math:`\mathrm{K}`.  :math:`\partial T / \partial n = \mathbf{n} \cdot \nabla T`
is the derivative along the outward unit normal :math:`\mathbf{n}`.

.. code-block:: python

   problem = dualmesh.Problem(mesh, method="DMCDM")
   heat = problem.add_physics(
       "general_form_PDE",
       variables="T",
       rho=7800.0, cp=460.0, k=45.0, s=1.0e6, h_c=25.0, T_inf=293.15,
       equations="rho*cp*dT/dt = div(k*grad(T)) + s",
       initial_condition=293.15,
       units="K",
   )
   heat.add_boundary_condition("T = 293.15", boundary="left")
   heat.add_boundary_condition("-k*dT/dn = h_c*(T - T_inf)", boundary="top")
   problem.solve_transient(end_time=60.0, time_step=10.0)

Advection and diffusion
~~~~~~~~~~~~~~~~~~~~~~~

A species of concentration :math:`c` decays at the rate :math:`R` in
:math:`\mathrm{1/s}` and diffuses with the diffusivity :math:`D` in
:math:`\mathrm{m^2/s}`.  The Poiseuille flow
:math:`\mathbf{v} = (4 y (1 - y), 0)` in :math:`\mathrm{m/s}` carries it down a
channel.  It enters with :math:`c = 1` on the left and leaves on the right.  The
non-conservative form

.. math::

   \mathbf{v} \cdot \nabla c - \nabla \cdot (D \nabla c) + R c = 0

holds only the diffusion inside the divergence, so a side without a condition
has no diffusive flux, :math:`-D \, \partial c / \partial n = 0`.  That is the
outlet, where the species leaves with the flow, and the walls, where the
velocity vanishes:

.. code-block:: python

   problem = dualmesh.Problem(mesh, method="FEM")
   pde = problem.add_physics(
       "general_form_PDE",
       variables="c",
       D=0.05,
       R=2.0,
       equations="dot([4*y*(1 - y), 0], grad(c)) - div(D*grad(c)) + R*c = 0",
   )
   pde.add_boundary_condition("c = 1", boundary="left")
   problem.solve()

The conservative form

.. math::

   \nabla \cdot (-D \nabla c + \mathbf{v} c) + R c = 0

holds the advective flux inside the divergence as well, so a side without a
condition has no total flux, which at the outlet would keep the species in.
The outlet condition therefore sets the total flux to the advective flux, the
flow carrying the species out:

.. code-block:: python

   problem = dualmesh.Problem(mesh, method="FEM")
   pde = problem.add_physics(
       "general_form_PDE",
       variables="c",
       D=0.05,
       R=2.0,
       equations="div(-D*grad(c) + [4*y*(1 - y), 0]*c) + R*c = 0",
   )
   pde.add_boundary_condition("c = 1", boundary="left")
   pde.add_boundary_condition("normal_flux = 4*y*(1 - y)*c", boundary="right")
   problem.solve()

This velocity is divergence free, so the two forms state the same equation.
When :math:`\nabla \cdot \mathbf{v} \neq 0` they differ, and the conservative
form is the balance of the species.

Anisotropic diffusion
~~~~~~~~~~~~~~~~~~~~~

.. math::

   -\nabla \cdot (\mathbf{K} \nabla u) = 1 , \qquad
   \mathbf{K} = \begin{pmatrix} 10 & 1 \\ 1 & 1 \end{pmatrix} ,

where :math:`\mathbf{K}` is a dimensionless diffusivity tensor.

.. code-block:: python

   problem = dualmesh.Problem(mesh, method="FEM")
   pde = problem.add_physics(
       "general_form_PDE", variables="u", equations="-div(dot([[10, 1], [1, 1]], grad(u))) = 1"
   )
   pde.add_boundary_condition("u = 0", boundary=walls)
   problem.solve()

By its coefficients the equation is ``c=[10, 1, 1, 1], f=1``.

Nonlinear diffusion
~~~~~~~~~~~~~~~~~~~

.. math::

   -\nabla \cdot \left( \left( \delta^2 + |\nabla u|^2 \right)^{(\gamma - 2)/2} \nabla u \right) = 1 ,

with the dimensionless exponent :math:`\gamma = 3`.  The small dimensionless
:math:`\delta` keeps the coefficient differentiable where the gradient
vanishes, as it does at the start of Newton's method from :math:`u = 0`:

.. code-block:: python

   problem = dualmesh.Problem(mesh, method="FEM")
   pde = problem.add_physics(
       "general_form_PDE",
       variables="u",
       exponent=3.0,
       delta=1.0e-3,
       k="(delta^2 + dot(grad(u), grad(u)))^((exponent - 2)/2)",
       equations="-div(k*grad(u)) = 1",
   )
   pde.add_boundary_condition("u = 0", boundary=walls)
   problem.solve()

The Cahn-Hilliard equation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The fourth-order equation of the dimensionless concentration :math:`c` of a
mixture,

.. math::

   \frac{\partial c}{\partial t} = \nabla \cdot \left( M \nabla \left( c^3 - c - \ell^2 \nabla^2 c \right) \right) ,

with the mobility :math:`M` in :math:`\mathrm{m^2/s}` and the interface width
:math:`\ell` in :math:`\mathrm{m}`, is written as two second-order equations,
with the dimensionless chemical potential :math:`w` as a second field:

.. math::

   \frac{\partial c}{\partial t} = \nabla \cdot (M \nabla w) , \qquad
   w = c^3 - c - \ell^2 \nabla^2 c .

.. code-block:: python

   problem = dualmesh.Problem(mesh, method="FEM")
   mixture = problem.add_physics(
       "general_form_PDE",
       variables=["c", "w"],
       M=1.0,
       ell=0.03,
       equations=[
           "dc/dt = div(M*grad(w))",
           "w = c^3 - c - ell^2*laplacian(c)",
       ],
       initial_condition={"c": "0.1*cos(2*pi*x)*cos(2*pi*y)"},
   )
   problem.solve_transient(end_time=1.0e-3, time_step=1.0e-4)

The first equation holds the time derivative of :math:`c`, and the second
governs :math:`w`, the field left over.

Two-group neutron diffusion: an eigenvalue problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fast neutron flux :math:`\phi_1` and the thermal neutron flux
:math:`\phi_2` of a bare reactor core, in :math:`\mathrm{1/(m^2\,s)}`, satisfy

.. math::

   -\nabla \cdot (D_1 \nabla \phi_1) + (\Sigma_{a1} + \Sigma_{12}) \phi_1 = \frac{1}{k_\mathrm{eff}} P_2 \phi_2 , \qquad
   -\nabla \cdot (D_2 \nabla \phi_2) + \Sigma_{a2} \phi_2 = \Sigma_{12} \phi_1 ,

with the diffusion coefficients :math:`D_1` and :math:`D_2` in
:math:`\mathrm{m}`, the absorption cross sections :math:`\Sigma_{a1}` and
:math:`\Sigma_{a2}`, the scattering cross section :math:`\Sigma_{12}` from the
fast to the thermal group, and the fission production cross section
:math:`P_2` of the thermal group, all in :math:`\mathrm{1/m}`.  The effective
multiplication factor :math:`k_\mathrm{eff}` is dimensionless.  The symbol
``eigenvalue`` is :math:`1/k_\mathrm{eff}`, so the smallest eigenvalue gives
the largest :math:`k_\mathrm{eff}` (see :ref:`tutorial-criticality`):

.. code-block:: python

   problem = dualmesh.Problem(mesh, method="FEM")
   neutrons = problem.add_physics(
       "general_form_PDE",
       variables=["phi1", "phi2"],
       D1=0.015, D2=0.004, Sigma_a1=1.0, Sigma_a2=8.0, Sigma_12=2.0, nu_Sigma_f2=13.5,
       equations=[
           "-div(D1*grad(phi1)) + (Sigma_a1 + Sigma_12)*phi1 = eigenvalue*nu_Sigma_f2*phi2",
           "-div(D2*grad(phi2)) + Sigma_a2*phi2 = Sigma_12*phi1",
       ],
   )
   neutrons.add_boundary_condition(["phi1 = 0", "phi2 = 0"], boundary=walls)
   k_eff = 1.0 / problem.solve_eigenvalue(num_modes=1).eigenvalues[0]

The script ``examples/reactor_criticality.py`` solves the same core on a
square of side :math:`2\ \mathrm{m}` through the input, with one model of
your own per group, and compares :math:`k_\mathrm{eff}` with the exact value
of two-group theory.

Axisymmetric heat conduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In axisymmetric coordinates the coordinates are :math:`r` and :math:`z`, and
the divergence of a vector :math:`\mathbf{a}` carries the factor of the radius,

.. math::

   \nabla \cdot \mathbf{a} = \frac{1}{r} \frac{\partial (r a_r)}{\partial r}
   + \frac{\partial a_z}{\partial z} ,

with no change to the text.  This cylinder of radius :math:`0.01\ \mathrm{m}`
has a heat source that falls to zero at its surface, which is held at
:math:`600\ \mathrm{K}`:

.. code-block:: python

   cylinder = dualmesh.generate_rectangle_mesh(
       x_min=0.0, x_max=0.01, y_min=0.0, y_max=0.02, num_x_elements=10, num_y_elements=20
   )
   problem = dualmesh.Problem(cylinder, method="FEM", coordinates="axisymmetric")
   heat = problem.add_physics(
       "general_form_PDE",
       variables="T",
       k=3.0,
       equations="-div(k*grad(T)) = 3.0e8*(1 - (r/0.01)^2)",
   )
   heat.add_boundary_condition("T = 600", boundary="right")
   problem.solve()

The gradient of a vector field and the divergence of a tensor have hoop terms
in axisymmetric coordinates, which the language does not write, so it refuses
them there.  The physics with an axisymmetric formulation, such as
``solid_mechanics``, carry those terms.

Linear elasticity: a vector field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \nabla \cdot \boldsymbol{\sigma} + \mathbf{b} = \mathbf{0} , \qquad
   \boldsymbol{\sigma} = \lambda (\operatorname{tr} \boldsymbol{\varepsilon}) \mathbf{I} + 2 G \boldsymbol{\varepsilon} , \qquad
   \boldsymbol{\varepsilon} = \tfrac12 \left( \nabla \mathbf{U} + \nabla \mathbf{U}^\mathsf{T} \right) ,

with the displacement :math:`\mathbf{U}` in :math:`\mathrm{m}`, the Cauchy
stress :math:`\boldsymbol{\sigma}`, the small strain
:math:`\boldsymbol{\varepsilon}`, the first Lame parameter :math:`\lambda` and
the shear modulus :math:`G` in :math:`\mathrm{Pa}`, and the body force
:math:`\mathbf{b}` in :math:`\mathrm{N/m^3}`, here the weight of steel.  In
plane strain:

.. code-block:: python

   problem = dualmesh.Problem(mesh, method="FEM")
   solid = problem.add_physics(
       "general_form_PDE",
       variables=[{"U": "vector"}],
       lame_lambda=1.15e11,
       G=7.7e10,
       strain="symmetric(grad(U))",
       stress="lame_lambda*trace(strain)*I + 2*G*strain",
       equations="div(stress) + [0, -7800*9.81] = 0",
   )
   solid.add_boundary_condition(["U_x = 0", "U_y = 0"], boundary="left")
   solid.add_boundary_condition("normal_flux = [0, -1.0e6]", boundary="right")
   problem.solve()

The text puts ``div(stress)`` on the left with a plus sign, so the flux of this
equation is :math:`\boldsymbol{\sigma}`, and ``normal_flux`` is the traction
:math:`\boldsymbol{\sigma} \cdot \mathbf{n}`, here a downward load of
:math:`10^6\ \mathrm{Pa}`.  The ``solid_mechanics`` physics solves these
equations and adds finite strain and contact.

Incompressible flow
~~~~~~~~~~~~~~~~~~~

The Navier-Stokes equations of the velocity :math:`\mathbf{U}` in
:math:`\mathrm{m/s}` and the pressure :math:`p` in :math:`\mathrm{Pa}`, with
the density :math:`\rho` and the dynamic viscosity :math:`\mu` in
:math:`\mathrm{Pa\,s}`, are written as printed:

.. code-block:: text

   variables=[{"U": "vector"}, "p"]
   equations=[
       "rho*(dU/dt + dot(grad(U), U)) = div(-p*I + mu*(grad(U) + transpose(grad(U))))",
       "div(U) = 0",
   ]

``general_form_PDE`` solves exactly what is written.  With the same
interpolation for the velocity and the pressure, this saddle-point system needs
a stabilization [BrezziPitkaranta1984]_ that the text does not hold.  The
``fluid_dynamics`` physics adds it, so solve a flow with that physics.
