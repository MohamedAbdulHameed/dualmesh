From paper to code
==================

``general_form_PDE`` solves equations written as they are printed.  This page
defines the language of the equations and of their boundary conditions, and
then writes a series of equations, from Poisson's equation to the
Cahn-Hilliard equation and linear elasticity, each beside its mathematical
form.

Every example starts from a mesh of the unit square:

.. code-block:: python

   import dualmesh as dm

   mesh = dm.generate_rectangle_mesh(
       x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=20, num_y_elements=20
   )
   walls = ["left", "right", "bottom", "top"]

The fields and the coordinates
------------------------------

A field is a function of position and time: :math:`u = u(x, y, z, t)` in
three dimensions, :math:`u(x, y, t)` on a two-dimensional mesh, and
:math:`u(r, z, t)` in axisymmetric coordinates, where :math:`r` is the radius
and :math:`z` the axial position.  The fields are named in ``variables``:

.. code-block:: python

   variables="u"                                # one scalar field
   variables=["u", "v"]                         # two scalar fields
   variables=[{"U": "vector"}, "p"]             # a vector field and a scalar field

A field written without a kind is scalar, so ``[{"U": "vector"}, "p"]`` and
``[{"U": "vector"}, {"p": "scalar"}]`` are the same.  A vector field
:math:`\mathbf{U}` has one component per dimension, named
``U_x``, ``U_y`` and ``U_z`` (``U_r`` and ``U_z`` in axisymmetric
coordinates).  The components are the unknowns that the output files hold.

The notation
------------

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Mathematics
     - Text
     - Notes
   * - :math:`\partial u / \partial t`
     - ``du/dt``
     - a derivative whenever ``u`` is a field
   * - :math:`\partial^2 u / \partial t^2`
     - ``d2u/dt2``
     - needs the second-order time integrator (not yet available)
   * - :math:`\partial u / \partial x`
     - ``du/dx`` or ``grad_x(u)``
     - also ``dy``, ``dz``; ``dr``, ``dz`` in axisymmetric coordinates
   * - :math:`\nabla u`
     - ``grad(u)``
     - a vector; of a vector field, the tensor :math:`\partial U_i / \partial x_j`
   * - :math:`\nabla \cdot \mathbf{F}`
     - ``div(F)``
     - of a vector, a scalar; of a tensor, the vector of the divergences of its rows
   * - :math:`\nabla^2 u`
     - ``laplacian(u)``
     - the same as ``div(grad(u))``
   * - :math:`\nabla \times \mathbf{F}`
     - ``curl(F)``
     - in two dimensions, a scalar
   * - :math:`\mathbf{a} \cdot \mathbf{b}`, :math:`\mathbf{a} \times \mathbf{b}`, :math:`|\mathbf{a}|`
     - ``dot(a, b)``, ``cross(a, b)``, ``norm(a)``
     - ``dot`` also takes tensors: :math:`\mathbf{A} \mathbf{b}`, :math:`\mathbf{a} \mathbf{A}`, :math:`\mathbf{A} \mathbf{B}`
   * - a vector, a tensor
     - ``[a, b]``, ``[[a, b], [c, d]]``
     - one entry per dimension; a tensor row by row
   * - :math:`\mathbf{I}`, :math:`\mathbf{A}^\mathsf{T}`, :math:`\operatorname{tr} \mathbf{A}`, :math:`\det \mathbf{A}`, :math:`\mathbf{A}^{-1}`
     - ``I``, ``transpose(A)``, ``trace(A)``, ``det(A)``, ``inverse(A)``
     -
   * - :math:`\tfrac12 (\mathbf{A} + \mathbf{A}^\mathsf{T})`, :math:`\tfrac12 (\mathbf{A} - \mathbf{A}^\mathsf{T})`
     - ``sym(A)``, ``skew(A)``
     -
   * - :math:`\mathbf{A} - \tfrac1n (\operatorname{tr} \mathbf{A}) \mathbf{I}`
     - ``dev(A)``
     - :math:`n` is the dimension of the mesh
   * - :math:`\mathbf{A} : \mathbf{B}`, :math:`\mathbf{a} \otimes \mathbf{b}`
     - ``ddot(A, B)``, ``outer(a, b)``
     -
   * - functions
     - ``sin``, ``cos``, ``tan``, ``asin``, ``acos``, ``atan``, ``sinh``,
       ``cosh``, ``tanh``, ``exp``, ``log``, ``log10``, ``sqrt``, ``abs``,
       ``erf``, ``sign``, ``floor``, ``ceil``, ``min``, ``max``, ``atan2``,
       ``hypot``, ``pow``
     -
   * - a condition
     - ``if(c, a, b)`` with ``<``, ``>``, ``<=``, ``>=``, ``==``, ``!=``, ``and``, ``or``
     - a comparison is 1 when true and 0 when false
   * - :math:`\pi`, :math:`e`, the eigenvalue
     - ``pi``, ``e``, ``eigenvalue``
     - ``eigenvalue`` in an eigenvalue study

The rules of the text are those of mathematics, written out:

* multiplication is always written: ``2*u``, never ``2u``;
* ``^`` (or ``**``) is a power, binds tighter than a minus sign and groups
  from the right: ``-u^2`` is :math:`-(u^2)` and ``a^b^c`` is
  :math:`a^{(b^c)}`;
* names are case-sensitive;
* ``d<field>/d<coordinate>`` is a derivative only when ``<field>`` is a field,
  so a definition may not be named ``du`` when ``u`` is a field;
* any other name is a definition, a registered function of
  :math:`(x, y, z, t)`, a property or an element field, used by its name alone.

How an equation is read
-----------------------

Each equation is reduced to the general form of each of its scalar
components,

.. math::

   d \, \frac{\partial u}{\partial t} + \nabla \cdot \boldsymbol{\Gamma} = f ,

by these rules:

* everything inside ``div(...)``, or ``laplacian(...)``, that holds a
  derivative of a field is the flux :math:`\boldsymbol{\Gamma}`.  It is
  integrated by parts, so that the second derivatives never appear, and its
  normal component :math:`\mathbf{n} \cdot \boldsymbol{\Gamma}` is what a
  flux condition prescribes;
* a ``div`` term may be multiplied only by a factor that is constant in space.
  ``k(x)*laplacian(u)`` is refused, with the advice to write the conservative
  form ``div(k*grad(u))``;
* a ``div`` of an argument without derivatives, such as ``div(U)`` of a
  velocity, is a first-order term and is expanded in place;
* ``du/dt`` gives :math:`d`, with the factor that multiplies it, which may
  depend on the fields;
* everything else is moved to the right side and is the source :math:`f`;
* a term that would need a second derivative outside a divergence, such as
  ``laplacian(laplacian(u))``, is refused: a fourth-order equation is written
  as two second-order ones, as the Cahn-Hilliard example below does.

The reduction is symbolic: a derivative of an expression follows the chain
rule, the vector and tensor operators are expanded component by component, and
the result is compiled once and differentiated automatically, so Newton's
method has the exact Jacobian.

Each equation is matched to the field it governs: the field whose time
derivative it holds, then the field whose second derivative it holds, and a
field left over takes the equation left over.  ``div(U) = 0``, which holds no
pressure, is thus the equation of the pressure.  When two equations could
govern the same field, the input is refused with a request to name the field
of each equation, as ``equations={"u": "...", "v": "..."}``.

Definitions
-----------

A keyword of ``add_physics`` that is not a parameter is a definition: a
number is a named constant and a text a named expression.  A definition may
use the fields and the definitions before it:

.. code-block:: python

   pde = problem.add_physics(
       "general_form_PDE",
       variables="u",
       k0=2.0,
       grad_u_squared="dot(grad(u), grad(u))",
       k="k0*(1 + grad_u_squared)",
       equations="-div(k*grad(u)) = 1",
   )

A definition that no equation, definition or boundary condition uses is
refused, so that a misspelled parameter is not taken silently for a
definition.  A definition may not take the name of a field, a parameter or a
name of the language.

Boundary conditions
-------------------

A boundary condition is written the same way, on the side sets or node sets
named by ``boundary``.  The normal :math:`\mathbf{n}` points out of the
body.

* A field alone on the left prescribes its value: ``"u = 0"``,
  ``"u = 300 + 10*y"``.
* The normal flux on the left prescribes the flux through the boundary. It is
  written out, ``"-k*du/dn = h*(u - 1)"``, where ``du/dn`` is
  :math:`\mathbf{n} \cdot \nabla u`, or by its name, ``"normal_flux = 5"``,
  which is :math:`\mathbf{n} \cdot \boldsymbol{\Gamma}` of the equation of the
  field.  With several fields, the value tells the field: a vector, as
  ``"normal_flux = [0, -1e6]"``, is the flux of the vector field and a scalar
  that of the scalar field; when two fields of one kind have a flux, the
  flux is written out with ``d<field>/dn``.  A flux written out is compared with :math:`\mathbf{n} \cdot
  \boldsymbol{\Gamma}`, and a condition that differs from it, ``"k*du/dn =
  1"`` for the equation ``-div(k*grad(u)) = 0``, is refused with the flux
  that it should be.
* The right side may depend on the fields, so convection, radiation and any
  nonlinear condition are written as they are printed.

A boundary without a condition has a zero flux.

.. code-block:: python

   pde.add_boundary_condition("u = 0", boundary=["left", "right"])
   pde.add_boundary_condition("-k*du/dn = 5.67e-8*0.8*(u^4 - 1)", boundary="top")

The coefficient form
--------------------

An equation of a scalar field may instead be given by its coefficients, as in
the coefficient form of COMSOL and the ``specifyCoefficients`` of MATLAB:

.. math::

   d \, \frac{\partial u}{\partial t}
   + \nabla \cdot \left( -c \, \nabla u - \boldsymbol{\alpha} \, u + \boldsymbol{\gamma} \right)
   + \boldsymbol{\beta} \cdot \nabla u + a \, u = f .

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Keyword
     - Name
     - Value
   * - ``d``
     - damping (or mass) coefficient
     - an expression
   * - ``c``
     - diffusion coefficient
     - an expression, one per dimension (orthotropic), or the dimension squared row by row (a tensor)
   * - ``alpha``
     - conservative flux convection coefficient
     - one per dimension
   * - ``gamma``
     - conservative flux source
     - one per dimension
   * - ``beta``
     - convection coefficient
     - one per dimension
   * - ``a``
     - absorption coefficient
     - an expression
   * - ``f``
     - source term
     - an expression

With several fields, each coefficient is a dict keyed by the field of the
equation, and a value that is itself a dict couples the equation to other
fields: ``a={"u": {"u": 2.0, "v": -1.0}}`` is the term :math:`2u - v` of the
equation of :math:`u`.

Examples
--------

Poisson's equation
^^^^^^^^^^^^^^^^^^

.. math::

   -\nabla^2 u = 1 \ \text{in } \Omega , \qquad u = 0 \ \text{on } \partial \Omega .

.. code-block:: python

   problem = dm.Problem(mesh, method="FEM")
   pde = problem.add_physics("general_form_PDE", variables="u", equations="-laplacian(u) = 1")
   pde.add_boundary_condition("u = 0", boundary=walls)
   problem.solve()

The same by its coefficients: ``c=1, f=1``.

Transient heat conduction with a source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \rho c_p \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + q ,
   \qquad -k \frac{\partial T}{\partial n} = h (T - T_\infty) \ \text{on the top}.

.. code-block:: python

   problem = dm.Problem(mesh, method="DMCDM")
   heat = problem.add_physics(
       "general_form_PDE",
       variables="T",
       rho=7800.0, cp=460.0, k=45.0, q=1.0e6, h=25.0, T_inf=293.0,
       equations="rho*cp*dT/dt = div(k*grad(T)) + q",
       initial_condition=293.0,
   )
   heat.add_boundary_condition("T = 293", boundary="left")
   heat.add_boundary_condition("-k*dT/dn = h*(T - T_inf)", boundary="top")
   problem.solve_transient(end_time=60.0, time_step=10.0)

Advection-diffusion
^^^^^^^^^^^^^^^^^^^

A species that decays at the rate :math:`k` is carried down a channel by the
Poiseuille flow :math:`\mathbf{v} = (4 y (1 - y), 0)`, enters with
:math:`c = 1` and leaves at the right.  The non-conservative form,

.. math::

   \mathbf{v} \cdot \nabla c - \nabla \cdot (D \nabla c) + k c = 0 ,

holds only the diffusion inside the divergence, so a side without a
condition has no diffusive flux, :math:`-D \, \partial c / \partial n = 0`:
that is the outlet, where the species leaves with the flow, and the walls,
where the velocity vanishes:

.. code-block:: python

   problem = dm.Problem(mesh, method="FEM")
   pde = problem.add_physics(
       "general_form_PDE",
       variables="c",
       D=0.05,
       k=2.0,
       equations="dot([4*y*(1 - y), 0], grad(c)) - div(D*grad(c)) + k*c = 0",
   )
   pde.add_boundary_condition("c = 1", boundary="left")
   problem.solve()

The conservative form,

.. math::

   \nabla \cdot (-D \nabla c + \mathbf{v} c) + k c = 0 ,

holds the advective flux inside the divergence as well, so a side without a
condition has no *total* flux, which at the outlet would stop the species
from leaving.  The outlet is therefore given the advective flux, the flow
carrying the species out:

.. code-block:: python

   problem = dm.Problem(mesh, method="FEM")
   pde = problem.add_physics(
       "general_form_PDE",
       variables="c",
       D=0.05,
       k=2.0,
       equations="div(-D*grad(c) + [4*y*(1 - y), 0]*c) + k*c = 0",
   )
   pde.add_boundary_condition("c = 1", boundary="left")
   pde.add_boundary_condition("normal_flux = 4*y*(1 - y)*c", boundary="right")
   problem.solve()

The velocity is divergence free, so the two forms are the same equation, and
they give the same solution to round-off.  They differ when
:math:`\nabla \cdot \mathbf{v} \neq 0`, as in a compressible flow, where
only the conservative form conserves the species.

Reaction-diffusion: the Fisher-KPP equation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \frac{\partial u}{\partial t} = D \nabla^2 u + r u (1 - u) .

.. code-block:: python

   problem = dm.Problem(mesh, method="FEM")
   pde = problem.add_physics(
       "general_form_PDE",
       variables="u",
       D=1.0e-3,
       r=5.0,
       equations="du/dt = D*laplacian(u) + r*u*(1 - u)",
       initial_condition="exp(-50*(x^2 + y^2))",
   )
   problem.solve_transient(end_time=0.5, time_step=0.05)

Anisotropic diffusion
^^^^^^^^^^^^^^^^^^^^^

.. math::

   -\nabla \cdot (\mathbf{K} \nabla u) = 1 , \qquad
   \mathbf{K} = \begin{pmatrix} 10 & 1 \\ 1 & 1 \end{pmatrix} .

.. code-block:: python

   problem = dm.Problem(mesh, method="FEM")
   pde = problem.add_physics(
       "general_form_PDE", variables="u", equations="-div(dot([[10, 1], [1, 1]], grad(u))) = 1"
   )
   pde.add_boundary_condition("u = 0", boundary=walls)
   problem.solve()

By its coefficients, ``c=[10, 1, 1, 1]``.

Nonlinear diffusion: the p-Laplacian
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   -\nabla \cdot \left( \left( \varepsilon^2 + |\nabla u|^2 \right)^{(p-2)/2} \nabla u \right) = 1 .

The small :math:`\varepsilon` keeps the coefficient differentiable where the
gradient vanishes, as it does at the start of Newton's method from
:math:`u = 0`:

.. code-block:: python

   problem = dm.Problem(mesh, method="FEM")
   pde = problem.add_physics(
       "general_form_PDE",
       variables="u",
       p=3.0,
       eps=1.0e-3,
       k="(eps^2 + dot(grad(u), grad(u)))^((p - 2)/2)",
       equations="-div(k*grad(u)) = 1",
   )
   pde.add_boundary_condition("u = 0", boundary=walls)
   problem.solve()

The Allen-Cahn equation
^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \frac{\partial \phi}{\partial t} = \varepsilon^2 \nabla^2 \phi - (\phi^3 - \phi) .

.. code-block:: python

   problem = dm.Problem(mesh, method="FEM")
   phase = problem.add_physics(
       "general_form_PDE",
       variables="phi",
       eps=0.05,
       equations="dphi/dt = eps^2*laplacian(phi) - (phi^3 - phi)",
       initial_condition="tanh((sqrt((x - 0.5)^2 + (y - 0.5)^2) - 0.3)/0.05)",
   )
   problem.solve_transient(end_time=0.1, time_step=0.01)

The Cahn-Hilliard equation
^^^^^^^^^^^^^^^^^^^^^^^^^^

The fourth-order equation

.. math::

   \frac{\partial c}{\partial t} = \nabla \cdot \left( M \nabla \left( c^3 - c - \kappa \nabla^2 c \right) \right)

is written as two second-order equations, with the chemical potential
:math:`\mu` as a second field:

.. math::

   \frac{\partial c}{\partial t} = \nabla \cdot (M \nabla \mu) , \qquad
   \mu = c^3 - c - \kappa \nabla^2 c .

.. code-block:: python

   problem = dm.Problem(mesh, method="FEM")
   mixture = problem.add_physics(
       "general_form_PDE",
       variables=["c", "mu"],
       M=1.0,
       kappa=1.0e-3,
       equations=[
           "dc/dt = div(M*grad(mu))",
           "mu = c^3 - c - kappa*laplacian(c)",
       ],
       initial_condition={"c": "0.1*cos(2*pi*x)*cos(2*pi*y)"},
   )
   problem.solve_transient(end_time=1.0e-3, time_step=1.0e-4)

The first equation has the time derivative of :math:`c`, and the second is
left for :math:`\mu`.

The Gray-Scott equations
^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \frac{\partial u}{\partial t} = D_u \nabla^2 u - u v^2 + F (1 - u) , \qquad
   \frac{\partial v}{\partial t} = D_v \nabla^2 v + u v^2 - (F + k) v .

.. code-block:: python

   problem = dm.Problem(mesh, method="FEM")
   species = problem.add_physics(
       "general_form_PDE",
       variables=["u", "v"],
       D_u=2.0e-5, D_v=1.0e-5, F=0.04, k=0.06,
       equations=[
           "du/dt = D_u*laplacian(u) - u*v^2 + F*(1 - u)",
           "dv/dt = D_v*laplacian(v) + u*v^2 - (F + k)*v",
       ],
       initial_condition={"u": 1.0, "v": "0.25*exp(-100*((x - 0.5)^2 + (y - 0.5)^2))"},
   )
   problem.solve_transient(end_time=10.0, time_step=1.0)

Two-group neutron diffusion: an eigenvalue problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   -\nabla \cdot (D_1 \nabla \phi_1) + (\Sigma_{a1} + \Sigma_{12}) \phi_1 = \frac{1}{k} \nu \Sigma_{f2} \phi_2 , \qquad
   -\nabla \cdot (D_2 \nabla \phi_2) + \Sigma_{a2} \phi_2 = \Sigma_{12} \phi_1 .

The symbol ``eigenvalue`` is :math:`1/k`, and the smallest eigenvalue gives
the largest :math:`k` (see :ref:`tutorial-criticality`):

.. code-block:: python

   problem = dm.Problem(mesh, method="FEM")
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
   k = 1.0 / problem.solve_eigenvalue(num_modes=1).eigenvalues[0]

Axisymmetric heat conduction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In axisymmetric coordinates the coordinates are :math:`r` and :math:`z`, and
the divergence carries the factor of the radius,
:math:`\nabla \cdot \boldsymbol{\Gamma} = r^{-1} \partial (r \Gamma_r) /
\partial r + \partial \Gamma_z / \partial z`, without a change to the text:

.. code-block:: python

   cylinder = dm.generate_rectangle_mesh(
       x_min=0.0, x_max=0.01, y_min=0.0, y_max=0.02, num_x_elements=10, num_y_elements=20
   )
   problem = dm.Problem(cylinder, method="FEM", coordinates="axisymmetric")
   heat = problem.add_physics(
       "general_form_PDE",
       variables="T",
       k=3.0,
       equations="-div(k*grad(T)) = 3.0e8*(1 - (r/0.01)^2)",
   )
   heat.add_boundary_condition("T = 600", boundary="right")
   problem.solve()

Linear elasticity: a vector field
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \nabla \cdot \boldsymbol{\sigma} + \mathbf{b} = \mathbf{0} , \qquad
   \boldsymbol{\sigma} = \lambda (\operatorname{tr} \boldsymbol{\varepsilon}) \mathbf{I} + 2 \mu \boldsymbol{\varepsilon} , \qquad
   \boldsymbol{\varepsilon} = \tfrac12 \left( \nabla \mathbf{U} + \nabla \mathbf{U}^\mathsf{T} \right) .

In plane strain:

.. code-block:: python

   problem = dm.Problem(mesh, method="FEM")
   solid = problem.add_physics(
       "general_form_PDE",
       variables=[{"U": "vector"}],
       lame_lambda=1.15e11,
       mu=7.7e10,
       strain="sym(grad(U))",
       stress="lame_lambda*trace(strain)*I + 2*mu*strain",
       equations="div(stress) + [0, -7800*9.81] = 0",
   )
   solid.add_boundary_condition(["U_x = 0", "U_y = 0"], boundary="left")
   solid.add_boundary_condition("normal_flux = [0, -1.0e6]", boundary="right")
   problem.solve()

The flux of a vector equation is the tensor inside the divergence, here
:math:`\boldsymbol{\sigma}`, and ``normal_flux`` is its traction
:math:`\boldsymbol{\sigma} \cdot \mathbf{n}`.  The ``solid_mechanics`` physics
solves the same equations with stress models, finite strain and contact.

Incompressible flow
^^^^^^^^^^^^^^^^^^^

The Navier-Stokes equations are written as printed:

.. code-block:: text

   variables=[{"U": "vector"}, "p"]
   equations=[
       "rho*(dU/dt + dot(grad(U), U)) = div(-p*I + mu*(grad(U) + transpose(grad(U))))",
       "div(U) = 0",
   ]

``general_form_PDE`` solves exactly what is written, and with the same
interpolation for the velocity and the pressure this saddle-point system needs
a stabilization that the text does not hold.  The ``incompressible_flow``
physics adds it, and is the way to solve a flow.

What is not yet written as text
-------------------------------

* the second time derivative ``d2u/dt2`` and the coefficient ``m``, which
  wait for the generalized-:math:`\alpha` time integrator;
* the gradient and the divergence of a vector field in axisymmetric
  coordinates, whose hoop terms the language does not yet write;
* ``curl`` equations of electromagnetics, which need edge elements.
