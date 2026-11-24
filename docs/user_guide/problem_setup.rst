Setting up a problem in Python
==============================

You describe a model at one of two levels.  The input, :class:`dualmesh.problem`
or a case file that ``dualmesh run`` reads, names the physics and the conditions
of a model (see :doc:`/user_guide/input`).  The object level, which this page
describes, builds a :class:`dualmesh.Problem` from physics, variables, kernels,
boundary conditions and property objects, and it reaches every object of the
library.

.. contents::
   :local:
   :depth: 1

Create the problem
------------------

A problem holds a mesh and the equations on it.

.. code-block:: python

   import dualmesh

   mesh = dualmesh.generate_rectangle_mesh(
       x_min=0.0, x_max=0.1, y_min=0.0, y_max=0.1, num_x_elements=10, num_y_elements=10
   )
   problem = dualmesh.Problem(mesh, method="automatic", coordinates="cartesian")

``method`` chooses the discretization.  The rest of the problem definition is
the same for every method.

``"FEM"``
   The Galerkin finite element method (FEM), for solids.

``"DMCDM"``
   The dual mesh control domain method (DMCDM) of [Reddy2024]_, for fluids.
   The unknowns sit at the nodes of the mesh, and the balance law holds over
   the control domain of each node, so mass and heat are conserved in every
   control domain (see :doc:`/theory/finite_volume`).

``"automatic"``
   The default.  The problem takes DMCDM when it has a ``fluid_dynamics``
   physics of the ``equal_order`` formulation, and FEM otherwise.  The choice
   is made when the physics are built, at the first solve.

The lower-case spellings ``"fem"`` and ``"dmcdm"`` are accepted.

``coordinates`` chooses the coordinate system, which sets the factor of the
volume and area integrals.

``"cartesian"``
   The default, with a factor of one.

``"axisymmetric"``
   The first coordinate is the radius :math:`r` and the second the axial
   position :math:`z`, both in :math:`\mathrm{m}`.  Every integral carries the
   factor :math:`2\pi r`, so a one-dimensional radial mesh or a two-dimensional
   :math:`(r, z)` mesh describes a body of revolution.  The mesh must not reach
   :math:`r < 0`.

``"spherical"``
   The mesh is one-dimensional, its coordinate is the radius :math:`r`, and
   every integral carries the factor :math:`4\pi r^2`.

``distributed`` splits the problem among the processes of the message passing
interface (MPI).  The default, ``None``, splits it when the script runs on
more than one process (``mpirun -n 4 python script.py``), so the same script
runs serially and in parallel.  ``partitioner`` and ``gather_scatter`` choose
how a distributed problem is split and how its processes exchange values (see
:doc:`/user_guide/solving` and :doc:`/theory/parallel`).

Add a physics
-------------

A physics names a set of equations in physical terms.  It creates its
variables when you add it, and it generates its kernels and property objects
when the problem is built at the first solve.

.. code-block:: python

   heat = problem.add_physics("heat_transfer", thermal_conductivity=20.0, heat_source=1.0e5)
   heat.add_boundary_condition("temperature = 313.15", boundary="left")
   heat.add_boundary_condition("heat_flux = 75*(temperature - 293.15)", boundary="top")
   problem.solve()

The first argument of :meth:`~dualmesh.Problem.add_physics` is the type of the
physics, and the keywords are its parameters.  The call returns the physics,
and you refer to it through that object, here ``heat``.

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Physics
     - Variables
     - Equations
   * - ``heat_transfer``
     - ``temperature``
     - Heat conduction,
       :math:`\rho c_p \partial T / \partial t + \nabla \cdot \mathbf{q} = s`
       with :math:`\mathbf{q} = -k \nabla T`, where :math:`s` is the heat
       source in :math:`\mathrm{W/m^3}`.  The heat capacity term is present
       when both ``density`` and ``specific_heat`` are given.
   * - ``general_form_PDE``
     - named by ``variables``
     - Equations written as text or given by their coefficients (see
       :doc:`/user_guide/equations`).
   * - ``solid_mechanics``
     - ``disp_x``, ``disp_y``, ``disp_z``
     - The equilibrium of a solid.  ``formulation`` is ``plane_stress``,
       ``plane_strain``, ``generalized_plane_strain``, ``axisymmetric``,
       ``three_dimensional``, one of the structural formulations ``plate``,
       ``beam`` and ``circular_plate``, or one of the members ``truss`` and
       ``shell``.
   * - ``fluid_dynamics``
     - ``velocity_x``, ``velocity_y``, ``velocity_z``, ``pressure``
     - The flow of an incompressible Newtonian fluid.  ``formulation`` is
       ``equal_order``, ``Taylor-Hood`` or ``penalty``.  With the default
       ``density`` of 0 the flow is a Stokes flow.

In this table :math:`T` is the temperature in :math:`\mathrm{K}`,
:math:`t` the time in :math:`\mathrm{s}`, :math:`\rho` the density in
:math:`\mathrm{kg/m^3}`, :math:`c_p` the specific heat capacity in
:math:`\mathrm{J/(kg\,K)}`, :math:`k` the thermal conductivity in
:math:`\mathrm{W/(m\,K)}` and :math:`\mathbf{q}` the heat flux in
:math:`\mathrm{W/m^2}`.

``dualmesh describe heat_transfer`` prints every parameter of a physics with
its unit and its description, which states the default.  ``dualmesh list
--category physics`` lists the physics.

Boundary conditions of a physics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You write a boundary condition of a physics as text with one ``=``, and it acts
on the side sets or node sets named by ``boundary``.

* A variable of the physics on the left prescribes its value, for example
  ``"temperature = 300 + 10*y"`` or ``"velocity_x = 1"``.  The right side is a
  number, the name of a registered function or an expression in :math:`x`,
  :math:`y`, :math:`z` and :math:`t`.
* The name of a flux of the physics on the left prescribes that flux.  For
  ``heat_transfer`` the name is ``heat_flux``, the heat flux along the outward
  unit normal :math:`\mathbf{n}`, :math:`\mathbf{n} \cdot \mathbf{q}`, which is
  the heat leaving the body.  For ``solid_mechanics`` the names are
  ``traction``, the surface traction
  :math:`\boldsymbol{\sigma} \cdot \mathbf{n}` in :math:`\mathrm{Pa}` with
  :math:`\boldsymbol{\sigma}` the Cauchy stress, ``total_force``, a force in
  :math:`\mathrm{N}` spread uniformly over the boundary, and ``pressure``, the
  traction :math:`-p\,\mathbf{n}` of a pressure :math:`p` in
  :math:`\mathrm{Pa}`.  A vector is written ``[a, b, c]``, one value per
  component.  For ``general_form_PDE`` the name is ``normal_flux``.  The right
  side of ``heat_flux``, ``traction`` and ``normal_flux`` may use the fields,
  so a convection or radiation condition is written as it is printed.
* The words ``"fixed"`` and ``"symmetry"`` need no data.  They apply to a
  physics with a vector unknown, ``solid_mechanics`` or ``fluid_dynamics``.
  ``"fixed"`` holds every component of the vector at zero, and for a
  structural formulation every rotation too.  ``"symmetry"`` holds the normal
  component at zero on a boundary that lies in a plane normal to a coordinate
  axis.
* The word ``"periodic"`` joins two boundaries, named by ``primary`` and
  ``secondary`` in place of ``boundary`` (see :ref:`problem-setup-boundary-conditions`).
* ``point`` in place of ``boundary`` prescribes a value at the node closest to
  a point, for example the pressure level of an enclosed flow.

A list of texts adds each condition.  Any other keyword is a definition, a
number or a text, that the condition uses by its name.  These calls assume a
``solid``, a ``flow`` and a ``heat`` physics on a mesh with the side sets
they name:

.. code-block:: python

   solid.add_boundary_condition("fixed", boundary="jaws")
   solid.add_boundary_condition("symmetry", boundary="bottom")
   solid.add_boundary_condition("total_force = [0, -force, 0]", boundary="grip", force=150.0)
   flow.add_boundary_condition(["velocity_x = 1", "velocity_y = 0"], boundary="inlet")
   flow.add_boundary_condition("pressure = 0", point=[0.5, 0.0])
   heat.add_boundary_condition("heat_flux = 5.67e-8*0.8*(temperature^4 - 293.15^4)", boundary="surface")

A boundary without a condition carries a zero flux.  It is an insulated wall
for heat transfer and a traction-free surface for a solid.

``solid_mechanics`` takes five more conditions:

* ``"contact"`` with ``primary``, ``secondary`` and, for friction,
  ``friction_coefficient``, and no ``boundary``: mortar contact of two
  surfaces.
* ``"cohesive"`` with ``primary``, ``secondary``, ``strength`` in
  :math:`\mathrm{Pa}` and ``toughness`` in :math:`\mathrm{J/m^2}`, and no
  ``boundary``: a cohesive zone between two surfaces that coincide at the
  start.
* ``"rotation"`` with ``angle`` in :math:`\mathrm{rad}`, an expression of
  :math:`t`, ``center`` and, in three dimensions, ``axis``: a prescribed
  rotation of the boundary.
* ``"torque = ..."`` in :math:`\mathrm{N\,m}`, with ``center`` and, in three
  dimensions, ``axis``: a torque on the boundary.
* ``"normal_displacement = ..."``: a displacement along the outward normal,
  with the tangential displacements free.

``"rotation"`` and ``"torque"`` apply to the ``plane_strain``,
``plane_stress`` and ``three_dimensional`` formulations.

Properties from property objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A material parameter of a physics that you leave out is read from the property
of the same name.  This holds for ``thermal_conductivity`` of
``heat_transfer`` and for ``youngs_modulus`` and ``poissons_ratio`` of
``solid_mechanics``.  A property object on a block then supplies the value on
that block:

.. code-block:: python

   problem.add_property(
       "parsed",
       block=["wall"],
       property_name="thermal_conductivity",
       expression="10.0 + 0.01*temperature",
   )
   problem.add_property("constant", block=["coolant"], properties={"thermal_conductivity": 0.6})
   heat = problem.add_physics("heat_transfer")

This example needs a mesh with the blocks ``wall`` and ``coolant``.

Extra terms of a physics
~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~dualmesh.physics.Physics.add_kernel` adds a kernel of the object
reference to the equations of a physics and fills in the variable when the
physics has one.  A physics with several variables needs ``variable``.
``heat_transfer`` also takes the word ``heat_source``, which adds a
``body_force`` kernel:

.. code-block:: python

   heat.add_kernel("heat_source", heat_source=1.0e5, block=["wall"])

Couple two physics
------------------

A coupling joins two physics.  It takes each physics as the object that
:meth:`~dualmesh.Problem.add_physics` returned:

.. code-block:: python

   heat = problem.add_physics("heat_transfer", thermal_conductivity=20.0)
   solid = problem.add_physics(
       "solid_mechanics", youngs_modulus=200.0e9, poissons_ratio=0.3, formulation="plane_stress"
   )
   problem.add_coupling(
       "thermal_expansion",
       heat_transfer=heat,
       solid_mechanics=solid,
       linear_expansion_coefficient=1.2e-5,
       stress_free_temperature=293.15,
   )

You may leave out a physics when the problem has only one physics of its type,
so ``heat_transfer=heat`` and ``solid_mechanics=solid`` are optional here.  Add
the couplings before the first solve.

``thermal_expansion``
   The thermal strain :math:`\alpha (T - T_0)` of a solid, with
   :math:`\alpha` the linear expansion coefficient
   (``linear_expansion_coefficient``) in :math:`\mathrm{1/K}` and :math:`T_0`
   the stress-free temperature (``stress_free_temperature``) in
   :math:`\mathrm{K}`.

``nonisothermal_flow``
   The heat that a flow carries, :math:`\rho c_p \mathbf{v} \cdot \nabla T`
   with :math:`\mathbf{v}` the velocity in :math:`\mathrm{m/s}`.  With
   ``gravity`` and ``volumetric_expansion_coefficient`` it adds the Boussinesq
   buoyancy force :math:`-\rho \beta (T - T_0) \mathbf{g}` to the flow, with
   :math:`\mathbf{g}` the gravitational acceleration in :math:`\mathrm{m/s^2}`,
   :math:`\beta` the volumetric expansion coefficient in :math:`\mathrm{1/K}`
   and :math:`T_0` the reference temperature (``reference_temperature``,
   default 0).

The streamline upwind Petrov-Galerkin (SUPG) term stabilizes the heat that a
``nonisothermal_flow`` coupling carries.  ``advection_scheme`` of
``heat_transfer`` chooses another discretization of that heat: ``"limited"``,
``"linear_upwind"``, ``"upwind"`` or ``"central"`` (see
:doc:`/theory/finite_volume`).  A ``heat_transfer`` physics without the
coupling refuses ``advection_scheme``.

``dualmesh list --category coupling`` lists the couplings.

Add a variable
--------------

A variable is one scalar unknown field.  A physics adds its own variables,
and :meth:`~dualmesh.Problem.add_variable` adds one for the kernels that you add
yourself:

.. code-block:: python

   problem.add_variable("temperature", initial_condition=293.15, unit="K")
   problem.add_variable("concentration", block=["fluid"])
   problem.add_variable("u", initial_condition="sin(pi*x)*sin(pi*y)")

``block`` restricts the variable to some blocks of the mesh, and the default,
an empty list, defines it everywhere.  ``initial_condition`` sets the starting
values of a transient run and the first iterate of a nonlinear solve.  It is a
number, the name of a registered function, an expression in :math:`x`,
:math:`y`, :math:`z` and :math:`t`, or a Python callable ``f(x, y, z, t)``, and
the default is zero.  ``unit`` is the unit that the output files write beside
the values.  The call returns the index of the variable.

A vector field, such as a displacement, is one variable per component.

``order`` chooses how the variable is interpolated:

``"mesh"``
   The default: the order of the elements, linear on a linear mesh and
   quadratic on a quadratic one.

``"first"``
   Linear on every element.  On a quadratic mesh only the corner nodes carry
   the variable.  This is the pressure of the Taylor-Hood element, which
   combines a quadratic velocity with a linear pressure [TaylorHood1973]_, and
   it needs ``method="FEM"``:

   .. code-block:: python

      mesh = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 16, 16, element_type="Quad9")
      problem = dualmesh.Problem(mesh, method="FEM")
      problem.add_variable("velocity_x")
      problem.add_variable("velocity_y")
      problem.add_variable("pressure", order="first")

``fluid_dynamics`` with ``formulation="Taylor-Hood"`` declares its variables
in this way.

The number of variables has a limit.  Automatic differentiation seeds one
derivative slot per degree of freedom of an element, one per node and
variable, and the corner nodes only for a first-order variable.  The default
limit is 96 slots, so a ``Quad9`` mesh carries up to ten variables and a
``Hex27`` mesh three, and the Taylor-Hood element on ``Hex27`` uses
:math:`27 \times 3 + 8 = 89`.  A problem over the limit is refused with a
message that names the element and the CMake setting
``DUALMESH_MAX_AD_DERIVATIVES`` that raises it.

.. _problem-setup-kernels:

Add a kernel
------------

A kernel adds one term to the equation of a variable.

The sign of the flux
~~~~~~~~~~~~~~~~~~~~

The documentation writes each equation as the balance law

.. math::

   \frac{\partial m}{\partial t} + \nabla \cdot \mathbf{F} = s ,

where :math:`m` is the stored quantity per unit volume, :math:`\mathbf{F}` its
flux and :math:`s` its source per unit volume.  :math:`\mathbf{F}` is the
physical flux, positive along the outward unit normal :math:`\mathbf{n}` of a
body.  For heat conduction, :math:`m = \rho c_p T`,
:math:`\mathbf{F} = \mathbf{q} = -k \nabla T`, and :math:`s` is the heat
source.

A kernel of the C++ library returns its term in the form

.. math::

   -\nabla \cdot \hat{\mathbf{F}} + \hat{S} = 0 ,

with the kernel flux :math:`\hat{\mathbf{F}}` and the kernel source
:math:`\hat{S}`.  The two forms are the same equation when

.. math::

   \hat{\mathbf{F}} = -\mathbf{F} , \qquad
   \hat{S} = \frac{\partial m}{\partial t} - s ,

so the kernel flux is minus the physical flux.  The ``heat_conduction`` kernel
returns :math:`\hat{\mathbf{F}} = k \nabla T`, which is the physical flux
:math:`\mathbf{F} = \mathbf{q} = -k \nabla T`.  The same sign holds at a
boundary.  The ``flux`` of a ``Neumann`` condition is
:math:`\mathbf{n} \cdot \hat{\mathbf{F}} = -\mathbf{n} \cdot \mathbf{F}`, the
flux entering the body.

The table gives the terms of common kernels in both forms.  :math:`D` is the
``diffusivity`` of ``diffusion``, :math:`f` the ``value`` of ``body_force``,
:math:`c` the ``coefficient`` of ``reaction`` or ``time_derivative``, and
:math:`p_e` the dimensionless ``exponent`` of ``reaction``.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Kernel
     - Kernel term
     - Balance law term
   * - ``diffusion``
     - :math:`\hat{\mathbf{F}} = D \nabla u`
     - :math:`\mathbf{F} = -D \nabla u`
   * - ``heat_conduction``
     - :math:`\hat{\mathbf{F}} = k \nabla T`
     - :math:`\mathbf{F} = -k \nabla T`
   * - ``body_force``
     - :math:`\hat{S} = -f`
     - :math:`s = f`
   * - ``reaction``
     - :math:`\hat{S} = c\, u^{p_e}`
     - :math:`s = -c\, u^{p_e}`
   * - ``time_derivative``, constant :math:`c`
     - :math:`\hat{S} = c\, \partial u / \partial t`
     - :math:`m = c\, u`

The kernels of one variable add their terms, so you assemble a
convection-diffusion-reaction equation from three kernels.
``dualmesh list --category kernel`` lists the kernels.

Add the kernels
~~~~~~~~~~~~~~~

:meth:`~dualmesh.Problem.add_kernel` takes the type name of the kernel and its
parameters as keywords.  This steady heat conduction problem in a plate of
side :math:`0.1\ \mathrm{m}` has a fixed temperature on the left, a convection
condition on the top and insulated sides elsewhere:

.. code-block:: python

   import dualmesh

   mesh = dualmesh.generate_rectangle_mesh(
       x_min=0.0, x_max=0.1, y_min=0.0, y_max=0.1, num_x_elements=10, num_y_elements=10
   )
   problem = dualmesh.Problem(mesh, method="DMCDM")
   problem.add_variable("temperature", initial_condition=293.15, unit="K")
   problem.add_kernel("heat_conduction", variable="temperature", thermal_conductivity=20.0)
   problem.add_kernel("body_force", variable="temperature", value=1.0e5)
   problem.add_boundary_condition("Dirichlet", "left", variable="temperature", value=313.15)
   problem.add_boundary_condition(
       "Robin", "top", variable="temperature", transfer_coefficient=75.0, ambient_value=293.15
   )
   problem.solve()

``variable`` takes one variable or a list, which adds the object once for each
variable.  The call returns the object, the last one for a list, which
:meth:`~dualmesh.Problem.kernel_boundary_flux` accepts.

.. _problem-setup-boundary-conditions:

Add a boundary condition
------------------------

:meth:`~dualmesh.Problem.add_boundary_condition` takes the type of the
condition, the side sets or node sets of ``boundary``, and the parameters of the
condition.  It refuses a boundary that the mesh does not have and names the
closest one.  The conditions belong to three categories.

An integrated condition adds a boundary integral to the equation of its
variable.  ``Neumann`` prescribes :math:`\mathbf{n} \cdot \hat{\mathbf{F}}`,
the flux entering the body.  ``Robin`` prescribes
:math:`\mathbf{n} \cdot \hat{\mathbf{F}} = q_0 - h_\mathrm{c} (u - u_\infty)`,
with :math:`h_\mathrm{c}` the transfer coefficient
(``transfer_coefficient``), :math:`u_\infty` the ambient value
(``ambient_value``) and :math:`q_0` the prescribed part (``flux``).  For heat
transfer, :math:`h_\mathrm{c}` is in :math:`\mathrm{W/(m^2\,K)}`.
``radiative_heat_flux`` and the ``traction`` and ``pressure`` of a solid are
integrated conditions too.

A nodal condition prescribes the value at nodes.  ``Dirichlet`` writes the
value into the solution at every node of the boundary.  ``point_Dirichlet``
holds the node closest to ``point``, and takes no ``boundary``.

A constraint joins degrees of freedom.  ``periodic`` makes the field on the
``secondary`` boundary equal to the field on the ``primary`` boundary, carried
over by a translation:

.. code-block:: python

   problem.add_boundary_condition("periodic", primary="left", secondary="right")

Each node of the secondary boundary is joined to the node of the primary
boundary at the position shifted back by the translation.  The translation is
``translation`` when you give it, and the vector between the centroids of the
two boundaries otherwise.  The two nodes then act as one unknown, whose
equation collects the equations of both.  ``variable`` names the periodic
variables, and the default is every variable.  The two boundaries must match
node by node, and a value prescribed on a secondary node must also be
prescribed on its primary node.

A boundary without a condition carries a zero flux,
:math:`\mathbf{n} \cdot \hat{\mathbf{F}} = 0`.

Add a property object
---------------------

A property object computes named properties at the integration points of its
blocks.  Kernels, boundary conditions, physics and expressions read a property
by its name, and a coefficient given as the name of a property reads it, as in
``thermal_conductivity="thermal_conductivity"``.

``constant``
   Properties constant in space and time:
   ``properties={"thermal_conductivity": 400.0}``.

``function``
   Properties that are functions of position and time, each a registered
   function or an expression in :math:`x`, :math:`y`, :math:`z` and :math:`t`:
   ``properties={"thermal_conductivity": "20 + 5*x"}``.

``parsed``
   One property, ``property_name``, given by ``expression``.  The expression may
   use the fields, the registered functions, the element fields, the properties
   of the property objects added before it and the named constants of
   ``constants``.  Automatic differentiation carries the dependence on the
   solution into the Jacobian.

Every property object takes ``block`` and ``property_factors``, a dict of
factors on the properties it computes, for sensitivity studies.
``dualmesh list --category property`` lists the property objects.

Add a point source
------------------

:meth:`~dualmesh.Problem.add_point_source` applies a concentrated source at
nodes, a force for a displacement or a heat rate in :math:`\mathrm{W}` for a
temperature.  ``value`` is the magnitude.  ``points`` gives the loaded points,
one list of coordinates each, and ``boundary`` the side sets or node sets whose
nodes carry the load:

.. code-block:: python

   problem.add_point_source(variable="temperature", value=50.0, points=[[0.05, 0.05]])

Each point moves to the nearest node of the mesh, however far that node is.  A
positive value adds heat, or acts along the positive direction of the
variable.

Parameters that many objects accept
-----------------------------------

The base classes declare these parameters, and each page of :doc:`/objects`
lists them with the parameters of its object.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Objects
     - Parameters
   * - Kernels and integrated conditions
     - ``variable``, ``block``, ``quadrature``, ``reduced_integration``, and
       ``boundary`` for the conditions
   * - Nodal conditions
     - ``variable``, ``boundary``
   * - Property objects
     - ``block``, ``property_factors``

``block``
   The block names this object acts on.  The default, an empty list, means
   every block.  Two sets of objects on two blocks give two materials.

``quadrature``
   The rule that integrates the term.  The default, ``"automatic"``, is
   Gauss-Legendre with one more point per direction than the polynomial order
   of the mesh: two points on a linear mesh and three on a quadratic one.  The
   other rules are ``"gauss1"`` to ``"gauss20"``, ``"midpoint"``,
   ``"trapezoid"``, ``"simpson"``, ``"nodal"`` (one point at the owning node,
   which lumps the term), ``"interface"`` (one point at the control domain
   interface) and ``"control_domain_trapezoid"`` (the trapezoidal rule over the
   whole control domain).  The names ``"auto"``, ``"trapezoidal"``,
   ``"lumped"``, ``"centroid"`` and ``"cd_trapezoid"`` are accepted as aliases,
   ``"gauss"``
   means ``"gauss2"``, and the case of the letters does not matter (see
   :doc:`/theory/elements`).

``reduced_integration``
   When true, the solution and its gradient are evaluated at the centroid of
   the element, and the integration keeps the rule of ``quadrature``.  This is
   selective reduced integration [HughesCohenHaroun1978]_.  It removes locking
   in the transverse shear term of a thin beam or plate and in the penalty term
   of an incompressible flow.  On a bending or diffusion term it degrades the
   accuracy, so the default is false.

Whether a steady solve ramps an object when it continues a problem that does
not converge depends on the type of the object (see :doc:`/user_guide/solving`).

.. _problem-setup-values:

The values of a parameter
-------------------------

``dualmesh describe`` prints the kind of each parameter.  A parameter of the
kind ``real or function`` takes four kinds of value:

.. code-block:: python

   # A constant.
   problem.add_kernel("body_force", variable="u", value=2.5)

   # An expression, as text or compiled first.
   problem.add_kernel("body_force", variable="u", value="sin(pi*x)*sin(pi*y)")
   problem.add_kernel("body_force", variable="u", value=dualmesh.parsed_function("sin(pi*x)*sin(pi*y)"))

   # The name of a function registered on the problem.
   problem.add_function("ramp", lambda x, y, z, t: min(t, 1.0))
   problem.add_kernel("body_force", variable="u", value="ramp")

   # A Python callable of (x, y, z, t).
   problem.add_kernel("body_force", variable="u", value=lambda x, y, z, t: x * x + y)

A text that is not the name of a registered function is an expression.  Which
names an expression may use depends on the parameter, and the description of
the parameter states it.

* A value such as ``value`` of ``body_force`` or ``Dirichlet`` is a function of
  :math:`x`, :math:`y`, :math:`z` and :math:`t`.
* A coefficient such as ``diffusivity`` of ``diffusion``, ``coefficient`` of
  ``reaction`` or ``flux`` of ``Neumann`` may also use the fields, their
  gradients as ``grad_x(u)``, ``grad_y(u)`` and ``grad_z(u)``, the symbol
  ``eigenvalue`` of an eigenvalue study, the registered functions, the element
  fields and the properties of the property objects added before it.
  Automatic differentiation [Wengert1964]_ carries its dependence on the
  solution into the Jacobian.

A constant and an expression are evaluated in C++, and the assembly runs on
every thread.  A Python callable, given directly or registered by name, is
evaluated through the Python interpreter, which holds the global interpreter
lock, so it puts the whole assembly on one thread.  The problem reports this:

.. code-block:: python

   problem.thread_safe  # False once a Python callable or object is attached
   problem.effective_threads()  # 1 in that case

Use an expression wherever the expression language holds the formula, and a
callable only where it does not.

The expression language
~~~~~~~~~~~~~~~~~~~~~~~

An expression holds these elements:

* Numbers, ``+``, ``-``, ``*``, ``/``, and ``^`` or ``**`` for a power.  A
  power binds tighter than a minus sign and groups from the right, so
  ``-u^2`` is :math:`-(u^2)` and ``2^-1`` is :math:`0.5`.
* The comparisons ``<``, ``>``, ``<=``, ``>=``, ``==`` and ``!=``, which give 1
  when true and 0 when false, and ``if(condition, a, b)``.
* The constants ``pi`` and ``e``.
* The functions ``sin``, ``cos``, ``tan``, ``asin``, ``acos``, ``atan``,
  ``sinh``, ``cosh``, ``tanh``, ``exp``, ``log``, ``log10``, ``sqrt``,
  ``abs``, ``floor``, ``ceil``, ``erf`` and ``sign`` of one argument, and
  ``atan2``, ``pow``, ``hypot``, ``min`` and ``max`` of two.

The spellings ``Abs`` and ``E`` that SymPy prints are accepted.  An
expression compiles once to a short program that C++ evaluates.  A text that
is neither a registered name nor a valid expression is refused with the
position at which it fails to parse.  The equations of ``general_form_PDE``
use a larger language with vectors and tensors (see
:doc:`/user_guide/equations`).

Find what exists
----------------

The registry of the library describes every object, physics and coupling, so
the description always matches the code.

.. code-block:: bash

   dualmesh list                        # every type
   dualmesh list --category kernel      # the kernels
   dualmesh list --module heat_transfer # the types of one module
   dualmesh describe heat_conduction    # the parameters of one type

The categories are ``physics``, ``coupling``, ``kernel``,
``boundary_condition``, ``nodal_boundary_condition``, ``nodal_load``,
``property`` and ``constraint``.  The same queries run in Python:

.. code-block:: python

   dualmesh.list_objects(category="kernel")
   dualmesh.registered_types()  # every object type
   dualmesh.object_category("heat_conduction")  # 'kernel'
   dualmesh.object_module("heat_conduction")  # 'heat_transfer'
   print(dualmesh.describe("heat_conduction"))

:func:`dualmesh.describe` prints the description of a type, then each
parameter with its kind, whether it is required, and its description, which
states the default.
:doc:`/objects` has one page per object, generated from the same registry when
the documentation is built.

.. _problem-setup-python-objects:

Write a kernel in Python
------------------------

You can add a term without changing the C++ code.  Subclass
:class:`dualmesh.PythonKernel` and define ``compute_flux``, which returns the
components of the kernel flux :math:`\hat{\mathbf{F}}`, or ``compute_source``,
which returns the kernel source :math:`\hat{S}`, or both (see
:ref:`problem-setup-kernels`).  This kernel is heat conduction with the
conductivity :math:`k = k_0 (1 + b\, T)`, where :math:`k_0` is the
conductivity at :math:`T = 0` in :math:`\mathrm{W/(m\,K)}` and :math:`b` a
coefficient in :math:`\mathrm{1/K}`:

.. code-block:: python

   import dualmesh


   class NonlinearConduction(dualmesh.PythonKernel):
       """Kernel flux k grad T with k = k0 (1 + b T)."""

       def compute_flux(self, ctx):
           T = ctx.value(self.variable)
           k = self.k0 * (1.0 + self.b * T)
           return [k * g for g in ctx.gradient(self.variable)]


   problem.add_kernel(NonlinearConduction(variable="temperature", k0=20.0, b=1.0e-3))

The constructor takes ``variable``, ``name``, ``block``, ``quadrature`` (default
``"gauss2"``) and ``reduced_integration``.  Every other keyword becomes an
attribute, here ``self.k0`` and ``self.b``, and an entry of
``self.parameters``.  ``self.variable`` is the index of the variable once the
problem is set up.

The method ``setup(self, problem)`` runs when the problem is set up.  Resolve
other names there, as ``problem.variable_index("temperature")`` for a coupled
variable or ``problem.property_id("thermal_conductivity")`` for a property.

``ctx`` is the state at one integration point:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Member
     - Meaning
   * - ``ctx.x``, ``ctx.time``, ``ctx.dt``
     - The position, the time and the time step.
   * - ``ctx.dim``, ``ctx.element``, ``ctx.block``
     - The space dimension, the element and its block.
   * - ``ctx.value(v)``, ``ctx.gradient(v)``
     - The value and the gradient (three entries) of variable ``v`` at the
       current iterate.
   * - ``ctx.coefficient_value(v)``, ``ctx.coefficient_gradient(v)``
     - The value and the gradient to use in a coefficient: the current
       iterate under Newton's method and the previous iterate under the
       Picard iteration.
   * - ``ctx.old_value(v)``, ``ctx.old_gradient(v)``
     - The value and the gradient at the end of the last time step.
   * - ``ctx.property(id, component=0)``
     - The value of a property.
   * - ``ctx.normal``, ``ctx.on_boundary``
     - The outward unit normal, and whether the point lies on a boundary.
   * - ``ctx.is_picard``
     - Whether the nonlinear solver is the Picard iteration
       (``nonlinear_solver="picard"``).

The arithmetic inside these methods runs on :class:`dualmesh.ADReal`
numbers, which carry their derivatives with respect to the degrees of freedom
of the element.  The Jacobian is exact without a derivative written by hand,
so Newton's method converges quadratically.  Use ``dualmesh.exp``,
``dualmesh.log``, ``dualmesh.sqrt``, ``dualmesh.sin``, ``dualmesh.cos``,
``dualmesh.tanh``, ``dualmesh.abs`` and ``dualmesh.pow``, which accept both
floats and these numbers.  The functions of ``math`` and ``numpy`` drop the
derivatives.

A kernel of a time derivative defines ``is_time_kernel`` to return true, so
that a steady solve skips it and the time scheme does not weight it.

The other objects follow the same pattern:

:class:`dualmesh.PythonBoundaryCondition`
   An integrated condition, built with ``variable`` and ``boundary``.
   ``compute_boundary_flux(ctx)`` returns
   :math:`\mathbf{n} \cdot \hat{\mathbf{F}}`, the flux entering the body.

:class:`dualmesh.PythonNodalBoundaryCondition`
   A prescribed value, built with ``variable`` and ``boundary``.
   ``compute_value(x, t)`` returns the value at the position ``x`` and the
   time ``t``.

:class:`dualmesh.PythonProperty`
   A property object, built with ``block``.  ``declare_properties(registry)``
   declares each property with ``registry.declare(name)`` and keeps the id that
   it returns, and ``compute_properties(ctx)`` sets each value with
   ``ctx.set_property(id, value)``.

An object written in Python puts the assembly on one thread, for the reason
given in :ref:`problem-setup-values`.  Move a term to C++ once its form is
settled (see :doc:`/developing`).  :doc:`/tutorials/custom_kernel` solves a
problem with a kernel written in Python.
