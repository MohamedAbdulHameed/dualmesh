Setting up a problem
====================

A mesh on its own describes a region of space.  A :class:`dualmesh.Problem`
attaches unknowns to that region, says which equations they satisfy, and says
what happens on the boundary.  This chapter explains every part of that
description and every keyword it takes.

.. contents::
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

Creating the problem
--------------------

.. code-block:: python

   import dualmesh as dm

   mesh = dm.generate_rectangle_mesh(
       x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, num_x_elements=10, num_y_elements=10
   )

   problem = dm.Problem(mesh, method="automatic", coordinates="cartesian")

``method`` chooses the discretisation.  Both methods read the same problem
description.

``"FEM"``
   The Galerkin finite element method, for solids.  It reaches order
   :math:`p + 1` in :math:`L^2` on elements of order :math:`p`, accepts every
   element type, and carries the structures, the mixed formulations and the
   Taylor-Hood element.

``"DMCDM"``
   The dual mesh control domain method of [Reddy2019a]_ and [Reddy2024]_, for
   fluids.  The unknowns are located at the mesh nodes.  The balance law is
   integrated over each node's control domain, and the flux at a control domain
   interface is taken from the element interpolation, so mass and heat are
   conserved in every control domain (see :doc:`/theory/finite_volume`).

``"automatic"``
   The default: DMCDM when the problem has a ``fluid_dynamics`` of the pressure
   formulation, FEM otherwise.  The choice is made when the physics are built,
   at the first solve.

``coordinates`` chooses the coordinate system, which determines the volume and
area factors in every integral.

``"cartesian"``
   Plane or three-dimensional Cartesian coordinates, with a factor of one.  This
   is the default.

``"axisymmetric"``
   The first coordinate is the radius :math:`r` and the second the axial
   coordinate :math:`z`.  Every volume integral carries a factor :math:`2\pi r`
   and every area integral the same, so a two-dimensional mesh describes a body
   of revolution.  The mesh must not cross :math:`r < 0`.

``"spherical"``
   The first coordinate is the radius and the problem is a function of it alone.
   Volume integrals carry :math:`4\pi r^2`.

``distributed`` and ``partitioner`` concern a problem split among
MPI processes.  A script run with ``mpirun -n 4 python script.py`` splits the
problem among the four processes, and the same script runs serially with
``python script.py`` (see :doc:`/theory/parallel`).

Physics
-------

A model is described in physical terms by adding *physics* to the problem.  A
physics names a set of equations, creates its variables, and generates the
kernels and property objects of its equations when the problem is solved.  Its
boundary conditions and extra terms are added to the physics, which fills in
the variables.

.. code-block:: python

   heat = problem.add_physics("heat_transfer", thermal_conductivity=20.0, heat_source=1.0e6)
   heat.add_boundary_condition("temperature = 40", boundary="left")
   heat.add_boundary_condition("heat_flux = 75*(temperature - 293)", boundary="top")

The first argument is the type of the physics, and the keywords are its
parameters.  The physics has no name: the variable it is assigned to, here
``heat``, refers to it.  The physics available are:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Physics
     - Equations
   * - ``heat_transfer``
     - Heat conduction with a heat source and a heat capacity,
       :math:`\rho c_p \partial T / \partial t - \nabla \cdot (k \nabla T) = q`.
   * - ``general_form_PDE``
     - Equations that no other physics names, written as they are printed
       (``"du/dt = div(k*grad(u)) + f"``) or by their coefficients,
       :math:`d \partial u / \partial t + \nabla \cdot (-c \nabla u - \boldsymbol{\alpha} u + \boldsymbol{\gamma}) + \boldsymbol{\beta} \cdot \nabla u + a u = f`
       (see :doc:`/user_guide/equations`).
   * - ``solid_mechanics``
     - Linear elasticity at small strain in plane stress, plane strain,
       axisymmetric or three-dimensional form, and the structural
       formulations: a plate (first-order theory), a beam (Euler-Bernoulli
       or Timoshenko) and an axisymmetric circular plate (classical or
       first-order).
   * - ``fluid_dynamics``
     - The Navier-Stokes or Stokes equations of an incompressible Newtonian
       fluid, in the penalty, pressure or Taylor-Hood formulation.

``dualmesh describe <physics>`` lists the parameters of each one, with its
unit, its default and the reason for the default.

**Boundary conditions of a physics.**  A boundary condition is written as
text and acts on the side sets or node sets named by ``boundary``, which is
always given:

* A variable of the physics on the left prescribes its value on the
  boundary, for example ``"temperature = 300 + 10*y"`` or
  ``"velocity_x = 1"``.
* The name of the flux of the physics on the left prescribes the flux through
  the boundary along the outward normal :math:`\mathbf{n}`.  The names are
  ``heat_flux`` (:math:`\mathbf{n} \cdot \mathbf{q}`, the heat leaving the
  body) for ``heat_transfer``, ``traction`` (:math:`\boldsymbol{\sigma}
  \cdot \mathbf{n}`), ``total_force`` (a force spread over the boundary) and
  ``pressure`` for ``solid_mechanics``, and ``normal_flux``
  (:math:`\mathbf{n} \cdot \boldsymbol{\Gamma}`) for ``general_form_PDE``,
  which also takes the flux written out, as ``-k*du/dn``.

The right side is an expression of :math:`x`, :math:`y`, :math:`z`,
:math:`t`, the fields, the registered functions and the properties, so a
condition that depends on the solution (convection, radiation) is written as
it is printed.  A list of texts adds several conditions on one boundary, and
the words ``"fixed"`` (every displacement zero) and ``"symmetry"`` (the
normal displacement zero) of a solid need no data:

.. code-block:: python

   solid.add_boundary_condition("fixed", boundary="jaws")
   solid.add_boundary_condition("symmetry", boundary="bottom")
   solid.add_boundary_condition("total_force = [0, -150, 0]", boundary="grip")
   solid.add_boundary_condition("disp_x = 0.001", boundary="end")
   flow.add_boundary_condition(["velocity_x = 1", "velocity_y = 0"], boundary="inlet")
   heat.add_boundary_condition("heat_flux = 5.67e-8*0.8*(temperature^4 - 293^4)", boundary="surface")

A boundary without a condition has a zero flux: it is insulated, free of
traction, or a free outflow.

**Properties from the property objects.**  A property of a physics that is not given
as a number (``thermal_conductivity`` of ``heat_transfer``, ``youngs_modulus``
and ``poissons_ratio`` of ``solid_mechanics``) is read from the
property of the same name, so that a property object added to a block, such as
``parsed`` or ``constant``, supplies it on that block:

.. code-block:: python

   problem.add_property(
       "parsed",
       block=["wall"],
       property_name="thermal_conductivity",
       expression="10.0 + 0.01*temperature",
   )
   problem.add_property("constant", block=["coolant"], properties={"thermal_conductivity": 0.6})
   heat = problem.add_physics("heat_transfer")

**Couplings.**  Two physics are coupled by a coupling, which takes them as the
objects that ``add_physics`` returned:

.. code-block:: python

   problem.add_coupling(
       "thermal_expansion",
       heat_transfer=heat,
       solid_mechanics=solid,
       linear_expansion_coeff=1.2e-5,
       stress_free_temperature=293.15,
   )

A physics may be left out when the problem has only one physics of its type,
so ``heat_transfer=heat`` and ``solid_mechanics=solid`` are optional here.
The couplings are ``thermal_expansion`` (the thermal strain of a solid) and
``nonisothermal_flow`` (the transport of heat by a flow and, with ``gravity``
and ``vol_expansion_coeff``, the buoyancy force that the
temperature exerts on the flow).  ``dualmesh list --category coupling``
lists them.

**Convection-dominated transport.**  When a flow carries heat much faster than
the heat conducts (a cell Péclet number above about 1), the central
discretisation oscillates, so the heat that a ``nonisothermal_flow`` coupling
carries is stabilised by default with the streamline upwind Petrov-Galerkin
term (SUPG).  The inertia of the flow is treated the same way, with no option.  ``advection_scheme`` of ``heat_transfer`` chooses another
scheme where the case calls for it, for example ``"central"`` where conduction
matters as much as convection:

.. code-block:: python

   heat = problem.add_physics("heat_transfer", thermal_conductivity=1.0,
                              advection_scheme="central")

:doc:`../theory/finite_volume` and :doc:`../theory/heat_and_fluids` give the
schemes with their measured order and bounds.

**Extra terms.**  ``add_kernel`` adds a term of the object reference to the
equations of a physics, e.g., a heat source on one block, for which
``heat_transfer`` takes the word ``heat_source`` (the object ``body_force``):

.. code-block:: python

   heat.add_kernel("heat_source", heat_source=2.0e8, block=["wall"])

The sections below describe the object level, i.e., the variables, kernels,
property objects and boundary conditions that the physics generate.  A model whose
equations have no physics is written with these objects directly.  The same
equations are written with one of the two levels, never with both.

Variables
---------

A variable is one scalar unknown field.

.. code-block:: python

   problem.add_variable("temperature")
   problem.add_variable("disp_x", initial_condition=0.0)
   problem.add_variable("pressure", blocks=["fluid"])

``blocks`` restricts a variable to part of the mesh, and the default, an empty
list, defines it everywhere.  ``initial_condition`` may be a number, the name of
a registered function or a Python callable of :math:`(x, y, z, t)`.  It sets the
starting values, which matter for a transient run and for the first iterate of a
nonlinear one.

Every variable has one unknown at every entity of the problem, so the total
number of unknowns is the number of entities times the number of variables.  A
vector field such as a displacement is entered as one variable per component,
which is the form that the beam, plate and fluid kernels expect.

``order`` chooses how a variable is interpolated.  The default, ``"mesh"``,
follows the elements: linear on a linear mesh, quadratic on a quadratic one.
``"first"`` makes the variable linear on every element: on a quadratic mesh
only the corner nodes then carry it, and the values reported at the other
nodes are those of the linear field.  This is the pressure interpolation of the
Taylor-Hood element, which combines a quadratic velocity with a linear pressure
(see :doc:`../theory/heat_and_fluids`), and it needs the finite element method:

.. code-block:: python

   mesh = dm.generate_rectangle_mesh(0, 1, 0, 1, 16, 16, element_type="Quad9")
   problem = dm.Problem(mesh, method="FEM")
   problem.add_variable("velocity_x")
   problem.add_variable("velocity_y")
   problem.add_variable("pressure", order="first")

The ``fluid_dynamics`` physics with ``formulation="Taylor-Hood"``
declares the variables in this way itself.

The number of variables is limited by the automatic differentiation budget: the
Jacobian is seeded with one derivative slot per local degree of freedom of an
element (one per node and variable, the corners only for a first-order
variable), and the default budget is 96 slots.  A ``Quad9`` mesh therefore
supports ten variables, a ``Hex27`` mesh three, and the Taylor-Hood element on
``Hex27`` needs :math:`27 \times 3 + 8 = 89`.  Exceeding the budget raises an
immediate error that names the element, the arithmetic and the CMake setting
that raises the limit.

The four kinds of object
------------------------

Everything else about a problem is added as a named, registered *object* with
validated parameters.
There are six kinds, and they differ in the part of the residual to which they
contribute.

A **kernel** contributes a volume term.  Every equation is written in the
canonical conservation form

.. math::

   -\nabla \cdot \mathbf{F}(u, \nabla u, \mathbf{x}, t)
   + S(u, \nabla u, \mathbf{x}, t) = 0 ,

and a kernel supplies a flux :math:`\mathbf{F}`, a source :math:`S`, or both.
For example, ``diffusion`` supplies :math:`\mathbf{F} = \nabla u`,
``body_force`` supplies a source and ``time_derivative`` supplies the storage
term.  Several kernels acting on the same variable add their fluxes and sources
together, so that a convection-diffusion-reaction equation is assembled from
three separate, independently tested kernels.

An **integrated boundary condition** contributes a surface term, evaluated over
the faces of a side set.  ``Neumann`` prescribes the normal
flux, ``Robin`` and ``radiative_heat_flux`` make it depend on the local
value,
and ``pressure`` prescribes a traction that follows the
surface normal.

A **nodal boundary condition** replaces an equation, whereas an integrated
boundary condition adds to it.  ``Dirichlet`` is the only
common one: the residual row of a constrained degree of freedom is replaced by
:math:`u - g = 0`, and the corresponding Jacobian row by the identity.

A **property** object computes properties at the integration points and makes them
available to the kernels by name.  ``small_strain_stress`` computes ``stress``
and ``strain`` from the displacement gradients, and
``constant_property`` declares named constants.  The property objects are
evaluated before the kernels at every integration point, so a kernel may depend
on a property that depends on the solution, and the automatic differentiation
carries the dependence through into the Jacobian without additional code.

A **nodal load** applies a concentrated force or flux at a single node,
identified either by node number or by the coordinates of the nearest node.

A **constraint** joins degrees of freedom.  ``periodic``
makes the field on a secondary boundary equal to the field on a primary
boundary, carried over by a translation:

.. code-block:: python

   problem.add_boundary_condition("periodic", primary="left", secondary="right")

Every node :math:`s` of the secondary boundary is joined to the node :math:`p`
of the primary boundary at :math:`\mathbf{x}_s - \mathbf{t}`, where
:math:`\mathbf{t}` is the translation (by default the vector between the
centroids of the two boundaries, or the ``translation`` given).  The two act as
one unknown: the equation of :math:`s` is added to that of :math:`p`, so that
the control domain of the joined node is the union of the two half domains,
and the Jacobian becomes :math:`P^{\mathsf{T}} J P`, where :math:`P` maps the
unknowns onto the nodes and keeps a symmetric matrix symmetric.  A corner of a
domain periodic in two directions is joined to a single primary node.  The two
boundaries must match node by node after the translation, and a value
prescribed on a secondary node must also be prescribed on its primary.  The
condition works with ``FEM`` and ``DMCDM``, serially and on any number of
processes.  On a manufactured solution periodic in one direction and in two,
both methods converge at second order.

.. code-block:: python

   problem.add_kernel("heat_conduction", variable="temperature", thermal_conductivity=20.0)
   problem.add_kernel("body_force", variable="temperature", value=1.0e6)  # the heat source
   problem.add_boundary_condition("Dirichlet", "left", variable="temperature", value=40.0)
   problem.add_boundary_condition(
       "Robin", "top", variable="temperature", transfer_coeff=75.0, ambient_value=20.0
   )

The object level names each object by its type, as the C++ library does, and
gives its data as keywords.  ``variable`` takes one variable or a list, which
adds the object to each.  A call returns the object, which
:meth:`~dualmesh.Problem.boundary_flux_integral` takes.

Parameters every object accepts
-------------------------------

Four parameters are declared by the base class and so are accepted by every
kernel, boundary condition and property object.

``block``
   A list of block names or numbers this object acts on.  The default, an empty
   list, means everywhere.  This is how two materials (e.g., two sets of properties) are given to two parts of
   a mesh.

``quadrature``
   The rule used to integrate this object's contribution.  The default,
   ``"automatic"``, is Gauss-Legendre with one more point per direction than the
   polynomial order of the mesh: two points on a linear mesh and three on a
   quadratic one.  The explicit choices are ``"gauss1"`` to ``"gauss20"``,
   ``"midpoint"``, ``"trapezoid"``, ``"simpson"``, and three rules that lump the
   integrand onto the nodes: ``"nodal"``, ``"interface"`` and
   ``"control_domain_trapezoid"``.  The lumped rules are provided because the
   finite volume formulations of [Reddy2024]_ use them, and because a
   lumped capacity matrix makes an explicit time step stable.  They are
   described in :doc:`/theory/elements`.

``reduced_integration``
   When true, the *solution* is evaluated at the element centroid while the
   integration still uses the rule named by ``quadrature``.  This is selective
   reduced integration [HughesCohenHaroun1978]_, and it exists to remove
   locking: turn it on for the transverse shear term of a thin beam or plate and
   for the penalty term of an incompressible flow.  Turning it on for a bending
   or diffusion term instead degrades the accuracy, so it is off by default and
   should be switched on deliberately.

``scale_with_load``
   When true, this object's contribution is multiplied by the load factor during
   load stepping.  Use it on the terms that represent the applied load, so that
   load stepping ramps the load while the stiffness keeps its full value.

What a parameter value may be
-----------------------------

Any parameter declared as a real number accepts four kinds of value.

.. code-block:: python

   # 1. A constant.
   problem.add_kernel("body_force", variable="u", value=2.5)

   # 2. An expression in x, y, z and t, given as text or compiled first.
   problem.add_kernel("body_force", variable="u", value="sin(pi*x) * sin(pi*y)")
   problem.add_kernel("body_force", variable="u", value=dm.parsed_function("sin(pi*x) * sin(pi*y)"))

   # 3. The name of a function registered on the problem.
   problem.add_function("ramp", lambda x, y, z, t: min(t, 1.0))
   problem.add_kernel("body_force", variable="u", value="ramp")

   # 4. A Python callable of (x, y, z, t).
   problem.add_kernel("body_force", variable="u", value=lambda x, y, z, t: x * x + y)

The four kinds give the same values but differ greatly in computational cost.
A constant and a parsed expression are evaluated in C++.  A Python callable,
whether registered by name or passed directly, is evaluated by calling back
into the interpreter, and because that requires the global interpreter lock it
**forces the whole assembly onto one thread**.  The problem reports this:

.. code-block:: python

   problem.thread_safe  # False once any Python callable is attached
   problem.effective_threads()  # 1, whatever set_num_threads was told

A string that is not the name of a registered function is compiled as an
expression.  If the string is neither a registered name nor a valid expression,
the error message states both the name that was looked up and the position at
which the expression fails to parse.  The expression language has the usual
arithmetic with ``^`` or ``**`` for powers, the comparisons ``<``, ``>``,
``<=``, ``>=``, ``==`` and ``!=`` (which give 1 or 0), the constants ``pi`` and
``e``, the variables ``x``, ``y``, ``z`` and ``t``, and the functions ``sin``,
``cos``, ``tan``, ``asin``, ``acos``, ``atan``, ``sinh``, ``cosh``, ``tanh``,
``exp``, ``log``, ``log10``, ``sqrt``, ``abs``, ``floor``, ``ceil``, ``erf``,
``sign``, ``atan2``, ``pow``, ``hypot``, ``min``, ``max`` and
``if(condition, a, b)``.  The text SymPy prints is accepted unchanged.  An
expression is compiled once, to a short program evaluated in C++, so that its
cost is comparable to that of a constant and the assembly remains threaded.

An expression is therefore preferable wherever the expression language suffices,
and a callable is needed only where it does not.  The same applies to a kernel,
property object or boundary condition written as a Python class: such a class is
suited to testing a new term, which should be moved into C++ once its form is
settled.

Finding out what exists
-----------------------

The registry can be queried, so that the documentation of an object always
matches the object itself.

.. code-block:: python

   dm.registered_types()  # every object name
   dm.object_category("heat_conduction")  # 'Kernel'
   dm.object_module("heat_conduction")  # 'heat_transfer'
   print(dm.describe_object("heat_conduction"))

``describe_object`` prints the class description, then every parameter with its
kind, whether it is required, its default and its meaning.  The same information
is rendered as one page per object under :doc:`/objects`.  Those pages are
generated from the registry when the documentation is built, so that they always
agree with the code.

Writing a kernel in Python
--------------------------

A new term can be tested without modifying the C++ code.  Subclass
:class:`dualmesh.PythonKernel` and supply a flux, a source, or both, in terms of
the integration-point context.

.. code-block:: python

   import dualmesh as dm


   class NonlinearConduction(dm.PythonKernel):
       """F = k(u) grad u with k(u) = k0 (1 + beta u)."""

       def __init__(self, variable, k0, beta):
           super().__init__(variable=variable)
           self.k0, self.beta = k0, beta

       def compute_flux(self, ctx):
           u = ctx.value(self.variable)
           k = self.k0 * (1.0 + self.beta * u)
           return [k * g for g in ctx.gradient(self.variable)]

The arithmetic inside ``compute_flux`` is done on :class:`dualmesh.ADReal`
numbers, which carry their derivatives with respect to the local degrees of
freedom [Wengert1964]_.  No derivative has to be written by hand: the resulting
Jacobian is exact, and Newton's method therefore converges quadratically.  Use
the functions in the ``dm`` namespace (``dm.exp``, ``dm.sqrt``, ``dm.tanh`` and
the rest), because the functions in ``math`` or ``numpy`` do not propagate
derivatives.
