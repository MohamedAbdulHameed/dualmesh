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
       x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
       num_x_elements=10, num_y_elements=10)

   problem = dm.Problem(mesh, method="dmcdm", coordinates="cartesian")

``method`` chooses the discretisation.  All four discretisations read the same
problem description, so that a comparison of the methods requires a change of
this keyword alone.

``"dmcdm"``
   The dual mesh control domain method of [Reddy2019a]_ and [Reddy2024]_, and
   the default.  The unknowns are located at the mesh nodes.  The balance law is
   integrated over each node's control domain, and the flux at a control domain
   interface is taken from the element interpolation.

``"fem"``
   The standard Galerkin finite element method, provided so that the two can be
   compared on identical input.  The unknowns are again located at the nodes.

``"hfvm"``
   The vertex-centred finite volume method, the "half control volume"
   formulation of Chapter 3 of [Reddy2024]_.  Its control volumes are the same
   dual mesh control domains, but the flux at a face is computed from a
   two-point difference along the edge joining the two nodes, whereas the dual
   mesh control domain method takes it from the element interpolation.  See
   :doc:`/theory/finite_volume`.

``"zfvm"``
   The cell-centred finite volume method, the "zero-thickness control volume"
   formulation.  It is the only method whose unknowns are located away from the
   mesh nodes: there is one unknown at every element centroid and one at every
   boundary face centroid.  Code that reads results back must therefore use the
   points returned by :meth:`~dualmesh.Problem.entity_points` in place of the
   mesh nodes (see :doc:`output`).

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

``boundary_gradient`` applies only to the cell-centred finite volume method and
selects how the gradient at a boundary face is reconstructed.  ``"first_order"``,
the default, uses the difference between the face value and the owning cell
value over the normal distance.  ``"second_order"`` also uses the neighbouring
cell, which makes it exact for a quadratic normal profile and noticeably more
accurate on a coarse mesh.  The two are compared in
:doc:`/theory/finite_volume`.

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
   problem = dm.Problem(mesh, method="fem")
   problem.add_variable("velocity_x")
   problem.add_variable("velocity_y")
   problem.add_variable("pressure", order="first")

In an input file the same is ``pressure: {order: first}`` in the
``variables`` block.  ``physics.add_incompressible_flow(...,
formulation="taylor_hood")`` declares the variables this way itself.

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
validated parameters, a design taken from MOOSE [MOOSE2025]_ [MOOSE2020]_.
There are four kinds, and they differ in the part of the residual to which they
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
the faces of a side set.  ``Neumann_boundary_condition`` prescribes the normal
flux, ``Robin_boundary_condition`` and
``convective_heat_flux_boundary_condition`` make it depend on the local value,
and ``pressure_boundary_condition`` prescribes a traction that follows the
surface normal.

A **nodal boundary condition** replaces an equation, whereas an integrated
boundary condition adds to it.  ``Dirichlet_boundary_condition`` is the only
common one: the residual row of a constrained degree of freedom is replaced by
:math:`u - g = 0`, and the corresponding Jacobian row by the identity.

A **material** computes properties at the integration points and makes them
available to the kernels by name.  ``linear_elastic_stress`` computes ``stress``
and ``strain`` from the displacement gradients, and
``generic_constant_material`` declares named constants.  Materials are
evaluated before the kernels at every integration point, so a kernel may depend
on a property that depends on the solution, and the automatic differentiation
carries the dependence through into the Jacobian without additional code.

A **nodal load** applies a concentrated force or flux at a single node,
identified either by node number or by the coordinates of the nearest node.

.. code-block:: python

   problem.add_kernel("heat_conduction", "conduction",
                      variable="temperature", thermal_conductivity=20.0)
   problem.add_kernel("heat_source", "source",
                      variable="temperature", heat_source=1.0e6)
   problem.add_boundary_condition("Dirichlet_boundary_condition", "cold",
                                  variable="temperature",
                                  boundary="left", value=40.0)
   problem.add_boundary_condition("convective_heat_flux_boundary_condition", "film",
                                  variable="temperature", boundary="top",
                                  heat_transfer_coefficient=75.0,
                                  ambient_temperature=20.0)

The second argument of each call is the object's name.  It is optional, and one
is generated when it is omitted, but explicit names are recommended: the name
appears in error messages, it is the argument that
:meth:`~dualmesh.Problem.boundary_flux_integral` takes, and it makes a long
problem definition readable.  When a boundary condition is given no
``boundary`` parameter, its name is taken as its boundary, so that a condition
named after a side set (e.g., ``"left"``) acts on that side set.

Parameters every object accepts
-------------------------------

Four parameters are declared by the base class and so are accepted by every
kernel, boundary condition and material.

``block``
   A list of block names or numbers this object acts on.  The default, an empty
   list, means everywhere.  This is how two materials are given to two parts of
   a mesh.

``quadrature``
   The rule used to integrate this object's contribution.  The default,
   ``"automatic"``, is Gauss-Legendre with one more point per direction than the
   polynomial order of the mesh: two points on a linear mesh and three on a
   quadratic one.  The explicit choices are ``"gauss1"`` to ``"gauss10"``,
   ``"midpoint"``, ``"trapezoid"``, ``"simpson"``, and three rules that lump the
   integrand onto the nodes: ``"nodal"``, ``"interface"`` and
   ``"control_domain_trapezoid"``.  The lumped rules are provided because the
   finite volume tables of Chapter 3 of [Reddy2024]_ use them, and because a
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
   problem.add_kernel("body_force", "constant", variable="u", value=2.5)

   # 2. An expression in x, y, z and t, given as text or compiled first.
   problem.add_kernel("body_force", "parsed", variable="u",
                      value="sin(pi*x) * sin(pi*y)")
   problem.add_kernel("body_force", "compiled", variable="u",
                      value=dm.parsed_function("sin(pi*x) * sin(pi*y)"))

   # 3. The name of a function registered on the problem.
   problem.add_function("ramp", lambda x, y, z, t: min(t, 1.0))
   problem.add_kernel("body_force", "named", variable="u", value="ramp")

   # 4. A Python callable of (x, y, z, t).
   problem.add_kernel("body_force", "callable", variable="u",
                      value=lambda x, y, z, t: x * x + y)

The four kinds give the same values but differ greatly in computational cost.
A constant and a parsed expression are evaluated in C++.  A Python callable,
whether registered by name or passed directly, is evaluated by calling back
into the interpreter, and because that requires the global interpreter lock it
**forces the whole assembly onto one thread**.  The problem reports this:

.. code-block:: python

   problem.thread_safe          # False once any Python callable is attached
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
material or boundary condition written as a Python class: such a class is
suited to testing a new term, which should be moved into C++ once its form is
settled.

Finding out what exists
-----------------------

The registry can be queried, so that the documentation of an object always
matches the object itself.

.. code-block:: python

   dm.registered_types()                       # every object name
   dm.object_category("heat_conduction")        # 'Kernel'
   dm.object_module("heat_conduction")          # 'heat_transfer'
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

The convenience module
----------------------

``dualmesh.physics`` assembles the objects of a standard model in one call, for
the cases where the pieces are always the same.

.. code-block:: python

   from dualmesh import physics

   physics.add_plane_elasticity(problem, ["disp_x", "disp_y"],
                                youngs_modulus=200.0e9, poissons_ratio=0.3,
                                formulation="plane_strain")
   physics.add_incompressible_flow(problem, ["velocity_x", "velocity_y"],
                                   dynamic_viscosity=1.0e-3, density=1000.0)

The module also has ``add_beam``, ``add_plate`` and ``add_circular_plate`` for
the structural theories of [ReddyBeams2022]_.

These are ordinary helpers: each one adds the same named objects that could be
added one by one, so that anything they set can afterwards be inspected,
replaced or added to.  They shorten the problem definition, and every object
they create remains visible in the problem.
