# SPDX-License-Identifier: LGPL-2.1-or-later
"""Worked usage examples for the generated syntax reference.

Every registered object has one entry.  The text is prose that explains when
the object is used and what its keywords do in that context, and the code
block below it is a complete, runnable fragment.  An object missing from this
file still receives a generated page without the usage section, and the
documentation build prints a warning that names the object, so that every
missing entry is reported.
"""

from __future__ import annotations

EXAMPLES: dict[str, str] = {}


def _add(name: str, text: str) -> None:
    EXAMPLES[name] = text.strip("\n")


# ---------------------------------------------------------------------------
# Framework
# ---------------------------------------------------------------------------
_add(
    "diffusion",
    """
``diffusion`` is the generic second-order operator.  It is the appropriate
kernel whenever the flux is proportional to the gradient of the unknown and no
physics module provides a kernel with a more specific name.  The diffusivity
may be a number, a named function of position and time, or a
property, and it may additionally be multiplied by a polynomial in the unknown
itself, which is the simplest way to make a problem nonlinear:

.. code-block:: python

   problem.add_kernel("diffusion", "conduction", variable="u", diffusivity=2.5)

   # A diffusivity that varies through the body.
   problem.add_function("k", lambda x, y, z, t: 1.0 + 4.0 * x)
   problem.add_kernel("diffusion", "graded", variable="u", diffusivity="k")

   # A diffusivity that depends on the solution: k_eff = 1.0 * (1 + 0.5 u).
   problem.add_kernel("diffusion", "nonlinear", variable="u", diffusivity=1.0, solution_polynomial=[1.0, 0.5])

The last form requires a nonlinear solve.  With ``nonlinear_solver="newton"``
the derivative of the polynomial is carried exactly through the automatic
differentiation, so the iteration converges quadratically.
""",
)

_add(
    "anisotropic_diffusion",
    """
Use ``anisotropic_diffusion`` when the conductivity or permeability depends on
direction, for example in a laminate, a fibre-reinforced solid, or a layered
rock.  Give the tensor either as its diagonal or in full:

.. code-block:: python

   # Orthotropic: ten times more conductive along x than along y.
   problem.add_kernel("anisotropic_diffusion", "conduction", variable="temperature", diffusivity_tensor=[10.0, 1.0])

   # The full tensor, row by row, for principal axes rotated by 30 degrees.
   import numpy as np

   c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
   k = np.array([[10.0, 0.0], [0.0, 1.0]])
   r = np.array([[c, -s], [s, c]])
   problem.add_kernel("anisotropic_diffusion", "rotated", variable="temperature", diffusivity_tensor=list((r @ k @ r.T).ravel()))
""",
)

_add(
    "reaction",
    """
``reaction`` adds a term proportional to the unknown, which models radioactive
decay, chemical consumption, heat loss along a fin, and the restoring force of
an elastic foundation.  The sign of the coefficient determines the direction
of the term: a positive coefficient removes the unknown.

.. code-block:: python

   # A cooling fin: -k A u'' + P beta u = 0, with P beta / (k A) = 400.
   problem.add_kernel("diffusion", "conduction", variable="u")
   problem.add_kernel("reaction", "convection", variable="u", coefficient=400.0)

   # A beam on an elastic foundation of modulus 1e6.
   problem.add_kernel("reaction", "foundation", variable="deflection", coefficient=1.0e6)

   # A second-order chemical reaction, which is nonlinear.
   problem.add_kernel("reaction", "consumption", variable="concentration", coefficient=0.1, exponent=2.0)
""",
)

_add(
    "body_force",
    """
``body_force`` is the generic distributed source.  Each physics module provides
an equivalent source with a physically descriptive name.  ``body_force``
accepts an arbitrary function, and the verification problems use it when the
source has no physical name:

.. code-block:: python

   problem.add_function("source", lambda x, y, z, t: 10.0 * np.cos(x))
   problem.add_kernel("body_force", "source", variable="u", value="source")

The quadrature keyword has a larger effect on this kernel than on any other,
because the source is the only term that the method integrates approximately.
Reddy's finite volume examples integrate it with the trapezoidal rule over the
whole control domain, which this library names ``control_domain_trapezoid``:

.. code-block:: python

   problem.add_kernel("body_force", "source", variable="u", value="source", quadrature="control_domain_trapezoid")
""",
)

_add(
    "advection",
    """
``advection`` transports the unknown with a prescribed velocity, so together
with ``diffusion`` it forms the advection-diffusion equation.  The velocity may
be constant or given by functions, one per component:

.. code-block:: python

   problem.add_kernel("diffusion", "diffusion", variable="c", diffusivity=1.0)
   problem.add_kernel("advection", "transport", variable="c", velocity=[75.0, 0.0])

At a high cell Peclet number, here :math:`Pe = 75`, the central differencing
that this kernel performs produces oscillations on a coarse mesh.  Refining the
mesh until the cell Peclet number is below about two removes the cause of the
oscillations.  Alternatively, the oscillations can be accepted and reported
with the results.

Choose ``form="conservative"`` when the velocity field is divergence free and
the discrete equations must conserve the transported quantity exactly over
each control domain.
""",
)

_add(
    "coupled_force",
    """
``coupled_force`` is the simplest way to couple two equations: it adds to one
equation a source proportional to another variable.  It represents, e.g., a
heat source proportional to a concentration, or a simple two-species
reaction:

.. code-block:: python

   problem.add_variable("a")
   problem.add_variable("b")
   problem.add_kernel("diffusion", "diff_a", variable="a")
   problem.add_kernel("diffusion", "diff_b", variable="b")
   problem.add_kernel("coupled_force", "b_feeds_a", variable="a", coupled_variable="b", coefficient=-1.0)

The derivative with respect to the coupled variable is carried through the
automatic differentiation, so the off-diagonal block of the Jacobian is exact
and Newton's method converges quadratically even for a strongly coupled system.
""",
)

_add(
    "time_derivative",
    """
``time_derivative`` makes a steady problem transient.  Add it to every
equation that evolves in time.  An equation without it remains a constraint,
which is what a mixed formulation requires for its moment equation.

.. code-block:: python

   problem.add_variable("u", initial_condition=lambda x, y, z, t: np.sin(np.pi * x))
   problem.add_kernel("diffusion", "diffusion", variable="u")
   problem.add_kernel("time_derivative", "time", variable="u", coefficient=1.0)
   problem.solve_transient(end_time=0.1, time_step=0.001, implicitness=0.5)

The capacity is lumped, as is common in explicit and nearly explicit schemes,
by a change of the quadrature:

.. code-block:: python

   problem.add_kernel("time_derivative", "time", variable="u", quadrature="nodal")
""",
)

_add(
    "Dirichlet_boundary_condition",
    """
``Dirichlet_boundary_condition`` prescribes the value of a variable, which is
the essential member of the duality pair.  The value is written directly into
the solution vector and the equation of that degree of freedom is replaced, so
the residual of the replaced equation becomes available afterwards as the
reaction:

.. code-block:: python

   problem.add_boundary_condition("Dirichlet_boundary_condition", "hot", variable="temperature", boundary="left", value=300.0)

   # A value that varies along the boundary.
   problem.add_function("profile", lambda x, y, z, t: np.cos(np.pi * x / 6.0))
   problem.add_boundary_condition("Dirichlet_boundary_condition", "top", variable="temperature", value="profile")

   # One value on several variables: the condition is created once per variable.
   problem.add_boundary_condition("Dirichlet_boundary_condition", "bottom", variables=["u", "v"], value=0.0)

   # The heat that must flow in through the left edge to hold it at 300.
   print(problem.total_reaction("temperature", "left"))

A boundary without a boundary condition receives the natural condition with a
zero secondary variable, i.e., an insulated surface in heat transfer and a
traction-free surface in elasticity.
""",
)

_add(
    "Neumann_boundary_condition",
    """
``Neumann_boundary_condition`` prescribes the secondary variable of the duality
pair, the normal flux :math:`n \\cdot F`.  Because the canonical flux of a
diffusion problem is :math:`F = k \\nabla u`, a positive value is a flux of the
unknown *into* the body:

.. code-block:: python

   # One watt per square metre entering through the right edge.
   problem.add_boundary_condition("Neumann_boundary_condition", "inflow", variable="temperature", boundary="right", flux=1.0)

A boundary without a boundary condition has a zero flux, so an insulated or
symmetry boundary requires no object.  Add this object only when the flux is
non-zero or when a named object makes the input clearer.
""",
)

_add(
    "Robin_boundary_condition",
    """
``Robin_boundary_condition`` is the mixed condition
:math:`n \\cdot F = q_0 - h (u - u_\\infty)`, which describes convection into a
surrounding medium, a contact resistance, and a spring support.  It is the
general form of which ``Neumann_boundary_condition`` is the special case
:math:`h = 0`:

.. code-block:: python

   # Convection from a fin tip into air at 20 degrees.
   problem.add_boundary_condition("Robin_boundary_condition", "tip", variable="temperature", boundary="right", transfer_coefficient=2.0, ambient_value=20.0)

For heat transfer, ``convective_heat_flux_boundary_condition`` expresses the
same condition in the terminology of heat transfer and is the preferred
object.
""",
)

_add(
    "point_source",
    """
``point_source`` applies a concentrated load or a point source of heat at one or
more nodes.  The load is applied at the node nearest to the given point,
without a tolerance, so the point should coincide with a node of the mesh:

.. code-block:: python

   # A concentrated force at the centre of the span.
   problem.add_nodal_load("point_source", "tip_load", variable="deflection", value=-1000.0, points=[0.5, 0.0, 0.0])

   # A heat source at every node of a named node set.
   problem.add_nodal_load("point_source", "heaters", variable="temperature", value=50.0, boundary="heater_nodes")

In a plane-stress or plane-strain model a traction is multiplied by the
thickness, while the value of a point source is applied as given, so give the
total force through the thickness.
""",
)

_add(
    "constant_property",
    """
``constant_property`` defines named constant properties that kernels
read.  It allows several kernels to share one value, and a property to take
different values in different blocks:

.. code-block:: python

   problem.add_property("constant_property", "copper", property_names=["conductivity"], property_values=[400.0], block=["copper_bar"])
   problem.add_property("constant_property", "steel", property_names=["conductivity"], property_values=[45.0], block=["steel_bar"])
   problem.add_kernel("heat_conduction", "conduction", variable="temperature", thermal_conductivity_property="conductivity")
""",
)

_add(
    "function_property",
    """
``function_property`` is the counterpart of
``constant_property`` for a property that varies through the body.
The functions must be registered on the problem first:

.. code-block:: python

   problem.add_function("k_of_x", lambda x, y, z, t: 50.0 * (1.0 + x))
   problem.add_property("function_property", "graded", property_names=["conductivity"], functions=["k_of_x"])

A property that depends on the *solution* is given as a ``parsed_property``,
or as a Python subclass of ``PythonProperty``, as shown in
:doc:`/user_guide/problem_setup`.
""",
)

# ---------------------------------------------------------------------------
# Heat transfer
# ---------------------------------------------------------------------------
_add(
    "heat_conduction",
    """
``heat_conduction`` is Fourier's law of heat conduction and is the starting
point of almost every thermal analysis:

.. code-block:: python

   problem.add_variable("temperature")
   problem.add_kernel("heat_conduction", "conduction", variable="temperature", thermal_conductivity=20.0)

A temperature-dependent conductivity is written as a polynomial multiplying the
base value, which is the form Reddy's nonlinear examples use:

.. code-block:: python

   # k(T) = 45 (1 + 0.002 T)
   problem.add_kernel("heat_conduction", "conduction", variable="temperature", thermal_conductivity=45.0, temperature_polynomial=[1.0, 0.002])
   problem.solve(nonlinear_solver="newton")
""",
)

_add(
    "heat_source",
    """
``heat_source`` is internal heat generation, e.g., ohmic heating in a
conductor, the absorption of radiation in a shield, or the heat released by
curing in a resin.

.. code-block:: python

   problem.add_kernel("heat_source", "joule", variable="temperature", heat_source=1.0e6)

   # Generation that decays with depth.
   problem.add_function("q", lambda x, y, z, t: 2.0e6 * np.exp(-x / 0.01))
   problem.add_kernel("heat_source", "absorption", variable="temperature", heat_source="q")
""",
)

_add(
    "heat_conduction_time_derivative",
    """
``heat_conduction_time_derivative`` is the storage term of the energy equation
and makes a thermal problem transient:

.. code-block:: python

   problem.add_kernel("heat_conduction", "conduction", variable="temperature", thermal_conductivity=45.0)
   problem.add_kernel("heat_conduction_time_derivative", "storage", variable="temperature", density=7850.0, specific_heat=460.0)
   problem.solve_transient(end_time=600.0, time_step=5.0, implicitness=0.5)

Only the product of density and specific heat enters, so a volumetric heat
capacity may be given as the density with the specific heat set to one.
""",
)

_add(
    "heat_flux_boundary_condition",
    """
``heat_flux_boundary_condition`` prescribes the heat flux entering a surface.
It applies, e.g., a heater, a measured flux, or an absorbed radiation load:

.. code-block:: python

   problem.add_boundary_condition("heat_flux_boundary_condition", "heater", variable="temperature", boundary="bottom", heat_flux=5.0e4)

An insulated surface is the default and needs no object.
""",
)

_add(
    "convective_heat_flux_boundary_condition",
    """
``convective_heat_flux_boundary_condition`` is Newton's law of cooling and is
the most common thermal boundary condition in practice, because both the
temperature of a surface exposed to a fluid and the heat flux through it are
unknown in advance:

.. code-block:: python

   problem.add_boundary_condition("convective_heat_flux_boundary_condition", "air", variable="temperature", boundary=["right", "top"], heat_transfer_coefficient=25.0, ambient_temperature=20.0)

The condition is linear in the temperature, so it requires no additional
Newton iterations.
""",
)

_add(
    "radiative_heat_flux_boundary_condition",
    """
``radiative_heat_flux_boundary_condition`` adds grey-body radiation to a
surface.  It is strongly nonlinear, because the flux varies with the fourth
power of the temperature, and it requires temperatures on an absolute scale:

.. code-block:: python

   # A surface at a few hundred kelvin radiating to surroundings at 300 K.
   problem.add_boundary_condition("radiative_heat_flux_boundary_condition", "radiation", variable="temperature", boundary="top", emissivity=0.8, ambient_temperature=300.0)
   problem.solve(nonlinear_solver="newton")

Radiation and convection usually act together.  When both objects are added
on the same side set, their fluxes are summed.
""",
)

# ---------------------------------------------------------------------------
# Solid mechanics
# ---------------------------------------------------------------------------
_add(
    "linear_elastic_stress",
    """
``linear_elastic_stress`` computes the stress from the displacement gradients
and provides it as the property that ``stress_divergence`` reads.
One instance supplies the stress to all the displacement equations:

.. code-block:: python

   for name in ("displacement_x", "displacement_y"):
       problem.add_variable(name)
   problem.add_property("linear_elastic_stress", "steel", displacements=["displacement_x", "displacement_y"], formulation="plane_stress", youngs_modulus=200.0e9, poissons_ratio=0.3)
   for component, name in enumerate(("displacement_x", "displacement_y")):
       problem.add_kernel("stress_divergence", f"equilibrium_{name}", variable=name, component=component)

Thermal strain is included by naming a temperature variable *and* giving a
non-zero expansion coefficient:

.. code-block:: python

   problem.add_property("linear_elastic_stress", "steel", displacements=["displacement_x", "displacement_y"], youngs_modulus=200.0e9, poissons_ratio=0.3, temperature="temperature", thermal_expansion_coefficient=1.2e-5, stress_free_temperature=20.0)

The ``solid_mechanics`` physics adds the stress property object and all the kernels, and the
``thermal_expansion`` coupling adds the thermal strain (see
:doc:`/user_guide/problem_setup`).
""",
)

_add(
    "stress_divergence",
    """
``stress_divergence`` is the equilibrium equation of one displacement component.
Add one instance per displacement, with ``component`` matching the variable:

.. code-block:: python

   problem.add_kernel("stress_divergence", "equilibrium_x", variable="displacement_x", component=0, thickness=0.01)
   problem.add_kernel("stress_divergence", "equilibrium_y", variable="displacement_y", component=1, thickness=0.01)

The thickness multiplies the whole equation in a plane problem and must match
the thickness given to every traction and pressure boundary condition of the
same model.  In an axisymmetric problem keep it at one, because the
integration measure already carries the factor :math:`2 \\pi r`.
""",
)

_add(
    "traction_boundary_condition",
    """
``traction_boundary_condition`` applies a surface traction component.  The
object has no ``component`` keyword: the component is that of the equation
named by ``variable``.  The load is given either as the traction itself, in
force per unit area, or as the total force on the boundary, which the code
spreads uniformly over the area of the boundary.  ``traction`` and
``total_force`` define the same load, so a condition gives one of them and
never both:

.. code-block:: python

   # A uniform shear of 1 MPa on the right edge, acting along +y.
   problem.add_boundary_condition("traction_boundary_condition", "shear", variable="displacement_y", boundary="right", traction=1.0e6, thickness=0.01)

   # A force of 150 N along -y on the side set 'grip', whatever its area.
   problem.add_boundary_condition("traction_boundary_condition", "grip", variable="v", total_force=-150.0)
""",
)

_add(
    "fixed_constraint",
    """
``fixed_constraint`` clamps a boundary: every displacement listed in
``displacements`` is zero there.  It is the usual support of a solid that is
held by a much stiffer part, e.g., a bolted or welded face.  It is equivalent
to ``Dirichlet_boundary_condition`` with ``variables`` set to the same list
and ``value`` 0, and the two define the same constraint, so a boundary is
given one of them and never both:

.. code-block:: python

   # The jaws of a wrench held by a rigid nut.
   problem.add_boundary_condition("fixed_constraint", "jaws", displacements=["u", "v", "w"])
""",
)

_add(
    "pressure_boundary_condition",
    """
``pressure_boundary_condition`` applies a pressure normal to a surface, which
is the usual load on a vessel, a dam, or the surface of a hole in a plate.  It
requires one instance per displacement component, each with the ``component``
to which it contributes:

.. code-block:: python

   for component, name in enumerate(("displacement_x", "displacement_y")):
       problem.add_boundary_condition("pressure_boundary_condition", f"internal_{name}", variable=name, boundary="inner", component=component, pressure=10.0e6)

A positive pressure acts inward, opposite to the outward normal.
""",
)

# ---------------------------------------------------------------------------
# Beams and plates
# ---------------------------------------------------------------------------
_add(
    "beam_Euler_Bernoulli_mixed",
    """
The classical beam theory gives a fourth-order equation in the deflection.
The dual mesh control domain method discretizes second-order equations, so the
bending moment is carried as a third unknown and the system becomes three
second-order equations.  Add the kernel once per variable, with identical parameters:

.. code-block:: python

   for name in ("axial", "deflection", "moment"):
       problem.add_variable(name)
   for name in ("axial", "deflection", "moment"):
       problem.add_kernel("beam_Euler_Bernoulli_mixed", f"beam_{name}", variable=name, axial_displacement="axial", transverse_displacement="deflection", bending_moment="moment", extensional_stiffness=a_xx, bending_stiffness=d_xx, transverse_load=-1.0e3)

The boundary conditions follow the duality pairs: a clamped end prescribes the
deflection and leaves the moment free, because the vanishing slope is the
natural condition of the moment equation.  A simply supported end prescribes
the deflection *and* prescribes the moment to be zero.

The ``beam`` physics adds all three kernels.
""",
)

_add(
    "beam_Timoshenko_displacement",
    """
The shear-deformable beam in displacement form carries the axial displacement,
the deflection, and the rotation of the cross-section.  A thin beam exhibits
shear locking unless the shear terms are integrated with one point, and the
kernel is therefore split into two instances:

.. code-block:: python

   common = dict(axial_displacement="axial", transverse_displacement="deflection", rotation="rotation", extensional_stiffness=a_xx, bending_stiffness=d_xx, shear_stiffness=s_xz, transverse_load=-1.0e3)
   for name in ("axial", "deflection", "rotation"):
       problem.add_kernel("beam_Timoshenko_displacement", f"bend_{name}", variable=name, shear_treatment="exclude", **common)
       problem.add_kernel("beam_Timoshenko_displacement", f"shear_{name}", variable=name, shear_treatment="only", quadrature="midpoint", reduced_integration=True, **common)

Using ``shear_treatment`` with only one of the two kernels omits terms and
gives a wrong result.
""",
)

_add(
    "beam_Timoshenko_mixed",
    """
The mixed form of the shear-deformable beam carries the axial displacement,
the deflection, and the bending moment as unknowns.  It is free of shear
locking with full integration of all terms, which makes it the more robust of
the two Timoshenko models, and the rotation is recovered afterwards from
:math:`\\phi = -dw/dx + (1/S)\\,dM/dx`:

.. code-block:: python

   for name in ("axial", "deflection", "moment"):
       problem.add_kernel("beam_Timoshenko_mixed", f"beam_{name}", variable=name, axial_displacement="axial", transverse_displacement="deflection", bending_moment="moment", extensional_stiffness=a_xx, bending_stiffness=d_xx, shear_stiffness=s_xz, transverse_load=-1.0e3)
""",
)

_add(
    "circular_plate_first_order",
    """
This object is the shear-deformable model of an axisymmetric circular plate,
solved on a one-dimensional radial mesh with the axisymmetric coordinate
system.  As for the shear-deformable beam, the shear terms require their own
reduced-integration kernel:

.. code-block:: python

   mesh = dm.generate_line_mesh(start=0.0, end=radius, num_elements=32)
   problem = dm.Problem(mesh, coordinates="axisymmetric")
   plate = problem.add_physics("circular_plate", "plate", theory="first_order", extensional_stiffness=a_rr, coupling_stiffness=b_rr, bending_stiffness=d_rr, shear_stiffness=s_rz, poissons_ratio=0.3, transverse_load=q0)

Symmetry at the centre prescribes the radial displacement and the rotation to
be zero.  The edge condition prescribes the deflection, and a clamped edge
additionally prescribes the rotation.
""",
)

_add(
    "circular_plate_classical_mixed",
    """
The classical, shear-rigid model of an axisymmetric circular plate, written in
terms of the radial displacement, the deflection, and the radial bending
moment.  This is Reddy's DM-CP(M) model:

.. code-block:: python

   mesh = dm.generate_line_mesh(start=0.0, end=radius, num_elements=32)
   problem = dm.Problem(mesh, coordinates="axisymmetric")
   plate = problem.add_physics("circular_plate", "plate", theory="classical", extensional_stiffness=a_rr, coupling_stiffness=b_rr, bending_stiffness=d_rr, poissons_ratio=0.3, transverse_load=q0)
   plate.add_boundary_condition("Dirichlet_boundary_condition", "left", variable="radial_displacement", value=0.0)
   plate.add_boundary_condition("Dirichlet_boundary_condition", "right", variables=["radial_displacement", "deflection"], value=0.0)
   # A simply supported edge adds: bending_moment = 0 on "right".
   # A clamped edge adds nothing: the vanishing slope is natural there.

The moment is a nodal unknown, so it is available directly, without
differentiation of the deflection.  It is accurate everywhere except at the
centre, which is a known property of mixed models that Reddy states in
Section 8.6.
""",
)

_add(
    "plate_first_order",
    """
The shear-deformable rectangular plate carries five variables: two in-plane
displacements, the deflection, and two rotations.  With ``von_karman=True`` it
models moderate rotations, as required for a thin plate under a load large
enough to stretch its mid-plane:

.. code-block:: python

   plate = problem.add_physics("plate", "plate", in_plane_displacements=["u", "v"], transverse_displacement="w", rotations=["phi_x", "phi_y"], extensional_stiffness=a, coupling_stiffness=b, bending_stiffness=d, shear_stiffness=s, poissons_ratio=0.3, transverse_load=q0, von_karman=True)
   problem.solve(load_factors=[0.1 * k for k in range(1, 11)])

The nonlinear case requires load stepping, because a solve that starts from
the undeformed state at the full load usually fails to converge.
""",
)

# ---------------------------------------------------------------------------
# Fluids
# ---------------------------------------------------------------------------
_add(
    "viscous_stress",
    """
``viscous_stress`` is the viscous term of the momentum equation.  Together with
the penalty term it describes Stokes flow, and the addition of
``convective_inertia`` gives the Navier-Stokes equations:

.. code-block:: python

   velocities = ["velocity_x", "velocity_y"]
   for component, name in enumerate(velocities):
       problem.add_kernel("viscous_stress", f"viscous_{name}", variable=name, velocities=velocities, component=component, dynamic_viscosity=1.0)

The ``incompressible_flow`` physics adds the complete set of kernels.
""",
)

_add(
    "penalty_incompressibility",
    """
The penalty formulation replaces the pressure by :math:`P = -\\gamma \\nabla
\\cdot v`, which eliminates the pressure unknown and leaves a problem in the
velocities alone.  The penalty term must then be under-integrated:

.. code-block:: python

   for component, name in enumerate(velocities):
       problem.add_kernel("penalty_incompressibility", f"penalty_{name}", variable=name, velocities=velocities, component=component, penalty_parameter=1.0e8)

The default settings select the one-point rule and the centroid evaluation.  A
velocity field that is identically zero almost certainly indicates that the
penalty parameter is too large for the viscosity and that the constraint has
locked the element.
""",
)

_add(
    "convective_inertia",
    """
``convective_inertia`` adds the nonlinear transport term of the Navier-Stokes
equations.  A Stokes flow is modelled by omitting this kernel, which is
preferable to setting its density to zero:

.. code-block:: python

   for component, name in enumerate(velocities):
       problem.add_kernel("convective_inertia", f"inertia_{name}", variable=name, velocities=velocities, component=component, density=1.0)
   problem.solve(nonlinear_solver="newton", load_factors=[0.1, 0.25, 0.5, 0.75, 1.0])

At a high Reynolds number, a solve that starts from rest at the full velocity
usually diverges.  Either increase the boundary velocity in increments with
load stepping, as above, or use direct iteration with a relaxation factor of
about one half, which is the strategy of Reddy's cavity example.
""",
)

_add(
    "Boussinesq_buoyancy",
    """
``Boussinesq_buoyancy`` adds the buoyancy force that a temperature field exerts
on the flow.  Add it to the momentum equation of every velocity component that
has a component of gravity, usually only the vertical one.  In the
non-dimensional form of the natural convection benchmark (lengths scaled by
the cavity width, velocities by :math:`\\kappa / L`, temperature by the wall
temperature difference) the coefficient :math:`\\rho_0 \\beta g` becomes :math:`Ra \\, Pr`:

.. code-block:: python

   problem.add_kernel("Boussinesq_buoyancy", "buoyancy", variable="v", component=1, temperature="temperature", gravity=[0.0, -1.0], thermal_expansion_coefficient=rayleigh * prandtl, scale_with_load=True)
   problem.solve(load_factors=[0.01, 0.1, 1.0])

With ``scale_with_load=True`` load stepping increases the Rayleigh number in
increments, which allows a strongly buoyant flow to be computed from a fluid
at rest.  The ``nonisothermal_flow`` coupling of a ``heat_transfer`` and an
``incompressible_flow`` physics adds this kernel when it is given ``gravity``
and ``thermal_expansion_coefficient``.  The complete problem is in
``examples/natural_convection.py``.
""",
)

_add(
    "heat_convection",
    """
``heat_convection`` is the complementary term of the coupling, through which
the flow transports heat.  The velocities are variables of the same problem,
so the Jacobian of Newton's method contains the coupling in both directions:

.. code-block:: python

   problem.add_kernel("heat_convection", "convection", variable="temperature", velocities=["u", "v"], density=1.0, specific_heat=1.0)

The ``nonisothermal_flow`` coupling of a ``heat_transfer`` and an
``incompressible_flow`` physics adds this kernel:

.. code-block:: python

   flow = problem.add_physics("incompressible_flow", "flow", velocities=["u", "v"], dynamic_viscosity=0.71, density=1.0)
   heat = problem.add_physics("heat_transfer", "heat", thermal_conductivity=1.0)
   problem.add_coupling("nonisothermal_flow", "coupling", heat_transfer="heat", incompressible_flow="flow")

The term is written in the advective (non-conservative) form, which is exact
for an incompressible flow.
""",
)

_add(
    "penalty_pressure",
    """
The penalty formulation eliminates the pressure from the unknowns, so the
pressure is recovered afterwards from the velocity divergence.
``penalty_pressure`` provides it as a property, which the
post-processing evaluates at the element centroids:

.. code-block:: python

   problem.add_property("penalty_pressure", "pressure", velocities=velocities, penalty_parameter=1.0e8)
   problem.solve()
   pressure = problem.property_at_centroids("pressure")

The penalty parameter must equal the one given to
``penalty_incompressibility``, otherwise the recovered pressure is meaningless.
""",
)

_add(
    "pressure_gradient",
    """
``pressure_gradient`` is the pressure part of the stress in one momentum
equation of the pressure-velocity formulation.  It is normally added by the
``incompressible_flow`` physics, together with the viscous stress and the
mass equation:

.. code-block:: python

   flow = problem.add_physics("incompressible_flow", "flow", velocities=["u", "v"], dynamic_viscosity=1.0, density=100.0, formulation="pressure", pressure_pin_point=(0.5, 0.0))

To add it manually, create one instance per velocity component:

.. code-block:: python

   problem.add_kernel("pressure_gradient", "grad_p_u", variable="u", pressure="pressure", component=0)
""",
)

_add(
    "mass_conservation",
    """
``mass_conservation`` is the mass equation, attached to the pressure variable.
With equal-order interpolation it must include the stabilisation, and every
force of the momentum equations must also be given to it:

.. code-block:: python

   problem.add_kernel("mass_conservation", "mass", variable="pressure", velocities=["u", "v"], density=100.0, dynamic_viscosity=1.0, body_force=["0", "-9.81"])
   problem.add_boundary_condition("mass_flux_boundary_condition", "mass_flux", variable="pressure", boundary=["left", "right", "bottom", "top"], velocities=["u", "v"])

In an enclosed flow the pressure level must also be fixed, with a
``point_Dirichlet_boundary_condition``.
""",
)

_add(
    "momentum_stabilization",
    """
``momentum_stabilization`` adds the streamline-upwind term to one momentum
equation of a flow with inertia.  It takes the same parameters as
``mass_conservation``, plus the component:

.. code-block:: python

   problem.add_kernel("momentum_stabilization", "supg_u", variable="u", component=0, pressure="pressure", velocities=["u", "v"], density=100.0, dynamic_viscosity=1.0)
""",
)

_add(
    "mass_flux_boundary_condition",
    """
``mass_flux_boundary_condition`` adds to the mass equation the flow through
the boundary, computed from the discrete velocity.  Apply it on every boundary
without a prescribed pressure.  The ``incompressible_flow`` physics applies it
on the side sets of its ``mass_flux_boundaries``, by default every side set:

.. code-block:: python

   problem.add_boundary_condition("mass_flux_boundary_condition", "mass_flux", variable="pressure", boundary=["inlet", "outlet", "walls"], velocities=["u", "v"])
""",
)

_add(
    "point_Dirichlet_boundary_condition",
    """
``point_Dirichlet_boundary_condition`` fixes a variable at the single entity
nearest to a given point.  It sets, e.g., the pressure level of an enclosed
flow:

.. code-block:: python

   problem.add_boundary_condition("point_Dirichlet_boundary_condition", "pressure_level", variable="pressure", point=[0.5, 0.0], value=0.0)
""",
)

# ---------------------------------------------------------------------------
# Contact between bodies, eigenstrains and creep
# ---------------------------------------------------------------------------
_add(
    "gap_heat_transfer",
    """
``gap_heat_transfer`` transfers heat across a thin gap between two bodies that
are meshed separately, such as a shaft and the sleeve around it.  Each
integration point of ``boundary`` (the primary surface) is paired with the
closest point of ``secondary_boundary``.  The heat flux
:math:`q = h (T_s - T_p)` enters the primary body and exactly the same heat
leaves the secondary body, so the condition conserves energy on matching and
non-matching meshes alike.  The conductance :math:`h` is a number or a
function, and radiation is added when both emissivities are positive:

.. code-block:: python

   problem.add_boundary_condition("gap_heat_transfer", "gap", variable="T", boundary=["shaft_outer"], secondary_boundary=["sleeve_inner"], gap_conductance=5000.0)
""",
)

_add(
    "gas_gap_heat_transfer",
    """
``gas_gap_heat_transfer`` is ``gap_heat_transfer`` with the conductance of a
gas-filled gap computed at every point: conduction through the gas mixture
(the model of Ross and Stoute, with the mixture conductivity of Brokaw and the
temperature jump distances of Kennard), radiation between the two surfaces,
and solid conduction once they are in contact.  With ``displacements`` the
gap width follows the deformation.  With ``contact_penalty`` (the same value
as in ``gap_contact``) the contact pressure adds the solid contact
conductance, which needs the Meyer hardness and the conductivities of the two
bodies.  The gas pressure and the mole fractions are numbers or functions:

.. code-block:: python

   problem.add_boundary_condition("gas_gap_heat_transfer", "gap", variable="T", boundary=["shaft_outer"], secondary_boundary=["sleeve_inner"], helium_fraction=1.0, gas_pressure=2.0e6, primary_roughness=1.0e-6, secondary_roughness=1.0e-6, primary_emissivity=0.3, secondary_emissivity=0.3)
""",
)

_add(
    "gap_contact",
    """
``gap_contact`` prevents the interpenetration of two bodies with a penalty
force normal to the contact surface that acts only when the gap is closed
(frictionless contact).  One object is added per displacement component.
``penalty`` is the stiffness per unit area, which should be large compared
with the stiffness of the bodies divided by their size:

.. code-block:: python

   for i, d in enumerate(["disp_x", "disp_y"]):
       problem.add_boundary_condition("gap_contact", f"contact_{d}", variable=d, component=i, displacements=["disp_x", "disp_y"], boundary=["shaft_outer"], secondary_boundary=["sleeve_inner"], penalty=1e15)
""",
)

_add(
    "thermal_expansion_eigenstrain",
    """
``thermal_expansion_eigenstrain`` is the thermal strain of a material with a
constant expansion coefficient, :math:`\\varepsilon^* = \\alpha (T - T_0)
\\mathbf{I}`, stored as a property under ``eigenstrain_name`` so
that ``small_strain_stress`` can subtract it from the total strain:

.. code-block:: python

   problem.add_property("thermal_expansion_eigenstrain", "expansion", temperature="T", thermal_expansion_coefficient=1.0e-5, stress_free_temperature=293.15, eigenstrain_name="thermal_strain")
""",
)

_add(
    "small_strain_stress",
    """
``small_strain_stress`` computes the stress of an isotropic elastic solid
from the total strain minus the listed eigenstrains and, optionally, the
creep strain, which it integrates itself by a backward Euler radial return
and stores between time steps.  ``formulation`` selects axisymmetric
:math:`(r, z)`, plane strain, three-dimensional, or ``axisymmetric_1d``, the
radial slice of a long body with a uniform ``axial_strain`` (generalized plane
strain).  Young's modulus and Poisson's ratio are numbers or the properties of
another property object.  A creep law is an expression of
``von_mises_stress``, ``temperature``, ``grain_size`` and named constants:

.. code-block:: python

   creep_law = "A * von_mises_stress^n * exp(-Q / temperature)"
   problem.add_property("small_strain_stress", "stress", displacements=["disp_x", "disp_y"], formulation="plane_strain", youngs_modulus=2.0e11, poissons_ratio=0.3, eigenstrain_names=["thermal_strain"], creep_model="parsed", creep_rate=creep_law, creep_constant_names=["A", "n", "Q"], creep_constant_values=[1e-30, 3.0, 2.0e4], temperature="T")
""",
)

# ---------------------------------------------------------------------------
# Physics and couplings
# ---------------------------------------------------------------------------
_add(
    "symmetry_boundary_condition",
    """
``symmetry_boundary_condition`` models a plane of symmetry of the body and of
its loads, so that only one side of the plane is meshed.  The displacement
normal to the plane is zero and the tangential displacements are free.  The
plane is found from the coordinates of the nodes of the side set, which must
all share one coordinate:

.. code-block:: python

   solid = problem.add_physics("solid_mechanics", "solid", displacements=["u", "v"], formulation="plane_stress", youngs_modulus=200.0e9, poissons_ratio=0.3)
   solid.add_boundary_condition("symmetry_boundary_condition", "left")
   solid.add_boundary_condition("symmetry_boundary_condition", "bottom")
""",
)

_add(
    "heat_transfer",
    """
The bus bar of Reddy's Example 5.4.3, with fixed temperatures on two sides
and convection on the top:

.. code-block:: python

   heat = problem.add_physics("heat_transfer", "heat", thermal_conductivity=20.0, heat_source=1.0e6)
   heat.add_boundary_condition("Dirichlet_boundary_condition", "left", value=40.0)
   heat.add_boundary_condition("Dirichlet_boundary_condition", "right", value=10.0)
   heat.add_boundary_condition("convective_heat_flux_boundary_condition", "top", heat_transfer_coefficient=75.0)

A transient problem gives ``density`` and ``specific_heat``, and a
conductivity that differs between blocks comes from the property objects when
``thermal_conductivity`` is not given.
""",
)

_add(
    "coefficient_form_PDE",
    """
The coefficient form writes one or several equations by their coefficients.
The equation of each field :math:`u` is

.. math::

   d \\frac{\\partial u}{\\partial t} + \\nabla \\cdot \\left( -c \\nabla u - \\boldsymbol{\\alpha} u \\right) + \\boldsymbol{\\beta} \\cdot \\nabla u + a u = f ,

where :math:`c` is the diffusion coefficient, :math:`\\boldsymbol{\\alpha}` the
conservative flux convection coefficient, :math:`\\boldsymbol{\\beta}` the
convection coefficient, :math:`a` the absorption coefficient, :math:`f` the
source and :math:`d` the time derivative coefficient.  A coefficient is a
constant or an expression of the fields, their gradients, :math:`x`, :math:`y`,
:math:`z` and :math:`t`.  An expression that depends on a field makes the
equations nonlinear, and Newton's method receives its exact derivative.  A
steady advection-diffusion-reaction problem of one field reads:

.. code-block:: python

   pde = problem.add_physics("coefficient_form_PDE", "transport", variables=["c"], diffusion_coefficient="1 + 0.5*c", convection_coefficient=[1.0, 0.5], absorption_coefficient=2.0, source="sin(pi*y)")
   pde.add_boundary_condition("Dirichlet_boundary_condition", "left", value=1.0)
   pde.add_boundary_condition("Neumann_boundary_condition", "right", flux=0.5)

With several fields, every input is a dict keyed by the field of the
equation.  A coefficient of an equation is one value, which acts on the field
of that equation, or a dict field name -> value, which couples the equation to
other fields.  In this example, the equation of :math:`u` has the absorption
term :math:`2u - v`:

.. code-block:: python

   diffusion = {"u": 1.0, "v": "1 + u^2"}
   absorption = {"u": {"u": 2.0, "v": -1.0}}
   pde = problem.add_physics("coefficient_form_PDE", "system", variables=["u", "v"], diffusion_coefficient=diffusion, absorption_coefficient=absorption, source={"u": "sin(x)"})
""",
)

_add(
    "general_form_PDE",
    """
The general form writes a system of equations by the flux
:math:`\\boldsymbol{\\Gamma}` and the source :math:`f` of each field,

.. math::

   d \\frac{\\partial u}{\\partial t} + \\nabla \\cdot \\boldsymbol{\\Gamma} = f ,

where :math:`\\boldsymbol{\\Gamma}` and :math:`f` are expressions of all the fields,
the components of their gradients (``grad_x(u)``, ``grad_y(u)`` and
``grad_z(u)``), :math:`x`, :math:`y`, :math:`z`, :math:`t` and named constants.
The expressions are compiled and differentiated automatically, so the
Jacobian is exact and the assembly runs on every thread and every process.
A p-Laplacian coupled to a second field reads:

.. code-block:: python

   gradient = "(grad_x(u)^2 + grad_y(u)^2)"
   flux = {"u": [f"-(1 + {gradient})*grad_x(u)", f"-(1 + {gradient})*grad_y(u)"], "v": ["-k*grad_x(v)", "-k*grad_y(v)"]}
   source = {"u": "v - u^3", "v": "1"}
   pde = problem.add_physics("general_form_PDE", "system", variables=["u", "v"], flux=flux, source=source, constants={"k": 2.0})

The boundary flux :math:`-\\mathbf{n} \\cdot \\boldsymbol{\\Gamma}` is prescribed
by ``Neumann_boundary_condition``, whose ``flux`` may also be an expression of
the fields.
""",
)

_add(
    "parsed_kernel",
    """
``parsed_kernel`` adds one term to the equation of ``variable`` in the general
form :math:`\\nabla \\cdot \\boldsymbol{\\Gamma} = f`, with the flux and the source
given as expressions of the fields and their gradients.  ``general_form_PDE``
and ``coefficient_form_PDE`` generate it.  A nonlinear diffusion term:

.. code-block:: python

   problem.add_kernel("parsed_kernel", "nonlinear_diffusion", variable="u", flux=["-(1 + u^2)*grad_x(u)", "-(1 + u^2)*grad_y(u)"], source="sin(x)")
""",
)

_add(
    "solid_mechanics",
    """
A three-dimensional wrench held at its jaws and loaded at its grip by a force
of 150 N (``examples/wrench``):

.. code-block:: python

   solid = problem.add_physics("solid_mechanics", "solid", displacements=["u", "v", "w"], youngs_modulus=200.0e9, poissons_ratio=0.3)
   solid.add_boundary_condition("fixed_constraint", "jaws")
   solid.add_boundary_condition("traction_boundary_condition", "grip", total_force=[0.0, -150.0, 0.0])

A two-dimensional problem gives ``formulation="plane_stress"`` or
``formulation="plane_strain"``.
""",
)

_add(
    "incompressible_flow",
    """
The lid-driven cavity of Reddy's Example 9.8.4 at a Reynolds number of 1000,
with the velocity of the lid increased in load steps:

.. code-block:: python

   flow = problem.add_physics("incompressible_flow", "flow", velocities=["u", "v"], dynamic_viscosity=1.0, density=1000.0)
   flow.add_boundary_condition("Dirichlet_boundary_condition", "walls", boundary=["left", "right", "bottom"], value=[0.0, 0.0])
   flow.add_boundary_condition("Dirichlet_boundary_condition", "top", value=[1.0, 0.0], scale_with_load=True)
   problem.solve(load_factors=[0.1, 0.25, 0.5, 0.75, 1.0])
""",
)

_add(
    "beam",
    """
A pinned beam under a uniform load, of which half is modelled, in the mixed
Euler-Bernoulli model:

.. code-block:: python

   beam = problem.add_physics("beam", "beam", model="beam_Euler_Bernoulli_mixed", extensional_stiffness=3.0e7, bending_stiffness=2.5e6, transverse_load=0.5)
   beam.add_boundary_condition("Dirichlet_boundary_condition", "left", variables=["axial_displacement", "deflection", "bending_moment"], value=0.0)
   beam.add_boundary_condition("Dirichlet_boundary_condition", "right", variable="axial_displacement", value=0.0)
""",
)

_add(
    "plate",
    """
A rectangular plate of the first-order theory under a uniform load:

.. code-block:: python

   plate = problem.add_physics("plate", "plate", in_plane_displacements=["u", "v"], transverse_displacement="w", rotations=["phi_x", "phi_y"], extensional_stiffness=a, bending_stiffness=d, shear_stiffness=s, poissons_ratio=0.3, transverse_load=q0)
""",
)

_add(
    "circular_plate",
    """
A clamped circular plate in the classical theory, on a radial mesh in
axisymmetric coordinates:

.. code-block:: python

   problem = dm.Problem(dm.generate_line_mesh(start=0.0, end=radius, num_elements=32), coordinates="axisymmetric")
   plate = problem.add_physics("circular_plate", "plate", theory="classical", extensional_stiffness=a_rr, bending_stiffness=d_rr, poissons_ratio=0.3, transverse_load=q0)
   plate.add_boundary_condition("Dirichlet_boundary_condition", "left", variable="radial_displacement", value=0.0)
   plate.add_boundary_condition("Dirichlet_boundary_condition", "right", variables=["radial_displacement", "deflection"], value=0.0)
""",
)

_add(
    "thermal_expansion",
    """
The thermal stress of a body heated by conduction:

.. code-block:: python

   heat = problem.add_physics("heat_transfer", "heat", thermal_conductivity=50.0)
   solid = problem.add_physics("solid_mechanics", "solid", formulation="plane_strain", youngs_modulus=200.0e9, poissons_ratio=0.3)
   problem.add_coupling("thermal_expansion", "expansion", heat_transfer="heat", solid_mechanics="solid", thermal_expansion_coefficient=1.2e-5, stress_free_temperature=293.15)
""",
)

_add(
    "nonisothermal_flow",
    """
Natural convection in a cavity (``examples/natural_convection.py``), in the
non-dimensional form of de Vahl Davis (1983):

.. code-block:: python

   flow = problem.add_physics("incompressible_flow", "flow", velocities=["u", "v"], dynamic_viscosity=0.71, density=1.0, penalty_parameter=1.0e7)
   heat = problem.add_physics("heat_transfer", "heat", thermal_conductivity=1.0)
   problem.add_coupling("nonisothermal_flow", "coupling", heat_transfer="heat", incompressible_flow="flow", gravity=[0.0, -1.0], thermal_expansion_coefficient=1.0e5 * 0.71, scale_with_load=True)
""",
)
