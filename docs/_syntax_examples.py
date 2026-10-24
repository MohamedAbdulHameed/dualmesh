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

   problem.add_kernel("diffusion", variable="u", diffusivity=2.5)

   # A diffusivity that varies through the body.
   problem.add_function("k", lambda x, y, z, t: 1.0 + 4.0 * x)
   problem.add_kernel("diffusion", variable="u", diffusivity="k")

   # A diffusivity that depends on the solution: k_eff = 1.0 * (1 + 0.5 u).
   problem.add_kernel("diffusion", variable="u", diffusivity=1.0, solution_polynomial=[1.0, 0.5])

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
   problem.add_kernel("anisotropic_diffusion", variable="temperature", diffusivity_tensor=[10.0, 1.0])

   # The full tensor, row by row, for principal axes rotated by 30 degrees.
   import numpy as np

   c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
   k = np.array([[10.0, 0.0], [0.0, 1.0]])
   r = np.array([[c, -s], [s, c]])
   problem.add_kernel("anisotropic_diffusion", variable="temperature", diffusivity_tensor=list((r @ k @ r.T).ravel()))
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
   problem.add_kernel("diffusion", variable="u")
   problem.add_kernel("reaction", variable="u", coeff=400.0)

   # A beam on an elastic foundation of modulus 1e6.
   problem.add_kernel("reaction", variable="deflection", coeff=1.0e6)

   # A second-order chemical reaction, which is nonlinear.
   problem.add_kernel("reaction", variable="concentration", coeff=0.1, exponent=2.0)
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
   problem.add_kernel("body_force", variable="u", value="source")

The quadrature keyword has a larger effect on this kernel than on any other,
because the source is the only term that the method integrates approximately.
The trapezoidal rule over the whole control domain, the rule of the published
finite volume formulations [Reddy2024]_, is named ``control_domain_trapezoid``:

.. code-block:: python

   problem.add_kernel("body_force", variable="u", value="source", quadrature="control_domain_trapezoid")
""",
)

_add(
    "advection",
    """
``advection`` transports the unknown with a prescribed velocity, so together
with ``diffusion`` it forms the advection-diffusion equation.  The velocity may
be constant or given by functions, one per component:

.. code-block:: python

   problem.add_kernel("diffusion", variable="c", diffusivity=1.0)
   problem.add_kernel("advection", variable="c", velocity=[75.0, 0.0])

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
   problem.add_kernel("diffusion", variable="a")
   problem.add_kernel("diffusion", variable="b")
   problem.add_kernel("coupled_force", variable="a", coupled_variable="b", coeff=-1.0)

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
   problem.add_kernel("diffusion", variable="u")
   problem.add_kernel("time_derivative", variable="u", coeff=1.0)
   problem.solve_transient(end_time=0.1, time_step=0.001, implicitness=0.5)

The capacity is lumped, as is common in explicit and nearly explicit schemes,
by a change of the quadrature:

.. code-block:: python

   problem.add_kernel("time_derivative", variable="u", quadrature="nodal")
""",
)

_add(
    "Dirichlet",
    """
``Dirichlet`` prescribes the value of a variable, which is
the essential member of the duality pair.  The value is written directly into
the solution vector and the equation of that degree of freedom is replaced, so
the residual of the replaced equation becomes available afterwards as the
reaction:

.. code-block:: python

   problem.add_boundary_condition("Dirichlet", variable="temperature", boundary="left", value=300.0)

   # A value that varies along the boundary.
   problem.add_function("profile", lambda x, y, z, t: np.cos(np.pi * x / 6.0))
   problem.add_boundary_condition("Dirichlet", variable="temperature", value="profile")

   # One value on several variables: the condition is created once per variable.
   problem.add_boundary_condition("Dirichlet", variable=["u", "v"], value=0.0)

   # The heat that must flow in through the left edge to hold it at 300.
   print(problem.total_reaction("temperature", "left"))

A boundary without a boundary condition receives the natural condition with a
zero secondary variable, i.e., an insulated surface in heat transfer and a
traction-free surface in elasticity.
""",
)

_add(
    "Neumann",
    """
``Neumann`` prescribes the secondary variable of the duality
pair, the normal flux :math:`n \\cdot F`.  Because the canonical flux of a
diffusion problem is :math:`F = k \\nabla u`, a positive value is a flux of the
unknown *into* the body:

.. code-block:: python

   # One watt per square metre entering through the right edge.
   problem.add_boundary_condition("Neumann", variable="temperature", boundary="right", flux=1.0)

A boundary without a boundary condition has a zero flux, so an insulated or
symmetry boundary requires no object.  Add this object only when the flux is
non-zero or when a named object makes the input clearer.
""",
)

_add(
    "Robin",
    """
``Robin`` is the mixed condition
:math:`n \\cdot F = q_0 - h (u - u_\\infty)`, which describes convection into a
surrounding medium, a contact resistance, and a spring support.  It is the
general form of which ``Neumann`` is the special case
:math:`h = 0`:

.. code-block:: python

   # Convection from a fin tip into air at 20 degrees.
   problem.add_boundary_condition("Robin", variable="temperature", boundary="right", transfer_coeff=2.0, ambient_value=20.0)

The physics write the same condition as text, in their own terms:
``heat.add_boundary_condition("heat_flux = 2*(temperature - 20)", boundary="right")``.
""",
)

_add(
    "periodic",
    """
``periodic`` joins two opposite boundaries, so that every
node of the secondary boundary takes the value of the primary node it faces.
It models one cell of a structure that repeats, such as a channel of a heat
exchanger or a unit cell of a composite.  The translation, from a primary
node to its secondary node, is the vector between the centroids of the two
boundaries unless it is given, and the meshes of the two boundaries must match
node by node after it:

.. code-block:: python

   # Periodic in x on the unit square: the right edge repeats the left edge.
   problem.add_boundary_condition("periodic", primary="left", secondary="right")

   # Periodic in y as well; the corner nodes are then joined in both directions.
   problem.add_boundary_condition("periodic", primary="bottom", secondary="top")

   # Only the velocity is periodic, in a channel two metres long.
   problem.add_boundary_condition("periodic", primary="inlet", secondary="outlet", variable=["u", "v"], translation=[2.0, 0.0])

A value prescribed on the secondary boundary must also be prescribed on the
primary one.  The condition works with ``fem``, ``hfvm`` and ``dmcdm``, in
serial and in parallel.
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
   problem.add_point_source(variable="deflection", value=-1000.0, points=[0.5, 0.0, 0.0])

   # A heat source at every node of a named node set.
   problem.add_point_source(variable="temperature", value=50.0, boundary="heater_nodes")

In a plane-stress or plane-strain model a traction is multiplied by the
thickness, while the value of a point source is applied as given, so give the
total force through the thickness.
""",
)

_add(
    "constant",
    """
``constant`` defines named constant properties that kernels
read.  It allows several kernels to share one value, and a property to take
different values in different blocks:

.. code-block:: python

   problem.add_property("constant", properties={"conductivity": 400.0}, block=["copper_bar"])
   problem.add_property("constant", properties={"conductivity": 45.0}, block=["steel_bar"])
   problem.add_kernel("heat_conduction", variable="temperature", thermal_conductivity="conductivity")
""",
)

_add(
    "function",
    """
``function`` is the counterpart of
``constant`` for a property that varies through the body.
The functions must be registered on the problem first:

.. code-block:: python

   problem.add_function("k_of_x", lambda x, y, z, t: 50.0 * (1.0 + x))
   problem.add_property("function", properties={"conductivity": "k_of_x"})

A property that depends on the *solution* is given as a ``parsed``,
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
   problem.add_kernel("heat_conduction", variable="temperature", thermal_conductivity=20.0)

A temperature-dependent conductivity is written as a polynomial multiplying the
base value:

.. code-block:: python

   # k(T) = 45 (1 + 0.002 T)
   problem.add_kernel("heat_conduction", variable="temperature", thermal_conductivity=45.0, temperature_polynomial=[1.0, 0.002])
   problem.solve(nonlinear_solver="newton")
""",
)

_add(
    "heat_conduction_time_derivative",
    """
``heat_conduction_time_derivative`` is the storage term of the energy equation
and makes a thermal problem transient:

.. code-block:: python

   problem.add_kernel("heat_conduction", variable="temperature", thermal_conductivity=45.0)
   problem.add_kernel("heat_conduction_time_derivative", variable="temperature", density=7850.0, specific_heat=460.0)
   problem.solve_transient(end_time=600.0, time_step=5.0, implicitness=0.5)

Only the product of density and specific heat enters, so a volumetric heat
capacity may be given as the density with the specific heat set to one.
""",
)

_add(
    "radiative_heat_flux",
    """
``radiative_heat_flux`` adds grey-body radiation to a
surface.  It is strongly nonlinear, because the flux varies with the fourth
power of the temperature, and it requires temperatures on an absolute scale:

.. code-block:: python

   # A surface at a few hundred kelvin radiating to surroundings at 300 K.
   problem.add_boundary_condition("radiative_heat_flux", variable="temperature", boundary="top", emissivity=0.8, ambient_temperature=300.0)
   problem.solve(nonlinear_solver="newton")

Radiation and convection usually act together.  When both objects are added
on the same side set, their fluxes are summed.
""",
)

# ---------------------------------------------------------------------------
# Solid mechanics
# ---------------------------------------------------------------------------
_add(
    "stress_divergence",
    """
``stress_divergence`` is the equilibrium equation of the displacements.  One
call adds the equation of every displacement listed in ``displacements``, and
``variable`` restricts it to some of them:

.. code-block:: python

   problem.add_kernel("stress_divergence", displacements=["displacement_x", "displacement_y"], thickness=0.01)

The thickness multiplies the whole equation in a plane problem, and the
traction, pressure and gap contact conditions of the same displacement read it
from here.  A three-dimensional or axisymmetric problem refuses it, because the
integration measure of the axisymmetric case already carries the factor
:math:`2 \\pi r`.
""",
)

_add(
    "traction",
    """
``traction`` applies a surface traction component, that of the equation
named by ``variable``.  The load is given either as the traction itself, in
force per unit area, or as the total force on the boundary, which the code
spreads uniformly over the area of the boundary.  ``traction`` and
``total_force`` define the same load, so a condition gives one of them and
never both:

.. code-block:: python

   # A uniform shear of 1 MPa on the right edge, acting along +y.
   problem.add_boundary_condition("traction", variable="displacement_y", boundary="right", traction=1.0e6)

   # A force of 150 N along -y on the side set 'grip', whatever its area.
   problem.add_boundary_condition("traction", variable="v", boundary="grip", total_force=-150.0)

In a plane problem the thickness is that of the ``stress_divergence`` of the
same displacement.  The ``solid_mechanics`` physics writes the load as text:
``solid.add_boundary_condition("traction = [0, 1e6]", boundary="right")``.
""",
)

_add(
    "pressure",
    """
``pressure`` applies a pressure normal to a surface, which
is the usual load on a vessel, a dam, or the surface of a hole in a plate.  One
call loads every displacement listed in ``displacements``:

.. code-block:: python

   problem.add_boundary_condition("pressure", boundary="inner", displacements=["displacement_x", "displacement_y"], pressure=10.0e6)

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
second-order equations.  One call adds the equations of the three variables:

.. code-block:: python

   for name in ("axial", "deflection", "moment"):
       problem.add_variable(name)
   problem.add_kernel("beam_Euler_Bernoulli_mixed", axial_displacement="axial", transverse_displacement="deflection", bending_moment="moment", extensional_stiffness=a_xx, bending_stiffness=d_xx, transverse_load=-1.0e3)

The boundary conditions follow the duality pairs: a clamped end prescribes the
deflection and leaves the moment free, because the vanishing slope is the
natural condition of the moment equation.  A simply supported end prescribes
the deflection *and* prescribes the moment to be zero.

``solid_mechanics`` with ``formulation="beam"`` adds them.
""",
)

_add(
    "beam_Timoshenko_displacement",
    """
The shear-deformable beam in displacement form carries the axial displacement,
the deflection, and the rotation of the cross-section.  A thin beam exhibits
shear locking unless the shear terms are integrated with one point, so the
kernel of the rotation equation is added with the one-point rule, as the
``solid_mechanics`` physics does:

.. code-block:: python

   beam = dict(axial_displacement="axial", transverse_displacement="deflection", rotation="rotation", extensional_stiffness=a_xx, bending_stiffness=d_xx, shear_stiffness=s_xz, transverse_load=-1.0e3)
   for variable in ("axial", "deflection"):
       problem.add_kernel("beam_Timoshenko_displacement", variable=variable, **beam)
   problem.add_kernel("beam_Timoshenko_displacement", variable="rotation", quadrature="midpoint", reduced_integration=True, **beam)
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

   problem.add_kernel("beam_Timoshenko_mixed", axial_displacement="axial", transverse_displacement="deflection", bending_moment="moment", extensional_stiffness=a_xx, bending_stiffness=d_xx, shear_stiffness=s_xz, transverse_load=-1.0e3)
""",
)

_add(
    "circular_plate_first_order",
    """
This object is the shear-deformable model of an axisymmetric circular plate,
solved on a one-dimensional radial mesh with the axisymmetric coordinate
system.  ``solid_mechanics`` with ``formulation="circular_plate"`` adds it twice, once without the
transverse shear (``shear_treatment="exclude"``) and once with the shear
alone on a reduced rule (``shear_treatment="only"``), which removes shear
locking in thin plates:

.. code-block:: python

   mesh = dm.generate_line_mesh(start=0.0, end=radius, num_elements=32)
   problem = dm.Problem(mesh, coordinates="axisymmetric")
   plate = problem.add_physics("solid_mechanics", formulation="circular_plate", theory="first_order", extensional_stiffness=a_rr, coupling_stiffness=b_rr, bending_stiffness=d_rr, shear_stiffness=s_rz, poissons_ratio=0.3, transverse_load=q0)

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
   plate = problem.add_physics("solid_mechanics", formulation="circular_plate", theory="classical", extensional_stiffness=a_rr, coupling_stiffness=b_rr, bending_stiffness=d_rr, poissons_ratio=0.3, transverse_load=q0)
   plate.add_boundary_condition("disp_r = 0", boundary="left")
   plate.add_boundary_condition(["disp_r = 0", "disp_z = 0"], boundary="right")
   # The physics holds the moment of the simply supported edge at zero.  A clamped
   # edge ("fixed") keeps its zero slope, the natural condition of the moment.

The moment is a nodal unknown, so it is available directly, without
differentiation of the deflection.  It is accurate everywhere except at the
centre, which is a known property of mixed models.
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

   plate = problem.add_physics("solid_mechanics", formulation="plate", extensional_stiffness=a, coupling_stiffness=b, bending_stiffness=d, shear_stiffness=s, poissons_ratio=0.3, transverse_load=q0, von_karman=True)
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
   problem.add_kernel("viscous_stress", variable=velocities, velocities=velocities, dynamic_viscosity=1.0)

The ``fluid_dynamics`` physics adds the complete set of kernels.
""",
)

_add(
    "penalty_incompressibility",
    """
The penalty formulation replaces the pressure by :math:`P = -\\gamma \\nabla
\\cdot v`, which eliminates the pressure unknown and leaves a problem in the
velocities alone.  The penalty term must then be under-integrated:

.. code-block:: python

   problem.add_kernel("penalty_incompressibility", variable=velocities, velocities=velocities, penalty_parameter=1.0e8)

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

   problem.add_kernel("convective_inertia", variable=velocities, velocities=velocities, density=1.0)
   problem.solve(nonlinear_solver="newton", load_factors=[0.1, 0.25, 0.5, 0.75, 1.0])

At a high Reynolds number, a solve that starts from rest at the full velocity
usually diverges.  Either increase the boundary velocity in increments with
load stepping, as above, or use direct iteration with a relaxation factor of
about one half.
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

   problem.add_kernel("Boussinesq_buoyancy", variable="v", velocities=["u", "v"], temperature="temperature", gravity=[0.0, -1.0], vol_expansion_coeff=rayleigh * prandtl, scale_with_load=True)
   problem.solve(load_factors=[0.01, 0.1, 1.0])

With ``scale_with_load=True`` load stepping increases the Rayleigh number in
increments, which allows a strongly buoyant flow to be computed from a fluid
at rest.  The ``nonisothermal_flow`` coupling of a ``heat_transfer`` and an
``fluid_dynamics`` physics adds this kernel when it is given ``gravity``
and ``vol_expansion_coeff``.  The complete problem is in
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

   problem.add_kernel("heat_convection", variable="temperature", velocities=["u", "v"], density=1.0, specific_heat=1.0)

The ``nonisothermal_flow`` coupling of a ``heat_transfer`` and an
``fluid_dynamics`` physics adds this kernel:

.. code-block:: python

   flow = problem.add_physics("fluid_dynamics", dynamic_viscosity=0.71, density=1.0)
   heat = problem.add_physics("heat_transfer", thermal_conductivity=1.0)
   problem.add_coupling("nonisothermal_flow", heat_transfer=heat, fluid_dynamics=flow)

With ``advection_scheme="central"``, the default, the term is written in the
advective (non-conservative) form, which is exact for an incompressible flow.
``"upwind"``, ``"linear_upwind"`` and ``"limited"`` make it a face flux with the
face value of that scheme, for the control volume methods.
""",
)

_add(
    "heat_streamline_stabilization",
    """
``heat_streamline_stabilization`` adds the streamline upwind Petrov-Galerkin
term of the energy equation, for the finite element method.  Its residual is
the sum of the sources of the other kernels of the temperature (the convection,
the heat sources and the heat capacity term), so a source added as a kernel is
in it too:

.. code-block:: python

   problem.add_kernel("heat_streamline_stabilization", variable="temperature", velocities=["u", "v"],
                      thermal_conductivity="1e-4")

The ``nonisothermal_flow`` coupling adds it, with the values of the
``heat_transfer`` physics, when that physics has ``advection_scheme="SUPG"``:

.. code-block:: python

   heat = problem.add_physics("heat_transfer", thermal_conductivity=1e-4, advection_scheme="SUPG")
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

   problem.add_property("penalty_pressure", velocities=velocities)
   problem.solve()
   pressure = problem.property_at_centroids("pressure")

It reads the penalty parameter of the ``penalty_incompressibility`` of the
same velocities, so the two always agree.
""",
)

_add(
    "pressure_gradient",
    """
``pressure_gradient`` is the pressure part of the stress in one momentum
equation of the pressure-velocity formulation.  It is normally added by the
``fluid_dynamics`` physics, together with the viscous stress and the
mass equation:

.. code-block:: python

   flow = problem.add_physics("fluid_dynamics", dynamic_viscosity=1.0, density=100.0, formulation="pressure", pressure_pin_point=(0.5, 0.0))

To add it manually, give the momentum equations it belongs to:

.. code-block:: python

   problem.add_kernel("pressure_gradient", variable=["u", "v"], velocities=["u", "v"], pressure="pressure")
""",
)

_add(
    "mass_conservation",
    """
``mass_conservation`` is the mass equation, attached to the pressure variable.
With equal-order interpolation it includes the stabilisation, whose residual
of the momentum equations it reads from the momentum kernels of the same
velocities (the density, the viscosity and the body forces):

.. code-block:: python

   problem.add_kernel("mass_conservation", variable="pressure", velocities=["u", "v"])
   problem.add_boundary_condition("mass_flux", variable="pressure", boundary=["left", "right", "bottom", "top"], velocities=["u", "v"])

In an enclosed flow the pressure level must also be fixed, with a
``point_Dirichlet``.
""",
)

_add(
    "momentum_stabilization",
    """
``momentum_stabilization`` adds the streamline-upwind term to one momentum
equation of a flow with inertia.  Like ``mass_conservation``, it reads the
momentum equations from their kernels:

.. code-block:: python

   problem.add_kernel("momentum_stabilization", variable=["u", "v"], pressure="pressure", velocities=["u", "v"])
""",
)

_add(
    "mass_flux",
    """
``mass_flux`` adds to the mass equation the flow through
the boundary, computed from the discrete velocity.  Apply it on every boundary
without a prescribed pressure.  The ``fluid_dynamics`` physics applies it
on the side sets of its ``mass_flux_boundaries``, by default every side set:

.. code-block:: python

   problem.add_boundary_condition("mass_flux", variable="pressure", boundary=["inlet", "outlet", "walls"], velocities=["u", "v"])
""",
)

_add(
    "point_Dirichlet",
    """
``point_Dirichlet`` fixes a variable at the single entity
nearest to a given point.  It sets, e.g., the pressure level of an enclosed
flow:

.. code-block:: python

   problem.add_boundary_condition("point_Dirichlet", variable="pressure", point=[0.5, 0.0], value=0.0)
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

   problem.add_boundary_condition("gap_heat_transfer", variable="T", boundary=["shaft_outer"], secondary_boundary=["sleeve_inner"], gap_conductance=5000.0)
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

   problem.add_boundary_condition("gas_gap_heat_transfer", variable="T", boundary=["shaft_outer"], secondary_boundary=["sleeve_inner"], helium_fraction=1.0, gas_pressure=2.0e6, primary_roughness=1.0e-6, secondary_roughness=1.0e-6, primary_emissivity=0.3, secondary_emissivity=0.3)
""",
)

_add(
    "gap_contact",
    """
``gap_contact`` prevents the interpenetration of two bodies with a penalty
force normal to the contact surface that acts only when the gap is closed
(frictionless contact).  One call adds it to every displacement listed in
``displacements``.  ``penalty`` is the stiffness per unit area, which should be large compared
with the stiffness of the bodies divided by their size:

.. code-block:: python

   problem.add_boundary_condition("gap_contact", displacements=["disp_x", "disp_y"], boundary=["shaft_outer"], secondary_boundary=["sleeve_inner"], penalty=1e15)
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

   problem.add_property("thermal_expansion_eigenstrain", temperature="T", linear_expansion_coeff=1.0e-5, stress_free_temperature=293.15, eigenstrain_name="thermal_strain")
""",
)

_add(
    "small_strain_stress",
    """
``small_strain_stress`` computes the stress of an isotropic elastic solid
from the total strain minus the listed eigenstrains and, optionally, the
creep strain, which it integrates itself by a backward Euler radial return
and stores between time steps.  It provides the property that
``stress_divergence`` reads, and one instance supplies the stress to all the
displacement equations:

.. code-block:: python

   for name in ("displacement_x", "displacement_y"):
       problem.add_variable(name)
   problem.add_property("small_strain_stress", displacements=["displacement_x", "displacement_y"], formulation="plane_stress", youngs_modulus=200.0e9, poissons_ratio=0.3)
   problem.add_kernel("stress_divergence", displacements=["displacement_x", "displacement_y"])

``formulation`` selects plane stress, plane strain, axisymmetric :math:`(r, z)`,
three-dimensional, or ``axisymmetric_1d``, the radial slice of a long body with
a uniform ``axial_strain`` (generalized plane strain).  Young's modulus and
Poisson's ratio are numbers, expressions or the properties of another property
object.  A thermal strain is an eigenstrain of ``thermal_expansion_eigenstrain``.
A creep law is an expression of ``von_mises_stress``, ``temperature``,
``grain_size`` and named constants:

.. code-block:: python

   creep_law = "A * von_mises_stress^n * exp(-Q / temperature)"
   problem.add_property("small_strain_stress", displacements=["disp_x", "disp_y"], formulation="plane_strain", youngs_modulus=2.0e11, poissons_ratio=0.3, eigenstrain_names=["thermal_strain"], creep_model="parsed", creep_rate=creep_law, creep_constants={"A": 1e-30, "n": 3.0, "Q": 2.0e4}, temperature="T")
""",
)

_add(
    "parsed",
    """
``parsed`` declares a scalar property given by an expression of the fields,
the functions, the element fields and the properties added before it.  It is
how a correlation, e.g., a conductivity :math:`k(T)`, is added without
writing C++, and its derivative with respect to the fields reaches Newton's
method exactly:

.. code-block:: python

   problem.add_property("parsed", property_name="thermal_conductivity", expression="k0*(1 + b*(temperature - 300))", constants={"k0": 20.0, "b": 1.0e-3})
   heat = problem.add_physics("heat_transfer")  # reads the property thermal_conductivity
""",
)

_add(
    "parsed_eigenstrain",
    """
``parsed_eigenstrain`` is an eigenstrain (a stress-free strain such as
thermal expansion or swelling) given by an expression, for the
``eigenstrain_names`` of ``small_strain_stress``.  A volumetric swelling that
grows with the temperature, zero at 293.15 K:

.. code-block:: python

   problem.add_property("parsed_eigenstrain", eigenstrain_name="swelling", expression="2.0e-5*(temperature - 293.15)", strain_type="volumetric", formulation="plane_strain")
   problem.add_property("small_strain_stress", displacements=["disp_x", "disp_y"], formulation="plane_strain", youngs_modulus=2.0e11, poissons_ratio=0.3, eigenstrain_names=["swelling"])

With ``stress_free_state={"temperature": 293.15}`` the expression is shifted
so that it vanishes in that state, which avoids repeating the reference value.
""",
)

_add(
    "finite_strain_stress",
    """
``finite_strain_stress`` is the stress at large strain, for equations written
on the undeformed mesh.  It provides the first Piola-Kirchhoff stress that
``stress_divergence`` reads when the deformation is large, with the material
laws of ``small_strain_stress`` (including creep) carried to finite rotation,
or a neo-Hookean solid:

.. code-block:: python

   problem.add_property("finite_strain_stress", displacements=["disp_x", "disp_y"], formulation="plane_strain", youngs_modulus=1.0e6, poissons_ratio=0.45, stress_update="neo_Hookean")
""",
)

_add(
    "property_scaling",
    """
``property_scaling`` multiplies properties computed by the property objects
added before it by constant factors, so that a sensitivity or uncertainty
study scales a correlation without changing the object that computes it:

.. code-block:: python

   problem.add_property("property_scaling", property_factors={"thermal_conductivity": 1.1}, block=["fuel"])
""",
)

# ---------------------------------------------------------------------------
# Physics and couplings
# ---------------------------------------------------------------------------
_add(
    "symmetry",
    """
``symmetry`` models a plane of symmetry of the body and of
its loads, so that only one side of the plane is meshed.  The displacement
normal to the plane is zero and the tangential displacements are free.  The
plane is found from the coordinates of the nodes of the side set, which must
all share one coordinate:

.. code-block:: python

   solid = problem.add_physics("solid_mechanics", formulation="plane_stress", youngs_modulus=200.0e9, poissons_ratio=0.3)
   solid.add_boundary_condition("symmetry", boundary="left")
   solid.add_boundary_condition("symmetry", boundary="bottom")
""",
)

_add(
    "heat_transfer",
    """
A bus bar with internal heat generation, fixed temperatures on two sides and
convection on the top.  A boundary condition is written as text: the
temperature on the left prescribes its value, and ``heat_flux``, the heat
flux :math:`\\mathbf{n} \\cdot \\mathbf{q}` leaving the body through the
boundary, prescribes the flux:

.. code-block:: python

   heat = problem.add_physics("heat_transfer", thermal_conductivity=20.0, heat_source=1.0e6)
   heat.add_boundary_condition("temperature = 40", boundary="left")
   heat.add_boundary_condition("temperature = 10", boundary="right")
   heat.add_boundary_condition("heat_flux = 75*temperature", boundary="top")

A heater is a negative ``heat_flux`` (heat enters), and radiation is
``"heat_flux = 5.67e-8*0.8*(temperature^4 - 293^4)"``.  A transient problem
gives ``density`` and ``specific_heat``, and a conductivity that differs
between blocks comes from the property objects when ``thermal_conductivity``
is not given.
""",
)

_add(
    "general_form_PDE",
    """
``general_form_PDE`` solves equations written as they are printed.  The
fields are named in ``variables`` and the equations in ``equations``, with
:math:`\\partial u / \\partial t` written ``du/dt``, :math:`\\nabla u`
``grad(u)``, :math:`\\nabla \\cdot` ``div`` and :math:`\\nabla^2`
``laplacian``.  A p-Laplacian coupled to a Poisson equation,

.. math::

   \\frac{\\partial u}{\\partial t} - \\nabla \\cdot \\left[ \\left( 1 + |\\nabla u|^2 \\right) \\nabla u \\right] = v - u^3 ,
   \\qquad
   -\\nabla \\cdot \\left( k \\nabla v \\right) = 1 ,

reads:

.. code-block:: python

   pde = problem.add_physics(
       "general_form_PDE",
       variables=["u", "v"],
       k=2.0,
       grad_u_squared="dot(grad(u), grad(u))",
       equations=[
           "du/dt - div((1 + grad_u_squared)*grad(u)) = v - u^3",
           "-div(k*grad(v)) = 1",
       ],
   )
   pde.add_boundary_condition("u = 0", boundary=["left", "right"])
   pde.add_boundary_condition("-k*dv/dn = 2*(v - 1)", boundary="top")

Every keyword that is not a parameter is a definition: ``k`` is a named
constant and ``grad_u_squared`` a named expression.  Each equation is matched
to the field it governs.  Everything inside ``div`` is the flux
:math:`\\boldsymbol{\\Gamma}`, which a boundary condition prescribes as
:math:`\\mathbf{n} \\cdot \\boldsymbol{\\Gamma}` along the outward normal,
written out (``-k*dv/dn``) or as ``normal_flux``.

An equation of a scalar field may instead be given by its coefficients, in

.. math::

   d \\frac{\\partial u}{\\partial t} + \\nabla \\cdot \\left( -c \\nabla u - \\boldsymbol{\\alpha} u + \\boldsymbol{\\gamma} \\right) + \\boldsymbol{\\beta} \\cdot \\nabla u + a u = f ,

with :math:`d` the damping (or mass) coefficient, :math:`c` the diffusion
coefficient, :math:`\\boldsymbol{\\alpha}` the conservative flux convection
coefficient, :math:`\\boldsymbol{\\gamma}` the conservative flux source,
:math:`\\boldsymbol{\\beta}` the convection coefficient, :math:`a` the
absorption coefficient and :math:`f` the source term:

.. code-block:: python

   pde = problem.add_physics("general_form_PDE", variables="c", c="1 + 0.5*c", beta=[1.0, 0.5], a=2.0, f="sin(pi*y)")

With several fields, each coefficient is a dict keyed by the field of the
equation, and a value that is itself a dict couples the equation to other
fields: ``a={"u": {"u": 2.0, "v": -1.0}}`` is the term :math:`2u - v` of the
equation of :math:`u`.
""",
)

_add(
    "parsed_kernel",
    """
``parsed_kernel`` adds one term to the equation of ``variable`` in the general
form :math:`\\nabla \\cdot \\boldsymbol{\\Gamma} = f`, with the flux and the source
given as expressions of the fields and their gradients.  ``general_form_PDE``
generates it from the equations.  A nonlinear diffusion term:

.. code-block:: python

   problem.add_kernel("parsed_kernel", variable="u", flux=["-(1 + u^2)*grad_x(u)", "-(1 + u^2)*grad_y(u)"], source="sin(x)")
""",
)

_add(
    "solid_mechanics",
    """
A three-dimensional wrench held at its jaws and loaded at its grip by a force
of 150 N (``examples/wrench``):

.. code-block:: python

   solid = problem.add_physics("solid_mechanics", youngs_modulus=200.0e9, poissons_ratio=0.3)
   solid.add_boundary_condition("fixed", boundary="jaws")
   solid.add_boundary_condition("total_force = [0, -150, 0]", boundary="grip")

Its variables are ``disp_x``, ``disp_y`` and ``disp_z`` (``disp_r`` and
``disp_z`` in axisymmetric coordinates).  A boundary condition is written as
text: a displacement on the left prescribes its value (``"disp_x = 0"``), ``traction`` the traction
:math:`\\boldsymbol{\\sigma} \\cdot \\mathbf{n}` per unit area (``"traction = [0,
-1e6, 0]"``), ``total_force`` the force spread over the boundary, and
``pressure`` a pressure :math:`p`, the traction :math:`-p \\mathbf{n}`.  The
words ``"fixed"`` and ``"symmetry"`` need no data.  A two-dimensional problem
gives ``formulation="plane_stress"`` or ``formulation="plane_strain"``.

The structural formulations reduce a thin body to its mid-surface or its
axis.  A steel plate 10 mm thick under a uniform pressure, simply supported,
and a Timoshenko beam:

.. code-block:: python

   plate = problem.add_physics("solid_mechanics", formulation="plate", youngs_modulus=2.0e11, poissons_ratio=0.3, thickness=0.01, transverse_load=1.0e3)
   plate.add_boundary_condition(["disp_x = 0", "disp_y = 0", "disp_z = 0"], boundary=["left", "right", "bottom", "top"])

   beam = problem.add_physics("solid_mechanics", formulation="beam", theory="Timoshenko", youngs_modulus=2.0e11, poissons_ratio=0.3, cross_section={"area": 1.0e-3, "second_moment": 2.0e-6}, transverse_load=-1.0e3)
   beam.add_boundary_condition("fixed", boundary="left")

A plate's variables are ``disp_x``, ``disp_y``, ``disp_z`` (the deflection),
``phi_x`` and ``phi_y``; a beam's ``disp_x``, ``disp_z`` and
``bending_moment`` (``phi`` with ``form="displacement"``); a circular plate's
``disp_r``, ``disp_z`` and ``phi`` or ``bending_moment``.  The section is
computed from the material at every point, by integrating its law through
the thickness, so a graded modulus is an expression of ``z``, the height above
the mid-surface (``youngs_modulus="2e11*(1 + 0.5*z/0.01)"``), and
``thermal_expansion`` gives the thermal force and moment of the section.  A
laminated section is given by its section stiffnesses instead.  The stress at
a height comes from the same law:

.. code-block:: python

   top = plate.stress(z="top")        # sigma_xx, sigma_yy, sigma_xy at the element centres
   bottom = beam.stress(z=-0.05)      # sigma_xx at 5 cm below the axis
""",
)

_add(
    "fluid_dynamics",
    """
The lid-driven cavity at a Reynolds number of 1000,
with the velocity of the lid increased in load steps:

.. code-block:: python

   flow = problem.add_physics("fluid_dynamics", dynamic_viscosity=1.0, density=1000.0)
   flow.add_boundary_condition(["velocity_x = 0", "velocity_y = 0"], boundary=["left", "right", "bottom"])
   flow.add_boundary_condition(["velocity_x = 1", "velocity_y = 0"], boundary="top", scale_with_load=True)
   problem.solve(load_factors=[0.1, 0.25, 0.5, 0.75, 1.0])
""",
)

_add(
    "thermal_expansion",
    """
The thermal stress of a body heated by conduction:

.. code-block:: python

   heat = problem.add_physics("heat_transfer", thermal_conductivity=50.0)
   solid = problem.add_physics("solid_mechanics", formulation="plane_strain", youngs_modulus=200.0e9, poissons_ratio=0.3)
   problem.add_coupling("thermal_expansion", heat_transfer=heat, solid_mechanics=solid, linear_expansion_coeff=1.2e-5, stress_free_temperature=293.15)
""",
)

_add(
    "nonisothermal_flow",
    """
Natural convection in a cavity (``examples/natural_convection.py``), in the
non-dimensional form of de Vahl Davis (1983):

.. code-block:: python

   flow = problem.add_physics("fluid_dynamics", dynamic_viscosity=0.71, density=1.0, penalty_parameter=1.0e7)
   heat = problem.add_physics("heat_transfer", thermal_conductivity=1.0)
   problem.add_coupling("nonisothermal_flow", heat_transfer=heat, fluid_dynamics=flow, gravity=[0.0, -1.0], vol_expansion_coeff=1.0e5 * 0.71, scale_with_load=True)
""",
)
