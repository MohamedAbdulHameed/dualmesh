# SPDX-License-Identifier: LGPL-2.1-or-later
"""Worked usage examples for the generated syntax reference.

Every registered object gets one entry.  The text is prose that explains when
the object is used and what its keywords do in that context; the code block
below it is a complete, runnable fragment.  Anything missing from this file
still gets a generated page, just without the usage section, and the
documentation build prints a warning naming the object, so the two cannot
drift apart silently.
"""

from __future__ import annotations

EXAMPLES: dict[str, str] = {}


def _add(name: str, text: str) -> None:
    EXAMPLES[name] = text.strip("\n")


# ---------------------------------------------------------------------------
# Framework
# ---------------------------------------------------------------------------
_add(
    "Diffusion",
    """
``Diffusion`` is the plain second-order operator and is the right kernel
whenever the physics is "flux proportional to gradient" and no physics module
offers a more specific name.  The diffusivity may be a number, a named function
of position and time, or a material property, and it may additionally be
multiplied by a polynomial in the unknown itself, which is the simplest way to
make a problem nonlinear:

.. code-block:: python

   problem.add_kernel("Diffusion", "conduction", variable="u", diffusivity=2.5)

   # A diffusivity that varies through the body.
   problem.add_function("k", lambda x, y, z, t: 1.0 + 4.0 * x)
   problem.add_kernel("Diffusion", "graded", variable="u", diffusivity="k")

   # A diffusivity that depends on the solution: k_eff = 1.0 * (1 + 0.5 u).
   problem.add_kernel("Diffusion", "nonlinear", variable="u",
                      diffusivity=1.0, solution_polynomial=[1.0, 0.5])

The last form needs a nonlinear solve; with ``nonlinear_solver="newton"`` the
derivative of the polynomial is carried exactly through the automatic
differentiation, so the iteration converges quadratically.
""",
)

_add(
    "AnisotropicDiffusion",
    """
Use ``AnisotropicDiffusion`` when the conductivity or permeability depends on
direction, for example in a laminate, a fibre-reinforced solid, or a layered
rock.  Give the tensor either as its diagonal or in full:

.. code-block:: python

   # Orthotropic: ten times more conductive along x than along y.
   problem.add_kernel("AnisotropicDiffusion", "conduction",
                      variable="temperature", diffusivity_tensor=[10.0, 1.0])

   # The full tensor, row by row, for principal axes rotated by 30 degrees.
   import numpy as np
   c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
   k = np.array([[10.0, 0.0], [0.0, 1.0]])
   r = np.array([[c, -s], [s, c]])
   problem.add_kernel("AnisotropicDiffusion", "rotated", variable="temperature",
                      diffusivity_tensor=list((r @ k @ r.T).ravel()))
""",
)

_add(
    "Reaction",
    """
``Reaction`` adds a term proportional to the unknown, which models radioactive
decay, chemical consumption, heat loss along a fin, and the restoring force of
an elastic foundation.  The sign matters: a positive coefficient removes the
unknown.

.. code-block:: python

   # A cooling fin: -k A u'' + P beta u = 0, with P beta / (k A) = 400.
   problem.add_kernel("Diffusion", "conduction", variable="u")
   problem.add_kernel("Reaction", "convection", variable="u", coefficient=400.0)

   # A beam on an elastic foundation of modulus 1e6.
   problem.add_kernel("Reaction", "foundation", variable="deflection",
                      coefficient=1.0e6)

   # A second-order chemical reaction, which is nonlinear.
   problem.add_kernel("Reaction", "consumption", variable="concentration",
                      coefficient=0.1, exponent=2.0)
""",
)

_add(
    "BodyForce",
    """
``BodyForce`` is the generic distributed source.  Any physics module has a
better-named equivalent, but this one accepts an arbitrary function and is what
the verification problems use when the source has no physical name:

.. code-block:: python

   problem.add_function("source", lambda x, y, z, t: 10.0 * np.cos(x))
   problem.add_kernel("BodyForce", "source", variable="u", value="source")

The quadrature keyword matters more here than anywhere else, because the source
is the only term whose integral the method does not compute exactly.  Reddy's
finite volume examples integrate it with the trapezoidal rule over the whole
control domain, which this library spells ``control_domain_trapezoid``:

.. code-block:: python

   problem.add_kernel("BodyForce", "source", variable="u", value="source",
                      quadrature="control_domain_trapezoid")
""",
)

_add(
    "Advection",
    """
``Advection`` transports the unknown with a prescribed velocity, so together
with ``Diffusion`` it forms the advection-diffusion equation.  The velocity may
be constant or given by functions, one per component:

.. code-block:: python

   problem.add_kernel("Diffusion", "diffusion", variable="c", diffusivity=1.0)
   problem.add_kernel("Advection", "transport", variable="c", velocity=[75.0, 0.0])

At a high cell Peclet number, here :math:`Pe = 75`, the central differencing
that this kernel performs will oscillate on a coarse mesh; refine the mesh
until the cell Peclet number is below about two, which is the honest cure, or
accept the oscillation and note it.

Choose ``form="conservative"`` when the velocity field is divergence free and
you want the discrete equations to conserve the transported quantity exactly
over each control domain.
""",
)

_add(
    "CoupledForce",
    """
``CoupledForce`` is the simplest way to couple two equations: it adds a source
to one equation proportional to another variable.  It is how a concentration
drives a temperature, or how a simple two-species reaction is written:

.. code-block:: python

   problem.add_variable("a")
   problem.add_variable("b")
   problem.add_kernel("Diffusion", "diff_a", variable="a")
   problem.add_kernel("Diffusion", "diff_b", variable="b")
   problem.add_kernel("CoupledForce", "b_feeds_a", variable="a",
                      coupled_variable="b", coefficient=-1.0)

The derivative with respect to the coupled variable is carried through the
automatic differentiation, so the off-diagonal block of the Jacobian is exact
and Newton's method converges quadratically even for a strongly coupled system.
""",
)

_add(
    "TimeDerivative",
    """
``TimeDerivative`` turns a steady problem into a transient one.  Add it to
every equation that should evolve; an equation without it stays a constraint,
which is what a mixed formulation needs for its moment equation.

.. code-block:: python

   problem.add_variable("u", initial_condition=lambda x, y, z, t: np.sin(np.pi * x))
   problem.add_kernel("Diffusion", "diffusion", variable="u")
   problem.add_kernel("TimeDerivative", "time", variable="u", coefficient=1.0)
   problem.solve_transient(end_time=0.1, dt=0.001, theta=0.5)

Lumping the capacity, which is common in explicit and nearly explicit schemes,
is a matter of changing the quadrature:

.. code-block:: python

   problem.add_kernel("TimeDerivative", "time", variable="u", quadrature="nodal")
""",
)

_add(
    "DirichletBC",
    """
``DirichletBC`` prescribes the value of a variable, which is the essential
member of the duality pair.  The value is written straight into the solution
vector and the equation of that degree of freedom is replaced, so the residual
of the replaced equation becomes available afterwards as the reaction:

.. code-block:: python

   problem.add_boundary_condition("DirichletBC", "hot", variable="temperature",
                                  boundary="left", value=300.0)

   # A value that varies along the boundary.
   problem.add_function("profile", lambda x, y, z, t: np.cos(np.pi * x / 6.0))
   problem.add_boundary_condition("DirichletBC", "top", variable="temperature",
                                  boundary="top", value="profile")

   # The heat that must flow in through the left edge to hold it at 300.
   print(problem.total_reaction("temperature", "left"))

A boundary that carries no condition at all is *not* unconstrained: it gets the
natural condition with a zero secondary variable, which means insulated for
heat transfer and traction free for elasticity.
""",
)

_add(
    "NeumannBC",
    """
``NeumannBC`` prescribes the secondary variable of the duality pair, the normal
flux :math:`n \\cdot F`.  Because the canonical flux of a diffusion problem is
:math:`F = k \\nabla u`, a positive value drives the unknown *into* the body:

.. code-block:: python

   # One watt per square metre entering through the right edge.
   problem.add_boundary_condition("NeumannBC", "inflow", variable="temperature",
                                  boundary="right", flux=1.0)

Leaving a boundary out entirely is the same as prescribing a zero flux there,
so an insulated or symmetry boundary needs no object at all.  Write one only
when the flux is non-zero or when naming it makes the input clearer.
""",
)

_add(
    "RobinBC",
    """
``RobinBC`` is the mixed condition :math:`n \\cdot F = q_0 - h (u - u_\\infty)`,
which covers convection into a surrounding medium, a contact resistance, and a
spring support.  It is the general form of which ``NeumannBC`` is the special
case :math:`h = 0`:

.. code-block:: python

   # Convection from a fin tip into air at 20 degrees.
   problem.add_boundary_condition("RobinBC", "tip", variable="temperature",
                                  boundary="right",
                                  transfer_coefficient=2.0, ambient_value=20.0)

For heat transfer specifically, ``ConvectiveHeatFluxBC`` says the same thing in
the vocabulary of the subject and should be preferred.
""",
)

_add(
    "PointSource",
    """
``PointSource`` applies a concentrated load or a point source of heat at one or
more nodes.  The point is snapped to the nearest node without a tolerance, so
place the load on a node the mesh actually has:

.. code-block:: python

   # A concentrated force at the centre of the span.
   problem.add_nodal_load("PointSource", "tip_load", variable="deflection",
                          value=-1000.0, points=[0.5, 0.0, 0.0])

   # A heat source at every node of a named node set.
   problem.add_nodal_load("PointSource", "heaters", variable="temperature",
                          value=50.0, boundary="heater_nodes")

In a plane-stress or plane-strain model the value is *not* multiplied by the
thickness the way a traction is, so give the total force through the thickness.
""",
)

_add(
    "GenericConstantMaterial",
    """
``GenericConstantMaterial`` publishes named constants that kernels can consume,
which is how one number is shared by several kernels and how a property is made
to differ between blocks:

.. code-block:: python

   problem.add_material("GenericConstantMaterial", "copper",
                        property_names=["conductivity"], property_values=[400.0],
                        block=["copper_bar"])
   problem.add_material("GenericConstantMaterial", "steel",
                        property_names=["conductivity"], property_values=[45.0],
                        block=["steel_bar"])
   problem.add_kernel("HeatConduction", "conduction", variable="temperature",
                      thermal_conductivity_property="conductivity")
""",
)

_add(
    "GenericFunctionMaterial",
    """
``GenericFunctionMaterial`` is the same idea for a property that varies through
the body.  The functions must be registered on the problem first:

.. code-block:: python

   problem.add_function("k_of_x", lambda x, y, z, t: 50.0 * (1.0 + x))
   problem.add_material("GenericFunctionMaterial", "graded",
                        property_names=["conductivity"], functions=["k_of_x"])

A property that depends on the *solution* rather than on position cannot be
written this way; subclass ``Material`` in Python instead, which is shown in
:doc:`/user_guide/problem_setup`.
""",
)

# ---------------------------------------------------------------------------
# Heat transfer
# ---------------------------------------------------------------------------
_add(
    "HeatConduction",
    """
``HeatConduction`` is Fourier conduction and is the kernel almost every thermal
analysis starts from:

.. code-block:: python

   problem.add_variable("temperature")
   problem.add_kernel("HeatConduction", "conduction", variable="temperature",
                      thermal_conductivity=20.0)

A temperature-dependent conductivity is written as a polynomial multiplying the
base value, which is the form Reddy's nonlinear examples use:

.. code-block:: python

   # k(T) = 45 (1 + 0.002 T)
   problem.add_kernel("HeatConduction", "conduction", variable="temperature",
                      thermal_conductivity=45.0,
                      temperature_polynomial=[1.0, 0.002])
   problem.solve(nonlinear_solver="newton")
""",
)

_add(
    "HeatSource",
    """
``HeatSource`` is internal heat generation: ohmic heating in a conductor,
nuclear heating in a fuel pin, curing in a resin.

.. code-block:: python

   problem.add_kernel("HeatSource", "joule", variable="temperature",
                      heat_source=1.0e6)

   # Generation that decays with depth.
   problem.add_function("q", lambda x, y, z, t: 2.0e6 * np.exp(-x / 0.01))
   problem.add_kernel("HeatSource", "absorption", variable="temperature",
                      heat_source="q")
""",
)

_add(
    "HeatConductionTimeDerivative",
    """
``HeatConductionTimeDerivative`` is the storage term of the energy equation and
is what makes a thermal problem transient:

.. code-block:: python

   problem.add_kernel("HeatConduction", "conduction", variable="temperature",
                      thermal_conductivity=45.0)
   problem.add_kernel("HeatConductionTimeDerivative", "storage",
                      variable="temperature", density=7850.0, specific_heat=460.0)
   problem.solve_transient(end_time=600.0, dt=5.0, theta=0.5)

Only the product of density and specific heat enters, so a volumetric heat
capacity may be given as the density with the specific heat left at one.
""",
)

_add(
    "HeatFluxBC",
    """
``HeatFluxBC`` prescribes the heat entering a surface, which is how a heater, a
measured flux, or an absorbed radiation load is applied:

.. code-block:: python

   problem.add_boundary_condition("HeatFluxBC", "heater", variable="temperature",
                                  boundary="bottom", heat_flux=5.0e4)

An insulated surface is the default and needs no object.
""",
)

_add(
    "ConvectiveHeatFluxBC",
    """
``ConvectiveHeatFluxBC`` is Newton's law of cooling and is the most common
thermal boundary condition in practice, because a surface exposed to a fluid is
neither at a known temperature nor carrying a known flux:

.. code-block:: python

   problem.add_boundary_condition("ConvectiveHeatFluxBC", "air",
                                  variable="temperature", boundary=["right", "top"],
                                  heat_transfer_coefficient=25.0,
                                  ambient_temperature=20.0)

The condition is linear in the temperature, so it costs no extra Newton
iterations.
""",
)

_add(
    "RadiativeHeatFluxBC",
    """
``RadiativeHeatFluxBC`` adds grey-body radiation to a surface.  It is strongly
nonlinear, because the flux goes as the fourth power of the temperature, and it
demands an absolute temperature scale:

.. code-block:: python

   # A surface at a few hundred kelvin radiating to surroundings at 300 K.
   problem.add_boundary_condition("RadiativeHeatFluxBC", "radiation",
                                  variable="temperature", boundary="top",
                                  emissivity=0.8, ambient_temperature=300.0)
   problem.solve(nonlinear_solver="newton")

Radiation and convection usually act together; add both objects on the same
side set and their fluxes are summed.
""",
)

# ---------------------------------------------------------------------------
# Solid mechanics
# ---------------------------------------------------------------------------
_add(
    "LinearElasticStress",
    """
``LinearElasticStress`` turns the displacement gradients into a stress and
publishes it as the material property that ``StressDivergence`` consumes.  One
instance serves all the displacement equations:

.. code-block:: python

   for name in ("displacement_x", "displacement_y"):
       problem.add_variable(name)
   problem.add_material("LinearElasticStress", "steel",
                        displacements=["displacement_x", "displacement_y"],
                        formulation="plane_stress",
                        youngs_modulus=200.0e9, poissons_ratio=0.3)
   for component, name in enumerate(("displacement_x", "displacement_y")):
       problem.add_kernel("StressDivergence", f"equilibrium_{name}",
                          variable=name, component=component)

Thermal strain is switched on by naming a temperature variable *and* giving a
non-zero expansion coefficient:

.. code-block:: python

   problem.add_material("LinearElasticStress", "steel",
                        displacements=["displacement_x", "displacement_y"],
                        youngs_modulus=200.0e9, poissons_ratio=0.3,
                        temperature="temperature",
                        thermal_expansion_coefficient=1.2e-5,
                        reference_temperature=20.0)

The helper :func:`dualmesh.physics.add_plane_elasticity` writes the material and
all the kernels in one call and is what most analyses should use.
""",
)

_add(
    "StressDivergence",
    """
``StressDivergence`` is the equilibrium equation of one displacement component.
Add one instance per displacement, with ``component`` matching the variable:

.. code-block:: python

   problem.add_kernel("StressDivergence", "equilibrium_x",
                      variable="displacement_x", component=0, thickness=0.01)
   problem.add_kernel("StressDivergence", "equilibrium_y",
                      variable="displacement_y", component=1, thickness=0.01)

The thickness multiplies the whole equation in a plane problem and must match
the thickness given to every traction and pressure boundary condition of the
same model.  In an axisymmetric problem leave it at one, because the
integration measure already carries the factor :math:`2 \\pi r`.
""",
)

_add(
    "TractionBC",
    """
``TractionBC`` applies a surface traction component.  There is no ``component``
keyword: the component is the one belonging to the equation named by
``variable``, so add one object per displacement:

.. code-block:: python

   # A uniform shear of 1 MPa on the right edge, acting along +y.
   problem.add_boundary_condition("TractionBC", "shear",
                                  variable="displacement_y", boundary="right",
                                  traction=1.0e6, thickness=0.01)
""",
)

_add(
    "PressureBC",
    """
``PressureBC`` applies a pressure normal to a surface, which is the natural way
to load a vessel, a dam, or a hole in a plate.  It needs one instance per
displacement component, each told which component it contributes to:

.. code-block:: python

   for component, name in enumerate(("displacement_x", "displacement_y")):
       problem.add_boundary_condition("PressureBC", f"internal_{name}",
                                      variable=name, boundary="inner",
                                      component=component, pressure=10.0e6)

A positive pressure pushes inward, against the outward normal.
""",
)

# ---------------------------------------------------------------------------
# Beams and plates
# ---------------------------------------------------------------------------
_add(
    "BeamEulerBernoulliMixed",
    """
The classical beam theory gives a fourth-order equation in the deflection,
which the dual mesh control domain method cannot discretize, so the bending
moment is carried as a third unknown and the system becomes three second-order
equations.  Add the kernel once per variable, with identical parameters:

.. code-block:: python

   for name in ("axial", "deflection", "moment"):
       problem.add_variable(name)
   for name in ("axial", "deflection", "moment"):
       problem.add_kernel("BeamEulerBernoulliMixed", f"beam_{name}", variable=name,
                          axial_displacement="axial",
                          transverse_displacement="deflection",
                          bending_moment="moment",
                          extensional_stiffness=a_xx, bending_stiffness=d_xx,
                          transverse_load=-1.0e3)

The boundary conditions follow the duality pairs: a clamped end prescribes the
deflection and leaves the moment free, because the vanishing slope is the
natural condition of the moment equation; a simply supported end prescribes the
deflection *and* prescribes the moment to be zero.

:func:`dualmesh.physics.add_beam` writes all three kernels in one call.
""",
)

_add(
    "BeamTimoshenkoDisplacement",
    """
The shear-deformable beam in displacement form carries the axial displacement,
the deflection, and the rotation of the cross-section.  A thin beam locks
unless the shear terms are integrated with one point, which is what splitting
the kernel in two achieves:

.. code-block:: python

   common = dict(axial_displacement="axial", transverse_displacement="deflection",
                 rotation="rotation", extensional_stiffness=a_xx,
                 bending_stiffness=d_xx, shear_stiffness=s_xz,
                 transverse_load=-1.0e3)
   for name in ("axial", "deflection", "rotation"):
       problem.add_kernel("BeamTimoshenkoDisplacement", f"bend_{name}",
                          variable=name, shear_treatment="exclude", **common)
       problem.add_kernel("BeamTimoshenkoDisplacement", f"shear_{name}",
                          variable=name, shear_treatment="only",
                          quadrature="midpoint", reduced_integration=True, **common)

Using ``shear_treatment`` alone, without the matching second kernel, drops terms
and gives a wrong answer.
""",
)

_add(
    "BeamTimoshenkoMixed",
    """
The mixed form of the shear-deformable beam carries the bending moment instead
of the rotation.  It is free of shear locking without any reduced integration,
which makes it the more robust of the two Timoshenko models, and the rotation
is recovered afterwards from :math:`\\phi = -dw/dx + (1/S)\\,dM/dx`:

.. code-block:: python

   for name in ("axial", "deflection", "moment"):
       problem.add_kernel("BeamTimoshenkoMixed", f"beam_{name}", variable=name,
                          axial_displacement="axial",
                          transverse_displacement="deflection",
                          bending_moment="moment",
                          extensional_stiffness=a_xx, bending_stiffness=d_xx,
                          shear_stiffness=s_xz, transverse_load=-1.0e3)
""",
)

_add(
    "CircularPlateFirstOrder",
    """
The shear-deformable model of an axisymmetric circular plate, on a
one-dimensional radial mesh with the axisymmetric coordinate system.  As for
the shear-deformable beam, the shear terms need their own reduced-integration
kernel:

.. code-block:: python

   mesh = dm.generate_line_mesh(start=0.0, end=radius, num_elements=32)
   problem = dm.Problem(mesh, coordinates="axisymmetric")
   dm.physics.add_circular_plate(problem, theory="first_order",
                                 extensional_stiffness=a_rr,
                                 coupling_stiffness=b_rr,
                                 bending_stiffness=d_rr,
                                 shear_stiffness=s_rz,
                                 poissons_ratio=0.3, transverse_load=q0)

Symmetry at the centre prescribes the radial displacement and the rotation to
be zero; the edge condition prescribes the deflection, and a clamped edge
additionally prescribes the rotation.
""",
)

_add(
    "CircularPlateClassicalMixed",
    """
The classical, shear-rigid model of an axisymmetric circular plate, written in
terms of the radial displacement, the deflection, and the radial bending
moment.  This is Reddy's DM-CP(M) model:

.. code-block:: python

   mesh = dm.generate_line_mesh(start=0.0, end=radius, num_elements=32)
   problem = dm.Problem(mesh, coordinates="axisymmetric")
   dm.physics.add_circular_plate(problem, theory="classical",
                                 extensional_stiffness=a_rr,
                                 coupling_stiffness=b_rr,
                                 bending_stiffness=d_rr,
                                 poissons_ratio=0.3, transverse_load=q0)
   problem.add_boundary_condition("DirichletBC", "centre",
                                  variable="radial_displacement",
                                  boundary="left", value=0.0)
   problem.add_boundary_condition("DirichletBC", "edge_u",
                                  variable="radial_displacement",
                                  boundary="right", value=0.0)
   problem.add_boundary_condition("DirichletBC", "edge_w",
                                  variable="deflection", boundary="right", value=0.0)
   # A simply supported edge adds: bending_moment = 0 on "right".
   # A clamped edge adds nothing: the vanishing slope is natural there.

The moment is a nodal unknown, so it is available directly rather than having
to be differentiated out of the deflection.  It is accurate everywhere except
at the centre, which is a known property of mixed models and is stated as such
in Reddy's Section 8.6.
""",
)

_add(
    "PlateFirstOrder",
    """
The shear-deformable rectangular plate carries five variables: two in-plane
displacements, the deflection, and two rotations.  With ``von_karman=True`` it
models moderate rotations, which is what a thin plate under a load large enough
to stretch its mid-plane needs:

.. code-block:: python

   dm.physics.add_plate(problem,
                        in_plane_displacements=["u", "v"],
                        transverse_displacement="w",
                        rotations=["phi_x", "phi_y"],
                        extensional_stiffness=a, coupling_stiffness=b,
                        bending_stiffness=d, shear_stiffness=s,
                        poissons_ratio=0.3, transverse_load=q0,
                        von_karman=True)
   problem.solve(load_factors=[0.1 * k for k in range(1, 11)])

Load stepping is not optional for the nonlinear case: starting from rest at the
full load usually fails to converge.
""",
)

# ---------------------------------------------------------------------------
# Fluids
# ---------------------------------------------------------------------------
_add(
    "ViscousStress",
    """
``ViscousStress`` is the viscous term of the momentum equation.  With the
penalty term it forms a Stokes flow; adding ``ConvectiveInertia`` makes it the
Navier-Stokes equations:

.. code-block:: python

   velocities = ["velocity_x", "velocity_y"]
   for component, name in enumerate(velocities):
       problem.add_kernel("ViscousStress", f"viscous_{name}", variable=name,
                          velocities=velocities, component=component,
                          dynamic_viscosity=1.0)

:func:`dualmesh.physics.add_incompressible_flow` assembles the whole set.
""",
)

_add(
    "PenaltyIncompressibility",
    """
The penalty formulation replaces the pressure by :math:`P = -\\gamma \\nabla
\\cdot v`, which removes the pressure unknown altogether and leaves a problem in
the velocities alone.  The price is that the term must be under-integrated:

.. code-block:: python

   for component, name in enumerate(velocities):
       problem.add_kernel("PenaltyIncompressibility", f"penalty_{name}",
                          variable=name, velocities=velocities,
                          component=component, penalty_parameter=1.0e8)

The defaults already select the one-point rule and the centroid evaluation.  If
the velocity field comes out identically zero, the penalty parameter is almost
certainly too large for the viscosity, and the constraint has locked the
element.
""",
)

_add(
    "ConvectiveInertia",
    """
``ConvectiveInertia`` adds the nonlinear transport term of the Navier-Stokes
equations.  Leave it out for a Stokes flow rather than setting the density to
zero:

.. code-block:: python

   for component, name in enumerate(velocities):
       problem.add_kernel("ConvectiveInertia", f"inertia_{name}", variable=name,
                          velocities=velocities, component=component, density=1.0)
   problem.solve(nonlinear_solver="newton",
                 load_factors=[0.1, 0.25, 0.5, 0.75, 1.0])

At a high Reynolds number, starting from rest at the full velocity usually
diverges.  Either ramp the boundary velocity with load stepping, as above, or
use direct iteration with a relaxation factor of about one half, which is the
strategy Reddy's cavity example uses.
""",
)

_add(
    "BoussinesqBuoyancy",
    """
``BoussinesqBuoyancy`` makes a temperature field drive the flow.  Add it to the
momentum equation of every velocity component that has a component of gravity,
usually just the vertical one.  In the non-dimensional form of the natural
convection benchmark (lengths scaled by the cavity width, velocities by
:math:`\\kappa / L`, temperature by the wall temperature difference) the
coefficient :math:`\\rho_0 \\beta g` becomes :math:`Ra \\, Pr`:

.. code-block:: python

   problem.add_kernel("BoussinesqBuoyancy", "buoyancy", variable="v", component=1,
                      temperature="temperature", gravity=[0.0, -1.0],
                      thermal_expansion=rayleigh * prandtl, scale_with_load=True)
   problem.solve(load_factors=[0.01, 0.1, 1.0])

With ``scale_with_load=True`` load stepping ramps the Rayleigh number, which is
how a strongly buoyant flow is reached from rest.  The complete problem is
``examples/natural_convection.py``.
""",
)

_add(
    "HeatConvection",
    """
``HeatConvection`` is the other half of the coupling: the flow carries the
heat.  The velocities are variables of the same problem, so Newton's method
sees the coupling in both directions:

.. code-block:: python

   velocities = dm.physics.add_incompressible_flow(problem, velocities=["u", "v"],
                                                   dynamic_viscosity=0.71, density=1.0)
   problem.add_variable("temperature")
   problem.add_kernel("HeatConduction", "conduction", variable="temperature")
   problem.add_kernel("HeatConvection", "convection", variable="temperature",
                      velocities=velocities, density=1.0, specific_heat=1.0)

The term is written in the advective (non-conservative) form, which is exact
for an incompressible flow.
""",
)

_add(
    "PenaltyPressure",
    """
The penalty formulation never solves for a pressure, so the pressure has to be
recovered afterwards from the velocity divergence.  ``PenaltyPressure``
publishes it as a material property, which the post-processing can then
evaluate at the element centroids:

.. code-block:: python

   problem.add_material("PenaltyPressure", "pressure", velocities=velocities,
                        penalty_parameter=1.0e8)
   problem.solve()
   pressure = problem.property_at_centroids("pressure")

The penalty parameter must be exactly the one given to
``PenaltyIncompressibility``, or the recovered pressure is meaningless.
""",
)
