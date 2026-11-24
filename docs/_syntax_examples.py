# SPDX-License-Identifier: LGPL-2.1-or-later
"""Worked usage examples for the generated syntax reference.

Every registered object has one entry.  The text says when the object is used
and what its keywords do in that context, and the code block below it is a
fragment that runs on a problem with the variables and boundaries it names.
An object missing from this file still receives a generated page without the
usage section, and the documentation build prints a warning that names the
object.
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
    r"""
``diffusion`` is the generic second-order operator, with the flux
:math:`\mathbf{F} = -k \nabla u` of the unknown :math:`u` and the diffusivity
:math:`k`.  Use it when the flux is proportional to the gradient of the
unknown and no physics module has a kernel with a more specific name.  The
diffusivity is a number, a named function of position and time, or a
property, and a polynomial in the unknown can multiply it, which makes the
problem nonlinear:

.. code-block:: python

   problem.add_kernel("diffusion", variable="u", diffusivity=2.5)

   # A diffusivity that varies through the body.
   problem.add_function("k", lambda x, y, z, t: 1.0 + 4.0 * x)
   problem.add_kernel("diffusion", variable="u", diffusivity="k")

   # A diffusivity that depends on the solution: k_eff = 1.0 * (1 + 0.5 u).
   problem.add_kernel("diffusion", variable="u", diffusivity=1.0, solution_polynomial=[1.0, 0.5])

The last form needs a nonlinear solve.  With ``nonlinear_solver="Newton"``
the automatic differentiation carries the derivative of the polynomial
exactly, so the iteration converges quadratically.
""",
)

_add(
    "anisotropic_diffusion",
    """
Use ``anisotropic_diffusion`` when the conductivity or the permeability
depends on direction, as in a laminate, a fibre-reinforced solid or a layered
rock.  Give the tensor as its diagonal or in full, row by row:

.. code-block:: python

   # Orthotropic: ten times more conductive along x than along y.
   problem.add_kernel("anisotropic_diffusion", variable="temperature", diffusivity_tensor=[10.0, 1.0])

   # The full tensor, row by row, for principal axes rotated by 30 degrees.
   import numpy

   c, s = numpy.cos(numpy.pi / 6), numpy.sin(numpy.pi / 6)
   k = numpy.array([[10.0, 0.0], [0.0, 1.0]])
   r = numpy.array([[c, -s], [s, c]])
   problem.add_kernel("anisotropic_diffusion", variable="temperature", diffusivity_tensor=list((r @ k @ r.T).ravel()))
""",
)

_add(
    "reaction",
    r"""
``reaction`` adds a term proportional to the unknown, which models
radioactive decay, chemical consumption, the heat loss along a fin and the
restoring force of an elastic foundation.  A positive coefficient removes the
unknown:

.. code-block:: python

   # A cooling fin, -u'' + 400 u = 0, with 400 = P h_c / (k A) for the perimeter P,
   # the film coefficient h_c, the conductivity k and the cross-section area A.
   problem.add_kernel("diffusion", variable="u")
   problem.add_kernel("reaction", variable="u", coefficient=400.0)

   # A beam on an elastic foundation of modulus 1e6.
   problem.add_kernel("reaction", variable="deflection", coefficient=1.0e6)

   # A second-order chemical reaction, which is nonlinear.
   problem.add_kernel("reaction", variable="concentration", coefficient=0.1, exponent=2.0)
""",
)

_add(
    "body_force",
    """
``body_force`` is the generic distributed source.  Each physics module has an
equivalent source with a physical name.  ``body_force`` takes any function,
and the verification problems use it for a source that has no physical name:

.. code-block:: python

   import numpy

   problem.add_function("source", lambda x, y, z, t: 10.0 * numpy.cos(x))
   problem.add_kernel("body_force", variable="u", value="source")

The ``quadrature`` keyword sets the rule that integrates the source.  The
trapezoidal rule over the whole control domain, the source rule of the finite
volume method of [Reddy2024]_, is ``control_domain_trapezoid``:

.. code-block:: python

   problem.add_kernel("body_force", variable="u", value="source", quadrature="control_domain_trapezoid")
""",
)

_add(
    "advection",
    r"""
``advection`` transports the unknown with a prescribed velocity, so with
``diffusion`` it forms the advection-diffusion equation.  The velocity is
constant or given by functions, one per component:

.. code-block:: python

   problem.add_kernel("diffusion", variable="c", diffusivity=1.0)
   problem.add_kernel("advection", variable="c", velocity=[75.0, 0.0])

The cell Peclet number :math:`Pe_h = |\mathbf{v}| h / k`, with the velocity
:math:`\mathbf{v}`, the element size :math:`h` and the diffusivity :math:`k`,
compares advection with diffusion over one element.  The default
``advection_scheme="central"`` oscillates where :math:`Pe_h > 2`, which here
means :math:`h > 2/75`.  Refine the mesh below that size, or, with
``method="DMCDM"``, choose a face flux: ``advection_scheme="upwind"``,
``"linear_upwind"`` or ``"limited"``.

Choose ``advection_form="conservative"`` when the velocity field is
divergence free and the discrete equations must conserve the transported
quantity exactly over each control domain.
""",
)

_add(
    "coupled_force",
    """
``coupled_force`` couples two equations: it adds to one equation a source
proportional to another variable.  It represents, for example, a heat source
proportional to a concentration, or a two-species reaction:

.. code-block:: python

   problem.add_variable("a")
   problem.add_variable("b")
   problem.add_kernel("diffusion", variable="a")
   problem.add_kernel("diffusion", variable="b")
   problem.add_kernel("coupled_force", variable="a", coupled_variable="b", coefficient=-1.0)

The automatic differentiation carries the derivative with respect to the
coupled variable, so the off-diagonal block of the Jacobian is exact and
Newton's method converges quadratically for a strongly coupled system.
""",
)

_add(
    "second_time_derivative",
    r"""
``second_time_derivative`` is the inertia of a wave equation.  A string of
unit length fixed at both ends, :math:`\partial^2 u/\partial t^2 =
\partial^2 u/\partial x^2`, has the natural frequencies :math:`n/2` in
:math:`\mathrm{Hz}` for :math:`n = 1, 2, \dots`.  A transient run integrates
the inertia by the generalized-alpha method, and an eigenvalue study takes it
as the mass of the natural frequencies:

.. code-block:: python

   import numpy
   import dualmesh

   mesh = dualmesh.generate_line_mesh(start=0.0, end=1.0, num_elements=40)
   problem = dualmesh.Problem(mesh)
   problem.add_variable("u", initial_condition=lambda x, y, z, t: numpy.sin(numpy.pi * x))
   problem.add_kernel("diffusion", variable="u")
   problem.add_kernel("second_time_derivative", variable="u", coefficient=1.0)
   problem.add_boundary_condition("Dirichlet", ["left", "right"], variable="u", value=0.0)
   problem.solve_transient(end_time=1.0, time_step=0.01)
   problem.solve_eigenvalue(num_modes=3).frequencies  # close to 0.5, 1.0 and 1.5 Hz

The ``density`` of ``solid_mechanics`` adds it to every displacement.
""",
)

_add(
    "stiffness_damping",
    r"""
``stiffness_damping`` is the stiffness-proportional part of Rayleigh damping,
which ``solid_mechanics`` adds for ``damping={"stiffness": b}``:

.. code-block:: python

   problem.add_kernel("stiffness_damping", variable=["u", "v"], displacements=["u", "v"], formulation="plane_strain", coefficient=1e-5)

A mode of angular frequency :math:`\omega` in :math:`\mathrm{rad/s}` then has
the damping ratio :math:`\zeta = b \omega / 2`, with the coefficient :math:`b`
in :math:`\mathrm{s}`.
""",
)

_add(
    "time_derivative",
    """
``time_derivative`` makes a steady equation transient.  Add it to every
equation that evolves in time.  An equation without it holds at every
instant, as the moment equation of a mixed beam does.

.. code-block:: python

   problem.add_variable("u", initial_condition=lambda x, y, z, t: numpy.sin(numpy.pi * x))
   problem.add_kernel("diffusion", variable="u")
   problem.add_kernel("time_derivative", variable="u", coefficient=1.0)
   problem.solve_transient(end_time=0.1, time_step=0.001, time_scheme="Crank-Nicolson")

To lump the capacity, give the nodal quadrature:

.. code-block:: python

   problem.add_kernel("time_derivative", variable="u", quadrature="nodal")
""",
)

_add(
    "Dirichlet",
    """
``Dirichlet`` prescribes the value of a variable, the primary member of the
duality pair.  It replaces the equation of each prescribed degree of freedom,
so the residual of that equation is the reaction, the flux that the condition
supplies:

.. code-block:: python

   problem.add_boundary_condition("Dirichlet", variable="temperature", boundary="left", value=300.0)

   # A value that varies along the boundary.
   problem.add_function("profile", lambda x, y, z, t: numpy.cos(numpy.pi * x / 6.0))
   problem.add_boundary_condition("Dirichlet", variable="temperature", boundary="top", value="profile")

   # One value on several variables: the condition is created once per variable.
   problem.add_boundary_condition("Dirichlet", variable=["u", "v"], boundary="bottom", value=0.0)

   # The heat that flows in through the left edge to hold it at 300 K.
   print(-problem.boundary_flux("temperature", "left"))

A boundary without a boundary condition has a zero normal flux, which is an
insulated surface in heat transfer and a traction-free surface in elasticity.
""",
)

_add(
    "Neumann",
    r"""
``Neumann`` prescribes the secondary member of the duality pair: ``flux`` is
the flux :math:`-\mathbf{n} \cdot \mathbf{F}` that enters the body, with the
flux :math:`\mathbf{F}` of the equation and the outward unit normal
:math:`\mathbf{n}`.  In heat conduction :math:`\mathbf{F}` is the heat flux,
so a positive ``flux`` is heat that enters the body:

.. code-block:: python

   # One watt per square metre entering through the right edge.
   problem.add_boundary_condition("Neumann", variable="temperature", boundary="right", flux=1.0)

A boundary without a boundary condition has a zero flux, so an insulated or
symmetry boundary needs no object.  The physics write the same condition as
text, with the flux that leaves the body:
``heat.add_boundary_condition("heat_flux = -1", boundary="right")``.
""",
)

_add(
    "Robin",
    r"""
``Robin`` is the mixed condition :math:`-\mathbf{n} \cdot \mathbf{F} = q_0 -
h_c (u - u_\infty)` on the flux that enters the body, with the flux
:math:`\mathbf{F}` of the equation, the outward unit normal :math:`\mathbf{n}`,
the prescribed part :math:`q_0` (``flux``), the transfer coefficient
:math:`h_c` (``transfer_coefficient``) and the ambient value
:math:`u_\infty` (``ambient_value``).  It describes convection into a
surrounding medium, a contact resistance and a spring support.  ``Neumann``
is the case :math:`h_c = 0`:

.. code-block:: python

   # Convection from a fin tip into air at 293.15 K.
   problem.add_boundary_condition("Robin", variable="temperature", boundary="right", transfer_coefficient=2.0, ambient_value=293.15)

The physics write the same condition as text, in their own terms:
``heat.add_boundary_condition("heat_flux = 2*(temperature - 293.15)", boundary="right")``.
""",
)

_add(
    "periodic",
    """
``periodic`` joins two opposite boundaries, so that every node of the
secondary boundary takes the value of the primary node it faces.  It models
one cell of a structure that repeats, such as a channel of a heat exchanger
or a unit cell of a composite.  The translation from a primary node to its
secondary node is the vector between the centroids of the two boundaries
unless you give it, and the meshes of the two boundaries must match node by
node after it:

.. code-block:: python

   # Periodic in x on the unit square: the right edge repeats the left edge.
   problem.add_boundary_condition("periodic", primary="left", secondary="right")

   # Periodic in y as well. The corner nodes are then joined in both directions.
   problem.add_boundary_condition("periodic", primary="bottom", secondary="top")

   # Only the velocity is periodic, in a channel two metres long.
   problem.add_boundary_condition("periodic", primary="inlet", secondary="outlet", variable=["u", "v"], translation=[2.0, 0.0])

A value prescribed on the secondary boundary must also be prescribed on the
primary one.  The condition works with ``method="FEM"`` and
``method="DMCDM"``, in serial and in parallel.
""",
)

_add(
    "point_source",
    """
``point_source`` applies a concentrated load or a point source of heat at one
or more nodes.  The load acts at the node nearest to each given point,
however far away, so place the point on a node of the mesh:

.. code-block:: python

   # A concentrated force at the centre of the span.
   problem.add_point_source(variable="deflection", value=-1000.0, points=[0.5])

   # A heat source at every node of a named node set.
   problem.add_point_source(variable="temperature", value=50.0, boundary="heater_nodes")

In a plane-stress or plane-strain model the thickness multiplies a traction
and leaves a point source unchanged, so give the total force through the
thickness.
""",
)

_add(
    "constant",
    """
``constant`` defines named constant properties that kernels read by name.
Several kernels then share one value, and a property can take different
values in different blocks:

.. code-block:: python

   problem.add_property("constant", properties={"conductivity": 400.0}, block=["copper_bar"])
   problem.add_property("constant", properties={"conductivity": 45.0}, block=["steel_bar"])
   problem.add_kernel("heat_conduction", variable="temperature", thermal_conductivity="conductivity")
""",
)

_add(
    "function",
    """
``function`` is the counterpart of ``constant`` for a property that varies
through the body.  Register the functions on the problem first:

.. code-block:: python

   problem.add_function("k_of_x", lambda x, y, z, t: 50.0 * (1.0 + x))
   problem.add_property("function", properties={"conductivity": "k_of_x"})

A property that depends on the solution is a ``parsed`` property, or a Python
subclass of ``PythonProperty``, as :doc:`/user_guide/problem_setup` shows.
""",
)

# ---------------------------------------------------------------------------
# Heat transfer
# ---------------------------------------------------------------------------
_add(
    "heat_conduction",
    r"""
``heat_conduction`` states Fourier's law, the heat flux
:math:`\mathbf{q} = -k \nabla T` with the thermal conductivity :math:`k` in
:math:`\mathrm{W/(m\,K)}` and the temperature :math:`T` in :math:`\mathrm{K}`:

.. code-block:: python

   problem.add_variable("temperature")
   problem.add_kernel("heat_conduction", variable="temperature", thermal_conductivity=20.0)

A polynomial in the temperature can multiply the conductivity:

.. code-block:: python

   # k(T) = 45 (1 + 0.002 T)
   problem.add_kernel("heat_conduction", variable="temperature", thermal_conductivity=45.0, temperature_polynomial=[1.0, 0.002])
   problem.solve(nonlinear_solver="Newton")
""",
)

_add(
    "heat_conduction_time_derivative",
    """
``heat_conduction_time_derivative`` is the heat capacity term of the energy
equation and makes a thermal problem transient:

.. code-block:: python

   problem.add_kernel("heat_conduction", variable="temperature", thermal_conductivity=45.0)
   problem.add_kernel("heat_conduction_time_derivative", variable="temperature", density=7850.0, specific_heat=460.0)
   problem.solve_transient(end_time=600.0, time_step=5.0, time_scheme="Crank-Nicolson")

Only the product of the density and the specific heat enters, so you can
give a volumetric heat capacity as the density with the specific heat set to
one.
""",
)

_add(
    "radiative_heat_flux",
    """
``radiative_heat_flux`` adds grey-body radiation to a surface.  The flux
varies with the fourth power of the temperature, so the condition is strongly
nonlinear, and it needs temperatures on an absolute scale:

.. code-block:: python

   # A surface at a few hundred kelvin radiating to surroundings at 300 K.
   problem.add_boundary_condition("radiative_heat_flux", variable="temperature", boundary="top", emissivity=0.8, ambient_temperature=300.0)
   problem.solve(nonlinear_solver="Newton")

Radiation and convection often act together.  The fluxes of two objects on
the same side set add up.
""",
)

# ---------------------------------------------------------------------------
# Solid mechanics
# ---------------------------------------------------------------------------
_add(
    "stress_divergence",
    r"""
``stress_divergence`` is the equilibrium equation of the displacements.  One
call adds the equation of every displacement listed in ``displacements``, and
``variable`` restricts it to some of them:

.. code-block:: python

   problem.add_kernel("stress_divergence", displacements=["displacement_x", "displacement_y"], thickness=0.01)

The thickness multiplies the whole equation of a plane problem, and the
``traction`` and ``pressure`` conditions of the same displacement read it
from here.  ``mortar_contact`` takes it as its own parameter.  A
three-dimensional or axisymmetric problem refuses a thickness other than 1,
because the integration measure of the axisymmetric case already carries the
factor :math:`2 \pi r`, with the radius :math:`r`.
""",
)

_add(
    "traction",
    """
``traction`` applies one component of a surface traction, that of the
equation named by ``variable``.  The load is the traction itself, in force
per unit area, or the total force on the boundary, which the condition
spreads uniformly over the area of the boundary.  ``traction`` and
``total_force`` define the same load, so give one of them:

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
``pressure`` applies a pressure normal to a surface, the usual load on a
vessel, a dam or the surface of a hole in a plate.  One call loads every
displacement listed in ``displacements``:

.. code-block:: python

   problem.add_boundary_condition("pressure", boundary="inner", displacements=["displacement_x", "displacement_y"], pressure=10.0e6)

A positive pressure acts inward, against the outward normal.
""",
)

_add(
    "normal_displacement",
    r"""
``normal_displacement`` prescribes the displacement along the outward unit
normal :math:`\mathbf{n}` of a boundary, :math:`\mathbf{n} \cdot \mathbf{u} =
g`, and leaves the tangential displacements free: a roller on an inclined or
curved face.  A Lagrange multiplier, the variable of the condition, is the
normal force at each node of the boundary and holds the condition exactly.
``stiffness_scale``, of the size of the stiffness of a node, scales the
equation of the condition and changes the conditioning only.  The physics
level adds it with
``solid.add_boundary_condition("normal_displacement = 0", boundary=...)``:

.. code-block:: python

   problem.add_variable("normal_force", unit="N")
   problem.add_boundary_condition("normal_displacement", "inclined_face", variable="normal_force", displacements=["disp_x", "disp_y"], value=0.0, stiffness_scale=2.0e10)
""",
)

# ---------------------------------------------------------------------------
# Beams and plates
# ---------------------------------------------------------------------------
_add(
    "beam_Euler-Bernoulli_mixed",
    """
The classical beam theory gives a fourth-order equation in the deflection.
The dual mesh control domain method (DMCDM) discretizes second-order
equations, so the bending moment is a third unknown and the system becomes
three second-order equations.  One call adds the equations of the three
variables:

.. code-block:: python

   for name in ("axial", "deflection", "moment"):
       problem.add_variable(name)
   problem.add_kernel("beam_Euler-Bernoulli_mixed", axial_displacement="axial", transverse_displacement="deflection", bending_moment="moment", extensional_stiffness=2.0e9, bending_stiffness=2.0e6, transverse_load=-1.0e3)

The boundary conditions follow the duality pairs.  A clamped end prescribes
the deflection and leaves the moment free, because the zero slope is the
natural condition of the moment equation.  A simply supported end prescribes
the deflection and a zero moment.  ``solid_mechanics`` with
``formulation="beam"`` adds these kernels.
""",
)

_add(
    "beam_Timoshenko_displacement",
    """
The shear-deformable beam in displacement form carries the axial
displacement, the deflection and the rotation of the cross-section.  A thin
beam locks in shear unless the shear terms are integrated with one point, so
the kernel of the rotation equation takes the one-point rule, as the
``solid_mechanics`` physics adds it:

.. code-block:: python

   beam = dict(axial_displacement="axial", transverse_displacement="deflection", rotation="rotation", extensional_stiffness=2.0e9, bending_stiffness=2.0e6, shear_stiffness=6.4e8, transverse_load=-1.0e3)
   for variable in ("axial", "deflection"):
       problem.add_kernel("beam_Timoshenko_displacement", variable=variable, **beam)
   problem.add_kernel("beam_Timoshenko_displacement", variable="rotation", quadrature="midpoint", reduced_integration=True, **beam)
""",
)

_add(
    "beam_Timoshenko_mixed",
    r"""
The mixed form of the shear-deformable beam carries the axial displacement,
the deflection and the bending moment as unknowns.  It is free of shear
locking with full integration of every term.  The rotation of the
cross-section follows from :math:`\phi = -dw/dx + (1/S)\, dM/dx`, with the
deflection :math:`w` in :math:`\mathrm{m}`, the bending moment :math:`M` in
:math:`\mathrm{N\,m}` and the shear stiffness :math:`S` in :math:`\mathrm{N}`:

.. code-block:: python

   problem.add_kernel("beam_Timoshenko_mixed", axial_displacement="axial", transverse_displacement="deflection", bending_moment="moment", extensional_stiffness=2.0e9, bending_stiffness=2.0e6, shear_stiffness=6.4e8, transverse_load=-1.0e3)
""",
)

_add(
    "circular_plate_first_order",
    """
``circular_plate_first_order`` is the shear-deformable model of an
axisymmetric circular plate, on a one-dimensional radial mesh in axisymmetric
coordinates [ReddyBeams2022]_.  ``solid_mechanics`` with
``formulation="circular_plate"`` adds it twice: once without the transverse
shear (``shear_treatment="exclude"``) and once with the shear alone on a
reduced rule (``shear_treatment="only"``), which removes shear locking in
thin plates:

.. code-block:: python

   import dualmesh

   mesh = dualmesh.generate_line_mesh(start=0.0, end=1.0, num_elements=32)
   problem = dualmesh.Problem(mesh, coordinates="axisymmetric")
   plate = problem.add_physics("solid_mechanics", formulation="circular_plate", theory="first_order", youngs_modulus=2.0e11, poissons_ratio=0.3, thickness=0.01, transverse_load=1.0e3)
   plate.add_boundary_condition(["disp_r = 0", "phi = 0"], boundary="left")
   plate.add_boundary_condition("disp_z = 0", boundary="right")

Symmetry at the centre prescribes the radial displacement and the rotation
to be zero.  The edge condition prescribes the deflection, and a clamped edge
prescribes the rotation as well.
""",
)

_add(
    "circular_plate_classical_mixed",
    """
``circular_plate_classical_mixed`` is the classical, shear-rigid model of an
axisymmetric circular plate [ReddyBeams2022]_, written in terms of the radial
displacement, the deflection and the radial bending moment:

.. code-block:: python

   import dualmesh

   mesh = dualmesh.generate_line_mesh(start=0.0, end=1.0, num_elements=32)
   problem = dualmesh.Problem(mesh, coordinates="axisymmetric")
   plate = problem.add_physics("solid_mechanics", formulation="circular_plate", theory="classical", youngs_modulus=2.0e11, poissons_ratio=0.3, thickness=0.01, transverse_load=1.0e3)
   plate.add_boundary_condition("disp_r = 0", boundary="left")
   plate.add_boundary_condition(["disp_r = 0", "disp_z = 0"], boundary="right")
   # The physics holds the moment of the simply supported edge at zero.  A clamped
   # edge ("fixed") keeps its zero slope, the natural condition of the moment.

The moment is a nodal unknown, so the solution gives it without a
differentiation of the deflection.
""",
)

_add(
    "plate_first_order",
    """
The shear-deformable rectangular plate carries five variables: two in-plane
displacements, the deflection and two rotations.  With ``von_karman=True`` it
models moderate rotations, as a thin plate needs under a load large enough to
stretch its mid-plane:

.. code-block:: python

   import dualmesh

   mesh = dualmesh.generate_rectangle_mesh(x=[0.0, 1.0], y=[0.0, 1.0], num_elements=[16, 16])
   problem = dualmesh.Problem(mesh)
   plate = problem.add_physics("solid_mechanics", formulation="plate", youngs_modulus=2.0e11, poissons_ratio=0.3, thickness=0.01, transverse_load=1.0e5, von_karman=True)
   plate.add_boundary_condition(["disp_x = 0", "disp_y = 0", "disp_z = 0"], boundary=["left", "right", "bottom", "top"])
   problem.solve()

When Newton's method does not converge at the full load, the solve ramps the
load up from zero by itself.
""",
)

# ---------------------------------------------------------------------------
# Fluids
# ---------------------------------------------------------------------------
_add(
    "viscous_stress",
    """
``viscous_stress`` is the viscous term of the momentum equation.  With a
treatment of the pressure, the penalty term or the ``pressure_gradient`` and
``mass_conservation`` kernels, it describes a Stokes flow, and
``convective_inertia`` adds the inertia of the Navier-Stokes equations:

.. code-block:: python

   velocities = ["velocity_x", "velocity_y"]
   problem.add_kernel("viscous_stress", variable=velocities, velocities=velocities, dynamic_viscosity=1.0)

The ``fluid_dynamics`` physics adds the complete set of kernels.
""",
)

_add(
    "penalty_incompressibility",
    r"""
The penalty formulation replaces the pressure by
:math:`p = -\gamma \nabla \cdot \mathbf{v}`, with the pressure :math:`p` in
:math:`\mathrm{Pa}`, the penalty parameter :math:`\gamma` in
:math:`\mathrm{Pa\,s}` and the velocity :math:`\mathbf{v}` in
:math:`\mathrm{m/s}`.  This leaves a problem in the velocities alone:

.. code-block:: python

   velocities = ["velocity_x", "velocity_y"]
   problem.add_kernel("penalty_incompressibility", variable=velocities, velocities=velocities, penalty_parameter=1.0e8)

The penalty term needs a reduced quadrature rule, which is its default, since
a rule with more points locks the velocity field.
""",
)

_add(
    "convective_inertia",
    """
``convective_inertia`` adds the nonlinear transport term of the
Navier-Stokes equations.  A Stokes flow leaves this kernel out:

.. code-block:: python

   velocities = ["velocity_x", "velocity_y"]
   problem.add_kernel("convective_inertia", variable=velocities, velocities=velocities, density=1.0)
   problem.solve(nonlinear_solver="Newton")

When Newton's method does not converge from rest at a high Reynolds number,
the solve ramps the boundary velocity up from rest by itself.
""",
)

_add(
    "Boussinesq_buoyancy",
    r"""
``Boussinesq_buoyancy`` adds the buoyancy force that a temperature field
exerts on the flow.  Add it to the momentum equation of every velocity
component along which gravity has a component, often the vertical one only.
In the non-dimensional form of the natural convection benchmark
[DeVahlDavis1983]_, the lengths are scaled by the cavity width :math:`L`, the
velocities by :math:`\kappa / L` with the thermal diffusivity :math:`\kappa`,
and the temperature by the temperature difference :math:`\Delta T` of the
walls.  The coefficient :math:`\rho \beta |\mathbf{g}|` of the buoyancy, with
the density :math:`\rho`, the volumetric expansion coefficient :math:`\beta`
and the gravitational acceleration :math:`\mathbf{g}`, then becomes
:math:`Ra \, Pr`.  Here :math:`Ra = \rho \beta |\mathbf{g}| \Delta T L^3 /
(\mu \kappa)` is the Rayleigh number and :math:`Pr = \mu / (\rho \kappa)` the
Prandtl number, with the dynamic viscosity :math:`\mu`:

.. code-block:: python

   rayleigh, prandtl = 1.0e5, 0.71
   problem.add_kernel("Boussinesq_buoyancy", variable="v", velocities=["u", "v"], temperature="temperature", gravity=[0.0, -1.0], volumetric_expansion_coefficient=rayleigh * prandtl)
   problem.solve()

The default ``density`` of 1 is the value of the non-dimensional equations.
When Newton's method does not converge on a strongly buoyant flow, the steady
solve ramps the buoyancy up from a fluid at rest.  The ``nonisothermal_flow``
coupling of a ``heat_transfer`` and a ``fluid_dynamics`` physics adds this
kernel when it is given ``gravity`` and
``volumetric_expansion_coefficient``.  The complete problem is in
``examples/natural_convection.py``.
""",
)

_add(
    "heat_convection",
    r"""
``heat_convection`` is the heat that a flow carries,
:math:`\rho c_p \mathbf{v} \cdot \nabla T`, with the density :math:`\rho`,
the specific heat capacity :math:`c_p`, the velocity :math:`\mathbf{v}` and
the temperature :math:`T`.  The velocities are variables of the same problem,
so the Jacobian of Newton's method holds the coupling in both directions:

.. code-block:: python

   problem.add_kernel("heat_convection", variable="temperature", velocities=["u", "v"], density=1.0, specific_heat=1.0)

The ``nonisothermal_flow`` coupling of a ``heat_transfer`` and a
``fluid_dynamics`` physics adds this kernel:

.. code-block:: python

   flow = problem.add_physics("fluid_dynamics", dynamic_viscosity=0.71, density=1.0)
   heat = problem.add_physics("heat_transfer", thermal_conductivity=1.0)
   problem.add_coupling("nonisothermal_flow", heat_transfer=heat, fluid_dynamics=flow)

``advection_scheme="central"``, the default, takes the temperature at each
integration point.  ``"upwind"``, ``"linear_upwind"`` and ``"limited"`` make
the term a face flux with the face value of that scheme, for
``method="DMCDM"``.
""",
)

_add(
    "heat_streamline_stabilization",
    """
``heat_streamline_stabilization`` adds the streamline upwind Petrov-Galerkin
(SUPG) term of the energy equation.  Its residual is the sum of the sources
of the other kernels of the temperature (the convection, the heat sources and
the heat capacity term), so a source added as a kernel is part of it:

.. code-block:: python

   problem.add_kernel("heat_streamline_stabilization", variable="temperature", velocities=["u", "v"], thermal_conductivity=1.0e-4)

The ``nonisothermal_flow`` coupling adds it, with the values of the
``heat_transfer`` physics, when that physics has ``advection_scheme="SUPG"``,
its default:

.. code-block:: python

   heat = problem.add_physics("heat_transfer", thermal_conductivity=1.0e-4)
""",
)

_add(
    "penalty_pressure",
    """
The penalty formulation eliminates the pressure from the unknowns, and the
pressure follows from the divergence of the velocity.  ``penalty_pressure``
gives it as the property ``pressure``, which the post-processing evaluates at
the element centroids:

.. code-block:: python

   velocities = ["velocity_x", "velocity_y"]
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
equation of a formulation with a pressure variable.  The ``fluid_dynamics``
physics adds it, with the viscous stress and the mass equation:

.. code-block:: python

   flow = problem.add_physics("fluid_dynamics", dynamic_viscosity=1.0, density=100.0, formulation="equal_order")

To add it yourself, give the momentum equations it belongs to:

.. code-block:: python

   problem.add_kernel("pressure_gradient", variable=["u", "v"], velocities=["u", "v"], pressure="pressure")
""",
)

_add(
    "mass_conservation",
    """
``mass_conservation`` is the mass equation, on the pressure variable.  With
equal-order interpolation it includes the pressure stabilisation, and it
reads the momentum residual of that stabilisation from the momentum kernels
of the same velocities (the density, the viscosity and the body forces):

.. code-block:: python

   problem.add_kernel("mass_conservation", variable="pressure", velocities=["u", "v"])
   problem.add_boundary_condition("mass_flux", variable="pressure", boundary=["left", "right", "bottom", "top"], velocities=["u", "v"])

An enclosed flow, whose boundaries all prescribe the velocity, fixes its
pressure up to a constant only.  The solve gives that pressure a mean of
zero, unless a ``point_Dirichlet`` holds it at a point.
""",
)

_add(
    "momentum_stabilization",
    """
``momentum_stabilization`` adds the streamline upwind term to one momentum
equation of a flow with inertia.  Like ``mass_conservation``, it reads the
momentum equations from their kernels:

.. code-block:: python

   problem.add_kernel("momentum_stabilization", variable=["u", "v"], pressure="pressure", velocities=["u", "v"])
""",
)

_add(
    "mass_flux",
    """
``mass_flux`` adds to the mass equation the flow through the boundary,
computed from the discrete velocity.  Apply it on every boundary without a
prescribed pressure.  The ``fluid_dynamics`` physics applies it on every side
set:

.. code-block:: python

   problem.add_boundary_condition("mass_flux", variable="pressure", boundary=["inlet", "outlet", "walls"], velocities=["u", "v"])
""",
)

_add(
    "point_Dirichlet",
    """
``point_Dirichlet`` fixes a variable at the single entity nearest to a given
point.  It sets, for example, the pressure level of an enclosed flow:

.. code-block:: python

   problem.add_boundary_condition("point_Dirichlet", variable="pressure", point=[0.5, 0.0], value=0.0)
""",
)

# ---------------------------------------------------------------------------
# Contact between bodies, eigenstrains and creep
# ---------------------------------------------------------------------------
_add(
    "gap_heat_transfer",
    r"""
``gap_heat_transfer`` transfers heat across a thin gap between two bodies
meshed separately, such as a shaft and the sleeve around it.  Each
integration point of ``boundary``, the primary surface, is paired with the
closest point of ``secondary_boundary``.  The heat flux
:math:`h_g (T_s - T_p)`, with the gap conductance :math:`h_g` in
:math:`\mathrm{W/(m^2\,K)}` and the temperatures :math:`T_s` and :math:`T_p`
of the secondary and primary surfaces, enters the primary body, and the same
heat leaves the secondary body.  So the condition conserves energy on
matching and non-matching meshes.  The conductance is a number or a
function, and radiation is added when both emissivities are positive:

.. code-block:: python

   problem.add_boundary_condition("gap_heat_transfer", variable="T", boundary=["shaft_outer"], secondary_boundary=["sleeve_inner"], gap_conductance=5000.0)
""",
)

_add(
    "out_of_plane_stress",
    r"""
``out_of_plane_stress`` is the axial equation of generalized plane strain, on
the uniform strain ``strain_zz``: the integral of the axial stress
:math:`\sigma_{zz}` over the section balances the axial force, which a
``body_force`` of :math:`F/A` on the same variable gives, with the force
:math:`F` and the area :math:`A` of the section.  The ``solid_mechanics``
physics adds it with ``formulation="generalized_plane_strain"``:

.. code-block:: python

   problem.add_kernel("out_of_plane_stress", variable="strain_zz")
""",
)

_add(
    "uniform_field",
    """
``uniform_field`` makes a variable one unknown on the whole mesh, as the
uniform axial strain of generalized plane strain is.  Every node becomes a
copy of the first, and the equations of all are added into one:

.. code-block:: python

   problem.add_boundary_condition("uniform_field", variable="strain_zz")
""",
)

_add(
    "mortar_contact",
    r"""
``mortar_contact`` keeps two surfaces of a two-dimensional problem from
passing through each other.  The contact pressure :math:`p_c`, a variable on
the nodes of the secondary surface, and the weighted gap :math:`g` satisfy
the complementarity conditions :math:`g \ge 0`, :math:`p_c \ge 0` and
:math:`p_c g = 0`, which Newton's method solves with an active set.  With a
``friction_coefficient`` the tangential traction, a second variable, follows
Coulomb's law.  The surfaces need not match, and they may slide.
``stiffness_scale``, of the order of :math:`E/h` with Young's modulus
:math:`E` and the element size :math:`h`, changes the convergence and leaves
the solution unchanged.  The physics level adds it with
``solid.add_boundary_condition("contact", primary=..., secondary=...)``:

.. code-block:: python

   problem.add_variable("contact_pressure")
   problem.add_boundary_condition("mortar_contact", "block_top", variable="contact_pressure", primary_boundary="punch_face", displacements=["disp_x", "disp_y"], stiffness_scale=1e13)
""",
)

_add(
    "truss",
    r"""
``truss`` is the axial force of the bars of a line mesh in a space of one,
two or three dimensions, :math:`N = E A\, \mathbf{t} \cdot \partial
\mathbf{u}/\partial s`, with no bending.  Here :math:`N` is the axial force in
:math:`\mathrm{N}`, :math:`E` Young's modulus, :math:`A` the area of the bar,
:math:`\mathbf{t}` the unit tangent of the bar, :math:`\mathbf{u}` the
displacement and :math:`s` the length along the bar.  One call adds the
equation of every displacement.  The physics level adds it with
``formulation="truss"``:

.. code-block:: python

   problem.add_kernel("truss", displacements=["disp_x", "disp_y", "disp_z"], youngs_modulus=2e11, area=1e-4)
""",
)

_add(
    "shell",
    """
``shell`` is a shell of flat facets on a quadrilateral surface mesh in three
dimensions: membrane, Mindlin bending and transverse shear in the axes of
each facet, with the rotation about the normal tied to the rotation of the
membrane.  The unknowns are the displacements and the rotations about the
global axes.  The second instance integrates the transverse shear alone at
the centre of the element, which removes shear locking.  The physics level
adds both with ``formulation="shell"``:

.. code-block:: python

   options = dict(displacements=["disp_x", "disp_y", "disp_z"], rotations=["rotation_x", "rotation_y", "rotation_z"], youngs_modulus=4.32e8, poissons_ratio=0.0, thickness=0.25)
   problem.add_kernel("shell", shear_treatment="exclude", **options)
   problem.add_kernel("shell", shear_treatment="only", quadrature="midpoint", reduced_integration=True, **options)
""",
)

_add(
    "cohesive_zone",
    r"""
``cohesive_zone`` bonds two surfaces of a two-dimensional problem that
coincide at the start, by a bilinear traction-separation law.  The traction
rises with the stiffness :math:`K` in :math:`\mathrm{Pa/m}` to the
``strength``, then softens to zero, so that the work of separation per unit
area is the ``toughness``.  The damage does not heal, and a closing zone
keeps its full stiffness.  The surfaces need not match.  The physics level
adds it with ``solid.add_boundary_condition("cohesive", primary=...,
secondary=..., strength=..., toughness=...)``, with :math:`K = 50 E/h`, Young's
modulus :math:`E` and the mean length :math:`h` of the sides of the secondary
surface:

.. code-block:: python

   problem.add_boundary_condition("cohesive_zone", "upper_bottom", variable="disp_x", primary_boundary="lower_top", displacements=["disp_x", "disp_y"], strength=3e6, toughness=100.0, stiffness=2.5e13)
""",
)

_add(
    "phase_field_damage",
    r"""
``phase_field_damage`` is the damage equation of phase-field fracture (AT2),
:math:`-G_c l \nabla^2 d + (G_c/l + 2H)\, d = 2H`, with the damage :math:`d`,
0 intact and 1 broken, the fracture toughness :math:`G_c` in
:math:`\mathrm{J/m^2}`, the length scale :math:`l` in :math:`\mathrm{m}` and
the history :math:`H` in :math:`\mathrm{J/m^3}` of the tensile elastic
energy, which the stress object stores as ``fracture_driving_energy``.  The
stress object degrades the tensile energy by :math:`(1 - d)^2`.  The physics
level adds it with ``phase_field_fracture={"toughness": ..., "length": ...}``:

.. code-block:: python

   problem.add_kernel("phase_field_damage", variable="damage", toughness=2.7e3, length=0.01)
""",
)

_add(
    "thermal_expansion_eigenstrain",
    r"""
``thermal_expansion_eigenstrain`` is the thermal strain of a material of
constant expansion coefficient,
:math:`\boldsymbol{\varepsilon}_{\mathrm{th}} = \alpha (T - T_0) \mathbf{I}`,
with the linear expansion coefficient :math:`\alpha` in :math:`\mathrm{1/K}`,
the temperature :math:`T`, the stress-free temperature :math:`T_0` and the
identity :math:`\mathbf{I}`.  It is stored as a property under
``eigenstrain_name``, so that ``small_strain_stress`` subtracts it from the
total strain:

.. code-block:: python

   problem.add_property("thermal_expansion_eigenstrain", temperature="T", linear_expansion_coefficient=1.0e-5, stress_free_temperature=293.15, eigenstrain_name="thermal_strain")
""",
)

_add(
    "small_strain_stress",
    r"""
``small_strain_stress`` computes the stress of an isotropic elastic solid
from the total strain minus the listed eigenstrains and, if a creep law is
given, the creep strain.  It integrates the creep strain by a backward Euler
radial return and stores it between time steps.  It provides the property
that ``stress_divergence`` reads, and one instance supplies the stress to the
equations of every displacement:

.. code-block:: python

   for name in ("displacement_x", "displacement_y"):
       problem.add_variable(name)
   problem.add_property("small_strain_stress", displacements=["displacement_x", "displacement_y"], formulation="plane_stress", youngs_modulus=200.0e9, poissons_ratio=0.3)
   problem.add_kernel("stress_divergence", displacements=["displacement_x", "displacement_y"])

``formulation`` selects plane stress, plane strain, axisymmetric
:math:`(r, z)`, three-dimensional, or ``axisymmetric_1d``, the radial slice of
a long body with a uniform ``axial_strain`` (generalized plane strain).
Young's modulus and Poisson's ratio are numbers, expressions or the
properties of another property object.  A thermal strain is an eigenstrain of
``thermal_expansion_eigenstrain``.  A creep law is an expression of
``von_mises_stress``, ``temperature``, ``grain_size`` and named constants:

.. code-block:: python

   creep_law = "A * von_mises_stress^n * exp(-Q / temperature)"
   problem.add_property("small_strain_stress", displacements=["disp_x", "disp_y"], formulation="plane_strain", youngs_modulus=2.0e11, poissons_ratio=0.3, eigenstrain_names=["thermal_strain"], creep_model="parsed", creep_rate=creep_law, creep_constants={"A": 1e-30, "n": 3.0, "Q": 2.0e4}, temperature="T")
""",
)

_add(
    "parsed",
    r"""
``parsed`` declares a scalar property given by an expression of the fields,
the functions, the element fields and the properties added before it.  It
adds a correlation, such as a conductivity :math:`k(T)`, without C++ code,
and Newton's method receives its exact derivative with respect to the
fields:

.. code-block:: python

   problem.add_property("parsed", property_name="thermal_conductivity", expression="k0*(1 + b*(temperature - 300))", constants={"k0": 20.0, "b": 1.0e-3})
   heat = problem.add_physics("heat_transfer")  # reads the property thermal_conductivity
""",
)

_add(
    "parsed_eigenstrain",
    """
``parsed_eigenstrain`` is an eigenstrain, a stress-free strain such as
thermal expansion or swelling, given by an expression, for the
``eigenstrain_names`` of ``small_strain_stress``.  A volumetric swelling that
grows with the temperature, zero at 293.15 K:

.. code-block:: python

   problem.add_property("parsed_eigenstrain", eigenstrain_name="swelling", expression="2.0e-5*(temperature - 293.15)", strain_type="volumetric", formulation="plane_strain")
   problem.add_property("small_strain_stress", displacements=["disp_x", "disp_y"], formulation="plane_strain", youngs_modulus=2.0e11, poissons_ratio=0.3, eigenstrain_names=["swelling"])

With ``stress_free_state={"temperature": 293.15}`` the expression is shifted
so that it vanishes in that state, which avoids repeating the reference
value.
""",
)

_add(
    "finite_strain_stress",
    """
``finite_strain_stress`` is the stress at large strain, for equations written
on the undeformed mesh.  It provides the first Piola-Kirchhoff stress
``first_piola_kirchhoff_stress``, which ``stress_divergence`` reads with
``stress_property``.  Its material laws are those of
``small_strain_stress``, creep included, carried to finite rotation, or a
neo-Hookean solid:

.. code-block:: python

   problem.add_property("finite_strain_stress", displacements=["disp_x", "disp_y"], formulation="plane_strain", youngs_modulus=1.0e6, poissons_ratio=0.45, stress_update="neo-Hookean")
   problem.add_kernel("stress_divergence", displacements=["disp_x", "disp_y"], stress_property="first_piola_kirchhoff_stress")
""",
)

_add(
    "property_scaling",
    """
``property_scaling`` multiplies the properties computed by the property
objects added before it by constant factors, so that a parameter study
scales a correlation without a change to the object that computes it:

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
``symmetry`` models a plane of symmetry of the body and of its loads, so that
only one side of the plane is meshed.  The displacement normal to the plane
is zero and the tangential displacements are free.  The side set must lie in
a plane normal to a coordinate axis, which the condition finds from the
coordinates of its nodes:

.. code-block:: python

   solid = problem.add_physics("solid_mechanics", formulation="plane_stress", youngs_modulus=200.0e9, poissons_ratio=0.3)
   solid.add_boundary_condition("symmetry", boundary="left")
   solid.add_boundary_condition("symmetry", boundary="bottom")
""",
)

_add(
    "heat_transfer",
    r"""
A bus bar with internal heat generation, fixed temperatures on two sides and
convection on the top.  A boundary condition is written as text.  The
temperature on the left side prescribes its value, and ``heat_flux``, the
heat flux :math:`\mathbf{n} \cdot \mathbf{q}` that leaves the body along the
outward unit normal :math:`\mathbf{n}`, prescribes the flux:

.. code-block:: python

   heat = problem.add_physics("heat_transfer", thermal_conductivity=20.0, heat_source=1.0e6)
   heat.add_boundary_condition("temperature = 40", boundary="left")
   heat.add_boundary_condition("temperature = 10", boundary="right")
   heat.add_boundary_condition("heat_flux = 75*temperature", boundary="top")

A heater is a negative ``heat_flux``, heat that enters, and radiation is
``"heat_flux = 5.67e-8*0.8*(temperature^4 - 293^4)"``.  A transient problem
gives ``density`` and ``specific_heat``, and a conductivity that differs
between blocks comes from the property objects when ``thermal_conductivity``
is not given.
""",
)

_add(
    "general_form_PDE",
    r"""
``general_form_PDE`` solves equations written as they are printed.  The
fields are named in ``variables`` and the equations in ``equations``, with
:math:`\partial u / \partial t` written ``du/dt``, :math:`\nabla u` written
``grad(u)``, the divergence :math:`\nabla \cdot` written ``div`` and the
Laplacian :math:`\nabla^2` written ``laplacian``.  A nonlinear diffusion
equation coupled to a Poisson equation for the fields :math:`u` and
:math:`w`, with the constant :math:`k = 2`,

.. math::

   \frac{\partial u}{\partial t} - \nabla \cdot \left[ \left( 1 + |\nabla u|^2 \right) \nabla u \right] = w - u^3 ,
   \qquad
   -\nabla \cdot \left( k \nabla w \right) = 1 ,

reads:

.. code-block:: python

   pde = problem.add_physics(
       "general_form_PDE",
       variables=["u", "w"],
       k=2.0,
       grad_u_squared="dot(grad(u), grad(u))",
       equations=[
           "du/dt - div((1 + grad_u_squared)*grad(u)) = w - u^3",
           "-div(k*grad(w)) = 1",
       ],
   )
   pde.add_boundary_condition("u = 0", boundary=["left", "right"])
   pde.add_boundary_condition("-k*dw/dn = 2*(w - 1)", boundary="top")

Every keyword that is not a parameter is a definition: ``k`` is a named
constant and ``grad_u_squared`` a named expression.  Each equation is matched
to the field it governs.  Everything inside ``div`` is the flux
:math:`\mathbf{F}`, which a boundary condition prescribes as
:math:`\mathbf{n} \cdot \mathbf{F}` along the outward unit normal
:math:`\mathbf{n}`, written out (``-k*dw/dn``) or as ``normal_flux``.

An equation of a scalar field may instead be given by its coefficients, in

.. math::

   m \frac{\partial^2 u}{\partial t^2} + d \frac{\partial u}{\partial t} + \nabla \cdot \left( -c \nabla u - \boldsymbol{\alpha} u + \boldsymbol{\gamma} \right) + \boldsymbol{\beta} \cdot \nabla u + a u = f ,

with :math:`m` the mass coefficient, :math:`d` the damping (or mass)
coefficient, :math:`c` the diffusion coefficient, :math:`\boldsymbol{\alpha}`
the conservative flux convection coefficient, :math:`\boldsymbol{\gamma}` the
conservative flux source, :math:`\boldsymbol{\beta}` the convection
coefficient, :math:`a` the absorption coefficient and :math:`f` the source
term.  These symbols name the parameters of this physics only:

.. code-block:: python

   pde = problem.add_physics("general_form_PDE", variables="u", c="1 + 0.5*u", beta=[1.0, 0.5], a=2.0, f="sin(pi*y)")

With several fields, each coefficient is a dict keyed by the field of the
equation, and a value that is itself a dict couples the equation to other
fields: ``a={"u": {"u": 2.0, "w": -1.0}}`` is the term :math:`2u - w` of the
equation of :math:`u`.
""",
)

_add(
    "parsed_kernel",
    r"""
``parsed_kernel`` adds one term to the equation of ``variable`` in the form
:math:`\nabla \cdot \mathbf{F} = s`, with the flux :math:`\mathbf{F}`
(``flux``) and the source :math:`s` (``source``) given as expressions of the
fields and their gradients.  ``general_form_PDE`` generates it from the
equations.  A nonlinear diffusion term:

.. code-block:: python

   problem.add_kernel("parsed_kernel", variable="u", flux=["-(1 + u^2)*grad_x(u)", "-(1 + u^2)*grad_y(u)"], source="sin(x)")
""",
)

_add(
    "solid_mechanics",
    r"""
A three-dimensional wrench held at its jaws and loaded at its grip by a
force of 150 N (``examples/wrench``):

.. code-block:: python

   solid = problem.add_physics("solid_mechanics", youngs_modulus=200.0e9, poissons_ratio=0.3)
   solid.add_boundary_condition("fixed", boundary="jaws")
   solid.add_boundary_condition("total_force = [0, -150, 0]", boundary="grip")

Its variables are ``disp_x``, ``disp_y`` and ``disp_z``, or ``disp_r`` and
``disp_z`` in axisymmetric coordinates.  A boundary condition is written as
text.  A displacement on the left side prescribes its value
(``"disp_x = 0"``), ``traction`` prescribes the traction
:math:`\boldsymbol{\sigma} \mathbf{n}` per unit area, with the Cauchy stress
:math:`\boldsymbol{\sigma}` and the outward unit normal :math:`\mathbf{n}`
(``"traction = [0, -1e6, 0]"``), ``total_force`` prescribes the force spread
over the boundary, and ``pressure`` prescribes a pressure :math:`p`, the
traction :math:`-p \mathbf{n}`.  The words ``"fixed"`` and ``"symmetry"``
need no data.  A two-dimensional problem gives
``formulation="plane_stress"`` or ``formulation="plane_strain"``.

The structural formulations reduce a thin body to its mid-surface or its
axis.  A steel plate 10 mm thick under a uniform pressure, simply supported,
and a Timoshenko beam, each on a problem of its own:

.. code-block:: python

   plate = plate_problem.add_physics("solid_mechanics", formulation="plate", youngs_modulus=2.0e11, poissons_ratio=0.3, thickness=0.01, transverse_load=1.0e3)
   plate.add_boundary_condition(["disp_x = 0", "disp_y = 0", "disp_z = 0"], boundary=["left", "right", "bottom", "top"])

   beam = beam_problem.add_physics("solid_mechanics", formulation="beam", theory="Timoshenko", youngs_modulus=2.0e11, poissons_ratio=0.3, cross_section={"area": 1.0e-3, "second_moment": 2.0e-6, "shear_correction": 5.0 / 6.0}, transverse_load=-1.0e3)
   beam.add_boundary_condition("fixed", boundary="left")

A plate has the variables ``disp_x``, ``disp_y``, ``disp_z`` (the
deflection), ``phi_x`` and ``phi_y``.  A beam has ``disp_x``, ``disp_z`` and
``bending_moment``, or ``phi`` with ``beam_formulation="displacement"``.  A
circular plate has ``disp_r``, ``disp_z`` and ``phi`` or ``bending_moment``.
The section follows from the material at every point, by integration of its
law through the thickness.  So a graded modulus is an expression of ``z``,
the height above the mid-surface (``youngs_modulus="2e11*(1 + 0.5*z/0.01)"``),
and ``thermal_expansion`` gives the thermal force and moment of the section.
A laminated section is given by its section stiffnesses instead.  The stress
at a height comes from the same law:

.. code-block:: python

   top = plate.stress(z="top")        # sigma_xx, sigma_yy, sigma_xy at the element centres
   bottom = beam.stress(z=-0.05)      # sigma_xx at 5 cm below the axis
""",
)

_add(
    "fluid_dynamics",
    r"""
The lid-driven cavity on the unit square at the Reynolds number :math:`Re = \rho U L / \mu =
1000`, with the density :math:`\rho`, the lid speed :math:`U`, the cavity size
:math:`L` and the dynamic viscosity :math:`\mu`.  The solve ramps the speed of
the lid up from rest when Newton's method needs it:

.. code-block:: python

   flow = problem.add_physics("fluid_dynamics", dynamic_viscosity=1.0, density=1000.0)
   flow.add_boundary_condition(["velocity_x = 0", "velocity_y = 0"], boundary=["left", "right", "bottom"])
   flow.add_boundary_condition(["velocity_x = 1", "velocity_y = 0"], boundary="top")
   problem.solve()
""",
)

_add(
    "thermal_expansion",
    """
The thermal stress of a body heated by conduction:

.. code-block:: python

   heat = problem.add_physics("heat_transfer", thermal_conductivity=50.0)
   solid = problem.add_physics("solid_mechanics", formulation="plane_strain", youngs_modulus=200.0e9, poissons_ratio=0.3)
   problem.add_coupling("thermal_expansion", heat_transfer=heat, solid_mechanics=solid, linear_expansion_coefficient=1.2e-5, stress_free_temperature=293.15)
""",
)

_add(
    "nonisothermal_flow",
    """
Natural convection in a square cavity in the non-dimensional form of
[DeVahlDavis1983]_, at a Rayleigh number of 1e5 and a Prandtl number of
0.71, as in ``examples/natural_convection.py``:

.. code-block:: python

   flow = problem.add_physics("fluid_dynamics", dynamic_viscosity=0.71, density=1.0, formulation="penalty", penalty_parameter=1.0e7)
   heat = problem.add_physics("heat_transfer", thermal_conductivity=1.0)
   problem.add_coupling("nonisothermal_flow", heat_transfer=heat, fluid_dynamics=flow, gravity=[0.0, -1.0], volumetric_expansion_coefficient=1.0e5 * 0.71)
""",
)
