Models of the fuel rod
======================

This chapter gives the equations that :class:`~dualmesh.fuel.FuelRod` solves
and their numerical treatment.  The material correlations are listed in
:doc:`correlations`, and the comparison with measurements is given in
:doc:`benchmarks`.

Background
----------

Three fuel performance codes are the reference points of the design.  BISON
[Williamson2012]_ [Williamson2021]_, which is built on MOOSE, solves the
thermal and mechanical equations of a rod in one fully coupled Newton
iteration, in one, two or three dimensions, and uses the physics-based fission
gas model of Pastore et al. [Pastore2013]_.  TRANSURANUS [Lassmann1992]_ models
a rod as a stack of axial slices, each of which is a one-dimensional radial
problem (a 1.5-dimensional model), which makes it fast enough for routine and
statistical analyses.  OFFBEAT [Scolaro2020]_ is a multidimensional code in
the finite volume framework OpenFOAM, in which the gap is a boundary condition
between the two bodies.  Van Uffelen et al. [VanUffelen2019]_ review these and
other codes.

dualmesh adopts from these codes the three rod models (1.5-dimensional,
axisymmetric and three-dimensional, all using the same materials), the fully
coupled Newton solution with an exact Jacobian, and the treatment of the gap
as a pair of coupled boundaries.  The module is written from the published
equations only, and every coefficient is cited to its source in
:doc:`correlations`.  In addition, the module offers four discretisations of
the same rod (finite elements, the dual mesh control domain method and the
vertex-centred and cell-centred finite volume methods), which provide a
verification between methods within one program, and a Python interface to
the rod, its fields and every correlation.

Heat conduction
---------------

In the fuel and in the cladding, the temperature :math:`T` obeys

.. math::
   :label: fuel_energy

   \rho c_p \frac{\partial T}{\partial t} = \nabla \cdot \left( k \nabla T \right) + q''',

where :math:`\rho`, :math:`c_p` and :math:`k` are the density, the specific
heat and the thermal conductivity of the material, and :math:`q'''` is the
fission heat generated per unit volume in the fuel.  The materials
``UO2_thermal``, ``UN_thermal`` and ``Zircaloy_thermal`` provide :math:`k`,
:math:`c_p` and :math:`\rho` as material properties, which are used by the
kernels ``heat_conduction`` and ``heat_conduction_time_derivative``.

The input of the calculation is the linear heat rate :math:`q'(z, t)` (W/m)
of the rod.  The heat generated per unit volume is :math:`q''' = q' f(r) / (\pi
a^2)`, where :math:`a` is the pellet radius and :math:`f(r)` is the radial
profile, normalised to an area average of one (flat unless given).  The fission
rate density is :math:`\dot F = q''' / E_f`, with the energy per fission
:math:`E_f = 200` MeV.  The local burnup, i.e., the fraction of the heavy-metal
atoms that have fissioned (FIMA), is :math:`\beta = \int \dot F \, dt /
n_{HM}`, where :math:`n_{HM}` is the number of heavy-metal atoms per unit
volume.  With these values, a burnup of 1 % FIMA corresponds to 9.383 GWd/tU.
The fast neutron flux is proportional to the linear heat rate, with a factor
set by the user for the reactor.  The function
:func:`dualmesh.fuel.irradiation_fields` tabulates these fields on a grid of
radius, height and time, integrates the piecewise-linear power history exactly,
and passes them to the solver as ``CylinderTableFunction`` objects.

The gap
-------

The pellet surface and the inner surface of the cladding are two separate
boundaries of the mesh.  The boundary condition ``gas_gap_heat_transfer``
pairs every integration point of the pellet surface (the *primary* side) with
the closest point of the cladding surface (the *secondary* side), and applies
the heat flux

.. math::
   :label: gap_flux

   q = h_{gap} \left( T_s - T_p \right),
   \qquad
   h_{gap} = \frac{k_{gas}}{g + C_r (R_p + R_s) + j}
   + \sigma F_\epsilon \left( T_p^2 + T_s^2 \right) \left( T_p + T_s \right)
   + h_{solid} ,

where :math:`T_p` and :math:`T_s` are the temperatures of the primary and the
secondary surface.  The first term is the gas conduction across the gap of
width :math:`g`, measured along the normal of the pellet between the displaced
surfaces, with the roughnesses :math:`R_p` and :math:`R_s`, the roughness
coefficient :math:`C_r = 1.5` and the temperature jump distance :math:`j`.  The
second term is the radiation between two grey surfaces, with :math:`F_\epsilon
= 1 / (1/\epsilon_p + 1/\epsilon_s - 1)` and the Stefan-Boltzmann constant
:math:`\sigma`.  The third term is the solid contact conductance, which acts
once the surfaces are in contact, :math:`h_{solid} = C_s k_m P_c /
(\sqrt{\delta} H)`, where :math:`k_m` is the harmonic mean of the
conductivities, :math:`P_c` the contact pressure, :math:`H` the Meyer hardness
of the cladding, :math:`C_s = 10\ \mathrm{m^{-1/2}}` and :math:`\delta = 0.8
(R_p + R_s)`.  This is the form of Ross and Stoute [RossStoute1962]_.  The gas
conductivity is that of the helium fill mixed with the released xenon and
krypton, by the rule of Lindsay and Bromley [LindsayBromley1950]_ in the form
of Brokaw [Brokaw1958]_.  The jump distance is the result of the kinetic
theory of Kennard as used by Lanning and Hann [LanningHann1975]_,

.. math::

   j = 0.013757\, \frac{2 - \alpha}{\alpha}\, \frac{k_{gas} \sqrt{T}}{P}
   \left(\sum_i \frac{x_i}{M_i}\right)^{-1/2},

in SI units, where :math:`\alpha` is the accommodation coefficient,
:math:`P` the gas pressure, :math:`x_i` the mole fraction and :math:`M_i` the
molar mass (g/mol) of gas :math:`i`.

The flux :math:`q` of :eq:`gap_flux` is the heat entering the pellet per unit
area of the pellet surface.  The same heat, with the opposite sign and
integrated with the same weights, enters the cladding at the paired point.  The
heat that leaves the pellet therefore enters the cladding exactly, for any
meshes on the two sides.  The primary side is integrated with the rule of the
discretisation.  The secondary side receives its share through its shape
functions at the paired point (finite elements), through the control domain
that contains the paired point (dual mesh and vertex-centred finite volumes),
or through the boundary face that contains it (cell-centred finite volumes).
The paired points are found once, by a Gauss-Newton projection onto the
candidate faces of the cladding.  The derivatives of the flux with respect to
the unknowns on both sides are evaluated by automatic differentiation, so that
the Jacobian of the coupled rod is exact.  The same mechanism, the
``InterfaceBC`` base class, is used by ``gap_heat_transfer`` (a prescribed
conductance) and by the mechanical ``gap_contact``.  The finer of the two
surface meshes should be the primary side, since a secondary face onto which
no primary point is projected receives no heat.

The coolant
-----------

The outer surface of the cladding transfers heat to the coolant with a heat
transfer coefficient :math:`h`, i.e., :math:`q = h (T - T_{cool})`.  With
:class:`~dualmesh.fuel.ForcedConvection`, the coolant temperature follows from
the enthalpy balance of the channel,

.. math::

   \dot m \left[ h_{cool}(z) - h_{cool}(0) \right] = \int_0^z q'(z')\, dz',

where :math:`\dot m = G A` is the mass flow rate, with the mass flux :math:`G`
and the flow area :math:`A` of a square-lattice pin cell, and the water
properties are those of IAPWS-IF97.  The heat transfer coefficient is given by
the correlation of Dittus and Boelter [DittusBoelter1930]_ or of Weisman, on
the hydraulic diameter of the pin cell, with the local properties of the
coolant.  With :class:`~dualmesh.fuel.PrescribedCladdingTemperature`, the
outer temperature of the cladding is prescribed as a function of the linear
heat rate and the time, as is usual for test reactors such as Halden.  The
coolant can also be computed with the flow equations of
:doc:`../theory/heat_and_fluids`, sharing its nodes with the cladding, as in
``examples/fuel_rod_conjugate.py``.

Mechanics
---------

The fuel and the cladding are in quasi-static equilibrium,
:math:`\nabla \cdot \boldsymbol{\sigma} = 0`.  Each is isotropic and linear
elastic, with temperature-dependent elastic constants, and the stress follows
from the strain that remains after the stress-free strains (*eigenstrains*)
are removed:

.. math::
   :label: fuel_stress

   \boldsymbol{\sigma} = \lambda \operatorname{tr}(\boldsymbol{\varepsilon}_e) \mathbf{I}
   + 2 \mu \boldsymbol{\varepsilon}_e ,
   \qquad
   \boldsymbol{\varepsilon}_e = \boldsymbol{\varepsilon}
   - \boldsymbol{\varepsilon}_{th} - \boldsymbol{\varepsilon}_{sw}
   - \boldsymbol{\varepsilon}_{rel} - \boldsymbol{\varepsilon}_{g}
   - \boldsymbol{\varepsilon}_{cr} ,

where :math:`\lambda` and :math:`\mu` are the Lamé constants,
:math:`\boldsymbol{\varepsilon}` is the total strain, and the eigenstrains are
the thermal expansion :math:`\boldsymbol{\varepsilon}_{th}`, the densification
and swelling of the fuel :math:`\boldsymbol{\varepsilon}_{sw}`, the relocation
of the cracked fuel fragments towards the cladding
:math:`\boldsymbol{\varepsilon}_{rel}`, the irradiation growth of the cladding
:math:`\boldsymbol{\varepsilon}_{g}`, and the creep strain
:math:`\boldsymbol{\varepsilon}_{cr}`.  The stress is computed by
``small_strain_stress`` or, for large deformations, by
``finite_strain_stress``, and the eigenstrains by materials such as
``UO2_thermal_expansion_eigenstrain``, ``UO2_volumetric_swelling_eigenstrain``,
``UO2_relocation_eigenstrain`` and ``Zircaloy_irradiation_growth_eigenstrain``.

**Creep.**  The creep rate depends on the stress and the temperature, and the
creep strain accumulates over time.  The creep strain is integrated by the
backward Euler method with a radial return on the von Mises stress.  With the
trial stress :math:`\boldsymbol{\sigma}^{tr}`, computed from the creep strain
at the start of the step, its deviator :math:`\mathbf{s}` and its von Mises
value :math:`q`, the increment of the equivalent creep strain :math:`\Delta p`
solves

.. math::

   \Delta p = \Delta t\, \dot\varepsilon_{cr}\!\left(q - 3 G \Delta p,\ T\right),

where :math:`G` is the shear modulus, and the creep strain increases by
:math:`\frac{3}{2} \Delta p\, \mathbf{s} / q`.  This scalar equation is solved
by Newton's method, safeguarded by bisection on :math:`[0, q / 3G]`.  The
derivatives of :math:`\Delta p` with respect to the unknowns follow from the
implicit function theorem, so that the Jacobian remains exact and the global
Newton iteration converges quadratically.  The creep strain is stored at every
integration point of every method.  It is committed when a time step is
accepted and discarded when a step is rejected.

**Contact.**  The contact between the pellet and the cladding is frictionless
and is enforced by a penalty (``gap_contact``).  Where the gap :math:`g =
(\mathbf{x}_s + \mathbf{u}_s - \mathbf{x}_p - \mathbf{u}_p) \cdot \mathbf{n}_p`
becomes negative, the contact pressure :math:`P_c = k_p (-g)` pushes the two
surfaces apart with equal and opposite forces, where :math:`\mathbf{x}` and
:math:`\mathbf{u}` are the positions and displacements of the paired points
and :math:`k_p` is the penalty.  The default penalty of :math:`10^{15}` Pa/m
gives a penetration below 1 :math:`\mu\mathrm{m}` at the contact pressures of
pellet-cladding interaction.

**Pressures.**  The gas in the rod acts on the pellet surface, on the top of
the fuel stack and on the inner surface of the cladding.  The coolant pressure
acts on the outer surface of the cladding.  The difference of the two
pressures, carried by the end plugs, loads the cladding axially.  The gas
pressure follows the ideal gas law,

.. math::

   p = \frac{n R}{\sum_i V_i / T_i},

where :math:`n` is the amount of gas (the fill gas plus the released fission
gas), :math:`R` is the gas constant, and the sum runs over the free volumes
:math:`V_i` of the rod at their temperatures :math:`T_i`:

* The gap, with its current width, at the mean of the temperatures of the
  pellet surface and the inner cladding surface.
* The bore of annular pellets, at the centre temperature.
* The cracks opened by relocation (twice the relocation strain times the
  pellet volume), at the mean of the centre and surface temperatures of the
  pellet.
* The plenum.

The pressure is updated after every time step.

**Time steps.**  The time steps end at the output times and at the points of
the power history needed to follow the history within 1 % of its largest
linear heat rate (``RodNumerics.power_history_tolerance``).  The shutdowns and
power steps are therefore resolved regardless of the length of the other
steps.  The largest step is set by ``RodNumerics.max_time_step``.

**The 1.5-dimensional model.**  Each axial slice is a radial line through the
fuel and the cladding in *generalized plane strain*.  The axial strain is
uniform in each body, with one value for the fuel and one for the cladding.
The two values are found at every time step from the axial force balance, in
which the fuel stack carries the gas pressure on its top and the cladding
carries the load of the end plugs, by a secant iteration around the solution
of the slice.

Fission gas
-----------

Fission produces the noble gases xenon and krypton, about 0.32 atoms per
fission, which are almost insoluble in the fuel.  In UO2 the gas atoms diffuse
through each grain to its boundary, collect there in bubbles on the grain
faces, and are released to the free volume of the rod once the bubbles
interconnect.  The released gas lowers the conductance of the gap and raises
the pressure in the rod.  :class:`~dualmesh.fuel.BoothFissionGasRelease`
follows this sequence at every fuel element.

**Intragranular diffusion.**  The gas concentration :math:`C` in an
equivalent spherical grain of radius :math:`a` obeys the equation of Booth,

.. math::

   \frac{\partial C}{\partial t} = D \nabla^2 C + \beta, \qquad C(a) = 0,

where :math:`D` is the effective diffusion coefficient and :math:`\beta` the
rate of gas production.  With :math:`D` and :math:`\beta` constant over a time
step, the equation is solved exactly by expanding :math:`u = r C` in the
eigenfunctions :math:`\sin(n \pi r / a)`.  Each coefficient obeys :math:`\dot
A_n = -\lambda_n A_n + \beta b_n`, with :math:`\lambda_n = D n^2 \pi^2 / a^2`
and :math:`b_n = 2 a (-1)^{n+1} / (n \pi)`, whose solution over a step is
exact.  This is the spectral form of the method of Forsberg and Massih
[ForsbergMassih1985]_, with 200 terms of the series.

**Diffusion coefficient, trapping and re-solution.**  The single-atom
diffusion coefficient has a thermal, an irradiation-enhanced and an athermal
term (Turnbull et al. [Turnbull1982]_).  Gas atoms are trapped in
intragranular bubbles and are returned to the lattice by fission fragments
(re-solution), and the total gas then diffuses with the effective coefficient
:math:`D_{eff} = b D / (b + g)` of Speight [Speight1969]_, where :math:`b` is
the re-solution rate and :math:`g` the trapping rate.  The bubbles nucleate in
the wake of fission fragments and are destroyed by re-solution.

**Grain-face bubbles.**  The gas that reaches the grain boundaries collects in
lenticular bubbles on the grain faces.  The bubbles grow by absorbing
vacancies, coalesce, and vent their gas once they cover half of the face
(Pastore et al. [Pastore2013]_ [Pastore2015]_).  On temperature changes, part
of the faces crack and release their gas in a burst, and the cracks heal with
burnup (Barani et al. 2017).  The volume of the intragranular and grain-face
bubbles is the gaseous swelling of the fuel.

The fission gas model is advanced between time steps, from the temperature and
the fission rate of each element.  The released gas enters the rod through the
gas pressure and the composition of the gas in the gap, and the gaseous
swelling enters the mechanics as an eigenstrain.  This explicit coupling
follows FRAPCON and TRANSURANUS, and it is accurate when the time steps resolve
the changes of temperature.  For UN, :class:`~dualmesh.fuel.StormsFissionGasRelease`
uses the empirical release fraction of Storms [Storms1988]_.

Correlations at low temperature
-------------------------------

The correlations are fits to data above room temperature, and several of them
are singular at absolute zero: the polaron term of Fink contains
:math:`t^{-5/2} \exp(-16.35/t)`, a factor of Lucuta et al. contains a square
root of the temperature, and the gas conductivity :math:`A T^B` with :math:`B
< 1` has an infinite derivative at :math:`T = 0`.  Newton's method can visit
such temperatures when it starts from a cold initial guess, e.g., a
temperature left at its default initial value of zero.  The materials
therefore evaluate the correlations at :math:`\max(T, 200\ \mathrm{K})`, with a
zero derivative below this value.  A converged solution of a reactor rod lies
far above 200 K, so that this lower bound affects only the first iterations.

Model factors
-------------

The input group :class:`~dualmesh.fuel.ModelFactors` multiplies the models of
the rod by factors, all equal to one by default: the linear heat rate, the
conductivities of the fuel and the cladding, the gas conduction and solid
contact terms of the gap conductance, the heat transfer to the coolant, the
thermal expansion, relocation, densification and swelling of the fuel, the
creep rates of the fuel and the cladding, and, in the fission gas model only,
the temperature, the grain radius, the intragranular diffusion coefficient,
the re-solution rate and the vacancy diffusivity on the grain boundaries.  The
factors are intended for sensitivity and uncertainty studies.  A factor that
cannot act on a given rod, e.g., a densification factor with the densification
model switched off, is an error.  The material factors are applied by the
material ``property_scaling``, which multiplies properties computed by the
materials added before it.

Verification
------------

``tests/python/test_fuel.py`` and ``tests/python/test_fuel_factors.py`` check
every part of the model against an exact answer, with every method where the
part depends on the method:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Test
     - Result
   * - Correlations
     - Every correlation reproduces the values tabulated in, or computed
       from, its source (e.g., the conductivity of Fink of 3.47 W/(m K) at
       1000 K and the factors of Lucuta et al. of 0.8345, 1.0023 and 0.9555
       at 3 at. %).
   * - A heated rod in a tube across a gap (radial slice, all four methods)
     - Second-order convergence to the exact temperatures.  The power that
       leaves the cladding equals the fission power to round-off.
   * - The same in (r, z) and in three dimensions
     - Energy balance to round-off in (r, z), and in three dimensions against
       the meshed volume, with the temperature error decreasing as the
       section is refined.
   * - Thermal stress of a long cylinder (all four methods)
     - The stresses of Timoshenko and Goodier [TimoshenkoGoodier1970]_,
       converging at order 1.5 or better.
   * - Thick tube under pressure (all four methods)
     - The hoop stress of Lamé within 0.2 %.
   * - Creep under constant stress (UO2, UN and Zircaloy, three methods)
     - The exact strain :math:`\sigma / E + \dot\varepsilon t` to
       :math:`10^{-7}`, with quadratic convergence of Newton's method.
   * - Booth diffusion
     - The release fraction of Booth to :math:`10^{-5}` over time steps of
       any size.
   * - Rod models
     - The (r, z) and 1.5-dimensional rods, with every model active, agree
       within 10 K and 2.5 :math:`\mu\mathrm{m}` of gap width after 100
       days.  The three-dimensional and (r, z) rods agree within the
       geometric error of the three-dimensional section.  The four methods
       agree on a rod in pellet-cladding contact.
   * - Rod with a flowing coolant
     - The heat removed by the coolant equals the fission power within 1 %
       for three methods, which agree with each other.
   * - Model factors
     - A conductivity factor of 1.1 changes the conductivity integral
       :math:`\int_{T_s}^{T_c} k\, dT` by the factor 1.1 within 0.5 %, a
       power factor of 1.05 scales the burnup by 1.05 exactly, and a factor
       of 2 on the heat transfer coefficient of the coolant halves the
       temperature rise across the film.

The validation against measured rods, and the comparison with the published
results of BISON, are given in :doc:`benchmarks`.

Limitations
-----------

The module does not model the friction between the pellet and the cladding
(the contact is frictionless, and the slices of the 1.5-dimensional model are
independent in the axial direction), the cracking of the fuel other than
through relocation, the primary creep and the plasticity of the cladding,
waterside corrosion and hydriding, the radial power profile of the plutonium
build-up at the pellet rim (the profile is flat or given), and the high burnup
structure.  The models are those of normal operation.  Interface conditions
are not available in distributed (MPI) calculations, and the cell-centred
method cannot hold a no-slip condition on a fluid-solid interface inside the
mesh.
