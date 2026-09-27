Nuclear fuel performance
========================

A nuclear fuel rod is a stack of ceramic pellets inside a metal tube.  The
fission of uranium in the pellets releases heat, which crosses the pellets,
a thin gas-filled gap and the tube (the *cladding*) and is carried away by the
coolant.  Over months and years of irradiation the rod changes: the pellets
crack, densify and then swell, release part of the fission gas they produce,
and eventually press on the cladding, which creeps down onto them under the
coolant pressure.  A *fuel performance code* follows these coupled thermal,
mechanical and chemical changes through a power history, to show that the
fuel stays below its melting point, that the cladding is not overstrained and
that the pressure in the rod stays acceptable.

The ``fuel_performance`` module and the Python package :mod:`dualmesh.fuel`
add this capability to dualmesh.  This chapter describes what they compute,
where every correlation comes from, how the rod is solved, and how the result
is verified.

Where this module comes from
----------------------------

Three fuel performance codes shaped the design:

* **BISON** [Williamson2012]_ [Williamson2021]_, built on MOOSE, solves the
  thermal and mechanical equations of a rod in one fully coupled Newton
  iteration, in one, two or three dimensions, with a large library of
  material models; its fission gas model is the physics-based one of Pastore
  et al. [Pastore2013]_.
* **TRANSURANUS** [Lassmann1992]_ models a rod as a stack of axial slices,
  each a one-dimensional radial problem (a "1.5-dimensional" model), which
  makes it fast and robust enough for routine and statistical analyses.
* **OFFBEAT** [Scolaro2020]_ is a multi-dimensional code in the finite
  volume framework OpenFOAM, with cell-centred unknowns and the gap treated as
  a boundary condition between the two bodies.

The review of Van Uffelen et al. [VanUffelen2019]_ compares these and other
codes.  dualmesh takes from them the three rod models (1.5-dimensional,
axisymmetric and three-dimensional, all driven by the same materials), the
fully coupled Newton solution with an exact Jacobian, and the treatment of the
gap as a paired boundary condition.  None of their source code was consulted:
BISON's source is distributed under controlled access, TRANSURANUS under a
licence agreement, and OFFBEAT under the GNU General Public License version 3,
whose code cannot be copied into a library released under the Lesser General
Public License 2.1.  Everything here is written from the published equations,
and every coefficient is cited to its source in the tables below.

What the module adds that none of the three has:

* **Four discretisations of the same rod.**  The finite element, dual mesh,
  vertex-centred and cell-centred finite volume methods solve the same fuel
  rod from the same problem description, which gives a code-to-code
  verification within one program.  BISON is a finite element code and
  OFFBEAT a finite volume code.
* **Three rod models from one set of materials**, which the tests compare
  with each other.
* **Verification against exact solutions of every building block**, in the
  test suite, for every method (see `Verification`_).
* **A Python interface** to the rod, its fields and every correlation, and
  the exact Jacobian of the automatic differentiation, which also gives the
  sensitivity of any result to a parameter.

The equations
-------------

Heat
^^^^

In the fuel and the cladding the temperature :math:`T` obeys

.. math::
   :label: fuel_energy

   \rho c_p \frac{\partial T}{\partial t} = \nabla \cdot \left( k \nabla T \right) + q''' ,

with the density :math:`\rho`, specific heat :math:`c_p` and conductivity
:math:`k` of the material, and the fission heat :math:`q'''` in the fuel.  The
materials ``UO2_thermal``, ``UN_thermal`` and ``Zircaloy_thermal`` supply
:math:`k`, :math:`c_p` and :math:`\rho` as material properties, which the
kernels ``heat_conduction`` and ``heat_conduction_time_derivative`` use.

The linear power :math:`q'(z, t)` (W/m) of the rod is the input.  The heat
generated per unit volume is :math:`q''' = q' f(r) / (\pi a^2)`, with the
pellet radius :math:`a` and the radial profile :math:`f(r)`, normalised to an
area average of one (flat unless given).  The fission rate density is
:math:`\dot F = q''' / E_f` with :math:`E_f = 200` MeV, and the local burnup,
the fraction of the heavy-metal atoms that have fissioned (FIMA), is
:math:`\beta = \int \dot F \, dt / n_{HM}`, where :math:`n_{HM}` is the number
of heavy-metal atoms per unit volume.  With these values one atom per cent of
burnup is 9.383 GWd/tU.  The fast neutron flux is taken proportional to the
linear power, with a factor that the user sets for the reactor.
:func:`dualmesh.fuel.rod_fields` tabulates these fields on a grid of radius,
height and time, integrating the piecewise-linear power history exactly, and
hands them to the solver as ``CylinderTableFunction`` objects.

The gap
^^^^^^^

The pellet surface and the inner cladding surface are two separate
boundaries of the mesh.  ``FuelGapHeatTransfer`` pairs every integration
point of the pellet surface (the *primary* side) with the closest point of the
cladding surface (the *secondary* side) and applies the heat flux

.. math::
   :label: gap_flux

   q = h_{gap} \left( T_s - T_p \right),
   \qquad
   h_{gap} = \frac{k_{gas}}{g + C_r (R_p + R_s) + j}
   + \sigma F_\epsilon \left( T_p^2 + T_s^2 \right) \left( T_p + T_s \right)
   + h_{solid} ,

which is the form of Ross and Stoute [RossStoute1962]_ used by the fuel codes:
gas conduction across the width :math:`g` of the gap, measured along the
pellet normal between the displaced surfaces, plus the roughnesses
:math:`R` (with :math:`C_r = 1.5`) and the temperature jump distance
:math:`j`; radiation between two grey surfaces with
:math:`F_\epsilon = 1 / (1/\epsilon_p + 1/\epsilon_s - 1)`; and, once the
surfaces are in contact, the solid contact conductance
:math:`h_{solid} = C_s k_m P_c / (\sqrt{\delta} H)`, with the harmonic mean
conductivity :math:`k_m`, the contact pressure :math:`P_c`, the Meyer hardness
:math:`H = 680` MPa, :math:`C_s = 10\ \mathrm{m^{-1/2}}` and :math:`\delta =
0.8 (R_p + R_s)`.  The gas conductivity is that of the helium fill mixed with
the released xenon and krypton, by the rule of Lindsay and Bromley
[LindsayBromley1950]_ in the form of Brokaw [Brokaw1958]_, with the pure-gas
fits :math:`k = A T^B` of MATPRO.  The jump distance is Kennard's kinetic
theory result as used by Lanning and Hann [LanningHann1975]_,
:math:`j = 0.013757\, \frac{2 - \alpha}{\alpha}\, k_{gas} \sqrt{T} / P \,
(\sum_i x_i / M_i)^{-1/2}` in SI units (with the molar masses in g/mol), with
the accommodation coefficient :math:`\alpha` of MATPRO.

**How the pairing conserves energy.**  The flux :math:`q` of
:eq:`gap_flux` is the heat entering the pellet per unit *pellet* area.  The
same heat, with the opposite sign and integrated with the same weights, enters
the cladding at the paired point.  Whatever leaves the pellet therefore enters
the cladding exactly, whatever the meshes on the two sides.  The primary side
is integrated with the rule of the discretisation; the secondary side receives
its share through its shape functions at the paired point (finite elements),
through the control domain that contains the paired point (dual mesh and
vertex-centred finite volumes), or through the boundary face that contains it
(cell-centred finite volumes).  The paired points are found once, by a
Gauss-Newton projection onto the candidate cladding faces.  The derivatives
of the flux with respect to the unknowns on both sides are carried by the
automatic differentiation, so the Jacobian of the coupled rod stays exact.
The same mechanism, the ``InterfaceBC`` base class, carries the simpler
``gap_heat_transfer`` (a given conductance) and the mechanical ``gap_contact``.
The finer of the two surface meshes should be the primary side, because a
secondary face that no primary point maps into receives nothing.

The coolant
^^^^^^^^^^^

The cladding surface loses heat to the coolant through a heat transfer
coefficient, :math:`q = h (T - T_{cool})`.  :class:`dualmesh.fuel.Coolant`
computes :math:`h` from the Dittus-Boelter correlation
:math:`Nu = 0.023 Re^{0.8} Pr^{0.4}` [DittusBoelter1930]_ on the hydraulic
diameter of a square-lattice pin cell, and the coolant temperature from the
energy balance of the channel,
:math:`T_{cool}(z) = T_{in} + \int_0^z q' \, dz' / (G A c_p)`, with the mass
flux :math:`G` and the flow area :math:`A`.  The coolant properties are held
constant.  The coolant can instead be solved with the flow equations
(:doc:`heat_and_fluids`), sharing its nodes with the cladding, which is
what ``examples/fuel_rod_conjugate.py`` does.

Mechanics
^^^^^^^^^

The fuel and the cladding are in quasi-static equilibrium,
:math:`\nabla \cdot \boldsymbol{\sigma} = 0`, and each is isotropic and
linear elastic with temperature-dependent constants, from the strain that is
left after the stress-free strains (*eigenstrains*) are removed:

.. math::
   :label: fuel_stress

   \boldsymbol{\sigma} = \lambda \operatorname{tr}(\boldsymbol{\varepsilon}_e) \mathbf{I}
   + 2 \mu \boldsymbol{\varepsilon}_e ,
   \qquad
   \boldsymbol{\varepsilon}_e = \boldsymbol{\varepsilon}
   - \boldsymbol{\varepsilon}_{th} - \boldsymbol{\varepsilon}_{sw}
   - \boldsymbol{\varepsilon}_{rel} - \boldsymbol{\varepsilon}_{g}
   - \boldsymbol{\varepsilon}_{cr} .

``EigenstrainElasticStress`` implements :eq:`fuel_stress`.  The eigenstrains
are the thermal expansion (``FuelThermalExpansionEigenstrain``), the
densification and swelling of the fuel (``UO2VolumetricEigenstrain``,
``UNSwellingEigenstrain``), the relocation of the cracked fuel fragments
towards the cladding (``FuelRelocationEigenstrain``), the irradiation growth
of the cladding (``ZircaloyGrowthEigenstrain``), and the creep strain.

**Creep** is a history: its rate depends on the stress and the temperature,
and the strain accumulates.  It is integrated by the backward Euler method
with a radial return on the von Mises stress.  With the trial stress
:math:`\boldsymbol{\sigma}^{tr}` computed from the creep strain at the start of
the step, its deviator :math:`\mathbf{s}` and von Mises value :math:`q`, the
equivalent creep strain increment :math:`\Delta p` solves

.. math::

   \Delta p = \Delta t\, \dot\varepsilon_{cr}\!\left(q - 3 G \Delta p,\ T\right),

and the creep strain grows by :math:`\frac{3}{2} \Delta p\, \mathbf{s} / q`.
The scalar equation is solved by Newton's method safeguarded by bisection on
:math:`[0, q / 3G]`, and the derivatives of :math:`\Delta p` with respect to
the unknowns follow from the implicit function theorem, so that the Jacobian
remains exact and the global Newton iteration quadratic.  The creep strain is
kept at every integration point of every method, committed when a time step
is accepted and discarded when it is rejected (the history machinery of
:class:`dualmesh.Problem` that any material can use through ``stateSize``).

**Contact** between the pellet and the cladding is frictionless and enforced
by a penalty (``gap_contact``): where the gap
:math:`g = (\mathbf{x}_s + \mathbf{u}_s - \mathbf{x}_p - \mathbf{u}_p) \cdot
\mathbf{n}_p` becomes negative, the contact pressure :math:`P_c = k_p (-g)`
pushes the two surfaces apart, with equal and opposite forces.  The default
penalty :math:`10^{15}` Pa/m leaves a penetration of about ten nanometres at
the pressures of pellet-cladding interaction.

**Pressures.**  The gas in the rod presses on the pellet surface, the top of
the fuel stack and the inside of the cladding; the coolant presses on the
outside; and the difference of the two, carried by the end plugs, pulls on
the cladding along its axis.  The gas pressure follows the ideal gas law
:math:`p = n R / \sum_i V_i / T_i` over the free volumes of the rod (the gap,
at the mean of the pellet surface and cladding temperatures and with its
current width, the bore of annular pellets at the centre temperature, the
cracks that relocation opens between the fuel fragments, twice the
relocation strain times the pellet volume, at the mean of the centre and
surface temperatures, and the plenum), with the amount of gas :math:`n` of
the fill plus the released fission gas.  It is updated after every time
step.

**Time steps.**  The steps land on the output times and on the points of
the power history needed to follow it to within 1 % of its largest linear
heat rate (``RodNumerics.power_history_tolerance``), so that shutdowns and
power steps are resolved however long the other steps are.

**The 1.5-dimensional model.**  Each axial slice is a radial line through the
fuel and the cladding, in *generalized plane strain*: the axial strain is
uniform across each body, one value for the fuel and one for the cladding,
and is found at every time step from the axial force balance (the fuel stack
carries the gas pressure on its top, the cladding the end-plug load), by a
secant iteration around the solve of the slice.

Fission gas
^^^^^^^^^^^

In UO2 the gas atoms diffuse through each grain to its boundary, gather there
in bubbles, and are released once the bubbles on the grain faces
interconnect.  :class:`dualmesh.fuel.gas.BoothGasModel` models this in every
fuel element.  The diffusion in an equivalent sphere of radius :math:`a`
(Booth) is solved exactly over each time step, by expanding :math:`r C` in the
eigenfunctions :math:`\sin(n \pi r / a)`: every coefficient decays or grows
exponentially with its eigenvalue :math:`D n^2 \pi^2 / a^2`.  This is the
spectral form of the method of Forsberg and Massih [ForsbergMassih1985]_,
with a few hundred terms in place of their four-term fit.  The diffusion
coefficient is that of Turnbull et al. [Turnbull1982]_ in the form used by
Pastore et al. [Pastore2015]_, optionally reduced by a trapping factor
:math:`b / (b + g)` for the trapping of gas in intragranular bubbles and its
re-solution (Speight [Speight1969]_).  The grain boundaries hold gas up to the
saturation density of the FRAPCON-4.0 code description, and release the
excess.  For UN, :class:`dualmesh.fuel.gas.StormsGasModel` uses the empirical
release fraction of Storms [Storms1988]_.

The gas model runs between time steps, from the temperature and fission rate
of each element; its release enters the rod through the gas pressure and the
composition of the gap gas.  This explicit coupling is the one of FRAPCON and
TRANSURANUS, and is accurate when the time step resolves the changes of
temperature.

The materials
-------------

The correlations are implemented in ``include/dualmesh/modules/FuelProperties.h``
and are callable from Python through :mod:`dualmesh.fuel.properties`.  The
column "checked" says how the coefficients were verified: "original" means
against the original paper (its abstract where only that could be read);
"secondary" means against public secondary sources that agree with each
other (the FRAPCON-4.0 material property report PNNL-19417, IAEA-TECDOC-1496,
the public BISON documentation), the original being paywalled or
unpublished.

The correlations are fits to data above room temperature, and several of
them are singular at absolute zero: Fink's polaron term contains
:math:`t^{-5/2} \exp(-16.35/t)`, Lucuta's factor a square root of the
temperature, and the gas conductivity :math:`A T^B` with :math:`B < 1` has an
infinite derivative at :math:`T = 0`.  Newton's method can visit such
temperatures when it starts from a cold guess, for example a temperature left
at its default initial value of zero.  The materials therefore evaluate the
correlations at :math:`\max(T, 200\ \mathrm{K})`, with zero derivative below
the floor.  A converged reactor solution is far above it, so the floor only
makes the first iterations well defined.

.. list-table::
   :header-rows: 1
   :widths: 24 44 32

   * - Property
     - Correlation
     - Source; checked
   * - UO2 conductivity
     - Fink (95 % TD, with Fink's porosity correction); Fink times the four
       factors of Lucuta et al. (burnup, porosity, radiation damage); modified
       NFI (burnup, gadolinia); Halden (burnup)
     - [Fink2000]_, [Lucuta1996]_, FRAPCON-4.0; secondary (values reproduced)
   * - UO2 specific heat
     - Fink; MATPRO with Kerrisk and Clifton's constants
     - [Fink2000]_, [KerriskClifton1972]_; secondary
   * - UO2 thermal expansion
     - Martin's length polynomials
     - [Martin1988]_; secondary (coefficient checked numerically)
   * - UO2 densification, swelling
     - MATPRO FUDENS densification (ESCORE as an option); MATPRO solid
       swelling 2.5e-29 per fission per m^3; gaseous swelling from the
       bubbles of the fission gas model (MATPRO FSWELL as an option)
     - NUREG/CR-6150; [Pastore2013]_; see :doc:`correlations`
   * - UO2 relocation
     - ESCORE, applied in the plane normal to the axis
     - BISON theory manual; secondary
   * - UO2 creep
     - MATPRO FCREEP (diffusional, dislocation and irradiation creep)
     - MATPRO via BISON; secondary
   * - UO2 elastic constants
     - MATPRO FELMOD, :math:`E = 2.334 \times 10^{11} (1 - 1.0915 \times
       10^{-4} T)(1 - 2.752 P)` Pa with :math:`P` the porosity; FPOIR
       :math:`\nu = 0.316`
     - MATPRO via the public report CASL-U-2019-1870-000 (Gamble et al.,
       2019); secondary
   * - UN conductivity, heat capacity, density, expansion, elasticity
     - Hayes, Thomas and Peddicord (Einstein temperature 365.7 K)
     - [HayesI1990]_ [HayesII1990]_ [HayesIII1990]_ [HayesIV1990]_; original
   * - UN creep
     - Hayes thermal creep with its porosity factor; irradiation creep 2e-22
       :math:`\sigma \dot F` per hour
     - [HayesII1990]_; porosity factor and irradiation term secondary
   * - UN swelling
     - Ross, El-Genk and Matthews
     - [Ross1990]_; original
   * - UN gas release
     - Storms
     - [Storms1988]_; original
   * - Zircaloy-4 conductivity, heat capacity, density
     - MATPRO; IAEA-TECDOC-1496
     - secondary
   * - Zircaloy-4 expansion
     - FRAPCON-4.0 axial and diametral strains (alpha phase)
     - secondary
   * - Zircaloy-4 elastic constants
     - MATPRO CELMOD and CSHEAR with the fluence factor
     - secondary
   * - Zircaloy-4 creep
     - Limbäck and Andersson thermal creep with Hoppe's irradiation creep
       (secondary creep only), constants of the BISON documentation
     - [LimbackAndersson1996]_; secondary, and the FRAPCON-4.0 report gives a
       different irradiation constant
   * - Zircaloy-4 growth
     - Franklin
     - [Franklin1982]_; secondary
   * - Gap gases
     - MATPRO :math:`k = A T^B`; Brokaw mixing; Kennard jump distance
     - [Brokaw1958]_, [LanningHann1975]_; secondary

Three discrepancies between sources are worth knowing.  The UN Einstein
temperature is 365.7 K in the original and 367.5 K in the BISON
documentation; the difference is below 0.1 %.  The Ross swelling correlation
is implemented as published, and one older BISON class is ten times lower.
The Zircaloy-4 creep and corrosion constants differ between the BISON
documentation and the FRAPCON-4.0 report; the BISON set is used for creep and
corrosion is not yet modelled.

Verification
------------

``tests/python/test_fuel.py`` checks every building block against an exact
answer, with every method where the building block is method-dependent:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Test
     - Result
   * - Correlations
     - Every correlation reproduces the values tabulated in, or computed
       from, its source (for instance Fink's 3.47 W/m/K at 1000 K and the
       Lucuta factors 0.8345, 1.0023, 0.9268, 0.9555 at 3 at. %).
   * - Heated rod in a tube across a gap (1D, all four methods)
     - Second-order convergence to the exact temperatures; the power leaving
       the cladding equals the fission power to round-off.
   * - The same in (r, z) and in 3D
     - Energy balance to round-off in (r, z); in 3D against the meshed
       volume, with the temperature error falling as the section is refined.
   * - Thermal stress of a long cylinder (all four methods)
     - The stresses of Timoshenko and Goodier [TimoshenkoGoodier1970]_,
       converging at order 1.5 or better.
   * - Thick tube under pressure (all four methods)
     - Lamé's hoop stress within 0.2 %.
   * - Creep under constant stress (UO2, UN, Zircaloy-4; three methods)
     - The exact strain :math:`\sigma / E + \dot\varepsilon t` to
       :math:`10^{-7}`, with Newton's method quadratic.
   * - Booth diffusion
     - Booth's release fraction to :math:`10^{-5}` over steps of any size.
   * - Rod models
     - The (r, z) and 1.5-dimensional rods, with every model active, agree
       within 10 K and 1 micrometre of gap; the 3D and (r, z) rods agree
       within the geometric error of the 3D section; the four methods agree
       on a rod in pellet-cladding contact.
   * - Rod with a flowing coolant
     - The heat leaving with the coolant equals the fission power within 1 %
       for three methods, which agree with each other.

Validation against measured rods is the next step: the IAEA FUMEX
benchmarks, the Halden rods IFA-431 and IFA-432, and the Risø and
Super-Ramp experiments of the OECD/NEA International Fuel Performance
Experiments database are the standard cases.  The release of fission gas in
particular needs calibration (the trapping factor), because the physics of
intragranular trapping and re-solution is represented by one number here.

Limitations
-----------

The module does not yet model: friction between pellet and cladding (the
contact is frictionless, and the 1.5-dimensional slices are independent
axially); fuel cracking, other than through relocation; primary creep and
plasticity of the cladding; waterside corrosion and hydriding; the radial
power profile of the plutonium build-up at the pellet rim (the profile is
flat or given); the high burnup structure; transients beyond normal
operation.  Interface conditions are not available in distributed (MPI)
runs, and the cell-centred method cannot hold a no-slip condition on a
fluid-solid interface interior to the mesh.
