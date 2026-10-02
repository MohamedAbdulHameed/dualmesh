.. _fuel:

Nuclear fuel performance
========================

A nuclear fuel rod consists of a stack of ceramic pellets inside a metal tube,
the *cladding*.  The fission of uranium in the pellets releases heat, which is
conducted through the pellets, across a thin gas-filled gap and through the
cladding, and is removed by the coolant.  During months and years of
irradiation the rod evolves: the pellets crack, densify and swell, they release
part of the fission gas they produce, and they eventually come into contact
with the cladding, which creeps down onto them under the coolant pressure.  A
*fuel performance code* follows these coupled thermal and mechanical changes
through a power history, in order to show that the fuel stays below its
melting point, that the strain of the cladding stays acceptable, and that the
pressure inside the rod stays below its limit.

The ``fuel_performance`` module and the Python package :mod:`dualmesh.fuel`
provide this capability.  They compute:

* The temperature of the fuel and the cladding, with temperature- and
  burnup-dependent properties, the heat transfer across the gap (gas
  conduction, radiation and solid contact), and the heat transfer to the
  coolant (forced convection with the heat-up of the coolant along the rod,
  or a prescribed cladding temperature).
* The stresses and strains of the fuel and the cladding, with thermal
  expansion, densification, swelling and relocation of the fuel, thermal
  expansion and irradiation growth of the cladding, creep of both, the gas
  and coolant pressures, and frictionless pellet-cladding contact.
* The fission gas: its diffusion to the grain boundaries, its storage in
  grain-face bubbles, its release (including the burst release on power
  changes), the gaseous swelling, and the pressure and composition of the gas
  in the rod.
* The stresses and the failure probability of TRISO coated particles.

The fuels are UO2, Cr2O3-doped UO2, UN and U3Si2, and the claddings are
Zircaloy, FeCrAl, SiC/SiC composite and chromium-coated Zircaloy.  A fuel or
cladding of the user's own can be defined by expressions
(:class:`~dualmesh.fuel.CustomFuel`, :class:`~dualmesh.fuel.CustomCladding`).
A rod is computed in one of three forms, which share the materials: an
axisymmetric (r-z) model of the whole rod, a 1.5-dimensional model of axial
slices in generalized plane strain, and a three-dimensional model.  Each of
the four discretisations of dualmesh (finite elements, the dual mesh control
domain method and the two finite volume methods) solves the same rod.

The chapters of this part are:

* :doc:`models`: the equations of the rod and their numerical treatment.
* :doc:`correlations`: every material correlation, with its source and its
  check values.
* :doc:`benchmarks`: the verification against the IAEA FUMEX-II exercises and
  the validation against Halden irradiations, in comparison with the
  published results of BISON.
* :doc:`triso`: the TRISO particle model and the IAEA CRP-6 benchmark.

A first fuel rod
----------------

A rod is described by input groups, each a Python dataclass whose fields carry
their units and their defaults.  The following script computes a PWR rod at a
constant linear heat rate of 20 kW/m up to 30 MWd/kgHM, with time steps of at
most ten days:

.. code-block:: python

   from dualmesh import fuel

   rod = fuel.FuelRod(
       geometry=fuel.RodGeometry.from_diameters(
           pellet_outer_diameter=8.19e-3, clad_inner_diameter=8.36e-3,
           clad_outer_diameter=9.50e-3, fuel_stack_height=0.1),
       fuel=fuel.UO2Fuel(enrichment=0.045),
       cladding=fuel.ZircaloyCladding(),
       fill_gas=fuel.FillGas(pressure=2.0e6, plenum_volume=0.3e-6),
       coolant=fuel.ForcedConvection(
           inlet_temperature=565.0, pressure=15.5e6, mass_flux=3800.0, rod_pitch=12.6e-3),
       power_history=fuel.PowerHistory(
           linear_heat_rate=[20e3, 20e3], burnup=[0.0, 30.0], burnup_unit="MWd/kgHM"),
       numerics=fuel.RodNumerics(max_time_step=10 * 86400.0),
   )
   result = rod.run()
   print(result.summary())

The input groups are:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Group
     - Content
   * - :class:`~dualmesh.fuel.RodGeometry`
     - The radii of the pellet and the cladding, the height of the fuel stack,
       an optional central bore, the plenum and an optional coating.
   * - Fuel
     - :class:`~dualmesh.fuel.UO2Fuel`, :class:`~dualmesh.fuel.DopedUO2Fuel`,
       :class:`~dualmesh.fuel.UNFuel`, :class:`~dualmesh.fuel.U3Si2Fuel` or
       :class:`~dualmesh.fuel.CustomFuel`: the fabrication data and the choice
       of correlations.
   * - Cladding
     - :class:`~dualmesh.fuel.ZircaloyCladding`,
       :class:`~dualmesh.fuel.FeCrAlCladding`,
       :class:`~dualmesh.fuel.SiCCladding`,
       :class:`~dualmesh.fuel.CoatedCladding` or
       :class:`~dualmesh.fuel.CustomCladding`.
   * - :class:`~dualmesh.fuel.FillGas`
     - The fill pressure and temperature, the composition and the plenum
       volume.
   * - Coolant
     - :class:`~dualmesh.fuel.ForcedConvection` (inlet temperature, pressure,
       mass flux and pitch of the channel) or
       :class:`~dualmesh.fuel.PrescribedCladdingTemperature`.
   * - :class:`~dualmesh.fuel.PowerHistory`
     - The linear heat rate against time or burnup, the axial and radial
       profiles, and the fast neutron flux per unit linear heat rate.
   * - :class:`~dualmesh.fuel.RodModels`
     - Which physical models are active.
   * - :class:`~dualmesh.fuel.RodNumerics`
     - The rod model (axisymmetric, 1.5-dimensional or three-dimensional),
       the discretisation, the mesh and the time steps.
   * - :class:`~dualmesh.fuel.RodOutput`
     - The output times and files.
   * - :class:`~dualmesh.fuel.ModelFactors`
     - Multipliers on the models (all equal to one by default), for
       sensitivity and uncertainty studies.

Before a calculation, the rod prints every input value with its unit, and
marks the values that were left at their defaults.  The result holds the
history of the rod (e.g., the gas pressure, the fission gas release and the
centre temperature) and the axial profiles at every output time, and it can be
written to CSV files.

The script ``examples/custom_fuel.py`` defines a fuel by expressions, and
``examples/fuel_rod_conjugate.py`` couples a rod to a coolant channel solved
with the flow equations.  The benchmark scripts in
``verification/benchmarks/`` compute the measured rods of :doc:`benchmarks`.
