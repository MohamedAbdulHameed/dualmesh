.. _work_in_progress:

Work in progress
================

The following parts of dualmesh are being developed.  Everything else in
this manual describes what the released code does.

Fuel performance
----------------

* **Benchmarks.**  The open items of the fuel benchmarks (the fission gas
  release of IFA-677.1 rod 1, the IFA-677.1 temperatures in the first three
  cycles, the default diffusivity of Cr2O3-doped fuel, the onset of release
  at low power, and the benchmarks of U3Si2, UN, FeCrAl, SiC and Cr-coated
  cladding) are listed at the end of :doc:`fuel/benchmarks`.
* **Stress in the fission gas model.**  The external pressure on the
  grain-face bubbles will be the hydrostatic stress of the mechanics solution
  (Pastore et al. 2013, Eq. 11).  At present it is the gas pressure of the
  rod.

Uncertainty quantification and sensitivity analysis
---------------------------------------------------

The package :mod:`dualmesh.uq` is described in :doc:`theory/uncertainty`.
Its application to the fuel benchmarks is being written.  The comparison of
a calculation with a measurement is meaningful only when the uncertainty of
the calculated result is known, and the benchmark results of this manual
will therefore be given with their standard deviations.

Verification
------------

* The method of manufactured solutions for finite strain and for heat
  conduction in the deformed configuration.

Documentation
-------------

* A page that lists every default value of the library with its reason.
* Syntax examples for every object of the object reference.
