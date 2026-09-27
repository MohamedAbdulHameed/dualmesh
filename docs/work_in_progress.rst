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
  cladding) are listed at the end of :doc:`theory/fuel_benchmarks`.
* **Stress in the fission gas model.**  The grain-face bubbles will feel the
  hydrostatic stress of the mechanics solution (Pastore et al. 2013, Eq.
  11).  They feel the rod gas pressure at present.

Uncertainty quantification and sensitivity analysis
---------------------------------------------------

A module that propagates the uncertainties of model parameters and inputs to
the results of any dualmesh calculation (heat transfer, solid mechanics,
fluid dynamics and fuel performance), ranks the parameters by their
influence, and calibrates them against measurements with their
uncertainties, from Python and from YAML input files.  Agreement with a
measurement means little when the uncertainty of the calculated result is
larger than the difference, so the benchmark results of this manual will be
given with their standard deviations once the module is complete.

Input files
-----------

Every calculation in dualmesh will have two forms of input, a Python script
and a YAML input file (:doc:`input_files`).  The fuel rod, the TRISO
particle and adaptive refinement are run from Python at present.  Their YAML
input files are being added.

Verification
------------

* The method of manufactured solutions for finite strain and for heat
  conduction in the deformed configuration.

Documentation
-------------

* A page that lists every default value of the library with its reason.
* Syntax examples for every object of the object reference.
