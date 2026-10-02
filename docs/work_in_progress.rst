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

The package :mod:`dualmesh.uq` propagates the uncertainties of model
parameters and inputs to the results of any dualmesh calculation, computes
the Sobol' sensitivity indices of the parameters, builds Gaussian process
surrogates, and calibrates parameters against measurements, from Python and
from the ``uq`` block of YAML input files.  Its theory chapter and its
application to the fuel benchmarks are being written.  The comparison of a
calculation with a measurement is meaningful only when the uncertainty of the
calculated result is known, and the benchmark results of this manual will
therefore be given with their standard deviations.

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
