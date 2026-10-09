.. _work_in_progress:

Work in progress
================

The following parts of dualmesh are being developed.  Everything else in
this manual describes what the released code does.

Uncertainty quantification and sensitivity analysis
---------------------------------------------------

The package :mod:`dualmesh.uq` is described in :doc:`theory/uncertainty`.
The comparison of a calculation with a measurement is meaningful only when
the uncertainty of the calculated result is known, and the benchmark results
of this manual will therefore be given with their standard deviations.

Verification
------------

* The method of manufactured solutions for finite strain and for heat
  conduction in the deformed configuration.
* A mortar coupling of the two surfaces of a gap.  With the closest-point
  pairing, ``gap_heat_transfer`` converges at first order when the meshes of
  the two surfaces do not match.

Documentation
-------------

* A page that lists every default value of the library with its reason.
* Syntax examples for every object of the object reference.
