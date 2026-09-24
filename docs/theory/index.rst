Theory manual
=============

This part of the manual states exactly what the code computes.  It is meant to
be read rather than searched: each chapter gives the background, then the
equations, then what the implementation does with them, and says plainly where
a method has a limitation.

The notation follows J. N. Reddy, *Computational Methods in Engineering*
[Reddy2024]_, referred to throughout as "the book".

The first chapter is the one to read first: everything else in the library is
built on the canonical conservation form it introduces.

.. toctree::
   :maxdepth: 2
   :caption: The discretisations

   foundations
   elements
   finite_volume

.. toctree::
   :maxdepth: 2
   :caption: Solving

   nonlinear_and_time
   adaptivity
   parallel

.. toctree::
   :maxdepth: 2
   :caption: The physics modules

   heat_and_fluids
   solid_mechanics
