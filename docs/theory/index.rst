Theory manual
=============

This part of the manual states what the code computes.  Each chapter gives the
background, the governing equations and their discretisation, and the
limitations of each method.

The notation follows J. N. Reddy, *Computational Methods in Engineering*
[Reddy2024]_, referred to throughout as "the book".

The first chapter introduces the canonical conservation form on which the rest
of the library is built, and it should be read before the others.

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
   neutronics

.. toctree::
   :maxdepth: 2
   :caption: Studies

   uncertainty
