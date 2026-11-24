Theory manual
=============

The theory pages state what the code computes: the governing equations, their
discretization by the finite element method (FEM) and the dual mesh control
domain method (DMCDM), the properties of each method and its limits.  The
source of DMCDM is [Reddy2024]_, and each page cites the sources of its
equations.

Every equation is a balance law, :math:`\partial m / \partial t + \nabla \cdot
\mathbf{F} = s`, with :math:`m` the storage, :math:`\mathbf{F}` the physical
flux and :math:`s` the source of the balanced quantity.  Read
:doc:`foundations` first: it states this form and the dual mesh on which the
other pages build.  :doc:`/verification` gives the cases that check the
methods.

.. toctree::
   :maxdepth: 2
   :caption: The discretizations

   foundations
   elements
   finite_volume

.. toctree::
   :maxdepth: 2
   :caption: The physics

   heat_transfer
   fluids
   solid_mechanics

.. toctree::
   :maxdepth: 2
   :caption: Solving

   nonlinear_and_time
   adaptivity
   parallel
   eigenvalues
