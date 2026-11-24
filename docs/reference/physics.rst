Physics and couplings
=====================

.. currentmodule:: dualmesh.physics

A physics is a set of equations named in physical terms.
:meth:`dualmesh.Problem.add_physics` adds one and returns its object, whose
methods add boundary conditions and extra terms to its equations.  A coupling
joins two physics, and :meth:`dualmesh.Problem.add_coupling` adds it.
:doc:`/syntax/index` lists the parameters of each physics and coupling type.

The physics object
------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Physics
   Coupling

Physics types
-------------

.. autosummary::
   :toctree: generated/
   :template: class_alone.rst

   HeatTransfer
   SolidMechanics
   FluidDynamics
   GeneralFormPDE

Coupling types
--------------

.. autosummary::
   :toctree: generated/
   :template: class_alone.rst

   ThermalExpansion
   NonisothermalFlow

The registry
------------

.. autosummary::
   :toctree: generated/

   registered
   describe
   create
