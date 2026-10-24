Physics and couplings
=====================

.. currentmodule:: dualmesh.physics

A physics is a set of equations named in physical terms, added with
:meth:`dualmesh.Problem.add_physics`, which returns the object below.  Its
methods add boundary conditions and extra terms to its equations.  A
coupling joins two physics and is added with
:meth:`dualmesh.Problem.add_coupling`.  The parameters of each physics and
coupling type are listed in the :doc:`/syntax/index`.

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
