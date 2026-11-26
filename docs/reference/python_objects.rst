Objects written in Python
=========================

.. currentmodule:: dualmesh

You write a kernel, a boundary condition or a property object that the
library does not provide by deriving a class from one of the classes below.
Your object computes with :class:`ADReal` numbers, so Newton's method gets the
exact Jacobian without differentiation by hand.  A problem with an object
written in Python assembles on one thread.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   PythonKernel
   PythonBoundaryCondition
   PythonNodalBoundaryCondition
   PythonProperty
   QpContext

The base classes of the compiled objects
----------------------------------------

The compiled kernels and boundary conditions derive from these classes, and
:class:`InputParameters` holds the declared parameters of an object.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Kernel
   IntegratedBC
   NodalBC
   InputParameters
