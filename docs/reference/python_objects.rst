Objects written in Python
=========================

.. currentmodule:: dualmesh

A kernel, a boundary condition or a property object that the library does
not provide can be written in Python by deriving from one of these classes.
Its values are computed with :class:`ADReal` numbers, so Newton's method has
the exact Jacobian without hand differentiation.  A problem with an object
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

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Kernel
   IntegratedBC
   NodalBC
   InputParameters
