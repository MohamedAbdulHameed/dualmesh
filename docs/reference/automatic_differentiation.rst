Automatic differentiation
=========================

.. currentmodule:: dualmesh

An object written in Python computes with :class:`ADReal` numbers, which
carry their derivatives with respect to the unknowns.  The functions below
act on these numbers and carry the derivatives through.  The operator ``**``
raises a number or an :class:`ADReal` number to a power.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ADReal

.. autosummary::
   :toctree: generated/

   sqrt
   exp
   log
   sin
   cos
   tanh
   abs
