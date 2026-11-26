Expressions and functions
=========================

.. currentmodule:: dualmesh

A coefficient, a boundary value or a source takes a number, a function of
position and time, or an expression written as text.  The code compiles an
expression once and differentiates it by automatic differentiation.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ParsedFunction
   Expression
   SettableFunction

.. autosummary::
   :toctree: generated/

   parsed_function
