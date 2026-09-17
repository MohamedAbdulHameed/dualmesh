Heat conduction
===============

A cooling fin in one dimension
------------------------------

The equation :math:`-u'' + 400\,u = 0` on :math:`(0, 0.05)` with
:math:`u(0) = 300` and :math:`u'(L) + 2u(L) = 0` is Example 5.3.1 of the book.
It shows the three ingredients of every problem — kernels, boundary conditions,
and a solve — and the recovery of the secondary variable at the fixed end.

.. literalinclude:: ../../examples/cooling_fin.py
   :language: python

Running it prints

.. code-block:: text

   dmcdm  u =  300.000  257.618  225.593  202.637  187.827  180.568
   dmcdm  Q(0) = 4817.0
   fem    u =  300.000  257.567  225.507  202.527  187.703  180.437

which are the values of Table 5.3.1 of the book.  Note how little separates the
two methods: only the ``method`` argument of :class:`dualmesh.Problem`.

A bus bar in two dimensions
---------------------------

Example 5.4.3 adds internal heat generation and a convective boundary.  The
same problem is also available as an input file (:doc:`../input_files`).

.. literalinclude:: ../../examples/bus_bar.yaml
   :language: yaml

Things worth noticing
---------------------

* An insulated boundary needs no boundary condition at all: a zero natural
  condition is the default, which in the dual mesh method means that the
  boundary face of the control domain simply contributes nothing.
* ``total_reaction`` returns the heat flow through a boundary, computed from
  the equations that the Dirichlet conditions replaced — not by
  differentiating the solution afterwards.
* A temperature-dependent conductivity :math:`k = k_0 (1 + k_1 T)` needs only
  ``temperature_polynomial=[1.0, k1]`` on the ``HeatConduction`` kernel; the
  problem then solves by Newton's method with the exact Jacobian, or by direct
  iteration with ``nonlinear_solver="picard"``.
