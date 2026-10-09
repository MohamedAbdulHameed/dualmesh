Heat conduction
===============

A cooling fin in one dimension
------------------------------

The equation :math:`-u'' + 400\,u = 0` on :math:`(0, 0.05)` with
:math:`u(0) = 300` and :math:`u'(L) + 2u(L) = 0` is Example 5.3.1 of the book.
It shows the three components of every problem (kernels, boundary conditions
and a solve) and the recovery of the secondary variable at the fixed end.

.. literalinclude:: ../../examples/cooling_fin.py
   :language: python

Running it prints

.. code-block:: text

   dmcdm  u =  300.000  257.618  225.593  202.637  187.827  180.568
   dmcdm  Q(0) = 4817.0
   fem    u =  300.000  257.567  225.507  202.527  187.703  180.437

which are the values of Table 5.3.1 of the book.  The two computations differ
only in the ``method`` argument of :class:`dualmesh.Problem`.

A bus bar in two dimensions
---------------------------

Example 5.4.3 adds internal heat generation and a convective boundary.  The
script ``examples/bus_bar.py`` defines the problem, adds five
post-processors (the heat flows through the left and right sides, the
temperatures at the middle of the bottom and top edges, and the largest
temperature), and writes the field and the post-processors to files:

.. literalinclude:: ../../examples/bus_bar.py
   :language: python
   :pyobject: bus_bar

The temperature at the middle of the bottom edge is 83.142, as in Table 5.4.3
of the book.

Points to note
--------------

* An insulated boundary needs no boundary condition: a zero natural condition
  is the default, which in the dual mesh method means that the boundary face of
  the control domain contributes nothing.
* ``total_reaction`` returns the heat flow through a boundary, computed directly
  from the equations that the Dirichlet conditions replaced, which avoids
  differentiating the solution afterwards.
* A temperature-dependent conductivity :math:`k = k_0 (1 + k_1 T)` needs only
  ``temperature_polynomial=[1.0, k1]`` on the ``heat_conduction`` kernel.  The
  problem is then solved by Newton's method with the exact Jacobian, or by direct
  iteration with ``nonlinear_solver="picard"``.
