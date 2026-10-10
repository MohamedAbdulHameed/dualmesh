Benchmarks
==========

The benchmarks are published problems with known answers, solved with every
method that applies to them.  Each page gives the problem and its source, the
setup as Python calls, the results of every method against the known answer,
and the script that produced them.  The scripts are in
``verification/benchmarks``.  Each saves its results beside itself, and
``--plot-only`` redraws the figures from them.

.. list-table::
   :header-rows: 1
   :widths: 10 45 45

   * - ID
     - Problem
     - Known answer
   * - F1
     - :doc:`dfg_steady`, the steady flow around a cylinder in a channel at
       :math:`Re = 20` (DFG 2D-1)
     - drag, lift and pressure difference [SchaeferTurek1996]_
   * - F2
     - :doc:`dfg_unsteady`, the vortex shedding behind a cylinder at
       :math:`Re = 100` (DFG 2D-2), with the projection time integration
     - the largest drag and lift and the Strouhal number [SchaeferTurek1996]_
   * - F4
     - :doc:`kovasznay`, the steady flow behind a row of cylinders at
       :math:`Re = 40`
     - an exact solution of the Navier-Stokes equations [Kovasznay1948]_
   * - F5
     - :doc:`cavity`, the lid-driven cavity at :math:`Re = 1000`
     - the centerline velocities on a :math:`601 \times 601` grid [Erturk2005]_

.. toctree::
   :maxdepth: 1

   dfg_steady
   dfg_unsteady
   kovasznay
   cavity
