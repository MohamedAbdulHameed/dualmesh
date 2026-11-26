Benchmarks
==========

The benchmarks are published flow problems with known answers, solved with the
finite element method (FEM), the dual mesh control domain method (DMCDM) and
the formulations of incompressible flow that apply to each.  Each page gives
the problem and its source, the setup as Python calls, the results of every
method against the known answer, and the command that reproduces them.  The
scripts are in ``verification/benchmarks/fluids``.  Each script saves its
results in a cache file beside itself, and ``--plot-only`` redraws the figures
from that cache.  :doc:`../verification` gives the other verification cases.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Problem
     - Known answer
   * - :doc:`dfg_steady`, the steady flow around a cylinder in a channel at
       the Reynolds number :math:`Re = 20` (the benchmark 2D-1 of the German
       Research Foundation, DFG)
     - drag, lift and pressure difference [SchaeferTurek1996]_
   * - :doc:`dfg_unsteady`, the vortex shedding behind a cylinder at
       :math:`Re = 100` (DFG 2D-2)
     - the largest drag and lift and the Strouhal number [SchaeferTurek1996]_
   * - :doc:`kovasznay`, the steady flow behind a row of cylinders at
       :math:`Re = 40`
     - an exact solution of the Navier-Stokes equations [Kovasznay1948]_
   * - :doc:`cavity`, the lid-driven cavity at :math:`Re = 1000`
     - the centerline velocities on a :math:`601 \times 601` grid [Erturk2005]_

.. toctree::
   :maxdepth: 1

   dfg_steady
   dfg_unsteady
   kovasznay
   cavity
