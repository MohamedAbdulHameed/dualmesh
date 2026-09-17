Cross-verification against OpenFOAM
===================================

Agreement with a book's tables shows that the implementation matches the
method.  Agreement with an independent code on the same problem shows that the
method and the implementation together describe the physics.  The scripts in
``verification/openfoam`` do the second: they run `OpenFOAM
<https://www.openfoam.com>`_, run dualmesh on the same problem — on the same
mesh, where possible — and print the two sets of results side by side.

Running them needs an OpenFOAM installation:

.. code-block:: console

   cd verification/openfoam
   . /usr/lib/openfoam/openfoam2406/etc/bashrc     # or your own installation

   ./cavity_openfoam.sh 64      && python compare_cavity.py --resolution 64
   ./conduction_openfoam.sh 40 20 && python compare_conduction.py --nx 40 --ny 20
   ./plate_hole_openfoam.sh     && python compare_plate_hole.py

The results below were obtained with OpenFOAM v2406.

Lid-driven cavity at Re = 100
-----------------------------

``icoFoam`` (transient incompressible solver, marched to steady state) against
the penalty Navier–Stokes solver of dualmesh on a uniform 64 × 64 mesh, with the
tabulated data of Ghia, Ghia and Shin (1982) as the reference.

.. list-table:: :math:`u` on the vertical centreline
   :header-rows: 1
   :widths: 15 25 25 25

   * - :math:`y`
     - dualmesh
     - OpenFOAM
     - Ghia et al.
   * - 0.1719
     - −0.09537
     - −0.10153
     - −0.10150
   * - 0.4531
     - −0.19636
     - −0.21229
     - −0.21090
   * - 0.7344
     - 0.00719
     - 0.00364
     - 0.00332
   * - 0.9531
     - 0.69136
     - 0.69019
     - 0.68717

On this mesh the largest difference from the reference data is 0.0145 for
dualmesh and 0.0049 for OpenFOAM (in units of the lid velocity).  Refining the
dual mesh calculation shows clean convergence towards the reference data:

.. list-table::
   :header-rows: 1

   * - mesh
     - max :math:`|u - u_{\text{Ghia}}|`
     - max :math:`|v - v_{\text{Ghia}}|`
   * - 32 × 32
     - 0.0311
     - 0.0266
   * - 64 × 64
     - 0.0145
     - 0.0096
   * - 128 × 128
     - 0.0059
     - 0.0034

so that at 128 × 128 the dual mesh solution matches the reference data as
closely as the finite volume solution does at 64 × 64.  The two codes are
solving genuinely different discretizations — a segregated PISO finite volume
scheme with a pressure equation, against a coupled penalty formulation with
bilinear elements — so this level of agreement is what one hopes for rather
than an identity check.

Steady conduction in a plate
----------------------------

``laplacianFoam`` marched to steady state against the dual mesh solution of the
same problem (the geometry and boundary conditions of Example 5.4.2 of the
book) on the same 40 × 20 grid:

.. list-table:: :math:`T(x, y=0)` in kelvin
   :header-rows: 1

   * - :math:`x`
     - dualmesh
     - OpenFOAM
   * - 0.050
     - 464.560
     - 464.566
   * - 0.100
     - 420.524
     - 420.530
   * - 0.150
     - 364.560
     - 364.566

The largest difference is 0.006 K out of a 200 K range, that is one part in
:math:`10^5`.  The same script also checks global conservation: summing the
reactions over every boundary node gives :math:`-5 \times 10^{-12}` W/m.

Plate with a hole
-----------------

The most direct comparison of the three.  OpenFOAM's ``plateHole`` tutorial
(``solidDisplacementFoam``) is run, and its mesh — 1000 quadrilaterals, graded
around the hole — is then imported and solved by dualmesh, so that the two
codes use *identical* meshes, boundary conditions, and material data (a quarter
plate, 2 m × 2 m, hole radius 0.5 m, 10 kPa remote traction, E = 200 GPa,
:math:`\nu = 0.3`, plane stress).

.. list-table:: :math:`\sigma_{xx}` along :math:`x = 0` (Pa)
   :header-rows: 1

   * - :math:`y`
     - dualmesh
     - OpenFOAM
     - Kirsch (infinite plate)
   * - 0.50
     - 31875
     - 31955
     - 30000
   * - 0.75
     - 17322
     - 17365
     - 15185
   * - 1.00
     - 13267
     - 13286
     - 12188
   * - 1.50
     - 10341
     - 10324
     - 10741
   * - 2.00
     - 7656
     - 7643
     - 10371

The stress concentration factor at the hole is 3.187 (dualmesh) against 3.195
(OpenFOAM); both exceed the infinite-plate value of 3 because the plate is only
four hole diameters wide.  Away from the hole the two codes differ by less than
0.3 %, and they depart from Kirsch's solution together, as they must, because
that solution is for an infinite plate.

This case also exercises the mesh interoperability: ``read_openfoam_mesh`` in
``compare_plate_hole.py`` converts the one-cell-thick OpenFOAM mesh into a
two-dimensional dual mesh problem through meshio, which is the same path that
:func:`dualmesh.read_mesh` uses for Gmsh, Exodus, and Abaqus files.
