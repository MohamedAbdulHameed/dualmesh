.. _references:

References
==========

This page lists the works that the docs cite, and the papers that develop the
dual mesh control domain method (DMCDM).

The method
----------

.. [Reddy2024] J. N. Reddy, *Computational Methods in Engineering: Finite
   Difference, Finite Volume, Finite Element, and Dual Mesh Control Domain
   Methods*, 1st edition, CRC Press, Boca Raton, FL, 2024,
   ISBN 978-1-032-46637-8.  DOI: 10.1201/9781003382812.  The docs follow its
   Chapter 5 for DMCDM and its Chapter 3 for the finite volume methods.

.. [Reddy2019a] J. N. Reddy, "A dual mesh finite domain method for the
   numerical solution of differential equations", *International Journal for
   Computational Methods in Engineering Science and Mechanics*, 20(3):212–228,
   2019.  DOI: 10.1080/15502287.2019.1610987.  The paper that introduces the
   method.

.. [ReddyKimMartinez2020] J. N. Reddy, N. Kim and M. Martinez, "A dual mesh
   control domain method for the solution of nonlinear Poisson's equation and
   the Navier–Stokes equations for incompressible fluids", *Physics of
   Fluids*, 32(9):093608, 2020.  DOI: 10.1063/5.0026274.

.. [ReddyNampally2020] J. N. Reddy and P. Nampally, "A dual mesh finite domain
   method for the analysis of functionally graded beams", *Composite
   Structures*, 251:112648, 2020.  DOI: 10.1016/j.compstruct.2020.112648.

.. [ReddyNampallySrinivasa2020] J. N. Reddy, P. Nampally and A. R. Srinivasa,
   "Nonlinear analysis of functionally graded beams using the dual mesh finite
   domain method and the finite element method", *International Journal of
   Non-Linear Mechanics*, 127:103575, 2020.
   DOI: 10.1016/j.ijnonlinmec.2020.103575.

.. [NampallyReddy2020] P. Nampally and J. N. Reddy, "Bending analysis of
   functionally graded axisymmetric circular plates using the dual mesh finite
   domain method", *Latin American Journal of Solids and Structures*,
   17(7):e302, 2020.  DOI: 10.1590/1679-78256218.

.. [ReddyMartinez2021] J. N. Reddy and M. Martinez, "A dual mesh finite domain
   method for steady-state convection–diffusion problems", *Computers &
   Fluids*, 214:104760, 2021.  DOI: 10.1016/j.compfluid.2020.104760.

.. [ReddyNampallyPhan2021] J. N. Reddy, P. Nampally and N. Phan, "Dual mesh
   control domain analysis of functionally graded circular plates accounting
   for moderate rotations", *Composite Structures*, 257:113153, 2021.
   DOI: 10.1016/j.compstruct.2020.113153.

.. [NampallyRuoccoReddy2021] P. Nampally, E. Ruocco and J. N. Reddy, "Bending
   analysis of functionally graded rectangular plates using the dual mesh
   control domain method", *International Journal for Computational Methods in
   Engineering Science and Mechanics*, 22(5):425–437, 2021.
   DOI: 10.1080/15502287.2021.1890279.

.. [Jiao2023] Z. Jiao, T. Heblekar, G. Wang, R. Xu, W. Chen and J. N. Reddy,
   "Analysis of plane elasticity problems using the dual mesh control domain
   method", *Computer Methods in Applied Mechanics and Engineering*,
   416:116342, 2023.  DOI: 10.1016/j.cma.2023.116342.

.. [Heblekar2024] T. Y. Heblekar, J. N. Reddy and A. R. Srinivasa, "Analysis of
   nonlinear problems using the Dual Mesh Control Domain Method with arbitrary
   meshes", *Computer Methods in Applied Mechanics and Engineering*,
   427:117044, 2024.  DOI: 10.1016/j.cma.2024.117044.

.. [Jiao2024] Z. Jiao, T. Heblekar, G. Wang, R. Xu and J. N. Reddy, "Static,
   free vibration, and buckling analysis of functionally graded plates using
   the dual mesh control domain method", *Computers & Structures*, 305:107575,
   2024.  DOI: 10.1016/j.compstruc.2024.107575.

.. [Areias2025] P. Areias, A. R. Srinivasa, F. Moleiro and J. N. Reddy, "Finite
   strain analysis with the dual mesh control domain method", *International
   Journal for Numerical Methods in Engineering*, 126(1):e7654, 2025.
   DOI: 10.1002/nme.7654.

.. [HeblekarReddy2025] T. Y. Heblekar and J. N. Reddy, "An improved dual mesh
   control domain formulation for the unsteady flow of viscous incompressible
   fluids", *Physics of Fluids*, 37(3):033119, 2025.
   DOI: 10.1063/5.0259696.

.. [Heblekar2026p] T. Y. Heblekar, J. N. Reddy and A. R. Srinivasa,
   "p-Refinement in the dual-mesh control domain method", *Computing in Science
   & Engineering*, 28(1):64–73, 2026.  DOI: 10.1109/MCSE.2025.3641847.

.. [Heblekar2026spectral] T. Y. Heblekar, J. N. Reddy and A. R. Srinivasa, "A
   spectral dual mesh control domain framework for nonlinear fluid flow and
   coupled multi-field problems", *International Journal of Non-Linear
   Mechanics*, 186:105350, 2026.  DOI: 10.1016/j.ijnonlinmec.2026.105350.

.. [Heblekar2026nonlinear] T. Y. Heblekar, J. N. Reddy and A. R. Srinivasa,
   "Dual-mesh control-domain analysis of nonlinear problems in mechanics",
   *Theoretical and Applied Mechanics*, article in press, 2026.
   DOI: 10.2298/TAM260601007H.

Finite volume methods
---------------------

.. [BaligaPatankar1980] B. R. Baliga and S. V. Patankar, "A new finite-element
   formulation for convection-diffusion problems", *Numerical Heat Transfer*,
   3(4):393–409, 1980.  DOI: 10.1080/01495728008961767.  The control volume
   finite element method of :doc:`theory/finite_volume`.

.. [Patankar1980] S. V. Patankar, *Numerical Heat Transfer and Fluid Flow*,
   Hemisphere Publishing, Washington, DC, 1980.  The classical finite volume
   method of :doc:`theory/finite_volume`.

.. [Jasak1996] H. Jasak, *Error Analysis and Estimation for the Finite Volume
   Method with Applications to Fluid Flows*, PhD thesis, Imperial College
   London, 1996.  The bounded face schemes of the DMCDM advection in
   :doc:`theory/finite_volume`.

Finite elements
---------------

.. [Irons1966] B. M. Irons, "Engineering applications of numerical integration
   in stiffness methods", *AIAA Journal*, 4(11):2035–2037, 1966.
   DOI: 10.2514/3.3836.  The isoparametric map of :doc:`theory/elements`.

.. [Bedrosian1992] G. Bedrosian, "Shape functions and integration formulas for
   three-dimensional finite element analysis", *International Journal for
   Numerical Methods in Engineering*, 35(1):95–108, 1992.
   DOI: 10.1002/nme.1620350106.  The rational basis of the pyramid element.

.. [Duffy1982] M. G. Duffy, "Quadrature over a pyramid or cube of integrands
   with a singularity at a vertex", *SIAM Journal on Numerical Analysis*,
   19(6):1260–1262, 1982.  DOI: 10.1137/0719090.  The collapsed-coordinate map
   that integrates a triangle, a tetrahedron, a prism or a pyramid as a square
   or a cube with merged corners.

.. [Dunavant1985] D. A. Dunavant, "High degree efficient symmetrical Gaussian
   quadrature rules for the triangle", *International Journal for Numerical
   Methods in Engineering*, 21(6):1129–1148, 1985.
   DOI: 10.1002/nme.1620210612.  The symmetric quadrature rules of the
   triangle in :doc:`theory/elements`.

.. [Reddy2019b] J. N. Reddy, *Introduction to the Finite Element Method*, 4th
   edition, McGraw-Hill Education, New York, NY, 2019,
   ISBN 978-1-259-86190-1.  The serendipity elements of
   :doc:`theory/elements`.

.. [DouglasDupont1974] J. Douglas, Jr. and T. Dupont, "Galerkin approximations
   for the two point boundary problem using continuous, piecewise polynomial
   spaces", *Numerische Mathematik*, 22(2):99–109, 1974.
   DOI: 10.1007/BF01436724.  The superconvergence of the Galerkin method at
   the nodes of a one-dimensional mesh.

.. [Aubin1967] J.-P. Aubin, "Behavior of the error of the approximate
   solutions of boundary value problems for linear elliptic operators by
   Galerkin's and finite difference methods", *Annali della Scuola Normale
   Superiore di Pisa, Classe di Scienze*, Serie 3, 21(4):599–637, 1967.  With
   [Nitsche1968]_, the duality argument that gives the Galerkin method one more
   order of accuracy in the :math:`L^2` norm than in the energy norm.

.. [Nitsche1968] J. Nitsche, "Ein Kriterium für die Quasi-Optimalität des
   Ritzschen Verfahrens", *Numerische Mathematik*, 11(4):346–348, 1968.
   DOI: 10.1007/BF02166687.

.. [Barlow1976] J. Barlow, "Optimal stress locations in finite element models",
   *International Journal for Numerical Methods in Engineering*,
   10(2):243–251, 1976.  DOI: 10.1002/nme.1620100202.  The Gauss points as the
   points where the derivative of a finite element solution is most accurate.

.. [ZienkiewiczTaylorToo1971] O. C. Zienkiewicz, R. L. Taylor and J. M. Too,
   "Reduced integration technique in general analysis of plates and shells",
   *International Journal for Numerical Methods in Engineering*, 3(2):275–290,
   1971.  DOI: 10.1002/nme.1620030211.  Reduced integration of the shear terms
   of plates.

.. [HughesCohenHaroun1978] T. J. R. Hughes, M. Cohen and M. Haroun, "Reduced
   and selective integration techniques in the finite element analysis of
   plates", *Nuclear Engineering and Design*, 46(1):203–222, 1978.
   DOI: 10.1016/0029-5493(78)90184-X.  The selective reduced integration of
   the ``reduced_integration`` parameter.

.. [MalkusHughes1978] D. S. Malkus and T. J. R. Hughes, "Mixed finite element
   methods: reduced and selective integration techniques: a unification of
   concepts", *Computer Methods in Applied Mechanics and Engineering*,
   15(1):63–81, 1978.  DOI: 10.1016/0045-7825(78)90005-1.  The equivalence of
   selective reduced integration and a mixed formulation.

Incompressible flow
-------------------

.. [HughesLiuBrooks1979] T. J. R. Hughes, W. K. Liu and A. Brooks, "Finite
   element analysis of incompressible viscous flows by the penalty function
   formulation", *Journal of Computational Physics*, 30(1):1–60, 1979.
   DOI: 10.1016/0021-9991(79)90086-X.  The penalty formulation of
   :doc:`theory/fluids`.

.. [TaylorHood1973] C. Taylor and P. Hood, "A numerical solution of the
   Navier-Stokes equations using the finite element technique", *Computers &
   Fluids*, 1(1):73–100, 1973.  DOI: 10.1016/0045-7930(73)90027-3.  The
   element with quadratic velocity and linear pressure that carries their
   names.

.. [BoffiBrezziFortin2013] D. Boffi, F. Brezzi and M. Fortin, *Mixed Finite
   Element Methods and Applications*, Springer Series in Computational
   Mathematics 44, Springer, Berlin, 2013.  DOI: 10.1007/978-3-642-36519-5.
   The inf-sup condition and the error estimates of the Taylor-Hood elements.

.. [BrezziPitkaranta1984] F. Brezzi and J. Pitkäranta, "On the stabilization of
   finite element approximations of the Stokes equations", in *Efficient
   Solutions of Elliptic Systems*, Notes on Numerical Fluid Mechanics 10,
   pp. 11–19, Vieweg, 1984.  DOI: 10.1007/978-3-663-14169-3_2.  The pressure
   stabilization that equal-order elements need.

.. [HughesFrancaBalestra1986] T. J. R. Hughes, L. P. Franca and M. Balestra, "A
   new finite element formulation for computational fluid dynamics: V.
   Circumventing the Babuška-Brezzi condition: a stable Petrov-Galerkin
   formulation of the Stokes problem accommodating equal-order interpolations",
   *Computer Methods in Applied Mechanics and Engineering*, 59(1):85–99, 1986.
   DOI: 10.1016/0045-7825(86)90025-3.  The pressure stabilization of the
   ``mass_conservation`` kernel.

.. [BrooksHughes1982] A. N. Brooks and T. J. R. Hughes, "Streamline
   upwind/Petrov-Galerkin formulations for convection dominated flows with
   particular emphasis on the incompressible Navier-Stokes equations",
   *Computer Methods in Applied Mechanics and Engineering*, 32(1–3):199–259,
   1982.  DOI: 10.1016/0045-7825(82)90071-8.  The streamline stabilization of
   the ``momentum_stabilization`` kernel.

.. [Tezduyar1991] T. E. Tezduyar, "Stabilized finite element formulations for
   incompressible flow computations", *Advances in Applied Mechanics*,
   28:1–44, 1991.  DOI: 10.1016/S0065-2156(08)70153-4.  The stabilization
   parameter :math:`\tau`.

.. [TezduyarMittalRayShih1992] T. E. Tezduyar, S. Mittal, S. E. Ray and R. Shih,
   "Incompressible flow computations with stabilized bilinear and linear
   equal-order-interpolation velocity-pressure elements", *Computer Methods in
   Applied Mechanics and Engineering*, 95(2):221–242, 1992.
   DOI: 10.1016/0045-7825(92)90141-6.

.. [BochevGunzburgerLehoucq2007] P. B. Bochev, M. D. Gunzburger and R. B.
   Lehoucq, "On stabilized finite element methods for the Stokes problem in the
   small time step limit", *International Journal for Numerical Methods in
   Fluids*, 53(4):573–597, 2007.  DOI: 10.1002/fld.1295.  The loss of pressure
   stability of equal-order elements at small time steps.

.. [HughesWells2005] T. J. R. Hughes and G. N. Wells, "Conservation properties
   for the Galerkin and stabilised forms of the advection-diffusion and
   incompressible Navier-Stokes equations", *Computer Methods in Applied
   Mechanics and Engineering*, 194:1141–1159, 2005.
   DOI: 10.1016/j.cma.2004.06.034.  Erratum: 195:1277–1278, 2006.
   DOI: 10.1016/j.cma.2005.04.005.  The fine-scale velocity that keeps the
   stabilized forms conservative.

.. [RhieChow1983] C. M. Rhie and W. L. Chow, "Numerical study of the turbulent
   flow past an airfoil with trailing edge separation", *AIAA Journal*,
   21(11):1525–1532, 1983.  DOI: 10.2514/3.8284.  The momentum interpolation
   of collocated finite volumes.

.. [Fischer1998] P. F. Fischer, "Projection techniques for iterative solution of
   :math:`Ax = b` with successive right-hand sides", *Computer Methods in
   Applied Mechanics and Engineering*, 163:193–204, 1998.
   DOI: 10.1016/S0045-7825(98)00012-7.  The initial guess of the pressure
   solve from earlier pressures.

Structures
----------

.. [ReddyBeams2022] J. N. Reddy, *Theories and Analyses of Beams and
   Axisymmetric Circular Plates*, CRC Press, Boca Raton, FL, 2022,
   ISBN 978-1-032-14739-0.  DOI: 10.1201/9781003240846.  The beam and circular
   plate theories of :doc:`theory/solid_mechanics`.

.. [ReddyPlates2007] J. N. Reddy, *Theory and Analysis of Elastic Plates and
   Shells*, 2nd edition, CRC Press, Boca Raton, FL, 2007,
   ISBN 978-0-8493-8415-8.  The plate theories of
   :doc:`theory/solid_mechanics`.

.. [HughesBrezzi1989] T. J. R. Hughes and F. Brezzi, "On drilling degrees of
   freedom", *Computer Methods in Applied Mechanics and Engineering*,
   72(1):105–121, 1989.  The rotation about the normal of a shell, tied to the
   rotation of its membrane.

Linear and nonlinear solvers
----------------------------

.. [PETSc2023] S. Balay, S. Abhyankar, M. F. Adams, S. Benson, J. Brown,
   P. Brune, K. Buschelman, E. Constantinescu, L. Dalcin, A. Dener,
   V. Eijkhout, J. Faibussowitsch, W. D. Gropp, V. Hapla, T. Isaac,
   P. Jolivet, D. Karpeev, D. Kaushik, M. G. Knepley, F. Kong, S. Kruger,
   D. A. May, L. Curfman McInnes, R. Tran Mills, L. Mitchell, T. Munson,
   J. E. Roman, K. Rupp, P. Sanan, J. Sarich, B. F. Smith, S. Zampini,
   H. Zhang, H. Zhang and J. Zhang, *PETSc/TAO Users Manual*, Revision 3.19,
   Argonne National Laboratory report ANL-21/39, 2023.
   DOI: 10.2172/1968587.  The library that solves the linear systems.

.. [Eigen] G. Guennebaud, B. Jacob and others, *Eigen*, 2010,
   https://libeigen.gitlab.io.  The linear algebra library.

.. [Amestoy2001] P. R. Amestoy, I. S. Duff, J.-Y. L'Excellent and J. Koster,
   "A fully asynchronous multifrontal solver using distributed dynamic
   scheduling", *SIAM Journal on Matrix Analysis and Applications*,
   23(1):15–41, 2001.  DOI: 10.1137/S0895479899358194.  The MUltifrontal
   Massively Parallel sparse direct Solver (MUMPS).

.. [George1973] A. George, "Nested dissection of a regular finite element
   mesh", *SIAM Journal on Numerical Analysis*, 10(2):345–363, 1973.
   DOI: 10.1137/0710032.  With [LiptonRoseTarjan1979]_, the fill and the
   operation count of a sparse direct factorization in two and three
   dimensions.

.. [LiptonRoseTarjan1979] R. J. Lipton, D. J. Rose and R. E. Tarjan,
   "Generalized nested dissection", *SIAM Journal on Numerical Analysis*,
   16(2):346–358, 1979.  DOI: 10.1137/0716027.

.. [Saad1993] Y. Saad, "A flexible inner-outer preconditioned GMRES
   algorithm", *SIAM Journal on Scientific Computing*, 14(2):461–469, 1993.
   DOI: 10.1137/0914028.  Flexible GMRES, the generalized minimal residual
   method with a preconditioner that changes between iterations.

.. [FalgoutYang2002] R. D. Falgout and U. M. Yang, "hypre: a library of high
   performance preconditioners", in *Computational Science: ICCS 2002*,
   Lecture Notes in Computer Science 2331, pp. 632–641, Springer, 2002.
   DOI: 10.1007/3-540-47789-6_66.  The library of the algebraic multigrid
   preconditioner BoomerAMG.

.. [DeSterckYangHeys2006] H. De Sterck, U. M. Yang and J. J. Heys, "Reducing
   complexity in parallel algebraic multigrid preconditioners", *SIAM Journal
   on Matrix Analysis and Applications*, 27(4):1019–1039, 2006.
   DOI: 10.1137/040615729.

.. [DeSterckFalgoutNoltingYang2008] H. De Sterck, R. D. Falgout, J. W.
   Nolting and U. M. Yang, "Distance-two interpolation for parallel algebraic
   multigrid", *Numerical Linear Algebra with Applications*, 15(2–3):115–139,
   2008.  DOI: 10.1002/nla.559.

.. [VanekMandelBrezina1996] P. Vaněk, J. Mandel and M. Brezina, "Algebraic
   multigrid by smoothed aggregation for second and fourth order elliptic
   problems", *Computing*, 56(3):179–196, 1996.  DOI: 10.1007/BF02238511.
   The smoothed aggregation of the multigrid preconditioner GAMG of PETSc.

.. [KayLoghinWathen2002] D. Kay, D. Loghin and A. Wathen, "A preconditioner
   for the steady-state Navier-Stokes equations", *SIAM Journal on Scientific
   Computing*, 24(1):237–256, 2002.  DOI: 10.1137/S106482759935808X.  The
   pressure convection-diffusion preconditioner.

.. [ElmanTuminaro2009] H. C. Elman and R. S. Tuminaro, "Boundary conditions
   in approximate commutator preconditioners for the Navier-Stokes
   equations", *Electronic Transactions on Numerical Analysis*, 35:257–280,
   2009.  https://etna.math.kent.edu/vol.35.2009/pp257-280.dir/pp257-280.pdf.
   The form of the pressure convection-diffusion preconditioner, with the
   Robin condition at an inflow.

.. [ElmanHowleShadidShuttleworthTuminaro2006] H. Elman, V. E. Howle, J.
   Shadid, R. Shuttleworth and R. Tuminaro, "Block preconditioners based on
   approximate commutators", *SIAM Journal on Scientific Computing*,
   27(5):1651–1668, 2006.  DOI: 10.1137/040608817.

.. [ElmanHowleShadidSilvesterTuminaro2007] H. Elman, V. E. Howle, J. Shadid,
   D. Silvester and R. Tuminaro, "Least squares preconditioners for
   stabilized discretizations of the Navier-Stokes equations", *SIAM Journal
   on Scientific Computing*, 30(1):290–311, 2007.  DOI: 10.1137/060655742.

.. [EisenstatWalker1996] S. C. Eisenstat and H. F. Walker, "Choosing the
   forcing terms in an inexact Newton method", *SIAM Journal on Scientific
   Computing*, 17(1):16–32, 1996.  DOI: 10.1137/0917003.  The tolerance of
   the iterative linear solves within Newton's method.

.. [Wengert1964] R. E. Wengert, "A simple automatic derivative evaluation
   program", *Communications of the ACM*, 7(8):463–464, 1964.
   DOI: 10.1145/355586.364791.  Forward-mode automatic differentiation, which
   forms the exact Jacobian.

Parallel computing
------------------

.. [KarypisKumar1998] G. Karypis and V. Kumar, "A fast and high quality
   multilevel scheme for partitioning irregular graphs", *SIAM Journal on
   Scientific Computing*, 20(1):359–392, 1998.
   DOI: 10.1137/S1064827595287997.  The multilevel algorithm of the ParMETIS
   partitioner.

.. [Chevalier2008] C. Chevalier and F. Pellegrini, "PT-Scotch: a tool for
   efficient parallel graph ordering", *Parallel Computing*, 34(6–8):318–331,
   2008.  DOI: 10.1016/j.parco.2007.12.001.  The PT-Scotch partitioner.

.. [gslib] *gslib: Gather-scatter library*, release 1.0.9, 2024,
   https://github.com/Nek5000/gslib.  The gather-scatter library of Nek5000
   and nekRS.

.. [Fischer2021] P. Fischer, S. Kerkemeier, M. Min, Y.-H. Lan, M. Phillips,
   T. Rathnayake, E. Merzari, A. Tomboulides, A. Karakus, N. Chalmers and
   T. Warburton, "NekRS, a GPU-accelerated spectral element Navier-Stokes
   solver", arXiv:2104.05829, 2021.  DOI: 10.48550/arXiv.2104.05829.  The
   projection splitting of :doc:`theory/fluids` and the exchange methods of
   gslib.

Time integration
----------------

.. [CrankNicolson1947] J. Crank and P. Nicolson, "A practical method for
   numerical evaluation of solutions of partial differential equations of the
   heat-conduction type", *Mathematical Proceedings of the Cambridge
   Philosophical Society*, 43(1):50–67, 1947.
   DOI: 10.1017/S0305004100023197.  The Crank-Nicolson time scheme.

.. [ChungHulbert1993] J. Chung and G. M. Hulbert, "A time integration algorithm
   for structural dynamics with improved numerical dissipation: the
   generalized-alpha method", *Journal of Applied Mechanics*, 60(2):371–375,
   1993.  DOI: 10.1115/1.2900803.  The generalized-alpha method of the
   transient solid.

.. [RichardsonGaunt1927] L. F. Richardson and J. A. Gaunt, "The deferred
   approach to the limit", *Philosophical Transactions of the Royal Society A*,
   226:299–361, 1927.  DOI: 10.1098/rsta.1927.0008.  The step doubling that
   estimates the error of a time step.

.. [HairerNorsettWanner1993] E. Hairer, S. P. Nørsett and G. Wanner, *Solving
   Ordinary Differential Equations I: Nonstiff Problems*, 2nd revised edition,
   Springer Series in Computational Mathematics 8, Springer, Berlin, 1993,
   ISBN 978-3-540-56670-0.  The step-size controller of the error-controlled
   time stepper.

Adaptivity
----------

.. [ZienkiewiczZhu1987] O. C. Zienkiewicz and J. Z. Zhu, "A simple error
   estimator and adaptive procedure for practical engineering analysis",
   *International Journal for Numerical Methods in Engineering*,
   24(2):337–357, 1987.  DOI: 10.1002/nme.1620240206.  The averaging recovery
   of the gradient in the error indicator.

.. [ZienkiewiczZhu1992a] O. C. Zienkiewicz and J. Z. Zhu, "The superconvergent
   patch recovery and a posteriori error estimates. Part 1: The recovery
   technique", *International Journal for Numerical Methods in Engineering*,
   33(7):1331–1364, 1992.  DOI: 10.1002/nme.1620330702.

.. [ZienkiewiczZhu1992b] O. C. Zienkiewicz and J. Z. Zhu, "The superconvergent
   patch recovery and a posteriori error estimates. Part 2: Error estimates and
   adaptivity", *International Journal for Numerical Methods in Engineering*,
   33(7):1365–1382, 1992.  DOI: 10.1002/nme.1620330703.

.. [Dorfler1996] W. Dörfler, "A convergent adaptive algorithm for Poisson's
   equation", *SIAM Journal on Numerical Analysis*, 33(3):1106–1124, 1996.
   DOI: 10.1137/0733054.  The bulk marking of
   :func:`~dualmesh.mark_by_error_fraction`.

.. [BinevDahmenDeVore2004] P. Binev, W. Dahmen and R. DeVore, "Adaptive
   finite element methods with convergence rates", *Numerische Mathematik*,
   97:219–268, 2004.  The optimal rate of adaptive refinement by newest-vertex
   bisection.

.. [Stevenson2007] R. Stevenson, "Optimality of a standard adaptive finite
   element method", *Foundations of Computational Mathematics*, 7:245–269,
   2007.  The optimal rate of the standard adaptive loop with newest-vertex
   bisection.

.. [FunkenPraetoriusWissgott2011] S. Funken, D. Praetorius and P. Wissgott,
   "Efficient implementation of adaptive P1-FEM in Matlab", *Computational
   Methods in Applied Mathematics*, 11:460–490, 2011.  The edge marking of the
   refinement and the coarsening by newest-vertex bisection.

.. [RivaraLevin1992] M.-C. Rivara and C. Levin, "A 3-D refinement algorithm
   suitable for adaptive and multi-grid techniques", *Communications in
   Applied Numerical Methods*, 8:281–290, 1992.  The longest-edge bisection of
   tetrahedra.

Verification and benchmarks
---------------------------

.. [Roache2002] P. J. Roache, "Code verification by the method of manufactured
   solutions", *Journal of Fluids Engineering*, 124(1):4–10, 2002.
   DOI: 10.1115/1.1436090.  The method of manufactured solutions, which
   verifies the order of accuracy of a code.

.. [SalariKnupp2000] K. Salari and P. Knupp, *Code Verification by the Method
   of Manufactured Solutions*, Sandia National Laboratories report
   SAND2000-1444, 2000.  DOI: 10.2172/759450.

.. [ThomeeWahlbin2008] V. Thomée and L. B. Wahlbin, "On the existence of
   maximum principles in parabolic finite element equations", *Mathematics of
   Computation*, 77(261):11–19, 2008.  DOI: 10.1090/S0025-5718-07-02021-2.  The
   consistent capacity has no maximum principle for a short step, and the
   lumped capacity keeps it under conditions on the angles of the mesh.

.. [TimoshenkoGoodier1970] S. P. Timoshenko and J. N. Goodier, *Theory of
   Elasticity*, 3rd edition, McGraw-Hill, New York, NY, 1970.  The exact
   solutions of the thick cylinder and of the thermal stresses in a long
   cylinder.

.. [Ghia1982] U. Ghia, K. N. Ghia and C. T. Shin, "High-Re solutions for
   incompressible flow using the Navier-Stokes equations and a multigrid
   method", *Journal of Computational Physics*, 48(3):387–411, 1982.
   DOI: 10.1016/0021-9991(82)90058-4.  The centerline velocities of the
   lid-driven cavity in :doc:`verification`.

.. [Erturk2005] E. Erturk, T. C. Corke and C. Gokcol, "Numerical solutions of
   2-D steady incompressible driven cavity flow at high Reynolds numbers",
   *International Journal for Numerical Methods in Fluids*, 48:747–774, 2005.
   DOI: 10.1002/fld.953.  The centerline velocities of
   :doc:`benchmarks/cavity`.

.. [SchaeferTurek1996] M. Schäfer, S. Turek, F. Durst, E. Krause and
   R. Rannacher, "Benchmark computations of laminar flow around a cylinder", in
   E. H. Hirschel (ed.), *Flow Simulation with High-Performance Computers II*,
   Notes on Numerical Fluid Mechanics 52, pp. 547–566, Vieweg, 1996.
   DOI: 10.1007/978-3-322-89849-4_39.  The benchmarks 2D-1 and 2D-2 of
   :doc:`benchmarks/dfg_steady` and :doc:`benchmarks/dfg_unsteady`, with the
   reference values of the `FEATFLOW flow benchmark pages
   <https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow.html>`__.

.. [Kovasznay1948] L. I. G. Kovasznay, "Laminar flow behind a two-dimensional
   grid", *Mathematical Proceedings of the Cambridge Philosophical Society*,
   44:58–62, 1948.  DOI: 10.1017/S0305004100023999.  The exact solution of
   :doc:`benchmarks/kovasznay`.

.. [DeVahlDavis1983] G. de Vahl Davis, "Natural convection of air in a square
   cavity: a bench mark numerical solution", *International Journal for
   Numerical Methods in Fluids*, 3(3):249–264, 1983.
   DOI: 10.1002/fld.1650030305.  The natural convection benchmark of
   ``examples/natural_convection.py``.

.. [COMSOLWrench] COMSOL AB, "Stresses and Strains in a Wrench", COMSOL
   Multiphysics 6.1 Application Library, model ``wrench``,
   https://doc.comsol.com/6.1/doc/com.comsol.help.models.mph.wrench/wrench.html.
   The load case of :doc:`tutorials/wrench`.

.. [OpenFOAM1998] H. G. Weller, G. Tabor, H. Jasak and C. Fureby, "A tensorial
   approach to computational continuum mechanics using object-oriented
   techniques", *Computers in Physics*, 12(6):620–631, 1998.
   DOI: 10.1063/1.168744.  The code of the cross-checks in
   :ref:`verification-openfoam`.

Mesh files
----------

.. [Gmsh2009] C. Geuzaine and J.-F. Remacle, "Gmsh: a three-dimensional finite
   element mesh generator with built-in pre- and post-processing facilities",
   *International Journal for Numerical Methods in Engineering*,
   79(11):1309–1331, 2009.  DOI: 10.1002/nme.2579.  A mesh generator whose
   files the code reads through meshio.

.. [meshio] N. Schlömer, *meshio: Tools for mesh files*, Zenodo.
   DOI: 10.5281/zenodo.1173115.  The library that reads and writes the mesh
   files.

Neutronics
----------

.. [Lee2025] J. C. Lee, *Nuclear Reactor Physics and Engineering*, 2nd
   edition, John Wiley & Sons, Hoboken, NJ, 2025.
   DOI: 10.1002/9781394283583.  The multigroup diffusion equations and the
   bucklings of :doc:`tutorials/criticality`.

.. [ANL7416] Argonne Code Center, *Argonne Code Center: Benchmark Problem
   Book*, ANL-7416, Supplement 2, Argonne National Laboratory, Argonne, IL,
   1977.  DOI: 10.2172/5037820.  Problem 11-A2, the two-dimensional benchmark
   of the International Atomic Energy Agency (IAEA) for a pressurized water
   reactor (PWR), with its reference solution 11-A2-1.
