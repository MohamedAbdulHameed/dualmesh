.. _references:

References
==========

Every method that dualmesh implements is cited here to the work that
introduced it, not to a textbook that repeats it and not to another code that
uses it.  Where the implementation follows a design decision taken by another
project — the object model of MOOSE, the way OVITO organises its manual — the
project is named as the influence and the underlying method is cited
separately to its own source.

Each entry below was checked against the publisher's record: a DOI landing
page or the publisher-deposited metadata held by Crossref, the project's own
citation page for the software entries, or a library catalogue record for the
books.  Where two sources disagreed, the disagreement is noted in the entry.
The checks were made in September 2026.

The method
----------

.. [Reddy2024] J. N. Reddy, *Computational Methods in Engineering: Finite
   Difference, Finite Volume, Finite Element, and Dual Mesh Control Domain
   Methods*, 1st edition, CRC Press, Boca Raton, FL, 2024, 594 pp.,
   ISBN 978-1-032-46637-8.  DOI: 10.1201/9781003382812.  The dual mesh control
   domain method is Chapter 5; the finite volume methods of :doc:`theory/index` are
   Chapter 3.  Every verification case in :doc:`verification` that is marked
   "book" comes from this text.

.. [Reddy2019a] J. N. Reddy, "A dual mesh finite domain method for the
   numerical solution of differential equations", *International Journal for
   Computational Methods in Engineering Science and Mechanics*, 20(3):212–228,
   2019.  DOI: 10.1080/15502287.2019.1610987.  The paper that introduces the
   method.

.. [ReddyKimMartinez2020] J. N. Reddy, N. Kim and M. Martinez, "A dual mesh
   control domain method for the solution of nonlinear Poisson's equation and
   the Navier–Stokes equations for incompressible fluids", *Physics of
   Fluids*, 32(9):093608, 2020.  DOI: 10.1063/5.0026274.  The first nonlinear
   application, and the first paper to use the name *control* domain rather
   than *finite* domain.

.. [ReddyNampally2020] J. N. Reddy and P. Nampally, "A dual mesh finite domain
   method for the analysis of functionally graded beams", *Composite
   Structures*, 251:112648, 2020.
   DOI: 10.1016/j.compstruct.2020.112648.

.. [ReddyNampallySrinivasa2020] J. N. Reddy, P. Nampally and A. R. Srinivasa,
   "Nonlinear analysis of functionally graded beams using the dual mesh finite
   domain method and the finite element method", *International Journal of
   Non-Linear Mechanics*, 127:103575, 2020.
   DOI: 10.1016/j.ijnonlinmec.2020.103575.  The source of the von Kármán beam
   formulation used by the beams of the solid mechanics module.

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
   DOI: 10.1016/j.compstruct.2020.113153.  The source of the von Kármán
   circular plate formulation.

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
   427:117044, 2024.  DOI: 10.1016/j.cma.2024.117044.  The paper that takes the
   method to unstructured meshes and to Newton linearisation, which is what
   dualmesh implements.

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
   DOI: 10.1063/5.0259696.  The transient formulation for the Navier–Stokes
   equations.

.. [Heblekar2026p] T. Y. Heblekar, J. N. Reddy and A. R. Srinivasa,
   "p-Refinement in the dual-mesh control domain method", *Computing in Science
   & Engineering*, 28(1):64–73, 2026.  DOI: 10.1109/MCSE.2025.3641847.

.. [Heblekar2026spectral] T. Y. Heblekar, J. N. Reddy and A. R. Srinivasa, "A
   spectral dual mesh control domain framework for nonlinear fluid flow and
   coupled multi-field problems", *International Journal of Non-Linear
   Mechanics*, 186:105350, 2026.  DOI: 10.1016/j.ijnonlinmec.2026.105350.

.. [Heblekar2026nonlinear] T. Heblekar, J. N. Reddy and A. Srinivasa,
   "Dual-mesh control-domain analysis of nonlinear problems in mechanics",
   *Theoretical and Applied Mechanics* (Belgrade), 2026, online first.
   DOI: 10.2298/TAM260601007H.  Volume, issue and pages were not assigned at
   the time of checking.

Related control volume methods
------------------------------

The dual mesh control domain method belongs to a family of schemes that
integrate a balance law over node-centred control volumes and interpolate with
finite element shape functions.  The family is older than the name, and the
references below are its primary sources.  What distinguishes the dual mesh
control domain method from them is stated in :doc:`theory/index`.

.. [BaligaPatankar1980] B. R. Baliga and S. V. Patankar, "A new finite-element
   formulation for convection-diffusion problems", *Numerical Heat Transfer*,
   3(4):393–409, 1980.  DOI: 10.1080/01495728008961767.

.. [BaligaPatankar1983] B. R. Baliga and S. V. Patankar, "A control volume
   finite-element method for two-dimensional fluid flow and heat transfer",
   *Numerical Heat Transfer*, 6(3):245–261, 1983.
   DOI: 10.1080/01495728308963086.  The paper that names the control volume
   finite element method.

.. [BankRose1987] R. E. Bank and D. J. Rose, "Some error estimates for the box
   method", *SIAM Journal on Numerical Analysis*, 24(4):777–787, 1987.
   DOI: 10.1137/0724050.  The convergence analysis of the vertex-centred
   scheme on a median dual mesh.

.. [Hackbusch1989] W. Hackbusch, "On first and second order box schemes",
   *Computing*, 41(4):277–296, 1989.  DOI: 10.1007/BF02241218.

Finite volume methods
---------------------

.. [Patankar1980] S. V. Patankar, *Numerical Heat Transfer and Fluid Flow*,
   Hemisphere Publishing, Washington, DC, 1980.  The cell-centred finite volume
   method in the form Chapter 3 of [Reddy2024]_ presents it.

.. [Jasak1996] H. Jasak, *Error Analysis and Estimation for the Finite Volume
   Method with Applications to Fluid Flows*, PhD thesis, Imperial College
   London, 1996.  The non-orthogonal correction and the deferred correction
   treatment of it, in the form used by the two finite volume discretisations
   of dualmesh.

.. [DemirdzicMuzaferija1995] I. Demirdžić and S. Muzaferija, "Numerical method
   for coupled fluid flow, heat transfer and stress analysis using unstructured
   moving meshes with cells of arbitrary topology", *Computer Methods in
   Applied Mechanics and Engineering*, 125(1–4):235–255, 1995.
   DOI: 10.1016/0045-7825(95)00800-G.

.. [BarthJespersen1989] T. J. Barth and D. C. Jespersen, "The design and
   application of upwind schemes on unstructured meshes", AIAA Paper 89-0366,
   27th Aerospace Sciences Meeting, Reno, NV, 9–12 January 1989.
   DOI: 10.2514/6.1989-366.  The least-squares gradient reconstruction used by
   the cell-centred method.

.. [Barth1993] T. J. Barth, "Recent developments in high order K-exact
   reconstruction on unstructured meshes", AIAA Paper 93-0668, 31st Aerospace
   Sciences Meeting and Exhibit, Reno, NV, 11–14 January 1993.
   DOI: 10.2514/6.1993-668.

Finite elements
---------------

.. [Irons1966] B. M. Irons, "Engineering applications of numerical integration
   in stiffness methods", *AIAA Journal*, 4(11):2035–2037, 1966.
   DOI: 10.2514/3.3836.  The isoparametric map, which dualmesh uses for the
   geometry of every element and of every control domain patch.

.. [Bedrosian1992] G. Bedrosian, "Shape functions and integration formulas for
   three-dimensional finite element analysis", *International Journal for
   Numerical Methods in Engineering*, 35(1):95–108, 1992.
   DOI: 10.1002/nme.1620350106.  The rational basis of the pyramid element.

.. [Duffy1982] M. G. Duffy, "Quadrature over a pyramid or cube of integrands
   with a singularity at a vertex", *SIAM Journal on Numerical Analysis*,
   19(6):1260–1262, 1982.  DOI: 10.1137/0719090.  The collapsed-coordinate map
   by which a triangle, a tetrahedron, a prism or a pyramid is integrated as a
   square or a cube with some corners merged.

.. [Dunavant1985] D. A. Dunavant, "High degree efficient symmetrical Gaussian
   quadrature rules for the triangle", *International Journal for Numerical
   Methods in Engineering*, 21(6):1129–1148, 1985.
   DOI: 10.1002/nme.1620210612.  The symmetric triangle rules of degree 4 and
   5 used by the finite element method on triangles and prisms; the library
   recomputed them to 40 digits.

.. [DouglasDupont1974] J. Douglas, Jr. and T. Dupont, "Galerkin approximations
   for the two point boundary problem using continuous, piecewise polynomial
   spaces", *Numerische Mathematik*, 22(2):99–109, 1974.
   DOI: 10.1007/BF01436724.  The nodal superconvergence of the Galerkin method:
   the finite element solution of a two-point boundary value problem is more
   accurate at the nodes than anywhere else, which is why quadratic elements
   give fourth-order nodal values.  (The publisher's deposited metadata lists
   only the first author; both are confirmed by the zbMATH and EUDML records of
   the same article.)

.. [Aubin1967] J.-P. Aubin, "Behavior of the error of the approximate
   solutions of boundary value problems for linear elliptic operators by
   Galerkin's and finite difference methods", *Annali della Scuola Normale
   Superiore di Pisa, Classe di Scienze*, Serie 3, 21(4):599–637, 1967.  With
   [Nitsche1968]_, the duality argument that gives the Galerkin method one more
   order of accuracy in :math:`L^2` than in the energy norm.

.. [Nitsche1968] J. Nitsche, "Ein Kriterium für die Quasi-Optimalität des
   Ritzschen Verfahrens", *Numerische Mathematik*, 11(4):346–348, 1968.
   DOI: 10.1007/BF02166687.

.. [Roache2002] P. J. Roache, "Code verification by the method of manufactured
   solutions", *Journal of Fluids Engineering*, 124(1):4–10, 2002.
   DOI: 10.1115/1.1436090.  The method of manufactured solutions as a
   procedure for verifying the order of accuracy of a code.

.. [SalariKnupp2000] K. Salari and P. Knupp, *Code Verification by the Method
   of Manufactured Solutions*, Sandia National Laboratories report
   SAND2000-1444, 2000.  DOI: 10.2172/759450.

.. [Barlow1976] J. Barlow, "Optimal stress locations in finite element models",
   *International Journal for Numerical Methods in Engineering*,
   10(2):243–251, 1976.  DOI: 10.1002/nme.1620100202.  The Gauss points of an
   element are where the derivative of a finite element solution is most
   accurate.  This is the result that explains why the dual mesh control domain
   method gains no order from quadratic elements: its control domain interfaces
   are at the midpoints between nodes, not at the Gauss points.

.. [ZienkiewiczTaylorToo1971] O. C. Zienkiewicz, R. L. Taylor and J. M. Too,
   "Reduced integration technique in general analysis of plates and shells",
   *International Journal for Numerical Methods in Engineering*, 3(2):275–290,
   1971.  DOI: 10.1002/nme.1620030211.

.. [HughesCohenHaroun1978] T. J. R. Hughes, M. Cohen and M. Haroun, "Reduced
   and selective integration techniques in the finite element analysis of
   plates", *Nuclear Engineering and Design*, 46(1):203–222, 1978.
   DOI: 10.1016/0029-5493(78)90184-X.  The source of the
   ``reduced_integration`` parameter.

.. [MalkusHughes1978] D. S. Malkus and T. J. R. Hughes, "Mixed finite element
   methods — reduced and selective integration techniques: a unification of
   concepts", *Computer Methods in Applied Mechanics and Engineering*,
   15(1):63–81, 1978.  DOI: 10.1016/0045-7825(78)90005-1.  The equivalence
   between selective reduced integration and a mixed formulation, which is why
   reduced integration of the penalty and shear terms is legitimate rather than
   a trick.

.. [DeVahlDavis1983] G. de Vahl Davis, "Natural convection of air in a square
   cavity: a bench mark numerical solution", *International Journal for
   Numerical Methods in Fluids*, 3(3):249–264, 1983.
   DOI: 10.1002/fld.1650030305.  The benchmark of the coupled flow and heat
   transfer test, ``examples/natural_convection.py``.

.. [HughesLiuBrooks1979] T. J. R. Hughes, W. K. Liu and A. Brooks, "Finite
   element analysis of incompressible viscous flows by the penalty function
   formulation", *Journal of Computational Physics*, 30(1):1–60, 1979.
   DOI: 10.1016/0021-9991(79)90086-X.  The penalty formulation used by the
   fluids module.

Solvers
-------

.. [Wengert1964] R. E. Wengert, "A simple automatic derivative evaluation
   program", *Communications of the ACM*, 7(8):463–464, 1964.
   DOI: 10.1145/355586.364791.  Forward-mode automatic differentiation, which
   is how dualmesh forms exact Jacobians.

.. [GriewankWalther2008] A. Griewank and A. Walther, *Evaluating Derivatives:
   Principles and Techniques of Algorithmic Differentiation*, 2nd edition,
   SIAM, Philadelphia, PA, 2008, ISBN 978-0-89871-659-7.
   DOI: 10.1137/1.9780898717761.

.. [HestenesStiefel1952] M. R. Hestenes and E. Stiefel, "Methods of conjugate
   gradients for solving linear systems", *Journal of Research of the National
   Bureau of Standards*, 49(6):409–436, 1952 (Research Paper 2379).  Several
   secondary sources give the last page as 435; the scanned article ends on
   page 436.

.. [VanDerVorst1992] H. A. van der Vorst, "Bi-CGSTAB: a fast and smoothly
   converging variant of Bi-CG for the solution of nonsymmetric linear
   systems", *SIAM Journal on Scientific and Statistical Computing*,
   13(2):631–644, 1992.  DOI: 10.1137/0913035.

.. [Saad1994] Y. Saad, "ILUT: a dual threshold incomplete LU factorization",
   *Numerical Linear Algebra with Applications*, 1(4):387–402, 1994.
   DOI: 10.1002/nla.1680010405.  The threshold preconditioner offered as
   ``preconditioner = "ilut"``.

.. [Saad2003] Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd
   edition, SIAM, Philadelphia, 2003.  ISBN 978-0-89871-534-7.
   DOI: 10.1137/1.9780898718003.  Section 10.3.2, "Zero fill-in ILU
   (ILU(0))", is the default preconditioner of the serial Krylov solvers and
   the default subdomain solver of the Schwarz preconditioners.

.. [George1973] A. George, "Nested dissection of a regular finite element
   mesh", *SIAM Journal on Numerical Analysis*, 10(2):345–363, 1973.
   DOI: 10.1137/0710032.  With [LiptonRoseTarjan1979]_, the source of the
   fill and operation counts of sparse direct factorisation in two and three
   dimensions that decide when ``linear_solver = "automatic"`` factorises.

.. [LiptonRoseTarjan1979] R. J. Lipton, D. J. Rose and R. E. Tarjan,
   "Generalized nested dissection", *SIAM Journal on Numerical Analysis*,
   16(2):346–358, 1979.  DOI: 10.1137/0716027.

.. [Nicolaides1987] R. A. Nicolaides, "Deflation of conjugate gradients with
   applications to boundary value problems", *SIAM Journal on Numerical
   Analysis*, 24(2):355–365, 1987.  DOI: 10.1137/0724027.  The coarse space
   of one constant per subdomain used by the two-level Schwarz
   preconditioner.

.. [Tang2009] J. M. Tang, R. Nabben, C. Vuik and Y. A. Erlangga, "Comparison
   of two-level preconditioners derived from deflation, domain decomposition
   and multigrid methods", *Journal of Scientific Computing*, 39(3):340–370,
   2009.  DOI: 10.1007/s10915-009-9272-6.  Their operator A-DEF1,
   :math:`M^{-1}P + Q`, is how the two levels of the Schwarz preconditioner
   are combined.

.. [Demmel1999] J. W. Demmel, S. C. Eisenstat, J. R. Gilbert, X. S. Li and
   J. W. H. Liu, "A supernodal approach to sparse partial pivoting", *SIAM
   Journal on Matrix Analysis and Applications*, 20(3):720–755, 1999.
   DOI: 10.1137/S0895479895291765.  The algorithm behind the direct solver:
   Eigen's own documentation states that ``SparseLU`` uses the main techniques
   of the sequential SuperLU package, which is this paper.

.. [CaiSarkis1999] X.-C. Cai and M. Sarkis, "A restricted additive Schwarz
   preconditioner for general sparse linear systems", *SIAM Journal on
   Scientific Computing*, 21(2):792–797, 1999.
   DOI: 10.1137/S106482759732678X.  The subdomain preconditioner of the
   distributed solver.

.. [DryjaWidlund1994] M. Dryja and O. B. Widlund, "Domain decomposition
   algorithms with small overlap", *SIAM Journal on Scientific Computing*,
   15(3):604–620, 1994.  DOI: 10.1137/0915040.  The two-level additive Schwarz
   method with a coarse space.  The idea is older — it appears in a 1987
   Courant Institute technical report by the same authors — but that report has
   no DOI and no primary record that could be checked, so the journal paper is
   cited instead.

.. [ToselliWidlund2005] A. Toselli and O. Widlund, *Domain Decomposition
   Methods — Algorithms and Theory*, Springer Series in Computational
   Mathematics, volume 34, Springer, Berlin, 2005, ISBN 978-3-540-20696-5.
   DOI: 10.1007/b137868.  The analysis that explains why a coarse level is
   needed for the iteration count to stay bounded as ranks are added.

.. [KarypisKumar1998] G. Karypis and V. Kumar, "A fast and high quality
   multilevel scheme for partitioning irregular graphs", *SIAM Journal on
   Scientific Computing*, 20(1):359–392, 1998.
   DOI: 10.1137/S1064827595287997.  METIS, used to partition a mesh when it is
   available at build time.

.. [BergerBokhari1987] M. J. Berger and S. H. Bokhari, "A partitioning strategy
   for nonuniform problems on multiprocessors", *IEEE Transactions on
   Computers*, C-36(5):570–580, 1987.  DOI: 10.1109/TC.1987.1676942.  Recursive
   coordinate bisection, the built-in partitioner used when METIS is absent.

Time integration and adaptivity
-------------------------------

.. [CrankNicolson1947] J. Crank and P. Nicolson, "A practical method for
   numerical evaluation of solutions of partial differential equations of the
   heat-conduction type", *Proceedings of the Cambridge Philosophical Society*,
   43(1):50–67, 1947.  DOI: 10.1017/S0305004100023197.  The journal has since
   been renamed *Mathematical Proceedings of the Cambridge Philosophical
   Society*, under which name the publisher's record lists it.

.. [Richardson1911] L. F. Richardson, "The approximate arithmetical solution by
   finite differences of physical problems involving differential equations,
   with an application to the stresses in a masonry dam", *Philosophical
   Transactions of the Royal Society A*, 210:307–357, 1911.
   DOI: 10.1098/rsta.1911.0009.

.. [RichardsonGaunt1927] L. F. Richardson and J. A. Gaunt, "The deferred
   approach to the limit", *Philosophical Transactions of the Royal Society A*,
   226:299–361, 1927.  DOI: 10.1098/rsta.1927.0008.  The extrapolation that the
   error-controlled time stepper uses to estimate the error of a step by
   comparing one step with two half steps.  The phrase "deferred approach to
   the limit" belongs to this 1927 paper with Gaunt, not to the 1911 paper,
   which is often miscited for it.

.. [HairerNorsettWanner1993] E. Hairer, S. P. Nørsett and G. Wanner, *Solving
   Ordinary Differential Equations I: Nonstiff Problems*, 2nd revised edition,
   Springer Series in Computational Mathematics, volume 8, Springer, Berlin,
   1993, ISBN 978-3-540-56670-0.  The step-size controller of the
   error-controlled stepper follows the standard form given here.

.. [ZienkiewiczZhu1987] O. C. Zienkiewicz and J. Z. Zhu, "A simple error
   estimator and adaptive procedure for practical engineering analysis",
   *International Journal for Numerical Methods in Engineering*,
   24(2):337–357, 1987.  DOI: 10.1002/nme.1620240206.  The gradient recovery
   error indicator.  (The word "engineering" is misspelt in the publisher's
   deposited title; it is given correctly here.)

.. [ZienkiewiczZhu1992a] O. C. Zienkiewicz and J. Z. Zhu, "The superconvergent
   patch recovery and a posteriori error estimates. Part 1: The recovery
   technique", *International Journal for Numerical Methods in Engineering*,
   33(7):1331–1364, 1992.  DOI: 10.1002/nme.1620330702.

.. [ZienkiewiczZhu1992b] O. C. Zienkiewicz and J. Z. Zhu, "The superconvergent
   patch recovery and a posteriori error estimates. Part 2: Error estimates and
   adaptivity", *International Journal for Numerical Methods in Engineering*,
   33(7):1365–1382, 1992.  DOI: 10.1002/nme.1620330703.  dualmesh implements
   the simpler averaging recovery of [ZienkiewiczZhu1987]_, not the patch
   recovery of these two papers; they are cited because they are the reference
   a reader looking for a better recovery should go to.

.. [Dorfler1996] W. Dörfler, "A convergent adaptive algorithm for Poisson's
   equation", *SIAM Journal on Numerical Analysis*, 33(3):1106–1124, 1996.
   DOI: 10.1137/0733054.  Bulk marking, implemented as
   :func:`~dualmesh.mark_by_error_fraction`.

.. [Rivara1984] M.-C. Rivara, "Algorithms for refining triangular grids
   suitable for adaptive and multigrid techniques", *International Journal for
   Numerical Methods in Engineering*, 20(4):745–756, 1984.
   DOI: 10.1002/nme.1620200412.  Longest-edge bisection, the conforming
   refinement used by :func:`~dualmesh.refine_marked`.

.. [Rivara1991] M.-C. Rivara, "Local modification of meshes for adaptive and/or
   multigrid finite-element methods", *Journal of Computational and Applied
   Mathematics*, 36(1):79–89, 1991.  DOI: 10.1016/0377-0427(91)90227-B.  The
   propagation-to-conformity form of the algorithm, which is the form
   implemented.

Physics and verification data
-----------------------------

.. [Reddy2019b] J. N. Reddy, *Introduction to the Finite Element Method*, 4th
   edition, McGraw-Hill Education, New York, NY, 2019,
   ISBN 978-1-259-86190-1.  The third edition and earlier are titled *An
   Introduction to the Finite Element Method*; the fourth dropped the article.

.. [ReddyBeams2022] J. N. Reddy, *Theories and Analyses of Beams and
   Axisymmetric Circular Plates*, CRC Press, Boca Raton, FL, 2022,
   ISBN 978-1-032-14739-0.  DOI: 10.1201/9781003240846.  The beam and circular
   plate theories used by the solid mechanics module.

.. [ReddyPlates2007] J. N. Reddy, *Theory and Analysis of Elastic Plates and
   Shells*, 2nd edition, CRC Press, Boca Raton, FL, 2007,
   ISBN 978-0-8493-8415-8.

.. [ReddyGartling2010] J. N. Reddy and D. K. Gartling, *The Finite Element
   Method in Heat Transfer and Fluid Dynamics*, 3rd edition, CRC Press, Boca
   Raton, FL, 2010, ISBN 978-1-4200-8598-3.

.. [Reddy2000] J. N. Reddy, "Analysis of functionally graded plates",
   *International Journal for Numerical Methods in Engineering*,
   47(1–3):663–684, 2000.
   DOI: 10.1002/(SICI)1097-0207(20000110/30)47:1/3<663::AID-NME787>3.0.CO;2-8.
   The power-law through-thickness variation used by the functionally graded
   material module.

.. [Ghia1982] U. Ghia, K. N. Ghia and C. T. Shin, "High-Re solutions for
   incompressible flow using the Navier-Stokes equations and a multigrid
   method", *Journal of Computational Physics*, 48(3):387–411, 1982.
   DOI: 10.1016/0021-9991(82)90058-4.  The lid-driven cavity data used in
   :doc:`verification`.

Software that dualmesh builds on, or learns from
------------------------------------------------

.. [Eigen] G. Guennebaud, B. Jacob and others, *Eigen*, 2010,
   https://libeigen.gitlab.io.  The linear algebra library.  The project has
   moved from ``eigen.tuxfamily.org``, which now redirects; the citation above
   is the one the project's own BibTeX page currently asks for.

.. [pybind11] W. Jakob, J. Rhinelander and D. Moldovan, *pybind11 — Seamless
   operability between C++11 and Python*, 2017,
   https://github.com/pybind/pybind11.  The citation the project asks for in
   its documentation FAQ.

.. [meshio] N. Schlömer, *meshio: Tools for mesh files*, Zenodo.
   DOI: 10.5281/zenodo.1173115.  Mesh file reading and writing.  The DOI is the
   concept DOI naming all versions, which is what the project's ``CITATION.cff``
   specifies.

.. [MOOSE2025] L. Harbour, G. Giudicelli, A. D. Lindsay, P. German, J. Hansel,
   C. Icenhour, M. Li, J. M. Miller, R. H. Stogner, P. Behne, D. Yankura,
   Z. M. Prince, C. DeChant, D. Schwen, B. W. Spencer, M. Tano, N. Choi,
   Y. Wang, M. Nezdyur, Y. Miao, T. Hu, S. Kumar, C. Matthews, B. Langley,
   N. Nobre, A. Blair, C. MacMackin, H. Bergallo Rocha, E. Palmer, J. Carter,
   J. Meier, A. E. Slaughter, D. Andrš, R. W. Carlsen, F. Kong, D. R. Gaston
   and C. J. Permann, "4.0 MOOSE: enabling massively parallel multiphysics
   simulation", *SoftwareX*, 31:102264, 2025.
   DOI: 10.1016/j.softx.2025.102264.  The current reference for MOOSE, and
   the one the project's citation page asks for.  dualmesh follows MOOSE's
   object model -- named, registered kernels, boundary conditions and
   materials with validated, self-documenting parameters, assembled into one
   monolithic, fully coupled system with an automatically differentiated
   Jacobian.  No MOOSE source code is used; the influence is on the design,
   and every algorithm that MOOSE also implements is cited above to its own
   source.

.. [MOOSE2020] C. J. Permann, D. R. Gaston, D. Andrš, R. W. Carlsen, F. Kong,
   A. D. Lindsay, J. M. Miller, J. W. Peterson, A. E. Slaughter, R. H. Stogner
   and R. C. Martineau, "MOOSE: enabling massively parallel multiphysics
   simulation", *SoftwareX*, 11:100430, 2020.
   DOI: 10.1016/j.softx.2020.100430.  The previous framework paper, which
   [MOOSE2025]_ supersedes.

.. [MOOSE2009] D. Gaston, C. Newman, G. Hansen and D. Lebrun-Grandié, "MOOSE: a
   parallel computational framework for coupled systems of nonlinear
   equations", *Nuclear Engineering and Design*, 239(10):1768–1778, 2009.
   DOI: 10.1016/j.nucengdes.2009.05.021.

.. [libMesh2006] B. S. Kirk, J. W. Peterson, R. H. Stogner and G. F. Carey,
   "libMesh: a C++ library for parallel adaptive mesh refinement/coarsening
   simulations", *Engineering with Computers*, 22(3–4):237–254, 2006.
   DOI: 10.1007/s00366-006-0049-3.  The finite element library underneath
   MOOSE.  dualmesh does not use it; the reference is given because a reader
   comparing the two frameworks will want it.

.. [OpenFOAM1998] H. G. Weller, G. Tabor, H. Jasak and C. Fureby, "A tensorial
   approach to computational continuum mechanics using object-oriented
   techniques", *Computers in Physics*, 12(6):620–631, 1998.
   DOI: 10.1063/1.168744.  The code used for the independent cross-checks in
   :doc:`openfoam`.

.. [OVITO2010] A. Stukowski, "Visualization and analysis of atomistic
   simulation data with OVITO — the Open Visualization Tool", *Modelling and
   Simulation in Materials Science and Engineering*, 18(1):015012, 2010.
   DOI: 10.1088/0965-0393/18/1/015012.  Named as an influence on the
   organisation of this manual, in which every keyword has a page of its own
   that says what the keyword means, what it defaults to, and what happens when
   it is changed.

.. [Gmsh2009] C. Geuzaine and J.-F. Remacle, "Gmsh: a three-dimensional finite
   element mesh generator with built-in pre- and post-processing facilities",
   *International Journal for Numerical Methods in Engineering*,
   79(11):1309–1331, 2009.  DOI: 10.1002/nme.2579.  One of the mesh generators
   whose output dualmesh reads, through meshio.
