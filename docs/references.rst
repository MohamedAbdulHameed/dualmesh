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

.. [HughesFrancaBalestra1986] T. J. R. Hughes, L. P. Franca and M. Balestra, "A new
   finite element formulation for computational fluid dynamics: V.
   Circumventing the Babuška-Brezzi condition: a stable Petrov-Galerkin
   formulation of the Stokes problem accommodating equal-order interpolations",
   *Computer Methods in Applied Mechanics and Engineering*, 59(1):85–99, 1986.
   DOI: 10.1016/0045-7825(86)90025-3.  The pressure stabilisation of
   ``mass_conservation``.

.. [TaylorHood1973] C. Taylor and P. Hood, "A numerical solution of the
   Navier-Stokes equations using the finite element technique", *Computers &
   Fluids*, 1(1):73–100, 1973.  DOI: 10.1016/0045-7930(73)90027-3.  The
   element with quadratic velocity and linear pressure that carries their
   names.  Metadata checked against Crossref; the paper itself was not
   re-read for this documentation.

.. [BoffiBrezziFortin2013] D. Boffi, F. Brezzi and M. Fortin, *Mixed Finite
   Element Methods and Applications*, Springer Series in Computational
   Mathematics 44, Springer, Berlin, 2013.  DOI: 10.1007/978-3-642-36519-5.
   The standard reference for the inf-sup condition and for the stability and
   error estimates of the Taylor-Hood family.  Metadata checked against
   Crossref; the book was not re-read for this documentation, and the orders
   quoted from it in :doc:`theory/heat_and_fluids` are confirmed by the
   measurements reported there rather than taken on trust.

.. [ChapelleBathe1993] D. Chapelle and K. J. Bathe, "The inf-sup test",
   *Computers & Structures*, 47(4–5):537–545, 1993.
   DOI: 10.1016/0045-7949(93)90340-J.  The numerical inf-sup test: the discrete
   inf-sup constant computed as the smallest non-zero eigenvalue of a
   generalised eigenproblem on a sequence of refined meshes.  Metadata checked
   against Crossref; the paper was not read, and the test in
   ``tests/python/test_taylor_hood.py`` implements the eigenproblem as it is
   stated in :doc:`theory/heat_and_fluids`.

.. [Tezduyar1991] T. E. Tezduyar, "Stabilized finite element formulations for
   incompressible flow computations", *Advances in Applied Mechanics*,
   28:1–44, 1991.  DOI: 10.1016/S0065-2156(08)70153-4.  The stabilisation
   parameter :math:`\tau`.

.. [TezduyarMittalRayShih1992] T. E. Tezduyar, S. Mittal, S. E. Ray and R. Shih,
   "Incompressible flow computations with stabilized bilinear and linear
   equal-order-interpolation velocity-pressure elements", *Computer Methods in
   Applied Mechanics and Engineering*, 95(2):221–242, 1992.
   DOI: 10.1016/0045-7825(92)90141-6.

.. [BrooksHughes1982] A. N. Brooks and T. J. R. Hughes, "Streamline
   upwind/Petrov-Galerkin formulations for convection dominated flows with
   particular emphasis on the incompressible Navier-Stokes equations",
   *Computer Methods in Applied Mechanics and Engineering*, 32(1–3):199–259,
   1982.  DOI: 10.1016/0045-7825(82)90071-8.  The streamline stabilisation of
   ``momentum_stabilization``.

.. [RhieChow1983] C. M. Rhie and W. L. Chow, "Numerical study of the turbulent
   flow past an airfoil with trailing edge separation", *AIAA Journal*,
   21(11):1525–1532, 1983.  DOI: 10.2514/3.8284.  Momentum interpolation for
   colocated finite volumes.

.. [BrezziPitkaranta1984] F. Brezzi and J. Pitkäranta, "On the stabilization of
   finite element approximations of the Stokes equations", in *Efficient
   Solutions of Elliptic Systems*, Notes on Numerical Fluid Mechanics 10,
   pp. 11–19, Vieweg, 1984.  DOI: 10.1007/978-3-663-14169-3_2.

Solvers
-------

.. [PETSc2023] S. Balay, S. Abhyankar, M. F. Adams, S. Benson, J. Brown,
   P. Brune, K. Buschelman, E. Constantinescu, L. Dalcin, A. Dener,
   V. Eijkhout, J. Faibussowitsch, W. D. Gropp, V. Hapla, T. Isaac,
   P. Jolivet, D. Karpeev, D. Kaushik, M. G. Knepley, F. Kong, S. Kruger,
   D. A. May, L. Curfman McInnes, R. Tran Mills, L. Mitchell, T. Munson,
   J. E. Roman, K. Rupp, P. Sanan, J. Sarich, B. F. Smith, S. Zampini,
   H. Zhang, H. Zhang and J. Zhang, *PETSc/TAO Users Manual*, Revision 3.19,
   Argonne National Laboratory report ANL-21/39, 2023.
   DOI: 10.2172/1968587.  The library behind ``linear_solver="petsc"``.

.. [FalgoutYang2002] R. D. Falgout and U. M. Yang, "hypre: a library of high
   performance preconditioners", in *Computational Science — ICCS 2002*,
   Lecture Notes in Computer Science 2331, pp. 632–641, Springer, 2002.
   DOI: 10.1007/3-540-47789-6_66.

.. [Amestoy2001] P. R. Amestoy, I. S. Duff, J.-Y. L'Excellent and J. Koster,
   "A fully asynchronous multifrontal solver using distributed dynamic
   scheduling", *SIAM Journal on Matrix Analysis and Applications*,
   23(1):15–41, 2001.  DOI: 10.1137/S0895479899358194.  MUMPS.

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

Nuclear fuel performance
------------------------

These are the works behind :doc:`theory/fuel_performance`.  The module was
written from the open literature only; no source code of BISON, TRANSURANUS
or OFFBEAT was read.  Every DOI below was checked against Crossref in
September 2026.  Several of the original papers are paywalled or are reports
that are not public.  For those, the correlation was taken from a public
secondary source (the FRAPCON-4.0 and FAST-1.0 material property reports of
PNNL, the public BISON documentation pages, or the review of Van Uffelen et
al.), and the entry says so.

*The three codes that inspired the module*

.. [Williamson2012] R. L. Williamson, J. D. Hales, S. R. Novascone,
   M. R. Tonks, D. R. Gaston, C. J. Permann, D. Andrs and R. C. Martineau,
   "Multidimensional multiphysics simulation of nuclear fuel behavior",
   *Journal of Nuclear Materials*, 423(1–3):149–163, 2012.
   DOI: 10.1016/j.jnucmat.2012.01.012.  The first description of BISON.

.. [Williamson2021] R. L. Williamson, J. D. Hales, S. R. Novascone, G. Pastore,
   K. A. Gamble, B. W. Spencer et al., "BISON: a flexible code for advanced
   simulation of the performance of multiple nuclear fuel forms", *Nuclear
   Technology*, 207(7):954–980, 2021.  DOI: 10.1080/00295450.2020.1836940.

.. [Lassmann1992] K. Lassmann, "TRANSURANUS: a fuel rod analysis code ready for
   use", *Journal of Nuclear Materials*, 188:295–302, 1992.
   DOI: 10.1016/0022-3115(92)90487-6.

.. [Scolaro2020] A. Scolaro, I. Clifford, C. Fiorina and A. Pautz, "The OFFBEAT
   multi-dimensional fuel behavior solver", *Nuclear Engineering and Design*,
   358:110416, 2020.  DOI: 10.1016/j.nucengdes.2019.110416.

.. [VanUffelen2019] P. Van Uffelen, J. Hales, W. Li, G. Rossiter and
   R. Williamson, "A review of fuel performance modelling", *Journal of Nuclear
   Materials*, 516:373–412, 2019.  DOI: 10.1016/j.jnucmat.2018.12.037.  Open
   access; read in full.  The best single introduction to the field.

*Gap heat transfer and the coolant*

.. [RossStoute1962] A. M. Ross and R. L. Stoute, *Heat Transfer Coefficient
   Between UO2 and Zircaloy-2*, report AECL-1552 (CRFD-1075), Atomic Energy
   of Canada Limited, Chalk River, 1962.  No DOI; not read.  The form of the
   gap conductance was taken from [LanningHann1975]_ and the FRAPCON-4.0
   material property report.

.. [LanningHann1975] D. D. Lanning and C. R. Hann, *Review of Methods
   Applicable to the Calculation of Gap Conductance in Zircaloy-Clad UO2 Fuel
   Rods*, report BNWL-1894, Battelle Pacific Northwest Laboratories, 1975.
   DOI: 10.2172/4209005.

.. [LindsayBromley1950] A. L. Lindsay and L. A. Bromley, "Thermal conductivity
   of gas mixtures", *Industrial & Engineering Chemistry*, 42(8):1508–1511,
   1950.  DOI: 10.1021/ie50488a017.

.. [Brokaw1958] R. S. Brokaw, "Approximate formulas for the viscosity and
   thermal conductivity of gas mixtures", *The Journal of Chemical Physics*,
   29(2):391–397, 1958.  DOI: 10.1063/1.1744491.

.. [DittusBoelter1930] F. W. Dittus and L. M. K. Boelter, "Heat transfer in
   automobile radiators of the tubular type", *University of California
   Publications in Engineering*, 2(13):443–461, 1930.  Reprinted in
   *International Communications in Heat and Mass Transfer*, 12(1):3–22,
   1985, DOI: 10.1016/0735-1933(85)90003-X.  The coefficient 0.023 used by the
   module is the conventional one (it is due to McAdams, not to the 1930
   paper).

*Fission gas*

.. [Speight1969] M. V. Speight, "A calculation on the migration of fission gas
   in material exhibiting precipitation and re-solution of gas atoms under
   irradiation", *Nuclear Science and Engineering*, 37(2):180–185, 1969.
   DOI: 10.13182/NSE69-A20676.

.. [Turnbull1982] J. A. Turnbull, C. A. Friskney, J. R. Findlay, F. A. Johnson
   and A. J. Walter, "The diffusion coefficients of gaseous and volatile
   species during the irradiation of uranium dioxide", *Journal of Nuclear
   Materials*, 107(2–3):168–184, 1982.  DOI: 10.1016/0022-3115(82)90419-6.

.. [ForsbergMassih1985] K. Forsberg and A. R. Massih, "Diffusion theory of
   fission gas migration in irradiated nuclear fuel UO2", *Journal of Nuclear
   Materials*, 135(2–3):140–148, 1985.  DOI: 10.1016/0022-3115(85)90071-6.
   (The DOI ending "-8" that circulates in the literature does not resolve.)

.. [Pastore2013] G. Pastore, L. Luzzi, V. Di Marcello and P. Van Uffelen,
   "Physics-based modelling of fission gas swelling and release in UO2
   applied to integral fuel rod analysis", *Nuclear Engineering and Design*,
   256:75–86, 2013.  DOI: 10.1016/j.nucengdes.2012.12.002.  Not read
   (paywalled); cited for the grain-boundary saturation picture, whose
   equations were taken from [Pastore2015]_.

.. [Pastore2015] G. Pastore, L. P. Swiler, J. D. Hales, S. R. Novascone,
   D. M. Perez, B. W. Spencer, L. Luzzi, P. Van Uffelen and R. L. Williamson,
   "Uncertainty and sensitivity analysis of fission gas behavior in
   engineering-scale fuel modeling", *Journal of Nuclear Materials*,
   456:398–408, 2015.  DOI: 10.1016/j.jnucmat.2014.09.077.  Read in the
   authors' post-print; the source of the diffusion coefficient used here.

*Uranium dioxide*

.. [Fink2000] J. K. Fink, "Thermophysical properties of uranium dioxide",
   *Journal of Nuclear Materials*, 279(1):1–18, 2000.
   DOI: 10.1016/S0022-3115(99)00273-1.

.. [Lucuta1996] P. G. Lucuta, Hj. Matzke and I. J. Hastings, "A pragmatic
   approach to modelling thermal conductivity of irradiated UO2 fuel: review
   and recommendations", *Journal of Nuclear Materials*, 232(2–3):166–180,
   1996.  DOI: 10.1016/S0022-3115(96)00404-7.

.. [KerriskClifton1972] J. F. Kerrisk and D. G. Clifton, "Smoothed values of
   the enthalpy and heat capacity of UO2", *Nuclear Technology*,
   16(3):531–535, 1972.  DOI: 10.13182/NT72-6.

.. [Martin1988] D. G. Martin, "The thermal expansion of solid UO2 and (U,Pu)
   mixed oxides — a review and recommendations", *Journal of Nuclear
   Materials*, 152(2–3):94–101, 1988.  DOI: 10.1016/0022-3115(88)90315-7.

*Uranium mononitride*

.. [HayesI1990] S. L. Hayes, J. K. Thomas and K. L. Peddicord, "Material
   property correlations for uranium mononitride: I. Physical properties",
   *Journal of Nuclear Materials*, 171(2–3):262–270, 1990.
   DOI: 10.1016/0022-3115(90)90374-V.

.. [HayesII1990] S. L. Hayes, J. K. Thomas and K. L. Peddicord, "Material
   property correlations for uranium mononitride: II. Mechanical properties",
   *Journal of Nuclear Materials*, 171(2–3):271–288, 1990.
   DOI: 10.1016/0022-3115(90)90375-W.

.. [HayesIII1990] S. L. Hayes, J. K. Thomas and K. L. Peddicord, "Material
   property correlations for uranium mononitride: III. Transport properties",
   *Journal of Nuclear Materials*, 171(2–3):289–299, 1990.
   DOI: 10.1016/0022-3115(90)90376-X.

.. [HayesIV1990] S. L. Hayes, J. K. Thomas and K. L. Peddicord, "Material
   property correlations for uranium mononitride: IV. Thermodynamic
   properties", *Journal of Nuclear Materials*, 171(2–3):300–318, 1990.
   DOI: 10.1016/0022-3115(90)90377-Y.

.. [Ross1990] S. B. Ross, M. S. El-Genk and R. B. Matthews, "Uranium nitride
   fuel swelling correlation", *Journal of Nuclear Materials*,
   170(2):169–177, 1990.  DOI: 10.1016/0022-3115(90)90409-G.

.. [Storms1988] E. K. Storms, "An equation which describes fission gas release
   from UN reactor fuel", *Journal of Nuclear Materials*, 158:119–129, 1988.
   DOI: 10.1016/0022-3115(88)90161-4.

*Zircaloy cladding*

.. [LimbackAndersson1996] M. Limbäck and T. Andersson, "A model for analysis
   of the effect of final annealing on the in- and out-of-reactor creep
   behavior of Zircaloy cladding", in *Zirconium in the Nuclear Industry:
   Eleventh International Symposium*, ASTM STP 1295, pp. 448–468, 1996.
   DOI: 10.1520/STP16185S.  Not read (paywalled); the constants were taken
   from the public BISON and FRAPCON-4.0 documentation.

.. [Franklin1982] D. G. Franklin, "Zircaloy-4 cladding deformation during
   power reactor irradiation", in *Zirconium in the Nuclear Industry: Fifth
   Conference*, ASTM STP 754, pp. 235–267, 1982.  DOI: 10.1520/STP37057S.
   Not read; the growth constants were taken from the FRAPCON-4.0 material
   property report.

*Mechanics*

.. [TimoshenkoGoodier1970] S. P. Timoshenko and J. N. Goodier, *Theory of
   Elasticity*, 3rd edition, McGraw-Hill, New York, NY, 1970.  The Lamé
   solution of the thick cylinder and the thermal stresses in a long
   cylinder, used as exact solutions in the verification of the module.

TRISO particles
---------------

These are the works behind :doc:`theory/triso`.

.. [IAEA1674] International Atomic Energy Agency, *Advances in High
   Temperature Gas Cooled Reactor Fuel Technology*, IAEA-TECDOC-CD-1674,
   IAEA, Vienna, 2012.  Chapter 9 reports the fuel performance benchmark of
   the coordinated research project CRP-6: the specification of cases 1 to 8
   (Tables 9.5 to 9.8), the closed form solutions (Eqs. 9.24 to 9.31) and the
   results of the eight participating codes (Figs. 9.4 to 9.16).  Section 9.2
   was read in full, with the tables and equations checked on the rendered
   pages.

.. [Weibull1951] W. Weibull, "A statistical distribution function of wide
   applicability", *Journal of Applied Mechanics*, 18(3):293-297, 1951.
   DOI: 10.1115/1.4010337.  The weakest link model of the failure
   probability.

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
