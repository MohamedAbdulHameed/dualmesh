.. _triso:

TRISO particle coatings
=======================

.. contents::
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

A TRISO (tristructural isotropic) particle is the fuel of high temperature
gas cooled reactors.  It is a sphere of uranium oxide or oxycarbide, the
kernel, a few hundred micrometres across, wrapped in four coatings:

#. a porous carbon **buffer** (about 100 :math:`\mu\mathrm{m}`), which gives room to
   the fission gas and to the swelling of the kernel,
#. an **inner pyrocarbon** layer (IPyC, about 40 :math:`\mu\mathrm{m}`),
#. a **silicon carbide** layer (SiC, about 35 :math:`\mu\mathrm{m}`), which is the
   pressure vessel of the particle and the main barrier to fission products,
#. an **outer pyrocarbon** layer (OPyC, about 40 :math:`\mu\mathrm{m}`).

Fast neutrons change the dimensions of pyrocarbon.  It first shrinks, and at
higher fluence the radial direction turns to swelling.  Pyrocarbon also
creeps under irradiation, which relaxes the stresses that the shrinkage
builds up.  Silicon carbide is ten times stiffer and does not creep
appreciably at these temperatures.  The shrinking IPyC and OPyC therefore
pull on the SiC and put it in tangential compression, while the gas pressure
from the kernel puts it in tension.  A particle fails when the SiC tension
exceeds its strength, so the tangential stress in the SiC over the life of
the particle is the result that matters.

The module :mod:`dualmesh.fuel.triso` computes these stresses for a single
particle with any number of bonded layers.  Its verification is the
benchmark of the IAEA coordinated research project on TRISO fuel (CRP-6),
reported in IAEA-TECDOC-1674 (2012), chapter 9 [IAEA1674]_.

The model
---------

Geometry and kinematics
~~~~~~~~~~~~~~~~~~~~~~~

The particle is spherically symmetric.  The only displacement is the radial
displacement :math:`u(r)`, and the strains are

.. math::

   \varepsilon_r = \frac{du}{dr},\qquad
   \varepsilon_\theta = \varepsilon_\phi = \frac{u}{r}.

The stresses :math:`\sigma_r` and :math:`\sigma_\theta = \sigma_\phi` obey
the equilibrium equation of a sphere,

.. math::

   \frac{d\sigma_r}{dr} + \frac{2}{r}(\sigma_r - \sigma_\theta) = 0 .

The kernel and the buffer carry no load.  The gas pressure :math:`p_i` acts
on the inner surface of the first stressed layer, at radius
:math:`r_i = d_k/2 + t_b` for a kernel of diameter :math:`d_k` and a buffer
of thickness :math:`t_b`.  The ambient pressure :math:`p_o` acts on the
outer surface.  The layers are bonded, so :math:`u` is continuous across
every interface.

Constitutive law
~~~~~~~~~~~~~~~~

The total strain in each layer is the sum of four parts,

.. math::

   \boldsymbol\varepsilon = \mathbf C^{-1}\boldsymbol\sigma +
   \boldsymbol\varepsilon^{c} + \boldsymbol\varepsilon^{s}(\Phi) +
   \alpha\,(T - T_0)\,\mathbf 1 ,

where :math:`\mathbf C` is the isotropic elasticity matrix (Young's modulus
:math:`E`, Poisson's ratio :math:`\nu`), :math:`\boldsymbol\varepsilon^c` is
the irradiation creep strain, :math:`\boldsymbol\varepsilon^s` is the
irradiation induced dimensional change (called swelling here, with a
negative sign for shrinkage), :math:`\alpha` is the thermal expansion
coefficient and :math:`T_0` the stress free temperature.  Vectors hold the
three normal components :math:`(r, \theta, \phi)`.

Irradiation creep is linear in the stress and proportional to the fast
fluence :math:`\Phi`,

.. math::

   \frac{d\boldsymbol\varepsilon^c}{d\Phi} = K(T)\,\mathbf J\,\boldsymbol\sigma,
   \qquad
   \mathbf J = \begin{pmatrix} 1 & -\nu_c & -\nu_c \\ -\nu_c & 1 & -\nu_c \\
   -\nu_c & -\nu_c & 1 \end{pmatrix},

with the creep coefficient :math:`K` and the creep Poisson's ratio
:math:`\nu_c`.  A value :math:`\nu_c = 0.5` makes creep conserve volume,
since then each row of :math:`\mathbf J` sums to zero.

The swelling of pyrocarbon is anisotropic.  The CRP-6 correlations give the
rates of dimensional change in the radial and tangential directions as
polynomials in the fluence :math:`x` in units of :math:`10^{25}\,
\mathrm{n/m^2}` (IAEA-TECDOC-1674, Eq. 9.22),

.. math::

   \dot g_r = \frac{dg_r}{dx} = \sum_i A_i^{(r)} x^i,\qquad
   \dot g_\theta = \sum_i A_i^{(\theta)} x^i .

dualmesh integrates the polynomials exactly, so the swelling strain
:math:`g(x) = \sum_i A_i x^{i+1}/(i+1)` carries no time integration error.

Fluence convention
~~~~~~~~~~~~~~~~~~

All fluences in the module are fast fluences for neutron energies
:math:`E > 0.18` MeV, which is the convention of the pyrocarbon data.  A
fluence for :math:`E > 0.1` MeV is divided by 1.10 to convert it
(IAEA-TECDOC-1674, chapter 9, footnote 3).  The conversion of a fluence to
displacements per atom is the subject of :mod:`dualmesh.fuel.dpa`.

Discretisation and integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The equilibrium equation is solved in weak form with quadratic finite
elements in the radius,

.. math::

   \int_{r_i}^{r_o} (\sigma_r\,\delta\varepsilon_r +
   2\sigma_\theta\,\delta\varepsilon_\theta)\,r^2\,dr =
   p_i r_i^2\,\delta u(r_i) - p_o r_o^2\,\delta u(r_o),

with three point Gauss quadrature.  The fluence is divided into equal steps.
The creep law is integrated with the :math:`\theta` method,

.. math::

   \Delta\boldsymbol\varepsilon^c = K\,\Delta\Phi\,\mathbf J\,
   [(1-\theta)\boldsymbol\sigma_n + \theta\,\boldsymbol\sigma_{n+1}],

with :math:`\theta = 1/2` (the trapezoidal rule, second order accurate) by
default.  Since the law is linear, the stress at the end of a step follows
from one linear solve with the algorithmic stiffness

.. math::

   \mathbf C_{\mathrm{alg}} = (\mathbf I + \theta K\,\Delta\Phi\,
   \mathbf C\mathbf J)^{-1}\mathbf C .

The creep coefficient is evaluated at the temperature of the middle of the
step.  The stresses at the inner and outer surface of each layer are
evaluated directly from the element displacement field at the element ends,
where the history variables are stored as well.

The default of 32 elements per layer and 600 steps gives surface stresses
within 0.03 MPa of the closed form solutions below.  The spatial error falls
with the square of the element size.  It grows slowly with the fluence,
since the stress is the small difference between the total strain and the
creep and swelling strains, which keep growing.

The CRP-6 benchmark
-------------------

Specification
~~~~~~~~~~~~~

Cases 1 to 8 of CRP-6 test the coating mechanics alone.  The kernel and the
buffer do not interact with the coatings, and the internal pressure is
prescribed (IAEA-TECDOC-1674, Tables 9.5 to 9.8).  All cases use a buffer of
100 :math:`\mu\mathrm{m}` and an ambient pressure of 0.1 MPa.

.. list-table:: Geometry and loading of CRP-6 cases 1 to 8
   :header-rows: 1
   :widths: 10 34 16 22 18

   * - Case
     - Layers (thickness in :math:`\mu\mathrm{m}`)
     - Kernel diameter (:math:`\mu\mathrm{m}`)
     - Internal pressure (MPa)
     - Pyrocarbon swelling
   * - 1
     - SiC 35
     - 500
     - 25
     - none
   * - 2
     - IPyC 90 (BISO)
     - 500
     - 25
     - none
   * - 3
     - IPyC 40, SiC 35
     - 500
     - 25
     - none, no fluence
   * - 4a
     - IPyC 40, SiC 35, no creep
     - 500
     - 25
     - :math:`-0.005` isotropic
   * - 4b
     - IPyC 40, SiC 35
     - 500
     - 25
     - none
   * - 4c
     - IPyC 40, SiC 35
     - 500
     - 25
     - :math:`-0.005` isotropic
   * - 4d
     - IPyC 40, SiC 35
     - 500
     - 25
     - correlation (a)
   * - 5
     - IPyC 40, SiC 35, OPyC 40
     - 350
     - 0 to 15.54, linear
     - correlation (a)
   * - 6
     - IPyC 40, SiC 35, OPyC 40
     - 500
     - 0 to 26.20, linear
     - correlation (a)
   * - 7
     - IPyC 40, SiC 35, OPyC 40
     - 500
     - 0 to 26.20, linear
     - correlation (b)
   * - 8
     - IPyC 40, SiC 35, OPyC 40
     - 500
     - Table 9.7 (cycles)
     - correlation (c)

The swelling rates of cases 4a and 4c are per :math:`10^{25}\,\mathrm{n/m^2}`.
The end of life fluence is :math:`3\times10^{25}\,\mathrm{n/m^2}` for cases
4a to 8.  Pyrocarbon has :math:`E = 39.6` GPa, :math:`\nu = 0.33`, a creep
Poisson's ratio of 0.5 and a creep coefficient :math:`K = 2.71\times
10^{-4}\,(\mathrm{MPa}\cdot10^{25}\,\mathrm{n/m^2})^{-1}`.  The unit of
this coefficient is printed as :math:`\mathrm{MPa}/(10^{25}\,\mathrm{n/m^2})`
in Table 9.6 of the report, which is a misprint, as the closed form solution
of case 4c confirms.  Silicon carbide has :math:`E = 370` GPa and
:math:`\nu = 0.13` and is elastic.  The temperature is 1273 K, except in
case 8.

Case 8 cycles the temperature ten times from 873 K to 1273 K.  Each cycle
lasts :math:`0.3\times10^{25}\,\mathrm{n/m^2}`.  dualmesh raises the
temperature linearly over the first :math:`0.29\times10^{25}` of each cycle
and brings it back to 873 K over the last :math:`0.01\times10^{25}`, which
matches the resolution of the pressure table of the report (Table 9.7).
The particle is stress free at 873 K, the pyrocarbon expansion coefficient
is :math:`5.35\times10^{-6}` 1/K, that of SiC :math:`4.90\times10^{-6}` 1/K,
and the creep coefficient follows correlation (d),

.. math::

   K = 4.386\times10^{-4} - 9.70\times10^{-7}\,T + 8.0294\times10^{-10}\,
   T^2\quad(\mathrm{MPa}\cdot10^{25}\,\mathrm{n/m^2})^{-1},

with :math:`T` in degrees Celsius.  At 1000 :math:`^\circ`\ C it gives
:math:`2.72\times10^{-4}`, the constant of the isothermal cases.  The
function :func:`~dualmesh.fuel.triso.crp6_case` builds every case, and the
coefficients of Table 9.8 are in :data:`~dualmesh.fuel.triso.CRP6_SWELLING`.
They were checked against the rendered page of the table.

Closed form solutions
~~~~~~~~~~~~~~~~~~~~~

The report gives closed form solutions for cases 1 to 4c (section 9.2.3).
For a single shell of inner radius :math:`r_i` and outer radius :math:`r_o`
under pressures :math:`p_i` and :math:`p_o` (Lamé's solution, Eq. 9.24),

.. math::

   \sigma_\theta(r) = \frac{p_i r_i^3 (2r^3 + r_o^3) -
   p_o r_o^3 (2r^3 + r_i^3)}{2r^3(r_o^3 - r_i^3)} .

For two bonded shells, the interface pressure follows from the continuity of
the displacement at the interface (Eq. 9.27).  In case 4a an isotropic
swelling strain :math:`g` is added to the IPyC displacement.  For cases 4b and
4c the report gives the state that creep reaches after a long irradiation.
With creep only (4b), the IPyC relaxes to the hydrostatic stress
:math:`-p_i`, and the SiC carries the whole pressure.  With creep and a
constant swelling rate :math:`\dot g` (4c), and a SiC much stiffer than the
IPyC, the IPyC stresses reach the steady state

.. math::

   \sigma_\theta(r_i) = -\frac{2 m \dot g}{K} - p_i,\qquad
   \sigma_r(r_o) = -\frac{4\dot g (m - 1)}{3K} - p_i,\qquad
   m = \left(\frac{r_o}{r_i}\right)^3 ,

(Eqs. 9.28 and 9.29).  The report's :math:`m` in Eqs. 9.25 and 9.26 is the
inverse ratio :math:`(r_i/r_o)^3`.  The value :math:`(r_o/r_i)^3` is the one
that reproduces the report's numbers in Eqs. 9.28 and 9.29.  This form can
be derived directly: in the steady state the elastic strain is constant, the
creep strain conserves volume, and the rigid SiC holds the outer surface
still, so the displacement rate is :math:`\dot u = \dot g(r - r_o^3/r^2)`.
The creep law then gives :math:`\sigma_\theta - \sigma_r = -2\dot g
r_o^3/(K r^3)`, and the equilibrium equation integrates to the two
expressions above.

.. list-table:: CRP-6 cases 1 to 4c: dualmesh against the closed form solutions
   :header-rows: 1
   :widths: 10 36 18 18 18

   * - Case
     - Stress (MPa)
     - Report, closed form
     - Independent closed form
     - dualmesh
   * - 1
     - SiC :math:`\sigma_\theta`, inner surface
     - 125.19
     - 125.19
     - 125.19
   * - 2
     - IPyC :math:`\sigma_\theta`, inner surface
     - 50.20
     - 50.20
     - 50.20
   * - 3
     - Interface :math:`\sigma_r`
     - :math:`-18.8`
     - :math:`-18.76`
     - :math:`-18.76`
   * - 3
     - IPyC :math:`\sigma_\theta`, inner surface
     - 8.8
     - 8.78
     - 8.78
   * - 3
     - SiC :math:`\sigma_\theta`, inner surface
     - 104.4
     - 104.38
     - 104.38
   * - 4a
     - IPyC :math:`\sigma_\theta` at :math:`3\times10^{25}`
     - 926.80
     - 928.23
     - 928.23
   * - 4a
     - SiC :math:`\sigma_\theta` at :math:`3\times10^{25}`
     - :math:`-845.71`
     - :math:`-847.19`
     - :math:`-847.19`
   * - 4b
     - IPyC :math:`\sigma_\theta` at :math:`3\times10^{25}`
     - :math:`-25`
     - :math:`-25`
     - :math:`-25.00`
   * - 4b
     - SiC :math:`\sigma_\theta` at :math:`3\times10^{25}`
     - 139.34
     - 139.34
     - 139.34
   * - 4c
     - IPyC :math:`\sigma_\theta` at :math:`3\times10^{25}`
     - 26.05
     - 26.05 (rigid SiC)
     - 26.07
   * - 4c
     - SiC :math:`\sigma_\theta` at :math:`3\times10^{25}`
     - 86.5
     - 86.5 (rigid SiC)
     - 86.50

The column "independent closed form" was computed for this manual from
Lamé's solution and the continuity of the displacement, with the ambient
pressure of 0.1 MPa.  Without the ambient pressure, case 1 would give
125.79 MPa, so the report's 125.19 MPa confirms that the ambient pressure
belongs in the solution.  In case 4a the report's closed form values differ
from the exact solution of its own equations by 0.16 %, most likely by
rounding in the report.  The participants' codes span 923 to 962 MPa for the
IPyC in that case.

Case 4d has a quasi equilibrium solution (Eqs. 9.30 and 9.31 of the report,
after Eq. 17 of its reference 236),

.. math::

   \sigma_\theta(r_i) = -\frac{4.5\,m\,\hat g_\theta - 1.5\,(m-1)(\hat g_\theta
   - \hat g_r)}{2.25\,K} - p_i,\qquad
   \hat g = \dot g - C\ddot g + C^2\dddot g - \dots,

with the relaxation dose :math:`C = 1.7\times10^{24}\,\mathrm{n/m^2}`.  The
report prints the series as :math:`\dot g - \ddot g/C + \dddot g/C^2`, which
is not dimensionally consistent when :math:`C` is a dose.  The form with
:math:`C` as a multiplier is dimensionally consistent and agrees with
dualmesh within 0.25 MPa between :math:`1.5` and
:math:`3\times10^{25}\,\mathrm{n/m^2}` (for example 78.2 MPa from the
formula and 78.0 MPa from dualmesh at :math:`2\times10^{25}`).  Below
:math:`10^{25}` the quasi equilibrium has not been reached and the formula
does not apply.

Comparison with the participating codes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Eight codes took part in the benchmark: those of France, Japan, the Republic
of Korea, the Russian Federation, Turkey, the United Kingdom, General
Atomics and the Idaho National Laboratory (PARFUME).  The report shows their
results as curves (Figs. 9.5 to 9.16), which are in
``verification/benchmarks/crp6/crp6_participants.csv``.  The script
``run_crp6.py`` runs the cases and draws the figures below.  Three features of the report's figures
should be known when reading them:

* Some curves are incomplete in the PDF.  In the charts of case 8 the curves
  of General Atomics and INL stop at 0.3 and :math:`0.88\times10^{25}`, and
  the curves of Japan in cases 4d to 8 start at :math:`0.3\times10^{25}`.
* Several participants start their curves at zero stress and ramp to the
  initial elastic value over the first output step, which shows as a
  near vertical line at zero fluence.
* The fluence axes of Figs. 9.13 and 9.14 are labelled
  :math:`10^{24}\,\mathrm{n/m^2}`.  The end of life fluence of these cases is
  :math:`3\times10^{25}\,\mathrm{n/m^2}`, so the label is a misprint and
  the unit is :math:`10^{25}`.

.. list-table:: CRP-6 cases 4d to 8: dualmesh against the range of the participants (MPa)
   :header-rows: 1
   :widths: 10 40 16 34

   * - Case
     - Quantity
     - dualmesh
     - Participants
   * - 4d
     - IPyC :math:`\sigma_\theta`, maximum
     - 149.1
     - 146.9 to 150.3
   * - 4d
     - SiC :math:`\sigma_\theta`, minimum
     - :math:`-41.8`
     - :math:`-51.3` to :math:`-35.9`
   * - 4d
     - SiC :math:`\sigma_\theta`, end of life
     - 79.4
     - 71.0 to 88.0
   * - 5
     - IPyC :math:`\sigma_\theta`, maximum
     - 182.3
     - 178.4 to 197.8
   * - 5
     - SiC :math:`\sigma_\theta`, minimum
     - :math:`-307.9`
     - :math:`-334.4` to :math:`-278.0`
   * - 5
     - SiC :math:`\sigma_\theta`, end of life
     - :math:`-45.3`
     - :math:`-57.5` to :math:`-31.9`
   * - 6
     - IPyC :math:`\sigma_\theta`, maximum
     - 167.8
     - 165.1 to 168.6
   * - 6
     - SiC :math:`\sigma_\theta`, minimum
     - :math:`-288.5`
     - :math:`-316.8` to :math:`-265.8`
   * - 6
     - SiC :math:`\sigma_\theta`, end of life
     - 32.1
     - 27.5 to 45.6
   * - 7
     - IPyC :math:`\sigma_\theta`, maximum
     - 174.0
     - 171.3 to 177.5
   * - 7
     - SiC :math:`\sigma_\theta`, minimum
     - :math:`-299.6`
     - :math:`-324.8` to :math:`-276.6`
   * - 7
     - SiC :math:`\sigma_\theta`, end of life
     - 9.4
     - 3.8 to 23.6
   * - 8
     - IPyC :math:`\sigma_\theta`, maximum
     - 229.7
     - 216.9 to 228.7
   * - 8
     - SiC :math:`\sigma_\theta`, minimum
     - :math:`-417.7`
     - :math:`-420.4` to :math:`-383.4` (Japan :math:`-582.5`)

The most negative SiC values of the participants in cases 5 to 7 come from
the curve of Japan, which starts at :math:`0.3\times10^{25}` from a lower
value than the other codes.  Leaving it out, the ranges are :math:`-308.7`
to :math:`-278.0` (case 5), :math:`-290.5` to :math:`-265.8` (case 6) and
:math:`-300.7` to :math:`-276.6` (case 7), and dualmesh lies at the lower
end with the INL and General Atomics codes.  The maximum IPyC stress of
case 8, 229.7 MPa, is 1 MPa above the highest participant (INL, 228.7 MPa).
The table is regenerated by ``run_crp6.py``, which writes
``crp6_results.csv``.

.. figure:: ../_static/figures/crp6/crp6_case4a_ipyc_tangential.png
   :alt: CRP-6 case 4a, the tangential stress at the inner surface of the IPyC.
   :width: 85%

   CRP-6 case 4a, the tangential stress at the inner surface of the IPyC.
   IPyC with a constant shrinkage rate and no creep, bonded to SiC.  The
   stresses grow linearly with the fluence.  The orange marker is the closed
   form value of the report.

.. figure:: ../_static/figures/crp6/crp6_case4a_sic_tangential.png
   :alt: CRP-6 case 4a, the tangential stress at the inner surface of the SiC.
   :width: 85%

   CRP-6 case 4a, the tangential stress at the inner surface of the SiC.

.. figure:: ../_static/figures/crp6/crp6_case4a_interface_radial.png
   :alt: CRP-6 case 4a, the radial stress at the interface of the IPyC and the SiC.
   :width: 85%

   CRP-6 case 4a, the radial stress at the interface of the IPyC and the
   SiC.

.. figure:: ../_static/figures/crp6/crp6_case4b_ipyc_tangential.png
   :alt: CRP-6 case 4b, the tangential stress at the inner surface of the IPyC.
   :width: 85%

   CRP-6 case 4b, the tangential stress at the inner surface of the IPyC.
   Creep without swelling.  The IPyC relaxes to the hydrostatic stress of
   :math:`-25` MPa and the SiC carries the load.

.. figure:: ../_static/figures/crp6/crp6_case4b_sic_tangential.png
   :alt: CRP-6 case 4b, the tangential stress at the inner surface of the SiC.
   :width: 85%

   CRP-6 case 4b, the tangential stress at the inner surface of the SiC.

.. figure:: ../_static/figures/crp6/crp6_case4b_interface_radial.png
   :alt: CRP-6 case 4b, the radial stress at the interface of the IPyC and the SiC.
   :width: 85%

   CRP-6 case 4b, the radial stress at the interface of the IPyC and the
   SiC.

.. figure:: ../_static/figures/crp6/crp6_case4c_ipyc_tangential.png
   :alt: CRP-6 case 4c, the tangential stress at the inner surface of the IPyC.
   :width: 85%

   CRP-6 case 4c, the tangential stress at the inner surface of the IPyC.
   Creep and a constant shrinkage rate balance in a steady state.

.. figure:: ../_static/figures/crp6/crp6_case4c_sic_tangential.png
   :alt: CRP-6 case 4c, the tangential stress at the inner surface of the SiC.
   :width: 85%

   CRP-6 case 4c, the tangential stress at the inner surface of the SiC.

.. figure:: ../_static/figures/crp6/crp6_case4c_interface_radial.png
   :alt: CRP-6 case 4c, the radial stress at the interface of the IPyC and the SiC.
   :width: 85%

   CRP-6 case 4c, the radial stress at the interface of the IPyC and the
   SiC.

.. figure:: ../_static/figures/crp6/crp6_case4d_ipyc_tangential.png
   :alt: CRP-6 case 4d, the tangential stress at the inner surface of the IPyC.
   :width: 85%

   CRP-6 case 4d, the tangential stress at the inner surface of the IPyC.
   Anisotropic, fluence dependent swelling (correlation (a)). The IPyC
   tension peaks at about :math:`0.5\times10^{25}` when the shrinkage rate
   is highest, and the SiC turns from compression to tension as the radial
   swelling of the pyrocarbon takes over.

.. figure:: ../_static/figures/crp6/crp6_case4d_sic_tangential.png
   :alt: CRP-6 case 4d, the tangential stress at the inner surface of the SiC.
   :width: 85%

   CRP-6 case 4d, the tangential stress at the inner surface of the SiC.

.. figure:: ../_static/figures/crp6/crp6_case4d_interface_radial.png
   :alt: CRP-6 case 4d, the radial stress at the interface of the IPyC and the SiC.
   :width: 85%

   CRP-6 case 4d, the radial stress at the interface of the IPyC and the
   SiC.

.. figure:: ../_static/figures/crp6/crp6_case5_ipyc_tangential.png
   :alt: CRP-6 case 5, the tangential stress at the inner surface of the IPyC.
   :width: 85%

   CRP-6 case 5, the tangential stress at the inner surface of the IPyC. A
   TRISO particle with a 350 :math:`\mu\mathrm{m}` kernel and a pressure rising to
   15.54 MPa.

.. figure:: ../_static/figures/crp6/crp6_case5_sic_tangential.png
   :alt: CRP-6 case 5, the tangential stress at the inner surface of the SiC.
   :width: 85%

   CRP-6 case 5, the tangential stress at the inner surface of the SiC.

.. figure:: ../_static/figures/crp6/crp6_case6_ipyc_tangential.png
   :alt: CRP-6 case 6, the tangential stress at the inner surface of the IPyC.
   :width: 85%

   CRP-6 case 6, the tangential stress at the inner surface of the IPyC. A
   500 :math:`\mu\mathrm{m}` kernel and a pressure rising to 26.20 MPa.

.. figure:: ../_static/figures/crp6/crp6_case6_sic_tangential.png
   :alt: CRP-6 case 6, the tangential stress at the inner surface of the SiC.
   :width: 85%

   CRP-6 case 6, the tangential stress at the inner surface of the SiC.

.. figure:: ../_static/figures/crp6/crp6_case7_ipyc_tangential.png
   :alt: CRP-6 case 7, the tangential stress at the inner surface of the IPyC.
   :width: 85%

   CRP-6 case 7, the tangential stress at the inner surface of the IPyC. As
   case 6 with the swelling of a higher initial anisotropy (correlation (b),
   BAF 1.06).

.. figure:: ../_static/figures/crp6/crp6_case7_sic_tangential.png
   :alt: CRP-6 case 7, the tangential stress at the inner surface of the SiC.
   :width: 85%

   CRP-6 case 7, the tangential stress at the inner surface of the SiC.

.. figure:: ../_static/figures/crp6/crp6_case8_ipyc_tangential.png
   :alt: CRP-6 case 8, the tangential stress at the inner surface of the IPyC.
   :width: 85%

   CRP-6 case 8, the tangential stress at the inner surface of the IPyC. Ten
   temperature cycles between 873 K and 1273 K.  The IPyC stress drops and
   the SiC stress jumps at each return to 873 K, from the thermal
   contraction and the fall of the gas pressure.

.. figure:: ../_static/figures/crp6/crp6_case8_sic_tangential.png
   :alt: CRP-6 case 8, the tangential stress at the inner surface of the SiC.
   :width: 85%

   CRP-6 case 8, the tangential stress at the inner surface of the SiC.

Failure probability
-------------------

The strength of SiC is statistical.  The function
:func:`~dualmesh.fuel.triso.weibull_failure_probability` evaluates the
weakest link model of Weibull [Weibull1951]_ for a spherical layer,

.. math::

   P_f = 1 - \exp\left[-\int_V \left(\frac{\langle\sigma_\theta\rangle}
   {\sigma_0}\right)^m dV\right],

where :math:`\langle\sigma\rangle = \max(\sigma, 0)`, :math:`m` is the
Weibull modulus and :math:`\sigma_0` the characteristic strength in
:math:`\mathrm{Pa\,m^{3/m}}`.  The stress profile comes from
:class:`~dualmesh.fuel.triso.ParticleResult`.  The tangential stress is used
because it is the largest principal stress of a pressurised sphere.

Real particles differ from each other in their dimensions.
:func:`~dualmesh.fuel.triso.monte_carlo_failure_probability` draws the
kernel diameter, the buffer thickness and the layer thicknesses from normal
distributions, solves the stress history of every sampled particle and
evaluates for each the failure probability at the most critical step,
:math:`P_{f,j} = 1 - \exp(-\max_n I_n)` with :math:`I_n` the Weibull
integral at step :math:`n`.  The mean of :math:`P_{f,j}` is the expected
failure fraction of the population, since the strength of each particle is
an independent Weibull draw.  The function returns the mean and its
standard error.  With no scatter it reduces to the single particle result,
which the tests check.  The CRP-6 cases 1 to 8 are single particles, and the
report gives no failure probabilities for them.

Limitations
-----------

* The kernel and the buffer carry no load, and the internal pressure is
  prescribed.  The gas release and CO production models that compute the
  pressure (CRP-6 cases 9 to 13) and the contact between the buffer and the
  IPyC are not part of the module yet.
* The particle is isothermal, and the layers stay bonded.  Debonding and the
  cracking of the IPyC, which the full failure analysis of a particle needs,
  are not modelled.
* The strains are small.  With stresses of a few hundred MPa and strains of
  about 2 %, the small strain assumption is adequate.
* The pyrocarbon correlations are those fixed for the benchmark.  They are
  representative of pyrocarbon at 1273 K with a BAF of 1.03 to 1.06 and are
  not general material models.

Usage
-----

.. code-block:: python

   from dualmesh.fuel import triso

   particle, history = triso.crp6_case("6")
   result = triso.solve_particle(particle, history)
   print(result.maximum_tangential_stress("IPyC") / 1e6)   # MPa
   print(result.inner_tangential_stress["SiC"][-1] / 1e6)

A particle of one's own is built from layers:

.. code-block:: python

   pyc = dict(youngs_modulus=3.96e10, poissons_ratio=0.33,
              creep_coefficient=2.71e-35, swelling="a")
   particle = triso.TrisoParticle(
       kernel_diameter=425e-6,
       buffer_thickness=100e-6,
       layers=(
           triso.CoatingLayer("IPyC", 40e-6, **pyc),
           triso.CoatingLayer("SiC", 35e-6, 3.7e11, 0.13),
           triso.CoatingLayer("OPyC", 40e-6, **pyc),
       ),
   )
   history = triso.ParticleHistory(
       end_fluence=3e25,
       internal_pressure=((0.0, 3e25), (0.0, 20e6)),   # linear ramp
       temperature=1273.0,
   )
   result = triso.solve_particle(particle, history)

The creep coefficient is in SI units, :math:`1/(\mathrm{Pa}\cdot
\mathrm{n/m^2})`: :math:`2.71\times10^{-4}\,(\mathrm{MPa}\cdot10^{25}\,
\mathrm{n/m^2})^{-1} = 2.71\times10^{-35}\,(\mathrm{Pa}\cdot
\mathrm{n/m^2})^{-1}`.
