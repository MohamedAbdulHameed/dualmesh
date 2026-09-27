.. _correlations:

Material correlations and their verification
============================================

.. contents::
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

A fuel performance calculation is only as good as the material correlations
it uses.  This chapter lists every correlation of the ``fuel_performance``
module and of :mod:`dualmesh.fuel`, and for each one it gives the source, how
the source was checked, a few check values, the range of validity, and the
newer or alternative correlations that were considered.  The chapter
:doc:`fuel_performance` explains how the correlations enter the model.

How the correlations were checked
---------------------------------

Every coefficient was compared with a source in September 2026.  The
comparison followed four rules.

* **Primary sources.**  Each correlation is cited to the paper or report that
  published it.  The manuals of other fuel performance codes (BISON,
  TRANSURANUS, OFFBEAT, FRAPCON) were used, at most, to find a primary
  source.  They are never the citation of record, and none of their source
  code was read.
* **First-hand reading.**  Where a source was read, its equations were
  checked on the rendered page images whenever the text extraction lost
  exponents, fractions or signs.  The column "status" in the tables says
  which of the following applies:

  - *read*: the primary source was read and the coefficients agree.
  - *secondary*: the primary source could not be read, and the coefficients
    were read in the named secondary source (a report or review that
    reproduces them).
  - *abstract*: only the abstract of the primary source could be read.
  - *fit*: dualmesh fitted the correlation itself to tabulated data read in
    the primary source.  The fitting script is in
    ``verification/correlations/``.
  - *unverified*: no source of the coefficients could be read.  Such a value
    is kept only where no verified alternative was found, and it is flagged
    in the docstring of the object that uses it.

* **Check values.**  Every correlation reproduces the values printed in, or
  computed by hand from, its source.  The tests
  ``tests/python/test_correlations.py``, ``test_fuel.py`` and ``test_atf.py``
  assert these values.
* **Best available.**  For every property a search was made for newer or more
  accurate published correlations.  The alternatives found are listed with
  each property, with the reason each was used or left aside.

Sources that were blocked were not obtained by other means.  The OSTI
repository refuses automated access, several NRC and PNNL reports were
refused, and most Elsevier full texts could not be downloaded.  These
sources are marked as not read.

Uranium dioxide
---------------

Thermal conductivity
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 42 36

   * - Model
     - Correlation
     - Source and status
   * - ``fink`` (unirradiated)
     - :math:`k_{95} = \frac{100}{7.5408 + 17.692 t + 3.6142 t^2} +
       \frac{6400}{t^{5/2}} e^{-16.35/t}` W/(m K) with :math:`t = T/1000`,
       for 95 % dense fuel, 298 to 3120 K.  The Brandt and Neuer factor
       :math:`1 - (2.6 - 0.5 t) p` corrects it to the porosity :math:`p`.
     - Fink (2000), not read.  IAEA-TECDOC-1496 Sect. 6.1.1.7, Eqs. (1)-(2)
       and Table 1: *secondary*.  Uncertainty 10 % to 2000 K, 20 % above.
   * - ``fink_lucuta`` (default)
     - :math:`k = k_{95}(T)\,\frac{\kappa_{2p}(p)}{\kappa_{2p}(0.05)}\,
       \kappa_{1d}\,\kappa_{1p}\,\kappa_{4r}`, with Lucuta's factors for
       dissolved and precipitated fission products and radiation damage
       (:math:`\kappa_{4r}` for irradiated fuel only), and the Maxwell-Eucken
       pore factor :math:`\kappa_{2p} = (1 - p)/(1 + 0.5 p)`.
     - Lucuta, Matzke and Hastings (1996), not read.  :math:`\kappa_{1d}`,
       :math:`\kappa_{1p}`: IAEA-TECDOC-1496 Sect. 6.1.2.  All four factors:
       Popov et al., ORNL/TM-2000/351, Eqs. (6.4)-(6.7): *secondary*.
   * - ``nfi``
     - The modified NFI model with burnup and gadolinia terms (see
       ``UO2_thermal``).
     - Ohira and Itagaki (1997) and PNNL-19417: not accessible.
       *unverified*, kept as an option.  Its constant gadolinia resistance
       does not fade at high temperature as the data of IAEA-TECDOC-1496
       Table 2 (Sect. 6.1.3.2) do.
   * - ``halden``
     - :math:`k_{95} = [0.1148 + 0.0035 B + 2.475\times10^{-4} (1 -
       0.00333 B) T_C]^{-1} + 0.0132\, e^{0.00188 T_C}` W/(m K), with
       :math:`T_C` in degrees Celsius and the burnup :math:`B` in MWd/kgUO2,
       for 95 % dense fuel to 75 MWd/kgUO2, and the density factor
       :math:`1.0789\, d/(1 + 0.5(1 - d))`.
     - Wiesenack (1997), not read.  IAEA-TECDOC-1496 Sect. 6.1.2, which
       recommends it for irradiated fuel, and CASL-U-2019-1870 Eqs. (1)-(2),
       which BISON uses: *secondary*.  Check: 2.8465 W/(m K) at 1000 C fresh
       and 2.6567 at 10 MWd/kgUO2.

Check values: :math:`k_{95} = 7.612`, 3.467, 2.061 and 2.837 W/(m K) at
298.15, 1000, 2000 and 3000 K (TECDOC-1496: 7.61, 3.47, 2.06, 2.84).  At 1000 K
and 3 at. % the Lucuta factors are :math:`\kappa_{1d} = 0.8345`,
:math:`\kappa_{1p} = 1.0023` and :math:`\kappa_{4r} = 0.9555`.

The ``fink_lucuta`` model starts from Fink's 95 % dense conductivity and
applies the pore factor only to the departure from 95 % density, so that
fresh 95 % dense fuel has Fink's conductivity exactly.  Popov et al. write
the pore factor as :math:`(1 - p)/(1 + 2p)`.  The Maxwell-Eucken form for
spherical pores is used here.

*Alternatives considered.*  IAEA-TECDOC-1496 recommends the Halden correlation
for irradiated fuel to 75 MWd/kgUO2.  Its unirradiated high-temperature term
is older than Fink's.  The Ronchi et al. (2004) and Staicu et al. (2014)
models need the irradiation and annealing temperature history.  Against the
60 MWd/t data of Amaya et al. (TECDOC-1496 tables), ``fink_lucuta``, ``nfi``
and ``halden`` all lie within about 10 %, inside the scatter, and on the
Halden thermocouples of :doc:`fuel_benchmarks` ``fink_lucuta`` is the
closest, so it is the default.

Specific heat, thermal expansion and density
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 42 36

   * - Property
     - Correlation
     - Source and status
   * - Specific heat, ``fink`` (default)
     - Einstein term, linear term and Frenkel-defect term of Fink, per mole,
       divided by :math:`M = 0.2700277` kg/mol.
     - IAEA-TECDOC-1496 Sect. 6.1.1.1: *secondary*.  Check: 311.7 J/(kg K)
       at 1000 K, 725.8 at 3000 K.  Uncertainty 2 % to 1800 K, 13 % above.
   * - Specific heat, ``matpro``
     - MATPRO FCP with Kerrisk and Clifton's constants.
     - NUREG/CR-6150 Vol. 4 (MATPRO, SCDAP/RELAP5/MOD3.1, 1995) Eq. (2-5),
       Table 2-1: *read*.
   * - Thermal expansion
     - Martin's two cubics for :math:`L(T)/L(273\,\mathrm{K})`, split at 923 K.
     - Martin (1988), not read.  IAEA-TECDOC-1496 Sect. 6.1.1.3: *secondary*.
       Section 6.1.1.10 of the same report prints the cubic coefficient as
       4.291e-13, a misprint: only 4.391e-13 reproduces its tables.
   * - Theoretical density
     - 10963 kg/m^3 at 273 K.
     - IAEA-TECDOC-1496 Sect. 6.1.1.10: *secondary*.

Elastic constants
^^^^^^^^^^^^^^^^^

:math:`E = 2.334\times10^{11}\,[1 - 2.752(1 - D)]\,[1 - 1.0915\times10^{-4} T]`
Pa and :math:`\nu = 0.316`, MATPRO FELMOD and FPOIR, NUREG/CR-6150 Vol. 4 Eq.
(2-48): *read*.  Check: 194.69 GPa at 300 K and 179.31 GPa at 1000 K for 95 %
density.  The O/M and plutonium factor of Eq. (2-49) is left out
(stoichiometric UO2).

Creep
^^^^^

MATPRO FCREEP, NUREG/CR-6150 Vol. 4 Eqs. (2-60)-(2-63) and (2-69): *read*.

.. math::

   \dot\varepsilon = \frac{(A_1 + A_2 \dot F)\,\sigma_1\,e^{-Q_1/RT}}{(A_3 + D) G^2}
   + \frac{A_4\,\sigma^{4.5}\,e^{-Q_2/RT}}{A_6 + D}
   + A_7\,\dot F\,\sigma\,e^{-2616.8/T},
   \qquad \sigma_1 = \min(\sigma, \sigma_t),\quad
   \sigma_t = \frac{1.6547\times10^{7}}{G^{0.5714}}\ \mathrm{Pa},

with :math:`G` the grain size in micrometres, :math:`D` the density in per
cent of theoretical, :math:`Q_1 = 17884.8 f + 72124.23` and :math:`Q_2 = 19872 f
+ 111543.5` cal/mol, :math:`f = [e^{-20/\ln(x - 2) - 8} + 1]^{-1}` of the O/M
ratio :math:`x`, and :math:`A_1 = 0.3919`, :math:`A_2 = 1.31\times10^{-19}`,
:math:`A_3 = -87.7`, :math:`A_4 = 2.0391\times10^{-25}`, :math:`A_6 = -90.5`,
:math:`A_7 = 3.72264\times10^{-35}`.  The report says that above the
transition stress "the transition stress is used in the first term".

Check values: :math:`2.1008\times10^{-9}` 1/s at 20 MPa and 1500 K, and
:math:`6.3216\times10^{-7}` at 60 MPa and 1800 K, both with :math:`\dot F =
10^{19}` fissions/(m^3 s) and 10 um grains.

Notes on the source.  The last term uses the activation temperature
2616.8 K of Brucklacher's Eq. (2-69).  The constant list of Eq. (2-63)
prints :math:`Q_3 = 2.6167\times10^3` J/mol with a positive exponent, which
is a misprint.  The primary (time-dependent) factor of Eq. (2-65) is not
included.

Above the transition stress the diffusional term no longer grows with
stress.  A full Newton step across that kink can diverge over a long time
step, so the rod driver uses Newton's method with a backtracking line search.

*Alternatives considered.*  No open, assessed replacement for LWR UO2 was
found.  CASL-U-2019-1870 (Sect. 2.9) uses FCREEP for both UO2 and doped UO2.

Densification
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 42 36

   * - Model
     - Correlation
     - Source and status
   * - ``matpro`` (default)
     - :math:`\frac{\Delta L}{L} = \left(\frac{\Delta L}{L}\right)_m +
       e^{-3(Bu + B)} + 2 e^{-35(Bu + B)}` per cent, with
       :math:`(\Delta L/L)_m = -0.0015\,\mathrm{RSNTR}` below 1000 K and
       :math:`-0.00285\,\mathrm{RSNTR}` above, RSNTR the resintering density
       change (kg/m^3), Bu in MWd/kgU, and B such that the strain is zero at
       zero burnup.  The volumetric strain is three times the linear one.
     - MATPRO FUDENS, NUREG/CR-6150 Vol. 4 Eqs. (2-81), (2-82) and (2-85):
       *read*.
   * - ``escore``
     - :math:`\varepsilon_V = \Delta\rho_0 [\exp(Bu \ln 0.01 / (C_D Bu_D)) -
       1]`, :math:`C_D = 7.235 - 0.0086 (T_C - 25)` below 750 C.
     - Rashid et al., EPRI 1011308, not public: *unverified*.

Densification is irreversible, so both models use the highest temperature
each element has reached, which the rod driver keeps in the element field
``maximum_temperature``.  MATPRO FUDENS is the default.  Its step at 1000 K
is spread linearly over 950 to 1050 K, so that the strain of an element near
1000 K does not jump with a small change of its temperature.

Swelling
^^^^^^^^

Solid fission products: :math:`2.5\times10^{-29}` m^3 per fission, MATPRO
FSWELL, NUREG/CR-6150 Vol. 4 Eq. (2-91): *read*.  With 95 % dense fuel this is
0.5807 per FIMA.

Gaseous swelling (``gaseous_swelling_model``).  The default,
``fission_gas``, is the volume of the bubbles of the fission gas model:
:math:`(\Delta V/V)_{ig} = N_{ig} \tfrac{4}{3}\pi R_{ig}^3` for the
intragranular bubbles and :math:`(\Delta V/V)_{gf} = \tfrac{3}{2a} N_{gf}
V_{gf}` for the grain-face bubbles (Pastore et al., Nucl. Eng. Des. 256
(2013) 75, Eqs. 9 and 10: *read*), as BISON computes it
(CASL-U-2019-1870 Eq. 17).  The swelling and the release then come from the
same gas.  The option ``matpro`` is the empirical MATPRO correlation
:math:`\Delta\varepsilon_V = 8.8\times10^{-56}(2800 - T)^{11.73}
e^{-0.0162(2800 - T)} e^{-8\times10^{-27} B}\,\Delta B`, MATPRO Eqs.
(2-92)-(2-94): *read*.  Check: :math:`7.652\times10^{-4}` for 1500 K, 0.01
FIMA and an increment of 0.001 FIMA at 95 % density.  It is independent of
the fission gas model.

Relocation
^^^^^^^^^^

The ESCORE relocation model (Rashid et al., EPRI 1011308) is not public.  Its
form was taken from the public BISON theory manual: *unverified*.  The
FRAPCON-4.0 relocation model (PNNL-19418), the open alternative, could not be
accessed.  The relocation strain follows the current rod power, so it
vanishes at zero power.

Fission gas
^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 42 36

   * - Part
     - Model
     - Source and status
   * - Single-atom diffusivity
     - :math:`D = 7.6\times10^{-10} e^{-4.86\times10^{-19}/kT} +
       5.64\times10^{-25}\sqrt{\dot F} e^{-1.91\times10^{-19}/kT} +
       2\times10^{-40}\dot F` m^2/s.
     - :math:`D_1` and the athermal :math:`D_3 = 2\times10^{-40}\dot F`:
       Turnbull et al., J. Nucl. Mater. 107 (1982) 168, Eq. (9), and White
       and Tucker, J. Nucl. Mater. 118 (1983) 1, Eqs. (8) and (12): *read*.
       :math:`D_2`: the primary papers give it as a rate-theory expression
       without a closed coefficient.  The value :math:`4 \times 1.41 \times
       10^{-25}` is that of Zullo et al. (2023) Table 1, CASL-U-2019-1870
       Eq. (8) and Cooper et al. (2021) Eq. (10), who cite Turnbull, White
       and Wise (1989), not read: *secondary*.  Pastore et al. (2015), Eq.
       (9), used :math:`1.41\times10^{-25}` and no :math:`D_3`.
   * - Trapping and re-solution
     - Speight's effective diffusivity :math:`bD/(b + g)` (Speight 1969,
       *read*) with :math:`g = 4\pi D (R + R_s) N`, :math:`b =
       2\pi\mu_{ff}(R + R_{ff})^2\dot F` (Olander and Wongsawaeng, J. Nucl.
       Mater. 354 (2006) 94, Eq. 1, *read*),
       :math:`\dot N = 2\eta\dot F - bN`, :math:`R = (3\Omega m/4\pi N)^{1/3}`,
       :math:`\eta = 25`, :math:`\mu_{ff} = 6` um, :math:`R_{ff} = 1` nm,
       :math:`\Omega = 4.09\times10^{-29}` m^3.
     - Zullo et al. (2023) Tables 2-4 and Zullo's thesis Eqs. (2.2)-(2.3):
       *secondary* for :math:`\eta` and the nucleation law.  :math:`R_s`,
       for which no value was found, is taken as the radius of the sphere of
       volume :math:`\Omega`, 0.214 nm.  Option ``white_tucker``: the fitted
       radius and density of White and Tucker (1983), Eq. (27), with their
       :math:`b = 3.03\pi\mu_{ff}(\bar R + Z_0)^2 \dot F` (Eq. 24), *read*.
       The test suite reproduces their Table 1.
   * - Grain-face bubbles (default)
     - Lenticular bubbles that grow by vacancy absorption, coalesce and vent
       at the saturation coverage 0.5.  See
       :class:`dualmesh.fuel.GrainFaceBubbles` for the equations and every
       parameter.
     - Pastore et al., Nucl. Eng. Des. 256 (2013) 75, Eqs. (10)-(28),
       Pastore et al., J. Nucl. Mater. 456 (2015) 398, White, J. Nucl.
       Mater. 325 (2004) 61: *read*.  The surface energy and the dihedral
       angle are those of IAEA-TECDOC-1687 p. 90.  Two parameters are
       assumptions (the boundary layer thickness 0.5 nm and the initial
       bubble radius 10 nm).
   * - Burst release (default)
     - Micro-cracking of the grain faces on temperature changes and healing
       with burnup; see :class:`dualmesh.fuel.GrainFaceBubbles`.
     - Barani et al., J. Nucl. Mater. 486 (2017) 96, Eqs. (3)-(11): *read*.
   * - Re-solution from the grain boundaries (option, off)
     - :math:`\psi(a) = \kappa \dot F N_f / (2 D_{eff})`.
     - Speight (1969) and White and Tucker (1983), Sect. 5.2: *read*.
   * - Grain-boundary saturation (option)
     - :math:`N_s = \frac{4 r_b F(\theta) f_b}{3 k T \sin^2\theta}
       \left(\frac{2\gamma}{r_b} + P\right)`
     - IAEA-TECDOC-1687 p. 90 for the form (which misprints the sign of the
       :math:`\cos^3\theta` term): *secondary*.  The FRAPCON parameter values
       are *unverified*.
   * - Yield and xenon share
     - 0.32 atoms per fission, 88 % xenon.
     - IAEA LiveChart cumulative thermal fission yields of U-235 (stable Kr
       0.0387, stable Xe 0.2152, Xe-135 0.0661): *read*.

The grain-face bubble model is the default.  Grain-face bubbles that are
over-pressurised, because vacancies reach them slowly, hold the gas that
reaches the grain faces at low temperature, which keeps the release of LWR
fuel low.  After 45 MWd/kgU at a uniform temperature and :math:`10^{19}`
fissions/(m^3 s), with 5 um grains, the model releases nothing at 900 K,
0.7 % at 1000 K, 10 % at 1100 K, 24 % at 1200 K, 34 % at 1300 K and 46 % at
1500 K.  The grain-face area per unit volume is the geometric
:math:`3/(2a)`, since each face is shared by two grains.  Intragranular
trapping is weak in LWR conditions (:math:`b/(b + g) \approx 0.95` at 900
K), as the parameters imply.  The athermal coefficient is the primary
:math:`2\times10^{-40}` of Turnbull et al. (1982).  Venting at saturation
follows Pastore et al. (2013), Eqs. (26)-(28): whole bubbles vent, and the
density stops at :math:`10^{10}` m^-2.  The burst release of Barani et al.
(2017) is on by default.  :doc:`fuel_benchmarks` compares the model with
the FUMEX-II exercises and measured release.

Cr2O3-doped UO2
^^^^^^^^^^^^^^^

The diffusivity factors of Cooper et al., J. Nucl. Mater. 545 (2021) 152590
(Eq. 25 and Table 3, *read*), which INL/EXT-20-59969 Table 2.1 and SSM
2021:20 Sect. 4.3.2 repeat:
*secondary*.  Check: :math:`1.7511\times10^{-20}` m^2/s at 1000 K and
:math:`10^{19}` fissions/(m^3 s).  The factors are held at one above their
reference temperature of 1773 K, where the fit would otherwise make doping
slow the diffusion down, and the baseline is the UO2 diffusivity above.
Doping is measured to make creep faster, about five times at 1773 K and 45
MPa for 0.1 wt% Cr2O3 (Dugay et al. 1998, in SSM 2021:20 Table 16).  No
creep correlation for doped fuel exists, so ``creep_rate_factor`` (default
1) lets a user apply such a factor.  The total densification is 0.1 %, the
value measured on the doped rods of the Halden test IFA-677.1
(CASL-U-2019-1870, Sect. 2.7.1).

Surfaces
^^^^^^^^

Emissivity 0.8: MATPRO FEMISS gives :math:`0.7856 + 1.5263\times10^{-5} T`,
0.80 at 1000 K (NUREG/CR-6150 Vol. 4 Eq. (2-35), *read*).  IAEA-TECDOC-1496
recommends 0.85 (Sect. 6.1.1.2), inside the uncertainty.  The pellet and
cladding roughnesses (2 and 1 um) and the roughness coefficient 1.5 are
typical values: *unverified*.  They are fabrication inputs.

Uranium mononitride
-------------------

The Hayes, Thomas and Peddicord series (J. Nucl. Mater. 171, 1990) could be
read in the abstracts only.  The abstracts give the equations with their
ranges, and IAEA-THPH (2008) Eqs. 2.70-2.78 reproduces them.

.. list-table::
   :header-rows: 1
   :widths: 22 42 36

   * - Property
     - Correlation
     - Source and status
   * - Conductivity
     - :math:`k = 1.864\,e^{-2.14P} T^{0.361}` W/(m K), 298-1923 K,
       :math:`P \le 0.2`.
     - Hayes et al. part III: *abstract*.  IAEA-THPH Eq. 2.74.  Check: 20.28
       W/(m K) at 1000 K and 95 % density.
   * - Specific heat
     - Einstein, linear and defect terms, :math:`\Theta = 365.7` K.
     - Hayes part IV: *abstract*.  The enthalpy constant of the same paper
       implies 367.5 K, a difference below 0.1 % in :math:`c_p`.
   * - Density, expansion
     - :math:`\varepsilon = (7.096\times10^{-6} + 1.409\times10^{-9}T)(T -
       298)`, a mean coefficient from 298 K (derived from Hayes' lattice
       parameter).
     - Hayes part I: *abstract*.  Ranger et al., J. Nucl. Mater. 628 (2026)
       156623, is newer and could not be downloaded.
   * - Elastic constants
     - :math:`E = 0.258 D^{3.002}(1 - 2.375\times10^{-5}T)` MPa,
       :math:`\nu = 1.26\times10^{-3} D^{1.174}`, :math:`D` in % TD.
     - Hayes part II: *abstract*.  Check: 221.6 GPa at 300 K and 95 %.
   * - Creep, dislocation
     - :math:`2.054\times10^{-3}\sigma^{4.5}e^{-39369.5/T}` 1/s,
       :math:`\sigma` in MPa, dense UN.
     - Hayes part II: *abstract*, and AbdulHameed et al. (2025) Eq. (1):
       *read*.
   * - Creep, grain boundary
     - :math:`582610.427\,\frac{\sigma}{T d^3}\,e^{-2.28\,\mathrm{eV}/kT}`
       1/s, :math:`d` the grain size in um.
     - AbdulHameed, Beeler, Galvin, Cooper, Elamrawy and Claisse,
       arXiv:2503.03231v4 (2025) Eqs. (14)-(15): *read*.
   * - Creep, irradiation
     - :math:`2.9\times10^{-22}\sigma G e^{0.2P}` per hour, :math:`G` in
       fissions/(cm^3 s), :math:`P` in %.
     - Konovalov, Tarasov and Glagovsky (2016) Eq. (13), the middle of its
       range 2.5-3.3e-22: *read*.
   * - Swelling
     - :math:`4.7\times10^{-11}T^{3.12}B^{0.83}\rho^{0.5}` %, bounded below
       by 1 % per at. % of burnup.
     - S. B. Ross, El-Genk and Matthews (1990): *abstract*.  The floor is
       the lowest swelling rate measured for nitride fuel (NEA No. 7317,
       2018, Sect. 17): *read*.
   * - Gas release
     - Storms' release fraction of temperature, burnup and density.
     - Storms (1988): *abstract*.  The abstract as rendered lost its division
       signs, and a second coefficient set is quoted in the literature, so the
       implemented set remains to be checked in the full paper.

The creep includes the grain-boundary (Coble) creep term of AbdulHameed et
al.  At 30 MPa, 1400 K and 10 um grains it is :math:`7.74\times10^{-8}` 1/s,
14 times the dislocation term.  The irradiation creep follows Konovalov et
al.  Ross' swelling correlation falls below any measured swelling rate under
about 1100 K, so it is bounded below by 1 % per at. %.  The first author is
S. B. Ross.  The emissivity 0.8 has no UN source and is flagged as such.

*Newer work not yet usable.*  Kosmidou et al., Scripta Mater. 276 (2026)
117187, an updated UN creep formulation, was available in abstract only.

Uranium silicide (U3Si2)
------------------------

The LANL U3Si2 property handbook (J. T. White, LA-UR-18-28719, 2018) could not
be read.  Its correlations are reproduced in CASL-U-2019-1870 Sect. 3 and in
IAEA-TECDOC-1921 Sect. 2.2, which agree.

.. list-table::
   :header-rows: 1
   :widths: 22 42 36

   * - Property
     - Correlation
     - Source and status
   * - Conductivity
     - :math:`k = 4.996 + 0.0118 T` W/(m K), 300-1773 K, 5 %.
     - White et al. corrigendum, J. Nucl. Mater. 484 (2017) 386, not read.
       CASL Eq. 23, TECDOC-1921 Eq. 11: *secondary*.  Check: 8.536 at 300 K.
   * - Specific heat
     - :math:`c_p = (0.02582 T + 140.5)/0.77026` J/(kg K).
     - INL/EXT-16-40059 Eq. 4.2: *read*.  Within 1 % of the handbook form.
   * - Elastic constants
     - :math:`E = 142.68 - 6.425p`, :math:`G = 61.27 - 2.901p` GPa, :math:`p`
       the porosity in %, :math:`\nu = E/2G - 1`.
     - CASL Eqs. 39-41: *secondary*.  Check: 110.6 GPa and 0.182 at 95 %.
   * - Thermal expansion
     - :math:`16.0\times10^{-6}` 1/K, 273-1473 K, :math:`\pm 3\times10^{-6}`.
     - CASL Sect. 3.4.2 (handbook recommendation): *secondary*.
   * - Swelling
     - Solid :math:`0.34392\,Bu` (FIMA), and optionally the empirical
       gaseous term :math:`3.88008\,Bu^2 + 0.45419\,Bu`.
     - CASL Eqs. 68 and 70: *secondary*.  Metzger et al. (ICAPP 2014, *read*)
       prints 3.88008 and labels the fit in per cent, a mislabel: its own
       Fig. 4 shows a fraction.
   * - Creep
     - Nabarro-Herring, Coble and climb terms.
     - INL/EXT-20-59969 Eqs. 3.3-3.6: *read*.  Within a factor 2.3 of the
       compressive tests of Yingling et al. (INL/JOU-20-58799 Table 1).  The
       journal version (Cooper et al. 2021) was not read.
   * - Xe diffusivity
     - :math:`2.85\times10^{-4}e^{-3.17/kT} + 3.58\times10^{-42}\dot F`.
     - INL/EXT-20-59969 Eq. 3.1: *read*.

The empirical total swelling of Finlay's low-temperature dispersion fuel is
split, and only the solid part is on by default: power-reactor data disagree
with the gaseous part (about 12 % at 6 GWd/tU in AI-7-1, 0 to 1 % in the
ATF-1 rodlets to 20 GWd/tU).

ATF cladding
------------

.. list-table::
   :header-rows: 1
   :widths: 22 42 36

   * - Property
     - Correlation
     - Source and status
   * - FeCrAl conductivity, specific heat
     - Field et al. handbook Eqs. 3.1-3.4, Tables 1-2.
     - ORNL/SPR-2018/905 Rev. 1: *read*.  APMT within 6 % of the Kanthal
       datasheet.
   * - FeCrAl thermal strain
     - :math:`\varepsilon = 10^{-6}\alpha(T)(T - 293.15)` with the handbook's
       Table 3 polynomial read as a mean coefficient.
     - *read*.  Read as a mean coefficient it reproduces the datasheet's
       strain within 7 %.  The previous integral fell 11-15 % short.
   * - FeCrAl elastic constants
     - APMT: :math:`E = 219.85 - 0.07094 T_C - 1.928\times10^{-5}T_C^2`
       GPa, :math:`\nu = 0.30`.  ORNL alloys: handbook Eqs. 3.6-3.7.
     - Kanthal APMT datasheet: *fit* (within 2.6 GPa of its table).
   * - FeCrAl creep
     - Thermal :math:`0.83\sigma^{7.1}e^{-326\,\mathrm{kJ}/RT}`.  Irradiation
       :math:`B\sigma\dot d`, :math:`B = 5\times10^{-6}` /MPa/dpa, 0.9 dpa per
       :math:`10^{25}` n/m^2.
     - Handbook Eq. 3.9: *read*.  Terrani et al., ORNL/TM-2016/191 Sect. 4.2,
       and IAEA-TECDOC-1921 Eq. 5: *read*.
   * - FeCrAl surface
     - Emissivity 0.70 and Meyer hardness 2.45 GPa (250 HV) for APMT.
     - Kanthal datasheet: *read*.
   * - SiC/SiC
     - Handbook conductivity, expansion and swelling (Eqs. 1, 6-8), the
       swelling at the highest temperature reached.  Specific heat of Snead et
       al. (2007).
     - ORNL/TM-2018/912: *read*.  The specific heat via IAEA-TECDOC-1921 Eq.
       17 (*secondary*), within 1.6 % of NIST-JANAF beta-SiC (*read*).
   * - Chromium coating
     - Aragon et al. (2025) properties.  Thermal creep
       :math:`43.19\sigma^{4.769}e^{-333.6\,\mathrm{kJ}/RT}`, with irradiation
       creep when a dpa conversion is given.
     - Aragon et al.: *read* (its primaries not read).  Creep: *fit* to the
       49 points of Stephens and Klopp, NASA TM X-2499 Table I (*read*).

FeCrAl irradiation creep exceeds the thermal creep by about four orders of
magnitude at LWR conditions, and FeCrAl creeps down about a tenth as much as
Zircaloy.  The FeCrAl thermal strain and the APMT elastic constants,
emissivity and hardness follow the datasheet.  The SiC swelling uses the
highest irradiation temperature reached.  The chromium creep law of Wagih et
al., fitted to the 816 C data only, is 3 to 100 times too slow between 982
and 1316 C, and remains as the option ``thermal_creep="wagih"``.

Not resolved: the chromium thermal expansion (8.35e-6 1/K at room
temperature in Aragon et al., against 6.2-6.7e-6 quoted elsewhere) needs
Holzwarth and Stamm (2002), which could not be read.  No dpa conversion for
chromium was found.  Coating plasticity and cracking are not modelled.

Displacement damage
-------------------

:mod:`dualmesh.fuel.dpa` converts a fast neutron fluence and spectrum to a
dose in dpa, and back.

* The damage energy of a recoil follows Robinson's form of Lindhard's
  partition theory, as written by A. C. Kahler in INDC(NDS)-0648 (2013), p.
  35: *read*.
* The NRT and arc-dpa displacement functions, and the arc-dpa constants of
  Fe, Cu, Ni, Pd, Pt and W, are those of K. Nordlund et al., Nat. Commun. 9
  (2018) 1084, Eqs. 2-4 and Table 1: *read*.
* The damage energy cross sections (MF3 MT444) of C, Al, Si, Ti, Cr, Fe, Ni,
  Zr, Nb, Mo, W and U are the NJOY-2016 processing of ENDF/B-VIII.0 that the
  IAEA Nuclear Data Section distributes with the CRP "Primary Radiation
  Damage Cross Sections", averaged over 50 groups per decade.

Check values: the NRT cross section of iron is 2570 b at 14 MeV and 493 b at
1 MeV.  For elastic scattering that is isotropic in the centre-of-mass frame,
the implemented partition reproduces NJOY's mean damage energy per reaction
of C, Si, Cr and Fe within 1 % at 10 keV, an independent check of the
formula.

*Finding.*  For a U-235 fission spectrum the NRT dose per fluence above 0.1
MeV is 0.96 dpa per :math:`10^{25}` n/m^2 for APMT and 1.13 for SiC.  These
match the rules of thumb of IAEA-TECDOC-1921 (0.9 and 1), which therefore
refer to the fluence above about 0.1 MeV.  Per fluence above 1 MeV, the
fluence the rod carries, the values are about 1.4 and 1.6 times larger, and
larger still in an LWR spectrum with more intermediate neutrons.  The FeCrAl
default keeps the published factor, and its description states this.

TRISO coating layers
--------------------

:mod:`dualmesh.fuel.triso` carries the pyrocarbon correlations of the CRP-6
benchmark, IAEA-TECDOC-1674 (2012), Eqs. 9.22 and 9.23 and Table 9.8: the
swelling rate correlations (a), (b), (c), (e) and (f) and the creep
coefficient correlation (d).  Status: *read*, on the rendered page of the
table.  They are benchmark correlations, fixed by the project for a
comparison of codes, and are not general material models of pyrocarbon.
The chapter :doc:`triso` gives the model, the check values and the
comparison with the eight participating codes.

Zircaloy
--------

.. list-table::
   :header-rows: 1
   :widths: 22 42 36

   * - Property
     - Correlation
     - Source and status
   * - Conductivity
     - :math:`k = 12.767 - 5.4348\times10^{-4}T + 8.9818\times10^{-6}T^2`
       W/(m K), 300-1800 K, held above 1800 K.
     - IAEA-TECDOC-1496 Sect. 6.2.1.4 Eq. (1): *read*.  Check: 15.67 at 600 K,
       21.21 at 1000 K (its Table 1).
   * - Specific heat, density
     - The TECDOC-1496 alpha, transition and beta branches.
       :math:`\rho = 6595.2 - 0.1477 T`, held at its 300 K value.
     - TECDOC-1496 Sects. 6.2.1.1 and 6.2.1.4: *read*.
   * - Thermal expansion
     - Hoop :math:`7.092\times10^{-6}`, axial :math:`5.458\times10^{-6}`,
       radial :math:`9.999\times10^{-6}` 1/K (alpha phase).
     - TECDOC-1496 Sect. 6.2.1.5 Eqs. (4)-(6), after Bunnell et al.: *read*.
   * - Elastic constants
     - MATPRO CELMOD and CSHEAR with the cold-work and fluence factors.
     - NUREG/CR-6150 Vol. 4 Eqs. (4-71)-(4-76): *read*.
   * - Meyer hardness
     - MATPRO CMHARD, :math:`\exp(26.034 - 2.6394\times10^{-2}T +
       4.3504\times10^{-5}T^2 - 2.5621\times10^{-8}T^3)` Pa.
     - NUREG/CR-6150 Vol. 4 Eq. (4-280): *read*.  The scan shows the sign of
       the cubic term indistinctly.  The minus sign reproduces its Fig. 4-59.
   * - Creep
     - Limbäck and Andersson thermal creep with Hoppe's irradiation creep.
     - Neither primary was read: *unverified*.
   * - Irradiation growth
     - Franklin, :math:`2.18\times10^{-21}\Phi^{0.845}`, :math:`\Phi` in
       n/cm^2.
     - *unverified*.

TECDOC-1496 recommends a new conductivity fit and notes that MATPRO counted
one data set three times and runs 5 % high between 400 and 1200 K, so the
TECDOC fit is used.  The thermal expansion is the TECDOC recommendation,
with the radial strain distinct from the hoop strain.  The cold-work term of
CELMOD is available (``cold_work``).  The Meyer hardness falls with
temperature (MATPRO CMHARD).

*Open.*  The Zircaloy creep and growth primaries (Limbäck and Andersson 1996,
Hoppe 1991, Franklin 1982) and the review of Adamson, Coleman and Griffiths
(2019) could not be read.  MATPRO's own creep (CCSTRN) and growth (CAGROW)
models are in the report that was read, and are the verifiable alternative.
Waterside corrosion is not modelled.  A 50 um oxide adds about 15 K to the
cladding temperature.

The gap and the coolant
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 22 42 36

   * - Property
     - Correlation
     - Source and status
   * - Noble gas conductivities
     - :math:`\ln k = c_0 + c_1 \ln T + c_2 (\ln T)^2 + c_3 (\ln T)^3`,
       200-2273 K.
     - Kestin et al. (1984) Tables 1, 3, 4, 5: *fit* (within 0.005 % for He,
       0.05 % for Ar and Kr, 1 % for Xe).
   * - H2 and N2 conductivities
     - MATPRO :math:`k = A T^B`.
     - NUREG/CR-6150 Vol. 4 Table 13-2: *read*.
   * - Mixture rule
     - Lindsay-Bromley in Brokaw's form.
     - Brokaw (1958) not read.  Against Kestin's He-Xe mixture data (Tables
       15-17) it is 1 to 5.5 % low.
   * - Jump distance
     - Kennard's kinetic-theory form, with the accommodation coefficients of
       the FRAPCON gap model bounded below by 0.07.
     - The constant agrees with an independent derivation to 0.1 %.  The
       linear fits are *unverified*.  The bound 0.07 is MATPRO's estimate for
       helium on Zircaloy (Table 13-4, *read*).
   * - Water properties
     - IAPWS-IF97 region 1, IAPWS R12-08 viscosity, R15-11 conductivity
       (industrial forms).
     - The three releases: *read*.  Every verification value of their tables
       is reproduced.
   * - Coolant heat transfer
     - Dittus-Boelter (default) or Weisman, from the local properties.
     - EPRI 1000215 Eqs. 4-4 and 4-5 for Weisman: *read*.  Dittus and
       Boelter (1930) not read.

The helium accommodation coefficient is kept positive at all temperatures,
so that the gap conductance stays finite.  The coolant follows an enthalpy
balance with IAPWS properties, and the heat transfer coefficient varies along
the rod.  A run stops if the coolant reaches saturation, and warns if the
cladding surface exceeds it.  No subcooled boiling correlation (Thom, Jens and
Lottes, Chen) could be read, so none is implemented.

Sources
-------

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Source
     - Status
   * - IAEA-TECDOC-1496, *Thermophysical Properties Database of Materials for
       Light Water Reactors and Heavy Water Reactors*, IAEA, 2006.
     - read
   * - C. M. Allison and D. T. Hagrman (eds.), *SCDAP/RELAP5/MOD3.1 Code
       Manual Vol. IV: MATPRO*, NUREG/CR-6150 Vol. 4, EGG-2720, 1995.
     - read
   * - S. G. Popov, V. K. Ivanov, J. J. Carbajo and G. L. Yoder,
       ORNL/TM-2000/351, 2000.
     - read
   * - K. A. Gamble, G. Pastore, M. W. D. Cooper and D. Andersson,
       CASL-U-2019-1870-000, 2019.
     - read
   * - K. A. Gamble, G. Pastore and M. W. D. Cooper, INL/EXT-20-59969, 2020.
     - read
   * - K. A. Gamble, J. D. Hales, G. Pastore, T. Barani and D. Pizzocri,
       INL/EXT-16-40059, 2016.
     - read
   * - G. Zullo et al., J. Nucl. Mater. 587 (2023) 154744, and G. Zullo,
       PhD thesis, Politecnico di Milano.
     - read
   * - A. R. Massih and L. O. Jernkvist, SSM Report 2021:20.
     - read
   * - IAEA-TECDOC-1687 (FUMEX-II), 2012.
     - read
   * - Fission gas papers read on 27 September 2026: T. Barani et al., J.
       Nucl. Mater. 486 (2017) 96; G. Pastore et al., Nucl. Eng. Des. 256
       (2013) 75 and J. Nucl. Mater. 456 (2015) 398; R. J. White, J. Nucl.
       Mater. 325 (2004) 61; M. W. D. Cooper et al., J. Nucl. Mater. 545
       (2021) 152590; D. R. Olander and D. Wongsawaeng, J. Nucl. Mater. 354
       (2006) 94; J. A. Turnbull, J. Nucl. Mater. 38 (1971) 203; J. A.
       Turnbull et al., J. Nucl. Mater. 107 (1982) 168; M. V. Speight, Nucl.
       Sci. Eng. 37 (1969) 180; R. J. White and M. O. Tucker, J. Nucl.
       Mater. 118 (1983) 1; M. V. Speight and W. Beere, Metal Sci. 9 (1975)
       190; G. L. Reynolds and B. Burton, J. Nucl. Mater. 82 (1979) 22; J. C.
       Killeen, J. Nucl. Mater. 88 (1980) 177; A. Pagani et al., J. Nucl.
       Mater. 630 (2026) 156748; C. Vitanza, E. Kolstad and U. Graziani, ANS
       Topical Meeting on LWR Fuel Performance, Portland, 1979.
     - read
   * - IAEA LiveChart of Nuclides, cumulative fission yields, accessed
       2026-09-26.
     - read
   * - IAEA-THPH, *Thermophysical Properties of Materials for Nuclear
       Engineering*, 2008.
     - read
   * - M. AbdulHameed, B. Beeler, C. O. T. Galvin, M. W. D. Cooper, N.
       Elamrawy and A. Claisse, arXiv:2503.03231v4, 2025.
     - read
   * - I. I. Konovalov, B. A. Tarasov and E. M. Glagovsky, IOP Conf. Ser.
       Mater. Sci. Eng. 130 (2016) 012030.
     - read
   * - OECD/NEA, *State-of-the-Art Report on Light Water Reactor
       Accident-Tolerant Fuels*, NEA No. 7317, 2018.
     - read
   * - K. E. Metzger, T. W. Knight and R. L. Williamson, ICAPP 2014,
       INL/CON-13-30445.
     - read
   * - J. A. Yingling et al., INL/JOU-20-58799, 2021.
     - read
   * - IAEA-TECDOC-1921 (ACTOF), 2020.
     - read
   * - K. G. Field, M. A. Snead, Y. Yamamoto and K. A. Terrani,
       ORNL/SPR-2018/905 Rev. 1, 2018.
     - read
   * - K. A. Terrani, T. M. Karlsen and Y. Yamamoto, ORNL/TM-2016/191, 2016.
     - read
   * - T. Koyanagi, Y. Katoh, G. Jacobsen and C. Deck, ORNL/TM-2018/912, 2018.
     - read
   * - P. Aragon et al., Ann. Nucl. Energy 211 (2025) 110950.
     - read
   * - J. R. Stephens and W. D. Klopp, NASA TM X-2499, 1972.
     - read
   * - Kanthal APMT tube datasheet, kanthal.com, accessed 2026-09-26.
     - read
   * - NIST-JANAF Thermochemical Tables, C-101 and Cr-002.
     - read
   * - J. Kestin, K. Knierim, E. A. Mason, B. Najafi, S. T. Ro and M.
       Waldman, J. Phys. Chem. Ref. Data 13 (1984) 229-303.
     - read
   * - IAPWS R7-97(2012), R12-08 and R15-11.
     - read
   * - EPRI 1000215, *Rod Bundle Heat Transfer for PWRs at Operating
       Conditions*, 2000.
     - read
   * - J. K. Fink (2000), P. G. Lucuta et al. (1996), D. G. Martin (1988),
       J. A. Turnbull, R. J. White and C. Wise (1989), the bodies of S. L.
       Hayes et al. (1990),
       S. B. Ross et al. (1990) and E. K. Storms (1988), J. T. White et al.
       (2015, 2017, 2018), M. Limbäck and T. Andersson (1996), N. E. Hoppe
       (1991), D. G. Franklin (1982), A. M. Ross and R. L. Stoute (1962),
       R. S. Brokaw (1958), L. L. Snead et al. (2007), U. Holzwarth and H.
       Stamm (2002), M. Wagih et al. (2018), the FRAPCON-4.0 reports
       PNNL-19417 and PNNL-19418, and Rashid et al., EPRI 1011308.
     - not read (paywalled, blocked or not public), as stated with each
       correlation above
