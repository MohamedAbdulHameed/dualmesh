.. _correlations:

Material correlations
=====================

.. contents::
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

The accuracy of a fuel performance calculation is bounded by the accuracy of
the material correlations it uses.  This chapter lists every correlation of
the ``fuel_performance`` module and of the Python package
:mod:`dualmesh.fuel`.  For each correlation, the chapter gives the equation,
its source, its range of validity, and the check values that the test suite
asserts.  The chapter :doc:`models` explains how the correlations enter the
model of the rod.

Each correlation is cited to the paper or report that published it.  The
check values are values printed in the source, or computed by hand from its
equations, and they are reproduced by the tests
``tests/python/test_correlations.py``, ``test_fuel.py`` and ``test_atf.py``.
Where a correlation was fitted by dualmesh to tabulated data, the fitting
script is in ``verification/correlations/``.

Uranium dioxide
---------------

Thermal conductivity
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 46 32

   * - Model
     - Correlation
     - Source
   * - ``fink``
     - :math:`k_{95} = \frac{100}{7.5408 + 17.692 t + 3.6142 t^2} +
       \frac{6400}{t^{5/2}} e^{-16.35/t}` W/(m K), with :math:`t = T/1000`,
       for 95 % dense fuel between 298 and 3120 K.  The factor of Brandt and
       Neuer, :math:`1 - (2.6 - 0.5 t) p`, corrects it to the porosity
       :math:`p`.
     - Fink (2000) [Fink2000]_, Eqs. (18) and (19), also IAEA-TECDOC-1496
       Sect. 6.1.1.7.  The uncertainty is 10 % up to 2000 K and 20 % above.
   * - ``fink_lucuta`` (default)
     - :math:`k = k_{95}(T)\,\frac{\kappa_{2p}(p)}{\kappa_{2p}(0.05)}\,
       \kappa_{1d}\,\kappa_{1p}\,\kappa_{4r}`, where :math:`\kappa_{1d}`,
       :math:`\kappa_{1p}` and :math:`\kappa_{4r}` are the factors for the
       dissolved and precipitated fission products and for radiation damage
       (:math:`\kappa_{4r}` applies to irradiated fuel only), and
       :math:`\kappa_{2p} = (1 - p)/(1 + 0.5 p)` is the Maxwell-Eucken factor
       for spherical pores.
     - Lucuta, Matzke and Hastings (1996) [Lucuta1996]_, Eqs. (14b), (14c),
       (14d) and (14f), with the shape factor 1.5 of spherical pores in Eq.
       (14d).
   * - ``nfi``
     - The modified NFI model with burnup and gadolinia terms (see
       ``UO2_thermal``).
     - Ohira and Itagaki (1997), in the form of the FRAPCON-4.0 code
       description (PNNL-19418 Vol. 1 Rev. 2), Eqs. (2.52)-(2.56).  The
       gadolinia resistance of the model is
       constant, whereas the data of IAEA-TECDOC-1496 Table 2 (Sect.
       6.1.3.2) show a resistance that decreases at high temperature.
   * - ``halden``
     - :math:`k_{95} = [0.1148 + 0.0035 B + 2.475\times10^{-4} (1 -
       0.00333 B) \min(T_C, 1650)]^{-1} + 0.0132\, e^{0.00188 T_C}` W/(m K),
       where :math:`T_C` is the temperature in degrees Celsius and :math:`B`
       the burnup in MWd/kgUO2, for 95 % dense fuel up to 75 MWd/kgUO2.  The
       density factor is :math:`1.0789\, d/(1 + 0.5(1 - d))`.
     - W. Wiesenack, "Assessment of UO2 conductivity degradation based on
       in-pile temperature data", Sects. III.A and V.  IAEA-TECDOC-1496 Sect.
       6.1.2 recommends it for irradiated fuel, and CASL-U-2019-1870 Eqs.
       (1)-(2) give it with the density factor.  Check values: 2.8465 W/(m K) at 1000 :math:`^\circ\mathrm{C}` for fresh fuel and
       2.6567 W/(m K) at 10 MWd/kgUO2.

The check values of :math:`k_{95}` are 7.612, 3.467, 2.061 and 2.837 W/(m K) at
298.15, 1000, 2000 and 3000 K, against 7.61, 3.47, 2.06 and 2.84 W/(m K) in
IAEA-TECDOC-1496.  At 1000 K and a burnup of 3 at. %, the Lucuta factors are
:math:`\kappa_{1d} = 0.8345`, :math:`\kappa_{1p} = 1.0023` and
:math:`\kappa_{4r} = 0.9555`.

The ``fink_lucuta`` model starts from the conductivity of Fink for 95 % dense
fuel and applies the pore factor to the departure from 95 % density only.
Fresh fuel of 95 % density therefore has the conductivity of Fink exactly.  At
60 MWd/t the
``fink_lucuta``, ``nfi`` and ``halden`` models all lie within about 10 % of the
data of Amaya et al. tabulated in IAEA-TECDOC-1496, which is within the
scatter of the data.  On the Halden thermocouples of :doc:`benchmarks`, the
``fink_lucuta`` model is the closest to the measurements, and it is therefore
the default.

Specific heat, thermal expansion and density
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 46 32

   * - Property
     - Correlation
     - Source
   * - Specific heat, ``fink`` (default)
     - The Einstein term, the linear term and the Frenkel-defect term of Fink,
       per mole, divided by :math:`M = 0.2700277` kg/mol.
     - Fink (2000) [Fink2000]_, Eq. (2), also IAEA-TECDOC-1496 Sect. 6.1.1.1.
       The molar mass is :math:`238.02891 + 2 \times 15.9994` g/mol.  Check values:
       311.7 J/(kg K) at 1000 K and 725.8 J/(kg K) at 3000 K.  The
       uncertainty is 2 % up to 1800 K and 13 % above.
   * - Specific heat, ``matpro``
     - MATPRO FCP with the constants of Kerrisk and Clifton.
     - NUREG/CR-6150 Vol. 4 (MATPRO, SCDAP/RELAP5/MOD3.1, 1995), Eq. (2-5)
       and Table 2-1.
   * - Thermal expansion
     - The two cubic polynomials of Martin for :math:`L(T)/L(273\,\mathrm{K})`,
       joined at 923 K.
     - Martin (1988) [Martin1988]_, Eqs. (1a) and (1b).  Section 6.1.1.10 of
       IAEA-TECDOC-1496 prints the cubic coefficient of Eq. (1a) as
       4.291e-13, a misprint of Martin's 4.391e-13.
   * - Theoretical density
     - :math:`10963\ \mathrm{kg/m^3}` at 273 K.
     - Fink (2000) [Fink2000]_, Eq. (12).

Elastic constants
^^^^^^^^^^^^^^^^^

The Young's modulus and the Poisson's ratio are those of MATPRO FELMOD and
FPOIR (NUREG/CR-6150 Vol. 4, Eq. (2-48)):

.. math::

   E = 2.334\times10^{11}\,[1 - 2.752(1 - D)]\,[1 - 1.0915\times10^{-4} T]
   \ \mathrm{Pa}, \qquad \nu = 0.316,

where :math:`D` is the density as a fraction of the theoretical density and
:math:`T` is the temperature in K.  The check values are 194.69 GPa at 300 K
and 179.31 GPa at 1000 K for 95 % dense fuel.  The factor for the O/M ratio
and the plutonium content of Eq. (2-49) is omitted, since the fuel is
stoichiometric UO2.

Creep
^^^^^

The creep rate is that of MATPRO FCREEP (NUREG/CR-6150 Vol. 4, Eqs.
(2-60)-(2-63) and (2-69)):

.. math::

   \dot\varepsilon = \frac{(A_1 + A_2 \dot F)\,\sigma_1\,e^{-Q_1/RT}}{(A_3 + D) G^2}
   + \frac{A_4\,\sigma^{4.5}\,e^{-Q_2/RT}}{A_6 + D}
   + A_7\,\dot F\,\sigma\,e^{-2616.8/T},
   \qquad \sigma_1 = \min(\sigma, \sigma_t),\quad
   \sigma_t = \frac{1.6547\times10^{7}}{G^{0.5714}}\ \mathrm{Pa},

where :math:`\sigma` is the von Mises stress, :math:`\dot F` is the fission
rate density, :math:`G` is the grain size in micrometres, :math:`D` is the
density in per cent of the theoretical density, :math:`Q_1 = 17884.8 f +
72124.23` and :math:`Q_2 = 19872 f + 111543.5` cal/mol, :math:`f = [e^{-20/\ln(x
- 2) - 8} + 1]^{-1}` is a function of the O/M ratio :math:`x`, and
:math:`A_1 = 0.3919`, :math:`A_2 = 1.31\times10^{-19}`, :math:`A_3 = -87.7`,
:math:`A_4 = 2.0391\times10^{-25}`, :math:`A_6 = -90.5` and :math:`A_7 =
3.72264\times10^{-35}`.  Above the transition stress :math:`\sigma_t`, the
transition stress is used in the first term, as the report prescribes.  The
last term uses the activation temperature of 2616.8 K of Eq. (2-69).  The
constant list of Eq. (2-63) prints :math:`Q_3 = 2.6167\times10^3` J/mol with a
positive exponent, which is a misprint.  The primary (time-dependent) creep of
Eq. (2-65) is not included.

The check values are :math:`2.1008\times10^{-9}` 1/s at 20 MPa and 1500 K, and
:math:`6.3216\times10^{-7}` 1/s at 60 MPa and 1800 K, both with :math:`\dot F =
10^{19}` :math:`\mathrm{fissions/(m^3\,s)}` and grains of 10 :math:`\mu\mathrm{m}`.

Above the transition stress, the diffusional term does not increase with the
stress.  A full Newton step across this kink can diverge over a long time
step, and the rod driver therefore uses Newton's method with a backtracking
line search.  CASL-U-2019-1870 (Sect. 2.9) uses FCREEP for both UO2 and
Cr2O3-doped UO2.

Densification
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 46 32

   * - Model
     - Correlation
     - Source
   * - ``matpro``
     - :math:`\frac{\Delta L}{L} = \left(\frac{\Delta L}{L}\right)_m +
       e^{-3(Bu + B)} + 2 e^{-35(Bu + B)}` per cent, where
       :math:`(\Delta L/L)_m = -0.0015\,\mathrm{RSNTR}` below 1000 K and
       :math:`-0.00285\,\mathrm{RSNTR}` above, RSNTR is the density change in
       a resintering test (:math:`\mathrm{kg/m^3}`), :math:`Bu` is the burnup in
       MWd/kgU, and :math:`B` is chosen such that the strain is zero at zero
       burnup.  The volumetric strain is three times the linear strain.
     - MATPRO FUDENS, NUREG/CR-6150 Vol. 4, Eqs. (2-81), (2-82) and (2-85).
   * - ``escore`` (default)
     - :math:`\varepsilon_V = \Delta\rho_0 [\exp(Bu \ln 0.01 / (C\,Bu_D)) -
       1]`, with :math:`C = 7.235 - 0.0086 (T_C - 25)` below 750
       :math:`^\circ\mathrm{C}` and :math:`C = 1` above, the total
       densification :math:`\Delta\rho_0`, the burnup :math:`Bu_D` at which
       densification is complete, and the pellet-average burnup :math:`Bu`.
     - FALCON MOD01 Vol. 1 (EPRI 1011307), Eqs. (5-22) and (5-24).  The
       intercept :math:`7.235 = 1 + 0.0086 \times 725` makes :math:`C`, and
       so the densification, continuous at 750 :math:`^\circ\mathrm{C}`
       (FALCON prints it rounded to 7.2).

Densification is irreversible.  Both models therefore use the highest
temperature that each element has reached, which the rod driver keeps in the
element field ``maximum_temperature``.  The step of the MATPRO model at 1000 K
is spread linearly over the interval from 950 to 1050 K, so that the strain of
an element near 1000 K does not jump with a small change of its temperature.

Swelling
^^^^^^^^

The solid fission products swell the fuel by :math:`2.5\times10^{-29}\ \mathrm{m^3}`
per fission (MATPRO FSWELL, NUREG/CR-6150 Vol. 4, Eq. (2-91)),
which is 0.5807 per FIMA for 95 % dense fuel.

The gaseous swelling is selected by ``gaseous_swelling_model``.  The default,
``fission_gas``, is the volume of the bubbles of the fission gas model,
:math:`(\Delta V/V)_{ig} = N_{ig} \tfrac{4}{3}\pi R_{ig}^3` for the
intragranular bubbles and :math:`(\Delta V/V)_{gf} = \tfrac{3}{2a} N_{gf}
V_{gf}` for the grain-face bubbles, where :math:`N` is a number density,
:math:`R_{ig}` the radius of an intragranular bubble, :math:`V_{gf}` the
volume of a grain-face bubble and :math:`a` the grain radius (Pastore et al.
[Pastore2013]_, Eqs. 9 and 10, and CASL-U-2019-1870, Eq. 17).  The swelling
and the release are then computed from the same gas.  The option ``matpro`` is
the empirical correlation of MATPRO (Eqs. (2-92)-(2-94)),

.. math::

   \Delta\varepsilon_V = 8.8\times10^{-56}(2800 - T)^{11.73}
   e^{-0.0162(2800 - T)} e^{-8\times10^{-27} B}\,\Delta B,

where :math:`B` is the burnup in :math:`\mathrm{fissions/m^3}`.  Its check value is
:math:`7.652\times10^{-4}` for 1500 K, 0.01 FIMA and an increment of 0.001
FIMA at 95 % density.

Relocation
^^^^^^^^^^

The relocation strain is that of the ESCORE model as FALCON MOD01 Vol. 1
(EPRI 1011307) gives it in Eqs. (5-30) and (5-31):

.. math::

   \frac{\Delta D}{D} = 0.80\,Q\,\frac{G_0}{D_0}\left(0.005\,Bu^{0.3} - 0.20\,D_0 + 0.3\right),

with the linear heat rate :math:`q'` in kW/ft (:math:`Q = 0` below 6,
:math:`(q' - 6)^{1/3}` up to 14 and :math:`(q' - 10)/2` above), the cold
diametral gap :math:`G_0`, the pellet diameter :math:`D_0` in inches, and the
pellet-average burnup :math:`Bu` in MWd/tU.  It follows the current linear heat
rate of the rod and vanishes at zero power.

Fission gas
^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 46 32

   * - Part
     - Model
     - Source
   * - Single-atom diffusivity
     - :math:`D = 7.6\times10^{-10} e^{-4.86\times10^{-19}/kT} +
       5.64\times10^{-25}\sqrt{\dot F} e^{-1.91\times10^{-19}/kT} +
       2\times10^{-40}\dot F` :math:`\mathrm{m^2/s}`.
     - The thermal term :math:`D_1` and the athermal term :math:`D_3 =
       2\times10^{-40}\dot F` are those of Turnbull et al. [Turnbull1982]_,
       Eq. (9), and White and Tucker (1983), Eqs. (8) and (12).  The
       irradiation-enhanced term :math:`D_2` is that of Zullo et al. (2023),
       Table 1, CASL-U-2019-1870, Eq. (8), and Cooper et al. (2021), Eq.
       (10).
       Pastore et al. (2015), Eq. (9), used :math:`1.41\times10^{-25}` for
       :math:`D_2` and no :math:`D_3`.
   * - Trapping and re-solution
     - The effective diffusivity :math:`bD/(b + g)` of Speight (1969), with
       the trapping rate :math:`g = 4\pi D (R + R_s) N`, the re-solution rate
       :math:`b = 2\pi\mu_{ff}(R + R_{ff})^2\dot F` (Olander and Wongsawaeng
       2006, Eq. 1), :math:`\dot N = 2\eta\dot F - bN` and :math:`R =
       (3\Omega m/4\pi N)^{1/3}`, with :math:`\eta = 25`, :math:`\mu_{ff} =
       6` :math:`\mu\mathrm{m}`, :math:`R_{ff} = 1` nm and :math:`\Omega = 4.09\times10^{-29}\ \mathrm{m^3}`.
     - Pizzocri et al. (2018), J. Nucl. Mater. 502, 323, Eqs. (1)-(4) and
       Table 1, which also gives the radius of a gas atom, :math:`R_s = 0.2`
       nm, and Zullo et al. (2023), Tables 2-4.  The option
       ``white_tucker`` uses the fitted bubble radius and density of White
       and Tucker (1983), Eq. (27), with their :math:`b =
       3.03\pi\mu_{ff}(\bar R + Z_0)^2 \dot F` (Eq. 24).  The test suite
       reproduces their Table 1.
   * - Grain-face bubbles (default)
     - Lenticular bubbles that grow by the absorption of vacancies, coalesce,
       and vent at the saturation coverage of 0.5 (see
       :class:`dualmesh.fuel.GrainFaceBubbles` for the equations and every
       parameter).
     - Pastore et al. [Pastore2013]_, Eqs. (10)-(28), Pastore et al.
       [Pastore2015]_, and White (2004).  The surface energy and the dihedral
       angle are those of IAEA-TECDOC-1687, p. 90.  The thickness of the
       boundary diffusion layer (0.5 nm) and the initial bubble radius (10
       nm) are model parameters.
   * - Burst release (default)
     - Micro-cracking of the grain faces on temperature changes and healing
       with burnup (see :class:`dualmesh.fuel.GrainFaceBubbles`).
     - Barani et al. (2017), Eqs. (3)-(11).
   * - Re-solution from the grain boundaries (option, off by default)
     - :math:`\psi(a) = \kappa \dot F N_f / (2 D_{eff})`.
     - Speight (1969) and White and Tucker (1983), Sect. 5.2.
   * - Grain-boundary saturation (option)
     - :math:`N_s = \frac{4 r_b F(\theta) f_b}{3 k T \sin^2\theta}
       \left(\frac{2\gamma}{r_b} + P\right)`.
     - IAEA-TECDOC-1687, p. 90, which prints the sign of the
       :math:`\cos^3\theta` term incorrectly, and the FRAPCON-4.0 parameter
       values.
   * - Yield and xenon fraction
     - 0.32 atoms per fission, 88 % xenon.
     - The cumulative thermal fission yields of U-235 in the IAEA LiveChart of
       Nuclides: stable krypton 0.0387, stable xenon 0.2152 and Xe-135 0.0661.

The grain-face bubble model is the default.  Grain-face bubbles are
over-pressurised when vacancies reach them slowly, and they then hold the gas
that reaches the grain faces at low temperature, which keeps the release of
LWR fuel low.  After 45 MWd/kgU at a uniform temperature and :math:`10^{19}`
:math:`\mathrm{fissions/(m^3\,s)}`, with grains of 5 :math:`\mu\mathrm{m}` radius, the model releases 0 %
at 900 K, 0.7 % at 1000 K, 10 % at 1100 K, 24 % at 1200 K, 34 % at 1300 K and
46 % at 1500 K.  The grain-face area per unit volume is :math:`3/(2a)`, since
each face is shared by two grains.  Intragranular trapping is weak under LWR
conditions (:math:`b/(b + g) \approx 0.95` at 900 K).  At saturation, whole
bubbles vent according to Pastore et al. [Pastore2013]_, Eqs. (26)-(28), and the
bubble density does not fall below :math:`10^{10}\ \mathrm{m^{-2}}`.  The burst
release of Barani et al. (2017) is active by default.  The chapter
:doc:`benchmarks` compares the model with the FUMEX-II exercises and with
measured releases.

Cr2O3-doped UO2
^^^^^^^^^^^^^^^

The diffusivity of the doped fuel is the UO2 diffusivity above multiplied by
the factors of Cooper et al. (2021), Eq. (25) and Table 3, which are also given
in INL/EXT-20-59969, Table 2.1, and SSM 2021:20, Sect. 4.3.2.  The check value
is :math:`1.7511\times10^{-20}` :math:`\mathrm{m^2/s}` at 1000 K and :math:`10^{19}`
:math:`\mathrm{fissions/(m^3\,s)}`.  The factors are held at one above their reference
temperature of 1773 K, where the fit would otherwise make the doping slow the
diffusion down.  Doping is measured to accelerate creep, by a factor of about
five at 1773 K and 45 MPa for 0.1 wt% Cr2O3 (SSM 2021:20,
Table 16).  No creep correlation for doped fuel is available, and
the parameter ``creep_rate_factor`` (default 1) applies such a factor.  The
total densification is 0.1 %, the value measured on the doped rods of the
Halden test IFA-677.1 (CASL-U-2019-1870, Sect. 2.7.1).

Surfaces
^^^^^^^^

The emissivity is 0.8.  MATPRO FEMISS gives :math:`0.7856 + 1.5263\times10^{-5}
T`, i.e., 0.80 at 1000 K (NUREG/CR-6150 Vol. 4, Eq. (2-35)), and
IAEA-TECDOC-1496 recommends 0.85 (Sect. 6.1.1.2), which is within the
uncertainty.  The roughnesses of the pellet and the cladding (2 and 1 :math:`\mu\mathrm{m}`) are
fabrication inputs, and the roughness coefficient of the gap conductance is
1.5.

Uranium mononitride
-------------------

.. list-table::
   :header-rows: 1
   :widths: 22 46 32

   * - Property
     - Correlation
     - Source
   * - Conductivity
     - :math:`k = 1.864\,e^{-2.14P} T^{0.361}` W/(m K), 298-1923 K,
       :math:`P \le 0.2`.
     - Hayes et al. [HayesIII1990]_ and IAEA-THPH (2008), Eq. 2.74.  Check
       value: 20.28 W/(m K) at 1000 K and 95 % density.
   * - Specific heat
     - Einstein, linear and defect terms, with the Einstein temperature
       :math:`\Theta = 365.7` K.
     - Hayes et al. [HayesIV1990]_.  The enthalpy constant of the same paper
       implies :math:`\Theta = 367.5` K, a difference below 0.1 % in
       :math:`c_p`.
   * - Density and thermal expansion
     - :math:`\rho = 14.42 - 2.997\times10^{-4}T - 4.897\times10^{-8}T^2`
       :math:`\mathrm{g/cm^3}`, 298-2523 K, and :math:`\varepsilon =
       (7.096\times10^{-6} + 1.409\times10^{-9}T)(T - 298)`, a mean
       coefficient from 298 K derived from the lattice parameter of Hayes et
       al.  The theoretical density is the value at 298 K,
       :math:`14326\ \mathrm{kg/m^3}`.
     - Hayes et al. [HayesI1990]_, Eq. (3) in the body of the paper.  The
       abstract prints :math:`2.779\times10^{-4}` for the linear coefficient,
       which departs from the density of the lattice parameter by
       :math:`0.056\ \mathrm{g/cm^3}`.  The body value departs from it by
       less than :math:`0.004\ \mathrm{g/cm^3}`.
   * - Elastic constants
     - :math:`E = 0.258 D^{3.002}(1 - 2.375\times10^{-5}T)` MPa and
       :math:`\nu = 1.26\times10^{-3} D^{1.174}`, with :math:`D` in per cent
       of the theoretical density.
     - Hayes et al. [HayesII1990]_.  Check value: 221.6 GPa at 300 K and 95 %
       density.
   * - Creep, dislocation
     - :math:`2.054\times10^{-3}\sigma^{4.5}e^{-39369.5/T}` 1/s, with
       :math:`\sigma` in MPa, for dense UN.
     - Hayes et al. [HayesII1990]_.
   * - Creep, grain boundary
     - :math:`582610.427\,\frac{\sigma}{T d^3}\,e^{-2.28\,\mathrm{eV}/kT}`
       1/s, with :math:`d` the grain size in :math:`\mu\mathrm{m}`.
     - AbdulHameed, Beeler, Galvin, Cooper, Elamrawy and Claisse,
       J. Nucl. Mater. 617 (2025) 156153, Eqs. (14)-(15).
   * - Creep, irradiation
     - :math:`2.9\times10^{-22}\sigma G e^{0.2P}` per hour, with :math:`G` in
       :math:`\mathrm{fissions/(cm^3\,s)}` and :math:`P` in %.
     - Konovalov, Tarasov and Glagovsky (2016), Eq. (13), the middle of its
       range of :math:`2.5` to :math:`3.3\times10^{-22}`.
   * - Swelling
     - :math:`4.7\times10^{-11}T^{3.12}B^{0.83}\rho^{0.5}` %, bounded below by
       1 % per at. % of burnup.
     - Ross, El-Genk and Matthews [Ross1990]_.  The lower bound is the lowest
       swelling rate measured for nitride fuel (NEA No. 7317, 2018, Sect. 17).
   * - Gas release
     - The release fraction of Storms as a function of the temperature, the
       burnup and the density.
     - Storms [Storms1988]_.

The creep includes the grain-boundary (Coble) creep term of AbdulHameed et
al.  At 30 MPa, 1400 K and grains of 10 :math:`\mu\mathrm{m}`, this term is
:math:`7.74\times10^{-8}` 1/s, 14 times the dislocation term.  The irradiation
creep follows Konovalov et al.  The swelling correlation of Ross et al. falls
below every measured swelling rate under about 1100 K, and it is therefore
bounded below by 1 % per at. % of burnup.  The emissivity of UN is taken as
0.8.

Uranium silicide (U3Si2)
------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 46 32

   * - Property
     - Correlation
     - Source
   * - Conductivity
     - :math:`k = 4.996 + 0.0118 T` W/(m K), 300-1773 K, with an uncertainty
       of 5 %.
     - White et al. (2015) and its corrigendum (2017), Eq. (4), also
       CASL-U-2019-1870, Eq. 23, and IAEA-TECDOC-1921, Eq. 11.  Check value:
       8.536 W/(m K) at 300 K.
   * - Specific heat
     - :math:`c_p = (0.02582 T + 140.5)/0.77026` J/(kg K).
     - White et al. (2015), Eq. (2), divided by the molar mass, also
       INL/EXT-16-40059, Eq. 4.2.
   * - Elastic constants
     - :math:`E = 142.68 - 6.425p` and :math:`G = 61.27 - 2.901p` GPa, with
       :math:`p` the porosity in %, and :math:`\nu = E/2G - 1`.
     - CASL-U-2019-1870, Eqs. 39-41.  Check values: 110.6 GPa and 0.182 at
       95 % density.
   * - Thermal expansion
     - :math:`16.0\times10^{-6}` 1/K, 273-1473 K, :math:`\pm 3\times10^{-6}`
       1/K.
     - CASL-U-2019-1870, Sect. 3.4.2.
   * - Swelling
     - Solid swelling :math:`0.34392\,Bu`, with :math:`Bu` in FIMA, and
       optionally the empirical gaseous term :math:`3.88008\,Bu^2 +
       0.45419\,Bu`.
     - CASL-U-2019-1870, Eqs. 68 and 70, and Metzger et al. (ICAPP 2014).
       Metzger et al. label the fit in per cent, whereas their Fig. 4 shows
       a fraction, which is the unit used here.
   * - Creep
     - Nabarro-Herring, Coble and climb terms.
     - INL/EXT-20-59969, Eqs. 3.3-3.6.  The creep rate is within a factor of
       2.3 of the compressive tests of Yingling et al. (INL/JOU-20-58799,
       Table 1).
   * - Xe diffusivity
     - :math:`2.85\times10^{-4}e^{-3.17/kT} + 3.58\times10^{-42}\dot F`.
     - INL/EXT-20-59969, Eq. 3.1.

The empirical total swelling of the low-temperature dispersion fuel of Finlay
is split into a solid and a gaseous part, and only the solid part is active by
default, because the data of power reactors disagree with the gaseous part
(about 12 % at 6 GWd/tU in AI-7-1, against 0 to 1 % in the ATF-1 rodlets up
to 20 GWd/tU).

Accident-tolerant cladding
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 46 32

   * - Property
     - Correlation
     - Source
   * - FeCrAl conductivity and specific heat
     - Field et al. handbook, Eqs. 3.1-3.4 and Tables 1-2.
     - ORNL/SPR-2018/905 Rev. 1.  For APMT the values are within 6 % of the
       Kanthal datasheet.
   * - FeCrAl thermal strain
     - :math:`\varepsilon = 10^{-6}\alpha(T)(T - 293.15)`, with the
       polynomial of Table 3 of the handbook taken as a mean coefficient.
     - ORNL/SPR-2018/905 Rev. 1.  Taken as a mean coefficient, the
       polynomial reproduces the strain of the Kanthal datasheet within 7 %.
   * - FeCrAl elastic constants
     - APMT: :math:`E = 219.85 - 0.07094 T_C - 1.928\times10^{-5}T_C^2` GPa
       and :math:`\nu = 0.30`.  ORNL alloys: handbook Eqs. 3.6-3.7.
     - Fitted to the Kanthal APMT datasheet, within 2.6 GPa of its table.
   * - FeCrAl creep
     - Thermal creep :math:`0.83\sigma^{7.1}e^{-326\,\mathrm{kJ}/RT}`.
       Irradiation creep :math:`B\sigma\dot d`, with :math:`B =
       5\times10^{-6}` 1/(MPa dpa) and 0.9 dpa per :math:`10^{25}`
       :math:`\mathrm{n/m^2}`.
     - Handbook Eq. 3.9, Terrani et al., ORNL/TM-2016/191, Sect. 4.2, and
       IAEA-TECDOC-1921, Eq. 5.
   * - FeCrAl surface
     - Emissivity 0.70 and Meyer hardness 2.45 GPa (250 HV) for APMT.
     - Kanthal APMT datasheet.
   * - SiC/SiC
     - Conductivity, expansion and swelling of the handbook (Eqs. 1 and
       6-8), with the swelling evaluated at the highest temperature reached.
       Specific heat of Snead et al. (2007).
     - ORNL/TM-2018/912.  The specific heat of Snead et al. (2007), Eq. 10,
       which is within 1.6 % of the NIST-JANAF table of beta-SiC.
   * - Chromium coating
     - Properties of Aragon et al. (2025).  Thermal creep
       :math:`43.19\sigma^{4.769}e^{-333.6\,\mathrm{kJ}/RT}`, and irradiation
       creep when a dpa conversion is given.
     - Aragon et al. (2025).  The thermal creep law was fitted by dualmesh to
       the 49 measurements of Stephens and Klopp, NASA TM X-2499, Table I.

The irradiation creep of FeCrAl exceeds its thermal creep by about four orders
of magnitude under LWR conditions, and FeCrAl creeps down by about a tenth of
the creep-down of Zircaloy.  The creep law of chromium of Wagih et al. (2018),
which was fitted to the data at 816 :math:`^\circ\mathrm{C}` only, is 3 to 100 times slower than the
data of Stephens and Klopp between 982 and 1316 :math:`^\circ\mathrm{C}`, and it is available as the
option ``thermal_creep="wagih"``.

Displacement damage
-------------------

:mod:`dualmesh.fuel.dpa` converts a fast neutron fluence and spectrum to a
dose in dpa, and back.

* The damage energy of a recoil follows Robinson's form of the partition
  theory of Lindhard, as written by A. C. Kahler in INDC(NDS)-0648 (2013), p.
  35.
* The NRT and arc-dpa displacement functions, and the arc-dpa constants of
  Fe, Cu, Ni, Pd, Pt and W, are those of K. Nordlund et al., Nat. Commun. 9
  (2018) 1084, Eqs. 2-4 and Table 1.
* The damage energy cross sections (MF3 MT444) of C, Al, Si, Ti, Cr, Fe, Ni,
  Zr, Nb, Mo, W and U are the NJOY-2016 processing of ENDF/B-VIII.0 that the
  IAEA Nuclear Data Section distributes with the coordinated research project
  "Primary Radiation Damage Cross Sections", averaged over 50 groups per
  decade.

The NRT cross section of iron is 2570 b at 14 MeV and 493 b at 1 MeV.  For
elastic scattering that is isotropic in the centre-of-mass frame, the
implemented partition reproduces the mean damage energy per reaction of NJOY
for C, Si, Cr and Fe within 1 % at 10 keV, which is an independent check of
the formula.

For a U-235 fission spectrum, the NRT dose per fluence above 0.1 MeV is 0.96
dpa per :math:`10^{25}` :math:`\mathrm{n/m^2}` for APMT and 1.13 for SiC.  These values
agree with the rules of thumb of IAEA-TECDOC-1921 (0.9 and 1), which therefore
refer to the fluence above about 0.1 MeV.  Per fluence above 1 MeV, which is
the fluence carried by the rod model, the values are about 1.4 and 1.6 times
larger, and they are larger still in an LWR spectrum with more intermediate
neutrons.  The FeCrAl default keeps the published factor, and its parameter
description states this.

TRISO coating layers
--------------------

:mod:`dualmesh.fuel.triso` implements the pyrocarbon correlations of the CRP-6
benchmark, IAEA-TECDOC-1674 (2012), Eqs. 9.22 and 9.23 and Table 9.8: the
swelling rate correlations (a), (b), (c), (e) and (f) and the creep
coefficient correlation (d).  They are the correlations fixed by the
benchmark for a comparison between codes.  The chapter :doc:`triso` gives the
model, the check values and the comparison with the eight participating codes.

Zircaloy
--------

.. list-table::
   :header-rows: 1
   :widths: 22 46 32

   * - Property
     - Correlation
     - Source
   * - Conductivity
     - :math:`k = 12.767 - 5.4348\times10^{-4}T + 8.9818\times10^{-6}T^2`
       W/(m K), 300-1800 K, held constant above 1800 K.
     - IAEA-TECDOC-1496, Sect. 6.2.1.4, Eq. (1).  Check values: 15.67 W/(m K)
       at 600 K and 21.21 W/(m K) at 1000 K (its Table 1).
   * - Specific heat and density
     - The alpha, transition and beta branches of IAEA-TECDOC-1496.
       :math:`\rho = 6595.2 - 0.1477 T`, held at its value at 300 K.
     - IAEA-TECDOC-1496, Sects. 6.2.1.1 and 6.2.1.4.
   * - Thermal expansion
     - Hoop :math:`7.092\times10^{-6}`, axial :math:`5.458\times10^{-6}` and
       radial :math:`9.999\times10^{-6}` 1/K (alpha phase).
     - IAEA-TECDOC-1496, Sect. 6.2.1.5, Eqs. (4)-(6), after Bunnell et al.
   * - Elastic constants
     - MATPRO CELMOD and CSHEAR with the cold-work and fluence factors.
     - NUREG/CR-6150 Vol. 4, Eqs. (4-71)-(4-76).
   * - Meyer hardness
     - MATPRO CMHARD, :math:`\exp(26.034 - 2.6394\times10^{-2}T +
       4.3504\times10^{-5}T^2 - 2.5621\times10^{-8}T^3)` Pa.
     - NUREG/CR-6150 Vol. 4, Eq. (4-280).  The negative sign of the cubic term
       reproduces Fig. 4-59 of the report.
   * - Creep
     - The thermal creep of Limbäck and Andersson with the irradiation creep
       of Hoppe.
     - [LimbackAndersson1996]_ and Hoppe (1991).
   * - Irradiation growth
     - The growth of Franklin, :math:`2.18\times10^{-21}\Phi^{0.845}`, with
       the fast fluence :math:`\Phi` in :math:`\mathrm{n/cm^2}`.
     - [Franklin1982]_.

IAEA-TECDOC-1496 recommends a new fit of the conductivity and notes that
MATPRO counted one data set three times and runs 5 % high between 400 and 1200
K.  The fit of IAEA-TECDOC-1496 is therefore used.  The thermal expansion is
the recommendation of the same report, with a radial strain that differs from
the hoop strain.  The cold-work term of CELMOD is available through the
parameter ``cold_work``.  The Meyer hardness decreases with the temperature
(MATPRO CMHARD).  Waterside corrosion is not modelled.  An oxide layer of 50 :math:`\mu\mathrm{m}`
raises the cladding temperature by about 15 K.

The gap and the coolant
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 22 46 32

   * - Property
     - Correlation
     - Source
   * - Conductivities of the noble gases
     - :math:`\ln k = c_0 + c_1 \ln T + c_2 (\ln T)^2 + c_3 (\ln T)^3`,
       200-2273 K.
     - Fitted by dualmesh to Kestin et al. (1984), Tables 1, 3, 4 and 5,
       within 0.005 % for He, 0.05 % for Ar and Kr, and 1 % for Xe.
   * - Conductivities of H2 and N2
     - MATPRO :math:`k = A T^B`.
     - NUREG/CR-6150 Vol. 4, Table 13-2.
   * - Mixture rule
     - Brokaw's Eqs. (12) and (13), with the collision integral ratios
       :math:`A^* = B^* = 1.1`.
     - [Brokaw1958]_.  Against the He-Xe mixture data of Kestin et al.
       (Tables 15-17), the rule is 1 to 5.5 % low.
   * - Temperature jump distance
     - The equation of Kennard for a mixture of monatomic gases, with the
       accommodation coefficients of Ullman et al. for helium and xenon,
       interpolated by the mixture molar mass and bounded below by 0.07.
     - [LanningHann1975]_, Appendix B.  The bound of 0.07 is the MATPRO estimate for
       helium on Zircaloy (NUREG/CR-6150 Vol. 4, Table 13-4).
   * - Water properties
     - IAPWS-IF97 region 1, the IAPWS R12-08 viscosity and the R15-11
       conductivity (industrial forms).
     - The three IAPWS releases.  Every verification value of their tables
       is reproduced.
   * - Coolant heat transfer
     - Weisman (default), :math:`Nu = C\, Re^{0.8} Pr^{1/3}` with :math:`C =
       0.042 P/D - 0.024`, or Dittus-Boelter, :math:`Nu = 0.023\, Re^{0.8}
       Pr^{0.4}`, from the local properties.
     - The form of [DittusBoelter1930]_, Eq. (15), with :math:`n = 0.4` for a
       heated fluid.  Written in SI units (the paper takes :math:`d` in
       inches, :math:`V` in :math:`\mathrm{lb/(ft^2\,s)}` and :math:`z` in
       centipoise), the coefficient of Eq. (15) is 0.0241, 5 % above the
       0.023 used here.  EPRI 1000215, Eqs. 4-4 and 4-5, for the correlation of
       Weisman.

The accommodation coefficient of helium is kept positive at all temperatures,
so that the gap conductance stays finite.  The coolant follows an enthalpy
balance with IAPWS properties, and the heat transfer coefficient varies along
the rod.  A calculation stops if the coolant reaches saturation, and it warns
if the cladding surface exceeds the saturation temperature.  Subcooled boiling
is not modelled.
