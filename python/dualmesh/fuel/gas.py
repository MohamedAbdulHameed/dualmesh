# SPDX-License-Identifier: LGPL-2.1-or-later
r"""Fission gas: production, diffusion out of the grains, and release.

Fission produces the noble gases xenon and krypton, about 0.32 atoms per
fission.  They are almost insoluble in the fuel.  In UO2 the gas atoms
diffuse through each grain to its boundary, collect there in bubbles, and are
released to the free volume of the rod once the bubbles on the grain faces
interconnect (grain-boundary saturation).  The released gas lowers the
conductance of the pellet-cladding gap and raises the pressure in the rod.

:class:`BoothFissionGasRelease` follows this picture at every fuel element:

* **Intragranular diffusion** in an equivalent sphere of radius :math:`a`
  (Booth 1957), :math:`\partial C / \partial t = D \nabla^2 C + \beta` with
  :math:`C(a) = 0`, solved exactly over each time step (with :math:`D` and
  :math:`\beta` held at their values in the step) by expanding
  :math:`u = r C` in the eigenfunctions :math:`\sin(n \pi r / a)`:
  each coefficient obeys :math:`\dot A_n = -\lambda_n A_n + \beta b_n` with
  :math:`\lambda_n = D n^2 \pi^2 / a^2` and :math:`b_n = 2 a (-1)^{n+1} /
  (n \pi)`, whose solution over a step is exact.  This is the spectral form of
  the Forsberg-Massih method; many terms are kept instead of the four-term
  fit, so it is exact for a piecewise-constant history.
* **Single-atom diffusion coefficient** of Turnbull, White and Wise (1989),
  in the three-term form of G. Zullo et al., J. Nucl. Mater. 587 (2023)
  154744, Table 1, and K. A. Gamble et al., CASL-U-2019-1870 (2019), Eq. (8):
  the thermal term :math:`D_1 = 7.6 \times 10^{-10} \exp(-4.86 \times
  10^{-19} / k T)`, the irradiation-enhanced term :math:`D_2 = 5.64 \times
  10^{-25} \sqrt{\dot F} \exp(-1.91 \times 10^{-19} / k T)` and the
  athermal term :math:`D_3 = 2 \times 10^{-40} \dot F` (m^2/s, with the
  fission rate density :math:`\dot F` in fissions/(m^3 s)).  The athermal
  coefficient is that of the primary sources, J. A. Turnbull et al., J.
  Nucl. Mater. 107 (1982) 168, Eq. (9), and R. J. White and M. O. Tucker, J.
  Nucl. Mater. 118 (1983) 1, Eq. (12).  Zullo et al.
  and the CASL report use four times this value, after Turnbull, White and
  Wise (1989).  The value of the original papers is closer to the
  measurements of the fuel benchmarks (see the documentation chapter).
* **Trapping and re-solution** (Speight 1969): gas atoms are trapped in
  intragranular bubbles at the rate :math:`g = 4 \pi D (R + R_s) N` (Ham) and
  knocked back into the lattice by fission fragments at the rate :math:`b = 2
  \pi \mu_{ff} (R + R_{ff})^2 \dot F` (D. R. Olander and D. Wongsawaeng, J.
  Nucl. Mater. 354 (2006) 94, Eq. 1: bubbles of one size, two fragments per
  fission).  White and Tucker (1983), Eq. (24), write :math:`3.03 \pi` in
  place of :math:`2 \pi` for the mean radius of a distribution of sizes;
  that form goes with their fitted radius and density (Eq. 27), available as
  ``intragranular_bubbles="white_tucker"``.  With the bubbles in
  equilibrium with the atoms, the total gas diffuses with :math:`D_{eff} =
  b D / (b + g)`.  The bubbles nucleate at :math:`\nu = 2 \eta \dot F` and
  are destroyed by re-solution, :math:`\dot N = \nu - b N`, and their radius
  follows from the gas they hold, :math:`R = (3 \Omega m / 4 \pi N)^{1/3}`
  with :math:`m = g / (b + g)` of the gas in the grain.  This is the model of
  D. Pizzocri et al., J. Nucl. Mater. 502 (2018) 323, Eqs. (1)-(4), with the
  parameters of its Table 1, which Zullo et al. (2023), Tables 2-4, also use:
  :math:`\eta = 25`, :math:`\mu_{ff} = 6` um, :math:`R_{ff} = 1` nm, the radius
  of a gas atom :math:`R_s = 0.2` nm and :math:`\Omega = 4.09 \times 10^{-29}`
  m^3.
* **Grain-face bubbles** (the default, ``grain_boundary="bubbles"``): the gas
  that reaches the grain boundaries collects in lenticular bubbles on the
  grain faces, which grow by absorbing vacancies, coalesce, and vent the gas
  once they cover half of the face.  This is the engineering model of G.
  Pastore et al., Nucl. Eng. Des. 256 (2013) 75, Eqs. (10)-(28), with
  parameters from G. Pastore et al., J. Nucl. Mater. 456 (2015) 398, and
  R. J. White, J. Nucl. Mater. 325 (2004) 61.  See
  :class:`GrainFaceBubbles`.
* **Burst release by micro-cracking** of the grain faces when the
  temperature changes (T. Barani et al., J. Nucl. Mater. 486 (2017) 96,
  Eqs. 3-11), on by default.  See :class:`GrainFaceBubbles`.
* **Re-solution from the grain boundaries** (M. V. Speight, Nucl. Sci. Eng.
  37 (1969) 180; White and Tucker 1983, Sect. 5.2), off by default
  (``boundary_resolution=0``).  Fission fragments knock the gas on the faces
  back into the grains, which holds the concentration at the grain surface
  at :math:`\psi = b_{gb} \delta N_f / (2 D_{eff})`, with :math:`N_f` the gas
  per unit boundary area and :math:`b_{gb} \delta = \kappa \dot F`.  White
  and Tucker used :math:`b_{gb}\delta` = 1 to 3 :math:`\times 10^{-13}` m/s
  at :math:`\dot F \approx 6.9 \times 10^{18}` m^-3 s^-1, that is
  :math:`\kappa \approx 1.5` to :math:`4.4 \times 10^{-32}` m^4.  With these
  values the release of every benchmark falls far below the measurements,
  so the term is off by default; see the documentation chapter on fuel
  benchmarks.
* **Grain-boundary saturation** (``grain_boundary="saturation"``): the gas
  that reaches the grain boundaries is retained there up to the saturation
  density
  :math:`N_s = \frac{4 r_b F(\theta) V_c}{3 k T \sin^2\theta}
  (\frac{2 \gamma}{r_b} + P)` atoms per unit boundary area (the form of the
  FRAPCON-4.0 code description), with the bubble radius :math:`r_b = 0.5`
  micrometre, the dihedral half-angle :math:`\theta = 50^\circ`, the surface
  energy :math:`\gamma = 0.6` J/m^2, the fraction of the boundary covered at
  saturation :math:`V_c = 0.25`, :math:`F(\theta) = 1 - 1.5 \cos\theta +
  0.5 \cos^3\theta`, and the hydrostatic pressure :math:`P` taken as the rod
  gas pressure.  The boundary area per unit volume of fuel is
  :math:`3 / (2a)` (each boundary is shared by two grains).  Whatever exceeds
  saturation is released.

For UN, :class:`StormsFissionGasRelease` uses the empirical release fraction of
Storms (1988) as a function of the local temperature and burnup.
"""

from __future__ import annotations

import numpy as np

from .._core import fuel as properties

BOLTZMANN = 1.380649e-23
"""The Boltzmann constant, J/K."""

ATHERMAL_COEFFICIENT = 2.0e-40
"""m^5: the athermal diffusion term D3 = c F of Turnbull et al. (1982), Eq. (9)."""


class GrainFaceBubbles:
    r"""Lenticular bubbles on the grain faces, element by element.

    Every element carries the number of bubbles per unit face area
    :math:`N`, the gas atoms :math:`n_g` and the vacancies :math:`n_v` of
    one bubble.  The bubble volume is :math:`V = \omega n_g + \Omega n_v`
    (CASL-U-2019-1870 Eq. 12), so the gas pressure of the van der Waals
    covolume equation is :math:`p = k T n_g / (\Omega n_v)`.  The bubble is a
    lens of radius of curvature :math:`R` and semi-dihedral angle
    :math:`\theta`, :math:`V = \frac{4}{3}\pi R^3 \phi(\theta)` with
    :math:`\phi = 1 - 1.5 \cos\theta + 0.5 \cos^3\theta`, and it covers
    :math:`A = \pi R^2 \sin^2\theta` of the face.  In a step:

    1. the gas that arrived from the grains is shared among the bubbles;
    2. the bubbles absorb or emit vacancies (Speight and Beere),
       :math:`\dot n_v = \frac{2 \pi D_v \delta}{k T S}(p - p_{eq})`, with
       :math:`p_{eq} = 2\gamma/R + P` (P the hydrostatic compression, here
       the rod gas pressure) and :math:`S = -((3 - F)(1 - F) + 2 \ln F)/4`
       of the fractional coverage :math:`F = N A` (Zullo's thesis Eqs. 2.8,
       2.9), integrated by the backward Euler method;
    3. below saturation, growing bubbles coalesce, :math:`dN = -2 N^2
       dA_g` (White 2004, Eq. 7; Pastore et al. 2013, Eq. 17), with
       :math:`dA_g` the growth of one bubble's projected area in the step.
       The merged bubbles keep the gas and the vacancies (Pastore et al.
       2013, Sect. 2.2.2), and the density stops at the lower limit
       :math:`10^{10}` m^-2 (Pastore et al. 2015);
    4. with ``micro_cracking``, the faces that crack on a temperature change
       lose their bubbles and their gas (Barani et al. 2017, below);
    5. at the saturation coverage :math:`F_{sat} = 0.5` (White 2004; Pastore
       et al. 2013) further growth is balanced by venting.  With
       ``venting="bubbles"`` (the default, Pastore et al. 2013, Eqs. 26-28)
       whole bubbles vent, :math:`dN = -(N/A) dA`, and those left keep their
       size.  At the lower density limit the bubbles shrink instead.  With
       ``venting="shrink"`` the contents of every bubble are cut back until
       :math:`F = F_{sat}`.  The two give the same release to within 0.03 %
       on the benchmarks.

    **Micro-cracking** (T. Barani et al., J. Nucl. Mater. 486 (2017) 96).
    A fraction :math:`f` of the faces is intact.  On a
    temperature change the parameter :math:`m(T) = 1 - [1 + Q \exp(s (T -
    T_{infl})/B)]^{-1/Q}` rises (Eq. 6, :math:`s = +1` on heating and
    :math:`-1` on cooling, :math:`B = 10` K, :math:`Q = 33`), and the intact
    fraction falls as :math:`f = f_0 e^{-(m - m_0)}` (Eq. 5).  The cracked
    faces lose their gas, :math:`dF = F\,df_c`, and the storing capacity
    falls with them, :math:`dF_{sat} = F_{sat} (df_c + df_h)`, never above
    0.5 (Eq. 3).  The cracks heal with burnup, :math:`f = f_0 + (1 -
    f_0)(1 - e^{-(u - u_0)})` with :math:`u = bu / (1` GWd/tU) (Eqs. 10-11).
    The inflection temperature falls with burnup, :math:`T_{infl} = 1773 +
    520 e^{-bu/10}` K, bu in GWd/tU (Eq. 8).  On heating :math:`m` rises only
    within about :math:`B` of :math:`T_{infl}`, but on cooling it rises over
    about :math:`QB = 330` K below it, so a shutdown from 1400 K at high
    burnup cracks about a fifth of the faces.

    Parameters and their sources: the initial bubble density 20 per
    square micrometre (White 2004, Sect. 5.1; Pastore et al. 2013 used 40);
    :math:`D_v = (3.5/5) \cdot 8.86\times10^{-6} \exp(-4.17\times10^4/T)`
    m^2/s.  Pastore et al. (2015), Eq. (11), fitted :math:`8.86\times10^{-6}
    \exp(-5.75\times10^{-19}/kT)` to White's values of :math:`D_v \delta`
    (White 2004, Fig. 14), obtained with :math:`\delta = 3.5\times10^{-10}`
    m.  With the thickness :math:`5\times10^{-10}` m used here the
    diffusivity is scaled by 3.5/5 so that the product, the quantity the
    data fix, is White's.  The surface energy 0.626
    J/m^2 and the semi-dihedral angle 50 degrees (IAEA-TECDOC-1687, p. 90);
    the vacancy volume :math:`\Omega = 4.09\times10^{-29}` m^3 (a Schottky
    trio, Zullo et al. 2023); the xenon covolume :math:`\omega = 8.48\times
    10^{-29}` m^3 (the van der Waals b of xenon, 0.05105 L/mol, CRC
    Handbook).  Two values are assumptions, stated as such: the thickness of
    the boundary diffusion layer :math:`\delta = 5\times10^{-10}` m (a
    lattice spacing; no value was found in the sources read) and the initial
    bubble radius 10 nm, which is also the smallest the bubbles may shrink
    to.  The grain-face area per unit volume of fuel is :math:`3/(2a)`, since
    each boundary is shared by two grains (Pastore et al. 2013, Eq. 10).
    """

    def __init__(
        self,
        num_elements: int,
        grain_radius: float,
        initial_density: float = 2.0e13,
        initial_radius: float = 1.0e-8,
        surface_energy: float = 0.626,
        dihedral_half_angle: float = 50.0,
        saturation_coverage: float = 0.5,
        vacancy_volume: float = 4.09e-29,
        gas_covolume: float = 8.477e-29,
        boundary_thickness: float = 5.0e-10,
        vacancy_diffusivity=None,
        venting: str = "bubbles",
        minimum_density: float = 1.0e10,
        micro_cracking: bool = False,
        inflection_temperature=(1773.0, 520.0, 10.0),
        cracking_width: float = 10.0,
        cracking_asymmetry: float = 33.0,
        healing_burnup: float = 1.0,
    ):
        if venting not in ("bubbles", "shrink"):
            raise ValueError("GrainFaceBubbles: venting must be bubbles or shrink.")
        self.venting = venting
        self.minimum_density = float(minimum_density)
        self.micro_cracking = bool(micro_cracking)
        self.inflection_temperature = tuple(float(v) for v in inflection_temperature)
        self.cracking_width = float(cracking_width)
        self.cracking_asymmetry = float(cracking_asymmetry)
        self.healing_burnup = float(healing_burnup)
        self.intact = np.ones(num_elements)
        self.saturation = np.full(num_elements, float(saturation_coverage))
        self._previous_temperature = None
        self._previous_burnup = None
        self.area_per_volume = 3.0 / (2.0 * grain_radius)
        self.surface_energy = float(surface_energy)
        theta = np.radians(dihedral_half_angle)
        self._phi = 1.0 - 1.5 * np.cos(theta) + 0.5 * np.cos(theta) ** 3
        self._sin2 = np.sin(theta) ** 2
        self.saturation_coverage = float(saturation_coverage)
        self._initial_saturation = self.saturation_coverage
        self.vacancy_volume = float(vacancy_volume)
        self.gas_covolume = float(gas_covolume)
        self.boundary_thickness = float(boundary_thickness)
        self.vacancy_diffusivity = vacancy_diffusivity or (lambda T: (3.5 / 5.0) * 8.86e-6 * np.exp(-4.17e4 / np.asarray(T, dtype=float)))
        # A factor on the vacancy diffusivity, for sensitivity studies.
        self.vacancy_diffusivity_factor = 1.0
        volume0 = 4.0 / 3.0 * np.pi * initial_radius**3 * self._phi
        self.minimum_vacancies = volume0 / self.vacancy_volume
        self.density = np.full(num_elements, float(initial_density))
        self.gas = np.zeros(num_elements)  # atoms per bubble
        self.vacancies = np.full(num_elements, self.minimum_vacancies)

    def volume(self, gas=None, vacancies=None):
        gas = self.gas if gas is None else gas
        vacancies = self.vacancies if vacancies is None else vacancies
        return self.gas_covolume * gas + self.vacancy_volume * vacancies

    def radius(self, volume=None):
        volume = self.volume() if volume is None else volume
        return np.cbrt(3.0 * volume / (4.0 * np.pi * self._phi))

    def projected_area(self, volume=None):
        return np.pi * self.radius(volume) ** 2 * self._sin2

    def coverage(self):
        return self.density * self.projected_area()

    def stored(self):
        """Gas on the grain faces, atoms per m^3 of fuel."""
        return self.gas * self.density * self.area_per_volume

    def swelling(self):
        """Volumetric swelling from the grain-face bubbles."""
        return self.volume() * self.density * self.area_per_volume

    def cracking_parameter(self, temperature, burnup, heating):
        r"""The micro-cracking parameter :math:`m(T)` of Barani et al. (2017),
        Eq. (6), :math:`m = 1 - [1 + Q \exp(s (T - T_{infl}) / B)]^{-1/Q}`
        with :math:`s = +1` on heating and :math:`-1` on cooling, and the
        inflection temperature of Eq. (8), :math:`T_{infl} = \alpha + \beta
        \exp(-bu / \gamma)` (burnup in MWd/kgU, which is GWd/tU)."""
        a, b, g = self.inflection_temperature
        t_infl = a + b * np.exp(-np.asarray(burnup, dtype=float) / g)
        s = np.where(heating, 1.0, -1.0)
        Q, B = self.cracking_asymmetry, self.cracking_width
        x = np.clip(s * (np.asarray(temperature, dtype=float) - t_infl) / B, -700.0, 700.0)
        # [1 + Q e^x]^(-1/Q) = exp(-log1p(Q e^x) / Q), written to stay finite
        # for large x, where log1p(Q e^x) = x + log(Q) + log1p(e^-x / Q).
        big = x > 30.0
        log_term = np.where(big, x + np.log(Q) + np.log1p(np.exp(-np.where(big, x, 0.0)) / Q), np.log1p(Q * np.exp(np.where(big, 0.0, x))))
        return 1.0 - np.exp(-log_term / Q)

    def _crack(self, temperature, burnup):
        """Micro-cracking and healing over a step (Barani et al. 2017, Eqs.
        3-11): returns the relative change of the intact fraction by
        cracking, df_c (negative or zero)."""
        T = np.asarray(temperature, dtype=float)
        bu = np.broadcast_to(np.asarray(burnup, dtype=float), T.shape)
        if self._previous_temperature is None:
            self._previous_temperature, self._previous_burnup = T.copy(), bu.copy()
            return np.zeros_like(T)
        T0, bu0 = self._previous_temperature, self._previous_burnup
        heating = T >= T0
        # m increases on heating and on cooling (s = +1 and -1), so the
        # increment is m_s(T) - m_s(T0) >= 0 for the direction of the step.
        dm = np.maximum(self.cracking_parameter(T, bu, heating) - self.cracking_parameter(T0, bu, heating), 0.0)
        f = self.intact
        f_cracked = f * np.exp(-dm)  # Eq. (5)
        df_c = f_cracked - f
        du = np.maximum(bu - bu0, 0.0) / self.healing_burnup  # Eq. (11)
        f_healed = f_cracked + (1.0 - f_cracked) * (1.0 - np.exp(-du))  # Eq. (10)
        df_h = f_healed - f_cracked
        self.intact = f_healed
        # Eq. (3): dF_sat = F_sat (df_c + df_h), never above its initial value.
        self.saturation = np.minimum(self.saturation * (1.0 + df_c + df_h), self._initial_saturation)
        self._previous_temperature, self._previous_burnup = T.copy(), bu.copy()
        return df_c

    def advance(self, dt, temperature, pressure, arrived, burnup=None):
        """Take the gas that arrived (atoms per m^3 of fuel) through a step;
        returns the gas vented, atoms per m^3 of fuel.  ``burnup`` (MWd/kgU)
        is needed with ``micro_cracking``."""
        T = np.asarray(temperature, dtype=float)
        kT = BOLTZMANN * T
        area_old = self.projected_area()
        # Pastore et al. (2013), Eq. (27): coalescence governs the bubble
        # density below the saturation coverage and venting at it.
        saturated = self.density * area_old >= self.saturation * (1.0 - 1e-9)
        # A negative arrival is gas re-dissolved from the boundaries into the
        # grains; it cannot exceed the gas on the faces.
        self.gas = np.maximum(self.gas + arrived / (self.density * self.area_per_volume), 0.0)
        coverage = np.clip(self.coverage(), 1e-6, 0.99)
        S = -((3.0 - coverage) * (1.0 - coverage) + 2.0 * np.log(coverage)) / 4.0
        K = 2.0 * np.pi * self.vacancy_diffusivity_factor * self.vacancy_diffusivity(T) * self.boundary_thickness / (kT * S)
        old, gas = self.vacancies, self.gas

        def residual(nv):
            R = self.radius(self.volume(gas, nv))
            p = kT * gas / (self.vacancy_volume * nv)
            return nv - old - dt * K * (p - 2.0 * self.surface_energy / R - pressure)

        # Bisection on log(n_v): the residual is negative at the floor unless
        # the bubble is already at it, and positive for a large n_v.
        lo = np.full_like(old, np.log(self.minimum_vacancies))
        hi = np.log(np.maximum(old, self.minimum_vacancies)) + 1.0
        for _ in range(200):
            grow = residual(np.exp(hi)) < 0
            if not np.any(grow):
                break
            hi = np.where(grow, hi + 2.0, hi)
        at_floor = residual(np.exp(lo)) >= 0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            positive = residual(np.exp(mid)) > 0
            hi = np.where(positive, mid, hi)
            lo = np.where(positive, lo, mid)
        self.vacancies = np.where(at_floor, self.minimum_vacancies, np.exp(0.5 * (lo + hi)))
        # Coalescence (White): dN/dA = -2 N^2, with dA the growth of the
        # projected area of a bubble by the gas that arrived and the
        # vacancies it absorbed in the step (exact for this ODE with dA
        # given).  The merging bubbles keep their gas and vacancies, so the
        # volume per unit face area is conserved and the merged bubbles are
        # larger; the coverage therefore reaches saturation after a finite
        # growth.
        area_new = self.projected_area()
        grown = np.maximum(area_new - area_old, 0.0)
        density = self.density / (1.0 + 2.0 * self.density * grown)
        if self.venting == "bubbles":
            density = np.where(saturated, self.density, density)
        # Coalescence stops at the lower limit of the bubble density
        # (Pastore et al. 2015, 1e10 per m^2); a density already below it is
        # kept.
        density = np.maximum(density, np.minimum(self.density, self.minimum_density))
        scale = self.density / density
        self.gas, self.vacancies = self.gas * scale, self.vacancies * scale
        self.density = density
        vented = np.zeros_like(self.gas)
        # Micro-cracking (Barani et al. 2017): the cracked fraction of the
        # faces loses its bubbles and their gas, dF = F df_c, and the storing
        # capacity F_sat falls with the intact fraction.
        if self.micro_cracking:
            if burnup is None:
                raise ValueError("GrainFaceBubbles: micro_cracking needs the burnup.")
            df_c = self._crack(T, burnup)
            lost = -df_c * self.density
            vented += lost * self.gas * self.area_per_volume
            self.density = self.density - lost
        # Venting above the saturation coverage.
        area = self.projected_area()
        if self.venting == "bubbles":
            # Pastore et al. (2013), Eqs. (26)-(28): whole bubbles vent, so
            # the density falls to F_sat / A and the bubbles that remain
            # keep their size and contents, down to the lower density limit.
            target = self.saturation / area
            floor = np.minimum(self.density, self.minimum_density)
            new_density = np.where(target < self.density, np.maximum(target, floor), self.density)
            vented += (self.density - new_density) * self.gas * self.area_per_volume
            self.density = new_density
        # Whatever still exceeds F_sat (all of it with venting="shrink", or
        # at the lower density limit) is vented by shrinking the bubbles.
        area_sat = self.saturation / self.density
        shrink = np.where(area > area_sat, (area_sat / area) ** 1.5, 1.0)
        vented += (1.0 - shrink) * self.gas * self.density * self.area_per_volume
        self.gas = self.gas * shrink
        self.vacancies = np.maximum(self.vacancies * shrink, self.minimum_vacancies)
        return vented


class BoothFissionGasRelease:
    r"""Fission gas release from UO2, element by element; see the module
    documentation.

    Every parameter has a default taken from the source the model follows,
    listed with the parameter:

    ``grain_radius``
        m.  Default 5 um, a 10 um grain, typical of LWR fuel.
    ``num_modes``
        Terms of the eigenfunction series.  Default 200; the test suite
        checks the series against Booth's closed-form release fraction.
    ``fission_gas_yield``
        Stable xenon and krypton atoms per fission.  Default 0.32: the
        cumulative thermal-fission yields of U-235 in the IAEA LiveChart of
        Nuclides give 0.0387 for Kr-83 to Kr-86, 0.2152 for Xe-131, 132, 134
        and 136, and 0.0661 for Xe-135, most of which captures a neutron and
        becomes Xe-136 in a reactor (0.320 in all; Pu-239 gives 0.318).
    ``xenon_fraction``
        The xenon share of the fission gas.  Default 0.88, from the same
        yields (0.95 for Pu-239).
    ``irradiation_factor``
        Factor on the irradiation-enhanced diffusion term :math:`D_2`.
        Default 1.
    ``athermal_coefficient``
        m^5 per fission: the athermal term :math:`D_3 = c \dot F`.  Default
        2e-40 (Turnbull et al. 1982, Eq. 9; see the module documentation).
    ``intragranular_bubbles``
        ``"nucleation"`` (bubbles nucleated at :math:`2 \eta \dot F` and
        destroyed by re-solution, with the radius from the gas they hold)
        or ``"white_tucker"`` (the fitted mean radius :math:`\bar R = 5
        \times 10^{-10} (1 + 106 e^{-0.75\,\mathrm{eV}/kT})` m and density
        :math:`N = 1.52\times10^{27}/T - 3.3\times10^{23}` m^-3 of White and
        Tucker 1983, Eq. 27, with :math:`g = 4\pi \bar R N D`).  Default
        ``"nucleation"``, the approach BISON also takes (Cooper et al.,
        J. Nucl. Mater. 545 (2021) 152590, Sect. 2.2).
    ``resolution_coefficient``
        :math:`c` in :math:`b = c \pi \mu_{ff} (R + R_{ff})^2 \dot F`.
        Default 2 with ``"nucleation"`` and 3.03 with ``"white_tucker"``.
    ``micro_cracking``
        Burst release by micro-cracking of the grain faces (Barani et al.
        2017).  Default True.  :meth:`advance` then needs the burnup.
    ``boundary_resolution``
        :math:`\kappa` (m^4) of the re-solution from the grain boundaries.
        Default 0 (off).
    ``mwd_per_kg_per_fima``
        Converts the burnup given to :meth:`advance` (FIMA) to MWd/kgU for
        the micro-cracking.  Default 938.3; a rod sets the value of its
        fuel.
    ``trapping``
        ``"speight"`` (trapping and re-solution at intragranular bubbles, as
        above) or ``"constant"`` (the diffusion coefficient times
        ``trapping_factor``).  Default ``"speight"`` with the built-in UO2
        coefficient; ``"constant"`` with a ``diffusion_coefficient`` of
        another fuel, whose trapping parameters the UO2 values do not
        describe.
    ``nucleation_factor``, ``fragment_range``, ``fragment_radius``,
    ``gas_atom_volume``, ``gas_atom_radius``
        :math:`\eta`, :math:`\mu_{ff}` (m), :math:`R_{ff}` (m), :math:`\Omega`
        (m^3) and :math:`R_s` (m) of the Speight model; defaults as above.
    ``grain_boundary``
        ``"bubbles"`` (the grain-face bubble model, :class:`GrainFaceBubbles`)
        or ``"saturation"`` (an equilibrium saturation density).  Default
        ``"bubbles"``: bubbles that are over-pressurised because vacancies
        reach them slowly hold far more gas at low temperature than
        equilibrium bubbles, which is what keeps the release of LWR fuel low
        below about 1300 K.
    ``grain_face_parameters``
        Keyword arguments of :class:`GrainFaceBubbles`.  Default none.
    ``bubble_radius``, ``dihedral_half_angle`` (degrees), ``surface_energy``
    (J/m^2), ``saturation_coverage``
        The ``"saturation"`` model.  Defaults 0.5 um, 50 degrees, 0.6 J/m^2
        and 0.25, the values of the FRAPCON-4.0 code description.
    ``trapping_factor``
        With ``trapping="constant"``: a factor in (0, 1] on the diffusion
        coefficient.  Default 1 (no trapping).
    ``diffusion_coefficient``
        A function ``(temperature, fission_rate) -> D`` (m^2/s, arrays in and
        out) that replaces the Turnbull coefficient, for a fuel other than
        UO2.  Default None: the Turnbull coefficient of UO2.
    """

    def __init__(
        self,
        num_elements: int,
        grain_radius: float = 5.0e-6,
        num_modes: int = 200,
        fission_gas_yield: float = 0.32,
        xenon_fraction: float = 0.88,
        irradiation_factor: float = 1.0,
        athermal_coefficient: float = ATHERMAL_COEFFICIENT,
        bubble_radius: float = 0.5e-6,
        dihedral_half_angle: float = 50.0,
        surface_energy: float = 0.6,
        saturation_coverage: float = 0.25,
        trapping_factor: float = 1.0,
        diffusion_coefficient=None,
        trapping: str | None = None,
        nucleation_factor: float = 25.0,
        fragment_range: float = 6.0e-6,
        fragment_radius: float = 1.0e-9,
        gas_atom_volume: float = 4.09e-29,
        gas_atom_radius: float = 0.2e-9,
        grain_boundary: str = "bubbles",
        grain_face_parameters: dict | None = None,
        intragranular_bubbles: str = "nucleation",
        resolution_coefficient: float | None = None,
        mwd_per_kg_per_fima: float = 938.3,
        boundary_resolution: float = 0.0,
        micro_cracking: bool = True,
    ):
        if intragranular_bubbles not in ("nucleation", "white_tucker"):
            raise ValueError("BoothFissionGasRelease: intragranular_bubbles must be nucleation or white_tucker.")
        self.intragranular_bubbles = intragranular_bubbles
        # b = c pi l_f (R + Z0)^2 F: c = 2 for bubbles of one size (Olander and
        # Wongsawaeng 2006, Eq. 1), c = 3.03 for the size distribution of
        # White and Tucker (1983), Eq. (24), whose mean radius it goes with.
        self.resolution_coefficient = (3.03 if intragranular_bubbles == "white_tucker" else 2.0) if resolution_coefficient is None else float(resolution_coefficient)
        self.mwd_per_kg_per_fima = float(mwd_per_kg_per_fima)
        # Irradiation-induced re-solution from the grain boundaries (Speight
        # 1969; White and Tucker 1983, Sect. 5.2): b_gb delta = kappa F (m/s),
        # with kappa = boundary_resolution (m^4).
        self.boundary_resolution = float(boundary_resolution)
        self.boundary_concentration = np.zeros(num_elements)
        if not 0.0 < trapping_factor <= 1.0:
            raise ValueError("BoothFissionGasRelease: trapping_factor must lie in (0, 1].")
        if trapping is None:
            trapping = "speight" if diffusion_coefficient is None else "constant"
        if trapping not in ("speight", "constant"):
            raise ValueError("BoothFissionGasRelease: trapping must be speight or constant.")
        if trapping == "speight" and trapping_factor != 1.0:
            raise ValueError("BoothFissionGasRelease: trapping_factor applies to trapping='constant' only.")
        self.trapping = trapping
        self.nucleation_factor = float(nucleation_factor)
        self.fragment_range = float(fragment_range)
        self.fragment_radius = float(fragment_radius)
        self.gas_atom_volume = float(gas_atom_volume)
        self.gas_atom_radius = float(gas_atom_radius)
        if grain_boundary not in ("bubbles", "saturation"):
            raise ValueError("BoothFissionGasRelease: grain_boundary must be bubbles or saturation.")
        self.grain_boundary = grain_boundary
        self.faces = GrainFaceBubbles(num_elements, grain_radius, **{"micro_cracking": micro_cracking, **(grain_face_parameters or {})}) if grain_boundary == "bubbles" else None
        # Factors of set_factors, for sensitivity and uncertainty studies.
        self.temperature_factor = 1.0
        self.diffusivity_factor = 1.0
        self.resolution_factor = 1.0
        self.bubble_density = np.zeros(num_elements)  # intragranular bubbles per m^3
        self.intragranular_bubble_radius = np.zeros(num_elements)
        self.grain_radius = float(grain_radius)
        self.fission_gas_yield = float(fission_gas_yield)
        self.xenon_fraction = float(xenon_fraction)
        self.irradiation_factor = float(irradiation_factor)
        self.athermal_coefficient = float(athermal_coefficient)
        self.trapping_factor = float(trapping_factor)
        self.diffusion_coefficient = diffusion_coefficient
        a = self.grain_radius
        n = np.arange(1, num_modes + 1, dtype=float)
        self._eigenvalue_factor = (n * np.pi / a) ** 2
        # The source expanded in the eigenfunctions sin(n pi r / a) / r.
        self._source_coefficients = 2.0 * a * (-1.0) ** (n + 1) / (n * np.pi)
        # Gas in the grain per unit volume of grain from the coefficients:
        # (3 / a^3) sum_n A_n a^2 (-1)^(n+1) / (n pi).
        self._concentration_weights = 3.0 / a * (-1.0) ** (n + 1) / (n * np.pi)
        self.coefficients = np.zeros((num_elements, num_modes))
        self.boundary = np.zeros(num_elements)  # atoms per m^3 of fuel on the boundaries
        self.produced = np.zeros(num_elements)
        self.released = np.zeros(num_elements)
        theta = np.radians(dihedral_half_angle)
        f_theta = 1.0 - 1.5 * np.cos(theta) + 0.5 * np.cos(theta) ** 3
        self._saturation_prefactor = 4.0 * bubble_radius * f_theta * saturation_coverage / (3.0 * np.sin(theta) ** 2)
        self._capillary_pressure = 2.0 * surface_energy / bubble_radius

    def set_factors(self, temperature=1.0, diffusivity=1.0, resolution=1.0, grain_boundary_diffusivity=1.0):
        """Multipliers for sensitivity and uncertainty studies (see
        :class:`~dualmesh.fuel.ModelFactors`): on the temperature the model
        sees, the single-atom diffusion coefficient, the re-solution rate
        from the intragranular bubbles and the vacancy diffusivity of the
        grain faces."""
        self.temperature_factor = float(temperature)
        self.diffusivity_factor = float(diffusivity)
        self.resolution_factor = float(resolution)
        if self.faces is not None:
            self.faces.vacancy_diffusivity_factor = float(grain_boundary_diffusivity)
        elif grain_boundary_diffusivity != 1.0:
            raise ValueError("BoothFissionGasRelease: grain_boundary_diffusivity needs grain_boundary='bubbles'.")
        if resolution != 1.0 and self.trapping != "speight":
            raise ValueError("BoothFissionGasRelease: resolution needs trapping='speight'.")

    def diffusivity(self, temperature, fission_rate):
        """The single-atom diffusion coefficient, m^2/s (times the
        trapping factor with ``trapping="constant"``, and the diffusivity
        factor of :meth:`set_factors`)."""
        if self.diffusion_coefficient is not None:
            given = self.diffusion_coefficient(np.asarray(temperature, dtype=float), np.asarray(fission_rate, dtype=float))
            return self.diffusivity_factor * self.trapping_factor * np.broadcast_to(given, np.shape(temperature)).astype(float)
        kT = BOLTZMANN * np.asarray(temperature, dtype=float)
        fission_rate = np.asarray(fission_rate, dtype=float)
        thermal = 7.6e-10 * np.exp(-4.86e-19 / kT)
        irradiation = self.irradiation_factor * 5.64e-25 * np.sqrt(np.maximum(fission_rate, 0.0)) * np.exp(-1.91e-19 / kT)
        return self.diffusivity_factor * self.trapping_factor * (thermal + irradiation + self.athermal_coefficient * fission_rate)

    def _speight(self, D, fission_rate, grain_gas, temperature=None):
        """The effective diffusion coefficient b D / (b + g) and the
        re-solution rate b, at the bubble density of the start of the step
        and the bubble radius consistent with the gas in the grain."""
        F = np.maximum(np.asarray(fission_rate, dtype=float), 0.0)
        c = self.resolution_factor * self.resolution_coefficient * np.pi * self.fragment_range
        if self.intragranular_bubbles == "white_tucker":
            # White and Tucker (1983), Eqs. (24), (25) and (27): the mean
            # radius and density of the bubbles fitted to Baker's data.
            T = np.asarray(temperature, dtype=float)
            R = 5.0e-10 * (1.0 + 106.0 * np.exp(-0.75 * 1.602176634e-19 / (BOLTZMANN * T)))
            N = np.maximum(1.52e27 / T - 3.3e23, 0.0)
            self.intragranular_bubble_radius, self.bubble_density = R, N
            g = 4.0 * np.pi * D * R * N
            b = c * (R + self.fragment_radius) ** 2 * F
            with np.errstate(divide="ignore", invalid="ignore"):
                factor = np.where(b + g > 0, b / (b + g), 1.0)
            return factor * D, b
        N = self.bubble_density
        R = self.intragranular_bubble_radius.copy()
        has = N > 0
        for _ in range(60):
            g = 4.0 * np.pi * D * (R + self.gas_atom_radius) * N
            b = c * (R + self.fragment_radius) ** 2 * F
            with np.errstate(divide="ignore", invalid="ignore"):
                share = np.where(b + g > 0, g / (b + g), 0.0)
                target = np.where(has, np.cbrt(3.0 * self.gas_atom_volume * share * np.maximum(grain_gas, 0.0) / (4.0 * np.pi * np.where(has, N, 1.0))), 0.0)
            if np.all(np.abs(target - R) <= 1e-6 * np.maximum(target, 1e-12)):
                R = target
                break
            R = 0.5 * (R + target)
        self.intragranular_bubble_radius = R
        g = 4.0 * np.pi * D * (R + self.gas_atom_radius) * N
        b = c * (R + self.fragment_radius) ** 2 * F
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(b + g > 0, b / (b + g), 1.0)
        return factor * D, b

    def saturation(self, temperature, pressure):
        """Atoms per m^3 of fuel that the grain boundaries hold at saturation."""
        per_area = self._saturation_prefactor / (BOLTZMANN * np.asarray(temperature, dtype=float)) * (self._capillary_pressure + pressure)
        return per_area * 3.0 / (2.0 * self.grain_radius)

    def intragranular(self) -> np.ndarray:
        """Gas in the grains, atoms per m^3 of fuel, element by element."""
        return self.coefficients @ self._concentration_weights + self.boundary_concentration

    def advance(self, dt, temperature, fission_rate, pressure, burnup=None):
        """Advance every element by ``dt`` with the temperature, fission rate
        and pressure held over the step; returns the atoms per m^3 of fuel
        released in the step, element by element.  ``burnup`` (FIMA, at the
        end of the step) is used by the micro-cracking of the grain faces."""
        source = self.fission_gas_yield * np.asarray(fission_rate, dtype=float)
        temperature = self.temperature_factor * np.asarray(temperature, dtype=float)
        D = self.diffusivity(temperature, fission_rate)
        if self.trapping == "speight":
            D, b = self._speight(D, fission_rate, self.intragranular() * 1.0, temperature)
        if self.trapping == "speight" and self.intragranular_bubbles == "nucleation":
            # The bubbles nucleate at 2 eta F and are destroyed at b N; exact
            # over the step with b held.
            nu = 2.0 * self.nucleation_factor * np.maximum(np.asarray(fission_rate, float), 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                equilibrium = np.where(b > 0, nu / b, 0.0)
                self.bubble_density = np.where(b > 0, equilibrium + (self.bubble_density - equilibrium) * np.exp(-b * dt), self.bubble_density + nu * dt)
        rate = D[:, None] * self._eigenvalue_factor
        decay = np.exp(-rate * dt)
        with np.errstate(divide="ignore", invalid="ignore"):
            growth = np.where(rate > 0, (1.0 - decay) / rate, dt)
        before = self.intragranular()
        psi0 = self.boundary_concentration
        if self.boundary_resolution > 0.0:
            # The grain concentration at the boundary is psi = b_gb delta N_f /
            # (2 D_eff), with N_f the gas per unit boundary area (White and
            # Tucker 1983, Eq. 33, written for the total gas with the
            # effective diffusivity).  The grain holds psi(t) + v(r, t) with
            # v(a) = 0, and v takes the source beta - dpsi/dt.  The arrival
            # at the boundary is linear in the end value psi1, which is
            # therefore solved for exactly with psi1 = c (G0 + arrival).
            F = np.maximum(np.asarray(fission_rate, dtype=float), 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                c = np.where(D > 0, self.boundary_resolution * F * (2.0 * self.grain_radius / 3.0) / (2.0 * D), 0.0)
            weights = self._source_coefficients * self._concentration_weights
            Q = growth @ weights  # response of the mean of v to a unit source
            V0 = (self.coefficients * decay) @ self._concentration_weights
            v_mean0 = before - psi0
            alpha0 = source * dt - V0 - source * Q + v_mean0
            alpha1 = Q / dt - 1.0
            G0 = np.maximum(self.boundary, 0.0)
            psi1 = np.maximum(c * (G0 + alpha0 - alpha1 * psi0) / (1.0 - c * alpha1), 0.0)
        else:
            psi1 = psi0
        effective_source = source - (psi1 - psi0) / dt
        self.coefficients = self.coefficients * decay + effective_source[:, None] * self._source_coefficients * growth
        self.boundary_concentration = psi1
        produced = source * dt
        self.produced += produced
        arrived = before + produced - self.intragranular()
        if self.faces is not None:
            release = self.faces.advance(dt, temperature, pressure, arrived, None if burnup is None else np.asarray(burnup, float) * self.mwd_per_kg_per_fima)
            self.boundary = self.faces.stored()
        else:
            self.boundary += arrived
            release = np.maximum(self.boundary - self.saturation(temperature, pressure), 0.0)
            self.boundary -= release
        self.released += release
        return release

    def release_fraction(self, volumes) -> float:
        """The fraction of the gas produced so far that has been released,
        over elements of the given volumes."""
        total = np.sum(self.produced * volumes)
        return float(np.sum(self.released * volumes) / total) if total > 0 else 0.0

    def gaseous_swelling(self) -> np.ndarray:
        r"""The volumetric gaseous swelling of the fuel held in the bubbles
        of this model, element by element (Pastore et al., Nucl. Eng. Des.
        256 (2013) 75, Eqs. 9 and 10):

        * intragranular bubbles, :math:`(\Delta V/V)_{ig} = N_{ig} \tfrac43
          \pi R_{ig}^3` (with ``trapping="speight"``);
        * grain-face bubbles, :math:`(\Delta V/V)_{gf} = \tfrac{3}{2 a}
          N_{gf} V_{gf}` (with ``grain_boundary="bubbles"``), the factor 1/2
          because two grains share a face bubble.

        With ``grain_boundary="saturation"`` the boundary gas is not held in
        bubbles of a known volume and contributes nothing."""
        swelling = np.zeros(len(self.produced))
        if self.trapping == "speight":
            swelling += 4.0 / 3.0 * np.pi * self.intragranular_bubble_radius**3 * self.bubble_density
        if self.faces is not None:
            swelling += self.faces.swelling()
        return swelling


class FractionFissionGasRelease:
    """Fission gas release given as a fraction of the gas produced, from a
    function ``(temperature, burnup) -> fraction`` (burnup in FIMA, arrays in
    and out), applied element by element and never allowed to decrease.  This
    is the form of empirical release correlations.

    ``fission_gas_yield`` (default 0.32) and ``xenon_fraction`` (default
    0.88) are as for :class:`BoothFissionGasRelease`."""

    def __init__(self, num_elements: int, release_fraction, fission_gas_yield: float = 0.32, xenon_fraction: float = 0.88):
        self.release_fraction_function = release_fraction
        self.produced = np.zeros(num_elements)
        self.released = np.zeros(num_elements)
        self.fission_gas_yield = float(fission_gas_yield)
        self.xenon_fraction = float(xenon_fraction)

    def advance(self, dt, temperature, fission_rate, pressure, burnup=None):
        """Advance by ``dt``; ``burnup`` is the local burnup in FIMA at the end
        of the step.  Returns the atoms per m^3 released in the step."""
        if burnup is None:
            raise ValueError(f"{type(self).__name__}.advance needs the local burnup.")
        self.produced += self.fission_gas_yield * np.asarray(fission_rate, dtype=float) * dt
        fraction = np.clip(np.asarray(self.release_fraction_function(np.asarray(temperature, dtype=float), np.asarray(burnup, dtype=float)), dtype=float), 0.0, 1.0)
        release = np.maximum(fraction * self.produced - self.released, 0.0)
        self.released += release
        return release

    def release_fraction(self, volumes) -> float:
        total = np.sum(self.produced * volumes)
        return float(np.sum(self.released * volumes) / total) if total > 0 else 0.0


class StormsFissionGasRelease(FractionFissionGasRelease):
    """Fission gas release from UN, from the empirical correlation of E. K.
    Storms (J. Nucl. Mater. 158 (1988) 119-129): the released fraction of the
    gas produced as a function of the temperature, the burnup and the
    density (:class:`FractionFissionGasRelease` with that correlation)."""

    def __init__(self, num_elements: int, theoretical_density_fraction: float = 0.95, fission_gas_yield: float = 0.32, xenon_fraction: float = 0.88):
        self.theoretical_density_fraction = float(theoretical_density_fraction)
        density = self.theoretical_density_fraction

        def storms(temperature, burnup):
            return properties.un_fission_gas_release_fraction(temperature, burnup, density)

        super().__init__(num_elements, storms, fission_gas_yield, xenon_fraction)
