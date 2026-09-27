# SPDX-License-Identifier: LGPL-2.1-or-later
r"""Stresses in the coating layers of a TRISO fuel particle under irradiation.

A TRISO particle is a small sphere of fuel (the kernel, a few hundred
micrometres across) wrapped in four coatings: a porous carbon buffer that
takes up fission gas and kernel swelling, an inner pyrocarbon layer (IPyC), a
silicon carbide layer (SiC) that is the main pressure boundary, and an outer
pyrocarbon layer (OPyC).  Under fast neutron irradiation the pyrocarbon
shrinks (and later swells) anisotropically and creeps, while the stiff SiC
responds elastically.  The internal gas pressure pushes on the IPyC.  The
tangential stress in the SiC decides whether the particle fails.

This module solves the spherically symmetric problem for any number of
bonded layers.

Kinematics and equilibrium
    With the radial displacement :math:`u(r)`, the strains are
    :math:`\varepsilon_r = du/dr` and :math:`\varepsilon_\theta =
    \varepsilon_\phi = u/r`.  Equilibrium,
    :math:`d\sigma_r/dr + 2(\sigma_r - \sigma_\theta)/r = 0`, is solved in weak
    form with quadratic finite elements,

    .. math::

       \int_{r_i}^{r_o} (\sigma_r\,\delta\varepsilon_r +
       2\sigma_\theta\,\delta\varepsilon_\theta)\,r^2\,dr =
       p_i r_i^2\,\delta u(r_i) - p_o r_o^2\,\delta u(r_o),

    where :math:`p_i` is the gas pressure on the inner surface of the first
    layer and :math:`p_o` the ambient pressure.  The layers are bonded, so
    the displacement is continuous across every interface.

Constitutive law
    The strain in each layer is the sum of the elastic strain, the
    irradiation creep strain, the irradiation induced dimensional change
    (called swelling here, negative for shrinkage) and the thermal strain,

    .. math::

       \boldsymbol\varepsilon = \mathbf C^{-1}\boldsymbol\sigma +
       \boldsymbol\varepsilon^{c} + \boldsymbol\varepsilon^{s}(\Phi) +
       \alpha\,(T - T_0)\,\mathbf 1 .

    Irradiation creep is linear in stress and proportional to the fast
    fluence :math:`\Phi`,

    .. math::

       \frac{d\varepsilon^c_r}{d\Phi} = K(T)\,[\sigma_r -
       \nu_c(\sigma_\theta + \sigma_\phi)],

    and similarly for the other directions, with the creep coefficient
    :math:`K` and the creep Poisson's ratio :math:`\nu_c` (0.5 conserves
    volume).  The creep strain is integrated with the trapezoidal rule
    (:math:`\theta = 1/2`) over fluence steps, which is second order
    accurate and needs one linear solve per step, since the law is linear.

Fluence
    Fluences are fast neutron fluences in :math:`\mathrm{n/m^2}` for
    :math:`E > 0.18` MeV, which is the convention of the pyrocarbon
    correlations.  Divide a fluence for :math:`E > 0.1` MeV by 1.10 to
    convert it (IAEA-TECDOC-1674, 2012, chapter 9, footnote 3).

The pyrocarbon correlations of the IAEA coordinated research project CRP-6
(IAEA-TECDOC-1674, Table 9.8) are included as :data:`CRP6_SWELLING` and
:func:`crp6_creep_coefficient`, and :func:`crp6_case` builds the eight
benchmark cases of that report.  Section 9.2 of the report gives analytic
solutions for cases 1 to 4c, which the tests reproduce.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Union

import numpy as np

__all__ = [
    "CRP6_SWELLING",
    "CoatingLayer",
    "ParticleHistory",
    "ParticleResult",
    "PyrocarbonSwelling",
    "TrisoParticle",
    "crp6_case",
    "crp6_creep_coefficient",
    "monte_carlo_failure_probability",
    "solve_particle",
    "weibull_failure_probability",
]

# Internal units: micrometres, MPa and fluence in units of 1e25 n/m^2.
_LENGTH = 1.0e6
_STRESS = 1.0e-6
_FLUENCE = 1.0e-25


@dataclass(frozen=True)
class PyrocarbonSwelling:
    r"""A pyrocarbon swelling rate correlation of the CRP-6 form.

    The rates of dimensional change in the radial and tangential directions
    are polynomials in the fast fluence :math:`x` (units of
    :math:`10^{25}\,\mathrm{n/m^2}`, :math:`E > 0.18` MeV),

    .. math:: \dot g = \frac{dg}{dx} = \sum_{i} A_i\,x^i,

    per unit of :math:`x` (IAEA-TECDOC-1674, Eq. 9.22).  Above
    ``breakpoint`` the rates are the constants ``radial_after`` and
    ``tangential_after`` (correlation (e) of Table 9.8).  The strain is the
    exact integral of the rate from zero fluence.
    """

    radial: tuple[float, ...]
    tangential: tuple[float, ...]
    breakpoint: float | None = None
    radial_after: float = 0.0
    tangential_after: float = 0.0

    @staticmethod
    def _integral(coefficients: Sequence[float], x: np.ndarray) -> np.ndarray:
        return sum(a * x ** (i + 1) / (i + 1) for i, a in enumerate(coefficients))

    @staticmethod
    def _polynomial(coefficients: Sequence[float], x: np.ndarray) -> np.ndarray:
        return sum(a * x**i for i, a in enumerate(coefficients))

    def rate(self, fluence) -> tuple[np.ndarray, np.ndarray]:
        """Radial and tangential swelling rates per n/m^2 at ``fluence`` (n/m^2)."""
        x = np.asarray(fluence, dtype=float) * _FLUENCE
        radial = self._polynomial(self.radial, x)
        tangential = self._polynomial(self.tangential, x)
        if self.breakpoint is not None:
            above = x > self.breakpoint
            radial = np.where(above, self.radial_after, radial)
            tangential = np.where(above, self.tangential_after, tangential)
        return radial * _FLUENCE, tangential * _FLUENCE

    def strain(self, fluence) -> tuple[np.ndarray, np.ndarray]:
        """Radial and tangential swelling strains accumulated up to ``fluence``."""
        x = np.asarray(fluence, dtype=float) * _FLUENCE
        if self.breakpoint is None:
            return self._integral(self.radial, x), self._integral(self.tangential, x)
        xb = np.minimum(x, self.breakpoint)
        extra = np.maximum(x - self.breakpoint, 0.0)
        radial = self._integral(self.radial, xb) + self.radial_after * extra
        tangential = self._integral(self.tangential, xb) + self.tangential_after * extra
        return radial, tangential


CRP6_SWELLING: dict[str, PyrocarbonSwelling] = {
    "a": PyrocarbonSwelling(
        radial=(-2.22642e-2, 2.00861e-2, -7.77024e-3, 1.36334e-3),
        tangential=(-1.91253e-2, 2.63307e-3, 1.69251e-3, -3.53804e-4),
    ),
    "b": PyrocarbonSwelling(
        radial=(-2.12522e-2, 1.83715e-2, -5.05553e-3, 7.27026e-4),
        tangential=(-1.79113e-2, -3.42182e-3, 5.03465e-3, -8.88086e-4),
    ),
    "c": PyrocarbonSwelling(
        radial=(-1.80613e-2, 9.82884e-3, -2.25937e-3, 4.03266e-4),
        tangential=(-1.78392e-2, 1.71315e-3, 2.32979e-3, -4.91648e-4),
    ),
    "e": PyrocarbonSwelling(
        radial=(-1.43234e-1, 2.62692e-1, -1.74247e-1, 5.67549e-2, -8.36313e-3, 4.52013e-4),
        tangential=(-3.24737e-2, 9.07826e-3, -2.10029e-3, 1.30457e-4),
        breakpoint=6.08,
        radial_after=0.0954,
        tangential_after=-0.0249,
    ),
    "f": PyrocarbonSwelling(
        radial=(-2.13483e-2, 1.64999e-2, -3.80252e-3, 4.73765e-4),
        tangential=(-1.83549e-2, -3.29740e-3, 5.47396e-3, -1.03249e-3),
    ),
}
"""Pyrocarbon swelling rate correlations (a), (b), (c), (e) and (f) of
IAEA-TECDOC-1674 (2012), Table 9.8.  The report uses (a) for a bacon
anisotropy factor (BAF) of 1.03 at 1273 K, (b) for BAF 1.06 and (c) for the
temperature cycling case 8."""


def crp6_creep_coefficient(temperature) -> np.ndarray:
    r"""Pyrocarbon irradiation creep coefficient, correlation (d) of CRP-6.

    :math:`K = 4.386\times10^{-4} - 9.70\times10^{-7}\,T + 8.0294\times
    10^{-10}\,T^2` in :math:`(\mathrm{MPa}\cdot10^{25}\,\mathrm{n/m^2})^{-1}`
    with :math:`T` in degrees Celsius (IAEA-TECDOC-1674, Eq. 9.23 and Table
    9.8).  It is twice the creep coefficient of the CEGA report, after the
    NPR experiments.  At 1273 K it gives :math:`2.72\times10^{-4}`, the
    constant of the isothermal cases.

    Returns the coefficient in SI units, :math:`1/(\mathrm{Pa}\cdot
    \mathrm{n/m^2})`.
    """
    t = np.asarray(temperature, dtype=float) - 273.15
    k = 4.386e-4 - 9.70e-7 * t + 8.0294e-10 * t * t
    return k * _STRESS * _FLUENCE


SwellingSpec = Union[str, float, PyrocarbonSwelling, Callable, None]


@dataclass(frozen=True)
class CoatingLayer:
    """One coating layer of the particle.

    Attributes:
        name: A label used in the results, for example ``"IPyC"``.
        thickness: Layer thickness (m).
        youngs_modulus: Young's modulus (Pa).
        poissons_ratio: Poisson's ratio.
        thermal_expansion: Linear thermal expansion coefficient (1/K),
            isotropic.  The default 0 is right for isothermal problems.
        creep_coefficient: Irradiation creep coefficient :math:`K`
            (1/(Pa n/m^2)), a number or a function of the temperature (K).
            ``None`` (the default) makes the layer purely elastic, which is
            the usual assumption for SiC.
        creep_poissons_ratio: The creep Poisson's ratio :math:`\\nu_c`.  The
            default 0.5 conserves volume and is the CRP-6 value for
            pyrocarbon.
        swelling: The irradiation induced dimensional change.  ``None`` (the
            default) for none, a CRP-6 correlation name (``"a"``, ``"b"``,
            ``"c"``, ``"e"`` or ``"f"``), a :class:`PyrocarbonSwelling`, a
            number (an isotropic, constant rate per n/m^2) or a function of
            the fluence that returns the radial and tangential strains.
    """

    name: str
    thickness: float
    youngs_modulus: float
    poissons_ratio: float
    thermal_expansion: float = 0.0
    creep_coefficient: float | Callable | None = None
    creep_poissons_ratio: float = 0.5
    swelling: SwellingSpec = None

    def __post_init__(self):
        if self.thickness <= 0.0:
            raise ValueError(f"layer {self.name}: thickness must be positive")
        if self.youngs_modulus <= 0.0:
            raise ValueError(f"layer {self.name}: Young's modulus must be positive")
        if not -1.0 < self.poissons_ratio < 0.5:
            raise ValueError(f"layer {self.name}: Poisson's ratio must be in (-1, 0.5)")
        if isinstance(self.swelling, str) and self.swelling not in CRP6_SWELLING:
            raise ValueError(
                f"layer {self.name}: unknown swelling correlation {self.swelling!r}, "
                f"expected one of {sorted(CRP6_SWELLING)}"
            )

    def swelling_strain(self, fluence: float) -> tuple[float, float]:
        """Radial and tangential swelling strains at ``fluence`` (n/m^2)."""
        s = self.swelling
        if s is None:
            return 0.0, 0.0
        if isinstance(s, str):
            s = CRP6_SWELLING[s]
        if isinstance(s, PyrocarbonSwelling):
            radial, tangential = s.strain(fluence)
            return float(radial), float(tangential)
        if callable(s):
            radial, tangential = s(fluence)
            return float(radial), float(tangential)
        return float(s) * fluence, float(s) * fluence

    def creep(self, temperature: float) -> float:
        """Creep coefficient at ``temperature`` (1/(Pa n/m^2)), 0 if none."""
        k = self.creep_coefficient
        if k is None:
            return 0.0
        return float(k(temperature)) if callable(k) else float(k)


@dataclass(frozen=True)
class TrisoParticle:
    """Geometry and coating layers of a particle.

    Attributes:
        kernel_diameter: Diameter of the fuel kernel (m).
        buffer_thickness: Thickness of the porous buffer (m).  The buffer
            does not carry load in this model.  It only sets the inner
            radius of the first stressed layer.
        layers: The load bearing layers from the inside out, bonded to each
            other.
    """

    kernel_diameter: float
    buffer_thickness: float
    layers: tuple[CoatingLayer, ...]

    def __post_init__(self):
        if not self.layers:
            raise ValueError("a particle needs at least one coating layer")
        names = [layer.name for layer in self.layers]
        if len(set(names)) != len(names):
            raise ValueError(f"layer names must be unique, got {names}")

    @property
    def inner_radius(self) -> float:
        """Inner radius of the first stressed layer (m)."""
        return 0.5 * self.kernel_diameter + self.buffer_thickness

    def radii(self) -> np.ndarray:
        """Radii of the layer boundaries (m), from the inside out."""
        return self.inner_radius + np.concatenate(
            ([0.0], np.cumsum([layer.thickness for layer in self.layers]))
        )


def _as_function(value) -> Callable[[float], float]:
    if callable(value):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        xs = np.asarray(value[0], dtype=float)
        ys = np.asarray(value[1], dtype=float)
        return lambda f: float(np.interp(f, xs, ys))
    constant = float(value)
    return lambda f: constant


@dataclass(frozen=True)
class ParticleHistory:
    """Irradiation conditions as functions of the fast fluence.

    Attributes:
        end_fluence: Final fast fluence (n/m^2, E > 0.18 MeV).  Zero gives a
            single elastic solution.
        internal_pressure: Gas pressure on the inner surface of the first
            layer (Pa).  A number, a function of the fluence, or a pair of
            sequences (fluence, pressure) that is interpolated linearly.
        temperature: Particle temperature (K), in the same three forms.  The
            particle is isothermal.
        ambient_pressure: Pressure on the outer surface (Pa).  The default
            0.1 MPa is the CRP-6 value.
        stress_free_temperature: Temperature at which the thermal strain is
            zero (K).  ``None`` (the default) uses the temperature at zero
            fluence.
    """

    end_fluence: float
    internal_pressure: float | Callable | tuple = 0.0
    temperature: float | Callable | tuple = 1273.15
    ambient_pressure: float = 0.1e6
    stress_free_temperature: float | None = None

    def pressure_at(self, fluence: float) -> float:
        return _as_function(self.internal_pressure)(fluence)

    def temperature_at(self, fluence: float) -> float:
        return _as_function(self.temperature)(fluence)


@dataclass
class ParticleResult:
    """Stress histories of a particle calculation.

    All stresses are in Pa and all fluences in n/m^2 (E > 0.18 MeV).

    Attributes:
        fluence: The fluence at each output step.
        inner_tangential_stress: Tangential stress at the inner surface of
            each layer, keyed by layer name.
        outer_tangential_stress: Tangential stress at the outer surface of
            each layer.
        interface_radial_stress: Radial stress at each interface between
            two layers, keyed ``"inner/outer"`` (for example ``"IPyC/SiC"``).
        radius: Radii of the output points at the end of the calculation
            (m), undeformed.
        radial_stress: Radial stress at those points at the end (Pa).
        tangential_stress: Tangential stress at those points at the end.
        layer_of_point: Layer index of each output point.
        layers: The layer names, from the inside out.
        tangential_stress_history: Tangential stress at the output points
            at every step (Pa), an array of shape (steps + 1, points).
    """

    fluence: np.ndarray
    inner_tangential_stress: dict[str, np.ndarray]
    outer_tangential_stress: dict[str, np.ndarray]
    interface_radial_stress: dict[str, np.ndarray]
    radius: np.ndarray
    radial_stress: np.ndarray
    tangential_stress: np.ndarray
    layer_of_point: np.ndarray
    layers: tuple[str, ...] = field(default_factory=tuple)
    tangential_stress_history: np.ndarray | None = None

    def maximum_tangential_stress(self, layer: str) -> float:
        """Largest inner surface tangential stress of ``layer`` over the history."""
        return float(np.max(self.inner_tangential_stress[layer]))

    def minimum_tangential_stress(self, layer: str) -> float:
        """Smallest (most compressive) inner surface tangential stress of ``layer``."""
        return float(np.min(self.inner_tangential_stress[layer]))


def _elastic_matrix(e: float, nu: float) -> np.ndarray:
    c = e / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array([[1.0 - nu, nu, nu], [nu, 1.0 - nu, nu], [nu, nu, 1.0 - nu]], dtype=float)


def _creep_matrix(nu_c: float) -> np.ndarray:
    return np.array([[1.0, -nu_c, -nu_c], [-nu_c, 1.0, -nu_c], [-nu_c, -nu_c, 1.0]])


_GAUSS_POINTS = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
_GAUSS_WEIGHTS = np.array([5.0, 8.0, 5.0]) / 9.0


def _shape(xi: np.ndarray):
    n = np.stack([0.5 * xi * (xi - 1.0), 1.0 - xi * xi, 0.5 * xi * (xi + 1.0)], axis=-1)
    dn = np.stack([xi - 0.5, -2.0 * xi, xi + 0.5], axis=-1)
    return n, dn


def solve_particle(
    particle: TrisoParticle,
    history: ParticleHistory,
    *,
    steps: int = 600,
    elements_per_layer: int = 32,
    theta: float = 0.5,
) -> ParticleResult:
    """Integrate the layer stresses of ``particle`` over ``history``.

    Args:
        particle: Geometry and layers.
        history: Pressure and temperature as functions of the fluence.
        steps: Number of equal fluence steps (default 600).  The fluence
            integration error of the CRP-6 cases is below 0.01 MPa with the
            default.  A calculation with zero end fluence is a single
            elastic solution.
        elements_per_layer: Quadratic elements per layer (default 32).  The
            surface stresses converge with the square of the element size.
            The default brings the CRP-6 cases within 0.03 MPa of the
            closed form solutions.  The error grows slowly with fluence,
            because the stress is the small difference between the total
            strain and a growing creep and swelling strain.
        theta: Time integration parameter of the creep law, 0.5 for the
            trapezoidal rule (default) and 1 for backward Euler.

    Returns:
        A :class:`ParticleResult`.
    """
    if steps < 1 or elements_per_layer < 1:
        raise ValueError("steps and elements_per_layer must be positive")
    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta must be between 0 and 1")

    radii = particle.radii() * _LENGTH
    nlayers = len(particle.layers)
    nel = nlayers * elements_per_layer
    nodes = 2 * nel + 1

    # Element geometry.
    ends = np.concatenate(
        [np.linspace(radii[k], radii[k + 1], elements_per_layer + 1)[:-1] for k in range(nlayers)]
        + [[radii[-1]]]
    )
    element_layer = np.repeat(np.arange(nlayers), elements_per_layer)

    # Evaluation points: three Gauss points and the two element ends.
    xi_all = np.concatenate([_GAUSS_POINTS, [-1.0, 1.0]])
    n_all, dn_all = _shape(xi_all)
    npt = len(xi_all)
    h = np.diff(ends)
    r_pts = 0.5 * (ends[:-1, None] + ends[1:, None]) + 0.5 * h[:, None] * xi_all[None, :]
    jac = 0.5 * h
    # B matrices at every point: shape (nel, npt, 3, 3 local dofs).
    bmat = np.zeros((nel, npt, 3, 3))
    bmat[:, :, 0, :] = dn_all[None, :, :] / jac[:, None, None]
    bmat[:, :, 1, :] = n_all[None, :, :] / r_pts[:, :, None]
    bmat[:, :, 2, :] = bmat[:, :, 1, :]
    weight = np.zeros((nel, npt))
    weight[:, :3] = _GAUSS_WEIGHTS[None, :] * jac[:, None] * r_pts[:, :3] ** 2
    dofs = 2 * np.arange(nel)[:, None] + np.arange(3)[None, :]

    elastic = [
        _elastic_matrix(layer.youngs_modulus * _STRESS, layer.poissons_ratio)
        for layer in particle.layers
    ]
    creep_mat = [_creep_matrix(layer.creep_poissons_ratio) for layer in particle.layers]

    t_ref = history.stress_free_temperature
    if t_ref is None:
        t_ref = history.temperature_at(0.0)

    def eigen(layer_index: int, fluence: float, temperature: float) -> np.ndarray:
        layer = particle.layers[layer_index]
        sr, st = layer.swelling_strain(fluence)
        th = layer.thermal_expansion * (temperature - t_ref)
        return np.array([sr + th, st + th, st + th])

    creep_strain = np.zeros((nel, npt, 3))
    stress = np.zeros((nel, npt, 3))

    def step(f_old, f_new, first):
        nonlocal creep_strain, stress
        t_new = history.temperature_at(f_new)
        t_mid = history.temperature_at(0.5 * (f_old + f_new))
        dphi = (f_new - f_old) * _FLUENCE
        c_layers = np.empty((nlayers, 3, 3))
        inc_layers = np.empty((nlayers, 3, 3))
        base = np.empty((nlayers, 3))
        for k, layer in enumerate(particle.layers):
            kdphi = layer.creep(t_mid) / (_STRESS * _FLUENCE) * dphi
            a = np.eye(3) + theta * kdphi * elastic[k] @ creep_mat[k]
            c_layers[k] = np.linalg.solve(a, elastic[k])
            inc_layers[k] = kdphi * creep_mat[k]
            base[k] = eigen(k, f_new, t_new)
        ca = c_layers[element_layer]
        inc = inc_layers[element_layer]
        eig = (
            creep_strain
            + (1.0 - theta) * np.einsum("eij,epj->epi", inc, stress)
            + base[element_layer][:, None, :]
        )

        bq = bmat[:, :3]
        wq = weight[:, :3]
        ke = np.einsum("eq,eqia,eij,eqjb->eab", wq, bq, ca, bq)
        fe = np.einsum("eq,eqia,eij,eqj->ea", wq, bq, ca, eig[:, :3])
        kmat = np.zeros((nodes, nodes))
        rhs = np.zeros(nodes)
        np.add.at(kmat, (dofs[:, :, None], dofs[:, None, :]), ke)
        np.add.at(rhs, dofs, fe)
        rhs[0] += history.pressure_at(f_new) * _STRESS * radii[0] ** 2
        rhs[-1] -= history.ambient_pressure * _STRESS * radii[-1] ** 2
        u = np.linalg.solve(kmat, rhs)

        strain = np.einsum("epij,ej->epi", bmat, u[dofs])
        new_stress = np.einsum("eij,epj->epi", ca, strain - eig)
        if not first:
            creep_strain = creep_strain + np.einsum(
                "eij,epj->epi", inc, (1.0 - theta) * stress + theta * new_stress
            )
        stress = new_stress

    fluences = np.linspace(0.0, history.end_fluence, steps + 1 if history.end_fluence > 0 else 1)
    names = tuple(layer.name for layer in particle.layers)
    first_el = [k * elements_per_layer for k in range(nlayers)]
    last_el = [(k + 1) * elements_per_layer - 1 for k in range(nlayers)]
    inner = {n: [] for n in names}
    outer = {n: [] for n in names}
    interface_names = [f"{names[k]}/{names[k + 1]}" for k in range(nlayers - 1)]
    interface = {n: [] for n in interface_names}

    order = np.argsort(r_pts[:, :3].ravel(), kind="stable")
    profile_history = []

    def record():
        profile_history.append(stress[:, :3, 1].ravel()[order] / _STRESS)
        for k, n in enumerate(names):
            inner[n].append(stress[first_el[k], 3, 1] / _STRESS)
            outer[n].append(stress[last_el[k], 4, 1] / _STRESS)
        for k, n in enumerate(interface_names):
            a = stress[last_el[k], 4, 0]
            b = stress[first_el[k + 1], 3, 0]
            interface[n].append(0.5 * (a + b) / _STRESS)

    step(0.0, 0.0, True)
    record()
    for f_old, f_new in zip(fluences[:-1], fluences[1:]):
        step(f_old, f_new, False)
        record()

    radius = r_pts[:, :3].ravel()[order] / _LENGTH
    return ParticleResult(
        fluence=fluences,
        inner_tangential_stress={n: np.array(v) for n, v in inner.items()},
        outer_tangential_stress={n: np.array(v) for n, v in outer.items()},
        interface_radial_stress={n: np.array(v) for n, v in interface.items()},
        radius=radius,
        radial_stress=stress[:, :3, 0].ravel()[order] / _STRESS,
        tangential_stress=stress[:, :3, 1].ravel()[order] / _STRESS,
        layer_of_point=np.repeat(element_layer, 3)[order],
        layers=names,
        tangential_stress_history=np.array(profile_history),
    )


def weibull_failure_probability(
    radius: np.ndarray,
    tangential_stress: np.ndarray,
    characteristic_strength: float,
    modulus: float,
) -> float:
    r"""Failure probability of a spherical shell from the Weibull weakest link model.

    .. math::

       P_f = 1 - \exp\left[-\int_V \left(\frac{\langle\sigma_\theta
       \rangle}{\sigma_0}\right)^m dV\right],\qquad dV = 4\pi r^2\,dr,

    where :math:`\langle\sigma\rangle = \max(\sigma, 0)` so that only tension
    contributes, :math:`m` is the Weibull modulus and :math:`\sigma_0` the
    characteristic strength in :math:`\mathrm{Pa\cdot m^{3/m}}` (W. Weibull,
    J. Appl. Mech. 18 (1951) 293).  The tangential stress is used because it
    is the largest principal stress in a pressurised sphere.  The integral
    is evaluated with the trapezoidal rule over the given points.

    Args:
        radius: Radii of the points in the layer (m), increasing.
        tangential_stress: Tangential stress at those points (Pa).
        characteristic_strength: :math:`\sigma_0` (Pa m^(3/m)).
        modulus: :math:`m`.
    """
    r = np.asarray(radius, dtype=float)
    s = np.maximum(np.asarray(tangential_stress, dtype=float), 0.0)
    integrand = (s / characteristic_strength) ** modulus * 4.0 * np.pi * r * r
    integral = float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r)))
    return 1.0 - float(np.exp(-integral))


def monte_carlo_failure_probability(
    particle: TrisoParticle,
    history: ParticleHistory,
    *,
    characteristic_strength: float,
    modulus: float,
    layer: str = "SiC",
    samples: int = 100,
    kernel_diameter_sd: float = 0.0,
    buffer_thickness_sd: float = 0.0,
    thickness_sd: dict[str, float] | None = None,
    seed: int = 0,
    **solver_options,
) -> tuple[float, float]:
    r"""Failure fraction of a particle population by Monte Carlo sampling.

    Each sample draws the kernel diameter, the buffer thickness and the
    thickness of each layer from normal distributions around the values of
    ``particle`` (truncated at 5 % of the mean), solves the stress history
    and evaluates the Weibull failure probability of ``layer`` at the most
    critical step,

    .. math::

       P_{f,j} = 1 - \exp\left[-\max_n \int_V \left(\frac{\langle
       \sigma_\theta^n\rangle}{\sigma_0}\right)^m dV\right].

    The expected failure fraction of the population is the mean of
    :math:`P_{f,j}` over the samples, since the strength of each particle is
    an independent Weibull draw.  The maximum over the steps assumes that a
    layer that has survived a stress survives it again, which is the usual
    weakest link assumption for a stress history.

    Args:
        particle: The mean particle.
        history: The irradiation history.
        characteristic_strength: Weibull :math:`\sigma_0` of ``layer``
            (Pa m^(3/m)).
        modulus: Weibull modulus :math:`m`.
        layer: The layer that fails, ``"SiC"`` by default.
        samples: Number of sampled particles (default 100).
        kernel_diameter_sd: Standard deviation of the kernel diameter (m),
            0 by default.
        buffer_thickness_sd: Standard deviation of the buffer thickness (m),
            0 by default.
        thickness_sd: Standard deviation of the thickness of each layer (m),
            keyed by layer name.  Layers not named keep their thickness.
        seed: Seed of the random number generator (default 0), so that a
            calculation is reproducible.
        **solver_options: Passed to :func:`solve_particle`.

    Returns:
        The mean failure probability and its standard error.
    """
    names = [lay.name for lay in particle.layers]
    if layer not in names:
        raise ValueError(f"unknown layer {layer!r}, expected one of {names}")
    thickness_sd = dict(thickness_sd or {})
    unknown = set(thickness_sd) - set(names)
    if unknown:
        raise ValueError(f"thickness_sd names unknown layers {sorted(unknown)}")
    rng = np.random.default_rng(seed)

    def draw(mean, sd):
        return max(rng.normal(mean, sd), 0.05 * mean) if sd > 0.0 else mean

    index = names.index(layer)
    probabilities = np.empty(samples)
    for j in range(samples):
        layers = tuple(
            replace(lay, thickness=draw(lay.thickness, thickness_sd.get(lay.name, 0.0)))
            for lay in particle.layers
        )
        sample = TrisoParticle(
            draw(particle.kernel_diameter, kernel_diameter_sd),
            draw(particle.buffer_thickness, buffer_thickness_sd),
            layers,
        )
        result = solve_particle(sample, history, **solver_options)
        inside = result.layer_of_point == index
        r = result.radius[inside]
        s = np.maximum(result.tangential_stress_history[:, inside], 0.0)
        integrand = (s / characteristic_strength) ** modulus * 4.0 * np.pi * r * r
        integrals = np.sum(0.5 * (integrand[:, 1:] + integrand[:, :-1]) * np.diff(r), axis=1)
        probabilities[j] = 1.0 - np.exp(-np.max(integrals))
    mean = float(np.mean(probabilities))
    error = float(np.std(probabilities, ddof=1) / np.sqrt(samples)) if samples > 1 else 0.0
    return mean, error


# ---------------------------------------------------------------------------
# CRP-6 benchmark cases 1 to 8 (IAEA-TECDOC-1674, section 9.2).

_PYC_E = 3.96e10
_PYC_NU = 0.33
_SIC_E = 3.70e11
_SIC_NU = 0.13
_CRP6_K = 2.71e-4 * _STRESS * _FLUENCE
_CRP6_FLUENCE = 3.0e25
_CYCLE = 0.3e25


def _case8_temperature(fluence: float) -> float:
    """Temperature of CRP-6 case 8: ten ramps from 873 K to 1273 K.

    Each cycle is 0.3e25 n/m^2 long.  The temperature rises linearly over
    the first 0.29e25 and falls back to 873 K over the last 0.01e25, which
    is the resolution of the pressure table of the report.
    """
    x = fluence / _CYCLE
    k = min(int(np.floor(x)), 9)
    local = (fluence - k * _CYCLE) * _FLUENCE
    if local <= 0.29:
        return 873.0 + 400.0 * local / 0.29
    return 1273.0 - 400.0 * (local - 0.29) / 0.01


_CASE8_PRESSURE = (
    np.array(
        [0, 0.29, 0.30, 0.59, 0.60, 0.89, 0.90, 1.19, 1.20, 1.49, 1.50]
        + [1.79, 1.80, 2.09, 2.10, 2.39, 2.40, 2.69, 2.70, 2.99, 3.00]
    )
    * 1e25,
    np.array(
        [0.0, 0.14, 0.02, 0.94, 0.04, 2.59, 0.07, 4.87, 0.10, 7.64, 0.14]
        + [10.79, 0.20, 14.26, 0.26, 17.99, 0.33, 21.96, 0.41, 26.13, 0.50]
    )
    * 1e6,
)


def crp6_case(case: str) -> tuple[TrisoParticle, ParticleHistory]:
    """Particle and history of CRP-6 benchmark case ``case``.

    ``case`` is one of ``"1"``, ``"2"``, ``"3"``, ``"4a"``, ``"4b"``,
    ``"4c"``, ``"4d"``, ``"5"``, ``"6"``, ``"7"`` and ``"8"``.  The
    specification is IAEA-TECDOC-1674 (2012), Tables 9.5 to 9.8: a kernel of
    500 micrometres (350 in case 5), a buffer of 100, IPyC 40 (90 in the
    BISO case 2), SiC 35 and OPyC 40 (cases 5 to 8).  PyC has
    :math:`E = 39.6` GPa and :math:`\\nu = 0.33`, SiC :math:`E = 370` GPa and
    :math:`\\nu = 0.13`.  The PyC creep coefficient is
    :math:`2.71\\times10^{-4}\\,(\\mathrm{MPa}\\cdot10^{25}\\,\\mathrm{n/m^2})^{-1}`
    with a creep Poisson's ratio of 0.5.  The ambient pressure is 0.1 MPa.
    """
    case = str(case).lower()
    valid = ("1", "2", "3", "4a", "4b", "4c", "4d", "5", "6", "7", "8")
    if case not in valid:
        raise ValueError(f"unknown CRP-6 case {case!r}, expected one of {valid}")

    creep = None if case in ("1", "2", "3", "4a") else _CRP6_K
    swelling: SwellingSpec = None
    if case in ("4a", "4c"):
        swelling = -0.005 * _FLUENCE
    elif case in ("4d", "5", "6"):
        swelling = "a"
    elif case == "7":
        swelling = "b"
    elif case == "8":
        swelling = "c"
    alpha_pyc = 5.35e-6 if case == "8" else 5.5e-6
    if case == "8":
        creep = crp6_creep_coefficient

    def pyc(name, thickness):
        return CoatingLayer(
            name,
            thickness,
            _PYC_E,
            _PYC_NU,
            thermal_expansion=alpha_pyc,
            creep_coefficient=creep,
            creep_poissons_ratio=0.5,
            swelling=swelling,
        )

    sic = CoatingLayer("SiC", 35e-6, _SIC_E, _SIC_NU, thermal_expansion=4.9e-6)
    kernel = 350e-6 if case == "5" else 500e-6
    if case == "1":
        layers = (sic,)
    elif case == "2":
        layers = (pyc("IPyC", 90e-6),)
    elif case in ("3", "4a", "4b", "4c", "4d"):
        layers = (pyc("IPyC", 40e-6), sic)
    else:
        layers = (pyc("IPyC", 40e-6), sic, pyc("OPyC", 40e-6))
    particle = TrisoParticle(kernel, 100e-6, layers)

    end = 0.0 if case in ("1", "2", "3") else _CRP6_FLUENCE
    temperature: float | Callable = 1273.0
    pressure: float | tuple = 25.0e6
    if case == "5":
        pressure = ((0.0, _CRP6_FLUENCE), (0.0, 15.54e6))
    elif case in ("6", "7"):
        pressure = ((0.0, _CRP6_FLUENCE), (0.0, 26.20e6))
    elif case == "8":
        pressure = _CASE8_PRESSURE
        temperature = _case8_temperature
    history = ParticleHistory(
        end_fluence=end,
        internal_pressure=pressure,
        temperature=temperature,
        ambient_pressure=0.1e6,
        stress_free_temperature=873.0 if case == "8" else None,
    )
    return particle, history
