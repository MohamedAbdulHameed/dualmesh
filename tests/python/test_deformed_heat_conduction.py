# SPDX-License-Identifier: LGPL-2.1-or-later
"""Heat conduction through a deformed body, solved on the undeformed mesh.

The mesh never moves. With ``deformation_gradient_property`` the heat
equation of the deformed body is carried back to the mesh: the conductivity
becomes :math:`J k F^{-1} F^{-T}` and a boundary flux given per unit deformed
area is multiplied by :math:`da/dA = |J F^{-T} N|`. The source and the heat
capacity are per unit undeformed volume and do not change.

The tests use deformations for which the answer is known exactly:

* a uniform expansion :math:`F = (1 + \\varepsilon) I` of a solid cylinder
  heated from within. The deformed cylinder has the radius
  :math:`(1+\\varepsilon) R` and the length :math:`(1+\\varepsilon) H`. The
  heat per unit deformed length falls by :math:`1+\\varepsilon`, so the rise
  from the surface to the axis, :math:`q' / (4 \\pi k)`, falls by the same
  factor, and the lateral surface grows by :math:`(1+\\varepsilon)^2`;
* a rigid rotation, which must change nothing;
* a fuel rod, where the change is small: 1 % to 3 % of the temperature
  rise across the pellet.
"""

from __future__ import annotations

import dualmesh as dm
import numpy as np
import pytest

K, Q0, H = 3.0, 1.0e6, 2.0e4
RADIUS, HEIGHT = 0.004, 0.002


def _heated_cylinder(strain, method="fem", surface="dirichlet", deformed=True):
    """A solid cylinder (r-z mesh) expanded by a uniform isotropic eigenstrain
    that leaves it stress-free, heated uniformly, and cooled at its lateral
    surface. The ends are insulated."""
    mesh = dm.generate_rectangle_mesh(0.0, RADIUS, 0.0, HEIGHT, 16, 4)
    p = dm.Problem(mesh, method=method, coordinates="axisymmetric")
    for v in ("disp_r", "disp_z", "temperature"):
        p.add_variable(v)
    p.add_property("parsed_eigenstrain", "expansion", eigenstrain_name="expansion", expression=str(strain), strain_type="linear", formulation="axisymmetric")
    p.add_property("small_strain_stress", "stress", displacements=["disp_r", "disp_z"], formulation="axisymmetric", youngs_modulus=1e11, poissons_ratio=0.3, eigenstrain_names=["expansion"])
    for i, d in enumerate(("disp_r", "disp_z")):
        p.add_kernel("stress_divergence", f"eq_{d}", variable=d, component=i)
    p.add_boundary_condition("Dirichlet_boundary_condition", "axis", variable="disp_r", boundary=["left"], value=0.0)
    p.add_boundary_condition("Dirichlet_boundary_condition", "base", variable="disp_z", boundary=["bottom"], value=0.0)
    on_deformed = {"deformation_gradient_property": "deformation_gradient"} if deformed else {}
    p.add_kernel("heat_conduction", "conduction", variable="temperature", thermal_conductivity=K, **on_deformed)
    p.add_kernel("heat_source", "heat", variable="temperature", heat_source=Q0)
    if surface == "dirichlet":
        p.add_boundary_condition("Dirichlet_boundary_condition", "surface", variable="temperature", boundary=["right"], value=0.0)
    else:
        p.add_boundary_condition("convective_heat_flux_boundary_condition", "surface", variable="temperature", boundary=["right"], heat_transfer_coefficient=H, ambient_temperature=0.0, **on_deformed)
    p.solve(relative_tolerance=1e-12, absolute_tolerance=1e-12)
    return p


@pytest.mark.parametrize("method", ["fem", "dmcdm"])
def test_a_uniform_expansion_lowers_the_temperature_rise_by_one_plus_the_strain(method):
    """With the surface held at 0 the discrete problem is the undeformed one
    with k replaced by (1 + eps) k, so every temperature falls by exactly
    1 + eps. The displacement is checked to be the uniform expansion."""
    strain = 0.05
    deformed = _heated_cylinder(strain, method)
    undeformed = _heated_cylinder(strain, method, deformed=False)
    x = np.asarray(deformed.entity_points())
    assert np.asarray(deformed.values("disp_r")) == pytest.approx(strain * x[:, 0], abs=1e-12)
    assert np.asarray(deformed.values("disp_z")) == pytest.approx(strain * x[:, 1], abs=1e-12)
    ratio = np.asarray(deformed.values("temperature")) / (1 / (1 + strain))
    assert ratio == pytest.approx(np.asarray(undeformed.values("temperature")), rel=1e-9)
    # And the undeformed solution is q R^2 / (4 k) on the axis, to the
    # discretization error.
    axis = np.asarray(undeformed.values("temperature"))[x[:, 0] < 1e-12].mean()
    assert axis == pytest.approx(Q0 * RADIUS**2 / (4 * K), rel=5e-3)


def test_the_convective_flux_acts_on_the_deformed_surface():
    """All the heat, Q0 times the undeformed volume, leaves through the
    lateral surface, whose deformed area is (1 + eps)^2 times the undeformed
    one. The surface temperature is then Q0 R / (2 h (1 + eps)^2) exactly
    (the discrete energy balance is exact and the surface temperature is
    uniform)."""
    strain = 0.05
    p = _heated_cylinder(strain, surface="convective")
    x = np.asarray(p.entity_points())
    T = np.asarray(p.values("temperature"))
    surface = T[np.isclose(x[:, 0], RADIUS)]
    assert surface == pytest.approx(Q0 * RADIUS / (2 * H * (1 + strain) ** 2), rel=1e-9)
    rise = T[x[:, 0] < 1e-12].mean() - surface.mean()
    assert rise == pytest.approx(Q0 * RADIUS**2 / (4 * K * (1 + strain)), rel=5e-3)


def test_a_rigid_rotation_changes_nothing():
    """A square turned by 30 degrees as a rigid body (the displacement
    (R - I) X on the boundary, a neo-Hookean body, so the interior follows
    without stress): the temperatures equal those of the unrotated square to
    round-off."""
    angle = np.radians(30.0)
    c, s = np.cos(angle), np.sin(angle)
    results = []
    for deformed in (True, False):
        mesh = dm.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 6, 6)
        p = dm.Problem(mesh)
        # The initial guess is the rotation itself, so that no element starts
        # inside out.
        rotate_x = f"({c} - 1)*x - {s}*y"
        rotate_y = f"{s}*x + ({c} - 1)*y"
        p.add_variable("ux", initial_condition=dm.ParsedFunction(rotate_x))
        p.add_variable("uy", initial_condition=dm.ParsedFunction(rotate_y))
        p.add_variable("temperature")
        p.add_property("finite_strain_stress", "stress", displacements=["ux", "uy"], formulation="plane_strain", youngs_modulus=1e9, poissons_ratio=0.3, stress_update="neo_Hookean")
        for i, d in enumerate(("ux", "uy")):
            p.add_kernel("stress_divergence", f"eq_{d}", variable=d, component=i, stress_property="first_piola_kirchhoff_stress")
        p.add_function("rotate_x", dm.ParsedFunction(rotate_x))
        p.add_function("rotate_y", dm.ParsedFunction(rotate_y))
        everywhere = ["left", "right", "top", "bottom"]
        p.add_boundary_condition("Dirichlet_boundary_condition", "bx", variable="ux", boundary=everywhere, value="rotate_x")
        p.add_boundary_condition("Dirichlet_boundary_condition", "by", variable="uy", boundary=everywhere, value="rotate_y")
        on_deformed = {"deformation_gradient_property": "deformation_gradient"} if deformed else {}
        p.add_kernel("heat_conduction", "conduction", variable="temperature", thermal_conductivity=2.0, **on_deformed)
        p.add_kernel("heat_source", "heat", variable="temperature", heat_source=5.0)
        p.add_boundary_condition("Dirichlet_boundary_condition", "cold", variable="temperature", boundary=["left"], value=0.0)
        p.add_boundary_condition("convective_heat_flux_boundary_condition", "film", variable="temperature", boundary=["right"], heat_transfer_coefficient=3.0, ambient_temperature=1.0, **on_deformed)
        p.solve(relative_tolerance=1e-12, absolute_tolerance=1e-12)
        x = np.asarray(p.entity_points())
        assert np.asarray(p.values("ux")) == pytest.approx((c - 1) * x[:, 0] - s * x[:, 1], abs=1e-9)
        results.append(np.asarray(p.values("temperature")))
    assert results[0] == pytest.approx(results[1], abs=1e-10)


def test_the_property_must_have_nine_components():
    mesh = dm.generate_line_mesh(0.0, 1.0, 4)
    p = dm.Problem(mesh)
    p.add_variable("temperature")
    p.add_property("constant_property", "k", property_names=["k"], property_values=[1.0])
    p.add_kernel("heat_conduction", "conduction", variable="temperature", deformation_gradient_property="k")
    with pytest.raises(Exception, match="must have nine components"):
        p.solve()


@pytest.mark.parametrize("model", ["axisymmetric", "1.5d"])
def test_in_a_fuel_rod_the_deformed_body_runs_slightly_colder(model):
    """At 30 kW/m the pellet stack grows axially by about 1 %, so the heat per
    unit deformed length, and with it the rise from the pellet surface to the
    centre, fall by about as much. Accounting for the deformation lowers the
    centreline temperature by 1 % to 3 % of that rise."""
    from dualmesh import fuel

    def run(configuration):
        geometry = fuel.RodGeometry.from_diameters(8.19e-3, 8.36e-3, 9.50e-3, 0.05)
        rod = fuel.FuelRod(
            geometry,
            fuel.UO2Fuel(grain_radius=5.0e-6),
            fuel.ZircaloyCladding(),
            fuel.FillGas(pressure=2.0e6, plenum_volume=0.15e-6),
            fuel.ForcedConvection(inlet_temperature=565.0, pressure=15.5e6, mass_flux=3800.0, rod_pitch=12.6e-3),
            fuel.PowerHistory(linear_heat_rate=[1e3, 30e3], time=[0, 3600]),
            numerics=fuel.RodNumerics(model=model, heat_conduction_configuration=configuration, mesh=fuel.RodMesh(num_axial_elements=2)),
            output=fuel.RodOutput(),
        )
        return rod.run(report="none")

    deformed, undeformed = run("deformed"), run("undeformed")
    rise = undeformed.fuel_centerline_temperature[-1] - undeformed.fuel_surface_temperature[-1]
    drop = undeformed.fuel_centerline_temperature[-1] - deformed.fuel_centerline_temperature[-1]
    assert np.all(drop > 0.01 * rise)
    assert np.all(drop < 0.03 * rise)
    with pytest.raises(ValueError, match="deformed or undeformed"):
        fuel.RodNumerics(heat_conduction_configuration="current")
