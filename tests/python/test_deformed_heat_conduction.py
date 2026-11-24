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
* a rigid rotation, which must change nothing.
"""

from __future__ import annotations

import dualmesh
import numpy
import pytest

K, q0, H = 3.0, 1.0e6, 2.0e4
radius, height = 0.004, 0.002


def _heated_cylinder(strain, method="fem", surface="dirichlet", deformed=True):
    """A solid cylinder (r-z mesh) expanded by a uniform isotropic eigenstrain
    that leaves it stress-free, heated uniformly, and cooled at its lateral
    surface. The ends are insulated."""
    mesh = dualmesh.generate_rectangle_mesh(0.0, radius, 0.0, height, 16, 4)
    p = dualmesh.Problem(mesh, method=method, coordinates="axisymmetric")
    for v in ("disp_r", "disp_z", "temperature"):
        p.add_variable(v)
    p.add_property(
        "parsed_eigenstrain",
        eigenstrain_name="expansion",
        expression=str(strain),
        strain_type="linear",
        formulation="axisymmetric",
    )
    p.add_property(
        "small_strain_stress",
        displacements=["disp_r", "disp_z"],
        formulation="axisymmetric",
        youngs_modulus=1e11,
        poissons_ratio=0.3,
        eigenstrain_names=["expansion"],
    )
    for d in ("disp_r", "disp_z"):
        p.add_kernel("stress_divergence", variable=d, displacements=["disp_r", "disp_z"])
    p.add_boundary_condition("Dirichlet", "left", variable="disp_r", value=0.0)
    p.add_boundary_condition("Dirichlet", "bottom", variable="disp_z", value=0.0)
    on_deformed = {"deformation_gradient_property": "deformation_gradient"} if deformed else {}
    p.add_kernel(
        "heat_conduction",
        variable="temperature",
        thermal_conductivity=K,
        **on_deformed,
    )
    p.add_kernel("body_force", variable="temperature", value=q0)
    if surface == "dirichlet":
        p.add_boundary_condition(
            "Dirichlet",
            "right",
            variable="temperature",
            value=0.0,
        )
    else:
        p.add_boundary_condition(
            "Robin",
            "right",
            variable="temperature",
            transfer_coefficient=H,
            ambient_value=0.0,
            **on_deformed,
        )
    p.solve()
    return p


@pytest.mark.parametrize("method", ["fem", "dmcdm"])
def test_a_uniform_expansion_lowers_the_temperature_rise_by_one_plus_the_strain(method):
    """With the surface held at 0 the discrete problem is the undeformed one
    with k replaced by (1 + eps) k, so every temperature falls by exactly
    1 + eps. The displacement is checked to be the uniform expansion."""
    strain = 0.05
    deformed = _heated_cylinder(strain, method)
    undeformed = _heated_cylinder(strain, method, deformed=False)
    x = numpy.asarray(deformed.node_coordinates())
    assert numpy.asarray(deformed.values("disp_r")) == pytest.approx(strain * x[:, 0], abs=1e-12)
    assert numpy.asarray(deformed.values("disp_z")) == pytest.approx(strain * x[:, 1], abs=1e-12)
    ratio = numpy.asarray(deformed.values("temperature")) / (1 / (1 + strain))
    assert ratio == pytest.approx(numpy.asarray(undeformed.values("temperature")), rel=1e-9)
    # And the undeformed solution is q R^2 / (4 k) on the axis, to the
    # discretization error.
    axis = numpy.asarray(undeformed.values("temperature"))[x[:, 0] < 1e-12].mean()
    assert axis == pytest.approx(q0 * radius**2 / (4 * K), rel=5e-3)


def test_the_convective_flux_acts_on_the_deformed_surface():
    """All the heat, Q0 times the undeformed volume, leaves through the
    lateral surface, whose deformed area is (1 + eps)^2 times the undeformed
    one. The surface temperature is then Q0 R / (2 h (1 + eps)^2) exactly
    (the discrete energy balance is exact and the surface temperature is
    uniform)."""
    strain = 0.05
    p = _heated_cylinder(strain, surface="convective")
    x = numpy.asarray(p.node_coordinates())
    T = numpy.asarray(p.values("temperature"))
    surface = T[numpy.isclose(x[:, 0], radius)]
    assert surface == pytest.approx(q0 * radius / (2 * H * (1 + strain) ** 2), rel=1e-9)
    rise = T[x[:, 0] < 1e-12].mean() - surface.mean()
    assert rise == pytest.approx(q0 * radius**2 / (4 * K * (1 + strain)), rel=5e-3)


def test_a_rigid_rotation_changes_nothing():
    """A square turned by 30 degrees as a rigid body (the displacement
    (R - I) X on the boundary, a neo-Hookean body, so the interior follows
    without stress): the temperatures equal those of the unrotated square to
    round-off."""
    angle = numpy.radians(30.0)
    c, s = numpy.cos(angle), numpy.sin(angle)
    results = []
    for deformed in (True, False):
        mesh = dualmesh.generate_rectangle_mesh(0.0, 1.0, 0.0, 1.0, 6, 6)
        p = dualmesh.Problem(mesh)
        # The initial guess is the rotation itself, so that no element starts
        # inside out.
        rotate_x = f"({c} - 1)*x - {s}*y"
        rotate_y = f"{s}*x + ({c} - 1)*y"
        p.add_variable("ux", initial_condition=dualmesh.ParsedFunction(rotate_x))
        p.add_variable("uy", initial_condition=dualmesh.ParsedFunction(rotate_y))
        p.add_variable("temperature")
        p.add_property(
            "finite_strain_stress",
            displacements=["ux", "uy"],
            formulation="plane_strain",
            youngs_modulus=1e9,
            poissons_ratio=0.3,
            stress_update="neo-Hookean",
        )
        p.add_kernel(
            "stress_divergence",
            displacements=["ux", "uy"],
            stress_property="first_piola_kirchhoff_stress",
        )
        p.add_function("rotate_x", dualmesh.ParsedFunction(rotate_x))
        p.add_function("rotate_y", dualmesh.ParsedFunction(rotate_y))
        everywhere = ["left", "right", "top", "bottom"]
        p.add_boundary_condition(
            "Dirichlet",
            everywhere,
            variable="ux",
            value="rotate_x",
        )
        p.add_boundary_condition(
            "Dirichlet",
            everywhere,
            variable="uy",
            value="rotate_y",
        )
        on_deformed = {"deformation_gradient_property": "deformation_gradient"} if deformed else {}
        p.add_kernel(
            "heat_conduction",
            variable="temperature",
            thermal_conductivity=2.0,
            **on_deformed,
        )
        p.add_kernel("body_force", variable="temperature", value=5.0)
        p.add_boundary_condition(
            "Dirichlet",
            "left",
            variable="temperature",
            value=0.0,
        )
        p.add_boundary_condition(
            "Robin",
            "right",
            variable="temperature",
            transfer_coefficient=3.0,
            ambient_value=1.0,
            **on_deformed,
        )
        p.solve()
        x = numpy.asarray(p.node_coordinates())
        assert numpy.asarray(p.values("ux")) == pytest.approx(
            (c - 1) * x[:, 0] - s * x[:, 1], abs=1e-9
        )
        results.append(numpy.asarray(p.values("temperature")))
    assert results[0] == pytest.approx(results[1], abs=1e-10)


def test_the_property_must_have_nine_components():
    mesh = dualmesh.generate_line_mesh(0.0, 1.0, 4)
    p = dualmesh.Problem(mesh)
    p.add_variable("temperature")
    p.add_property("constant", properties={"k": 1.0})
    p.add_kernel("heat_conduction", variable="temperature", deformation_gradient_property="k")
    with pytest.raises(Exception, match="must have nine components"):
        p.solve()
