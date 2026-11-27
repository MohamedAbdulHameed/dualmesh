# SPDX-License-Identifier: LGPL-2.1-or-later
"""Members in space: trusses on line meshes and shells of flat facets on surface meshes.

A shell on a flat mesh is the Mindlin plate, for a graded section and a thermal strain too, and
stays it when the plate is turned in space; the shell converges to the Scordelis-Lo roof and the
pinched cylinder of the MacNeal-Harder set; a truss carries its loads by axial forces only, with
the closed forms of a tripod and a hanging bar."""

from __future__ import annotations

import math

import dualmesh
import numpy
import pytest
from dualmesh._core import InputError

methods = ["FEM", "DMCDM"]
rotation = numpy.array(
    [[math.cos(0.4), 0, math.sin(0.4)], [0, 1, 0], [-math.sin(0.4), 0, math.cos(0.4)]]
) @ numpy.array([[1, 0, 0], [0, math.cos(1.1), -math.sin(1.1)], [0, math.sin(1.1), math.cos(1.1)]])
shell_variables = ["disp_x", "disp_y", "disp_z", "rotation_x", "rotation_y", "rotation_z"]


def clamped_shell(points, method, load, thickness=0.01):
    """A square of side 1 meshed 6 x 6 at the given points, clamped all round, as a shell."""
    plane = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 6, 6)
    cells = numpy.array([plane.element_nodes(e) for e in range(plane.num_elements)])
    mesh = dualmesh.mesh_from_arrays(points, [("Quad4", cells)], dimension=3)
    flat = numpy.asarray(plane.points())[:, :2]
    edge = [n for n, (x, y) in enumerate(flat) if min(x, y, 1 - x, 1 - y) < 1e-12]
    mesh.add_nodeset("edge", edge)
    problem = dualmesh.Problem(mesh, method=method)
    shell = problem.add_physics(
        "solid_mechanics",
        formulation="shell",
        youngs_modulus=2e11,
        poissons_ratio=0.3,
        thickness=thickness,
        body_force=list(load / thickness),
    )
    for name in shell_variables:
        shell.add_boundary_condition(f"{name} = 0", boundary="edge")
    return problem, flat


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("thickness", [0.01, 0.2])
def test_a_flat_shell_is_the_plate_and_stays_it_when_turned(method, thickness):
    """A clamped square under a uniform pressure: the shell on the flat mesh deflects as the
    Mindlin plate of the plate formulation, thin and thick, with no membrane displacement and
    no rotation about the normal; turned in space with its load, its displacements turn with
    it."""
    q = -1e3
    plane = dualmesh.generate_rectangle_mesh(0, 1, 0, 1, 6, 6)
    problem = dualmesh.Problem(plane, method=method)
    plate = problem.add_physics(
        "solid_mechanics",
        formulation="plate",
        youngs_modulus=2e11,
        poissons_ratio=0.3,
        thickness=thickness,
        transverse_load=q,
    )
    for name in ["disp_x", "disp_y", "disp_z", "phi_x", "phi_y"]:
        plate.add_boundary_condition(f"{name} = 0", boundary=["left", "right", "bottom", "top"])
    problem.solve(report="none")
    w = problem.values("disp_z")
    points = numpy.c_[numpy.asarray(plane.points())[:, :2], numpy.zeros(plane.num_nodes)]
    flat, _ = clamped_shell(points, method, numpy.array([0, 0, q]), thickness)
    flat.solve(report="none")
    assert flat.values("disp_z") == pytest.approx(w, rel=1e-9, abs=1e-12 * abs(w).max())
    for name in ["disp_x", "disp_y", "rotation_z"]:
        assert numpy.abs(flat.values(name)).max() < 1e-12 * abs(w).max()
    # The plate's phi = beta = (theta_y, -theta_x).
    assert flat.values("rotation_y") == pytest.approx(
        problem.values("phi_x"), abs=1e-9 * abs(problem.values("phi_x")).max()
    )
    turned, _ = clamped_shell(points @ rotation.T, method, rotation @ [0, 0, q], thickness)
    turned.solve(report="none")
    u = numpy.c_[[turned.values(n) for n in ("disp_x", "disp_y", "disp_z")]].T @ rotation
    theta = (
        numpy.c_[[turned.values(n) for n in ("rotation_x", "rotation_y", "rotation_z")]].T
        @ rotation
    )
    assert u[:, 2] == pytest.approx(w, rel=1e-8, abs=1e-10 * abs(w).max())
    assert numpy.abs(u[:, :2]).max() < 1e-10 * abs(w).max()
    assert theta[:, 1] == pytest.approx(flat.values("rotation_y"), abs=1e-8 * abs(theta).max())


# A section of alumina on aluminium, with Young's modulus E and the expansion coefficient alpha
# linear in s = 1/2 + z/H, z the height above the mid-surface and H the thickness.
bottom_modulus, top_modulus = 7.0e10, 3.8e11
bottom_alpha, top_alpha = 2.3e-5, 7.0e-6


def graded(bottom, top, thickness):
    """The expression of a quantity linear through the thickness, bottom at z = -H/2."""
    return f"{bottom!r} + {top - bottom!r}*(0.5 + z/{thickness!r})"


def flat_shell_mesh(plane):
    """The plane mesh `plane` as a surface mesh in three dimensions, in the plane z = 0 with the
    same nodes and elements, and the node sets left, right, bottom and top."""
    cells = numpy.array([plane.element_nodes(e) for e in range(plane.num_elements)])
    points = numpy.c_[numpy.asarray(plane.points())[:, :2], numpy.zeros(plane.num_nodes)]
    mesh = dualmesh.mesh_from_arrays(points, [("Quad4", cells)], dimension=3)
    x, y = points[:, 0], points[:, 1]
    sets = {
        "left": numpy.isclose(x, x.min()),
        "right": numpy.isclose(x, x.max()),
        "bottom": numpy.isclose(y, y.min()),
        "top": numpy.isclose(y, y.max()),
    }
    for name, nodes in sets.items():
        mesh.add_nodeset(name, [int(n) for n in numpy.flatnonzero(nodes)])
    return mesh


@pytest.mark.parametrize("method", methods)
def test_a_flat_graded_shell_is_the_graded_plate(method):
    """A cantilever strip of a graded section under a uniform pressure, in cylindrical bending
    (v = 0 and phi_y = 0 on its long edges): the shell on the flat mesh gives the displacements
    and rotations of the plate formulation, the membrane displacement that the coupling
    stiffness B makes included.  The field depends on x alone, so the drilling strain is zero and
    the two discretisations are the same."""
    H, q = 0.05, -1e4
    modulus = graded(bottom_modulus, top_modulus, H)
    plane = dualmesh.generate_rectangle_mesh(0, 2, 0, 0.5, 8, 2)
    problem = dualmesh.Problem(plane, method=method)
    plate = problem.add_physics(
        "solid_mechanics",
        formulation="plate",
        youngs_modulus=modulus,
        poissons_ratio=0.3,
        thickness=H,
        transverse_load=q,
    )
    for name in ["disp_x", "disp_y", "disp_z", "phi_x", "phi_y"]:
        plate.add_boundary_condition(f"{name} = 0", boundary="left")
    for name in ["disp_y", "phi_y"]:
        plate.add_boundary_condition(f"{name} = 0", boundary=["bottom", "top"])
    problem.solve(report="none")
    shell_problem = dualmesh.Problem(flat_shell_mesh(plane), method=method)
    shell = shell_problem.add_physics(
        "solid_mechanics",
        formulation="shell",
        youngs_modulus=modulus,
        poissons_ratio=0.3,
        thickness=H,
        body_force=[0.0, 0.0, q / H],
    )
    for name in shell_variables:
        shell.add_boundary_condition(f"{name} = 0", boundary="left")
    for name in ["disp_y", "rotation_x"]:
        shell.add_boundary_condition(f"{name} = 0", boundary=["bottom", "top"])
    shell_problem.solve(report="none")
    w, u = problem.values("disp_z"), problem.values("disp_x")
    # The coupling stiffness B moves the mid-surface in its plane.
    assert numpy.abs(u).max() > 1e-3 * numpy.abs(w).max()
    scale = numpy.abs(w).max()
    assert shell_problem.values("disp_z") == pytest.approx(w, rel=1e-9, abs=1e-12 * scale)
    assert shell_problem.values("disp_x") == pytest.approx(u, rel=1e-9, abs=1e-12 * scale)
    assert shell_problem.values("rotation_y") == pytest.approx(
        problem.values("phi_x"), rel=1e-9, abs=1e-12 * numpy.abs(problem.values("phi_x")).max()
    )


@pytest.mark.parametrize("method", methods)
def test_a_heated_graded_flat_shell_bends_as_the_plate_and_the_closed_form(method):
    r"""A plate strip of thickness $H$, with $E$ and $\alpha$ linear in $s = 1/2 + z/H$ (alumina
    on aluminium), heated uniformly by $\Delta T$ and free to expand and bend along $x$.  Its
    long edges $y = 0$ and $y = b$ hold $v = 0$ and $\phi_y = 0$, the cylindrical bending of a
    wide plate, and the edge $x = 0$ holds $u = 0$, $w = 0$ and $\phi_x = 0$, which the free
    strip satisfies there, so the supports carry no reaction.  The strip carries $N_{xx} = 0$
    and $M_{xx} = 0$ with $\varepsilon_{yy} = \chi_{yy} = 0$, so its membrane strain $e_0$ and
    its curvature $k$ are uniform and solve $[A, B; B, D] [e_0; k] = (1 + \nu) \Delta T [F; G]$,
    with Poisson's ratio $\nu$ and the integrals per unit width

    $$ A = H \left( E_b + \frac{\Delta E}{2} \right), \quad
       B = \frac{H^2 \Delta E}{12}, \quad
       D = H^3 \left( \frac{E_b}{12} + \frac{\Delta E}{24} \right), $$

    $$ F = H \left( E_b \alpha_b + \frac{E_b \Delta\alpha + \Delta E \alpha_b}{2} +
       \frac{\Delta E \Delta\alpha}{3} \right), \quad
       G = \frac{H^2}{12} (E_b \Delta\alpha + \Delta E \alpha_b + \Delta E \Delta\alpha), $$

    where $E_b$, $\alpha_b$ are the values at the bottom and $\Delta E$, $\Delta\alpha$ the
    differences between top and bottom.  The factor $1 + \nu$ is that of the restrained
    direction $y$.  Then $u = e_0 x$, $v = 0$, $w = -k x^2/2$ and $\phi_x = k x$, which the plate
    and the shell on the flat mesh reproduce at the nodes.  The deflection is held along a whole
    edge: the transverse shear integrated at the element centre leaves the checkerboard mode of
    $w$ without stiffness, and a deflection held at one node only would leave it free."""
    H, nu, t0, t1 = 0.02, 0.3, 293.0, 393.0
    modulus = graded(bottom_modulus, top_modulus, H)
    alpha = graded(bottom_alpha, top_alpha, H)
    plane = dualmesh.generate_rectangle_mesh(0, 1, 0, 0.5, 4, 2)
    points = numpy.asarray(plane.points())
    walls = ["left", "right", "bottom", "top"]

    def heated(mesh, formulation, conditions):
        problem = dualmesh.Problem(mesh, method=method)
        heat = problem.add_physics("heat_conduction", thermal_conductivity=50.0)
        heat.add_boundary_condition("temperature = T1", boundary=walls, T1=t1)
        solid = problem.add_physics(
            "solid_mechanics",
            formulation=formulation,
            youngs_modulus=modulus,
            poissons_ratio=nu,
            thickness=H,
        )
        for condition, boundary in conditions:
            solid.add_boundary_condition(condition, boundary=boundary)
        problem.add_coupling(
            "thermal_expansion", linear_expansion_coefficient=alpha, stress_free_temperature=t0
        )
        problem.solve(report="none")
        return problem

    long_edges = ["bottom", "top"]
    plate = heated(
        plane,
        "plate",
        [
            ("disp_x = 0", "left"),
            ("disp_z = 0", "left"),
            ("phi_x = 0", "left"),
            ("disp_y = 0", long_edges),
            ("phi_y = 0", long_edges),
        ],
    )
    shell = heated(
        flat_shell_mesh(plane),
        "shell",
        [
            ("disp_x = 0", "left"),
            ("disp_z = 0", "left"),
            ("rotation_y = 0", "left"),
            ("disp_y = 0", long_edges),
            ("rotation_x = 0", long_edges),
        ],
    )
    E_b, dE = bottom_modulus, top_modulus - bottom_modulus
    a_b, da = bottom_alpha, top_alpha - bottom_alpha
    A = H * (E_b + dE / 2)
    B = H**2 * dE / 12
    D = H**3 * (E_b / 12 + dE / 24)
    F = H * (E_b * a_b + (E_b * da + dE * a_b) / 2 + dE * da / 3)
    G = H**2 / 12 * (E_b * da + dE * a_b + dE * da)
    heating = (1 + nu) * (t1 - t0)
    e0 = heating * (D * F - B * G) / (A * D - B**2)
    k = heating * (A * G - B * F) / (A * D - B**2)
    x = points[:, 0]
    w = -k * x**2 / 2
    # The bending is of the order of the stretching, so the check of w is not one of zero.
    assert abs(k) * H > 0.1 * abs(e0)
    for problem in (plate, shell):
        assert problem.values("disp_x") == pytest.approx(e0 * x, rel=1e-8, abs=1e-12 * abs(e0))
        assert numpy.abs(problem.values("disp_y")).max() < 1e-9 * abs(e0)
        assert problem.values("disp_z") == pytest.approx(w, rel=1e-8, abs=1e-12 * abs(k))
    assert plate.values("phi_x") == pytest.approx(k * x, rel=1e-8, abs=1e-12 * abs(k))
    assert shell.values("rotation_y") == pytest.approx(k * x, rel=1e-8, abs=1e-12 * abs(k))
    for name in ["rotation_x", "rotation_z"]:
        assert numpy.abs(shell.values(name)).max() < 1e-9 * abs(k)


def scordelis_lo(n, method):
    """The Scordelis-Lo roof: a cylindrical panel of radius 25 and length 50 over 80 degrees,
    on rigid diaphragms at its curved ends and free along its straight edges, under its weight
    of 90 per area; half of it, with a plane of symmetry at mid-length.  The vertical
    displacement at the middle of a free edge is 0.3024 (MacNeal and Harder)."""
    R, L, h = 25.0, 50.0, 0.25
    mesh = dualmesh.generate_cylinder_surface_mesh(
        radius=R, length=L / 2, arc_angle=80.0, num_elements=[2 * n, n]
    )
    problem = dualmesh.Problem(mesh, method=method)
    shell = problem.add_physics(
        "solid_mechanics",
        formulation="shell",
        youngs_modulus=4.32e8,
        poissons_ratio=0.0,
        thickness=h,
        body_force=[0, 0, -90 / h],
    )
    shell.add_boundary_condition("disp_x = 0", boundary="near_curved_edge")
    shell.add_boundary_condition("disp_z = 0", boundary="near_curved_edge")
    for condition in ("disp_y = 0", "rotation_x = 0", "rotation_z = 0"):
        shell.add_boundary_condition(condition, boundary="far_curved_edge")
    problem.solve(report="none")
    angle = math.radians(40)
    node = problem.node_at([R * math.sin(angle), L / 2, R * math.cos(angle)])
    return -problem.values("disp_z")[node] / 0.3024


@pytest.mark.parametrize("method, at_16", [("FEM", 0.99), ("DMCDM", 0.98)])
def test_the_scordelis_lo_roof_converges_to_its_reference(method, at_16):
    """From below, the error falling by about half or more with each refinement: 0.927, 0.969,
    0.990 of the reference on 4, 8 and 16 elements a side (FEM), 0.838, 0.935, 0.980
    (DMCDM)."""
    ratios = [scordelis_lo(n, method) for n in (4, 8, 16)]
    assert ratios[0] < ratios[1] < ratios[2] < 1.0
    assert 1 - ratios[2] < 0.6 * (1 - ratios[1]) < 0.6 * 0.6 * (1 - ratios[0])
    assert ratios[2] > at_16


def pinched_cylinder(n, method):
    """The pinched cylinder: radius 300, length 600, thickness 3, on rigid diaphragms at its
    ends, pinched by two opposite unit forces at mid-length; an eighth of it with three planes
    of symmetry, so a quarter of a force at the top.  The displacement under the force is
    1.8248e-5 (MacNeal and Harder)."""
    R, L = 300.0, 600.0
    t, y = numpy.linspace(0, math.pi / 2, n + 1), numpy.linspace(0, L / 2, n + 1)
    T, Y = numpy.meshgrid(t, y, indexing="ij")
    points = numpy.c_[R * numpy.sin(T).ravel(), Y.ravel(), R * numpy.cos(T).ravel()]
    index = lambda i, j: i * (n + 1) + j  # noqa: E731
    quads = [
        [index(i, j), index(i + 1, j), index(i + 1, j + 1), index(i, j + 1)]
        for i in range(n)
        for j in range(n)
    ]
    mesh = dualmesh.mesh_from_arrays(points, [("Quad4", numpy.array(quads))], dimension=3)
    for name, test in (
        ("end", lambda x, y, z: abs(y) < 1e-6),
        ("middle", lambda x, y, z: abs(y - L / 2) < 1e-6),
        ("x_plane", lambda x, y, z: abs(x) < 1e-6),
        ("z_plane", lambda x, y, z: abs(z) < 1e-6),
    ):
        mesh.add_sideset_by_predicate(name, test)
    problem = dualmesh.Problem(mesh, method=method)
    shell = problem.add_physics(
        "solid_mechanics",
        formulation="shell",
        youngs_modulus=3e6,
        poissons_ratio=0.3,
        thickness=3.0,
    )
    for boundary, conditions in (
        ("end", ("disp_x", "disp_z", "rotation_y")),
        ("middle", ("disp_y", "rotation_x", "rotation_z")),
        ("x_plane", ("disp_x", "rotation_y", "rotation_z")),
        ("z_plane", ("disp_z", "rotation_x", "rotation_y")),
    ):
        for name in conditions:
            shell.add_boundary_condition(f"{name} = 0", boundary=boundary)
    problem.add_point_source(variable="disp_z", value=-0.25, points=[[0.0, L / 2, R]])
    problem.solve(report="none")
    return -problem.values("disp_z")[problem.node_at([0.0, L / 2, R])] / 1.8248e-5


@pytest.mark.parametrize("method, at_32", [("FEM", 0.995), ("DMCDM", 0.99)])
def test_the_pinched_cylinder_converges_to_its_reference(method, at_32):
    """From below, as a bending-dominated shell does on flat facets: 0.35, 0.73, 0.93 and 0.997
    of the reference on 4, 8, 16 and 32 elements a side (FEM), 0.32, 0.71, 0.92 and 0.993
    (DMCDM)."""
    ratios = [pinched_cylinder(n, method) for n in (8, 16, 32)]
    assert ratios[0] < ratios[1] < ratios[2] < 1.0
    assert ratios[2] > at_32


@pytest.mark.parametrize("method", methods)
def test_a_tripod_carries_its_load_by_axial_forces(method):
    """Three bars from the apex (0, 0, H) to feet unequally spaced on a circle of radius a, a
    load P down at the apex, the whole turned in space: the bar forces are those of the statics
    of the apex, every bar is compressed, each shortens by N L / EA, and each support pushes on
    the body along its bar with its bar's force."""
    H, a, P, ea = 1.0, 0.75, 1e4, 2e11 * 1e-4
    feet = [[a * math.cos(phi), a * math.sin(phi), 0.0] for phi in (0.3, 0.3 + 2.1, 0.3 + 4.0)]
    points = numpy.array([[0.0, 0.0, H]] + feet) @ rotation.T
    mesh = dualmesh.mesh_from_arrays(points, [("Edge2", numpy.array([[0, 1], [0, 2], [0, 3]]))])
    for k in range(3):
        mesh.add_nodeset(f"foot_{k}", [k + 1])
    problem = dualmesh.Problem(mesh, method=method)
    truss = problem.add_physics(
        "solid_mechanics", formulation="truss", youngs_modulus=2e11, cross_section={"area": 1e-4}
    )
    for k in range(3):
        truss.add_boundary_condition("fixed", boundary=f"foot_{k}")
    load = rotation @ [0.0, 0.0, -P]
    for d, name in enumerate(("disp_x", "disp_y", "disp_z")):
        problem.add_point_source(variable=name, value=float(load[d]), points=[list(points[0])])
    problem.solve(report="none")
    # The statics of the apex: sum of N_k d_k = -load, with d_k from the apex to foot k.
    directions = numpy.array(
        [
            (points[k + 1] - points[0]) / numpy.linalg.norm(points[k + 1] - points[0])
            for k in range(3)
        ]
    )
    tension = numpy.linalg.solve(directions.T, -load)
    lengths = numpy.array([numpy.linalg.norm(points[k + 1] - points[0]) for k in range(3)])
    stretch = tension * lengths / ea
    apex = numpy.array([problem.values(n)[0] for n in ("disp_x", "disp_y", "disp_z")])
    # The apex moves so that each bar stretches by its N L / EA: -directions . apex = stretch.
    assert -directions @ apex == pytest.approx(stretch, rel=1e-10)
    # The support pushes on the body along the bar, and the three hold the load.
    for k in range(3):
        force = truss.reaction_force(f"foot_{k}")
        assert [force[c] for c in "xyz"] == pytest.approx(
            list(tension[k] * directions[k]), rel=1e-9, abs=1e-9 * P
        )
    assert all(t < 0.0 for t in tension)


@pytest.mark.parametrize("method", methods)
def test_a_hanging_bar_stretches_under_its_weight(method):
    """A bar of length L hanging from x = 0 along a line mesh, its weight rho g per volume along
    +x: the displacement is rho g (2 L x - x^2) / (2 E), exact at the nodes, and the support
    carries rho g A L."""
    L, E, A, w = 2.0, 2e11, 3e-4, 7.8e4
    mesh = dualmesh.generate_line_mesh(start=0.0, end=L, num_elements=8)
    problem = dualmesh.Problem(mesh, method=method)
    truss = problem.add_physics(
        "solid_mechanics",
        formulation="truss",
        youngs_modulus=E,
        cross_section={"area": A},
        body_force=[w],
    )
    truss.add_boundary_condition("disp_x = 0", boundary="left")
    problem.solve(report="none")
    x = numpy.asarray(mesh.points())[:, 0]
    assert problem.values("disp_x") == pytest.approx(w * (2 * L * x - x**2) / (2 * E), rel=1e-9)
    assert truss.reaction_force("left")["x"] == pytest.approx(-w * A * L, rel=1e-10)


def test_the_input_of_the_members_is_checked():
    line = dualmesh.mesh_from_arrays(
        numpy.array([[0.0, 0.0], [1.0, 1.0]]), [("Edge2", numpy.array([[0, 1]]))]
    )
    problem = dualmesh.Problem(line)
    with pytest.raises(InputError, match="give formulation='truss'"):
        problem.add_physics("solid_mechanics", youngs_modulus=1.0)
    with pytest.raises(InputError, match=r"cross_section=\{'area': A\}"):
        dualmesh.Problem(line).add_physics(
            "solid_mechanics", formulation="truss", youngs_modulus=1.0
        )
    with pytest.raises(InputError, match="static, and takes no .*plasticity"):
        dualmesh.Problem(line).add_physics(
            "solid_mechanics",
            formulation="truss",
            youngs_modulus=1.0,
            cross_section={"area": 1.0},
            plasticity={"yield_stress": 1.0},
        )
    surface = dualmesh.generate_cylinder_surface_mesh(
        radius=1.0, length=1.0, arc_angle=90.0, num_elements=[2, 2]
    )
    with pytest.raises(InputError, match="give formulation='shell'"):
        dualmesh.Problem(surface).add_physics("solid_mechanics", youngs_modulus=1.0)
    with pytest.raises(InputError, match="give the shell's thickness"):
        dualmesh.Problem(surface).add_physics(
            "solid_mechanics", formulation="shell", youngs_modulus=1.0, poissons_ratio=0.3
        )
