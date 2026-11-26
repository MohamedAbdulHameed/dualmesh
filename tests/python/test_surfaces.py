# SPDX-License-Identifier: LGPL-2.1-or-later
"""Surface meshes: 2-D elements in 3-D space (the meshes of shells).

The core maps a surface element with its 3 x 2 Jacobian, measures it with
sqrt(det(J^T J)) and takes tangential gradients; the generators build the
cylinder, the sphere, height and parametric surfaces and f(x, y, z) = c.
Every check has an exact answer: a tilted plane gives the flat solution, a
spherical harmonic solves the Laplace-Beltrami problem, and the generated
surfaces have known areas, topology and normals."""

import math

import dualmesh
import numpy
import pytest
from dualmesh._core import InputError
from dualmesh.surfaces import _corner_angles

rotation = numpy.array(
    [[math.cos(0.4), 0, math.sin(0.4)], [0, 1, 0], [-math.sin(0.4), 0, math.cos(0.4)]]
) @ numpy.array([[1, 0, 0], [0, math.cos(1.1), -math.sin(1.1)], [0, math.sin(1.1), math.cos(1.1)]])


def heat_on_rectangle(tilted, method, element_type):
    mesh = dualmesh.generate_rectangle_mesh(
        x_min=0,
        x_max=2,
        y_min=0,
        y_max=1,
        num_x_elements=8,
        num_y_elements=5,
        element_type=element_type,
    )
    if tilted:
        mesh.transform_nodes(
            lambda x, y, z: list(rotation @ numpy.array([x, y, z]) + [0.3, -0.2, 0.5])
        )
        mesh.space_dimension = 3
        mesh.fix_orientation()
    problem = dualmesh.Problem(mesh, method=method)
    heat = problem.add_physics(
        "heat_conduction", thermal_conductivity=3.0, heat_source=10.0, initial_condition=0.0
    )
    heat.add_boundary_condition("temperature = 1", boundary="left")
    heat.add_boundary_condition("heat_flux = 5*(temperature - 0.5)", boundary="top")
    heat.add_boundary_condition("heat_flux = -2", boundary="right")
    problem.solve()
    return problem.values("temperature")


@pytest.mark.parametrize(
    "method,element_type",
    [("fem", "Quad4"), ("fem", "Tri3"), ("fem", "Quad9"), ("dmcdm", "Quad4"), ("dmcdm", "Tri3")],
)
def test_a_tilted_plane_in_space_gives_the_flat_solution(method, element_type):
    # Conduction with a source, a fixed edge, a convective edge and a flux
    # edge: the mapping, the area element and the edge conormal of a surface
    # in space must reproduce the flat problem to round-off.
    flat = heat_on_rectangle(False, method, element_type)
    tilted = heat_on_rectangle(True, method, element_type)
    assert numpy.abs(tilted - flat).max() < 1e-12 * numpy.abs(flat).max()


def cubed_sphere(N):
    t = numpy.tan(numpy.linspace(-numpy.pi / 4, numpy.pi / 4, N + 1))
    sphere_points, quads = [], []
    for axis in range(3):
        for sign in (-1, 1):
            u, v = numpy.meshgrid(t, t, indexing="ij")
            p = numpy.empty(u.shape + (3,))
            p[..., axis] = sign
            o = [i for i in range(3) if i != axis]
            p[..., o[0]], p[..., o[1]] = u, v
            sphere_points.append(p.reshape(-1, 3))
    sphere_points = numpy.concatenate(sphere_points)
    key, inv = numpy.unique(
        numpy.round(sphere_points / numpy.linalg.norm(sphere_points, axis=1)[:, None], 12),
        axis=0,
        return_inverse=True,
    )
    inv = inv.ravel()
    for k in range(6):
        index = inv[k * (N + 1) ** 2 : (k + 1) * (N + 1) ** 2].reshape(N + 1, N + 1)
        quads.append(
            numpy.stack(
                [index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]], -1
            ).reshape(-1, 4)
        )
    return key, numpy.concatenate(quads)


@pytest.mark.parametrize("method", ["fem", "dmcdm"])
def test_a_spherical_harmonic_solves_Laplace_beltrami_at_second_order(method):
    # `z` is a spherical harmonic of degree 1 on the unit sphere:
    # laplacian_S z = -2 z, so -laplacian_S u + u = 3 z has u = z exactly.
    errors = []
    for N in (4, 8, 16):
        X, Q = cubed_sphere(N)
        mesh = dualmesh.mesh_from_arrays(X, Q, element_type="Quad4")
        assert mesh.space_dimension == 3
        problem = dualmesh.Problem(mesh, method=method)
        problem.add_physics("general_form_PDE", variables="u", equations="-laplacian(u) + u = 3*z")
        problem.solve()
        errors.append(numpy.abs(problem.values("u") - X[:, 2]).max())
    rates = numpy.log2(numpy.array(errors[:-1]) / numpy.array(errors[1:]))
    assert errors[-1] < 2.5e-3
    assert numpy.all(rates > 1.7)


def facets(mesh):
    P = numpy.asarray(mesh.points())
    area, normals, centres = 0.0, [], []
    for element_type, nodes, _block in mesh.cells():
        c = P[nodes[: 3 if element_type.startswith("Tri") else 4]]
        if len(c) == 3:
            a = numpy.cross(c[1] - c[0], c[2] - c[0]) / 2
        else:
            a = (numpy.cross(c[1] - c[0], c[2] - c[0]) + numpy.cross(c[2] - c[0], c[3] - c[0])) / 2
        area += numpy.linalg.norm(a)
        normals.append(a)
        centres.append(c.mean(axis=0))
    return area, numpy.array(normals), numpy.array(centres)


def Euler(mesh):
    cells = [n[:3] if t.startswith("Tri") else n[:4] for t, n, _ in mesh.cells()]
    edges = {tuple(sorted((c[i], c[(i + 1) % len(c)]))) for c in cells for i in range(len(c))}
    return len({n for c in cells for n in c}) - len(edges) + len(cells)


def check(mesh, area, Euler_characteristic, on_surface, outward=None, edges=None, rtol=1e-2):
    assert mesh.dimension == 2 and mesh.space_dimension == 3
    measured, normals, centres = facets(mesh)
    assert measured == pytest.approx(area, rel=rtol)
    assert Euler(mesh) == Euler_characteristic
    assert numpy.abs(on_surface(numpy.asarray(mesh.points()))).max() < 1e-12
    if outward is not None:
        assert numpy.all(numpy.einsum("ij,ij->i", normals, outward(centres)) > 0)
    if edges is not None:
        assert {name: len(mesh.sideset(name)) for name in mesh.sideset_names()} == edges


def test_the_scordelis_lo_roof_panel():
    mesh = dualmesh.generate_cylinder_surface_mesh(
        radius=25.0, length=50.0, arc_angle=80.0, num_elements=[16, 16]
    )
    check(
        mesh,
        25.0 * math.radians(80) * 50.0,
        1,
        lambda p: numpy.hypot(p[:, 0], p[:, 2]) - 25.0,
        outward=lambda c: c * [1, 0, 1],
        edges={
            "left_straight_edge": 16,
            "right_straight_edge": 16,
            "near_curved_edge": 16,
            "far_curved_edge": 16,
        },
        rtol=1e-3,
    )
    # The straight edges are at 40 degrees each side of the crown (+z).
    P = numpy.asarray(mesh.points())
    left = P[mesh.boundary_nodes("left_straight_edge")]
    assert numpy.allclose(numpy.degrees(numpy.arctan2(left[:, 0], left[:, 2])), -40.0)


def test_a_full_cylinder_closes_its_seam():
    mesh = dualmesh.generate_cylinder_surface_mesh(
        radius=2.0, length=3.0, num_elements=[24, 4], element_type="Quad9"
    )
    check(
        mesh,
        2 * math.pi * 2.0 * 3.0,
        0,
        lambda p: numpy.hypot(p[:, 0], p[:, 2]) - 2.0,
        edges={"near_curved_edge": 24, "far_curved_edge": 24},
    )


@pytest.mark.parametrize(
    "polar,azimuth,num,Euler_characteristic,edges",
    [
        ((0, 180), (0, 360), (16, 32), 2, {}),
        (
            (18, 90),
            (0, 90),
            (8, 8),
            1,
            {
                "upper_circular_edge": 8,
                "lower_circular_edge": 8,
                "start_meridian_edge": 8,
                "end_meridian_edge": 8,
            },
        ),
        (
            (0, 90),
            (0, 90),
            (8, 8),
            1,
            {"lower_circular_edge": 8, "start_meridian_edge": 8, "end_meridian_edge": 8},
        ),
        ((0, 180), (0, 90), (16, 8), 1, {"start_meridian_edge": 16, "end_meridian_edge": 16}),
        ((100, 180), (0, 360), (8, 16), 1, {"upper_circular_edge": 16}),
        ((30, 150), (0, 360), (8, 32), 0, {"upper_circular_edge": 32, "lower_circular_edge": 32}),
    ],
)
def test_zones_of_a_sphere(polar, azimuth, num, Euler_characteristic, edges):
    mesh = dualmesh.generate_spherical_shell_mesh(
        radius=10.0, polar_angle=polar, azimuth=azimuth, num_elements=num
    )
    area = (
        100
        * math.radians(azimuth[1] - azimuth[0])
        * (math.cos(math.radians(polar[0])) - math.cos(math.radians(polar[1])))
    )
    check(
        mesh,
        area,
        Euler_characteristic,
        lambda p: numpy.linalg.norm(p, axis=1) - 10.0,
        lambda c: c,
        edges,
        rtol=2e-2,
    )
    assert (
        _corner_angles(
            numpy.asarray(mesh.points()), numpy.array([n for _, n, _ in mesh.cells()])
        ).min()
        > 40
    )


@pytest.mark.parametrize("element_type", ["Quad9", "Tri6"])
def test_a_quadratic_sphere_has_its_nodes_on_the_sphere(element_type):
    mesh = dualmesh.generate_spherical_shell_mesh(
        radius=1.0, num_elements=[16, 32], element_type=element_type, center=[1, 2, 3]
    )
    check(
        mesh,
        4 * math.pi,
        2,
        lambda p: numpy.linalg.norm(p - [1, 2, 3], axis=1) - 1,
        lambda c: c - [1, 2, 3],
    )


def test_height_and_parametric_surfaces():
    saddle = dualmesh.generate_surface_mesh(
        surface="z = c*(x^2 - y^2)",
        c=0.1,
        x_min=-5,
        x_max=5,
        y_min=-5,
        y_max=5,
        num_elements=[16, 16],
    )
    P = numpy.asarray(saddle.points())
    assert numpy.abs(P[:, 2] - 0.1 * (P[:, 0] ** 2 - P[:, 1] ** 2)).max() < 1e-12
    _area, normals, _c = facets(saddle)
    assert numpy.all(normals[:, 2] > 0)
    assert sorted(saddle.sideset_names()) == [
        "x_max_edge",
        "x_min_edge",
        "y_max_edge",
        "y_min_edge",
    ]
    torus = dualmesh.generate_surface_mesh(
        surface=["x = (R + a*cos(v))*cos(u)", "y = (R + a*cos(v))*sin(u)", "z = a*sin(v)"],
        R=2.0,
        a=0.5,
        u_min=0,
        u_max="2*pi",
        v_min=0,
        v_max="2*pi",
        num_elements=[48, 16],
    )
    on = lambda p: (numpy.hypot(p[:, 0], p[:, 1]) - 2) ** 2 + p[:, 2] ** 2 - 0.25  # noqa: E731
    check(torus, 4 * math.pi**2 * 2 * 0.5, 0, on, edges={})


@pytest.mark.parametrize(
    "surface,box,area,Euler_characteristic,on",
    [
        (
            "x^2 + y^2 + z^2 = 1",
            [[-1.5] * 3, [1.5] * 3],
            4 * math.pi,
            2,
            lambda p: numpy.linalg.norm(p, axis=1) - 1,
        ),
        (
            "x^2 + y^2 + z^2 = 1",
            [[-1.5, -1.5, 0.0], [1.5, 1.5, 1.5]],
            2 * math.pi,
            1,
            lambda p: numpy.linalg.norm(p, axis=1) - 1,
        ),
        (
            "x^2 + y^2 + z^2 = 1",
            [[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]],
            math.pi / 2,
            1,
            lambda p: numpy.linalg.norm(p, axis=1) - 1,
        ),
        (
            "(sqrt(x^2 + y^2) - 2)^2 + z^2 = 0.25",
            [[-3, -3, -1], [3, 3, 1]],
            4 * math.pi**2,
            0,
            lambda p: (numpy.hypot(p[:, 0], p[:, 1]) - 2) ** 2 + p[:, 2] ** 2 - 0.25,
        ),
    ],
)
def test_any_surface_cut_out_of_a_box(surface, box, area, Euler_characteristic, on):
    mesh = dualmesh.generate_surface_mesh(surface=surface, box=box, element_size=0.1)
    check(mesh, area, Euler_characteristic, on, rtol=1e-2)
    P = numpy.asarray(mesh.points())
    assert _corner_angles(P, numpy.array([n for _, n, _ in mesh.cells()])).min() > 45
    # The normals point towards f > c, here outward.
    if Euler_characteristic != 0:
        _a, normals, centres = facets(mesh)
        assert numpy.all(numpy.einsum("ij,ij->i", normals, centres) > 0)
    # The cut lies on the faces of the box.
    if "box_edge" in mesh.sideset_names():
        lo, hi = numpy.array(box[0], float), numpy.array(box[1], float)
        edge = P[mesh.boundary_nodes("box_edge")]
        assert numpy.all(
            ((numpy.abs(edge - lo) < 1e-12) | (numpy.abs(edge - hi) < 1e-12)).any(axis=1)
        )


def test_surfaces_are_refused_where_they_do_not_make_sense():
    with pytest.raises(InputError, match="more than once"):
        dualmesh.generate_surface_mesh(
            surface="(sqrt(x^2 + y^2) - 2)^2 + z^2 = 0.25", center=[2, 0, 0], num_elements=[8, 16]
        )
    with pytest.raises(InputError, match="either center"):
        dualmesh.generate_surface_mesh(surface="x^2 + y^2 + z^2 = 1", num_elements=[8, 8])
    with pytest.raises(InputError, match="not used"):
        dualmesh.generate_surface_mesh(
            surface="x^2 + y^2 + z^2 = 1", k=2.0, center=[0, 0, 0], num_elements=[8, 16]
        )
    with pytest.raises(InputError, match="multiple of 8"):
        dualmesh.generate_spherical_shell_mesh(radius=1.0, num_elements=[8, 12])
    mesh = dualmesh.generate_spherical_shell_mesh(radius=1.0, num_elements=[8, 16])
    with pytest.raises(InputError, match="give formulation='shell'"):
        dualmesh.Problem(mesh).add_physics(
            "solid_mechanics", youngs_modulus=1.0, poissons_ratio=0.3
        )


def test_a_distributed_surface_problem_gives_the_serial_answer():
    # The distributed mesh carries the space dimension to every process (run
    # under mpirun, the same check covers several processes).
    mesh = dualmesh.generate_spherical_shell_mesh(radius=1.0, num_elements=[8, 16])

    def solve(distributed):
        problem = dualmesh.Problem(mesh, method="fem", distributed=distributed)
        problem.add_physics("general_form_PDE", variables="u", equations="-laplacian(u) + u = 3*z")
        problem.solve()
        return problem

    serial = solve(False).values("u")
    distributed = numpy.asarray(solve(True).gathered_values("u"))
    assert numpy.abs(distributed - serial).max() < 1e-10


def test_refinement_keeps_the_normals_a_generator_gave():
    # A height surface x = 3 y: its normals point towards increasing x, which
    # is not the generic rule's choice (largest summed component positive).
    mesh = dualmesh.generate_surface_mesh(
        surface="x = k*y", k=3.0, y_min=0, y_max=1, z_min=0, z_max=1, num_elements=[4, 4]
    )
    for m in (mesh, mesh.refined()):
        _area, normals, _c = facets(m)
        assert numpy.all(normals[:, 0] > 0)


def test_preferred_normals_choose_the_side_of_each_part():
    # Two separate plates in the xy plane: the default rule points both to +z,
    # and preferred normals turn the second one to -z on its own.
    points = [[x + shift, y, 0.0] for shift in (0.0, 2.0) for y in (0.0, 1.0) for x in (0.0, 1.0)]
    plates = dualmesh.mesh_from_arrays(
        points, [[0, 1, 3, 2], [4, 5, 7, 6]], element_type="Quad4", dimension=3
    )
    _area, normals, _c = facets(plates)
    assert numpy.allclose(normals[:, 2], [1.0, 1.0])
    plates.fix_orientation(preferred=numpy.array([[0.0, 0.0, 1.0], [0.1, 0.0, -1.0]]))
    _area, normals, _c = facets(plates)
    assert numpy.allclose(normals[:, 2], [1.0, -1.0])
    with pytest.raises(InputError, match="shape"):
        plates.fix_orientation(preferred=numpy.zeros((2, 2)))
    with pytest.raises(InputError, match="one preferred normal per element"):
        plates.fix_orientation(preferred=numpy.zeros((3, 3)))


def test_a_one_sided_surface_is_refused():
    with pytest.raises(InputError, match="cannot be oriented"):
        dualmesh.generate_surface_mesh(
            surface=[
                "x = (1 + v*cos(u/2))*cos(u)",
                "y = (1 + v*cos(u/2))*sin(u)",
                "z = v*sin(u/2)",
            ],
            u_min=0,
            u_max="2*pi",
            v_min=-0.3,
            v_max=0.3,
            num_elements=[32, 4],
        )


@pytest.mark.parametrize("radius", [5e-7, 1.0, 3e6])
def test_a_surface_seen_from_a_centre_at_any_scale(radius):
    mesh = dualmesh.generate_surface_mesh(
        surface="x^2 + y^2 + z^2 = R^2", R=radius, center=[0, 0, 0], num_elements=[8, 16]
    )
    P = numpy.asarray(mesh.points())
    assert numpy.abs(numpy.linalg.norm(P, axis=1) / radius - 1).max() < 1e-13
