# SPDX-License-Identifier: LGPL-2.1-or-later
"""Orthotropic and anisotropic elasticity, material axes, and Hill creep and plasticity."""

from __future__ import annotations

import dualmesh
import numpy
import pytest
from dualmesh._core import InputError

size = 1e-3
E = [140e9, 10e9, 12e9]
Poisson_ratios = [0.3, 0.25, 0.45]
G = [5e9, 6e9, 3.5e9]


def cube(**solid):
    mesh = dualmesh.generate_box_mesh(0, size, 0, size, 0, size, 1, 1, 1)
    problem = dualmesh.Problem(mesh)
    body = problem.add_physics("solid_mechanics", **solid)
    return problem, body


def uniaxial(problem, body, axis, stress):
    """Rollers on the three faces at 0 and a traction along `axis` on the opposite face."""
    for name, side in (("disp_x", "left"), ("disp_y", "bottom"), ("disp_z", "back")):
        body.add_boundary_condition(f"{name} = 0", boundary=side)
    face = {"x": "right", "y": "top", "z": "front"}[axis]
    traction = {"x": [stress, 0, 0], "y": [0, stress, 0], "z": [0, 0, stress]}[axis]
    body.add_boundary_condition(f"traction = {traction}", boundary=face)


def strains(problem):
    far = size * (1 - 1e-9)
    return (
        numpy.array(
            [
                problem.sample("disp_x", [[far, size / 2, size / 2]])[0],
                problem.sample("disp_y", [[size / 2, far, size / 2]])[0],
                problem.sample("disp_z", [[size / 2, size / 2, far]])[0],
            ]
        )
        / size
    )


@pytest.mark.parametrize("axis, column", [("x", 0), ("y", 1), ("z", 2)])
def test_an_orthotropic_cube_has_the_compliance_of_its_axes(axis, column):
    """Uniaxial stress along each material axis: the strains are the column of the compliance,
    eps_i = S_ij sigma, with S_12 = -nu12/E1, S_13 = -nu13/E1, S_23 = -nu23/E2."""
    S = numpy.array(
        [
            [1 / E[0], -Poisson_ratios[0] / E[0], -Poisson_ratios[1] / E[0]],
            [-Poisson_ratios[0] / E[0], 1 / E[1], -Poisson_ratios[2] / E[1]],
            [-Poisson_ratios[1] / E[0], -Poisson_ratios[2] / E[1], 1 / E[2]],
        ]
    )
    problem, body = cube(youngs_modulus=E, poissons_ratio=Poisson_ratios, shear_modulus=G)
    uniaxial(problem, body, axis, 50e6)
    problem.solve(report="none")
    # The strains are sampled 1e-9 inside the far faces.
    assert strains(problem) == pytest.approx(50e6 * S[:, column], rel=1e-8)


def rotated_tensor_stiffness(C, R):
    """C_ijkl = R_ip R_jq R_kr R_ls C_pqrs, from the 6 x 6 matrix, as 4th-order tensors."""
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    index = numpy.zeros((3, 3), dtype=int)
    for position, (i, j) in enumerate(pairs):
        index[i, j] = index[j, i] = position
    tensor = numpy.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for m in range(3):
                    tensor[i, j, k, m] = C[index[i, j], index[k, m]]
    return numpy.einsum("ip,jq,kr,ls,pqrs->ijkl", R, R, R, R, tensor)


def test_orientation_rotate_the_stiffness_as_a_fourth_order_tensor():
    """A homogeneous strain imposed on every face: the stress is C_ijkl eps_kl with the tensor
    of the material axes rotated to the global ones, computed here independently."""
    a1, a2 = numpy.array([1.0, 2.0, 0.5]), numpy.array([-2.0, 1.0, 0.3])
    problem, body = cube(
        youngs_modulus=E, poissons_ratio=Poisson_ratios, shear_modulus=G, orientation=[a1, a2]
    )
    strain = numpy.array([[1e-4, 3e-5, -2e-5], [3e-5, -5e-5, 4e-5], [-2e-5, 4e-5, 2e-4]])
    sides = list(problem.mesh.sideset_names())
    for i, name in enumerate(("disp_x", "disp_y", "disp_z")):
        expression = " + ".join(f"{float(strain[i, j])!r}*{c}" for j, c in enumerate("xyz"))
        body.add_boundary_condition(f"{name} = {expression}", boundary=sides)
    problem.solve(report="none")
    # The local stiffness, rotated as a tensor.
    S = numpy.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1 / E[0], 1 / E[1], 1 / E[2]
    S[0, 1] = S[1, 0] = -Poisson_ratios[0] / E[0]
    S[0, 2] = S[2, 0] = -Poisson_ratios[1] / E[0]
    S[1, 2] = S[2, 1] = -Poisson_ratios[2] / E[1]
    S[3, 3], S[4, 4], S[5, 5] = 1 / G[2], 1 / G[1], 1 / G[0]
    e1 = a1 / numpy.linalg.norm(a1)
    e2 = a2 - (a2 @ e1) * e1
    e2 /= numpy.linalg.norm(e2)
    R = numpy.column_stack([e1, e2, numpy.cross(e1, e2)])
    C = rotated_tensor_stiffness(numpy.linalg.inv(S), R)
    expected = numpy.einsum("ijkl,kl->ij", C, strain)
    stress = problem.property_at_centroids("stress")[0]
    Voigt_pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    assert stress == pytest.approx(
        [expected[i, j] for i, j in Voigt_pairs], rel=1e-8, abs=1e-6 * numpy.abs(expected).max()
    )


def test_an_isotropic_stiffness_matrix_is_the_isotropic_material():
    """The 6 x 6 stiffness of E and nu, given as a stiffness, and von Mises given as Hill's F =
    G = H = 1/2, L = M = N = 3/2: the plastic cube is the same to round-off."""
    E_v, nu = 200e9, 0.3
    lam, mu = E_v * nu / ((1 + nu) * (1 - 2 * nu)), E_v / (2 * (1 + nu))
    C = numpy.zeros((6, 6))
    C[:3, :3] = lam
    C[range(3), range(3)] += 2 * mu
    C[range(3, 6), range(3, 6)] = mu
    plasticity = {"yield_stress": "250e6 + 1e9*equivalent_plastic_strain"}
    results = []
    for solid in (
        dict(youngs_modulus=E_v, poissons_ratio=nu, plasticity=plasticity),
        dict(stiffness=C.tolist(), plasticity=plasticity),
        dict(
            youngs_modulus=E_v,
            poissons_ratio=nu,
            plasticity={**plasticity, "hill": [0.5, 0.5, 0.5, 1.5, 1.5, 1.5]},
        ),
    ):
        problem, body = cube(**solid)
        uniaxial(problem, body, "z", 300e6)
        problem.solve(report="none")
        results.append(strains(problem))
    assert results[0][2] == pytest.approx(300e6 / E_v + 0.05, rel=1e-8)
    for other in results[1:]:
        assert other == pytest.approx(results[0], rel=1e-8)


def test_hill_plasticity_yields_at_the_stress_of_its_coefficients():
    """Along axis 3, Hill's equivalent stress is sqrt(F + G) sigma, and the plastic strain along
    it is p sqrt(F + G): with sigma_y = 250 MPa + 1 GPa p, eps = sigma/E + p sqrt(F + G)."""
    F, G_h, H = 0.3, 0.9, 0.5
    sigma = 300e6
    problem, body = cube(
        youngs_modulus=200e9,
        poissons_ratio=0.3,
        plasticity={
            "yield_stress": "250e6 + 1e9*equivalent_plastic_strain",
            "hill": [F, G_h, H, 1.5, 1.5, 1.5],
        },
    )
    uniaxial(problem, body, "z", sigma)
    problem.solve(report="none")
    root = numpy.sqrt(F + G_h)
    plastic = (root * sigma - 250e6) / 1e9
    assert strains(problem)[2] == pytest.approx(sigma / 200e9 + plastic * root, rel=1e-8)
    # The lateral plastic strains follow the normal P sigma / q: -G and -F over sqrt(F + G).
    assert strains(problem)[0] == pytest.approx(
        -0.3 * sigma / 200e9 - plastic * G_h / root, rel=1e-8
    )


# Hill's F, G, H, L, M, N, all different, with linear hardening on an isotropic elastic cube.
Hill = [0.3, 0.9, 0.5, 1.2, 2.0, 0.9]
hardening = "250e6 + 1e9*equivalent_plastic_strain"
mu = 200e9 / (2 * (1 + 0.3))


def hill_cube(hill, **solid):
    return cube(
        youngs_modulus=200e9,
        poissons_ratio=0.3,
        plasticity={"yield_stress": hardening, "hill": hill},
        **solid,
    )


def uniaxial_x_strains(F, G_h, H, sigma):
    """The strains of uniaxial stress sigma along x for Hill's F, G, H in the global axes."""
    root = numpy.sqrt(G_h + H)
    plastic = (root * sigma - 250e6) / 1e9
    elastic = sigma / 200e9
    return numpy.array(
        [
            elastic + plastic * root,
            -0.3 * elastic - plastic * H / root,
            -0.3 * elastic - plastic * G_h / root,
        ]
    )


def simple_shear(problem, body, plane, shear):
    """Every face displaced as the simple shear u_i = shear x_j of the plane (i, j)."""
    i, j = plane
    conditions = [
        f"disp_{a} = {shear!r}*{'xyz'[j]}" if k == i else f"disp_{a} = 0"
        for k, a in enumerate("xyz")
    ]
    body.add_boundary_condition(conditions, boundary=list(problem.mesh.sideset_names()))


def shear_response(coefficient, shear):
    """The shear stress and the equivalent plastic strain of simple shear for the Hill
    coefficient of its plane."""
    root = numpy.sqrt(2 * coefficient)
    plastic = (root * mu * shear - 250e6) / (1e9 + mu * root**2)
    return mu * (shear - root * plastic), plastic


def test_hill_plasticity_along_x_yields_at_the_stress_of_G_and_H():
    """Uniaxial stress sigma along x, with F, G and H all different.  In the scale of the code,
    where F = G = H = 1/2 and L = M = N = 3/2 is von Mises, Hill's equivalent stress is q^2 =
    F (s_yy - s_zz)^2 + G (s_zz - s_xx)^2 + H (s_xx - s_yy)^2 + 2 L s_yz^2 + 2 M s_xz^2 +
    2 N s_xy^2 (Hill 1948, Proc. R. Soc. A 193, 281).  Along x, q = sqrt(G + H) sigma, so the
    material yields at sigma_y / sqrt(G + H).  The plastic strain p dq/dsigma has the components
    sqrt(G + H), -H / sqrt(G + H) and -G / sqrt(G + H) along x, y and z, and with sigma_y =
    250 MPa + 1 GPa p the yield q = sigma_y gives p = (sqrt(G + H) sigma - 250 MPa) / 1 GPa.
    Each strain is the elastic one plus its plastic part."""
    F, G_h, H = Hill[:3]
    problem, body = hill_cube(Hill)
    uniaxial(problem, body, "x", 300e6)
    problem.solve(report="none")
    assert strains(problem) == pytest.approx(uniaxial_x_strains(F, G_h, H, 300e6), rel=1e-8)


@pytest.mark.parametrize(
    "plane, voigt", [((1, 2), 3), ((0, 2), 4), ((0, 1), 5)], ids=["yz_L", "xz_M", "xy_N"]
)
def test_hill_plasticity_yields_in_shear_at_the_stress_of_L_M_and_N(plane, voigt):
    """Simple shear gamma = 1e-2 in the plane of the Hill coefficient c (L for yz, M for xz, N
    for xy): the stress is the shear tau alone, and Hill's equivalent stress of the previous test
    is q = sqrt(2 c) tau, so the shear yield stress is sigma_y / sqrt(2 c) (sigma_y / sqrt(3)
    for von Mises).  The plastic engineering shear is p dq/dtau = sqrt(2 c) p, so tau =
    mu (gamma - sqrt(2 c) p), with mu the shear modulus, and the yield sqrt(2 c) tau =
    250 MPa + h p, h = 1 GPa, gives p = (sqrt(2 c) mu gamma - 250 MPa) / (h + 2 c mu)."""
    problem, body = hill_cube(Hill)
    simple_shear(problem, body, plane, 1e-2)
    problem.solve(report="none")
    tau, plastic = shear_response(Hill[voigt], 1e-2)
    expected = numpy.zeros(6)
    expected[voigt] = tau
    assert problem.property_at_centroids("stress")[0] == pytest.approx(expected, abs=1e-8 * tau)
    assert problem.property_at_centroids("equivalent_plastic_strain")[0][0] == pytest.approx(
        plastic, rel=1e-8
    )


def test_the_orientation_turns_the_hill_coefficients_with_the_material_axes():
    """The material axes turned by 90 degrees about z, axis 1 along y and axis 2 along -x: the
    material stresses are s_11 = s_yy, s_22 = s_xx, s_33 = s_zz, s_23 = -s_xz, s_13 = s_yz and
    s_12 = -s_xy, so Hill's q of F, G, H, L, M, N in the material axes is the q of G, F, H, M,
    L, N in the global ones.  Uniaxial stress along x then has the strains of the previous
    uniaxial test with F and G swapped, and yields at sigma_y / sqrt(F + H), and simple shear in
    xz yields at the stress of L instead of M."""
    F, G_h, H, L = Hill[:4]
    turned = dict(orientation=[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    problem, body = hill_cube(Hill, **turned)
    uniaxial(problem, body, "x", 300e6)
    problem.solve(report="none")
    assert strains(problem) == pytest.approx(uniaxial_x_strains(G_h, F, H, 300e6), rel=1e-8)
    problem, body = hill_cube(Hill, **turned)
    simple_shear(problem, body, (0, 2), 1e-2)
    problem.solve(report="none")
    tau, plastic = shear_response(L, 1e-2)
    assert problem.property_at_centroids("stress")[0][4] == pytest.approx(tau, rel=1e-8)
    assert problem.property_at_centroids("equivalent_plastic_strain")[0][0] == pytest.approx(
        plastic, rel=1e-8
    )


def test_hill_creep_flows_along_its_normal():
    """Constant uniaxial stress along axis 3 with Hill creep at A q^n: the axial creep strain is
    A q^n sqrt(F + G) t, q = sqrt(F + G) sigma."""
    F, G_h, H = 0.4, 0.8, 0.5
    A, n, sigma = 1e-30, 3.0, 50e6
    problem, body = cube(
        youngs_modulus=200e9,
        poissons_ratio=0.3,
        temperature=700.0,
        creep={
            "rate": "A*von_mises_stress^n",
            "constants": {"A": A, "n": n},
            "hill": [F, G_h, H, 1.5, 1.5, 1.5],
        },
    )
    uniaxial(problem, body, "z", sigma)
    problem.solve(report="none")
    problem.solve_transient(end_time=1e6, time_step=2.5e5, report="none")
    root = numpy.sqrt(F + G_h)
    expected = sigma / 200e9 + A * (root * sigma) ** n * root * 1e6
    assert strains(problem)[2] == pytest.approx(expected, rel=1e-8)


def test_the_input_of_anisotropy_is_checked():
    with pytest.raises(InputError, match="not a stable material"):
        cube(youngs_modulus=E, poissons_ratio=[0.9, 0.9, 0.9], shear_modulus=G)[0].initialize()
    with pytest.raises(InputError, match="three numbers each"):
        cube(youngs_modulus=E, poissons_ratio=0.3, shear_modulus=G)[0].initialize()
    with pytest.raises(InputError, match="anisotropy act on the continua"):
        mesh = dualmesh.generate_line_mesh(start=0.0, end=1.0, num_elements=2)
        dualmesh.Problem(mesh).add_physics(
            "solid_mechanics",
            formulation="beam",
            youngs_modulus=E,
            poissons_ratio=Poisson_ratios,
            shear_modulus=G,
            cross_section={"width": 0.1, "height": 0.1},
        )


def test_a_fully_anisotropic_plate_in_plane_stress_has_the_compliance_of_the_material():
    """Plane stress sets sigma_zz = sigma_yz = sigma_xz = 0, so a uniform sigma_xx gives the
    in-plane strains of the full compliance S = C^-1: eps_xx = S_11 sigma, eps_yy = S_21 sigma and
    gamma_xy = S_61 sigma, also for a stiffness that couples the out-of-plane shears to zz.  The
    displacement u = (eps_xx x, eps_yy y + gamma_xy x) is linear, so the elements hold it."""
    rng = numpy.random.default_rng(3)
    A = rng.normal(size=(6, 6))
    C = (A @ A.T + 6.0 * numpy.eye(6)) * 1e10
    assert abs(C[2, 3]) > 1e8 and abs(C[2, 4]) > 1e8
    S = numpy.linalg.inv(C)
    sigma = 50e6
    mesh = dualmesh.generate_rectangle_mesh(0, size, 0, size, 2, 2)
    problem = dualmesh.Problem(mesh)
    body = problem.add_physics("solid_mechanics", formulation="plane_stress", stiffness=C.tolist())
    body.add_boundary_condition("disp_x = 0", boundary="left")
    body.add_boundary_condition("disp_y = 0", point=[0.0, 0.0])
    body.add_boundary_condition(f"traction = [{sigma}, 0]", boundary="right")
    problem.solve(report="none")
    far = size * (1 - 1e-9)
    stretch = problem.sample("disp_x", [[far, size / 2]])[0] / far
    lateral = problem.sample("disp_y", [[0.0, far]])[0] / far
    shear = problem.sample("disp_y", [[far, 0.0]])[0] / far
    assert [stretch, lateral, shear] == pytest.approx(
        [S[0, 0] * sigma, S[1, 0] * sigma, S[5, 0] * sigma], rel=1e-8
    )
