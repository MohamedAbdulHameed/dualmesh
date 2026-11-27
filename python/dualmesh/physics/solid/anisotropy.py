# SPDX-License-Identifier: LGPL-2.1-or-later
"""The anisotropic elasticity of a solid continuum: the stiffness of an orthotropic material
from its moduli, the rotation of the material axes, the Bond matrix that turns a Voigt matrix
from the material axes into the global axes, and Hill's matrix of the anisotropic equivalent
stress.  The stress object takes both matrices in the global axes."""

from __future__ import annotations

import numpy

from ...parameters import InputError
from ..base import label


def global_stiffness(solid):
    """The 6 x 6 stiffness in the global axes of an orthotropic or anisotropic continuum."""
    where = label(solid.type_name, solid.name)
    if solid.stiffness is not None:
        if (
            solid.youngs_modulus is not None
            or solid.poissons_ratio is not None
            or solid.shear_modulus is not None
        ):
            raise InputError(f"{where}: give the stiffness or the moduli, not both.")
        local = numpy.asarray(solid.stiffness, dtype=float)
        if local.shape != (6, 6):
            raise InputError(f"{where}: the stiffness is a 6 x 6 matrix.")
    else:
        try:
            E = [float(v) for v in solid.youngs_modulus]
            nu = [float(v) for v in solid.poissons_ratio]
            G = [float(v) for v in solid.shear_modulus]
        except TypeError:
            raise InputError(
                f"{where}: an orthotropic material takes three numbers each: youngs_modulus (E1, E2, E3), poissons_ratio (nu12, nu13, nu23) and shear_modulus (G12, G13, G23)."
            ) from None
        if not (len(E) == len(nu) == len(G) == 3) or min(E + G) <= 0.0:
            raise InputError(
                f"{where}: an orthotropic material takes three positive moduli E, three Poisson's ratios and three positive shear moduli."
            )
        compliance = numpy.zeros((6, 6))
        compliance[0, 0], compliance[1, 1], compliance[2, 2] = 1 / E[0], 1 / E[1], 1 / E[2]
        compliance[0, 1] = compliance[1, 0] = -nu[0] / E[0]
        compliance[0, 2] = compliance[2, 0] = -nu[1] / E[0]
        compliance[1, 2] = compliance[2, 1] = -nu[2] / E[1]
        compliance[3, 3], compliance[4, 4], compliance[5, 5] = 1 / G[2], 1 / G[1], 1 / G[0]
        if numpy.min(numpy.linalg.eigvalsh(compliance)) <= 0.0:
            raise InputError(f"{where}: these orthotropic constants are not a stable material.")
        local = numpy.linalg.inv(compliance)
    M = bond_matrix(solid)
    return M @ local @ M.T


def material_rotation(solid):
    """The rotation whose columns are the material axes in the global axes."""
    where = label(solid.type_name, solid.name)
    axes = solid.orientation
    if axes is None:
        return numpy.identity(3)
    if isinstance(axes, (int, float)):
        angle = numpy.radians(float(axes))
        return numpy.array(
            [
                [numpy.cos(angle), -numpy.sin(angle), 0.0],
                [numpy.sin(angle), numpy.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    a1, a2 = (numpy.asarray(v, dtype=float) for v in axes)
    if a1.shape != (3,) or a2.shape != (3,):
        raise InputError(
            f"{where}: orientation takes two vectors of three components, or an angle in 2D."
        )
    a1 = a1 / numpy.linalg.norm(a1)
    a2 = a2 - (a2 @ a1) * a1
    if numpy.linalg.norm(a2) < 1e-12:
        raise InputError(f"{where}: the two material axes are parallel.")
    a2 = a2 / numpy.linalg.norm(a2)
    return numpy.column_stack([a1, a2, numpy.cross(a1, a2)])


def bond_matrix(solid):
    """The 6 x 6 Bond matrix M with sigma_global = M sigma_material, in the Voigt order with
    tensor shears.  The engineering strains transform with M^T, so C_global = M C M^T."""
    R = material_rotation(solid)
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    M = numpy.zeros((6, 6))
    for J, (row, column) in enumerate(pairs):
        unit = numpy.zeros((3, 3))
        unit[row, column] = unit[column, row] = 1.0
        rotated = R @ unit @ R.T
        M[:, J] = [rotated[i, j] for i, j in pairs]
    return M


def hill_matrix(solid, coefficients):
    """Hill's matrix P of q^2 = sigma . P sigma in the global axes, from F, G, H, L, M, N in
    the material axes."""
    where = label(solid.type_name, solid.name)
    values = [float(v) for v in coefficients]
    if len(values) != 6:
        raise InputError(f"{where}: hill takes six coefficients F, G, H, L, M, N.")
    F, G, H, L, M_, N = values
    P = numpy.array(
        [
            [G + H, -H, -G, 0, 0, 0],
            [-H, F + H, -F, 0, 0, 0],
            [-G, -F, F + G, 0, 0, 0],
            [0, 0, 0, 2 * L, 0, 0],
            [0, 0, 0, 0, 2 * M_, 0],
            [0, 0, 0, 0, 0, 2 * N],
        ]
    )
    inverse = numpy.linalg.inv(bond_matrix(solid))
    return inverse.T @ P @ inverse
