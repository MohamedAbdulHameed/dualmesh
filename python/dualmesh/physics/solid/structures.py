# SPDX-License-Identifier: LGPL-2.1-or-later
"""The structural formulations and the members in space of a solid: the section of a plate,
a beam or a circular plate, the truss and shell members, their kernels, the free ends of the
mixed forms and the stress through a section."""

from __future__ import annotations

import numpy

from ...parameters import InputError
from ..base import boundary_names, label

#: The members in space: a line mesh (truss) or a surface mesh (shell) in a space of more dimensions.
MEMBERS = ("truss", "shell")
#: The formulations of the structural theories, which reduce the body to a surface or a line.
STRUCTURAL = ("plate", "beam", "circular_plate")
#: The theories of each structural formulation; the first is the default.
THEORIES = {
    "plate": ("first_order",),
    "beam": ("Euler-Bernoulli", "Timoshenko"),
    "circular_plate": ("first_order", "classical"),
}


def section(solid) -> dict:
    """The parameters of the section for the structural kernels: the
    material and the geometry, whose law the kernels integrate through
    the thickness, or the section stiffnesses."""
    where = label(solid.type_name, solid.name)
    given = {
        k: getattr(solid, k)
        for k in (
            "extensional_stiffness",
            "coupling_stiffness",
            "bending_stiffness",
            "shear_stiffness",
        )
        if getattr(solid, k) is not None
    }
    if given:
        if solid.cross_section is not None or solid.youngs_modulus is not None:
            raise InputError(
                f"{where}: give the section by its material (youngs_modulus, poissons_ratio and the thickness or cross_section) or by its section stiffnesses, not both."
            )
        if solid.thickness is not None:
            raise InputError(
                f"{where}: the section stiffnesses describe the section, so the thickness is not given with them."
            )
        needed = ["extensional_stiffness", "bending_stiffness"]
        if solid.theory in ("Timoshenko", "first_order"):
            needed.append("shear_stiffness")
        missing = [k for k in needed if k not in given]
        if missing:
            raise InputError(
                f"{where}: the {solid.formulation} of the {solid.theory} theory given by its section stiffnesses needs {' and '.join(missing)} as well (coupling_stiffness may be left out: it is 0 for a section symmetric about its mid-surface)."
            )
        unused = [k for k in given if k not in needed + ["coupling_stiffness"]]
        if unused:
            raise InputError(f"{where}: the {solid.theory} theory has no {' and '.join(unused)}.")
        stiffnesses = {k: float(v) for k, v in given.items()}
        if solid.poissons_ratio is not None:
            stiffnesses["poissons_ratio"] = solid.poissons_ratio
        return stiffnesses
    if solid.youngs_modulus is None:
        raise InputError(
            f"{where}: give youngs_modulus (with the thickness or the cross_section), or the section stiffnesses."
        )
    out = dict(youngs_modulus=solid.youngs_modulus)
    if solid.poissons_ratio is not None:
        out["poissons_ratio"] = solid.poissons_ratio
    if solid.formulation == "beam":
        shape = dict(solid.cross_section or {})
        shapes = ({"width", "height"}, {"area", "second_moment"})
        keys = set(shape) - {"shear_correction"}
        if keys not in shapes:
            raise InputError(
                f"{where}: give cross_section={{'width': b, 'height': h}} (a rectangle, through whose height the material is integrated) or {{'area': A, 'second_moment': I}} (a homogeneous section), and 'shear_correction' (default 5/6)."
            )
        if solid.theory == "Timoshenko" and "area" in shape and "shear_correction" not in shape:
            raise InputError(
                f"{where}: a Timoshenko beam with a section given by its area and second_moment needs its 'shear_correction' (5/6 for a rectangle, 0.9 for a solid circle), because it depends on the shape of the section."
            )
        if "shear_correction" in shape:
            out["shear_correction"] = float(shape.pop("shear_correction"))
        out["cross_section"] = {k: float(v) for k, v in shape.items()}
        return out
    if solid.cross_section is not None:
        raise InputError(f"{where}: a plate takes its thickness, not a cross_section.")
    if solid.thickness is None:
        raise InputError(
            f"{where}: give the thickness (m) of the {solid.formulation}, which its material needs to make its section."
        )
    out["thickness"] = float(solid.thickness)
    return out


#: The parameters of the structural formulations only, unset (None) on a continuum.
_STRUCTURAL_ONLY = (
    "theory",
    "beam_formulation",
    "extensional_stiffness",
    "coupling_stiffness",
    "bending_stiffness",
    "shear_stiffness",
    "in_plane_load",
)
#: The loads of the structural formulations only, zero on a continuum.
_STRUCTURAL_LOADS = ("transverse_load", "axial_load", "foundation_modulus")


def check_structural_parameters(solid) -> None:
    """The checks that need no problem: the parameters of a plate, a beam or a circular plate,
    and, for any other formulation, that it is given none of them."""
    if solid.formulation in STRUCTURAL:
        _check_structural(solid)
    else:
        _check_no_structural(solid)


def _check_structural(solid) -> None:
    """The damping, theory and loads of a plate, a beam or a circular plate, with its theory
    set to the default when not given."""
    if (solid.damping or {}).get("stiffness"):
        raise InputError(
            f"solid_mechanics: the {solid.formulation} takes mass-proportional damping only. Stiffness-proportional damping of the structures is not available."
        )
    if solid.density is not None and solid.cross_section is None and solid.thickness is None:
        raise InputError(
            f"solid_mechanics: the inertia of the {solid.formulation} needs its geometry (thickness or cross_section), not only its section stiffnesses."
        )
    theories = THEORIES[solid.formulation]
    solid.theory = solid.theory or theories[0]
    if solid.theory not in theories:
        raise InputError(
            f"solid_mechanics: the {solid.formulation} takes the theory {' or '.join(theories)}, not '{solid.theory}'."
        )
    if solid.beam_formulation is not None and not (
        solid.formulation == "beam" and solid.theory == "Timoshenko"
    ):
        raise InputError(
            "solid_mechanics: only the Timoshenko beam has a choice of beam_formulation (mixed or displacement)."
        )
    if solid.beam_formulation not in (None, "mixed", "displacement"):
        raise InputError(
            f"solid_mechanics: beam_formulation is mixed or displacement, not '{solid.beam_formulation}'."
        )
    if solid.body_force is not None:
        raise InputError(
            f"solid_mechanics: the {solid.formulation} takes its loads as transverse_load, axial_load or in_plane_load, not body_force."
        )
    if solid.in_plane_load is not None and solid.formulation != "plate":
        raise InputError("solid_mechanics: in_plane_load is the load of a plate.")


def _check_no_structural(solid) -> None:
    """That a continuum or a member is given no parameter of the structural formulations.  The
    truss takes a cross_section too."""
    given = [k for k in _STRUCTURAL_ONLY if getattr(solid, k) is not None]
    if solid.cross_section is not None and solid.formulation != "truss":
        given.append("cross_section")
    given += [k for k in _STRUCTURAL_LOADS if getattr(solid, k) not in (0, 0.0)]
    if solid.von_karman:
        given.append("von_karman")
    if given:
        raise InputError(
            f"solid_mechanics: {', '.join(given)} belong to the structural formulations (plate, beam, circular_plate{', and the truss' if 'cross_section' in given else ''}); give formulation= one of them."
        )


def check_member(solid, problem) -> None:
    """The checks of a truss or a shell: the mesh, the section and the parameters the members
    do not take."""
    where = label(solid.type_name, solid.name)
    mesh = problem.mesh
    if problem.coordinates == "axisymmetric":
        raise InputError(f"{where}: a {solid.formulation} takes Cartesian coordinates.")
    if solid.formulation == "truss":
        if mesh.dimension != 1:
            raise InputError(
                f"{where}: a truss takes a line mesh, and this mesh has {mesh.dimension} dimensions."
            )
        if set(solid.cross_section or {}) != {"area"} or solid.thickness is not None:
            raise InputError(f"{where}: give the bars' cross_section={{'area': A}}.")
        allowed = {"Edge2"}
    else:
        if mesh.dimension != 2 or mesh.space_dimension != 3:
            raise InputError(f"{where}: a shell takes a surface mesh in three dimensions.")
        if solid.thickness is None or solid.cross_section is not None:
            raise InputError(f"{where}: give the shell's thickness.")
        if not isinstance(solid.poissons_ratio, (int, float)):
            raise InputError(f"{where}: give the shell's poissons_ratio, a number.")
        # The one-point shear of a linear triangle still locks; the quadrilateral does not.
        allowed = {"Quad4"}
    other = sorted(mesh.element_types() - allowed)
    if other:
        raise InputError(
            f"{where}: a {solid.formulation} takes {', '.join(sorted(allowed))} elements, and this mesh has {', '.join(other)}."
        )
    if solid.formulation == "truss" and not isinstance(solid.youngs_modulus, (int, float)):
        raise InputError(f"{where}: give the truss's youngs_modulus, a number.")
    if solid.formulation == "shell" and not isinstance(solid.youngs_modulus, (int, float, str)):
        raise InputError(
            f"{where}: give the shell's youngs_modulus, a number, a function or an expression."
        )
    components = mesh.space_dimension if solid.formulation == "truss" else 3
    if solid.body_force is not None and (
        len(solid.body_force) != components
        or not all(isinstance(f, (int, float)) for f in solid.body_force)
    ):
        raise InputError(
            f"{where}: the body_force of a {solid.formulation} is {components} numbers (N/m^3), one per direction of space."
        )
    unavailable = [
        k
        for k in (
            "shear_modulus",
            "orientation",
            "stiffness",
            "creep",
            "plasticity",
            "phase_field_fracture",
            "viscoelasticity",
            "strain_energy",
            "temperature",
            "density",
            "damping",
        )
        if getattr(solid, k) is not None
    ]
    if solid.large_deformation:
        unavailable.append("large_deformation")
    if solid.elasticity != "linear":
        unavailable.append("elasticity")
    if solid.initial_condition:
        unavailable.append("initial_condition")
    if solid.axial_force not in (0, 0.0):
        unavailable.append("axial_force")
    if unavailable:
        raise InputError(
            f"{where}: a {solid.formulation} is linear elastic and static, and takes no {', '.join(unavailable)}."
        )


def build_member(solid, problem) -> None:
    """The kernels of a truss, or the membrane, bending and shear kernels of a shell."""
    expansion = solid._coupled.get("thermal_expansion") or {}
    if expansion and solid.formulation == "truss":
        raise InputError(
            f"{label(solid.type_name, solid.name)}: the thermal expansion of a truss is not available."
        )
    names = solid.variable_names()
    force = {"body_force": [float(f) for f in solid.body_force]} if solid.body_force else {}
    restrict = solid._restriction()
    if solid.formulation == "truss":
        problem.add_kernel(
            "truss",
            displacements=names,
            youngs_modulus=float(solid.youngs_modulus),
            area=float(solid.cross_section["area"]),
            **force,
            **restrict,
        )
        return
    options = dict(
        displacements=names[:3],
        rotations=names[3:],
        youngs_modulus=solid.youngs_modulus,
        poissons_ratio=float(solid.poissons_ratio),
        thickness=float(solid.thickness),
        # A thermal strain enters the section as its thermal force and moment, as in a plate.
        **expansion,
        **force,
        **restrict,
    )
    problem.add_kernel("shell", shear_treatment="exclude", **options)
    problem.add_kernel(
        "shell",
        shear_treatment="only",
        quadrature="midpoint",
        reduced_integration=True,
        **options,
    )


def beam_symmetry(solid, boundary, where: str):
    """A plane of symmetry of a beam end.  It holds the axial displacement and the slope
    (phi in the displacement form, the natural condition of the moment equation in the
    mixed forms), and its moment stays free."""
    if solid.formulation != "beam":
        raise InputError(
            f"{where}: a plane of symmetry of a {solid.formulation} is written as text, e.g. 'disp_x = 0' and 'phi_x = 0'."
        )
    solid._symmetric += boundary_names(boundary)
    held = ["disp_x"] + (["phi"] if solid._third() == "phi" else [])
    return solid._problem.add_boundary_condition("Dirichlet", boundary, variable=held, value=0.0)


def build_structural(solid, problem) -> None:
    """The section kernels of a plate, a beam or a circular plate, with the free ends of the
    mixed forms."""
    restrict = solid._restriction()
    # A thermal strain, uniform through the thickness, enters the section as its thermal force
    # and moment.
    expansion = solid._coupled.get("thermal_expansion") or {}
    sections = dict(
        section(solid),
        **expansion,
        transverse_load=solid.transverse_load,
        foundation_modulus=solid.foundation_modulus,
        von_karman=solid.von_karman,
    )
    # The plate kernel takes its in-plane load as in_plane_load and refuses an axial_load, so
    # it receives one only when the user gave a load.
    if solid.formulation != "plate" or solid.axial_load not in (0, 0.0):
        sections["axial_load"] = solid.axial_load
    variables = solid.variable_names()
    if solid.formulation == "plate":
        options = dict(
            sections,
            in_plane_displacements=variables[:2],
            transverse_displacement=variables[2],
            rotations=variables[3:],
            **({"in_plane_load": list(solid.in_plane_load)} if solid.in_plane_load else {}),
            **restrict,
        )
        solid._section_kernel = problem.add_kernel(
            "plate_first_order", shear_treatment="exclude", **options
        )
        problem.add_kernel(
            "plate_first_order",
            shear_treatment="only",
            quadrature="midpoint",
            reduced_integration=True,
            **options,
        )
        return
    third = variables[2]
    if solid.formulation == "beam":
        model = {
            ("Euler-Bernoulli", "bending_moment"): "beam_Euler-Bernoulli_mixed",
            ("Timoshenko", "bending_moment"): "beam_Timoshenko_mixed",
            ("Timoshenko", "phi"): "beam_Timoshenko_displacement",
        }[(solid.theory, third)]
        mapping = dict(axial_displacement=variables[0], transverse_displacement=variables[1])
        mapping["rotation" if third == "phi" else "bending_moment"] = third
        for variable in variables:
            options = dict(sections, **mapping, variable=variable, **restrict)
            if third == "phi" and variable == third:
                options.update(quadrature="midpoint", reduced_integration=True)
            solid._section_kernel = problem.add_kernel(model, **options)
        if third == "bending_moment":
            hold_free_ends(solid, problem, third)
        return
    mapping = dict(radial_displacement=variables[0], transverse_displacement=variables[1])
    options = dict(sections, **restrict)
    if solid.theory == "classical":
        solid._section_kernel = problem.add_kernel(
            "circular_plate_classical_mixed", bending_moment=third, **mapping, **options
        )
        hold_free_ends(solid, problem, third)
        return
    solid._section_kernel = problem.add_kernel(
        "circular_plate_first_order",
        rotation=third,
        shear_treatment="exclude",
        **mapping,
        **options,
    )
    problem.add_kernel(
        "circular_plate_first_order",
        rotation=third,
        shear_treatment="only",
        quadrature="midpoint",
        reduced_integration=True,
        **mapping,
        **options,
    )


def hold_free_ends(solid, problem, moment: str) -> None:
    """Hold the bending moment at zero on every end of a mixed beam or
    classical circular plate that is not clamped, not a plane of symmetry
    and has no moment of its own.  The natural condition of the moment equation is a zero slope,
    so an end left without a condition would act as a guided end.  The
    axis of a circular plate keeps its natural condition, the zero slope
    of symmetry."""
    mesh = problem._global_mesh
    points = numpy.asarray(mesh.points())
    # An end is an end of the field: the blocks of every physics that shares the moment (a
    # stepped beam of two physics joins at a node that is no end).
    sharing = [p for p in problem._physics.values() if moment in p.variable_names()]
    solid._holds_free_ends = True
    # The ends of the field are derived once, by the first physics that shares it, from the
    # end conditions of all of them.
    if sharing and sharing[0] is not solid:
        return
    blocks = set()
    for physics in sharing:
        if not getattr(physics, "block", ()):
            blocks = set()
            break
        blocks |= {str(b) for b in physics.block}
    names = mesh.block_names()
    uses = numpy.zeros(mesh.num_nodes, dtype=int)
    for _type, nodes, block in mesh.cells():
        if not blocks or str(block) in blocks or names.get(block) in blocks:
            uses[list(nodes[:2])] += 1
    ends = set(numpy.flatnonzero(uses == 1).tolist())
    for physics in sharing or [solid]:
        held = physics._clamped + physics._symmetric + physics._prescribed.get(moment, [])
        for boundary in held:
            ends -= set(mesh.boundary_nodes(boundary))
    size = float(numpy.ptp(points, axis=0).max())
    if solid.formulation == "circular_plate":
        ends = {n for n in ends if abs(points[n, 0]) > 1e-12 * size}
    if not ends:
        return
    targets = points[sorted(ends)]
    name = solid._object_name("free_ends")
    problem.mesh.add_nodeset_by_predicate(
        name,
        lambda x, y, z: bool(
            numpy.min(numpy.linalg.norm(targets - [x, y, z], axis=1)) <= 1e-9 * size
        ),
    )
    problem.add_boundary_condition("Dirichlet", name, variable=moment, value=0.0)


def section_stress(solid, z):
    """The stress at the height ``z`` of the section of a structural formulation, at the
    element centres, one row per element (:meth:`SolidMechanics.stress`)."""
    where = label(solid.type_name, solid.name)
    problem = solid._problem
    if z is None:
        raise InputError(
            f"{where}: give the height z of the {solid.formulation} section: a number in m, or 'top', 'mid' or 'bottom'."
        )
    # The height of the section, when its material and shape give it.
    if solid.formulation == "beam":
        height = (solid.cross_section or {}).get("height")
    else:
        height = solid.thickness
    if isinstance(z, str):
        if height is None:
            raise InputError(
                f"{where}: the faces of this section are not known (it is given by its area or its stiffnesses); give z as a number in m."
            )
        heights = {"top": 0.5 * height, "mid": 0.0, "bottom": -0.5 * height}
        if z not in heights:
            raise InputError(f"{where}: z is a number, 'top', 'mid' or 'bottom', not '{z}'.")
        z = heights[z]
    elif height is not None and abs(float(z)) > 0.5 * float(height) * (1 + 1e-12):
        raise InputError(
            f"{where}: z = {z} m lies outside the section, whose faces are at -{0.5 * float(height)} m and {0.5 * float(height)} m."
        )
    problem.initialize()
    rows = problem._problem.section_stress_at_centroids(solid._section_kernel.name, float(z))
    # An element of a block the physics does not cover has no row: NaN there.
    width = max((len(r) for r in rows), default=0)
    return numpy.array([list(r) if len(r) == width else [numpy.nan] * width for r in rows])
