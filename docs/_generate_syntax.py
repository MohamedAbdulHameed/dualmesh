# SPDX-License-Identifier: LGPL-2.1-or-later
"""Generate the syntax reference from the object registry.

The reference is generated rather than written by hand, so that it can never
disagree with the library: the parameter names, types, defaults and
descriptions all come from the same ``validParams`` declarations that validate
a user's input at run time.  Only the worked examples are written by hand, in
``_syntax_examples.py``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

MODULE_TITLES = {
    "framework": "Framework",
    "heat_transfer": "Heat transfer",
    "solid_mechanics": "Solid mechanics",
    "fluids": "Viscous incompressible flow",
}

MODULE_INTROS = {
    "framework": """
The framework objects are the ones that carry no physics of their own.  They
are the building blocks a new application is assembled from, and they are what
the verification problems of the theory chapters use, because a model equation
has no physical name.  Every other module is built on the same base classes,
so everything said here about blocks, quadrature and load scaling applies
throughout.
""",
    "heat_transfer": """
The heat transfer module states Fourier conduction, internal generation,
storage, and the three surface conditions of the subject: prescribed flux,
convection, and radiation.  The conductivity may depend on position, on time,
on a material property, and on the temperature itself, which is what makes the
nonlinear examples of Reddy's Chapter 6 reproducible.
""",
    "solid_mechanics": """
The solid mechanics module covers the deformation of solids, from the
three-dimensional continuum down to the reduced theories of structural members.
Its two groups share the same variables, loads and boundary conditions, and a
problem may use both.
""",
    "fluids": """
The fluids module solves the steady and unsteady Navier-Stokes equations for an
incompressible Newtonian fluid using the penalty formulation, in which the
pressure is eliminated by the constitutive assumption :math:`P = -\\gamma
\\nabla \\cdot v`.  This leaves a problem in the velocities alone, at the cost of
requiring the penalty term to be under-integrated.
""",
}

# Groups within a module, each with its own heading and introduction.  An
# object of the module that no group names is listed under the module heading.
MODULE_GROUPS = {
    "solid_mechanics": [
        (
            "Continuum elasticity",
            {"LinearElasticStress", "StressDivergence", "TractionBC", "PressureBC"},
            """
Small-strain linear elasticity in plane stress, plane strain, axisymmetry and
three dimensions.  The stress is computed by a material and consumed by one
equilibrium kernel per displacement component, which is the same separation
MOOSE uses and which makes it straightforward to replace the constitutive model
without touching the equilibrium equations.
""",
        ),
        (
            "Structural members: beams and plates",
            {
                "BeamEulerBernoulliMixed",
                "BeamTimoshenkoDisplacement",
                "BeamTimoshenkoMixed",
                "CircularPlateFirstOrder",
                "CircularPlateClassicalMixed",
                "PlateFirstOrder",
            },
            """
The reduced theories: a beam or a plate is an elastic body whose displacement
is assumed to vary through the thickness in a prescribed way, and whose
equilibrium equations have been integrated through it.  Beams, axisymmetric
circular plates and rectangular plates are each available in more than one
theory.  Every model is written as a system of *second-order* equations,
because that is what the dual mesh control domain method can discretize; a
theory whose natural statement is fourth order, such as the classical beam or
plate, is therefore recast in mixed form with the bending moment as an extra
unknown.  All of them accept functionally graded section stiffnesses and the
von Karman moderate-rotation nonlinearity.
""",
        ),
    ],
}

CATEGORY_NOTES = {
    "Kernel": (
        "A **kernel** states one term of the governing equation of one variable, "
        "as a contribution to the flux :math:`\\mathbf{F}`, to the source "
        ":math:`S`, or to both, in the canonical conservation form "
        ":math:`-\\nabla\\cdot\\mathbf{F} + S = 0`.  Several kernels on the same "
        "variable are added together."
    ),
    "BoundaryCondition": (
        "An **integrated boundary condition** prescribes the secondary variable "
        "of the duality pair, that is, the normal flux :math:`n\\cdot\\mathbf{F}` "
        "on a side set.  It is integrated over the boundary, so it enters the "
        "equations of the control domains that touch that boundary."
    ),
    "NodalBC": (
        "A **nodal boundary condition** prescribes the primary variable itself at "
        "the nodes of a boundary.  The equation of the prescribed degree of "
        "freedom is replaced, and its residual, evaluated at the converged "
        "solution, is the reaction."
    ),
    "NodalLoad": (
        "A **nodal load** is a concentrated source applied at a node rather than "
        "distributed over an element or a side."
    ),
    "Material": (
        "A **material** computes named properties at every integration point, "
        "which kernels and boundary conditions then consume by name.  Properties "
        "may depend on position, time, and the solution, and any dependence on "
        "the solution is differentiated automatically."
    ),
}


def _escape_inline(text: str) -> str:
    """Escape the characters reStructuredText treats as inline markup.

    A parameter description is free prose written by whoever wrote the object,
    and it may perfectly reasonably contain an asterisk (as in ``*_property``)
    or a backtick.  Left alone, those start inline markup that never closes and
    the documentation build warns about it, so they are escaped here rather
    than forbidden there.
    """
    for character in ("\\", "*", "`", "|"):
        text = text.replace(character, "\\" + character)
    return text


def _wrap(text: str, width: int = 79) -> str:
    return "\n".join(
        textwrap.fill(paragraph, width=width) if paragraph.strip() else ""
        for paragraph in text.strip().split("\n")
    )


def _parameter_table(rows: list[dict]) -> list[str]:
    if not rows:
        return ["None.", ""]
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 20 14 12 54",
        "   :class: parameter-table",
        "",
        "   * - Name",
        "     - Type",
        "     - Default",
        "     - Description",
    ]
    for row in rows:
        default = row["default"]
        default = f"``{default}``" if default else "--"
        description = _escape_inline(row["description"].replace("\n", " ").strip())
        lines += [
            f"   * - ``{row['name']}``",
            f"     - {row['type']}",
            f"     - {default}",
            f"     - {description}",
        ]
    lines.append("")
    return lines


def write_syntax_reference(source_dir: Path, have_library: bool) -> None:
    out_dir = source_dir / "syntax"
    out_dir.mkdir(exist_ok=True)
    if not have_library:
        (out_dir / "index.rst").write_text(
            "Syntax reference\n================\n\n"
            "The syntax reference is generated from the compiled library, which was not "
            "available when this copy of the documentation was built. Run "
            "``dualmesh list`` and ``dualmesh describe <type>`` locally instead.\n"
        )
        return

    import dualmesh as dm
    from _syntax_examples import EXAMPLES

    by_module: dict[str, list[str]] = {}
    for name in dm.registered_types():
        by_module.setdefault(dm.object_module(name), []).append(name)
    ordered_modules = sorted(
        by_module, key=lambda m: list(MODULE_TITLES).index(m) if m in MODULE_TITLES else 99
    )

    missing_examples = []

    # ---- one page per object ------------------------------------------------
    for module in ordered_modules:
        for name in sorted(by_module[module]):
            category = dm.object_category(name)
            parameters = dm._core.object_parameters(name)
            required = [p for p in parameters if p["required"]]
            optional = [p for p in parameters if not p["required"]]
            lines = [
                f".. _syntax-{name}:",
                "",
                name,
                "=" * len(name),
                "",
                f"*{category}* in the *{MODULE_TITLES.get(module, module)}* module.",
                "",
                _wrap(_escape_inline(dm._core.object_description(name))),
                "",
                _wrap(CATEGORY_NOTES.get(category, "")),
                "",
            ]
            if required:
                lines += ["Required parameters", "-------------------", ""]
                lines += _parameter_table(required)
            lines += ["Optional parameters", "-------------------", ""]
            lines += _parameter_table(optional)
            example = EXAMPLES.get(name)
            if example:
                lines += ["Usage", "-----", "", example, ""]
            else:
                missing_examples.append(name)
            (out_dir / f"{name}.rst").write_text("\n".join(lines) + "\n")

    # ---- the index ----------------------------------------------------------
    index = [
        "Syntax reference",
        "================",
        "",
        _wrap(
            """
Every piece of physics in dualmesh is a named object with a declared,
validated, self-documenting set of parameters, in the same way as MOOSE.  An
object is created by name and configured by keyword, either from Python or from
an input file, and the two routes accept exactly the same names:
"""
        ),
        "",
        ".. code-block:: python",
        "",
        '   problem.add_kernel("HeatConduction", "conduction",',
        '                      variable="temperature", thermal_conductivity=20.0)',
        "",
        ".. code-block:: yaml",
        "",
        "   kernels:",
        "     conduction:",
        "       type: HeatConduction",
        "       variable: temperature",
        "       thermal_conductivity: 20.0",
        "",
        _wrap(
            """
The pages below are generated from the library itself, so the parameter names,
types, defaults and descriptions are exactly the ones that validate your input
at run time.  The same information is available at the command line with
``dualmesh list`` and ``dualmesh describe <type>``, and from Python with
``dualmesh.describe("HeatConduction")``.

Misspelling a parameter is an error, not a silently ignored keyword, and the
error message lists the parameters the object does accept.  Omitting a required
parameter is likewise an error, and names the object and the parameter.
"""
        ),
        "",
        "Objects by module",
        "-----------------",
        "",
    ]
    for module in ordered_modules:
        title = MODULE_TITLES.get(module, module)
        index += [title, "^" * len(title), ""]
        intro = MODULE_INTROS.get(module)
        if intro:
            index += [_wrap(intro), ""]
        groups = MODULE_GROUPS.get(module, [])
        grouped = set().union(*(members for _, members, _ in groups)) if groups else set()
        ungrouped = [n for n in sorted(by_module[module]) if n not in grouped]
        sections = ([(None, ungrouped, None)] if ungrouped else []) + [
            (heading, sorted(n for n in by_module[module] if n in members), text)
            for heading, members, text in groups
        ]
        for heading, names, text in sections:
            if not names:
                continue
            if heading:
                index += [heading, '"' * len(heading), ""]
            if text:
                index += [_wrap(text), ""]
            index += [
                ".. list-table::",
                "   :header-rows: 1",
                "   :widths: 28 22 50",
                "",
                "   * - Object",
                "     - Category",
                "     - Summary",
            ]
            for name in names:
                summary = _escape_inline(dm._core.object_description(name).replace("\n", " "))
                first = summary.split(". ")[0].rstrip(".") + "."
                index += [
                    f"   * - :doc:`{name} <{name}>`",
                    f"     - {dm.object_category(name)}",
                    f"     - {first}",
                ]
            index += [""]
    index += [
        ".. toctree::",
        "   :hidden:",
        "   :maxdepth: 1",
        "",
    ]
    for module in ordered_modules:
        for name in sorted(by_module[module]):
            index += [f"   {name}"]
    index += [""]
    (out_dir / "index.rst").write_text("\n".join(index) + "\n")

    if missing_examples:
        print(
            "dualmesh docs: no usage example for "
            + ", ".join(missing_examples)
            + " (add one to docs/_syntax_examples.py)"
        )
