# SPDX-License-Identifier: LGPL-2.1-or-later
"""Generate the syntax reference from the object registry.

The reference is generated, so that it agrees with the library: the parameter
names, types, defaults and descriptions come from the same ``validParams``
declarations that validate a user's input at run time.  Only the worked
examples are written by hand, in ``_syntax_examples.py``.
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
    "framework": r"""
The framework objects carry no physics of their own.  Each one states a term
or a condition of the balance law :math:`\partial m/\partial t + \nabla \cdot
\mathbf{F} = s` of an unknown :math:`u`, in which :math:`m` is the storage,
:math:`\mathbf{F}` the flux and :math:`s` the source.  The flux is positive
along the outward unit normal :math:`\mathbf{n}` where it leaves a body.  The
framework objects build a model equation that has no physical name, such as
the equations of the verification problems.  The objects of the other modules
share their base classes, so the parameters ``variable``, ``block`` and
``quadrature`` mean the same in every module.
""",
    "heat_transfer": r"""
The heat transfer module states the energy balance :math:`\rho c_p \partial
T/\partial t + \nabla \cdot \mathbf{q} = s` of a body, with the temperature
:math:`T` in :math:`\mathrm{K}`, the density :math:`\rho` in
:math:`\mathrm{kg/m^3}`, the specific heat capacity :math:`c_p` in
:math:`\mathrm{J/(kg\,K)}` and the heat generated per unit volume :math:`s` in
:math:`\mathrm{W/m^3}`.  The heat flux :math:`\mathbf{q} = -k \nabla T` in
:math:`\mathrm{W/m^2}` follows Fourier's law, with the thermal conductivity
:math:`k` in :math:`\mathrm{W/(m\,K)}`.  The module holds the conduction, the
heat capacity term, the heat that a flow carries with its streamline
stabilisation, and the surface conditions of radiation and of heat transfer
across a gap.  The framework objects ``Neumann`` and ``Robin`` give a
prescribed heat flux and convection.  The conductivity may depend on the
position, the time, a property and the temperature, which makes the
conduction nonlinear.
""",
    "solid_mechanics": r"""
The solid mechanics module states the balance of momentum :math:`\rho\,
\partial^2 \mathbf{u}/\partial t^2 - \nabla \cdot \boldsymbol{\sigma} =
\mathbf{f}` of a solid, with the displacement :math:`\mathbf{u}` in
:math:`\mathrm{m}`, the Cauchy stress :math:`\boldsymbol{\sigma}` in
:math:`\mathrm{Pa}` and the body force per unit volume :math:`\mathbf{f}` in
:math:`\mathrm{N/m^3}`.  The flux of the equation of the displacement
component :math:`u_i` is minus row :math:`i` of :math:`\boldsymbol{\sigma}`.
The first table lists the objects outside the two groups below, such as
contact, fracture, damping and the members in space.
""",
    "fluids": r"""
The fluids module solves the steady and transient Navier-Stokes equations of
an incompressible Newtonian fluid, :math:`\rho (\partial \mathbf{v}/\partial t +
(\mathbf{v} \cdot \nabla) \mathbf{v}) - \nabla \cdot \boldsymbol{\sigma} =
\mathbf{f}` and :math:`\nabla \cdot \mathbf{v} = 0`, with the velocity
:math:`\mathbf{v}` in :math:`\mathrm{m/s}`, the pressure :math:`p` in
:math:`\mathrm{Pa}`, the dynamic viscosity :math:`\mu` in :math:`\mathrm{Pa\,s}`
and the stress :math:`\boldsymbol{\sigma} = -p \mathbf{I} + \mu (\nabla
\mathbf{v} + \nabla \mathbf{v}^{\mathsf{T}})`, where :math:`\mathbf{I}` is the
identity.  The equal-order formulation keeps the pressure as an unknown of the
same order as the velocity, with a residual-based stabilisation.  The
Taylor-Hood formulation keeps it with a quadratic velocity and a linear
pressure.  The penalty formulation eliminates the pressure by
:math:`p = -\gamma \nabla \cdot \mathbf{v}`, with the penalty parameter
:math:`\gamma` in :math:`\mathrm{Pa\,s}`, and integrates the penalty term with
a reduced rule.  A temperature drives the flow by the Boussinesq buoyancy
force.
""",
}

# Groups within a module, each with its own heading and introduction.  An
# object of the module that no group names is listed under the module heading.
MODULE_GROUPS = {
    "solid_mechanics": [
        (
            "Continuum elasticity",
            {
                "small_strain_stress",
                "stress_divergence",
                "traction",
                "pressure",
            },
            """
The stress of a continuum at small strain, in plane stress, plane strain,
axisymmetry and three dimensions, and its equilibrium.  The property object
``small_strain_stress`` computes the stress, and the kernel
``stress_divergence`` reads it in the equation of each displacement component,
so a constitutive model changes without a change to the equilibrium equations.
The conditions ``traction`` and ``pressure`` load the surface.
""",
        ),
        (
            "Structural members: beams and plates",
            {
                "beam_Euler-Bernoulli_mixed",
                "beam_Timoshenko_displacement",
                "beam_Timoshenko_mixed",
                "circular_plate_first_order",
                "circular_plate_classical_mixed",
                "plate_first_order",
            },
            """
In the reduced theories, a beam or a plate is an elastic body whose
displacement varies through the thickness in a prescribed way, and whose
equilibrium equations are integrated through the thickness.  Beams and
axisymmetric circular plates each have more than one theory, and rectangular
plates have the first-order theory.  Every model is a system of second-order
equations, the order that the dual mesh control domain method (DMCDM)
discretizes.  A theory whose natural statement is fourth order, such as the
classical beam or circular plate, is recast in mixed form with the bending
moment as an extra unknown.  Every model accepts a section graded through the
thickness and the von Karman nonlinearity of moderate rotations.
""",
        ),
    ]
}


def _category_title(category: str) -> str:
    """'nodal_boundary_condition' -> 'Nodal boundary condition'."""
    return category.replace("_", " ").capitalize()


CATEGORY_NOTES = {
    "kernel": (
        r"A **kernel** states one term of the equation of one variable :math:`u`, written as the balance law :math:`\partial m/\partial t + \nabla \cdot \mathbf{F} = s` with the storage :math:`m`, the flux :math:`\mathbf{F}` and the source :math:`s`.  The flux is the physical flux, positive along the outward unit normal :math:`\mathbf{n}` where it leaves the body, so the flux of a diffusion of diffusivity :math:`k` is :math:`\mathbf{F} = -k \nabla u`.  A kernel contributes to the storage, the flux, the source, or more than one of them, and the kernels of one variable add up."
    ),
    "boundary_condition": (
        r"An **integrated boundary condition** prescribes the normal flux :math:`\mathbf{n} \cdot \mathbf{F}` on a side set, with :math:`\mathbf{F}` the flux of the equation and :math:`\mathbf{n}` the outward unit normal.  The normal flux is the secondary variable of the equation, the partner of the unknown in their duality pair.  The condition is integrated over the side set, so it enters the equations of the control domains and the elements that touch the side set."
    ),
    "nodal_boundary_condition": (
        "A **nodal boundary condition** prescribes the unknown itself, the primary variable, at the nodes of a boundary.  It replaces the equation of each prescribed degree of freedom, and the residual of that equation at the converged solution is the reaction, the flux that the condition supplies."
    ),
    "nodal_load": ("A **nodal load** is a concentrated source applied at single nodes."),
    "property": (
        "A **property** object computes named properties at every integration point, and the kernels and the boundary conditions read them by name.  A property may depend on the position, the time and the solution, and its derivative with respect to the solution comes from automatic differentiation."
    ),
    "constraint": (
        r"A **constraint** makes some degrees of freedom copies of others.  A dependent degree of freedom takes the value of its primary, and its equation is added to the equation of the primary, so the two act as one unknown whose control domain is the union of theirs.  The Jacobian matrix :math:`\mathbf{J}` of the equations becomes :math:`\mathbf{Q}^{\mathsf{T}} \mathbf{J} \mathbf{Q}`, with :math:`\mathbf{Q}` the matrix that maps the independent degrees of freedom to all of them."
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
            "Syntax reference\n================\n\nThe syntax reference is generated from the compiled library, which was not available when this copy of the documentation was built. Run ``dualmesh list`` and ``dualmesh describe <type>`` to see the objects and their parameters.\n"
        )
        return

    import dualmesh
    from _syntax_examples import EXAMPLES

    by_module: dict[str, list[str]] = {}
    for name in dualmesh.registered_types():
        by_module.setdefault(dualmesh.object_module(name), []).append(name)
    ordered_modules = sorted(
        by_module, key=lambda m: list(MODULE_TITLES).index(m) if m in MODULE_TITLES else 99
    )

    missing_examples = []
    written = set()

    # ---- one page per object ------------------------------------------------
    for module in ordered_modules:
        for name in sorted(by_module[module]):
            category = dualmesh.object_category(name)
            parameters = dualmesh._core.object_parameters(name)
            required = [p for p in parameters if p["required"]]
            optional = [p for p in parameters if not p["required"]]
            lines = [
                f".. _syntax-{name}:",
                "",
                name,
                "=" * len(name),
                "",
                f"*{_category_title(category)}* in the *{MODULE_TITLES.get(module, module)}* module.",
                "",
                _wrap(_escape_inline(dualmesh._core.object_description(name))),
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
            written.add(f"{name}.rst")

    # ---- one page per physics and coupling ---------------------------------
    import dataclasses

    from dualmesh import physics as physics_module
    from dualmesh.parameters import _type_name

    physics_names = physics_module.registered()
    for name in physics_names:
        cls = physics_module._REGISTRY[name]
        required, optional = [], []
        for f in sorted(dataclasses.fields(cls), key=lambda f: f.name):
            if f.name.startswith("_"):
                continue
            missing = f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
            default = (
                ""
                if missing
                else repr(
                    f.default if f.default is not dataclasses.MISSING else f.default_factory()
                )
            )
            unit = f.metadata.get("unit", "")
            row = dict(
                name=f.name,
                type=_type_name(f.type) + (f" ({unit})" if unit else ""),
                default=default,
                description=f.metadata.get("description", ""),
            )
            (required if missing else optional).append(row)
        doc = " ".join((cls.__doc__ or "").split("\n\n")[0].split())
        lines = [
            f".. _syntax-{name}:",
            "",
            name,
            "=" * len(name),
            "",
            f"*{cls.category.capitalize()}* in the *{MODULE_TITLES.get(cls.module, cls.module)}* module.",
            "",
            _wrap(_escape_inline(doc)),
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
        written.add(f"{name}.rst")

    # ---- the index ----------------------------------------------------------
    index = [
        "Syntax reference",
        "================",
        "",
        _wrap(
            """
Each term, boundary condition and property of a model is an object with a
type name and a declared set of parameters.  You create an object by its type
name and set its parameters by keyword arguments:
"""
        ),
        "",
        ".. code-block:: python",
        "",
        '   problem.add_kernel("heat_conduction", variable="temperature", thermal_conductivity=20.0)',
        "",
        _wrap(
            """
The pages below are generated from the library, so the names, types, defaults
and descriptions of the parameters are the ones that check your input at run
time.  The command line gives the same information with ``dualmesh list`` and
``dualmesh describe <type>``, and Python gives it with
``dualmesh.describe("heat_conduction")``.

A misspelled parameter is an error, and its message names the closest valid
parameter and lists the parameters that the object accepts.  A missing required parameter is an error that names the
object and the parameter.
"""
        ),
        "",
        "Physics and couplings",
        "---------------------",
        "",
        _wrap(
            """
A physics adds a set of equations named in physical terms, and a coupling
joins two physics, as :doc:`/user_guide/problem_setup` describes.  The
boundary conditions and the extra terms of a physics are the objects of the
next section.
"""
        ),
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 28 22 50",
        "",
        "   * - Type",
        "     - Category",
        "     - Summary",
    ]
    for name in physics_names:
        cls = physics_module._REGISTRY[name]
        doc = " ".join((cls.__doc__ or "").split())
        first = _escape_inline(doc.split(". ")[0].rstrip(".")) + "."
        index += [
            f"   * - :doc:`{name} <{name}>`",
            f"     - {cls.category.capitalize()}",
            f"     - {first}",
        ]
    index += ["", "Objects by module", "-----------------", ""]
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
                summary = _escape_inline(dualmesh._core.object_description(name).replace("\n", " "))
                first = summary.split(". ")[0].rstrip(".") + "."
                index += [
                    f"   * - :doc:`{name} <{name}>`",
                    f"     - {_category_title(dualmesh.object_category(name))}",
                    f"     - {first}",
                ]
            index += [""]
    index += [".. toctree::", "   :hidden:", "   :maxdepth: 1", ""]
    for name in physics_names:
        index += [f"   {name}"]
    for module in ordered_modules:
        for name in sorted(by_module[module]):
            index += [f"   {name}"]
    index += [""]
    (out_dir / "index.rst").write_text("\n".join(index) + "\n")
    # Pages of objects that no longer exist (renamed or removed) would be
    # orphans that the strict build rejects: remove them.
    for page in out_dir.glob("*.rst"):
        if page.name != "index.rst" and page.name not in written:
            page.unlink()

    if missing_examples:
        print(
            "dualmesh docs: no usage example for "
            + ", ".join(missing_examples)
            + " (add one to docs/_syntax_examples.py)"
        )
