# SPDX-License-Identifier: LGPL-2.1-or-later
"""Input files: the physics level of a problem written as data.

A case file (YAML) holds the same input as a Python script at the physics
level, with the same names: the mesh (a generator of the library with its
keywords, or a mesh file), the method, the physics keyed by the name a script
would give their variables, their boundary conditions written as text, the
couplings, the study and the output.  ``dualmesh run case.yaml`` runs it, and
:func:`run_case` runs a file or a dict from Python::

    mesh:
      generate: rectangle
      x: [0.0, 0.1]
      y: [0.0, 0.02]
      num_elements: [40, 8]
    method: fem
    physics:
      heat:
        type: heat_transfer
        thermal_conductivity: 20.0
        conditions:
          - {boundary: left, condition: "temperature = 300"}
    solve:
      study: steady

Every name is checked: an unknown section, physics parameter or study is
refused with the closest valid name.  The object level stays a Python tool.
"""

from __future__ import annotations

import dataclasses
import difflib
from pathlib import Path

from ._core import InputError

SECTIONS = ("mesh", "method", "coordinates", "physics", "couplings", "solve", "output")
STUDIES = ("steady", "transient", "eigenvalue")
OUTPUTS = ("fields", "csv")


@dataclasses.dataclass
class Case:
    """A case that has run.

    Attributes
    ----------
    problem : Problem
        The solved problem.
    physics : dict
        The physics of the case by their keys in the file, for example
        ``{"heat": ..., "solid": ...}``.
    result : SolveResult
        The result of the study.
    """

    problem: object
    physics: dict
    result: object


def load_case(source) -> dict:
    """Read a case file.

    Parameters
    ----------
    source : str, os.PathLike or dict
        The path of a YAML file, or a dict of its sections, which is returned as it is.

    Returns
    -------
    dict
        The sections of the case.  Numbers follow YAML 1.2, so ``1.0e6`` is a number.

    Raises
    ------
    InputError
        If the file does not exist or holds no sections.
    ModuleNotFoundError
        If PyYAML is not installed.
    """
    if isinstance(source, dict):
        return source
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Case files need PyYAML: pip install pyyaml (or dualmesh[yaml])."
        ) from error
    path = Path(source)
    if not path.is_file():
        raise InputError(f"The case file '{source}' does not exist.")
    data = yaml.load(path.read_text(), Loader=_loader(yaml))
    if not isinstance(data, dict):
        raise InputError(f"The case file '{source}' holds no sections (mesh, physics, solve, ...).")
    return data


def _loader(yaml):
    """PyYAML's safe loader with the numbers of YAML 1.2: PyYAML follows YAML 1.1, which reads
    1.0e6 (an exponent without a sign) as a text, and a physical constant written so would
    become an expression."""

    class Loader(yaml.SafeLoader):
        pass

    import re

    number = re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    )
    Loader.yaml_implicit_resolvers = {
        key: [r for r in resolvers if r[0] != "tag:yaml.org,2002:float"]
        for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }
    Loader.add_implicit_resolver("tag:yaml.org,2002:float", number, list("-+0123456789."))
    return Loader


def run_case(source) -> Case:
    """Build and solve a case, and write its output.

    Parameters
    ----------
    source : str, os.PathLike or dict
        The path of a YAML file, or a dict of the same sections: ``mesh``,
        ``method``, ``coordinates``, ``physics``, ``couplings``, ``solve`` and
        ``output`` (see :doc:`/user_guide/case_files`).

    Returns
    -------
    Case
        The solved problem, its physics by key and the result of the study.

    Raises
    ------
    InputError
        For an unknown or missing section, generator, study, output, physics
        key or parameter, named with the closest valid name.

    Examples
    --------
    >>> case = dm.run_case("heated_plate.yaml")  # doctest: +SKIP
    >>> case.problem.values("temperature").max()  # doctest: +SKIP
    """
    data = load_case(source)
    _check_keys(data, SECTIONS, "the case")
    for required in ("mesh", "physics"):
        if required not in data:
            raise InputError(f"The case needs the section '{required}'.")
    # The sections that act after the solve are checked before it.
    _check_keys(_mapping(data.get("output", {}), "The section 'output'"), OUTPUTS, "output")
    from .problem import Problem

    problem = Problem(_mesh(data["mesh"]), **_problem_keywords(data))
    physics = _add_physics(problem, data["physics"])
    _add_couplings(problem, data.get("couplings", []), physics)
    result = _solve(problem, data.get("solve", {}))
    _write_output(problem, data.get("output", {}))
    return Case(problem=problem, physics=physics, result=result)


def _check_keys(section: dict, allowed, where: str) -> None:
    for key in section:
        if key not in allowed:
            close = difflib.get_close_matches(str(key), list(allowed), n=1)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            raise InputError(
                f"{where}: unknown key '{key}'.{hint} The keys are {', '.join(allowed)}."
            )


def _mapping(value, where: str) -> dict:
    if not isinstance(value, dict):
        raise InputError(f"{where} is a mapping of names to values, not {value!r}.")
    return dict(value)


def _problem_keywords(data: dict) -> dict:
    return {key: data[key] for key in ("method", "coordinates") if key in data}


def _mesh(spec) -> object:
    """A generated mesh (``generate: rectangle`` for generate_rectangle_mesh, with its keywords) or
    a mesh file (``file: path``, with the keywords of read_mesh)."""
    import dualmesh

    spec = _mapping(spec, "The section 'mesh'")
    if ("generate" in spec) == ("file" in spec):
        raise InputError("The section 'mesh' gives one of 'generate' (a generator) or 'file'.")
    if "file" in spec:
        return dualmesh.read_mesh(spec.pop("file"), **spec)
    name = spec.pop("generate")
    generators = sorted(
        n[len("generate_") : -len("_mesh")]
        for n in dir(dualmesh)
        if n.startswith("generate_") and n.endswith("_mesh")
    )
    if name not in generators:
        close = difflib.get_close_matches(str(name), generators, n=1)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise InputError(
            f"mesh: unknown generator '{name}'.{hint} The generators are {', '.join(generators)}."
        )
    return getattr(dualmesh, f"generate_{name}_mesh")(**spec)


def _add_physics(problem, spec) -> dict:
    """Every physics under the name a script would give its variable, with its conditions."""
    spec = _mapping(spec, "The section 'physics'")
    physics = {}
    for key, entry in spec.items():
        entry = _mapping(entry, f"physics '{key}'")
        if "type" not in entry:
            raise InputError(f"physics '{key}': give its type, for example heat_transfer.")
        conditions = entry.pop("conditions", [])
        physics[key] = problem.add_physics(entry.pop("type"), **entry)
        if not isinstance(conditions, list):
            raise InputError(f"physics '{key}': conditions is a list of conditions.")
        for condition in conditions:
            _add_condition(physics[key], condition, f"physics '{key}'")
    return physics


def _add_condition(physics, condition, where: str) -> None:
    condition = _mapping(condition, f"{where}: a condition")
    for required in ("boundary", "condition"):
        if required not in condition:
            raise InputError(
                f"{where}: a condition needs '{required}', for example "
                '{boundary: left, condition: "temperature = 300"}.'
            )
    text, boundary = condition.pop("condition"), condition.pop("boundary")
    # The other keys are the definitions that the text uses by their names.
    physics.add_boundary_condition(text, boundary=boundary, **condition)


def _add_couplings(problem, spec, physics: dict) -> None:
    """Every coupling, with its physics named by their keys."""
    from . import physics as registry

    if not isinstance(spec, list):
        raise InputError("The section 'couplings' is a list of couplings.")
    for entry in spec:
        entry = _mapping(entry, "a coupling")
        if "type" not in entry:
            raise InputError("A coupling needs its type, for example thermal_expansion.")
        kind = entry.pop("type")
        cls = registry._REGISTRY.get(kind)
        takes_physics = (
            {f.name for f in dataclasses.fields(cls) if "Physics" in str(f.type)} if cls else set()
        )
        for name in takes_physics & set(entry):
            key = entry[name]
            if key not in physics:
                raise InputError(
                    f"coupling '{kind}': {name} names the physics '{key}', and the physics are "
                    f"{', '.join(physics)}."
                )
            entry[name] = physics[key]
        problem.add_coupling(kind, **entry)


def _solve(problem, spec):
    spec = _mapping(spec, "The section 'solve'")
    study = spec.pop("study", "steady")
    if study not in STUDIES:
        close = difflib.get_close_matches(str(study), STUDIES, n=1)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise InputError(
            f"solve: unknown study '{study}'.{hint} The studies are {', '.join(STUDIES)}."
        )
    if study == "steady":
        return problem.solve(**spec)
    if study == "transient":
        return problem.solve_transient(**spec)
    return problem.solve_eigenvalue(**spec)


def _write_output(problem, spec) -> None:
    spec = _mapping(spec, "The section 'output'")
    _check_keys(spec, OUTPUTS, "output")
    if "fields" in spec:
        problem.write_vtu(spec["fields"])
    if "csv" in spec:
        problem.write_csv(spec["csv"])
