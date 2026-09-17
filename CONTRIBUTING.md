# Contributing to dualmesh

Thank you for considering a contribution. This document explains how to build
the project, what is expected of a change, and how the code is organized.

## Building and testing

```console
git clone https://github.com/OWNER/dualmesh.git
cd dualmesh
pip install -e .[all,test]

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure   # C++ unit tests
pytest                                       # verification suite
```

`cmake --build` also places the Python extension in `python/dualmesh`, so
`PYTHONPATH=$PWD/python` is enough to use the package from a source tree.

## What a change should include

- **Tests.** New physics needs a verification case with a reference value that
  comes from somewhere trustworthy: a published table (cite it in a comment,
  as the existing tests do), an analytical solution, or a convergence study.
  Bug fixes need a test that fails before the fix.
- **Documentation.** New objects are documented by their own parameter
  descriptions (`addRequired`/`addOptional` and `setClassDescription`), which
  is what `dualmesh describe` and the object reference in the documentation
  print. Anything conceptually new belongs in `docs/theory.rst` as well.
- **A note in `CHANGELOG.md`** under "Unreleased".

## Style

- C++ follows the MOOSE conventions: two-space indentation, `_member`
  variables, `camelCase` functions, `PascalCase` types, `.clang-format` in the
  repository root (`clang-format -i` before committing).
- Python follows PEP 8 with descriptive, unabbreviated names: `num_elements`,
  `thermal_conductivity`, `transverse_displacement` — not `ne`, `k`, `w`.
  `ruff check python tests` and `ruff format` are configured in
  `pyproject.toml`.
- Parameter names in `validParams` are the user interface: spell them out, and
  give each one a sentence that explains what it means physically.

## Adding physics

A kernel supplies a flux, a source, or both, for the canonical form
`-div F + S = 0`; see `docs/developing.rst` for a worked example and for how
registration works. Prototyping in Python (`dm.PythonKernel`) is encouraged:
the interface is the same, so a model that works can be moved to C++ almost
unchanged.

## Reporting problems

Please open an issue with the smallest input that reproduces the behaviour, the
version (`python -c "import dualmesh; print(dualmesh.__version__)"`), and the
platform. A failing test case is the most useful bug report of all.

## Licence

Contributions are accepted under the LGPL-2.1-or-later licence of the project.
