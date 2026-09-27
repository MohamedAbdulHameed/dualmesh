Developing
==========

Layout
------

.. code-block:: text

   include/dualmesh/      public C++ headers
     core/                ADReal, InputParameters, Function, Types
     mesh/                Mesh (nodes, elements, side sets, generators)
     fe/                  ReferenceElement (dual mesh geometry), Assembly
     base/                Object, Kernel, Material, Factory, Problem
     modules/             the physics modules' public headers
   src/                   the implementations, mirroring include/
   python/bindings/       the pybind11 module
   python/dualmesh/       the Python package
   tests/cpp/             C++ unit tests (self-contained, no framework needed)
   tests/python/          verification against the book
   verification/openfoam/ cross-verification scripts
   examples/              runnable examples and input files
   docs/                  this documentation

Adding a kernel in C++
----------------------

A kernel supplies a flux, a source, or both, and declares its parameters:

.. code-block:: c++

   class MyReaction : public Kernel
   {
   public:
     static InputParameters validParams()
     {
       InputParameters p = Kernel::validParams();
       p.setClassDescription("Reaction term c u^2.");
       p.addOptional("coefficient", ParameterKind::Real, 1.0, "The coefficient c.");
       return p;
     }
     explicit MyReaction(const InputParameters & p)
       : Kernel(p), _c(p.getReal("coefficient")) {}

     bool hasSource() const override { return true; }
     ADReal computeSource(const QpContext & ctx) const override
     {
       const ADReal & u = ctx.value(_var);
       return _c * u * u;
     }

   private:
     double _c;
   };

Register it in the module's registration function, for example

.. code-block:: c++

   void registerHeatTransferObjects(Factory & f)
   {
     f.add<MyReaction>("my_reaction", ObjectCategory::Kernel, "heat_transfer");
   }

and it becomes available by name from Python, from input files, and in
``dualmesh list``.  Registration is explicit (rather than by static
initialization) so that nothing is dropped when the library is linked
statically.

Writing kernels correctly
-------------------------

* Use ``ctx.value(v)`` and ``ctx.gradient(v)`` for the quantities that carry
  derivatives, and ``ctx.coefficient_value(v)`` /
  ``ctx.coefficient_gradient(v)`` inside nonlinear *coefficients*.  The latter
  return the current iterate under Newton's method and the previous iterate
  under direct iteration, which is what makes one kernel serve both schemes.
* Never compare AD numbers to decide a branch that depends on the unknown
  unless the derivative of the branch is what you intend.
* Material properties are requested by name with
  ``problem.propertyRegistry().id("stress")`` in ``initialSetup`` and read with
  ``ctx.property(id, component)``; list them in ``requiredProperties()`` so a
  missing material is reported before the solve starts.

Adding a kernel in Python
-------------------------

For prototyping, the same interface is available from Python, with exact
derivatives:

.. code-block:: python

   import dualmesh as dm

   class ArrheniusSource(dm.PythonKernel):
       def setup(self, problem):
           self.temperature = problem.variable_index(self.temperature_variable)

       def compute_source(self, ctx):
           T = ctx.value(self.temperature)
           return -self.pre_exponential * dm.exp(-self.activation_energy / T)

   problem.add_kernel(ArrheniusSource(
       variable="temperature", temperature_variable="temperature",
       pre_exponential=1.0e6, activation_energy=5000.0))

Testing
-------

* ``tests/cpp/unit_tests.cpp`` holds the fast structural tests: the automatic
  differentiation, the quadrature rules, the geometric closure of the dual
  mesh, and the patch test.
* ``tests/python`` holds the verification suite.  Every value in it comes from
  a published table; when adding a case, cite the table in a comment.
* Run both with ``ctest --test-dir build`` and ``pytest``.

Style
-----

C++ follows the MOOSE conventions (two-space indentation, ``_member``
variables, ``camelCase`` functions, ``PascalCase`` types); Python follows PEP 8
with descriptive, unabbreviated parameter names.  ``.clang-format`` and the
``ruff`` configuration in ``pyproject.toml`` encode both.
Formatting is checked in continuous integration by clang-format **18** (the
version on the ``ubuntu-latest`` runner); other major versions format some
constructs differently, so format with version 18 (``brew install llvm@18`` on
macOS, ``apt install clang-format-18`` on Ubuntu).  The exact gate is

.. code-block:: console

   find include src python/bindings tests/cpp \( -name '*.h' -o -name '*.cpp' \) \
     -print0 | xargs -0 clang-format --dry-run --Werror

No output means success.

Making a release
----------------

The package is published on PyPI as ``dualmesh-multiphysics``; it is imported
as ``dualmesh`` and its command is ``dualmesh``.  The repository
(``MohamedAbdulHameed/dualmesh``), the Read the Docs project (``dualmesh``,
https://dualmesh.readthedocs.io), the import name and the command all keep the
name ``dualmesh``.  The first release, v0.1.0, was tagged on commit
``bc96db4512378f525729f6dc48e83652fc982db6``.

``scripts/release.py`` runs every step of a release as a gate and stops at the
first one that fails.  It uses only the Python standard library.

.. code-block:: console

   python scripts/release.py check --version X.Y.Z   # gates 1-15, local
   python scripts/release.py push  --version X.Y.Z   # gate 16
   python scripts/release.py ci --wait               # gate 17
   python scripts/release.py dry-run --wait          # gates 18-19
   python scripts/release.py tag   --version X.Y.Z   # gate 20
   python scripts/release.py monitor --version X.Y.Z --wait   # gates 21-22
   python scripts/release.py verify-pypi --version X.Y.Z      # gates 23-24

or ``python scripts/release.py release --version X.Y.Z`` for all of them in
order.  ``python scripts/release.py check --dev`` runs the local gates at any
time without requiring a new version.

The gates, in order:

1. ``origin`` is ``git@github-personal:MohamedAbdulHameed/dualmesh.git``, the
   personal SSH alias (identity ``~/.ssh/id_ed25519_mohamed``).
2. The branch is ``main``.
3. ``git fetch`` and ``git pull --ff-only origin main`` succeed: the branch is
   not behind ``origin/main`` and has not diverged from it.
4. The working tree is clean and no tracked file contains a genuine merge
   conflict.  ``git diff --check`` is not used for this, because it reports
   every reStructuredText title underline of seven ``=`` (``Solving`` over
   ``=======``) as a leftover conflict marker; a line of ``=`` counts only
   between ``<<<<<<<`` and ``>>>>>>>``.
5. ``pyproject.toml`` names the distribution ``dualmesh-multiphysics``, the
   command ``dualmesh = "dualmesh.cli:main"`` and the documentation URL.
6. The version is the same in ``pyproject.toml``, ``python/dualmesh/__init__.py``
   and ``CITATION.cff`` (the documentation reads it from ``pyproject.toml``),
   and ``CHANGELOG.md`` has a ``## [X.Y.Z] - YYYY-MM-DD`` section.
7. The tag ``vX.Y.Z`` exists neither locally nor on ``origin``.

   A further static gate checks that the fixes of v0.1.0 are still in place:
   the wheel matrix ``[ubuntu-latest, macos-15-intel, macos-14,
   windows-latest]`` (no retired macOS 13 runner), cibuildwheel v4.2.1 or
   later, the ``workflow_dispatch`` trigger, the tag guard and the ``pypi``
   environment of the publish job, Trusted Publishing without a stored token,
   and the Furo class on every ``.. contents::`` directive.
8. clang-format 18 ``--dry-run --Werror`` on every C++ file.
9. The C++ library is configured, built and tested with CTest.
10. Ruff (``check`` and ``format --check``).
11. The sdist is built in isolation, and the wheel is built from that sdist,
    both from ``git archive HEAD`` so that no untracked or generated file can
    leak in.
12. ``twine check --strict``, and an audit of the wheel (only ``dualmesh/``
    and its ``.dist-info``, with the compiled extension) and of the sdist (no
    build products).
13. The wheel is installed with its ``all``, ``test`` and ``docs`` extras into
    a fresh virtual environment and the whole Python suite runs against it.
14. The documentation is built strictly:
    ``python -m sphinx -W --keep-going -b html docs docs/_build/html``.
15. The wheel alone is installed into another fresh environment, and
    ``import dualmesh``, ``importlib.metadata.version("dualmesh-multiphysics")``
    and ``dualmesh --version`` all report the version.
16. ``main`` is pushed, fast-forward only.
17. The CI workflow passes on the pushed commit.
18. The Wheels workflow is run by hand from ``main`` (the dry run: with the
    GitHub CLI the script starts it, otherwise Actions → Wheels → Run
    workflow → main), and every wheel job and the sdist pass on the same
    commit.
19. The Publish to PyPI job of that run is *skipped*.
20. Only then is the annotated tag created (``git tag -a vX.Y.Z -m "dualmesh
    vX.Y.Z"``), checked to point to ``HEAD``, and pushed alone.
21. The tag-triggered Wheels run is followed; its first job checks that the
    tag matches the package version.
22. The Publish to PyPI job must succeed (Trusted Publishing, with
    attestations).
23. PyPI must list the sdist and a wheel for each of CPython 3.9-3.13 on Linux
    x86_64, macOS x86_64, macOS arm64 and Windows x86_64, and ``pip install
    dualmesh-multiphysics==X.Y.Z`` in a fresh environment must import and run.
24. The temporary environment is deleted, and its deletion is checked.

What not to do, whatever a failure suggests:

* never ``git push --force`` main, and never resolve a divergence with a
  global ``--ours`` or ``--theirs``.  When ``git push`` is rejected with
  "fetch first" (for example after a documentation edit made in the GitHub web
  interface), run ``git fetch origin`` and ``git rebase origin/main``, resolve
  file by file, rerun the checks and push normally;
* never tag a dirty tree, a commit that is not on ``origin/main``, or a commit
  whose Wheels dry run has not passed;
* never upload with ``twine upload`` or a stored PyPI token: publication is by
  Trusted Publishing from ``wheels.yml`` in the ``pypi`` environment;
* never delete and re-push the tag of a version that reached PyPI (PyPI does
  not accept a file twice); fix forward with a new patch version;
* do not rename the repository, the Read the Docs project, the import or the
  command, and do not change the distribution name back to ``dualmesh``;
* do not remove the ``:class: this-will-duplicate-information-and-it-is-still-useful-here``
  option of the inline ``.. contents::`` blocks: Furo shows an error-style
  warning for an inline table of contents without it, and the inline tables
  are kept on purpose.
