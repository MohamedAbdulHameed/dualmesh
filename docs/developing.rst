Developing
==========

Layout
------

.. code-block:: text

   include/dualmesh/      public C++ headers
     core/                ADReal, InputParameters, Function, Types
     mesh/                Mesh (nodes, elements, side sets, generators)
     fe/                  ReferenceElement (dual mesh geometry), Assembly
     base/                Object, Kernel, Property, Factory, Problem
     modules/             the physics modules' public headers
   src/                   the implementations, mirroring include/
   python/bindings/       the pybind11 module
   python/dualmesh/       the Python package
   tests/cpp/             C++ unit tests (self-contained, no framework needed)
   tests/python/          verification against the book
   verification/openfoam/ cross-verification scripts
   examples/              runnable examples
   docs/                  this documentation

One input design for every capability
-------------------------------------

Every capability of dualmesh is used from Python through one input design.  A
user describes a model in the terms of its physics, and the code generates
the equations.  A capability belongs to one of three levels, and each level
generates the level below without changing its meaning.

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Level
     - What the user adds
     - Example
   * - Physics
     - A set of equations on any mesh, with its boundary conditions, and the
       couplings between physics.
     - ``problem.add_physics("solid_mechanics", "solid", ...)``,
       ``problem.add_coupling("thermal_expansion", ...)``
   * - Object
     - One term (kernel), property object or condition.
     - ``problem.add_kernel("heat_source", ...)``

A study of one model is a method of the model (``problem.solve()``,
``problem.solve_transient(...)``).  A study of many models
(uncertainty quantification, adaptive refinement, convergence studies) takes a
function that builds and solves one model.

Adding a capability
~~~~~~~~~~~~~~~~~~~

A new capability follows these rules, so that it fits the design without a
special case.

1. **Level.**  One term or condition is an object, a set of equations is a
   physics, and the interaction of two physics is a coupling.  A new physics
   generates objects.
2. **Names.**  Type names, parameters and keywords are full words in lower
   case joined by underscores.  Proper names and abbreviations keep their
   capitals (``Dirichlet_boundary_condition``, ``beam_Timoshenko_mixed``,
   ``Robin_boundary_condition``), and the very common abbreviations ``num``,
   ``max`` and ``min`` are used.  The same quantity has the same name in
   every module (e.g., ``poissons_ratio``, ``time_step``, ``implicitness``,
   ``standard_deviation``).  A Python class follows PEP 8 (``SolveResult``).
3. **Declaration.**  Every parameter is declared once, with its type, its SI
   unit, its default with the reason for the default, and a description of
   its physical meaning.  C++ objects use ``InputParameters``, and Python
   physics, couplings and input groups use
   :func:`dualmesh.parameters.parameter`.  The same declaration validates the
   input, answers ``dualmesh describe`` and generates the reference pages.
   ``scripts/audit_names.py`` checks every public name and declaration
   against rules 2 and 3 (forbidden abbreviations, capitals outside the
   proper names, a missing description, a default without its reason, and a
   semicolon, a dash or a Unicode symbol in a description), and the test
   suite runs it.
4. **One way to define each thing.**  Some pairs of inputs define the same
   thing, for example ``traction`` and ``total_force``.  The code stops with
   an error when the input gives the two inputs of such a pair.  The
   documentation tells the user to give only one of them.
5. **Validation.**  A misspelled name is answered with the closest valid name
   and the list of the valid ones, a missing required parameter is named with
   its description, and a value outside its physical range is refused with a
   message that says what to do.
6. **Physics.**  A physics declares its variables, fills in the variables,
   components and thicknesses of its objects, and passes every other
   parameter to them unchanged.  A property that is not given is read from
   the property of the same name.  A new physics is a dataclass
   subclass of :class:`dualmesh.physics.Physics` registered with
   ``@register``, with the methods ``variables`` and ``_build``.
7. **Results.**  A result provides ``summary()``, ``to_dict()``,
   ``write_json(path)`` and ``write_csv(path)``.
8. **Python style.**  One statement on one line, however long (ruff with a
   line length of 320), and comments on their own line.
9. **Accuracy and speed.**  A new capability comes with a test against the
   equivalent object-level input where one exists, a verification case
   against a published or analytical result, an example script, a tutorial
   section and its reference page.  A change to assembly or to a solver is
   timed before and after.

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

and it becomes available by name from Python and in ``dualmesh list``.  Registration is made by an explicit call so that no object
is dropped when the library is linked statically, as can happen with
registration by static initialization.

Writing kernels correctly
-------------------------

* Use ``ctx.value(v)`` and ``ctx.gradient(v)`` for the quantities that carry
  derivatives, and ``ctx.coefficient_value(v)`` /
  ``ctx.coefficient_gradient(v)`` inside nonlinear *coefficients*.  The latter
  return the current iterate under Newton's method and the previous iterate
  under direct iteration, so that one kernel serves both schemes.
* Never compare AD numbers to decide a branch that depends on the unknown
  unless the derivative of the branch is intended.
* Properties are requested by name with
  ``problem.propertyRegistry().id("stress")`` in ``initialSetup`` and read with
  ``ctx.property(id, component)``.  List them in ``requiredProperties()`` so
  that a missing property is reported before the solve starts.

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


   problem.add_kernel(ArrheniusSource(variable="temperature", temperature_variable="temperature", pre_exponential=1.0e6, activation_energy=5000.0))

Testing
-------

* ``tests/cpp/unit_tests.cpp`` holds the fast structural tests: the automatic
  differentiation, the quadrature rules, the geometric closure of the dual
  mesh, and the patch test.
* ``tests/python`` holds the verification suite.  Every value in it comes from
  a published table, and a new case must cite its table in a comment.
* Run both with ``ctest --test-dir build`` and ``pytest``.

Style
-----

C++ follows the MOOSE conventions (two-space indentation, ``_member``
variables, ``camelCase`` functions, ``PascalCase`` types), and Python follows
PEP 8 with descriptive, unabbreviated parameter names.  ``.clang-format`` and
the ``ruff`` configuration in ``pyproject.toml`` encode both.
Formatting is checked in continuous integration by clang-format **18** (the
version on the ``ubuntu-latest`` runner).  Other major versions format some
constructs differently, so format with version 18 (``brew install llvm@18`` on
macOS, ``apt install clang-format-18`` on Ubuntu).  The check applied in
continuous integration is

.. code-block:: console

   find include src python/bindings tests/cpp \( -name '*.h' -o -name '*.cpp' \) \
     -print0 | xargs -0 clang-format --dry-run --Werror

No output means success.

Making a release
----------------

The package is published on PyPI as ``dualmesh-multiphysics``.  It is imported
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

1. ``origin`` is the SSH remote of the GitHub repository
   ``MohamedAbdulHameed/dualmesh``.
2. The branch is ``main``.
3. ``git fetch`` and ``git pull --ff-only origin main`` succeed: the branch is
   not behind ``origin/main`` and has not diverged from it.
4. The working tree is clean and no tracked file contains a genuine merge
   conflict.  ``git diff --check`` is unsuitable for this, because it reports
   every reStructuredText title underline of seven ``=`` (``Solving`` over
   ``=======``) as a leftover conflict marker.  The gate counts a line of ``=``
   as a marker only between ``<<<<<<<`` and ``>>>>>>>``.
5. ``pyproject.toml`` names the distribution ``dualmesh-multiphysics``, the
   command ``dualmesh = "dualmesh.cli:main"`` and the documentation URL.
6. The version is the same in ``pyproject.toml``, ``python/dualmesh/__init__.py``
   and ``CITATION.cff`` (the documentation reads it from ``pyproject.toml``),
   and ``CHANGELOG.md`` has a ``## [X.Y.Z] - YYYY-MM-DD`` section.
7. The tag ``vX.Y.Z`` exists neither locally nor on ``origin``.

   A further static gate checks the release configuration: the wheel matrix
   ``[ubuntu-latest, macos-15-intel, macos-14, windows-latest]`` (which
   excludes the retired macOS 13 runner), cibuildwheel v4.2.1 or later, the
   ``workflow_dispatch`` trigger, the tag guard and the ``pypi`` environment
   of the publish job, Trusted Publishing without a stored token,
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
18. The Wheels workflow is run by hand from ``main`` as a dry run, and every
    wheel job and the sdist pass on the same commit.  The script starts the
    run with the GitHub CLI when it is installed.  Otherwise the run is started
    in the GitHub web interface by selecting Actions, Wheels, Run workflow and
    the branch ``main``.
19. The Publish to PyPI job of that run is *skipped*.
20. Only then is the annotated tag created (``git tag -a vX.Y.Z -m "dualmesh
    vX.Y.Z"``), checked to point to ``HEAD``, and pushed alone.
21. The tag-triggered Wheels run is followed.  Its first job checks that the
    tag matches the package version.
22. The Publish to PyPI job must succeed (Trusted Publishing, with
    attestations).
23. PyPI must list the sdist and a wheel for each of CPython 3.9-3.13 on Linux
    x86_64, macOS x86_64, macOS arm64 and Windows x86_64, and ``pip install
    dualmesh-multiphysics==X.Y.Z`` in a fresh environment must import and run.
24. The temporary environment is deleted, and its deletion is checked.

The following actions are prohibited, whatever a failure suggests:

* Never ``git push --force`` main, and never resolve a divergence with a
  global ``--ours`` or ``--theirs``.  When ``git push`` is rejected with
  "fetch first" (for example after a documentation edit made in the GitHub web
  interface), run ``git fetch origin`` and ``git rebase origin/main``, resolve
  file by file, rerun the checks and push normally.
* Never tag a dirty tree, a commit that is not on ``origin/main``, or a commit
  whose Wheels dry run has not passed.
* Never upload with ``twine upload`` or a stored PyPI token.  Publication is by
  Trusted Publishing from ``wheels.yml`` in the ``pypi`` environment.
* Never delete and re-push the tag of a version that reached PyPI, because
  PyPI does not accept a file twice.  Release the correction as a new patch
  version.
* Do not rename the repository, the Read the Docs project, the import or the
  command, and do not change the distribution name back to ``dualmesh``.
* Do not remove the ``:class: this-will-duplicate-information-and-it-is-still-useful-here``
  option of the inline ``.. contents::`` blocks: Furo shows an error-style
  warning for an inline table of contents without it, and the inline tables
  are kept on purpose.
