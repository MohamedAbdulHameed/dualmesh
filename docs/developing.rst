Developing
==========

This page is for contributors.  It describes the layout of the repository, the
levels of the code, the rules that a new capability follows, how to write a
kernel, and the commands that test, check and release the code.  Build the code
in the source tree first, as :doc:`installation` describes.

Layout
------

.. code-block:: text

   include/dualmesh/      the public C++ headers
     base/                Object, Kernel, Property, Factory, Problem, QpContext, Checkpoint,
                          and FlowSystem and SolidSystem, the hooks that modules implement
     core/                ADReal, InputParameters, Function, ParsedFunction, Types, and
                          Coordinates, the measure and hoop terms of each coordinate system
     fe/                  Assembly: the quadrature points and the map of each element
     linalg/              the PETSc solvers, their choice and the MPI communicator
     mesh/                Mesh, ReferenceElement (with the dual mesh geometry), the bisection,
                          the subdivision and the hierarchy of a mesh
     modules/             the headers of the physics modules, and Registry.h, whose
                          registerAllObjects() registers their objects
     parallel/            the distributed problem, its partition and its gather-scatter
   src/                   the implementations, in the folders of include/
   python/bindings/       the pybind11 module, one file per subsystem: conversions,
                          functions, mesh, objects, problem and parallel
   python/dualmesh/       the Python package
     units.py             the units, the dimensions and the standard names
     model/               the model language: its syntax, its expression graph, its calculus,
                          the conservation laws and their engine form
     physics/             the physics level: heat, solid, fluid, general_form, couplings and
                          base, the base classes and the registry
     engine.py            Problem, the object level
     input/               the input: its schema and its build
     case.py              problem, the input in Python
     solution.py          results, the result of the input
   tests/cpp/             the C++ tests
   tests/python/          the Python tests
   verification/          the verification and benchmark scripts
   examples/              the example scripts and case files
   scripts/               the checks, the benchmarks and the release script
   docs/                  this documentation

The C++ directories include in one direction.  Each of ``core``, ``mesh``,
``fe``, ``linalg`` and ``base`` includes only directories that come before it in
this list.  ``modules`` and ``parallel`` sit on top: they include ``base`` and
the levels below it, and not each other.  When ``base`` needs work that a module does, it calls a virtual function that the
module implements: a virtual of ``Kernel`` or of one of its interfaces, such as ``MassEquation::flowCoefficients``, or of
the systems ``FlowSystem`` and ``SolidSystem`` that a module registers in the
``Factory``.

The modules of the Python package import in one direction, from the top of each file.
``_core``, the compiled module, and ``units`` import no other module of the package.
The package ``model`` imports these two and ``parameters``.  The package ``physics``
imports ``model``, and ``engine`` imports ``physics``.  The package ``input`` and
the module ``case``, which holds :class:`dualmesh.problem`, import all of them.  When a
module needs code from a module above it, move that code down a level instead of
importing inside a function.

The levels of the code
----------------------

A problem is built at one of three levels, and each level builds the level
below it without changing its meaning.

.. list-table::
   :header-rows: 1
   :widths: 14 40 46

   * - Level
     - What you give
     - Example
   * - Input
     - The sections of a problem, in physical terms and with units.  The input
       lowers them onto the physics level.
     - ``dualmesh.problem(mesh=...)``, ``problem.physics("heat_conduction")``,
       or a case file run by ``dualmesh run``
   * - Physics
     - A set of equations with its boundary conditions, and the couplings
       between two physics.  A physics generates objects.
     - ``problem.add_physics("solid_mechanics", ...)``,
       ``problem.add_coupling("thermal_expansion", ...)``
   * - Object
     - One term (a kernel), a property object or a condition.
     - ``problem.add_kernel("body_force", ...)``

A study of one model is a method of the model, such as
:meth:`dualmesh.Problem.solve` or :meth:`dualmesh.Problem.solve_transient`.  A
study of many models, such as a convergence study, takes a function that builds
and solves one model.

Adding a capability
-------------------

A new capability follows these rules, so that it fits the levels without a
special case.

1. **Level.**  One term or condition is an object, a set of equations is a
   physics, and the interaction of two physics is a coupling.  A new physics
   generates objects.
2. **Names.**  Type names, parameters and keywords are full words in lower
   case joined by underscores.  Proper names and abbreviations keep their
   capitals, as in ``Dirichlet``, ``Robin`` and ``beam_Timoshenko_mixed``.  The
   common abbreviations ``num``, ``max`` and ``min`` are allowed.  The same
   quantity has the same name in every module, for example ``poissons_ratio``,
   ``time_step``, ``time_scheme`` and ``standard_deviation``.  A Python class
   follows PEP 8, as ``SolveResult`` does.
3. **Declaration.**  Every parameter is declared once, with its type, its unit
   in the International System of Units (SI), its default with the reason for the default, and a description of
   its physical meaning.  C++ objects declare their parameters with
   ``InputParameters``, and Python physics, couplings and input groups with
   :func:`dualmesh.parameters.parameter`.  The same declaration validates the
   input, answers ``dualmesh describe`` and generates the reference pages.
4. **One way to define each thing.**  Some pairs of inputs define the same
   thing, for example ``traction`` and ``total_force``.  The code stops with an
   error when the input gives both inputs of such a pair, and the
   documentation tells you to give one of them.
5. **Validation.**  A misspelled name is answered with the closest valid name
   and the list of the valid ones.  A missing required parameter is named with
   its description.  A value outside its physical range is refused with a
   message that says what to do.
6. **Physics.**  A physics declares its variables, fills in the variables,
   components and thicknesses of its objects, and passes every other parameter
   to them unchanged.  A property that you don't give is read from the property
   of the same name.  A new physics is a dataclass subclass of
   :class:`dualmesh.physics.Physics`, registered with ``@register``, with the
   methods ``variable_names`` and ``_build``.
7. **Results.**  A result provides ``summary()``, ``to_dict()``,
   ``write_json(path)`` and ``write_csv(path)``.
8. **Python style.**  The code follows ``ruff format`` with a line length of
   100 characters.  A call with many inputs is split into named pieces, and a
   comment has its own line.
9. **Accuracy and speed.**  A new capability comes with a test against the
   equivalent object-level input where one exists, a verification case against
   a published or analytical result, an example script, a tutorial section and
   its reference page.  A change to the assembly or to a solver is timed before
   and after, as `Check the performance`_ describes.

``scripts/audit_names.py`` checks every public name and declaration against
rules 2 and 3.  It reports a forbidden abbreviation, a capital outside the
proper names, a missing description, a default without its reason, and a
semicolon, a dash or a Unicode symbol in a description.  The Documentation job
of the continuous integration (CI) workflow runs it.

Add a kernel in C++
-------------------

The kernels of an equation together make its residual

.. math::

   R = -\nabla \cdot \mathbf{G} + S = 0,

where ``computeFlux`` of each kernel adds to the flux :math:`\mathbf{G}` and
``computeSource`` adds to the source :math:`S`.  In the balance law
:math:`\partial m/\partial t + \nabla \cdot \mathbf{F} = s` of the theory
pages, with :math:`m` the stored quantity per volume, :math:`\mathbf{F}` its
physical flux and :math:`s` its source per volume, the kernels give
:math:`\mathbf{G} = -\mathbf{F}` and :math:`S = \partial m/\partial t - s`.
For heat conduction, :math:`\mathbf{G} = k \nabla T`, with :math:`k` the
thermal conductivity in :math:`\mathrm{W/(m\,K)}` and :math:`T` the temperature
in :math:`\mathrm{K}`.

A kernel declares its parameters in ``validParams`` and overrides the terms it
supplies:

.. code-block:: c++

   class MyReaction : public Kernel
   {
   public:
     static InputParameters validParams()
     {
       InputParameters p = Kernel::validParams();
       p.setClassDescription("Reaction term c u^2.");
       p.addOptional("coefficient", ParameterKind::Real, 1.0, "The coefficient c. Default 1.");
       return p;
     }
     explicit MyReaction(const InputParameters & p) : Kernel(p), _c(p.getReal("coefficient")) {}

     bool hasSource() const override { return true; }
     ADReal computeSource(const QpContext & ctx) const override
     {
       const ADReal & u = ctx.value(_var);
       return _c * u * u;
     }

   private:
     double _c;
   };

``Kernel`` holds the hooks that every kernel needs.  A kernel of one family
derives from the interface of that family instead, which adds its hooks:

* ``StorageKernel``, a time derivative of the variable.
* ``MassEquation``, the mass equation of an incompressible flow, which gives
  the block preconditioners their coefficients and the advection the velocity
  that it conserves.
* ``AdvectiveKernel``, a term that carries the variable with a velocity.
* ``SectionKernel``, in the solid mechanics module, a beam or a plate whose
  stress varies over the height of its section.

Register the kernel in the registration function of its module, for example
``registerHeatTransferObjects`` in ``src/modules/HeatTransfer.cpp``:

.. code-block:: c++

   f.add<MyReaction>("my_reaction", ObjectCategory::Kernel, m);

The kernel is then available by its name from Python and in ``dualmesh list``.
Each module registers its objects by an explicit call, so a static link of the
library keeps every object.  ``registerAllObjects`` in
``include/dualmesh/modules/Registry.h`` calls the registration function of every
module.  The Python module calls it when it is imported.  A C++ program calls
it once before it makes a ``Problem``, as the C++ tests do in ``main``.

Write a kernel correctly
~~~~~~~~~~~~~~~~~~~~~~~~

* Read the quantities that carry derivatives with ``ctx.value(v)`` and
  ``ctx.gradient(v)``.  Inside a nonlinear coefficient, read them with
  ``ctx.coefficientValue(v)`` and ``ctx.coefficientGradient(v)``, which give
  the current iterate under Newton's method and the previous iterate under
  direct iteration.  One kernel then serves both schemes.
* Decide a branch by comparing automatic differentiation numbers only when you
  intend the derivative of that branch.
* Request a property by name with ``problem.propertyRegistry().id("stress")``
  in ``initialSetup``, and read it with ``ctx.property(id, component)``.  List
  it in ``requiredProperties()``, so that the setup reports a missing property.

Add a kernel in Python
----------------------

A subclass of :class:`dualmesh.PythonKernel` has the interface of a C++ kernel,
with exact derivatives.  It overrides ``compute_flux``, ``compute_source`` or
both, and it reads its keyword arguments as attributes.  This kernel gives the flux of a bar whose stiffness depends on the strain,
:math:`a(u')\, u'` with :math:`a(u') = EA (1 + 1.5 u' + 0.5 u'^2)`, where
:math:`u'` is the strain and :math:`EA` the axial stiffness.  It reads the
strain of the coefficient with ``ctx.coefficient_gradient``, as `Write a
kernel correctly`_ describes for C++:

.. code-block:: python

   import dualmesh


   class large_deformation_bar(dualmesh.PythonKernel):
       """Flux a(u') du/dx with a(u') = EA (1 + 1.5 u' + 0.5 u'^2)."""

       def compute_flux(self, ctx):
           strain = ctx.coefficient_gradient(self.variable)[0]
           stiffness = self.axial_stiffness * (1.0 + 1.5 * strain + 0.5 * strain * strain)
           return [stiffness * ctx.gradient(self.variable)[0]]


   mesh = dualmesh.generate_line_mesh(start=0.0, end=1.0, num_elements=8)
   problem = dualmesh.Problem(mesh)
   problem.add_variable("displacement")
   problem.add_kernel(large_deformation_bar(variable="displacement", axial_stiffness=1.0))
   problem.add_boundary_condition("Dirichlet", "left", variable="displacement", value=0.0)
   problem.add_boundary_condition("Neumann", "right", variable="displacement", flux=0.2)
   problem.solve(report="none")

``tests/python/test_heat_transfer.py`` solves this bar and checks the tip
displacement against Example 6.2.4 of [Reddy2024]_.
``examples/large_deformation_bar.py`` solves the same bar as a model of the
input, with no kernel.

Test the code
-------------

The C++ tests are in ``tests/cpp``.  ``unit_tests.cpp`` checks the automatic
differentiation, the quadrature rules, the dual mesh of each element, the shape
functions, the patch test and the mesh refinement.  ``parallel_tests.cpp``
checks the partitioners and the distributed solver.  The Python tests are in
``tests/python``.

1. Run the C++ tests from the ``build`` folder of the source tree:

   .. code-block:: console

      ctest --test-dir build --output-on-failure

2. Run the distributed C++ tests on 1 to 4 processes.  ``--oversubscribe`` is
   an option of Open MPI that allows more processes than cores:

   .. code-block:: console

      for n in 1 2 3 4
      do
        mpirun --oversubscribe -n $n ./build/dualmesh_parallel_tests
      done

3. Run the Python tests:

   .. code-block:: console

      pytest

4. Run the distributed Python tests on 1 to 4 processes:

   .. code-block:: console

      for n in 1 2 3 4
      do
        mpirun --oversubscribe -n $n python -m pytest -q -p no:cacheprovider tests/python/test_parallel.py
      done

A new test compares a computed value with an exact solution, a manufactured
solution or a published value, and its docstring names the source.

Check names, style and documentation
------------------------------------

* Audit the public names and declarations:

  .. code-block:: console

     python scripts/audit_names.py

* Check the naming rule and the comment rule in the sources.  Every comment
  starts with a capital letter, unless it starts with an identifier written as
  code:

  .. code-block:: console

     python scripts/check_names.py

* Check the Python code with ruff 0.16.10:

  .. code-block:: console

     pip install ruff==0.16.10
     ruff check python tests examples scripts
     ruff format --check python tests examples scripts

* Check the C++ format with clang-format 18.  The C++ code uses ``_member``
  variables, ``camelCase`` functions and ``PascalCase`` types, and
  ``.clang-format`` sets the layout.  The command prints nothing when every
  file is formatted:

  .. code-block:: console

     pip install clang-format==18.1.8
     find include src python/bindings tests/cpp -name '*.h' -o -name '*.cpp' | xargs clang-format --dry-run --Werror

* Build the documentation with every warning as an error:

  .. code-block:: console

     pip install ".[docs,all]"
     python -m sphinx -W --keep-going -b html docs docs/_build/html

  Each ``.. contents::`` directive keeps the option
  ``:class: this-will-duplicate-information-and-it-is-still-useful-here``,
  which the release script checks.

Check the performance
---------------------

``scripts/benchmark.py`` times one residual and Jacobian assembly of each
method for heat conduction, plane elasticity and pressure-velocity flow, and
``scripts/check_performance.py`` compares the times with their budgets:

.. code-block:: console

   python scripts/benchmark.py --quick --only assembly --repeat 5 --output results.json
   python scripts/check_performance.py results.json

The workflows
-------------

Every workflow starts by hand: CI, CodeQL, Benchmarks, FEniCSx, Mutation
tests, Performance and Wheels.  Wheels also starts on a pushed tag ``v*``.  No
workflow starts on a push, a pull request or a schedule.  The CI workflow runs
the C++ tests, the distributed tests, the Python tests, the examples and the
command line, the lint, the documentation, the sanitizers, the coverage and
clang-tidy.

To start a workflow on a branch:

1. Open the Actions tab of the repository on GitHub.
2. Pick the workflow.
3. Click Run workflow and choose the branch.

The GitHub command-line program starts a workflow by its file:

.. code-block:: console

   gh workflow run ci.yml --ref <branch>

Make a release
--------------

The distribution is named ``dualmesh-multiphysics``, its import name is
``dualmesh`` and its command is ``dualmesh``.  It is not published on the
Python Package Index (PyPI).  The wheels and the source distribution of a
release are the artifacts of the Wheels workflow that the tag of the release
starts.

``scripts/release.py`` runs each step of a release as a gate and stops at the
first gate that fails.  It uses only the Python standard library.  Run the
steps in order, from ``main``:

.. code-block:: console

   python scripts/release.py check --version X.Y.Z
   python scripts/release.py push --version X.Y.Z
   python scripts/release.py ci --wait
   python scripts/release.py dry-run --wait
   python scripts/release.py tag --version X.Y.Z
   python scripts/release.py monitor --version X.Y.Z --wait

``python scripts/release.py release --version X.Y.Z`` runs all six in order.
``python scripts/release.py check --dev`` runs the local gates at any time,
without a new version, and ``python scripts/release.py status`` prints the
state of the release without changing it.


The gates of ``check`` are:

1. The remote ``origin`` is the SSH remote that the script expects.
2. The branch is ``main``.
3. ``git pull --ff-only origin main`` succeeds, and the branch is not behind
   ``origin/main``.
4. The working tree is clean, and no tracked file holds a merge conflict.  A
   line of seven ``=`` counts as a conflict marker only between ``<<<<<<<`` and
   ``>>>>>>>``, so the title underlines of reStructuredText pass.
5. ``pyproject.toml`` names the distribution, maps the command ``dualmesh`` to
   ``dualmesh.cli:main`` and gives the project URLs that the script expects.
6. The version is the same in ``pyproject.toml``,
   ``python/dualmesh/__init__.py`` and ``CITATION.cff``, and ``CHANGELOG.md``
   has a section ``## [X.Y.Z] - YYYY-MM-DD``.
7. The tag ``vX.Y.Z`` exists neither locally nor on ``origin``.  A static check
   of the workflows follows: the wheel matrix, cibuildwheel v4.2.1 or newer,
   every action pinned to a commit, the manual and the tag triggers of the
   Wheels workflow, no upload to PyPI, and the class of every
   ``.. contents::`` directive.
8. clang-format 18 with ``--dry-run --Werror`` passes on every C++ file.
9. The C++ library configures, builds and passes CTest.  ``--skip-cxx`` skips
   this gate.
10. ``ruff check`` and ``ruff format --check`` pass.
11. The source distribution builds in isolation from ``git archive HEAD``, and
    the wheel builds from that source distribution.
12. ``twine check --strict`` passes.  The wheel holds only ``dualmesh/``, with
    its compiled extension, and its ``.dist-info``, and the source distribution
    holds no build products.
13. The wheel, installed with its ``all``, ``test`` and ``docs`` extras in a new
    virtual environment, passes the Python tests.
14. The documentation builds with ``-W --keep-going``.
15. The wheel alone, installed in another new virtual environment, reports the
    version through ``import dualmesh``, its metadata and ``dualmesh --version``.

The later commands run the remaining gates:

16. ``push`` pushes ``main``, fast-forward only.
17. ``ci`` runs the CI workflow by hand from ``main``, and waits for it to pass
    on the pushed commit.
18. ``dry-run`` runs the Wheels workflow by hand from ``main``, and every wheel
    and the source distribution must build and pass on that commit.  Without
    the GitHub command-line program, start the run in the GitHub web interface
    under Actions, Wheels, Run workflow, on the branch ``main``.
19. ``tag`` checks the gates again, creates the annotated tag ``vX.Y.Z`` on
    ``HEAD`` and pushes the tag alone.
20. ``monitor`` follows the Wheels workflow of the tag.  Its source
    distribution job checks that the tag matches the package version.

Never force-push ``main``, and never resolve a divergence with a global
``--ours`` or ``--theirs``.  When ``git push`` is rejected, run
``git fetch origin`` and ``git rebase origin/main``, resolve each file, run the
checks again and push.  Never tag a dirty tree, a commit that is not on
``origin/main``, or a commit whose Wheels dry run has not passed.  Never delete
and push again the tag of a released version: release the correction as a new
patch version.
