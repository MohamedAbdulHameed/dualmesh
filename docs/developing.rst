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
     f.add<MyReaction>("MyReaction", ObjectCategory::Kernel, "heat_transfer");
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
