// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Objects defined by expressions: parsed, which declares a scalar property, table, which
// declares a scalar property linear between the rows of a table of one argument expression,
// parsed_eigenstrain, which declares an eigenstrain for the stress property objects,
// parsed_tensor, which declares a property of several components from expressions of the fields
// and their gradients, and parsed_kernel, a term of an equation.  The expressions are compiled
// once (FieldExpressions) and evaluated with automatic differentiation, so a property that
// depends on the solution contributes its exact derivatives to the Jacobian.  They are the way to
// add a correlation (a conductivity, a swelling law) without writing C++, at the speed of
// compiled code.
#include "dualmesh/base/Factory.h"
#include "dualmesh/base/FieldExpression.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Problem.h"
#include "dualmesh/base/Property.h"

#include <algorithm>
#include <array>
#include <map>
#include <sstream>

namespace dualmesh
{

namespace
{

/// The parameter of the named constants of an object defined by expressions.
void
addConstants(InputParameters & p)
{
  p.addOptional("constants",
                ParameterKind::RealMap,
                RealMap{},
                "Named constants that the expressions use, as name -> value pairs. Default: "
                "none, for expressions that need no constant of their own.");
}

/// The named constants of @p p, as names in scope of FieldExpressions.
FieldExpressions::Names
constantNames(const InputParameters & p)
{
  FieldExpressions::Names names;
  for (const auto & [n, value] : p.getRealMap("constants"))
    names.constants.emplace_back(n, value);
  return names;
}

const char * kExpressionHelp =
    "The expression language: numbers, the arguments by name, the constants pi and e, "
    "+ - * / and ^ (or **), comparisons (< > <= >= == !=, which give 1 or 0), "
    "if(condition, a, b), and the functions sin cos tan asin acos atan sinh cosh tanh exp "
    "log log10 sqrt abs floor ceil erf sign atan2 pow hypot min max.";

} // namespace

// ---------------------------------------------------------------------------
/// A scalar property given by an expression.
class ParsedProperty : public Property
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Property::validParams();
    p.setClassDescription(
        std::string(
            "A scalar property given by an expression of the fields, the components of their "
            "gradients as grad_x(u), grad_y(u) and grad_z(u), the coordinates x, y and z, the "
            "time t, the registered functions, the element fields, the scalar properties of the "
            "property objects added before it and named constants, each used by its name. The "
            "code compiles the expression once and differentiates it automatically. It adds a "
            "correlation, for example a thermal conductivity k(T, x), without C++ code, and a "
            "property that varies through the body only, such as a graded conductivity '20 + "
            "5*x' or the name of a registered function of x, y, z and t. The "
            "fields take their values at the current iterate, with their derivatives, under "
            "Newton's method and Picard iteration alike, as in every expression of the code. ") +
        kExpressionHelp);
    p.addRequired("property_name", ParameterKind::String, "Name of the declared property.");
    p.addRequired("expression",
                  ParameterKind::String,
                  "The expression of the property, in the units that the objects reading it "
                  "expect.");
    addConstants(p);
    return p;
  }
  explicit ParsedProperty(const InputParameters & p) : Property(p) {}
  void initialSetup(Problem & problem) override
  {
    Property::initialSetup(problem);
    _expression.compile(problem, {_params.getString("expression")}, constantNames(_params), name());
  }
  void declareProperties(PropertyRegistry & r) override
  {
    _prop = r.declare(_params.getString("property_name"), 1);
  }
  void computeProperties(QpContext & ctx) const override
  {
    ctx.property(_prop) = _expression.value(0, ctx);
  }

private:
  FieldExpressions _expression;
  int _prop = -1;
};

// ---------------------------------------------------------------------------
/// A scalar property given by a table of one argument, linear between its rows.
class TableProperty : public Property
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Property::validParams();
    p.setClassDescription(
        "A scalar property given by a table of one argument, linear between its rows. The "
        "argument is an expression of the fields, the coordinates x, y and z and the time t, "
        "such as temperature. The property takes the derivatives of the argument, so a table of "
        "a field contributes its exact derivative to the Jacobian. A state whose argument lies "
        "outside the range of the table is inadmissible: the solve rejects it and tries a "
        "smaller step, and stops with a message that names the property, the region and the "
        "value when no smaller step keeps the argument in the range.");
    p.addRequired("property_name", ParameterKind::String, "Name of the declared property.");
    p.addRequired("argument",
                  ParameterKind::String,
                  "The expression of the argument of the table, such as temperature.");
    p.addRequired("points",
                  ParameterKind::RealList,
                  "The values of the argument at the rows of the table, in increasing order.");
    p.addRequired("values",
                  ParameterKind::RealList,
                  "The values of the property at the rows of the table, in the units that the "
                  "objects reading it expect.");
    return p;
  }
  explicit TableProperty(const InputParameters & p)
      : Property(p), _points(p.getRealList("points")), _values(p.getRealList("values"))
  {
    if (_points.size() < 2 || _points.size() != _values.size())
      throw InputError("'" + name() + "': a table takes at least two points and one value per " +
                       "point, and it has " + std::to_string(_points.size()) + " points and " +
                       std::to_string(_values.size()) + " values.");
    for (std::size_t k = 1; k < _points.size(); ++k)
      if (!(_points[k] > _points[k - 1]))
        throw InputError("'" + name() + "': the points of a table increase.");
  }
  void initialSetup(Problem & problem) override
  {
    Property::initialSetup(problem);
    _argument.compile(problem, {_params.getString("argument")}, {}, name());
    _block_names = problem.mesh().blockNames();
  }
  void declareProperties(PropertyRegistry & r) override
  {
    _prop = r.declare(_params.getString("property_name"), 1);
  }
  void computeProperties(QpContext & ctx) const override
  {
    const ADReal argument = _argument.value(0, ctx);
    const double a = argument.value();
    // The ends hold to round-off of the range, so that a state on an end of the table is read.
    const double tolerance = 1e-12 * (_points.back() - _points.front());
    if (!(a >= _points.front() - tolerance && a <= _points.back() + tolerance))
      throw InadmissibleStateError(outside(ctx, a));
    // The interval of the argument: the last one holds its upper end.
    const std::size_t k =
        std::upper_bound(_points.begin() + 1, _points.end() - 1, a) - _points.begin() - 1;
    const double slope = (_values[k + 1] - _values[k]) / (_points[k + 1] - _points[k]);
    ctx.property(_prop) = _values[k] + slope * (argument - _points[k]);
  }

private:
  std::string outside(const QpContext & ctx, double a) const
  {
    std::ostringstream os;
    os << "the table of the property " << _params.getString("property_name");
    const auto region = _block_names.find(ctx.block);
    if (region != _block_names.end())
      os << " in the region " << region->second;
    os << " reads " << _params.getString("argument") << " = " << a
       << ", outside the range of the table, from " << _points.front() << " to " << _points.back()
       << ".";
    return os.str();
  }

  std::vector<double> _points;
  std::vector<double> _values;
  FieldExpressions _argument;
  std::map<int, std::string> _block_names;
  int _prop = -1;
};

// ---------------------------------------------------------------------------
/// An eigenstrain given by an expression.
class ParsedEigenstrain : public Property
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Property::validParams();
    p.setClassDescription(
        std::string(
            "An eigenstrain, a strain free of stress such as swelling or a measured thermal "
            "strain curve f(T), given by an expression and stored as a property of six components "
            "for the 'eigenstrain_names' of small_strain_stress. The expression is a linear or a "
            "volumetric strain ('strain_type'), which acts equally in the three normal "
            "directions, only along the axis, or only across it ('direction'). With "
            "'stress_free_state' the code shifts the strain to zero in a given state: for a "
            "thermal strain f(T) of the temperature T, give the stress-free temperature T_sf, and "
            "the stored strain is f(T) - f(T_sf). A thermal strain given by its expansion "
            "coefficient alpha is the object thermal_expansion_eigenstrain. The expression may "
            "use the fields, the components of their gradients as grad_x(u), grad_y(u) and "
            "grad_z(u), the coordinates x, y and z, the time t, the registered functions, the "
            "element fields, the scalar properties of the property objects added before it and "
            "named constants, each by its name. The fields take their values at the current "
            "iterate, with their derivatives, under Newton's method and Picard iteration alike, "
            "as in every expression of the code. ") +
        kExpressionHelp);
    p.addRequired("expression", ParameterKind::String, "The expression of the strain.");
    p.addRequired("eigenstrain_name", ParameterKind::String, "Name of the declared property.");
    p.addRequired("strain_type",
                  ParameterKind::String,
                  "'linear': the expression is the strain along each direction it acts in. "
                  "'volumetric': the expression is the relative change of volume dV/V, and one "
                  "third of it acts along each of the three normal directions, so the direction "
                  "must then be isotropic.");
    p.addOptional("direction",
                  ParameterKind::String,
                  std::string("isotropic"),
                  "The directions the strain acts in: isotropic (the three normal directions), "
                  "axial (along the axis of the formulation only) or transverse (the two "
                  "directions normal to the axis). Default isotropic, as for the thermal "
                  "expansion of an isotropic material.");
    p.addOptional(
        "formulation",
        ParameterKind::String,
        std::string("three_dimensional"),
        "Formulation of the stress property object, which fixes the axial component: "
        "axisymmetric and axisymmetric_1d (the second, z), plane_strain (the third, out of the "
        "plane) or three_dimensional (the third, z). It matters only for an axial or a "
        "transverse direction. Default three_dimensional, whose axis is z.");
    p.addOptional("stress_free_state",
                  ParameterKind::RealMap,
                  RealMap{},
                  "The state in which the strain is zero, as name -> value pairs, with each name "
                  "an argument of the expression, for example {temperature: 293.15}. The "
                  "expression then reads no property, since a property keeps its value at the "
                  "current state: write the dependence of the property on the named arguments "
                  "in the expression. Default: none, which stores the expression as it is.");
    addConstants(p);
    return p;
  }
  explicit ParsedEigenstrain(const InputParameters & p)
      : Property(p), _scale(volumetricScale(p.getString("strain_type"))),
        _direction(directionFromName(p.getString("direction")))
  {
    if (_scale != 1.0 && _direction != Direction::Isotropic)
      throw InputError("'" + name() +
                       "': a volumetric strain acts equally in every direction. "
                       "Use strain_type = 'linear' for an axial or transverse "
                       "strain.");
    _axial = axialComponent(p.getString("formulation"));
  }
  void initialSetup(Problem & problem) override
  {
    Property::initialSetup(problem);
    _expression.compile(problem, {_params.getString("expression")}, constantNames(_params), name());
    _free_slots.clear();
    _free_values.clear();
    for (const auto & [n, value] : _params.getRealMap("stress_free_state"))
    {
      const int k = _expression.argumentIndex(n);
      if (k < 0)
        throw InputError("'" + name() + "': 'stress_free_state' names '" + n +
                         "', which the expression does not use.");
      _free_slots.push_back(k);
      _free_values.push_back(value);
    }
    // The shift evaluates the expression with the named arguments at their stress-free values.
    // A property that the expression reads is computed by its own object at the current state,
    // so f(T_sf) would take it at T and not at T_sf.
    if (!_free_slots.empty() && _expression.readsProperties())
      throw InputError("'" + name() +
                       "': with 'stress_free_state' the expression reads no property. A property "
                       "keeps its value at the current state, so the code would not subtract the "
                       "value of the expression in the stress-free state. Write the dependence of "
                       "the property on the named arguments in the expression itself.");
  }
  void declareProperties(PropertyRegistry & r) override
  {
    _prop = r.declare(_params.getString("eigenstrain_name"), 6);
  }
  void computeProperties(QpContext & ctx) const override
  {
    const ADReal e = strain(ctx);
    for (int i = 0; i < 6; ++i)
      ctx.property(_prop, i) = i < 3 && actsAlong(i) ? e : ADReal(0.0);
  }

private:
  /// The directions the strain acts in.
  enum class Direction
  {
    Isotropic,
    Axial,
    Transverse
  };

  /// The factor on the expression of the strain_type @p type: 1/3 of a volumetric strain acts
  /// along each normal direction.
  double volumetricScale(const std::string & type) const
  {
    if (type != "linear" && type != "volumetric")
      throw InputError("'" + name() + "': strain_type must be 'linear' or 'volumetric', not '" +
                       type + "'.");
    return type == "volumetric" ? 1.0 / 3.0 : 1.0;
  }

  Direction directionFromName(const std::string & direction) const
  {
    if (direction == "isotropic")
      return Direction::Isotropic;
    if (direction == "axial")
      return Direction::Axial;
    if (direction == "transverse")
      return Direction::Transverse;
    throw InputError("'" + name() + "': direction must be isotropic, axial or transverse, not '" +
                     direction + "'.");
  }

  /// The normal component along the axis of the formulation @p formulation.
  int axialComponent(const std::string & formulation) const
  {
    if (formulation == "axisymmetric" || formulation == "axisymmetric_1d")
      return 1;
    if (formulation == "plane_strain" || formulation == "three_dimensional")
      return 2;
    throw InputError("'" + name() + "': unknown formulation '" + formulation + "'.");
  }

  /// Whether the strain acts along the normal component @p i.
  bool actsAlong(int i) const
  {
    return _direction == Direction::Isotropic || (_direction == Direction::Axial) == (i == _axial);
  }

  /// The strain at the point of @p ctx: the expression, less its value in the stress-free state,
  /// times the factor of strain_type.
  ADReal strain(const QpContext & ctx) const
  {
    std::array<ADReal, FieldExpressions::kMaxArguments> args;
    _expression.arguments(ctx, args.data());
    ADReal e = _expression.value(0, args.data());
    if (!_free_slots.empty())
    {
      for (std::size_t k = 0; k < _free_slots.size(); ++k)
        args[_free_slots[k]] = ADReal(_free_values[k]);
      e = e - _expression.value(0, args.data());
    }
    return e * _scale;
  }

  FieldExpressions _expression;
  double _scale = 1.0;
  Direction _direction = Direction::Isotropic;
  int _axial = 2, _prop = -1;
  std::vector<int> _free_slots;
  std::vector<double> _free_values;
};

// ---------------------------------------------------------------------------
/// A property of several components, each an expression of the fields, their gradients, the
/// position and the time.
class ParsedTensor : public Property
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Property::validParams();
    p.setClassDescription(
        std::string(
            "A property of several components, each given by an expression, such as the nine "
            "components of a tensor, row by row. The expressions may use every field of the "
            "problem by its name, the components of its gradient as grad_x(u), grad_y(u) and "
            "grad_z(u), the coordinates x, y and z, the time t, the registered functions, the "
            "element fields, the scalar properties of the property objects added before it and "
            "named constants, each by its name. The code compiles them once and differentiates "
            "them automatically, so an object that reads the property receives its exact "
            "derivatives. ") +
        kExpressionHelp);
    p.addRequired("property_name", ParameterKind::String, "Name of the declared property.");
    p.addRequired("components",
                  ParameterKind::StringList,
                  "The expressions of the components, one per component of the declared "
                  "property, in the order that the objects reading it expect: row by row for a "
                  "tensor.");
    addConstants(p);
    return p;
  }

  explicit ParsedTensor(const InputParameters & p)
      : Property(p), _components(p.getStringList("components"))
  {
    if (_components.empty())
      throw InputError("'" + name() + "': give one expression per component in 'components'.");
  }

  void initialSetup(Problem & problem) override
  {
    Property::initialSetup(problem);
    _expressions.compile(problem, _components, constantNames(_params), name());
  }

  void declareProperties(PropertyRegistry & r) override
  {
    _prop = r.declare(_params.getString("property_name"), static_cast<int>(_components.size()));
  }

  void computeProperties(QpContext & ctx) const override
  {
    std::array<ADReal, FieldExpressions::kMaxArguments> args;
    _expressions.arguments(ctx, args.data());
    for (std::size_t c = 0; c < _components.size(); ++c)
      ctx.property(_prop, static_cast<int>(c)) = _expressions.value(c, args.data());
  }

private:
  std::vector<std::string> _components;
  FieldExpressions _expressions;
  int _prop = -1;
};

// ---------------------------------------------------------------------------
/// A term of an equation whose flux and source are expressions of the fields, their gradients,
/// the position and the time.
class ParsedKernel : public Kernel
{
public:
  static InputParameters validParams()
  {
    InputParameters p = Kernel::validParams();
    p.setClassDescription(
        std::string("A term of the equation of 'variable', with a flux F and a source s given as "
                    "expressions of the balance law dm/dt + div F = s. Other kernels on the same "
                    "variable add their terms, such as time_derivative for the storage m. ") +
        "The expressions may use every field of the problem by its name, the components of its "
        "gradient as grad_x(u), grad_y(u) and grad_z(u), the coordinates x, y and z, the time t, "
        "the registered functions, the element fields, the properties and named constants, each "
        "by its name. " +
        "The code compiles them once and differentiates them automatically, so the Jacobian is "
        "exact and the assembly runs on every thread and every process. " +
        "In axisymmetric coordinates x is the radius and y the axial position, and the "
        "divergence includes the factor of the radius. " +
        kExpressionHelp);
    p.addOptional("flux",
                  ParameterKind::StringList,
                  std::vector<std::string>{},
                  "The flux F, one expression for each space dimension of the mesh, positive "
                  "along the outward normal of a body. For diffusion with a coefficient c, "
                  "F = -c grad u. Default: none, for a term with a source only.");
    p.addOptional("source",
                  ParameterKind::String,
                  std::string(""),
                  "The source s, an expression. Default: none, for a term with a flux only.");
    p.addOptional("advective",
                  ParameterKind::Boolean,
                  false,
                  "Whether the term carries the variable, as an advection does: a flux that does "
                  "not vanish with the gradient of the variable, or a source that reads that "
                  "gradient. The automatic quadrature of the storage of the variable "
                  "(time_derivative) then takes the Gauss rule, since a lumped storage loses phase "
                  "accuracy in a transport. Default false, a diffusion or a reaction.");
    addConstants(p);
    return p;
  }

  explicit ParsedKernel(const InputParameters & p) : Kernel(p), _advective(p.getBool("advective"))
  {
    _flux_text = p.getStringList("flux");
    _source_text = p.getString("source");
    if (_flux_text.empty() && _source_text.empty())
      throw InputError("'" + name() + "': give a 'flux', a 'source' or both.");
  }

  bool hasFlux() const override { return !_flux_text.empty(); }
  bool hasSource() const override { return !_source_text.empty(); }
  bool transportsVariable() const override { return _advective; }

  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    const int dim = problem.mesh().spaceDimension();
    if (!_flux_text.empty() && static_cast<int>(_flux_text.size()) != dim)
      throw InputError("'" + name() + "': the flux has " + std::to_string(_flux_text.size()) +
                       " component(s), and the mesh has " + std::to_string(dim) +
                       " dimension(s). Give one expression for each dimension.");
    const FieldExpressions::Names names = constantNames(_params);
    std::vector<std::string> texts = _flux_text;
    if (!_source_text.empty())
      texts.push_back(_source_text);
    _expressions.compile(problem, texts, names, name());
    _num_flux = _flux_text.size();
  }

  void computeFlux(const QpContext & ctx, ADVector3 & flux) const override
  {
    std::array<ADReal, FieldExpressions::kMaxArguments> args;
    _expressions.arguments(ctx, args.data());
    // The kernel flux F of -div F + S = 0 is -Gamma.
    for (std::size_t c = 0; c < _num_flux; ++c)
      flux[c] = -_expressions.value(c, args.data());
  }

  ADReal computeSource(const QpContext & ctx) const override
  {
    // The kernel source S of -div F + S = 0 is -f.
    return -_expressions.value(_num_flux, ctx);
  }

  void fluxTermSize(const QpContext & ctx, const ADVector3 &, double size[3]) const override
  {
    std::array<ADReal, FieldExpressions::kMaxArguments> args;
    _expressions.arguments(ctx, args.data());
    for (std::size_t c = 0; c < 3; ++c)
      size[c] = c < _num_flux ? _expressions.termSize(c, args.data()) : 0.0;
  }

  double sourceTermSize(const QpContext & ctx, const ADReal &) const override
  {
    std::array<ADReal, FieldExpressions::kMaxArguments> args;
    _expressions.arguments(ctx, args.data());
    return _expressions.termSize(_num_flux, args.data());
  }

private:
  std::vector<std::string> _flux_text;
  std::string _source_text;
  bool _advective;
  FieldExpressions _expressions;
  std::size_t _num_flux = 0;
};

void
registerParsedObjects(Factory & f)
{
  f.add<ParsedProperty>("parsed", ObjectCategory::Property, "framework");
  f.add<TableProperty>("table", ObjectCategory::Property, "framework");
  f.add<ParsedEigenstrain>("parsed_eigenstrain", ObjectCategory::Property, "solid_mechanics");
  f.add<ParsedTensor>("parsed_tensor", ObjectCategory::Property, "framework");
  f.add<ParsedKernel>("parsed_kernel", ObjectCategory::Kernel, "framework");
}

} // namespace dualmesh
