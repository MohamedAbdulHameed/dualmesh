// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Property objects defined by expressions: parsed_property, which declares a scalar
// property, and parsed_eigenstrain, which declares an eigenstrain for the
// stress property objects.  The expressions are compiled once (ParsedExpression) and
// evaluated with automatic differentiation, so a property that depends on the
// solution contributes its exact derivatives to the Jacobian.  They are the
// way to add a correlation (a conductivity, a swelling law) without writing
// C++, at the speed of compiled code.
#include "dualmesh/base/Factory.h"
#include "dualmesh/base/FieldExpression.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Problem.h"
#include "dualmesh/base/Property.h"
#include "dualmesh/core/ParsedFunction.h"

#include <algorithm>
#include <array>
#include <memory>
#include <set>

namespace dualmesh
{

namespace
{

/// The arguments of the expressions of a property object.
/// The expressions use names without declaring them: the variables of the problem (their values
/// as coefficients, lagged under Picard iteration as in every property object), the registered
/// functions, the element fields, the scalar properties of the property objects added before this
/// one, the object's own leading arguments and its named constants.  The names are resolved when
/// the object is set up, and only those the expressions use are evaluated.
class ParsedArguments
{
public:
  static constexpr int kMaxArguments = 32;

  static void addParams(InputParameters & p)
  {
    p.addOptional("constants",
                  ParameterKind::RealMap,
                  RealMap{},
                  "Named constants that the expression uses: name -> value. Default none.");
  }

  explicit ParsedArguments(const InputParameters & p) : _constants(p.getRealMap("constants")) {}

  /// Compile @p texts, with the object's own @p leading names first in the argument array.
  void compile(Problem & problem,
               const std::vector<std::string> & texts,
               const std::vector<std::string> & leading,
               const std::string & owner)
  {
    struct Candidate
    {
      int kind; // 0 leading, 1 variable, 2 function, 3 element field, 4 property, 5 constant
      int index;
      double constant;
      std::string what;
    };
    std::vector<std::string> names;
    std::vector<std::vector<Candidate>> found;
    const auto add = [&](const std::string & n, Candidate c)
    {
      const auto it = std::find(names.begin(), names.end(), n);
      if (it == names.end())
      {
        names.push_back(n);
        found.push_back({c});
      }
      else
        found[static_cast<std::size_t>(it - names.begin())].push_back(c);
    };
    for (std::size_t k = 0; k < leading.size(); ++k)
      add(leading[k], {0, static_cast<int>(k), 0.0, "an argument of the object"});
    for (int v = 0; v < problem.numVariables(); ++v)
      add(problem.variable(v).name, {1, v, 0.0, "a field"});
    const auto functions = problem.functionNames();
    for (std::size_t k = 0; k < functions.size(); ++k)
      add(functions[k], {2, static_cast<int>(k), 0.0, "a function"});
    const auto fields = problem.elementFieldNames();
    for (std::size_t k = 0; k < fields.size(); ++k)
      add(fields[k], {3, static_cast<int>(k), 0.0, "an element field"});
    for (const auto & [n, slot] : problem.propertyRegistry().all())
      if (slot.second == 1)
        add(n, {4, slot.first, 0.0, "a property"});
    for (const auto & [n, value] : _constants)
      add(n, {5, 0, value, "a constant"});

    std::set<int> used;
    for (const auto & text : texts)
    {
      std::unique_ptr<ParsedExpression> all;
      try
      {
        all = std::make_unique<ParsedExpression>(text, names);
      }
      catch (const InputError & error)
      {
        throw InputError("'" + owner + "': " + error.what() +
                         " An expression may use the fields, the registered functions, the "
                         "element fields, the properties of the property objects added before "
                         "this one, and the named constants.");
      }
      for (std::size_t k = 0; k < names.size(); ++k)
        if (all->uses(static_cast<int>(k)))
          used.insert(static_cast<int>(k));
    }
    // The leading arguments keep their places, so that the object can fill them itself.
    for (std::size_t k = 0; k < leading.size(); ++k)
      used.insert(static_cast<int>(k));
    if (static_cast<int>(used.size()) > kMaxArguments)
      throw InputError("'" + owner + "': a parsed expression takes at most " +
                       std::to_string(kMaxArguments) + " arguments.");
    _names.clear();
    _slots.clear();
    for (int k : used)
    {
      if (found[k].size() > 1)
      {
        std::string what;
        for (std::size_t i = 0; i < found[k].size(); ++i)
          what += (i ? (i + 1 == found[k].size() ? " and " : ", ") : "") + found[k][i].what;
        throw InputError("'" + owner + "': the name '" + names[k] + "' is " + what +
                         ", so the expression is ambiguous. Rename one of them.");
      }
      const auto & c = found[k].front();
      Slot s;
      s.kind = c.kind;
      s.index = c.index;
      s.constant = c.constant;
      if (c.kind == 2)
        s.function = problem.function(functions[c.index]);
      else if (c.kind == 3)
        s.field = &problem.elementField(fields[c.index]);
      else if (c.kind == 4)
        problem.propertyRegistry().markRead(names[k]);
      _names.push_back(names[k]);
      _slots.push_back(s);
    }
  }

  /// The names in the order of the argument array, the leading names first.
  const std::vector<std::string> & names() const { return _names; }

  /// Fill @p out with every argument but the leading ones.
  template <typename T> void fill(const QpContext & ctx, T * out) const
  {
    for (std::size_t k = 0; k < _slots.size(); ++k)
    {
      const Slot & s = _slots[k];
      switch (s.kind)
      {
      case 1:
        out[k] = value<T>(ctx.coefficientValue(s.index));
        break;
      case 2:
        out[k] = T(s.function->value(ctx.x, ctx.time));
        break;
      case 3:
        out[k] = T(ctx.element >= 0 ? (*s.field)[ctx.element] : 0.0);
        break;
      case 4:
        out[k] = value<T>(ctx.property(s.index));
        break;
      case 5:
        out[k] = T(s.constant);
        break;
      default:
        break;
      }
    }
  }

private:
  template <typename T> static T value(const ADReal & a)
  {
    if constexpr (std::is_same_v<T, double>)
      return a.value();
    else
      return a;
  }

  struct Slot
  {
    int kind = 0;
    int index = 0;
    double constant = 0.0;
    FunctionPtr function;
    const std::vector<double> * field = nullptr;
  };
  RealMap _constants;
  std::vector<std::string> _names;
  std::vector<Slot> _slots;
};

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
        std::string("A scalar property given by an expression of the fields, the registered "
                    "functions, the element fields, the properties of the property objects "
                    "added before it and named constants, each used by its name, compiled once "
                    "and differentiated automatically. It is how a correlation, for instance a "
                    "thermal conductivity k(T, x), is added without writing C++. ") +
        kExpressionHelp);
    p.addRequired("property_name", ParameterKind::String, "Name of the declared property.");
    p.addRequired("expression", ParameterKind::String, "The expression of the property.");
    ParsedArguments::addParams(p);
    return p;
  }
  explicit ParsedProperty(const InputParameters & p) : Property(p), _arguments(p) {}
  void initialSetup(Problem & problem) override
  {
    Property::initialSetup(problem);
    const auto text = _params.getString("expression");
    _arguments.compile(problem, {text}, {}, name());
    _expression = std::make_unique<ParsedExpression>(text, _arguments.names());
  }
  void declareProperties(PropertyRegistry & r) override
  {
    _prop = r.declare(_params.getString("property_name"), 1);
  }
  void computeProperties(QpContext & ctx) const override
  {
    std::array<ADReal, ParsedArguments::kMaxArguments> args;
    _arguments.fill(ctx, args.data());
    ctx.property(_prop) = _expression->value(args.data());
  }

private:
  ParsedArguments _arguments;
  std::unique_ptr<ParsedExpression> _expression;
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
            "An eigenstrain (a stress-free strain, such as thermal expansion or swelling) "
            "given by an expression, stored as a six-component property for the "
            "'eigenstrain_names' of small_strain_stress. The expression is a linear strain or a "
            "volumetric strain (strain_type), applied equally in every direction or only along "
            "the axis or only across it (direction). With 'stress_free_state' the strain "
            "is shifted to be zero in a given state: for a thermal strain written as a function "
            "of temperature, give the temperature and the stress-free temperature, and the "
            "stored strain is f(T) - f(T_stress_free). ") +
        kExpressionHelp);
    p.addRequired("expression", ParameterKind::String, "The expression of the strain.");
    p.addRequired("eigenstrain_name", ParameterKind::String, "Name of the declared property.");
    p.addRequired("strain_type",
                  ParameterKind::String,
                  "'linear': the expression is the strain along each direction it acts in. "
                  "'volumetric': the expression is the relative change of volume dV/V, and "
                  "one third of it acts along each of the three directions (it must then be "
                  "isotropic).");
    p.addOptional("direction",
                  ParameterKind::String,
                  std::string("isotropic"),
                  "isotropic (the three normal directions), axial (along the axis of the "
                  "formulation only) or transverse (the two directions normal to the axis). "
                  "Default isotropic.");
    p.addOptional(
        "formulation",
        ParameterKind::String,
        std::string("three_dimensional"),
        "Formulation of the stress property object, which fixes which component is axial: "
        "axisymmetric (the second, z), axisymmetric_1d (the second, z), plane_strain "
        "(the third) or three_dimensional (the third, z). Default three_dimensional. "
        "The formulation matters only for an axial or transverse direction.");
    p.addOptional("stress_free_state",
                  ParameterKind::RealMap,
                  RealMap{},
                  "The state in which the strain is zero: name of an argument of the "
                  "expression -> its value in that state, for example {temperature: 293.15}. "
                  "Default none: the expression is used as it is.");
    ParsedArguments::addParams(p);
    return p;
  }
  explicit ParsedEigenstrain(const InputParameters & p) : Property(p), _arguments(p)
  {
    const auto type = p.getString("strain_type");
    if (type != "linear" && type != "volumetric")
      throw InputError("'" + name() + "': strain_type must be 'linear' or 'volumetric', not '" +
                       type + "'.");
    _scale = type == "volumetric" ? 1.0 / 3.0 : 1.0;
    const auto direction = p.getString("direction");
    if (direction != "isotropic" && direction != "axial" && direction != "transverse")
      throw InputError("'" + name() + "': direction must be isotropic, axial or transverse, not '" +
                       direction + "'.");
    if (type == "volumetric" && direction != "isotropic")
      throw InputError("'" + name() +
                       "': a volumetric strain acts equally in every direction; "
                       "use strain_type = 'linear' for an axial or transverse "
                       "strain.");
    _direction = direction == "isotropic" ? 0 : direction == "axial" ? 1 : 2;
    const auto formulation = p.getString("formulation");
    if (formulation == "axisymmetric" || formulation == "axisymmetric_1d")
      _axial = 1;
    else if (formulation == "plane_strain" || formulation == "three_dimensional")
      _axial = 2;
    else
      throw InputError("'" + name() + "': unknown formulation '" + formulation + "'.");
  }
  void initialSetup(Problem & problem) override
  {
    Property::initialSetup(problem);
    const auto text = _params.getString("expression");
    _arguments.compile(problem, {text}, {}, name());
    _expression = std::make_unique<ParsedExpression>(text, _arguments.names());
    _free_slots.clear();
    _free_values.clear();
    for (const auto & [n, value] : _params.getRealMap("stress_free_state"))
    {
      const int k = _expression->index(n);
      if (k < 0)
        throw InputError("'" + name() + "': 'stress_free_state' names '" + n +
                         "', which the expression does not use.");
      _free_slots.push_back(k);
      _free_values.push_back(value);
    }
  }
  void declareProperties(PropertyRegistry & r) override
  {
    _prop = r.declare(_params.getString("eigenstrain_name"), 6);
  }
  void computeProperties(QpContext & ctx) const override
  {
    std::array<ADReal, ParsedArguments::kMaxArguments> args;
    _arguments.fill(ctx, args.data());
    ADReal e = _expression->value(args.data());
    if (!_free_slots.empty())
    {
      for (std::size_t k = 0; k < _free_slots.size(); ++k)
        args[_free_slots[k]] = ADReal(_free_values[k]);
      e = e - _expression->value(args.data());
    }
    e = e * _scale;
    for (int i = 0; i < 6; ++i)
      ctx.property(_prop, i) = ADReal(0.0);
    for (int i = 0; i < 3; ++i)
    {
      const bool axial = i == _axial;
      if (_direction == 0 || (_direction == 1 && axial) || (_direction == 2 && !axial))
        ctx.property(_prop, i) = e;
    }
  }

private:
  ParsedArguments _arguments;
  std::unique_ptr<ParsedExpression> _expression;
  double _scale = 1.0;
  int _direction = 0, _axial = 2, _prop = -1;
  std::vector<int> _free_slots;
  std::vector<double> _free_values;
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
        std::string(
            "A term of the equation of 'variable' in the general form div Gamma = f, whose flux Gamma and source f are expressions. ") +
        "The expressions may use every field of the problem by its name, the components of its gradient as grad_x(u), grad_y(u) and grad_z(u), the coordinates x, y and z, the time t, the registered functions, the element fields, the properties and named constants, each by its name. " +
        "They are compiled once and differentiated automatically, so the Jacobian is exact and the assembly runs on every thread and every process. " +
        "In axisymmetric coordinates x is the radius and y the axial position, and the divergence includes the factor of the radius. " +
        kExpressionHelp);
    p.addOptional(
        "flux",
        ParameterKind::StringList,
        std::vector<std::string>{},
        "The flux Gamma of the equation, one expression for each space dimension of the mesh. For diffusion with a coefficient c, Gamma = -c grad u. Default none.");
    p.addOptional("source",
                  ParameterKind::String,
                  std::string(""),
                  "The source f of the equation, an expression. Default none.");
    p.addOptional("constants",
                  ParameterKind::RealMap,
                  RealMap{},
                  "Named constants that the expressions use: name -> value. Default none.");
    return p;
  }

  explicit ParsedKernel(const InputParameters & p) : Kernel(p)
  {
    _flux_text = p.getStringList("flux");
    _source_text = p.getString("source");
    if (_flux_text.empty() && _source_text.empty())
      throw InputError("'" + name() + "': give a 'flux', a 'source' or both.");
  }

  bool hasFlux() const override { return !_flux_text.empty(); }
  bool hasSource() const override { return !_source_text.empty(); }

  void initialSetup(Problem & problem) override
  {
    Kernel::initialSetup(problem);
    const int dim = problem.mesh().spaceDimension();
    if (!_flux_text.empty() && static_cast<int>(_flux_text.size()) != dim)
      throw InputError("'" + name() + "': the flux has " + std::to_string(_flux_text.size()) +
                       " component(s), and the mesh has " + std::to_string(dim) +
                       " dimension(s). Give one expression for each dimension.");
    FieldExpressions::Names names;
    for (const auto & [n, value] : _params.getRealMap("constants"))
      names.constants.emplace_back(n, value);
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

private:
  std::vector<std::string> _flux_text;
  std::string _source_text;
  FieldExpressions _expressions;
  std::size_t _num_flux = 0;
};

void
registerParsedObjects(Factory & f)
{
  f.add<ParsedProperty>("parsed", ObjectCategory::Property, "framework");
  f.add<ParsedEigenstrain>("parsed_eigenstrain", ObjectCategory::Property, "solid_mechanics");
  f.add<ParsedKernel>("parsed_kernel", ObjectCategory::Kernel, "framework");
}

} // namespace dualmesh
