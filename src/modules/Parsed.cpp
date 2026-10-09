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
#include "dualmesh/base/Problem.h"
#include "dualmesh/base/Property.h"
#include "dualmesh/core/ParsedFunction.h"

#include <array>
#include <memory>

namespace dualmesh
{

namespace
{

/// The arguments of a parsed expression: coupled variables (their values for
/// coefficients, lagged under Picard iteration as in every property object), functions of
/// position and time, scalar properties of other property objects and constants,
/// each used in the expression by its name.
class ParsedArguments
{
public:
  static constexpr int kMaxArguments = 32;

  static void addParams(InputParameters & p)
  {
    p.addOptional("coupled_variables",
                  ParameterKind::StringList,
                  std::vector<std::string>{},
                  "Solution variables the expression uses, each by its own name. Default none.");
    p.addOptional(
        "function_names",
        ParameterKind::StringList,
        std::vector<std::string>{},
        "Functions of position and time registered on the problem (for instance a "
        "heat source or a measured temperature), each used by its name and evaluated at the "
        "integration point. Default none.");
    p.addOptional("element_field_names",
                  ParameterKind::StringList,
                  std::vector<std::string>{},
                  "Element fields of the problem (one value per element, set from Python with "
                  "Problem.set_element_field, such as the highest temperature an element has "
                  "reached), each used by its name. Default none.");
    p.addOptional("coupled_properties",
                  ParameterKind::StringList,
                  std::vector<std::string>{},
                  "Scalar properties of other property objects, each used by its name. Those "
                  "property objects must be added to the problem before this one. Default none.");
    p.addOptional("constant_names",
                  ParameterKind::StringList,
                  std::vector<std::string>{},
                  "Names of constants used in the expression, matched by position with "
                  "'constant_values'. Default none.");
    p.addOptional("constant_values",
                  ParameterKind::RealList,
                  std::vector<double>{},
                  "Values of the constants named in 'constant_names'. Default none.");
  }

  explicit ParsedArguments(const InputParameters & p)
  {
    _variables = p.getStringList("coupled_variables");
    _functions = p.getStringList("function_names");
    _properties = p.getStringList("coupled_properties");
    _fields = p.getStringList("element_field_names");
    _constants = p.getStringList("constant_names");
    _values = p.getRealList("constant_values");
    if (_constants.size() != _values.size())
      throw InputError("'constant_names' and 'constant_values' must have the same length.");
  }

  /// The names in the order of the argument array, with @p leading names
  /// (the object's own arguments, such as the von Mises stress) first.
  std::vector<std::string> names(const std::vector<std::string> & leading = {}) const
  {
    std::vector<std::string> n = leading;
    for (const auto * list : {&_variables, &_functions, &_fields, &_properties, &_constants})
      n.insert(n.end(), list->begin(), list->end());
    if (n.size() > kMaxArguments)
      throw InputError("A parsed expression takes at most " + std::to_string(kMaxArguments) +
                       " arguments.");
    return n;
  }

  void initialSetup(Problem & problem, const std::string & owner)
  {
    _variable_ids.clear();
    for (const auto & v : _variables)
      _variable_ids.push_back(problem.variableIndex(v));
    _function_ptrs.clear();
    for (const auto & f : _functions)
      _function_ptrs.push_back(problem.function(f));
    _field_ptrs.clear();
    for (const auto & f : _fields)
      _field_ptrs.push_back(&problem.elementField(f));
    _property_ids.clear();
    const auto & registry = problem.propertyRegistry();
    for (const auto & name : _properties)
    {
      if (!registry.has(name))
        throw InputError("'" + owner + "': no property object declares the property '" + name +
                         "'. Add the property object that declares it before this one.");
      if (registry.components(name) != 1)
        throw InputError("'" + owner + "': the property '" + name +
                         "' has several components; only scalar properties can be used.");
      _property_ids.push_back(registry.id(name));
    }
  }

  /// Fill @p out from position @p offset on.
  template <typename T> void fill(const QpContext & ctx, T * out, int offset = 0) const
  {
    int k = offset;
    for (int id : _variable_ids)
      out[k++] = value<T>(ctx.coefficientValue(id));
    for (const auto & f : _function_ptrs)
      out[k++] = T(f->value(ctx.x, ctx.time));
    for (const auto * f : _field_ptrs)
      out[k++] = T(ctx.element >= 0 ? (*f)[ctx.element] : 0.0);
    for (int id : _property_ids)
      out[k++] = value<T>(ctx.property(id));
    for (double c : _values)
      out[k++] = T(c);
  }

private:
  template <typename T> static T value(const ADReal & a)
  {
    if constexpr (std::is_same_v<T, double>)
      return a.value();
    else
      return a;
  }

  std::vector<std::string> _variables, _functions, _fields, _properties, _constants;
  std::vector<const std::vector<double> *> _field_ptrs;
  std::vector<double> _values;
  std::vector<int> _variable_ids, _property_ids;
  std::vector<FunctionPtr> _function_ptrs;
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
        std::string("A scalar property given by an expression of solution variables, "
                    "functions, other properties and constants, compiled once and "
                    "differentiated automatically. It is how a correlation, for instance a "
                    "thermal conductivity k(T, x), is added without writing C++. ") +
        kExpressionHelp);
    p.addRequired("property_name", ParameterKind::String, "Name of the declared property.");
    p.addRequired("expression", ParameterKind::String, "The expression of the property.");
    ParsedArguments::addParams(p);
    return p;
  }
  explicit ParsedProperty(const InputParameters & p) : Property(p), _arguments(p)
  {
    _expression = std::make_unique<ParsedExpression>(p.getString("expression"), _arguments.names());
  }
  void initialSetup(Problem & problem) override
  {
    Property::initialSetup(problem);
    _arguments.initialSetup(problem, name());
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
            "the axis or only across it (direction). With 'stress_free_state_names' the strain "
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
    p.addOptional("stress_free_state_names",
                  ParameterKind::StringList,
                  std::vector<std::string>{},
                  "Arguments that take the values of 'stress_free_state_values' in the state "
                  "where the strain must be zero. Default none: the expression is used as it "
                  "is.");
    p.addOptional("stress_free_state_values",
                  ParameterKind::RealList,
                  std::vector<double>{},
                  "Values of the arguments named in 'stress_free_state_names'.");
    ParsedArguments::addParams(p);
    return p;
  }
  explicit ParsedEigenstrain(const InputParameters & p) : Property(p), _arguments(p)
  {
    const auto names = _arguments.names();
    _expression = std::make_unique<ParsedExpression>(p.getString("expression"), names);
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
    const auto free_names = p.getStringList("stress_free_state_names");
    _free_values = p.getRealList("stress_free_state_values");
    if (free_names.size() != _free_values.size())
      throw InputError("'" + name() +
                       "': 'stress_free_state_names' and 'stress_free_state_values' must have "
                       "the same length.");
    for (const auto & n : free_names)
    {
      const int k = _expression->index(n);
      if (k < 0)
        throw InputError("'" + name() + "': 'stress_free_state_names' lists '" + n +
                         "', which is not an argument of the expression.");
      _free_slots.push_back(k);
    }
  }
  void initialSetup(Problem & problem) override
  {
    Property::initialSetup(problem);
    _arguments.initialSetup(problem, name());
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

void
registerParsedObjects(Factory & f)
{
  f.add<ParsedProperty>("parsed_property", ObjectCategory::Property, "framework");
  f.add<ParsedEigenstrain>("parsed_eigenstrain", ObjectCategory::Property, "solid_mechanics");
}

} // namespace dualmesh
