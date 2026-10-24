// SPDX-License-Identifier: LGPL-2.1-or-later
//
// InputParameters: a typed, documented parameter dictionary.
// Every registered object declares the parameters it
// accepts (with a description and, optionally, a default value).  Unknown or
// missing required parameters are reported with a clear error message.
#pragma once

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace dualmesh
{

class Function;

/// A mapping from names to numbers, such as the values of named constants.
using RealMap = std::map<std::string, double>;
/// A mapping from names to texts, such as the functions of named properties.
using StringMap = std::map<std::string, std::string>;

/// A value that an input parameter can hold.
using ParameterValue = std::variant<std::monostate,
                                    bool,
                                    long,
                                    double,
                                    std::string,
                                    std::vector<double>,
                                    std::vector<long>,
                                    std::vector<std::string>,
                                    std::shared_ptr<Function>,
                                    RealMap,
                                    StringMap>;

/// The kind of value a parameter is declared to take.
enum class ParameterKind
{
  Boolean,
  Integer,
  Real,
  String,
  RealList,
  IntegerList,
  StringList,
  /// A constant, the name of a Function object, or a Function instance.
  Function,
  /// One name or a list of names, stored as a list (the variables of an object).
  Names,
  /// Names with a number each.
  RealMap,
  /// Names with a text each; a number given as a value is kept as its text.
  StringMap
};

std::string parameterKindName(ParameterKind kind);

/// The values a number of a parameter may take, checked when the object is created.  A function
/// or an expression is evaluated only where it is used, and is not checked here.
enum class ParameterRange
{
  Any,
  Positive,
  NonNegative
};

struct ParameterInfo
{
  ParameterKind kind;
  std::string description;
  bool required = false;
  ParameterValue value; ///< current value (default until set)
  bool set_by_user = false;
  ParameterRange range = ParameterRange::Any;
};

class InputParameters
{
public:
  InputParameters() = default;

  // ---- declaration (used in validParams) ---------------------------------
  void addRequired(const std::string & name, ParameterKind kind, const std::string & description);
  void addOptional(const std::string & name,
                   ParameterKind kind,
                   ParameterValue default_value,
                   const std::string & description);
  /// Restrict the numbers a declared parameter takes (a single number or every number of a list).
  void setRange(const std::string & name, ParameterRange range);
  /// Class-level description shown in the generated syntax documentation.
  void setClassDescription(const std::string & d) { _class_description = d; }
  const std::string & classDescription() const { return _class_description; }
  /// The parameters that name the variables of a model whose kernel assembles every equation of the model, one instance per variable (a beam or a plate).
  /// An instance added without 'variable' is added once for every variable these parameters name.
  void setModelVariables(std::vector<std::string> names) { _model_variables = std::move(names); }
  const std::vector<std::string> & modelVariables() const { return _model_variables; }

  // ---- assignment (from Python) --------------------------------------------
  /// Assign a value; performs lenient conversions (int->real, scalar->list).
  void set(const std::string & name, ParameterValue value);
  /// Remove a declared parameter (an object that does not use one of its base's parameters).
  void remove(const std::string & name) { _params.erase(name); }
  /// Set a parameter without declaration checks (framework-internal).
  void setPrivate(const std::string & name, ParameterValue value);
  /// Replace the default value of a declared parameter that the user did not set, keeping it
  /// marked as a default: for a default that depends on the problem, such as a value read from
  /// a property when one is declared and a stated constant otherwise.
  void replaceDefault(const std::string & name, ParameterValue value);

  /// Throw if a required parameter is missing.
  void validate(const std::string & object_description) const;

  // ---- queries -----------------------------------------------------------
  bool has(const std::string & name) const { return _params.count(name) > 0; }
  bool isSet(const std::string & name) const;
  /// Whether the user gave the parameter, as opposed to its default being used.
  bool isSetByUser(const std::string & name) const;
  const std::map<std::string, ParameterInfo> & all() const { return _params; }

  bool getBool(const std::string & name) const;
  long getInt(const std::string & name) const;
  double getReal(const std::string & name) const;
  std::string getString(const std::string & name) const;
  std::vector<double> getRealList(const std::string & name) const;
  std::vector<long> getIntList(const std::string & name) const;
  std::vector<std::string> getStringList(const std::string & name) const;
  RealMap getRealMap(const std::string & name) const;
  StringMap getStringMap(const std::string & name) const;
  /// Raw access (Function-valued parameters are resolved by the Problem).
  const ParameterValue & getRaw(const std::string & name) const;

  /// Human-readable dump of the declared parameters (for documentation).
  std::string describe() const;

private:
  const ParameterInfo & info(const std::string & name) const;
  std::map<std::string, ParameterInfo> _params;
  std::string _class_description;
  std::vector<std::string> _model_variables;
};

/// Error type used throughout dualmesh for user-facing input errors.
class InputError : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
};

/// The candidate closest to a misspelled @p name, for a "did you mean"
/// suggestion in an error message, or an empty string when none is close.
/// Closeness is the edit distance (insertions, deletions, substitutions and
/// transpositions of adjacent characters, ignoring case), and a candidate is
/// offered only when it is within a third of the length of the name.
std::string closestMatch(const std::string & name, const std::vector<std::string> & candidates);

/// " Did you mean 'x'?" when closestMatch finds a candidate, else "".
std::string didYouMean(const std::string & name, const std::vector<std::string> & candidates);

} // namespace dualmesh
