// SPDX-License-Identifier: LGPL-2.1-or-later
//
// InputParameters: a typed, documented parameter dictionary, in the spirit of
// MOOSE's InputParameters.  Every registered object declares the parameters it
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

/// A value that an input parameter can hold.
using ParameterValue = std::variant<std::monostate,
                                    bool,
                                    long,
                                    double,
                                    std::string,
                                    std::vector<double>,
                                    std::vector<long>,
                                    std::vector<std::string>,
                                    std::shared_ptr<Function>>;

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
  Function
};

std::string parameterKindName(ParameterKind kind);

struct ParameterInfo
{
  ParameterKind kind;
  std::string description;
  bool required = false;
  ParameterValue value; ///< current value (default until set)
  bool set_by_user = false;
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
  /// Class-level description shown in the generated syntax documentation.
  void setClassDescription(const std::string & d) { _class_description = d; }
  const std::string & classDescription() const { return _class_description; }

  // ---- assignment (from Python/input files) ------------------------------
  /// Assign a value; performs lenient conversions (int->real, scalar->list).
  void set(const std::string & name, ParameterValue value);
  /// Set a parameter without declaration checks (framework-internal).
  void setPrivate(const std::string & name, ParameterValue value);

  /// Throw if a required parameter is missing.
  void validate(const std::string & object_description) const;

  // ---- queries -----------------------------------------------------------
  bool has(const std::string & name) const { return _params.count(name) > 0; }
  bool isSet(const std::string & name) const;
  bool isDeclared(const std::string & name) const { return has(name); }
  const std::map<std::string, ParameterInfo> & all() const { return _params; }

  bool getBool(const std::string & name) const;
  long getInt(const std::string & name) const;
  double getReal(const std::string & name) const;
  std::string getString(const std::string & name) const;
  std::vector<double> getRealList(const std::string & name) const;
  std::vector<long> getIntList(const std::string & name) const;
  std::vector<std::string> getStringList(const std::string & name) const;
  /// Raw access (Function-valued parameters are resolved by the Problem).
  const ParameterValue & getRaw(const std::string & name) const;

  /// Human-readable dump of the declared parameters (for documentation).
  std::string describe() const;

private:
  const ParameterInfo & info(const std::string & name) const;
  std::map<std::string, ParameterInfo> _params;
  std::string _class_description;
};

/// Error type used throughout dualmesh for user-facing input errors.
class InputError : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
};

} // namespace dualmesh
