// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Base class of every user-configurable object (kernels, boundary conditions,
// materials, loads).  Mirrors MooseObject: an object is built from a validated
// InputParameters dictionary and later "set up" against a Problem, at which
// point names (variables, functions, material properties) are resolved.
#pragma once

#include "dualmesh/core/Function.h"
#include "dualmesh/core/InputParameters.h"
#include "dualmesh/fe/ReferenceElement.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

namespace dualmesh
{

class Problem;

class Object
{
public:
  explicit Object(const InputParameters & params);
  virtual ~Object() = default;

  /// Parameters common to all objects.
  static InputParameters validParams();

  const std::string & name() const { return _name; }
  const std::string & type() const { return _type; }
  const InputParameters & parameters() const { return _params; }

  /// Resolve names against the problem (called once before solving).
  virtual void initialSetup(Problem & problem);

  /// Whether this object may be evaluated from several threads at once.
  /// Objects implemented in C++ are thread safe because they only read their
  /// own parameters and write into the integration-point context, which is
  /// private to each thread.  Objects implemented in Python are not, because
  /// calling back into the interpreter requires the global interpreter lock;
  /// the Python bindings override this to return false, and the problem then
  /// assembles serially.
  ///
  /// A C++ object is still unsafe when one of its parameters holds a function
  /// that is not, which is what happens when a Python callable is passed
  /// straight to a parameter, as in value=lambda x, y, z, t: ...  The default
  /// implementation therefore checks every function-valued parameter as well,
  /// and an override in a derived class should call it rather than simply
  /// returning true.
  virtual bool threadSafe() const { return parametersAreThreadSafe(); }

  /// True when no function stored in this object's parameters refuses to be
  /// called from several threads.
  bool parametersAreThreadSafe() const;

protected:
  /// Resolve a Function-valued parameter (constant, function name, or object).
  FunctionPtr getFunction(Problem & problem, const std::string & param) const;

  InputParameters _params;
  std::string _name;
  std::string _type;
};

/// Common base of objects that contribute to the residual of one variable.
class ResidualObject : public Object
{
public:
  explicit ResidualObject(const InputParameters & params);
  static InputParameters validParams();

  void initialSetup(Problem & problem) override;

  /// Index of the variable (equation) this object contributes to.
  int variable() const { return _var; }
  const std::string & variableName() const { return _var_name; }
  /// Block ids this object acts on (empty = everywhere).
  const std::set<int> & blocks() const { return _blocks; }
  bool activeOnBlock(int b) const { return _blocks.empty() || _blocks.count(b); }
  const QuadratureSpec & quadrature() const { return _quad; }
  /// Whether the contribution is multiplied by the load factor (load stepping).
  bool scalesWithLoad() const { return _scale_with_load; }

protected:
  /// Resolve a coupled variable name given by parameter @p param.
  int coupledVariable(Problem & problem, const std::string & param) const;

  std::string _var_name;
  int _var = -1;
  std::set<int> _blocks;
  QuadratureSpec _quad;
  bool _scale_with_load = false;
};

} // namespace dualmesh
