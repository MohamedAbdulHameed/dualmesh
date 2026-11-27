// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Operators from the assembly: the Jacobian of a chosen set of terms (the operators of the
// projection time integration and of the eigenvalue study), the steady linear system of one
// Newton step, and the blocks of the preconditioner of a flow at the current solution.
#include "dualmesh/base/Problem.h"
#include "dualmesh/core/InputParameters.h"

#include <string>

namespace dualmesh
{

SparseMatrix
Problem::operatorOf(const OperatorOptions & options)
{
  if (options.order < 0 || options.order > 2)
    throw InputError("operatorOf: the order of the time derivative is 0, 1 or 2, not " +
                     std::to_string(options.order) + ".");
  // A residual-based stabilisation holds the time derivative inside a steady kernel, which the
  // time kernels alone do not include.
  if (options.time_kernels && !options.steady_terms)
  {
    const auto check = [](const Kernel & k)
    {
      if (k.residualHasTimeDerivative())
        throw InputError("The modes have no mass operator for '" + k.name() +
                         "', a stabilisation whose residual holds the time "
                         "derivative. Solve the modes without it.");
    };
    if (options.kernels)
      for (const Kernel * k : *options.kernels)
        check(*k);
    else
      for (const auto & k : _kernels)
        check(*k);
  }
  // The operators of an eigenvalue study take its rules: the consistent storage, where time
  // stepping lumps it (TimeDerivative).  The symbol and the rules are restored on every exit.
  struct Study
  {
    Problem & problem;
    double saved;
    bool on;
    ~Study()
    {
      if (on)
      {
        problem.setEigenvalue(saved);
        problem.setEigenvalueStudy(false);
      }
    }
  } study{*this, _eigenvalue, options.eigenvalue.has_value()};
  if (options.eigenvalue)
  {
    setEigenvalue(*options.eigenvalue);
    setEigenvalueStudy(true);
  }

  AssemblyOptions assembly;
  if (options.kernels)
  {
    assembly.kernels = &*options.kernels;
    assembly.conditions = &options.conditions;
  }
  const Vector rest = Vector::Zero(numDofs());
  assembly.old = &rest;
  assembly.dt = 1.0;
  assembly.stabilization = options.stabilization;
  assembly.include_steady_terms = options.steady_terms;
  assembly.include_time_kernels = options.time_kernels;
  // The first and the second time derivatives apart: c du/dt contributes c at order 1 and
  // c d2u/dt2 contributes c at order 2.
  if (options.order > 0 && hasSecondOrderVariables())
  {
    assembly.second_order = true;
    assembly.rate_scale = options.order == 1 ? 1.0 : 0.0;
    assembly.accel_scale = options.order == 2 ? 1.0 : 0.0;
    // The first time derivatives of the other variables take no part in order 2.
    if (options.order == 2)
      assembly.dt = 1e300;
  }
  else if (options.order == 2)
  {
    // No second time derivative: the operator of order 2 is zero.
    assembly.include_steady_terms = false;
    assembly.include_time_kernels = false;
  }
  const Vector & state = options.at_solution ? _U : rest;
  Vector R;
  SparseMatrix J;
  assemble(state, assembly, R, &J);
  if (options.fold_dependents)
    foldDependents(R, &J);
  J.makeCompressed();
  return J;
}

void
Problem::linearSystem(Vector & R, SparseMatrix & J)
{
  initialize();
  applyDirichlet(_U, 1.0);
  assemble(_U, AssemblyOptions{}, R, &J);
  dirichletRows(R, &J);
  J.makeCompressed();
}

} // namespace dualmesh
