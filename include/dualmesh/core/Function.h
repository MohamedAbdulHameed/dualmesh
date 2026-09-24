// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Functions of space and time, f(x, y, z, t), used for coefficients, sources,
// and boundary values.  Python callables are wrapped by the bindings.
#pragma once

#include "dualmesh/core/InputParameters.h"
#include "dualmesh/core/Types.h"

#include <memory>
#include <string>
#include <vector>

namespace dualmesh
{

class Function
{
public:
  virtual ~Function() = default;
  virtual double value(const Point & x, double t) const = 0;
  virtual std::string description() const { return "function"; }
  /// Whether this function may be called from several threads at once (see
  /// Object::threadSafe); Python callables override it to return false.
  virtual bool threadSafe() const { return true; }
};

using FunctionPtr = std::shared_ptr<Function>;

class ConstantFunction : public Function
{
public:
  explicit ConstantFunction(double c) : _c(c) {}
  double value(const Point &, double) const override { return _c; }
  std::string description() const override { return std::to_string(_c); }
  double constant() const { return _c; }

private:
  double _c;
};

/// Piecewise-linear interpolation in one coordinate (or time).
class PiecewiseLinearFunction : public Function
{
public:
  /// @param axis 0,1,2 for x,y,z or 3 for time.
  PiecewiseLinearFunction(std::vector<double> abscissa, std::vector<double> ordinate, int axis);
  double value(const Point & x, double t) const override;

private:
  std::vector<double> _x, _y;
  int _axis;
};

/// Polynomial in time multiplied by a spatial function: g(t) * f(x).
class ProductFunction : public Function
{
public:
  ProductFunction(FunctionPtr a, FunctionPtr b) : _a(std::move(a)), _b(std::move(b)) {}
  double value(const Point & x, double t) const override
  {
    return _a->value(x, t) * _b->value(x, t);
  }

private:
  FunctionPtr _a, _b;
};

} // namespace dualmesh
