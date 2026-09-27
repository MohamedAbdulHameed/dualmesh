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

/// A constant whose value can be changed between solves, for a quantity that
/// is updated outside the nonlinear iteration (the gas pressure in a fuel rod,
/// say).
class SettableFunction : public Function
{
public:
  explicit SettableFunction(double c = 0.0) : _c(c) {}
  double value(const Point &, double) const override { return _c; }
  void set(double c) { _c = c; }
  double get() const { return _c; }
  std::string description() const override { return "settable " + std::to_string(_c); }

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

/// A field of a cylindrical body (a fuel rod) tabulated on a grid of the
/// radius r, the axial position z and the time t, and interpolated linearly
/// in each of the three (constant beyond the ends of the grid).  The geometry
/// says how r and z follow from a point, in the vocabulary of the stress
/// formulations: "axisymmetric_1d" for a one-dimensional radial slice at a
/// fixed axial position (r = x), "axisymmetric" for an r-z mesh (r = x,
/// z = y), and "three_dimensional" for a mesh with the rod axis on the z axis
/// (r = sqrt(x^2 + y^2)).
class CylinderTableFunction : public Function
{
public:
  CylinderTableFunction(std::vector<double> r,
                        std::vector<double> z,
                        std::vector<double> t,
                        std::vector<double> values,
                        const std::string & geometry,
                        double slice_z = 0.0);
  double value(const Point & x, double t) const override;
  std::string description() const override { return "cylinder table"; }

private:
  std::vector<double> _r, _z, _t, _v;
  int _geometry = 0;
  double _slice_z = 0.0;
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
