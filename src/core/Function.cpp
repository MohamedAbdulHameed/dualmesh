// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/core/Function.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dualmesh
{

PiecewiseLinearFunction::PiecewiseLinearFunction(std::vector<double> abscissa,
                                                 std::vector<double> ordinate,
                                                 int axis)
    : _x(std::move(abscissa)), _y(std::move(ordinate)), _axis(axis)
{
  if (_x.size() != _y.size() || _x.empty())
    throw InputError("PiecewiseLinearFunction needs equally sized, non-empty lists.");
  if (!std::is_sorted(_x.begin(), _x.end()))
    throw InputError("PiecewiseLinearFunction abscissa must be increasing.");
  if (axis < 0 || axis > 3)
    throw InputError("PiecewiseLinearFunction axis must be 0, 1, 2 (x, y, z) or 3 (time).");
}

double
PiecewiseLinearFunction::value(const Point & x, double t) const
{
  const double s = _axis == 3 ? t : x[_axis];
  if (s <= _x.front())
    return _y.front();
  if (s >= _x.back())
    return _y.back();
  auto it = std::upper_bound(_x.begin(), _x.end(), s);
  std::size_t i = std::distance(_x.begin(), it) - 1;
  const double w = (s - _x[i]) / (_x[i + 1] - _x[i]);
  return (1 - w) * _y[i] + w * _y[i + 1];
}

} // namespace dualmesh

namespace dualmesh
{
namespace
{
/// Index of the interval containing x and the weight of its upper end.
void
bracket(const std::vector<double> & grid, double x, std::size_t & i, double & w)
{
  if (grid.size() == 1 || x <= grid.front())
  {
    i = 0;
    w = 0.0;
    return;
  }
  if (x >= grid.back())
  {
    i = grid.size() - 2;
    w = 1.0;
    return;
  }
  i = static_cast<std::size_t>(std::upper_bound(grid.begin(), grid.end(), x) - grid.begin()) - 1;
  w = (x - grid[i]) / (grid[i + 1] - grid[i]);
}
} // namespace

CylinderTableFunction::CylinderTableFunction(std::vector<double> r,
                                             std::vector<double> z,
                                             std::vector<double> t,
                                             std::vector<double> values,
                                             const std::string & geometry,
                                             double slice_z)
    : _r(std::move(r)), _z(std::move(z)), _t(std::move(t)), _v(std::move(values)), _slice_z(slice_z)
{
  if (geometry == "axisymmetric_1d")
    _geometry = 0;
  else if (geometry == "axisymmetric")
    _geometry = 1;
  else if (geometry == "three_dimensional")
    _geometry = 2;
  else
    throw InputError("CylinderTableFunction: unknown geometry '" + geometry +
                     "' (use axisymmetric_1d, axisymmetric or three_dimensional).");
  if (_r.empty() || _z.empty() || _t.empty())
    throw InputError("CylinderTableFunction: every grid needs at least one point.");
  if (_v.size() != _r.size() * _z.size() * _t.size())
    throw InputError("CylinderTableFunction: expected len(r) * len(z) * len(t) values.");
  for (const auto * g : {&_r, &_z, &_t})
    for (std::size_t i = 1; i < g->size(); ++i)
      if (!((*g)[i] > (*g)[i - 1]))
        throw InputError("CylinderTableFunction: every grid must increase strictly.");
}

double
CylinderTableFunction::value(const Point & x, double t) const
{
  double r = 0, z = 0;
  if (_geometry == 0)
  {
    r = std::abs(x[0]);
    z = _slice_z;
  }
  else if (_geometry == 1)
  {
    r = std::abs(x[0]);
    z = x[1];
  }
  else
  {
    r = std::hypot(x[0], x[1]);
    z = x[2];
  }
  std::size_t ir, iz, it;
  double wr, wz, wt;
  bracket(_r, r, ir, wr);
  bracket(_z, z, iz, wz);
  bracket(_t, t, it, wt);
  const std::size_t nr = _r.size(), nz = _z.size();
  const auto at = [&](std::size_t a, std::size_t b, std::size_t c)
  {
    a = std::min(a, nr - 1);
    b = std::min(b, nz - 1);
    c = std::min(c, _t.size() - 1);
    return _v[(c * nz + b) * nr + a];
  };
  double result = 0.0;
  for (int dc = 0; dc < 2; ++dc)
    for (int db = 0; db < 2; ++db)
      for (int da = 0; da < 2; ++da)
      {
        const double w = (da ? wr : 1 - wr) * (db ? wz : 1 - wz) * (dc ? wt : 1 - wt);
        if (w != 0.0)
          result += w * at(ir + da, iz + db, it + dc);
      }
  return result;
}

} // namespace dualmesh
