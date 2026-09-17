// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/core/Function.h"

#include <algorithm>
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
