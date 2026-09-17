// SPDX-License-Identifier: LGPL-2.1-or-later
//
// dualmesh: forward-mode automatic differentiation scalar.
//
// An ADReal carries a value and the partial derivatives of that value with
// respect to the local (element) degrees of freedom.  Residual objects are
// written once in terms of ADReal, and the element Jacobian needed by
// Newton's method is obtained exactly, without hand-coded derivatives.
// This mirrors the "AD" objects of the MOOSE framework.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <ostream>

namespace dualmesh
{

/// Maximum number of local degrees of freedom per element that can be seeded.
/// (Quad4 x 5 fields = 20, Hex8 x 3 fields = 24, Quad9 x 5 fields = 45.)
inline constexpr int kMaxDerivatives = 48;

class ADReal
{
public:
  ADReal() : _value(0.0), _size(0) {}
  ADReal(double v) : _value(v), _size(0) {} // NOLINT(implicit)

  /// Construct an independent variable: derivative 1 in slot @p index.
  static ADReal independent(double v, int index, int size)
  {
    ADReal r;
    r._value = v;
    r._size = size;
    std::fill_n(r._deriv.begin(), size, 0.0);
    r._deriv[index] = 1.0;
    return r;
  }

  /// A value with @p size zero derivatives (to be filled by setDerivative).
  static ADReal withZeroDerivatives(double v, int size)
  {
    ADReal r;
    r._value = v;
    r._size = size;
    std::fill_n(r._deriv.begin(), size, 0.0);
    return r;
  }
  void setDerivative(int i, double d) { _deriv[i] = d; }

  double value() const { return _value; }
  double & value() { return _value; }
  int size() const { return _size; }
  double derivative(int i) const { return i < _size ? _deriv[i] : 0.0; }
  const double * derivatives() const { return _deriv.data(); }

  /// Drop all derivative information (used for lagged/Picard coefficients).
  ADReal detached() const { return ADReal(_value); }

  ADReal & operator+=(const ADReal & b)
  {
    _value += b._value;
    addScaled(b, 1.0);
    return *this;
  }
  ADReal & operator-=(const ADReal & b)
  {
    _value -= b._value;
    addScaled(b, -1.0);
    return *this;
  }
  ADReal & operator*=(const ADReal & b)
  {
    if (&b == this)
      return (*this) = chain(_value * _value, 2.0 * _value);
    // d(ab) = a db + b da
    const double a = _value;
    scale(b._value);
    addScaled(b, a);
    _value = a * b._value;
    return *this;
  }
  ADReal & operator/=(const ADReal & b)
  {
    if (&b == this)
      return (*this) = ADReal(1.0);
    // d(a/b) = da/b - a db / b^2
    const double inv = 1.0 / b._value;
    const double q = _value * inv;
    scale(inv);
    addScaled(b, -q * inv);
    _value = q;
    return *this;
  }
  ADReal & operator*=(double s)
  {
    _value *= s;
    scale(s);
    return *this;
  }
  ADReal & operator/=(double s) { return (*this) *= (1.0 / s); }
  ADReal & operator+=(double s)
  {
    _value += s;
    return *this;
  }
  ADReal & operator-=(double s)
  {
    _value -= s;
    return *this;
  }

  ADReal operator-() const
  {
    ADReal r(*this);
    r._value = -r._value;
    r.scale(-1.0);
    return r;
  }

  /// Apply a scalar function with known derivative: f(a), f'(a).
  ADReal chain(double f, double dfda) const
  {
    ADReal r(*this);
    r._value = f;
    r.scale(dfda);
    return r;
  }

private:
  void scale(double s)
  {
    for (int i = 0; i < _size; ++i)
      _deriv[i] *= s;
  }
  void addScaled(const ADReal & b, double s)
  {
    if (b._size == 0)
      return;
    if (_size < b._size)
    {
      std::fill(_deriv.begin() + _size, _deriv.begin() + b._size, 0.0);
      _size = b._size;
    }
    for (int i = 0; i < b._size; ++i)
      _deriv[i] += s * b._deriv[i];
  }

  double _value;
  int _size;
  std::array<double, kMaxDerivatives> _deriv;
};

// ---- arithmetic ----------------------------------------------------------
inline ADReal
operator+(ADReal a, const ADReal & b)
{
  return a += b;
}
inline ADReal
operator-(ADReal a, const ADReal & b)
{
  return a -= b;
}
inline ADReal
operator*(ADReal a, const ADReal & b)
{
  return a *= b;
}
inline ADReal
operator/(ADReal a, const ADReal & b)
{
  return a /= b;
}
inline ADReal
operator+(ADReal a, double b)
{
  return a += b;
}
inline ADReal
operator-(ADReal a, double b)
{
  return a -= b;
}
inline ADReal
operator*(ADReal a, double b)
{
  return a *= b;
}
inline ADReal
operator/(ADReal a, double b)
{
  return a /= b;
}
inline ADReal
operator+(double a, ADReal b)
{
  return b += a;
}
inline ADReal
operator-(double a, const ADReal & b)
{
  return (-b) += a;
}
inline ADReal
operator*(double a, ADReal b)
{
  return b *= a;
}
inline ADReal
operator/(double a, const ADReal & b)
{
  const double inv = 1.0 / b.value();
  return b.chain(a * inv, -a * inv * inv);
}

inline bool
operator<(const ADReal & a, const ADReal & b)
{
  return a.value() < b.value();
}
inline bool
operator>(const ADReal & a, const ADReal & b)
{
  return a.value() > b.value();
}
inline bool
operator<=(const ADReal & a, const ADReal & b)
{
  return a.value() <= b.value();
}
inline bool
operator>=(const ADReal & a, const ADReal & b)
{
  return a.value() >= b.value();
}

// ---- elementary functions ------------------------------------------------
inline ADReal
sqrt(const ADReal & a)
{
  const double s = std::sqrt(a.value());
  return a.chain(s, s > 0 ? 0.5 / s : 0.0);
}
inline ADReal
exp(const ADReal & a)
{
  const double e = std::exp(a.value());
  return a.chain(e, e);
}
inline ADReal
log(const ADReal & a)
{
  return a.chain(std::log(a.value()), 1.0 / a.value());
}
inline ADReal
sin(const ADReal & a)
{
  return a.chain(std::sin(a.value()), std::cos(a.value()));
}
inline ADReal
cos(const ADReal & a)
{
  return a.chain(std::cos(a.value()), -std::sin(a.value()));
}
inline ADReal
tanh(const ADReal & a)
{
  const double t = std::tanh(a.value());
  return a.chain(t, 1.0 - t * t);
}
inline ADReal
abs(const ADReal & a)
{
  return a.chain(std::abs(a.value()), a.value() >= 0 ? 1.0 : -1.0);
}
inline ADReal
pow(const ADReal & a, double p)
{
  if (p == 0.0)
    return ADReal(1.0);
  if (p == 1.0)
    return a;
  if (p == 2.0)
    return a * a;
  const double f = std::pow(a.value(), p);
  return a.chain(f, p * std::pow(a.value(), p - 1.0));
}
inline ADReal
pow(const ADReal & a, const ADReal & b)
{
  return exp(b * log(a));
}

inline std::ostream &
operator<<(std::ostream & os, const ADReal & a)
{
  return os << a.value();
}

/// Plain-value helper that works for both double and ADReal.
inline double
rawValue(double v)
{
  return v;
}
inline double
rawValue(const ADReal & v)
{
  return v.value();
}

using ADVector3 = std::array<ADReal, 3>;

} // namespace dualmesh
