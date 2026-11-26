// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/core/ParsedFunction.h"

#include "dualmesh/core/ADReal.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace dualmesh
{

namespace
{
using Op = ParsedExpression::Op;
using Instruction = ParsedExpression::Instruction;

double
signOf(double v)
{
  return (v > 0) - (v < 0);
}

double
plainValue(double v)
{
  return v;
}
double
plainValue(const ADReal & v)
{
  return v.value();
}

using Unary = double (*)(double);
using Binary = double (*)(double, double);

struct UnaryEntry
{
  const char * name;
  Unary f;
};
struct BinaryEntry
{
  const char * name;
  Binary f;
};

const UnaryEntry unary_functions[] = {
    {"sin", [](double v) { return std::sin(v); }},
    {"cos", [](double v) { return std::cos(v); }},
    {"tan", [](double v) { return std::tan(v); }},
    {"asin", [](double v) { return std::asin(v); }},
    {"acos", [](double v) { return std::acos(v); }},
    {"atan", [](double v) { return std::atan(v); }},
    {"sinh", [](double v) { return std::sinh(v); }},
    {"cosh", [](double v) { return std::cosh(v); }},
    {"tanh", [](double v) { return std::tanh(v); }},
    {"exp", [](double v) { return std::exp(v); }},
    {"log", [](double v) { return std::log(v); }},
    {"log10", [](double v) { return std::log10(v); }},
    {"sqrt", [](double v) { return std::sqrt(v); }},
    {"abs", [](double v) { return std::abs(v); }},
    {"floor", [](double v) { return std::floor(v); }},
    {"ceil", [](double v) { return std::ceil(v); }},
    {"erf", [](double v) { return std::erf(v); }},
    {"sign", [](double v) { return signOf(v); }},
};

const BinaryEntry binary_functions[] = {
    {"atan2", [](double a, double b) { return std::atan2(a, b); }},
    {"pow", [](double a, double b) { return std::pow(a, b); }},
    {"hypot", [](double a, double b) { return std::hypot(a, b); }},
    {"min", [](double a, double b) { return std::min(a, b); }},
    {"max", [](double a, double b) { return std::max(a, b); }},
};

/// A recursive-descent compiler from text to a postfix program.
class Compiler
{
public:
  Compiler(const std::string & text, const std::vector<std::string> & names)
      : _s(text), _names(names)
  {
  }

  std::vector<Instruction> compile()
  {
    expression();
    skip();
    if (_i != _s.size())
      fail("unexpected '" + std::string(1, _s[_i]) + "'");
    return _program;
  }

private:
  [[noreturn]] void fail(const std::string & what) const
  {
    std::ostringstream os;
    os << what << " at position " << _i + 1 << " of \"" << _s << "\"";
    throw InputError(os.str());
  }

  void skip()
  {
    while (_i < _s.size() && std::isspace(static_cast<unsigned char>(_s[_i])))
      ++_i;
  }
  bool accept(const char * token)
  {
    skip();
    const std::size_t n = std::char_traits<char>::length(token);
    if (_s.compare(_i, n, token) == 0)
    {
      _i += n;
      return true;
    }
    return false;
  }
  void emit(Op op, int argument = 0, double value = 0)
  {
    _program.push_back({op, argument, value});
  }

  void expression() { comparison(); }

  void comparison()
  {
    sum();
    // Two-character operators first, so that "<=" is not read as "<".
    static const std::pair<const char *, Op> ops[] = {{"<=", Op::LessEqual},
                                                      {">=", Op::GreaterEqual},
                                                      {"==", Op::Equal},
                                                      {"!=", Op::NotEqual},
                                                      {"<", Op::Less},
                                                      {">", Op::Greater}};
    for (const auto & [token, op] : ops)
      if (accept(token))
      {
        sum();
        emit(op);
        return;
      }
  }

  void sum()
  {
    product();
    while (true)
    {
      if (accept("+"))
      {
        product();
        emit(Op::Add);
      }
      else if (peekMinus())
      {
        ++_i;
        product();
        emit(Op::Subtract);
      }
      else
        return;
    }
  }
  bool peekMinus()
  {
    skip();
    return _i < _s.size() && _s[_i] == '-';
  }

  void product()
  {
    unary();
    while (true)
    {
      skip();
      // "**" is exponentiation, handled in power(); a lone '*' multiplies.
      if (_i < _s.size() && _s[_i] == '*' && !(_i + 1 < _s.size() && _s[_i + 1] == '*'))
      {
        ++_i;
        unary();
        emit(Op::Multiply);
      }
      else if (accept("/"))
      {
        unary();
        emit(Op::Divide);
      }
      else
        return;
    }
  }

  void unary()
  {
    if (accept("-"))
    {
      unary();
      emit(Op::Negate);
    }
    else if (accept("+"))
      unary();
    else
      power();
  }

  void power()
  {
    primary();
    if (accept("**") || accept("^"))
    {
      unary(); // right-associative, and allows 2^-1
      emit(Op::Power);
    }
  }

  void primary()
  {
    skip();
    if (_i >= _s.size())
      fail("expression ends too early");
    const char c = _s[_i];
    if (std::isdigit(static_cast<unsigned char>(c)) || c == '.')
    {
      const char * begin = _s.c_str() + _i;
      char * end = nullptr;
      const double v = std::strtod(begin, &end);
      if (end == begin)
        fail("malformed number");
      _i += static_cast<std::size_t>(end - begin);
      emit(Op::Constant, 0, v);
      return;
    }
    if (c == '(')
    {
      ++_i;
      expression();
      if (!accept(")"))
        fail("missing ')'");
      return;
    }
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_')
    {
      const std::size_t start = _i;
      while (_i < _s.size() && (std::isalnum(static_cast<unsigned char>(_s[_i])) || _s[_i] == '_'))
        ++_i;
      const std::string name = _s.substr(start, _i - start);
      if (accept("("))
      {
        if (name == "grad_x" || name == "grad_y" || name == "grad_z")
        {
          gradient(name);
          return;
        }
        call(name);
        return;
      }
      for (std::size_t k = 0; k < _names.size(); ++k)
        if (name == _names[k])
        {
          emit(Op::Variable, static_cast<int>(k));
          return;
        }
      if (name == "pi")
      {
        emit(Op::Constant, 0, 3.14159265358979323846);
        return;
      }
      if (name == "e")
      {
        emit(Op::Constant, 0, std::exp(1.0));
        return;
      }
      _i = start;
      std::string known;
      for (std::size_t k = 0; k < _names.size(); ++k)
        known += (k == 0 ? "" : k + 1 == _names.size() ? " and " : ", ") + _names[k];
      fail("unknown name '" + name + "' (the variables are " +
           (known.empty() ? std::string("none") : known) + ")");
    }
    fail("unexpected '" + std::string(1, c) + "'");
  }

  /// A gradient component, grad_x(u), grad_y(u) or grad_z(u), after its opening parenthesis.
  /// It is the variable of the same name, "grad_x(u)", in the list of names.
  void gradient(const std::string & component)
  {
    skip();
    const std::size_t start = _i;
    while (_i < _s.size() && (std::isalnum(static_cast<unsigned char>(_s[_i])) || _s[_i] == '_'))
      ++_i;
    const std::string field = _s.substr(start, _i - start);
    if (field.empty())
      fail(component + " takes the name of a field");
    if (!accept(")"))
      fail("missing ')' after " + component + "(" + field);
    const std::string name = component + "(" + field + ")";
    for (std::size_t k = 0; k < _names.size(); ++k)
      if (name == _names[k])
      {
        emit(Op::Variable, static_cast<int>(k));
        return;
      }
    _i = start;
    fail("unknown gradient '" + name + "' ('" + field +
         "' is not a field of the problem, or the component does not exist in its dimension)");
  }

  void call(const std::string & name)
  {
    int count = 0;
    if (!accept(")"))
    {
      do
      {
        expression();
        ++count;
      } while (accept(","));
      if (!accept(")"))
        fail("missing ')' after the arguments of " + name);
    }
    if (name == "if")
    {
      if (count != 3)
        fail("if() takes three arguments: if(condition, value if true, value if false)");
      emit(Op::Select);
      return;
    }
    for (std::size_t k = 0; k < std::size(unary_functions); ++k)
      if (name == unary_functions[k].name)
      {
        if (count != 1)
          fail(name + "() takes one argument");
        emit(Op::Function1, static_cast<int>(k));
        return;
      }
    for (std::size_t k = 0; k < std::size(binary_functions); ++k)
      if (name == binary_functions[k].name)
      {
        if (count != 2)
          fail(name + "() takes two arguments");
        emit(Op::Function2, static_cast<int>(k));
        return;
      }
    fail("unknown function '" + name + "'");
  }

  const std::string & _s;
  const std::vector<std::string> & _names;
  std::size_t _i = 0;
  std::vector<Instruction> _program;
};

/// The stack depth the program reaches, so that evaluation can use a fixed
/// buffer and never allocate.
int
depthOf(const std::vector<Instruction> & program)
{
  int depth = 0, max_depth = 0;
  for (const auto & ins : program)
  {
    switch (ins.op)
    {
    case Op::Constant:
    case Op::Variable:
      ++depth;
      break;
    case Op::Negate:
    case Op::Function1:
      break;
    case Op::Select:
      depth -= 2;
      break;
    default:
      --depth;
    }
    max_depth = std::max(max_depth, depth);
  }
  return max_depth;
}

constexpr int kMaxStack = 64;

// ---- evaluation for double and ADReal -----------------------------------
double
unaryValue(int k, double a)
{
  return unary_functions[k].f(a);
}
double
binaryValue(int k, double a, double b)
{
  return binary_functions[k].f(a, b);
}

/// The derivative of unary function k at a, for the chain rule.
double
unaryDerivative(int k, double a)
{
  const std::string name = unary_functions[k].name;
  if (name == "sin")
    return std::cos(a);
  if (name == "cos")
    return -std::sin(a);
  if (name == "tan")
    return 1.0 / (std::cos(a) * std::cos(a));
  if (name == "asin")
    return 1.0 / std::sqrt(1.0 - a * a);
  if (name == "acos")
    return -1.0 / std::sqrt(1.0 - a * a);
  if (name == "atan")
    return 1.0 / (1.0 + a * a);
  if (name == "sinh")
    return std::cosh(a);
  if (name == "cosh")
    return std::sinh(a);
  if (name == "tanh")
    return 1.0 - std::tanh(a) * std::tanh(a);
  if (name == "exp")
    return std::exp(a);
  if (name == "log")
    return 1.0 / a;
  if (name == "log10")
    return 1.0 / (a * std::log(10.0));
  if (name == "sqrt")
    return 0.5 / std::sqrt(a);
  if (name == "abs")
    return signOf(a);
  if (name == "erf")
    return 2.0 / std::sqrt(3.14159265358979323846) * std::exp(-a * a);
  return 0.0; // floor, ceil, sign: piecewise constant
}

/// The second derivative of unary function k at a.
double
unarySecondDerivative(int k, double a)
{
  const std::string name = unary_functions[k].name;
  if (name == "sin")
    return -std::sin(a);
  if (name == "cos")
    return -std::cos(a);
  if (name == "tan")
    return 2.0 * std::tan(a) / (std::cos(a) * std::cos(a));
  if (name == "asin")
    return a / std::pow(1.0 - a * a, 1.5);
  if (name == "acos")
    return -a / std::pow(1.0 - a * a, 1.5);
  if (name == "atan")
    return -2.0 * a / ((1.0 + a * a) * (1.0 + a * a));
  if (name == "sinh")
    return std::sinh(a);
  if (name == "cosh")
    return std::cosh(a);
  if (name == "tanh")
    return -2.0 * std::tanh(a) * (1.0 - std::tanh(a) * std::tanh(a));
  if (name == "exp")
    return std::exp(a);
  if (name == "log")
    return -1.0 / (a * a);
  if (name == "log10")
    return -1.0 / (a * a * std::log(10.0));
  if (name == "sqrt")
    return -0.25 / (a * std::sqrt(a));
  if (name == "erf")
    return -4.0 * a / std::sqrt(3.14159265358979323846) * std::exp(-a * a);
  return 0.0; // abs, floor, ceil, sign: piecewise linear or constant
}

/// a scaled so that its value is v and its derivatives are d times those of a.
///
/// An infinite d (x^(2/3) or sqrt(x) at x = 0) multiplies only the nonzero
/// derivatives of a.  Where a does not depend on an unknown, the product
/// has the limit zero, and IEEE arithmetic would give inf * 0 = NaN, which
/// poisons the Jacobian.  (The SiC swelling [1 - exp(-dose/c(T))]^(2/3) at
/// zero dose is such a case: its temperature derivative is zero.)
ADReal
chain(const ADReal & a, double v, double d)
{
  if (std::isfinite(d))
    return (a - a.value()) * d + v;
  ADReal r = ADReal::withZeroDerivatives(v, a.size());
  for (int i = 0; i < a.size(); ++i)
    if (a.derivative(i) != 0.0)
      r.setDerivative(i, a.derivative(i) * d);
  return r;
}

ADReal
unaryValue(int k, const ADReal & a)
{
  return chain(a, unary_functions[k].f(a.value()), unaryDerivative(k, a.value()));
}

ADReal
binaryValue(int k, const ADReal & a, const ADReal & b)
{
  const std::string name = binary_functions[k].name;
  if (name == "pow")
    return pow(a, b);
  if (name == "min")
    return a.value() <= b.value() ? a : b;
  if (name == "max")
    return a.value() >= b.value() ? a : b;
  if (name == "hypot")
    return sqrt(a * a + b * b);
  // atan2(a, b): d = (b da - a db) / (a^2 + b^2).
  const double av = a.value(), bv = b.value(), r2 = av * av + bv * bv;
  return (a - av) * (bv / r2) - (b - bv) * (av / r2) + std::atan2(av, bv);
}

double
power(double a, double b)
{
  // An integer exponent is common (x**2) and is both faster and exact as
  // repeated multiplication.
  if (b == std::floor(b) && std::abs(b) <= 8)
  {
    const int n = static_cast<int>(b);
    double r = 1;
    for (int k = 0; k < std::abs(n); ++k)
      r *= a;
    return n >= 0 ? r : 1.0 / r;
  }
  return std::pow(a, b);
}

ADReal
power(const ADReal & a, const ADReal & b)
{
  // A constant exponent (the usual case) differentiates as p a^(p-1), which
  // is also defined for a negative base with an integer exponent.
  bool constant_exponent = true;
  for (int i = 0; i < b.size(); ++i)
    if (b.derivative(i) != 0.0)
    {
      constant_exponent = false;
      break;
    }
  if (constant_exponent)
  {
    const double p = b.value();
    const double av = a.value();
    return chain(a, power(av, p), p == 0.0 ? 0.0 : p * power(av, p - 1.0));
  }
  return pow(a, b);
}

// ---- evaluation with the size of the terms --------------------------------
/// A value and the size of the terms it is computed from: the sum, through the operations, of
/// their magnitudes, each scaled by the derivative that carries it to the result.  A sum of terms
/// that cancel is small and its size is not, and the round-off of the value is about the machine
/// epsilon times its size (a running error bound of forward error analysis, as in Higham,
/// Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM 2002, Sect. 3.3).
struct Bounded
{
  double value = 0.0;
  double size = 0.0;
  Bounded() = default;
  /// A number known exactly: its size is its magnitude.
  Bounded(double v) : value(v), size(std::abs(v)) {}
  Bounded(double v, double s) : value(v), size(s) {}
};

double
plainValue(const Bounded & v)
{
  return v.value;
}

Bounded
operator-(const Bounded & a)
{
  return {-a.value, a.size};
}

Bounded
operator+(const Bounded & a, const Bounded & b)
{
  return {a.value + b.value, a.size + b.size};
}

Bounded
operator-(const Bounded & a, const Bounded & b)
{
  return {a.value - b.value, a.size + b.size};
}

Bounded
operator*(const Bounded & a, const Bounded & b)
{
  return {a.value * b.value, a.size * b.size};
}

Bounded
operator/(const Bounded & a, const Bounded & b)
{
  const double q = a.value / b.value;
  return {q, a.size / std::abs(b.value) + std::abs(q) * b.size / std::abs(b.value)};
}

/// f(a) with the derivative @p d of f at a: the size of a carried by |d|, and that of the value.
Bounded
carried(double value, double d, const Bounded & a)
{
  return {value, std::abs(value) + (std::isfinite(d) ? std::abs(d) * a.size : 0.0)};
}

Bounded
power(const Bounded & a, const Bounded & b)
{
  const double v = power(a.value, b.value);
  const double d_base = b.value == 0.0 ? 0.0 : b.value * power(a.value, b.value - 1.0);
  Bounded r = carried(v, d_base, a);
  // An exponent computed from terms that cancel carries their size by d(a^b)/db = a^b log(a).
  if (b.size > std::abs(b.value) && a.value > 0.0)
    r.size += std::abs(v * std::log(a.value)) * b.size;
  return r;
}

Bounded
unaryValue(int k, const Bounded & a)
{
  return carried(unary_functions[k].f(a.value), unaryDerivative(k, a.value), a);
}

Bounded
binaryValue(int k, const Bounded & a, const Bounded & b)
{
  const std::string name = binary_functions[k].name;
  if (name == "pow")
    return power(a, b);
  if (name == "min")
    return a.value <= b.value ? a : b;
  if (name == "max")
    return a.value >= b.value ? a : b;
  const double v = binary_functions[k].f(a.value, b.value);
  if (name == "hypot")
    return {v, v > 0.0 ? v + (std::abs(a.value) * a.size + std::abs(b.value) * b.size) / v : 0.0};
  // atan2(a, b), whose derivatives are b / (a^2 + b^2) and -a / (a^2 + b^2).
  const double r2 = a.value * a.value + b.value * b.value;
  return {v,
          std::abs(v) +
              (r2 > 0.0 ? (std::abs(b.value) * a.size + std::abs(a.value) * b.size) / r2 : 0.0)};
}

// ---- evaluation with second derivatives ------------------------------------
constexpr int kOrder2 = ParsedExpression::kMaxSecondOrderVariables;

/// A value with its gradient and its Hessian in the variables of an expression: forward
/// differentiation to second order.
struct Taylor2
{
  double value = 0.0;
  std::array<double, kOrder2> gradient{};
  std::array<double, kOrder2 * kOrder2> hessian{};
  Taylor2() = default;
  Taylor2(double v) : value(v) {}
};

double
plainValue(const Taylor2 & v)
{
  return v.value;
}

/// x d, which is zero where x is zero, also for an infinite d (sqrt at 0), as in chain().
double
scaled(double x, double d)
{
  return x == 0.0 ? 0.0 : x * d;
}

/// f(a, b) from its value v and its first (fa, fb) and second (faa, fab, fbb) partial
/// derivatives at (a, b), by the chain rule to second order.
Taylor2
chain(const Taylor2 & a,
      const Taylor2 & b,
      double v,
      double fa,
      double fb,
      double faa,
      double fab,
      double fbb)
{
  Taylor2 r(v);
  for (int i = 0; i < kOrder2; ++i)
  {
    r.gradient[i] = scaled(a.gradient[i], fa) + scaled(b.gradient[i], fb);
    for (int j = 0; j < kOrder2; ++j)
    {
      const int k = i * kOrder2 + j;
      r.hessian[k] = scaled(a.hessian[k], fa) + scaled(b.hessian[k], fb) +
                     scaled(a.gradient[i] * a.gradient[j], faa) +
                     scaled(a.gradient[i] * b.gradient[j] + b.gradient[i] * a.gradient[j], fab) +
                     scaled(b.gradient[i] * b.gradient[j], fbb);
    }
  }
  return r;
}

/// f(a) from its value v and its first and second derivatives d1 and d2 at a.
Taylor2
chain(const Taylor2 & a, double v, double d1, double d2)
{
  return chain(a, Taylor2(), v, d1, 0.0, d2, 0.0, 0.0);
}

Taylor2
operator-(const Taylor2 & a)
{
  return chain(a, -a.value, -1.0, 0.0);
}

Taylor2
operator+(const Taylor2 & a, const Taylor2 & b)
{
  return chain(a, b, a.value + b.value, 1.0, 1.0, 0.0, 0.0, 0.0);
}

Taylor2
operator-(const Taylor2 & a, const Taylor2 & b)
{
  return chain(a, b, a.value - b.value, 1.0, -1.0, 0.0, 0.0, 0.0);
}

Taylor2
operator*(const Taylor2 & a, const Taylor2 & b)
{
  return chain(a, b, a.value * b.value, b.value, a.value, 0.0, 1.0, 0.0);
}

Taylor2
operator/(const Taylor2 & a, const Taylor2 & b)
{
  const double q = a.value / b.value, inverse = 1.0 / b.value;
  return chain(
      a, b, q, inverse, -q * inverse, 0.0, -inverse * inverse, 2.0 * q * inverse * inverse);
}

Taylor2
power(const Taylor2 & a, const Taylor2 & b)
{
  const double av = a.value, p = b.value, v = power(av, p);
  const bool constant_exponent =
      std::all_of(b.gradient.begin(), b.gradient.end(), [](double g) { return g == 0.0; });
  if (constant_exponent)
  {
    // p a^(p-1) and p (p - 1) a^(p-2), defined for a negative base with an integer exponent.
    const double d1 = p == 0.0 ? 0.0 : p * power(av, p - 1.0);
    const double d2 = p == 0.0 || p == 1.0 ? 0.0 : p * (p - 1.0) * power(av, p - 2.0);
    return chain(a, v, d1, d2);
  }
  const double ln = std::log(av), below = power(av, p - 1.0);
  return chain(a,
               b,
               v,
               p * below,
               v * ln,
               p * (p - 1.0) * power(av, p - 2.0),
               below * (1.0 + p * ln),
               v * ln * ln);
}

Taylor2
unaryValue(int k, const Taylor2 & a)
{
  return chain(a,
               unary_functions[k].f(a.value),
               unaryDerivative(k, a.value),
               unarySecondDerivative(k, a.value));
}

Taylor2
binaryValue(int k, const Taylor2 & a, const Taylor2 & b)
{
  const std::string name = binary_functions[k].name;
  if (name == "pow")
    return power(a, b);
  if (name == "min")
    return a.value <= b.value ? a : b;
  if (name == "max")
    return a.value >= b.value ? a : b;
  const double x = a.value, y = b.value, r2 = x * x + y * y;
  if (name == "hypot")
  {
    const double v = std::hypot(x, y), v3 = v * v * v;
    return chain(a, b, v, x / v, y / v, y * y / v3, -x * y / v3, x * x / v3);
  }
  // atan2(a, b), whose first derivatives are b / (a^2 + b^2) and -a / (a^2 + b^2).
  const double r4 = r2 * r2;
  return chain(a,
               b,
               std::atan2(x, y),
               y / r2,
               -x / r2,
               -2.0 * x * y / r4,
               (x * x - y * y) / r4,
               2.0 * x * y / r4);
}

template <typename T>
T
run(const std::vector<Instruction> & program, const T * variables)
{
  std::array<T, kMaxStack> stack;
  int top = -1;
  for (const auto & ins : program)
  {
    switch (ins.op)
    {
    case Op::Constant:
      stack[++top] = T(ins.value);
      break;
    case Op::Variable:
      stack[++top] = variables[ins.argument];
      break;
    case Op::Negate:
      stack[top] = -stack[top];
      break;
    case Op::Add:
      stack[top - 1] = stack[top - 1] + stack[top];
      --top;
      break;
    case Op::Subtract:
      stack[top - 1] = stack[top - 1] - stack[top];
      --top;
      break;
    case Op::Multiply:
      stack[top - 1] = stack[top - 1] * stack[top];
      --top;
      break;
    case Op::Divide:
      stack[top - 1] = stack[top - 1] / stack[top];
      --top;
      break;
    case Op::Power:
      stack[top - 1] = power(stack[top - 1], stack[top]);
      --top;
      break;
    case Op::Less:
      stack[top - 1] = T(plainValue(stack[top - 1]) < plainValue(stack[top]));
      --top;
      break;
    case Op::Greater:
      stack[top - 1] = T(plainValue(stack[top - 1]) > plainValue(stack[top]));
      --top;
      break;
    case Op::LessEqual:
      stack[top - 1] = T(plainValue(stack[top - 1]) <= plainValue(stack[top]));
      --top;
      break;
    case Op::GreaterEqual:
      stack[top - 1] = T(plainValue(stack[top - 1]) >= plainValue(stack[top]));
      --top;
      break;
    case Op::Equal:
      stack[top - 1] = T(plainValue(stack[top - 1]) == plainValue(stack[top]));
      --top;
      break;
    case Op::NotEqual:
      stack[top - 1] = T(plainValue(stack[top - 1]) != plainValue(stack[top]));
      --top;
      break;
    case Op::Function1:
      stack[top] = unaryValue(ins.argument, stack[top]);
      break;
    case Op::Function2:
      stack[top - 1] = binaryValue(ins.argument, stack[top - 1], stack[top]);
      --top;
      break;
    case Op::Select:
    {
      const T b = stack[top--];
      const T a = stack[top--];
      stack[top] = plainValue(stack[top]) != 0.0 ? a : b;
      break;
    }
    }
  }
  return stack[0];
}
} // namespace

// ---- ParsedExpression ----------------------------------------------------
ParsedExpression::ParsedExpression(std::string expression, std::vector<std::string> variables)
    : _text(std::move(expression)), _variables(std::move(variables))
{
  for (std::size_t i = 0; i < _variables.size(); ++i)
    for (std::size_t j = 0; j < i; ++j)
      if (_variables[i] == _variables[j])
        throw InputError("The variable name '" + _variables[i] + "' is given twice.");
  _program = Compiler(_text, _variables).compile();
  _max_depth = depthOf(_program);
  if (_max_depth > kMaxStack)
    throw InputError("The expression \"" + _text + "\" is nested too deeply to evaluate.");
  _used.assign(_variables.size(), false);
  for (const auto & ins : _program)
    if (ins.op == Op::Variable)
      _used[ins.argument] = true;
}

double
ParsedExpression::value(const double * variables) const
{
  return run(_program, variables);
}

ADReal
ParsedExpression::value(const ADReal * variables) const
{
  return run(_program, variables);
}

double
ParsedExpression::termSize(const double * variables) const
{
  std::array<Bounded, 256> values;
  if (_variables.size() > values.size())
    throw std::logic_error("dualmesh: too many variables for the size of the terms.");
  for (std::size_t k = 0; k < _variables.size(); ++k)
    values[k] = Bounded(variables[k]);
  return run(_program, values.data()).size;
}

void
ParsedExpression::secondOrder(const double * variables,
                              double & value,
                              double * gradient,
                              double * hessian) const
{
  const std::size_t n = _variables.size();
  if (n > static_cast<std::size_t>(kOrder2))
    throw std::logic_error("dualmesh: second derivatives take at most " + std::to_string(kOrder2) +
                           " variables.");
  std::array<Taylor2, kOrder2> seeds;
  for (std::size_t k = 0; k < n; ++k)
  {
    seeds[k] = Taylor2(variables[k]);
    seeds[k].gradient[k] = 1.0;
  }
  const Taylor2 r = run(_program, seeds.data());
  value = r.value;
  for (std::size_t i = 0; i < n; ++i)
  {
    gradient[i] = r.gradient[i];
    for (std::size_t j = 0; j < n; ++j)
      hessian[i * n + j] = r.hessian[i * kOrder2 + j];
  }
}

int
ParsedExpression::index(const std::string & name) const
{
  for (std::size_t k = 0; k < _variables.size(); ++k)
    if (_variables[k] == name)
      return static_cast<int>(k);
  return -1;
}

// ---- ParsedFunction ------------------------------------------------------
ParsedFunction::ParsedFunction(std::string expression)
    : _expression(std::move(expression), {"x", "y", "z", "t"})
{
}

bool
ParsedFunction::isExpression(const std::string & text, std::string * why)
{
  try
  {
    ParsedFunction f(text);
    return true;
  }
  catch (const std::exception & e)
  {
    if (why)
      *why = e.what();
    return false;
  }
}

double
ParsedFunction::value(const Point & x, double t) const
{
  const double variables[4] = {x[0], x[1], x[2], t};
  return _expression.value(variables);
}

} // namespace dualmesh
