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

constexpr double kPi = 3.14159265358979323846;

// ---- the functions -------------------------------------------------------
// The two tables below are the one list of the functions of the language.  Python reads their
// names and numbers of arguments through ParsedExpression::functions(), and its model language
// builds its calls of the same names from them.

using Unary = double (*)(double);

double
zero(double)
{
  return 0.0;
}

/// A function of one argument, with its first and second derivatives.  Every number type below
/// carries its derivatives, or the size of its terms, through the function from these by the
/// chain rule.  floor, ceil and sign are piecewise constant and abs is piecewise linear, so the
/// derivatives they leave out are zero.
struct UnaryFunction
{
  const char * name;
  Unary value;
  Unary first = zero;
  Unary second = zero;
};

const UnaryFunction unary_functions[] = {
    {"sin",
     [](double a) { return std::sin(a); },
     [](double a) { return std::cos(a); },
     [](double a) { return -std::sin(a); }},
    {"cos",
     [](double a) { return std::cos(a); },
     [](double a) { return -std::sin(a); },
     [](double a) { return -std::cos(a); }},
    {"tan",
     [](double a) { return std::tan(a); },
     [](double a) { return 1.0 / (std::cos(a) * std::cos(a)); },
     [](double a) { return 2.0 * std::tan(a) / (std::cos(a) * std::cos(a)); }},
    {"asin",
     [](double a) { return std::asin(a); },
     [](double a) { return 1.0 / std::sqrt(1.0 - a * a); },
     [](double a) { return a / std::pow(1.0 - a * a, 1.5); }},
    {"acos",
     [](double a) { return std::acos(a); },
     [](double a) { return -1.0 / std::sqrt(1.0 - a * a); },
     [](double a) { return -a / std::pow(1.0 - a * a, 1.5); }},
    {"atan",
     [](double a) { return std::atan(a); },
     [](double a) { return 1.0 / (1.0 + a * a); },
     [](double a) { return -2.0 * a / ((1.0 + a * a) * (1.0 + a * a)); }},
    {"sinh",
     [](double a) { return std::sinh(a); },
     [](double a) { return std::cosh(a); },
     [](double a) { return std::sinh(a); }},
    {"cosh",
     [](double a) { return std::cosh(a); },
     [](double a) { return std::sinh(a); },
     [](double a) { return std::cosh(a); }},
    {"tanh",
     [](double a) { return std::tanh(a); },
     [](double a) { return 1.0 - std::tanh(a) * std::tanh(a); },
     [](double a) { return -2.0 * std::tanh(a) * (1.0 - std::tanh(a) * std::tanh(a)); }},
    {"exp",
     [](double a) { return std::exp(a); },
     [](double a) { return std::exp(a); },
     [](double a) { return std::exp(a); }},
    {"log",
     [](double a) { return std::log(a); },
     [](double a) { return 1.0 / a; },
     [](double a) { return -1.0 / (a * a); }},
    {"log10",
     [](double a) { return std::log10(a); },
     [](double a) { return 1.0 / (a * std::log(10.0)); },
     [](double a) { return -1.0 / (a * a * std::log(10.0)); }},
    {"sqrt",
     [](double a) { return std::sqrt(a); },
     [](double a) { return 0.5 / std::sqrt(a); },
     [](double a) { return -0.25 / (a * std::sqrt(a)); }},
    {"abs", [](double a) { return std::abs(a); }, signOf},
    {"floor", [](double a) { return std::floor(a); }},
    {"ceil", [](double a) { return std::ceil(a); }},
    {"erf",
     [](double a) { return std::erf(a); },
     [](double a) { return 2.0 / std::sqrt(kPi) * std::exp(-a * a); },
     [](double a) { return -4.0 * a / std::sqrt(kPi) * std::exp(-a * a); }},
    {"sign", signOf},
};

/// The functions of two arguments.  Each number type evaluates them by its own rules below.
enum class Binary
{
  Atan2,
  Pow,
  Hypot,
  Min,
  Max
};

struct BinaryFunction
{
  const char * name;
  Binary function;
};

const BinaryFunction binary_functions[] = {
    {"atan2", Binary::Atan2},
    {"pow", Binary::Pow},
    {"hypot", Binary::Hypot},
    {"min", Binary::Min},
    {"max", Binary::Max},
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
        emit(Op::Constant, 0, kPi);
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

// ---- the number types and their chain rules -------------------------------
// Each number type takes a function of one argument from its row of unary_functions, and those
// of two arguments by the rules of binaryValue() at the end of this section.

/// The value and the partial derivatives of a function f(a, b) at a point: first f_a and f_b,
/// second f_aa, f_ab and f_bb.
struct Partials
{
  double value, a, b, aa, ab, bb;
};

/// atan2(x, y), whose first derivatives are y / (x^2 + y^2) and -x / (x^2 + y^2).
Partials
atan2Partials(double x, double y)
{
  const double r2 = x * x + y * y, r4 = r2 * r2;
  return {
      std::atan2(x, y), y / r2, -x / r2, -2.0 * x * y / r4, (x * x - y * y) / r4, 2.0 * x * y / r4};
}

// ---- double ----
double
unaryValue(const UnaryFunction & f, double a)
{
  return f.value(a);
}

double
binaryValue(Binary f, double a, double b)
{
  switch (f)
  {
  case Binary::Atan2:
    return std::atan2(a, b);
  case Binary::Pow:
    return std::pow(a, b);
  case Binary::Hypot:
    return std::hypot(a, b);
  case Binary::Min:
    return std::min(a, b);
  case Binary::Max:
    return std::max(a, b);
  }
  throw std::logic_error("dualmesh: unknown function of two arguments.");
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

// ---- ADReal ----
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
unaryValue(const UnaryFunction & f, const ADReal & a)
{
  return chain(a, f.value(a.value()), f.first(a.value()));
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

ADReal
atan2Of(const ADReal & a, const ADReal & b)
{
  const double av = a.value(), bv = b.value();
  const Partials f = atan2Partials(av, bv);
  return (a - av) * f.a + (b - bv) * f.b + f.value;
}

ADReal
hypotOf(const ADReal & a, const ADReal & b)
{
  return sqrt(a * a + b * b);
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
unaryValue(const UnaryFunction & f, const Bounded & a)
{
  return carried(f.value(a.value), f.first(a.value), a);
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

/// atan2(a, b), whose derivatives are b / (a^2 + b^2) and -a / (a^2 + b^2).
Bounded
atan2Of(const Bounded & a, const Bounded & b)
{
  const double v = std::atan2(a.value, b.value);
  const double r2 = a.value * a.value + b.value * b.value;
  return {v,
          std::abs(v) +
              (r2 > 0.0 ? (std::abs(b.value) * a.size + std::abs(a.value) * b.size) / r2 : 0.0)};
}

Bounded
hypotOf(const Bounded & a, const Bounded & b)
{
  const double v = std::hypot(a.value, b.value);
  return {v, v > 0.0 ? v + (std::abs(a.value) * a.size + std::abs(b.value) * b.size) / v : 0.0};
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

/// f(a, b) from its value and its partial derivatives @p f at (a, b), by the chain rule to second
/// order.
Taylor2
chain(const Taylor2 & a, const Taylor2 & b, const Partials & f)
{
  Taylor2 r(f.value);
  for (int i = 0; i < kOrder2; ++i)
  {
    r.gradient[i] = scaled(a.gradient[i], f.a) + scaled(b.gradient[i], f.b);
    for (int j = 0; j < kOrder2; ++j)
    {
      const int k = i * kOrder2 + j;
      r.hessian[k] = scaled(a.hessian[k], f.a) + scaled(b.hessian[k], f.b) +
                     scaled(a.gradient[i] * a.gradient[j], f.aa) +
                     scaled(a.gradient[i] * b.gradient[j] + b.gradient[i] * a.gradient[j], f.ab) +
                     scaled(b.gradient[i] * b.gradient[j], f.bb);
    }
  }
  return r;
}

/// f(a) from its value v and its first and second derivatives d1 and d2 at a.
Taylor2
chain(const Taylor2 & a, double v, double d1, double d2)
{
  return chain(a, Taylor2(), {v, d1, 0.0, d2, 0.0, 0.0});
}

Taylor2
operator-(const Taylor2 & a)
{
  return chain(a, -a.value, -1.0, 0.0);
}

Taylor2
operator+(const Taylor2 & a, const Taylor2 & b)
{
  return chain(a, b, {a.value + b.value, 1.0, 1.0, 0.0, 0.0, 0.0});
}

Taylor2
operator-(const Taylor2 & a, const Taylor2 & b)
{
  return chain(a, b, {a.value - b.value, 1.0, -1.0, 0.0, 0.0, 0.0});
}

Taylor2
operator*(const Taylor2 & a, const Taylor2 & b)
{
  return chain(a, b, {a.value * b.value, b.value, a.value, 0.0, 1.0, 0.0});
}

Taylor2
operator/(const Taylor2 & a, const Taylor2 & b)
{
  const double q = a.value / b.value, inverse = 1.0 / b.value;
  return chain(
      a, b, {q, inverse, -q * inverse, 0.0, -inverse * inverse, 2.0 * q * inverse * inverse});
}

Taylor2
unaryValue(const UnaryFunction & f, const Taylor2 & a)
{
  return chain(a, f.value(a.value), f.first(a.value), f.second(a.value));
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
               {v,
                p * below,
                v * ln,
                p * (p - 1.0) * power(av, p - 2.0),
                below * (1.0 + p * ln),
                v * ln * ln});
}

Taylor2
atan2Of(const Taylor2 & a, const Taylor2 & b)
{
  return chain(a, b, atan2Partials(a.value, b.value));
}

Taylor2
hypotOf(const Taylor2 & a, const Taylor2 & b)
{
  const double x = a.value, y = b.value, v = std::hypot(x, y), v3 = v * v * v;
  return chain(a, b, {v, x / v, y / v, y * y / v3, -x * y / v3, x * x / v3});
}

// ---- the functions of two arguments ----
/// A function of two arguments of a number type that carries derivatives or sizes (ADReal,
/// Bounded and Taylor2): pow is the power of the type, min and max choose an argument, and atan2
/// and hypot take the rules of the type above.
template <typename T>
T
binaryValue(Binary f, const T & a, const T & b)
{
  switch (f)
  {
  case Binary::Pow:
    return power(a, b);
  case Binary::Min:
    return plainValue(a) <= plainValue(b) ? a : b;
  case Binary::Max:
    return plainValue(a) >= plainValue(b) ? a : b;
  case Binary::Atan2:
    return atan2Of(a, b);
  case Binary::Hypot:
    return hypotOf(a, b);
  }
  throw std::logic_error("dualmesh: unknown function of two arguments.");
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
      stack[top] = unaryValue(unary_functions[ins.argument], stack[top]);
      break;
    case Op::Function2:
      stack[top - 1] =
          binaryValue(binary_functions[ins.argument].function, stack[top - 1], stack[top]);
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

std::vector<std::pair<std::string, int>>
ParsedExpression::functions()
{
  std::vector<std::pair<std::string, int>> table;
  for (const auto & f : unary_functions)
    table.emplace_back(f.name, 1);
  for (const auto & f : binary_functions)
    table.emplace_back(f.name, 2);
  return table;
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
