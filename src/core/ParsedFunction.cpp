// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/core/ParsedFunction.h"

#include "dualmesh/core/ADReal.h"

#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <sstream>

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
    {"Abs", [](double v) { return std::abs(v); }},
    {"fabs", [](double v) { return std::abs(v); }},
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
      if (name == "e" || name == "E")
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
  if (name == "abs" || name == "Abs" || name == "fabs")
    return signOf(a);
  if (name == "erf")
    return 2.0 / std::sqrt(3.14159265358979323846) * std::exp(-a * a);
  return 0.0; // floor, ceil, sign: piecewise constant
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
