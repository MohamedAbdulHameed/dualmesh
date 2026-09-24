// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/core/ParsedFunction.h"

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
using Op = ParsedFunction::Op;
using Instruction = ParsedFunction::Instruction;

double
signOf(double v)
{
  return (v > 0) - (v < 0);
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
  explicit Compiler(const std::string & text) : _s(text) {}

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
        call(name);
        return;
      }
      if (name == "x" || name == "y" || name == "z" || name == "t")
      {
        emit(Op::Variable, name == "x" ? 0 : name == "y" ? 1 : name == "z" ? 2 : 3);
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
      fail("unknown name '" + name + "' (the variables are x, y, z and t)");
    }
    fail("unexpected '" + std::string(1, c) + "'");
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
} // namespace

ParsedFunction::ParsedFunction(std::string expression) : _text(std::move(expression))
{
  _program = Compiler(_text).compile();
  _max_depth = depthOf(_program);
  if (_max_depth > kMaxStack)
    throw InputError("The expression \"" + _text + "\" is nested too deeply to evaluate.");
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
  std::array<double, kMaxStack> stack;
  int top = -1;
  const double variables[4] = {x[0], x[1], x[2], t};
  for (const auto & ins : _program)
  {
    switch (ins.op)
    {
    case Op::Constant:
      stack[++top] = ins.value;
      break;
    case Op::Variable:
      stack[++top] = variables[ins.argument];
      break;
    case Op::Negate:
      stack[top] = -stack[top];
      break;
    case Op::Add:
      stack[top - 1] += stack[top];
      --top;
      break;
    case Op::Subtract:
      stack[top - 1] -= stack[top];
      --top;
      break;
    case Op::Multiply:
      stack[top - 1] *= stack[top];
      --top;
      break;
    case Op::Divide:
      stack[top - 1] /= stack[top];
      --top;
      break;
    case Op::Power:
    {
      // An integer exponent is common in manufactured solutions (x**2) and
      // is both faster and exact as repeated multiplication.
      const double b = stack[top--];
      double & a = stack[top];
      if (b == std::floor(b) && std::abs(b) <= 8)
      {
        const int n = static_cast<int>(b);
        double r = 1;
        for (int k = 0; k < std::abs(n); ++k)
          r *= a;
        a = n >= 0 ? r : 1.0 / r;
      }
      else
        a = std::pow(a, b);
      break;
    }
    case Op::Less:
      stack[top - 1] = stack[top - 1] < stack[top];
      --top;
      break;
    case Op::Greater:
      stack[top - 1] = stack[top - 1] > stack[top];
      --top;
      break;
    case Op::LessEqual:
      stack[top - 1] = stack[top - 1] <= stack[top];
      --top;
      break;
    case Op::GreaterEqual:
      stack[top - 1] = stack[top - 1] >= stack[top];
      --top;
      break;
    case Op::Equal:
      stack[top - 1] = stack[top - 1] == stack[top];
      --top;
      break;
    case Op::NotEqual:
      stack[top - 1] = stack[top - 1] != stack[top];
      --top;
      break;
    case Op::Function1:
      stack[top] = unary_functions[ins.argument].f(stack[top]);
      break;
    case Op::Function2:
      stack[top - 1] = binary_functions[ins.argument].f(stack[top - 1], stack[top]);
      --top;
      break;
    case Op::Select:
    {
      const double b = stack[top--];
      const double a = stack[top--];
      stack[top] = stack[top] != 0.0 ? a : b;
      break;
    }
    }
  }
  return stack[0];
}

} // namespace dualmesh
