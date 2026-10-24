// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/core/Equation.h"

#include "dualmesh/core/InputParameters.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <sstream>

namespace dualmesh
{

namespace
{

// ---------------------------------------------------------------------------
// Text: tokens and the tree of an equation as written.
// ---------------------------------------------------------------------------
struct Token
{
  enum Kind
  {
    Number,
    Name,
    Op,
    End
  } kind;
  std::string text;
  double value = 0.0;
  std::size_t pos = 0;
};

std::vector<Token>
tokenize(const std::string & s, const std::string & what)
{
  std::vector<Token> out;
  std::size_t i = 0;
  while (i < s.size())
  {
    const char c = s[i];
    if (std::isspace(static_cast<unsigned char>(c)))
    {
      ++i;
      continue;
    }
    if (std::isdigit(static_cast<unsigned char>(c)) ||
        (c == '.' && i + 1 < s.size() && std::isdigit(static_cast<unsigned char>(s[i + 1]))))
    {
      const char * begin = s.c_str() + i;
      char * end = nullptr;
      const double v = std::strtod(begin, &end);
      const std::size_t n = static_cast<std::size_t>(end - begin);
      out.push_back({Token::Number, s.substr(i, n), v, i});
      i += n;
      continue;
    }
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_')
    {
      std::size_t j = i;
      while (j < s.size() && (std::isalnum(static_cast<unsigned char>(s[j])) || s[j] == '_'))
        ++j;
      out.push_back({Token::Name, s.substr(i, j - i), 0.0, i});
      i = j;
      continue;
    }
    static const char * two[] = {"**", "<=", ">=", "==", "!="};
    bool matched = false;
    for (const char * t : two)
      if (s.compare(i, 2, t) == 0)
      {
        out.push_back({Token::Op, t, 0.0, i});
        i += 2;
        matched = true;
        break;
      }
    if (matched)
      continue;
    if (std::string("+-*/^()[],<>=").find(c) != std::string::npos)
    {
      out.push_back({Token::Op, std::string(1, c), 0.0, i});
      ++i;
      continue;
    }
    throw InputError(what + ": the character '" + std::string(1, c) + "' at position " +
                     std::to_string(i + 1) + " is not part of the equation language.");
  }
  out.push_back({Token::End, "", 0.0, s.size()});
  return out;
}

struct Ast;
using AstPtr = std::shared_ptr<const Ast>;

struct Ast
{
  enum Kind
  {
    Number,
    Name,
    Call,
    Neg,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Compare,
    And,
    Or,
    List,
    Derivative
  } kind;
  double number = 0.0;
  std::string name;       ///< Name, Call, Compare operator, Derivative field
  std::string coordinate; ///< Derivative: t, x, y, z, r or n
  int order = 1;
  std::vector<AstPtr> args;
};

AstPtr
node(Ast::Kind kind, std::vector<AstPtr> args = {}, std::string name = "")
{
  auto a = std::make_shared<Ast>();
  a->kind = kind;
  a->args = std::move(args);
  a->name = std::move(name);
  return a;
}

/// Recursive descent over the tokens.  Names of definitions are replaced by the tree of their
/// text, so that the rules that sort the terms see through them.
class Parser
{
public:
  Parser(const std::string & text,
         std::string what,
         std::function<bool(const std::string &)> is_field,
         const std::map<std::string, AstPtr> & definitions,
         const std::set<std::string> & later)
      : _text(text), _what(std::move(what)), _tokens(tokenize(text, _what)),
        _is_field(std::move(is_field)), _definitions(definitions), _later(later)
  {
  }

  AstPtr expression()
  {
    AstPtr a = parseOr();
    expectEnd();
    return a;
  }

  /// "lhs = rhs": the pair.
  std::pair<AstPtr, AstPtr> equation()
  {
    AstPtr lhs = parseOr();
    if (!(peek().kind == Token::Op && peek().text == "="))
      fail("an equation needs '=' between its two sides");
    ++_k;
    AstPtr rhs = parseOr();
    expectEnd();
    return {lhs, rhs};
  }

private:
  const Token & peek(std::size_t ahead = 0) const
  {
    return _tokens[std::min(_k + ahead, _tokens.size() - 1)];
  }
  bool isOp(const char * op, std::size_t ahead = 0) const
  {
    return peek(ahead).kind == Token::Op && peek(ahead).text == op;
  }
  bool isWord(const char * word) const { return peek().kind == Token::Name && peek().text == word; }
  [[noreturn]] void fail(const std::string & why) const
  {
    throw InputError(_what + ": " + why + ", at position " + std::to_string(peek().pos + 1) +
                     " of '" + _text + "'.");
  }
  void expectEnd() const
  {
    if (peek().kind != Token::End)
      fail("'" + peek().text + "' is not expected here");
  }

  AstPtr parseOr()
  {
    AstPtr a = parseAnd();
    while (isWord("or"))
    {
      ++_k;
      a = node(Ast::Or, {a, parseAnd()});
    }
    return a;
  }
  AstPtr parseAnd()
  {
    AstPtr a = parseCompare();
    while (isWord("and"))
    {
      ++_k;
      a = node(Ast::And, {a, parseCompare()});
    }
    return a;
  }
  AstPtr parseCompare()
  {
    AstPtr a = parseSum();
    for (const char * op : {"<", ">", "<=", ">=", "==", "!="})
      if (isOp(op))
      {
        ++_k;
        return node(Ast::Compare, {a, parseSum()}, op);
      }
    return a;
  }
  AstPtr parseSum()
  {
    AstPtr a = parseProduct();
    while (isOp("+") || isOp("-"))
    {
      const bool plus = isOp("+");
      ++_k;
      a = node(plus ? Ast::Add : Ast::Sub, {a, parseProduct()});
    }
    return a;
  }
  AstPtr parseProduct()
  {
    AstPtr a = parseUnary();
    while (isOp("*") || isOp("/"))
    {
      const bool times = isOp("*");
      ++_k;
      a = node(times ? Ast::Mul : Ast::Div, {a, parseUnary()});
    }
    return a;
  }
  AstPtr parseUnary()
  {
    if (isOp("-"))
    {
      ++_k;
      return node(Ast::Neg, {parseUnary()});
    }
    if (isOp("+"))
    {
      ++_k;
      return parseUnary();
    }
    return parsePower();
  }
  AstPtr parsePower()
  {
    AstPtr a = parsePrimary();
    if (isOp("^") || isOp("**"))
    {
      ++_k;
      return node(Ast::Pow, {a, parseUnary()});
    }
    return a;
  }

  /// d<field>/d<coordinate> and d2<field>/d<coordinate>2, when <field> is a field.
  AstPtr derivative()
  {
    const Token & a = peek();
    if (a.kind != Token::Name || a.text.size() < 2 || a.text[0] != 'd' || !isOp("/", 1) ||
        peek(2).kind != Token::Name || peek(2).text.size() < 2 || peek(2).text[0] != 'd')
      return nullptr;
    const bool second = a.text.size() > 2 && a.text[1] == '2' && _is_field(a.text.substr(2));
    const std::string field = second ? a.text.substr(2) : a.text.substr(1);
    if (!_is_field(field))
      return nullptr;
    std::string coordinate = peek(2).text.substr(1);
    if (second)
    {
      if (coordinate.size() < 2 || coordinate.back() != '2')
        fail("write the second derivative as d2" + field + "/d" + coordinate + "2");
      coordinate.pop_back();
    }
    static const std::set<std::string> known{"t", "x", "y", "z", "r", "n"};
    if (!known.count(coordinate))
      fail("'" + a.text + "/" + peek(2).text + "' is not a derivative: the derivatives are d" +
           field + "/dt, d" + field + "/dx, d" + field + "/dy, d" + field + "/dz and d" + field +
           "/dn");
    if (second && coordinate != "t")
      fail("second derivatives in space are written with div, grad and laplacian, for example "
           "laplacian(" +
           field + ")");
    _k += 3;
    auto d = std::make_shared<Ast>();
    d->kind = Ast::Derivative;
    d->name = field;
    d->coordinate = coordinate;
    d->order = second ? 2 : 1;
    return d;
  }

  AstPtr parsePrimary()
  {
    const Token & t = peek();
    if (t.kind == Token::Number)
    {
      ++_k;
      auto a = std::make_shared<Ast>();
      a->kind = Ast::Number;
      a->number = t.value;
      return a;
    }
    if (isOp("("))
    {
      ++_k;
      AstPtr a = parseOr();
      if (!isOp(")"))
        fail("a ')' is missing");
      ++_k;
      return a;
    }
    if (isOp("["))
    {
      ++_k;
      std::vector<AstPtr> items{parseOr()};
      while (isOp(","))
      {
        ++_k;
        items.push_back(parseOr());
      }
      if (!isOp("]"))
        fail("a ']' is missing");
      ++_k;
      return node(Ast::List, items);
    }
    if (t.kind == Token::Name)
    {
      if (AstPtr d = derivative())
        return d;
      const std::string name = t.text;
      ++_k;
      if (isOp("("))
      {
        ++_k;
        std::vector<AstPtr> args;
        if (!isOp(")"))
        {
          args.push_back(parseOr());
          while (isOp(","))
          {
            ++_k;
            args.push_back(parseOr());
          }
        }
        if (!isOp(")"))
          fail("a ')' is missing after the arguments of " + name);
        ++_k;
        return node(Ast::Call, args, name);
      }
      if (const auto it = _definitions.find(name); it != _definitions.end())
        return it->second;
      if (_later.count(name))
        fail("'" + name +
             "' is defined after this expression (or is this expression itself); define a name "
             "before the expressions that use it");
      return node(Ast::Name, {}, name);
    }
    fail(t.kind == Token::End ? "the expression ends too early"
                              : "'" + t.text + "' is not expected here");
  }

  std::string _text;
  std::string _what;
  std::vector<Token> _tokens;
  std::size_t _k = 0;
  std::function<bool(const std::string &)> _is_field;
  const std::map<std::string, AstPtr> & _definitions;
  const std::set<std::string> & _later;
};

// ---------------------------------------------------------------------------
// Scalar expressions, built with simplification, differentiated and printed.
// ---------------------------------------------------------------------------
struct Sym;
using SymPtr = std::shared_ptr<const Sym>;

struct Sym
{
  enum Kind
  {
    Number,
    Field,      ///< name = the scalar field
    Gradient,   ///< name = the scalar field, axis
    Coordinate, ///< axis
    Normal,     ///< axis, the outward normal (boundary conditions only)
    Time,
    Constant, ///< a named constant, pi, e or eigenvalue
    External, ///< a name passed on: a function, a property, an element field
    Neg,
    Add,
    Mul,
    Div,
    Pow,
    Call,
    Compare,
    Select
  } kind;
  double number = 0.0;
  std::string name;
  int axis = 0;
  std::vector<SymPtr> args;
};

SymPtr
leaf(Sym::Kind kind, std::string name = "", int axis = 0)
{
  auto s = std::make_shared<Sym>();
  s->kind = kind;
  s->name = std::move(name);
  s->axis = axis;
  return s;
}
SymPtr
num(double v)
{
  auto s = std::make_shared<Sym>();
  s->kind = Sym::Number;
  s->number = v;
  return s;
}
SymPtr
op(Sym::Kind kind, std::vector<SymPtr> args, std::string name = "")
{
  auto s = std::make_shared<Sym>();
  s->kind = kind;
  s->args = std::move(args);
  s->name = std::move(name);
  return s;
}
bool
isNumber(const SymPtr & s, double v)
{
  return s->kind == Sym::Number && s->number == v;
}
SymPtr
neg(const SymPtr & a)
{
  if (a->kind == Sym::Number)
    return num(-a->number);
  if (a->kind == Sym::Neg)
    return a->args[0];
  return op(Sym::Neg, {a});
}
SymPtr
add(const SymPtr & a, const SymPtr & b)
{
  if (isNumber(a, 0.0))
    return b;
  if (isNumber(b, 0.0))
    return a;
  if (a->kind == Sym::Number && b->kind == Sym::Number)
    return num(a->number + b->number);
  return op(Sym::Add, {a, b});
}
SymPtr
sub(const SymPtr & a, const SymPtr & b)
{
  return add(a, neg(b));
}
SymPtr
mul(const SymPtr & a, const SymPtr & b)
{
  if (isNumber(a, 0.0) || isNumber(b, 0.0))
    return num(0.0);
  if (isNumber(a, 1.0))
    return b;
  if (isNumber(b, 1.0))
    return a;
  if (isNumber(a, -1.0))
    return neg(b);
  if (isNumber(b, -1.0))
    return neg(a);
  if (a->kind == Sym::Number && b->kind == Sym::Number)
    return num(a->number * b->number);
  if (a->kind == Sym::Neg)
    return neg(mul(a->args[0], b));
  if (b->kind == Sym::Neg)
    return neg(mul(a, b->args[0]));
  return op(Sym::Mul, {a, b});
}
SymPtr
divide(const SymPtr & a, const SymPtr & b)
{
  if (isNumber(a, 0.0))
    return num(0.0);
  if (isNumber(b, 1.0))
    return a;
  if (a->kind == Sym::Number && b->kind == Sym::Number && b->number != 0.0)
    return num(a->number / b->number);
  if (a->kind == Sym::Neg)
    return neg(divide(a->args[0], b));
  return op(Sym::Div, {a, b});
}
SymPtr
power(const SymPtr & a, const SymPtr & b)
{
  if (isNumber(b, 0.0))
    return num(1.0);
  if (isNumber(b, 1.0))
    return a;
  if (a->kind == Sym::Number && b->kind == Sym::Number)
    return num(std::pow(a->number, b->number));
  return op(Sym::Pow, {a, b});
}
SymPtr
call(const std::string & f, std::vector<SymPtr> args)
{
  return op(Sym::Call, std::move(args), f);
}

/// Whether @p s contains a node for which @p test holds.
bool
contains(const SymPtr & s, const std::function<bool(const Sym &)> & test)
{
  if (test(*s))
    return true;
  return std::any_of(
      s->args.begin(), s->args.end(), [&](const SymPtr & a) { return contains(a, test); });
}
bool
variesInSpace(const SymPtr & s)
{
  return contains(s,
                  [](const Sym & n)
                  {
                    return n.kind == Sym::Field || n.kind == Sym::Gradient ||
                           n.kind == Sym::Coordinate || n.kind == Sym::External ||
                           n.kind == Sym::Normal;
                  });
}

/// The derivative of @p s with respect to the coordinate @p axis.
SymPtr
derivative(const SymPtr & s, int axis, const std::string & what)
{
  const auto d = [&](const SymPtr & a) { return derivative(a, axis, what); };
  switch (s->kind)
  {
  case Sym::Number:
  case Sym::Time:
  case Sym::Constant:
  case Sym::Normal:
  case Sym::Compare:
    return num(0.0);
  case Sym::Coordinate:
    return num(s->axis == axis ? 1.0 : 0.0);
  case Sym::Field:
  {
    auto g = leaf(Sym::Gradient, s->name, axis);
    return g;
  }
  case Sym::Gradient:
    throw InputError(what + ": this needs a second derivative of " + s->name +
                     " outside a divergence. Second derivatives are allowed only as "
                     "div(... grad(...)) or laplacian(...), which are integrated by parts; "
                     "a fourth-order equation is written as two second-order ones.");
  case Sym::External:
    throw InputError(what + ": '" + s->name +
                     "' would be differentiated, and it is not a field, a coordinate or a "
                     "definition. Write it as a definition of the coordinates and the fields, "
                     "or keep it outside derivatives.");
  case Sym::Neg:
    return neg(d(s->args[0]));
  case Sym::Add:
    return add(d(s->args[0]), d(s->args[1]));
  case Sym::Mul:
    return add(mul(d(s->args[0]), s->args[1]), mul(s->args[0], d(s->args[1])));
  case Sym::Div:
  {
    const SymPtr & a = s->args[0];
    const SymPtr & b = s->args[1];
    return divide(sub(mul(d(a), b), mul(a, d(b))), power(b, num(2.0)));
  }
  case Sym::Pow:
  {
    const SymPtr & a = s->args[0];
    const SymPtr & b = s->args[1];
    const SymPtr db = d(b);
    if (isNumber(db, 0.0))
      return mul(mul(b, power(a, sub(b, num(1.0)))), d(a));
    return mul(s, add(mul(db, call("log", {a})), divide(mul(b, d(a)), a)));
  }
  case Sym::Select:
    return op(Sym::Select, {s->args[0], d(s->args[1]), d(s->args[2])});
  case Sym::Call:
  {
    const std::string & f = s->name;
    const SymPtr & a = s->args[0];
    if (s->args.size() == 1)
    {
      const SymPtr da = d(a);
      if (f == "floor" || f == "ceil" || f == "sign")
        return num(0.0);
      SymPtr outer;
      if (f == "sin")
        outer = call("cos", {a});
      else if (f == "cos")
        outer = neg(call("sin", {a}));
      else if (f == "tan")
        outer = divide(num(1.0), power(call("cos", {a}), num(2.0)));
      else if (f == "asin")
        outer = divide(num(1.0), call("sqrt", {sub(num(1.0), power(a, num(2.0)))}));
      else if (f == "acos")
        outer = neg(divide(num(1.0), call("sqrt", {sub(num(1.0), power(a, num(2.0)))})));
      else if (f == "atan")
        outer = divide(num(1.0), add(num(1.0), power(a, num(2.0))));
      else if (f == "sinh")
        outer = call("cosh", {a});
      else if (f == "cosh")
        outer = call("sinh", {a});
      else if (f == "tanh")
        outer = sub(num(1.0), power(call("tanh", {a}), num(2.0)));
      else if (f == "exp")
        outer = s;
      else if (f == "log")
        outer = divide(num(1.0), a);
      else if (f == "log10")
        outer = divide(num(1.0), mul(a, call("log", {num(10.0)})));
      else if (f == "sqrt")
        outer = divide(num(1.0), mul(num(2.0), s));
      else if (f == "abs")
        outer = call("sign", {a});
      else if (f == "erf")
        outer = mul(divide(num(2.0), call("sqrt", {leaf(Sym::Constant, "pi")})),
                    call("exp", {neg(power(a, num(2.0)))}));
      else
        throw InputError(what + ": the function " + f + " cannot be differentiated.");
      return mul(outer, da);
    }
    const SymPtr & b = s->args[1];
    if (f == "pow")
      return d(power(a, b));
    if (f == "min" || f == "max")
      return op(Sym::Select, {op(Sym::Compare, {a, b}, f == "min" ? "<" : ">"), d(a), d(b)});
    if (f == "atan2")
      return divide(sub(mul(b, d(a)), mul(a, d(b))), add(power(a, num(2.0)), power(b, num(2.0))));
    if (f == "hypot")
      return divide(add(mul(a, d(a)), mul(b, d(b))), s);
    throw InputError(what + ": the function " + f + " cannot be differentiated.");
  }
  }
  return num(0.0);
}

std::string
numberText(double v)
{
  if (v == std::floor(v) && std::abs(v) < 1e15)
  {
    std::ostringstream os;
    os << static_cast<long long>(v);
    return os.str();
  }
  for (int precision : {15, 16, 17})
  {
    std::ostringstream os;
    os.precision(precision);
    os << v;
    if (std::strtod(os.str().c_str(), nullptr) == v)
      return os.str();
  }
  return std::to_string(v);
}

const char * kAxes = "xyz";

/// The text of @p s in the scalar expression language.
std::string
print(const SymPtr & s, int parent = 0)
{
  const auto wrap = [&](int own, const std::string & text)
  { return own < parent ? "(" + text + ")" : text; };
  switch (s->kind)
  {
  case Sym::Number:
    return s->number < 0 ? wrap(4, numberText(s->number)) : numberText(s->number);
  case Sym::Field:
  case Sym::Constant:
  case Sym::External:
    return s->name;
  case Sym::Gradient:
    return std::string("grad_") + kAxes[s->axis] + "(" + s->name + ")";
  case Sym::Coordinate:
    return std::string(1, kAxes[s->axis]);
  case Sym::Normal:
    return std::string("n_") + kAxes[s->axis];
  case Sym::Time:
    return "t";
  case Sym::Neg:
  {
    // -a*b reads as (-a)*b, which has the same value, so a negated product needs no
    // parentheses; it binds as a product.
    const bool product = s->args[0]->kind == Sym::Mul || s->args[0]->kind == Sym::Div;
    return product ? wrap(3, "-" + print(s->args[0], 3)) : wrap(4, "-" + print(s->args[0], 4));
  }
  case Sym::Add:
  {
    const SymPtr & b = s->args[1];
    if (b->kind == Sym::Neg)
      return wrap(2, print(s->args[0], 2) + " - " + print(b->args[0], 3));
    if (b->kind == Sym::Number && b->number < 0)
      return wrap(2, print(s->args[0], 2) + " - " + numberText(-b->number));
    return wrap(2, print(s->args[0], 2) + " + " + print(b, 2));
  }
  case Sym::Mul:
    return wrap(3, print(s->args[0], 3) + "*" + print(s->args[1], 3));
  case Sym::Div:
    return wrap(3, print(s->args[0], 3) + "/" + print(s->args[1], 4));
  case Sym::Pow:
    return wrap(5, print(s->args[0], 6) + "^" + print(s->args[1], 5));
  case Sym::Compare:
    return wrap(1, print(s->args[0], 2) + " " + s->name + " " + print(s->args[1], 2));
  case Sym::Select:
    return "if(" + print(s->args[0]) + ", " + print(s->args[1]) + ", " + print(s->args[2]) + ")";
  case Sym::Call:
  {
    std::string text = s->name + "(";
    for (std::size_t k = 0; k < s->args.size(); ++k)
      text += (k ? ", " : "") + print(s->args[k]);
    return text + ")";
  }
  }
  return "";
}

/// The value of @p s with every name replaced by a number of @p sample.
double
evaluate(const SymPtr & s, const std::function<double(const Sym &)> & sample)
{
  const auto v = [&](std::size_t k) { return evaluate(s->args[k], sample); };
  switch (s->kind)
  {
  case Sym::Number:
    return s->number;
  case Sym::Field:
  case Sym::Gradient:
  case Sym::Coordinate:
  case Sym::Normal:
  case Sym::Time:
  case Sym::Constant:
  case Sym::External:
    return sample(*s);
  case Sym::Neg:
    return -v(0);
  case Sym::Add:
    return v(0) + v(1);
  case Sym::Mul:
    return v(0) * v(1);
  case Sym::Div:
    return v(0) / v(1);
  case Sym::Pow:
    return std::pow(v(0), v(1));
  case Sym::Select:
    return v(0) != 0.0 ? v(1) : v(2);
  case Sym::Compare:
  {
    const double a = v(0), b = v(1);
    const std::string & o = s->name;
    return (o == "<"    ? a < b
            : o == ">"  ? a > b
            : o == "<=" ? a <= b
            : o == ">=" ? a >= b
            : o == "==" ? a == b
                        : a != b)
               ? 1.0
               : 0.0;
  }
  case Sym::Call:
  {
    static const std::map<std::string, double (*)(double)> one{{"sin", std::sin},
                                                               {"cos", std::cos},
                                                               {"tan", std::tan},
                                                               {"asin", std::asin},
                                                               {"acos", std::acos},
                                                               {"atan", std::atan},
                                                               {"sinh", std::sinh},
                                                               {"cosh", std::cosh},
                                                               {"tanh", std::tanh},
                                                               {"exp", std::exp},
                                                               {"log", std::log},
                                                               {"log10", std::log10},
                                                               {"sqrt", std::sqrt},
                                                               {"abs", std::fabs},
                                                               {"floor", std::floor},
                                                               {"ceil", std::ceil},
                                                               {"erf", std::erf}};
    if (const auto it = one.find(s->name); it != one.end())
      return it->second(v(0));
    if (s->name == "sign")
      return v(0) > 0 ? 1.0 : v(0) < 0 ? -1.0 : 0.0;
    if (s->name == "pow")
      return std::pow(v(0), v(1));
    if (s->name == "min")
      return std::min(v(0), v(1));
    if (s->name == "max")
      return std::max(v(0), v(1));
    if (s->name == "atan2")
      return std::atan2(v(0), v(1));
    if (s->name == "hypot")
      return std::hypot(v(0), v(1));
    return 0.0;
  }
  }
  return 0.0;
}

// ---------------------------------------------------------------------------
// Values: scalars, vectors and tensors of scalar expressions.
// ---------------------------------------------------------------------------
struct Value
{
  int rank = 0; ///< 0 scalar, 1 vector, 2 tensor (row by row)
  std::vector<SymPtr> c;
};

Value
scalar(SymPtr s)
{
  return {0, {std::move(s)}};
}

const char *
rankName(int rank)
{
  return rank == 0 ? "a scalar" : rank == 1 ? "a vector" : "a tensor";
}

/// Evaluates trees to values in the scope of one physics.
class Evaluator
{
public:
  Evaluator(const EquationScope & scope, std::string what) : _scope(scope), _what(std::move(what))
  {
    _n = scope.dimension;
    for (const auto & [field, rank] : scope.fields)
    {
      _rank[field] = rank;
      if (rank == 1)
        for (int i = 0; i < _n; ++i)
          _component[component(field, i)] = {field, i};
    }
  }

  int dimension() const { return _n; }

  std::string component(const std::string & field, int i) const
  {
    static const char * cartesian[] = {"x", "y", "z"};
    static const char * axisymmetric[] = {"r", "z"};
    return field + "_" + (_scope.axisymmetric ? axisymmetric[i] : cartesian[i]);
  }

  /// Whether @p name is a field or a component of a vector field.
  bool isField(const std::string & name) const
  {
    return _rank.count(name) || _component.count(name);
  }

  /// The field that @p name (a scalar field or a component) belongs to.
  std::string fieldOf(const std::string & name) const
  {
    const auto it = _component.find(name);
    return it == _component.end() ? name : it->second.first;
  }

  Value eval(const AstPtr & a)
  {
    switch (a->kind)
    {
    case Ast::Number:
      return scalar(num(a->number));
    case Ast::Name:
      return name(a->name);
    case Ast::Neg:
      return map(eval(a->args[0]), [](const SymPtr & s) { return neg(s); });
    case Ast::Add:
    case Ast::Sub:
    {
      Value x = eval(a->args[0]), y = eval(a->args[1]);
      if (x.rank != y.rank)
        fail(std::string("'+' and '-' join terms of one kind, and here ") + rankName(x.rank) +
             " meets " + rankName(y.rank));
      for (std::size_t k = 0; k < x.c.size(); ++k)
        x.c[k] = a->kind == Ast::Add ? add(x.c[k], y.c[k]) : sub(x.c[k], y.c[k]);
      return x;
    }
    case Ast::Mul:
    {
      Value x = eval(a->args[0]), y = eval(a->args[1]);
      if (x.rank && y.rank)
        fail(std::string("'*' multiplies by a scalar; for ") + rankName(x.rank) + " and " +
             rankName(y.rank) + " use dot, ddot, cross or outer");
      if (x.rank)
        std::swap(x, y);
      return map(y, [&](const SymPtr & s) { return mul(x.c[0], s); });
    }
    case Ast::Div:
    {
      Value x = eval(a->args[0]), y = eval(a->args[1]);
      if (y.rank)
        fail("'/' divides by a scalar only");
      return map(x, [&](const SymPtr & s) { return divide(s, y.c[0]); });
    }
    case Ast::Pow:
    {
      Value x = eval(a->args[0]), y = eval(a->args[1]);
      if (x.rank || y.rank)
        fail("'^' takes scalars; for the square of a vector use dot(v, v)");
      return scalar(power(x.c[0], y.c[0]));
    }
    case Ast::Compare:
    {
      Value x = eval(a->args[0]), y = eval(a->args[1]);
      if (x.rank || y.rank)
        fail("comparisons take scalars");
      return scalar(op(Sym::Compare, {x.c[0], y.c[0]}, a->name));
    }
    case Ast::And:
    case Ast::Or:
    {
      Value x = eval(a->args[0]), y = eval(a->args[1]);
      if (x.rank || y.rank)
        fail("'and' and 'or' take conditions");
      const SymPtr p = op(Sym::Compare, {x.c[0], num(0.0)}, "!=");
      const SymPtr q = op(Sym::Compare, {y.c[0], num(0.0)}, "!=");
      return scalar(a->kind == Ast::And ? mul(p, q) : call("max", {p, q}));
    }
    case Ast::List:
      return list(a);
    case Ast::Derivative:
      return derivativeOf(a);
    case Ast::Call:
      return function(a);
    }
    return scalar(num(0.0));
  }

  [[noreturn]] void fail(const std::string & why) const
  {
    throw InputError(_what + ": " + why + ".");
  }

  /// The gradient of @p v: a vector of a scalar, a tensor (d v_i / d x_j) of a vector.
  Value grad(const Value & v)
  {
    if (v.rank == 0)
    {
      Value g{1, {}};
      for (int j = 0; j < _n; ++j)
        g.c.push_back(derivative(v.c[0], j, _what));
      return g;
    }
    if (v.rank == 1)
    {
      vectorsInPlane("the gradient of a vector");
      Value g{2, {}};
      for (int i = 0; i < _n; ++i)
        for (int j = 0; j < _n; ++j)
          g.c.push_back(derivative(v.c[i], j, _what));
      return g;
    }
    fail("the gradient of a tensor is not defined here");
  }

  /// The divergence of @p v, expanded in place.
  Value div(const Value & v)
  {
    if (v.rank == 1)
    {
      SymPtr s = num(0.0);
      for (int i = 0; i < _n; ++i)
        s = add(s, derivative(v.c[i], i, _what));
      if (_scope.axisymmetric)
        s = add(s, divide(v.c[0], leaf(Sym::Coordinate, "", 0)));
      return scalar(s);
    }
    if (v.rank == 2)
    {
      vectorsInPlane("the divergence of a tensor");
      Value out{1, {}};
      for (int i = 0; i < _n; ++i)
      {
        SymPtr s = num(0.0);
        for (int j = 0; j < _n; ++j)
          s = add(s, derivative(v.c[i * _n + j], j, _what));
        out.c.push_back(s);
      }
      return out;
    }
    fail("div takes a vector or a tensor, and here it has a scalar; the gradient is grad");
  }

private:
  void vectorsInPlane(const std::string & what) const
  {
    if (_scope.axisymmetric)
      fail(what +
           " in axisymmetric coordinates has hoop terms that the equation language does not yet "
           "write; use the axisymmetric physics");
  }

  Value map(Value v, const std::function<SymPtr(const SymPtr &)> & f) const
  {
    for (auto & s : v.c)
      s = f(s);
    return v;
  }

  Value name(const std::string & n)
  {
    if (const auto it = _rank.find(n); it != _rank.end())
    {
      if (it->second == 0)
        return scalar(leaf(Sym::Field, n));
      Value v{1, {}};
      for (int i = 0; i < _n; ++i)
        v.c.push_back(leaf(Sym::Field, component(n, i)));
      return v;
    }
    if (_component.count(n))
      return scalar(leaf(Sym::Field, n));
    if (n == "t")
      return scalar(leaf(Sym::Time));
    const int axis = coordinateAxis(n);
    if (axis >= 0)
      return scalar(leaf(Sym::Coordinate, "", axis));
    if (n == "I")
    {
      Value v{2, {}};
      for (int i = 0; i < _n; ++i)
        for (int j = 0; j < _n; ++j)
          v.c.push_back(num(i == j ? 1.0 : 0.0));
      return v;
    }
    if (n == "pi" || n == "e" || n == "eigenvalue" ||
        std::find(_scope.constants.begin(), _scope.constants.end(), n) != _scope.constants.end())
      return scalar(leaf(Sym::Constant, n));
    if (n == "n")
    {
      Value v{1, {}};
      for (int i = 0; i < _n; ++i)
        v.c.push_back(leaf(Sym::Normal, "", i));
      return v;
    }
    return scalar(leaf(Sym::External, n));
  }

  int coordinateAxis(const std::string & n) const
  {
    if (_scope.axisymmetric)
      return n == "r" ? 0 : n == "z" ? 1 : -1;
    for (int i = 0; i < _n; ++i)
      if (n == std::string(1, kAxes[i]))
        return i;
    return -1;
  }

  Value list(const AstPtr & a)
  {
    if (static_cast<int>(a->args.size()) != _n)
      fail("a vector here has " + std::to_string(_n) + " components, one per dimension, and [" +
           std::to_string(a->args.size()) + " items] has " + std::to_string(a->args.size()));
    std::vector<Value> items;
    for (const auto & x : a->args)
      items.push_back(eval(x));
    const int rank = items[0].rank;
    if (rank > 1)
      fail("a list holds scalars (a vector) or vectors (the rows of a tensor)");
    Value v{rank + 1, {}};
    for (const auto & x : items)
    {
      if (x.rank != rank)
        fail("the items of a list must be of one kind");
      v.c.insert(v.c.end(), x.c.begin(), x.c.end());
    }
    return v;
  }

  Value derivativeOf(const AstPtr & a)
  {
    if (a->coordinate == "t")
      fail("d" + a->name +
           "/dt may appear only as a term of the equation, multiplied by a "
           "coefficient");
    const Value f = name(a->name);
    if (a->coordinate == "n")
    {
      const Value g = grad(f);
      if (f.rank != 0)
        fail("d" + a->name + "/dn is defined for a scalar field");
      SymPtr s = num(0.0);
      for (int i = 0; i < _n; ++i)
        s = add(s, mul(leaf(Sym::Normal, "", i), g.c[i]));
      return scalar(s);
    }
    const int axis = coordinateAxis(a->coordinate);
    if (axis < 0)
      fail("d" + a->name + "/d" + a->coordinate + ": " + a->coordinate +
           " is not a coordinate of this problem");
    return map(f, [&](const SymPtr & s) { return derivative(s, axis, _what); });
  }

  Value function(const AstPtr & a)
  {
    const std::string & f = a->name;
    std::vector<Value> args;
    for (const auto & x : a->args)
      args.push_back(eval(x));
    const auto need = [&](std::size_t count)
    {
      if (args.size() != count)
        fail(f + " takes " + std::to_string(count) + " argument(s)");
    };
    const auto scalars = [&]()
    {
      for (const auto & v : args)
        if (v.rank)
          fail(f + " takes scalars");
    };
    if (f == "grad")
    {
      need(1);
      return grad(args[0]);
    }
    if (f == "grad_x" || f == "grad_y" || f == "grad_z")
    {
      need(1);
      const int axis = f.back() - 'x';
      if (axis >= _n)
        fail(f + " is not defined in " + std::to_string(_n) + " dimension(s)");
      return map(args[0], [&](const SymPtr & s) { return derivative(s, axis, _what); });
    }
    if (f == "div")
    {
      need(1);
      return div(args[0]);
    }
    if (f == "laplacian")
    {
      need(1);
      return div(grad(args[0]));
    }
    if (f == "curl")
    {
      need(1);
      const Value & v = args[0];
      if (_n == 3 && v.rank == 1)
      {
        const auto D = [&](int i, int j) { return derivative(v.c[i], j, _what); };
        return {1, {sub(D(2, 1), D(1, 2)), sub(D(0, 2), D(2, 0)), sub(D(1, 0), D(0, 1))}};
      }
      if (_n == 2 && v.rank == 1)
        return scalar(sub(derivative(v.c[1], 0, _what), derivative(v.c[0], 1, _what)));
      if (_n == 2 && v.rank == 0)
        return {1, {derivative(v.c[0], 1, _what), neg(derivative(v.c[0], 0, _what))}};
      fail("curl takes a vector in two or three dimensions, or a scalar in two");
    }
    if (f == "dot")
    {
      need(2);
      const Value & x = args[0];
      const Value & y = args[1];
      if (x.rank == 1 && y.rank == 1)
      {
        SymPtr s = num(0.0);
        for (int i = 0; i < _n; ++i)
          s = add(s, mul(x.c[i], y.c[i]));
        return scalar(s);
      }
      if (x.rank == 2 && y.rank == 1)
      {
        Value out{1, {}};
        for (int i = 0; i < _n; ++i)
        {
          SymPtr s = num(0.0);
          for (int j = 0; j < _n; ++j)
            s = add(s, mul(x.c[i * _n + j], y.c[j]));
          out.c.push_back(s);
        }
        return out;
      }
      if (x.rank == 1 && y.rank == 2)
      {
        Value out{1, {}};
        for (int j = 0; j < _n; ++j)
        {
          SymPtr s = num(0.0);
          for (int i = 0; i < _n; ++i)
            s = add(s, mul(x.c[i], y.c[i * _n + j]));
          out.c.push_back(s);
        }
        return out;
      }
      if (x.rank == 2 && y.rank == 2)
      {
        Value out{2, {}};
        for (int i = 0; i < _n; ++i)
          for (int j = 0; j < _n; ++j)
          {
            SymPtr s = num(0.0);
            for (int k = 0; k < _n; ++k)
              s = add(s, mul(x.c[i * _n + k], y.c[k * _n + j]));
            out.c.push_back(s);
          }
        return out;
      }
      fail("dot takes vectors and tensors; a scalar is multiplied with '*'");
    }
    if (f == "cross")
    {
      need(2);
      const Value & x = args[0];
      const Value & y = args[1];
      if (x.rank != 1 || y.rank != 1)
        fail("cross takes two vectors");
      if (_n == 3)
        return {1,
                {sub(mul(x.c[1], y.c[2]), mul(x.c[2], y.c[1])),
                 sub(mul(x.c[2], y.c[0]), mul(x.c[0], y.c[2])),
                 sub(mul(x.c[0], y.c[1]), mul(x.c[1], y.c[0]))}};
      if (_n == 2)
        return scalar(sub(mul(x.c[0], y.c[1]), mul(x.c[1], y.c[0])));
      fail("cross is defined in two and three dimensions");
    }
    if (f == "outer")
    {
      need(2);
      if (args[0].rank != 1 || args[1].rank != 1)
        fail("outer takes two vectors");
      Value out{2, {}};
      for (int i = 0; i < _n; ++i)
        for (int j = 0; j < _n; ++j)
          out.c.push_back(mul(args[0].c[i], args[1].c[j]));
      return out;
    }
    if (f == "ddot")
    {
      need(2);
      if (args[0].rank != 2 || args[1].rank != 2)
        fail("ddot takes two tensors");
      SymPtr s = num(0.0);
      for (std::size_t k = 0; k < args[0].c.size(); ++k)
        s = add(s, mul(args[0].c[k], args[1].c[k]));
      return scalar(s);
    }
    if (f == "norm")
    {
      need(1);
      if (args[0].rank == 0)
        return scalar(call("abs", {args[0].c[0]}));
      SymPtr s = num(0.0);
      for (const auto & x : args[0].c)
        s = add(s, power(x, num(2.0)));
      return scalar(call("sqrt", {s}));
    }
    if (f == "transpose" || f == "trace" || f == "sym" || f == "skew" || f == "dev" || f == "det" ||
        f == "inverse")
    {
      need(1);
      const Value & A = args[0];
      if (A.rank != 2)
        fail(f + " takes a tensor");
      const auto at = [&](int i, int j) { return A.c[i * _n + j]; };
      if (f == "trace")
      {
        SymPtr s = num(0.0);
        for (int i = 0; i < _n; ++i)
          s = add(s, at(i, i));
        return scalar(s);
      }
      if (f == "det" || f == "inverse")
      {
        SymPtr det;
        if (_n == 1)
          det = at(0, 0);
        else if (_n == 2)
          det = sub(mul(at(0, 0), at(1, 1)), mul(at(0, 1), at(1, 0)));
        else
          det = add(sub(mul(at(0, 0), sub(mul(at(1, 1), at(2, 2)), mul(at(1, 2), at(2, 1)))),
                        mul(at(0, 1), sub(mul(at(1, 0), at(2, 2)), mul(at(1, 2), at(2, 0))))),
                    mul(at(0, 2), sub(mul(at(1, 0), at(2, 1)), mul(at(1, 1), at(2, 0)))));
        if (f == "det")
          return scalar(det);
        Value out{2, std::vector<SymPtr>(A.c.size())};
        for (int i = 0; i < _n; ++i)
          for (int j = 0; j < _n; ++j)
          {
            SymPtr cofactor;
            if (_n == 1)
              cofactor = num(1.0);
            else if (_n == 2)
              cofactor = (i + j) % 2 ? neg(at(1 - i, 1 - j)) : at(1 - i, 1 - j);
            else
            {
              const int r0 = (i + 1) % 3, r1 = (i + 2) % 3, c0 = (j + 1) % 3, c1 = (j + 2) % 3;
              cofactor = sub(mul(at(r0, c0), at(r1, c1)), mul(at(r0, c1), at(r1, c0)));
            }
            out.c[j * _n + i] = divide(cofactor, det); // the adjugate is the transposed cofactor
          }
        return out;
      }
      Value out{2, {}};
      for (int i = 0; i < _n; ++i)
        for (int j = 0; j < _n; ++j)
        {
          if (f == "transpose")
            out.c.push_back(at(j, i));
          else if (f == "sym")
            out.c.push_back(mul(num(0.5), add(at(i, j), at(j, i))));
          else if (f == "skew")
            out.c.push_back(mul(num(0.5), sub(at(i, j), at(j, i))));
        }
      if (f == "dev")
      {
        SymPtr tr = num(0.0);
        for (int i = 0; i < _n; ++i)
          tr = add(tr, at(i, i));
        for (int i = 0; i < _n; ++i)
          for (int j = 0; j < _n; ++j)
            out.c.push_back(i == j ? sub(at(i, j), divide(tr, num(_n))) : at(i, j));
      }
      return out;
    }
    if (f == "if")
    {
      need(3);
      if (args[0].rank)
        fail("the condition of if is a scalar");
      if (args[1].rank != args[2].rank)
        fail("the two values of if must be of one kind");
      Value out{args[1].rank, {}};
      for (std::size_t k = 0; k < args[1].c.size(); ++k)
        out.c.push_back(op(Sym::Select, {args[0].c[0], args[1].c[k], args[2].c[k]}));
      return out;
    }
    static const std::set<std::string> one{"sin",
                                           "cos",
                                           "tan",
                                           "asin",
                                           "acos",
                                           "atan",
                                           "sinh",
                                           "cosh",
                                           "tanh",
                                           "exp",
                                           "log",
                                           "log10",
                                           "sqrt",
                                           "abs",
                                           "floor",
                                           "ceil",
                                           "erf",
                                           "sign"};
    static const std::set<std::string> two{"atan2", "pow", "hypot", "min", "max"};
    if (one.count(f) || two.count(f))
    {
      need(one.count(f) ? 1 : 2);
      scalars();
      std::vector<SymPtr> s;
      for (const auto & v : args)
        s.push_back(v.c[0]);
      return scalar(f == "pow" ? power(s[0], s[1]) : call(f, s));
    }
    fail("'" + f +
         "' is not a function of the equation language; a registered function of (x, "
         "y, z, t) is used by its name alone, as " +
         f);
  }

  const EquationScope & _scope;
  std::string _what;
  int _n = 2;
  std::map<std::string, int> _rank;
  std::map<std::string, std::pair<std::string, int>> _component;
};

// ---------------------------------------------------------------------------
// Sorting the terms of an equation.
// ---------------------------------------------------------------------------
/// A term of an equation: a factor times a part, with a sign.
struct Term
{
  enum Kind
  {
    Time1,      ///< factor * d<field>/dt
    Time2,      ///< factor * d2<field>/dt2
    Divergence, ///< factor * div(part)
    Other       ///< factor * part
  } kind;
  AstPtr part;
  std::vector<std::pair<AstPtr, bool>> factors; ///< (tree, divides)
  bool negative = false;
};

bool
isTime(const Ast & a)
{
  return a.kind == Ast::Derivative && a.coordinate == "t";
}
bool
isDivergence(const Ast & a)
{
  return a.kind == Ast::Call && (a.name == "div" || a.name == "laplacian") && a.args.size() == 1;
}
bool
special(const AstPtr & a)
{
  if (isTime(*a) || isDivergence(*a))
    return true;
  return std::any_of(a->args.begin(), a->args.end(), special);
}

void
collect(const AstPtr & a, Term base, std::vector<Term> & out, const std::string & what)
{
  if (!special(a))
  {
    base.kind = Term::Other;
    base.part = a;
    out.push_back(base);
    return;
  }
  switch (a->kind)
  {
  case Ast::Add:
    collect(a->args[0], base, out, what);
    collect(a->args[1], base, out, what);
    return;
  case Ast::Sub:
  {
    collect(a->args[0], base, out, what);
    Term negated = base;
    negated.negative = !negated.negative;
    collect(a->args[1], negated, out, what);
    return;
  }
  case Ast::Neg:
  {
    Term negated = base;
    negated.negative = !negated.negative;
    collect(a->args[0], negated, out, what);
    return;
  }
  case Ast::Mul:
  {
    const bool left = special(a->args[0]), right = special(a->args[1]);
    if (left && right)
      throw InputError(what + ": a product of two terms with time derivatives or divergences "
                              "cannot be sorted; each of them must be a term of its own.");
    Term t = base;
    t.factors.emplace_back(left ? a->args[1] : a->args[0], false);
    collect(left ? a->args[0] : a->args[1], t, out, what);
    return;
  }
  case Ast::Div:
  {
    if (special(a->args[1]))
      throw InputError(what + ": a time derivative or a divergence may not divide.");
    Term t = base;
    t.factors.emplace_back(a->args[1], true);
    collect(a->args[0], t, out, what);
    return;
  }
  default:
    break;
  }
  if (isTime(*a))
  {
    base.kind = a->order == 2 ? Term::Time2 : Term::Time1;
    base.part = a;
    out.push_back(base);
    return;
  }
  if (isDivergence(*a) && !special(a->args[0]))
  {
    base.kind = Term::Divergence;
    base.part = a->name == "laplacian" ? node(Ast::Call, {a->args[0]}, "grad") : a->args[0];
    out.push_back(base);
    return;
  }
  throw InputError(what +
                   ": a time derivative, div or laplacian may appear only as a term of the "
                   "equation, multiplied or divided by coefficients, not inside a function or "
                   "another operator.");
}

/// One equation, sorted and evaluated.
struct Sorted
{
  std::string text;
  int rank = 0;
  std::vector<SymPtr> time1, time2;      ///< per component
  std::vector<std::vector<SymPtr>> flux; ///< per component, one per dimension
  std::vector<SymPtr> rest; ///< per component, the terms that are not sorted out (moved: f = -rest)
  std::set<std::string> time_fields; ///< fields of the time derivatives
  std::set<std::string> flux_fields; ///< fields whose gradients are in the flux
  std::string time_field;
  int time_order = 0;
};

bool
hasGradient(const SymPtr & s, std::set<std::string> * fields = nullptr)
{
  bool found = false;
  std::function<void(const SymPtr &)> walk = [&](const SymPtr & n)
  {
    if (n->kind == Sym::Gradient)
    {
      found = true;
      if (fields)
        fields->insert(n->name);
    }
    for (const auto & a : n->args)
      walk(a);
  };
  walk(s);
  return found;
}

Sorted
sortEquation(Evaluator & ev,
             const std::string & text,
             const std::pair<AstPtr, AstPtr> & sides,
             const std::string & what)
{
  std::vector<Term> terms;
  collect(sides.first, Term{Term::Other, nullptr, {}, false}, terms, what);
  collect(sides.second, Term{Term::Other, nullptr, {}, true}, terms, what);
  const int n = ev.dimension();
  Sorted s;
  s.text = text;
  s.rank = -1;
  const auto factorOf = [&](const Term & t)
  {
    SymPtr f = num(t.negative ? -1.0 : 1.0);
    for (const auto & [tree, divides] : t.factors)
    {
      const Value v = ev.eval(tree);
      if (v.rank)
        ev.fail("a time derivative or a divergence is multiplied by a scalar only; here the "
                "factor is " +
                std::string(rankName(v.rank)));
      f = divides ? divide(f, v.c[0]) : mul(f, v.c[0]);
    }
    return f;
  };
  const auto setRank = [&](int rank)
  {
    if (s.rank == -1)
    {
      s.rank = rank;
      const std::size_t count = rank == 0 ? 1 : static_cast<std::size_t>(n);
      s.time1.assign(count, num(0.0));
      s.time2.assign(count, num(0.0));
      s.rest.assign(count, num(0.0));
      s.flux.assign(count, std::vector<SymPtr>(static_cast<std::size_t>(n), num(0.0)));
    }
    else if (s.rank != rank)
      ev.fail(std::string("the terms of an equation must be of one kind, and here ") +
              rankName(s.rank) + " meets " + rankName(rank));
  };
  for (const Term & t : terms)
  {
    const SymPtr factor = factorOf(t);
    if (t.kind == Term::Time1 || t.kind == Term::Time2)
    {
      const std::string field = ev.fieldOf(t.part->name);
      Value f = ev.eval(node(Ast::Name, {}, t.part->name));
      if (t.part->name != field)
        ev.fail("write the time derivative of the vector field, d" + field +
                "/dt, not of one "
                "of its components");
      setRank(f.rank);
      if (!s.time_field.empty() && s.time_field != field)
        ev.fail("an equation may have the time derivative of one field only, and this one has d" +
                s.time_field + "/dt and d" + field + "/dt");
      s.time_field = field;
      s.time_fields.insert(field);
      auto & target = t.kind == Term::Time1 ? s.time1 : s.time2;
      for (auto & c : target)
        c = add(c, factor);
      s.time_order = std::max(s.time_order, t.kind == Term::Time1 ? 1 : 2);
      continue;
    }
    if (t.kind == Term::Divergence)
    {
      const Value inner = ev.eval(t.part);
      std::set<std::string> fields;
      bool second_order = false;
      for (const auto & c : inner.c)
        second_order = hasGradient(c, &fields) || second_order;
      if (!second_order)
      {
        // A first-order term: expanded in place.
        const Value v = ev.div(inner);
        setRank(v.rank);
        for (std::size_t k = 0; k < v.c.size(); ++k)
          s.rest[k] = add(s.rest[k], mul(factor, v.c[k]));
        continue;
      }
      if (variesInSpace(factor))
        ev.fail("a div(...) or laplacian(...) term may be multiplied only by a factor that is "
                "constant in space, here '" +
                print(factor) +
                "'; move the factor inside, as in div(k*grad(u)), which is the conservative form");
      if (inner.rank == 0 || inner.rank > 2)
        ev.fail("div takes a vector or a tensor");
      setRank(inner.rank - 1);
      for (const auto & f : fields)
        s.flux_fields.insert(ev.fieldOf(f));
      for (std::size_t i = 0; i < s.flux.size(); ++i)
        for (int j = 0; j < n; ++j)
        {
          const SymPtr & c = inner.rank == 1 ? inner.c[static_cast<std::size_t>(j)]
                                             : inner.c[i * static_cast<std::size_t>(n) + j];
          s.flux[i][static_cast<std::size_t>(j)] =
              add(s.flux[i][static_cast<std::size_t>(j)], mul(factor, c));
        }
      continue;
    }
    const Value v = ev.eval(t.part);
    if (v.rank == 0 && isNumber(v.c[0], 0.0))
      continue; // "... = 0" holds for an equation of any kind
    setRank(v.rank);
    for (std::size_t k = 0; k < v.c.size(); ++k)
      s.rest[k] = add(s.rest[k], mul(factor, v.c[k]));
  }
  if (s.rank == -1)
    ev.fail("the equation has no terms");
  return s;
}

std::string
quoted(const std::string & text)
{
  return "'" + text + "'";
}

} // namespace

std::vector<ScalarEquation>
reduceEquations(const EquationScope & scope,
                const std::vector<std::string> & equations,
                const std::vector<std::string> & keys)
{
  if (scope.fields.empty())
    throw InputError("Give the fields of the equations in 'variables'.");
  if (equations.size() != scope.fields.size())
    throw InputError("There are " + std::to_string(scope.fields.size()) + " field(s) and " +
                     std::to_string(equations.size()) +
                     " equation(s); give one equation for each field.");
  if (!keys.empty() && keys.size() != equations.size())
    throw InputError("Give a key for every equation, or none.");

  Evaluator names(scope, "the definitions");
  const auto is_field = [&](const std::string & n) { return names.isField(n); };

  // The definitions, in order, each seeing the ones before it.
  std::map<std::string, AstPtr> definitions;
  std::set<std::string> later;
  for (const auto & [n, text] : scope.definitions)
    later.insert(n);
  for (const auto & [n, text] : scope.definitions)
  {
    later.erase(n);
    std::set<std::string> after = later;
    after.insert(n);
    Parser p(text, "the definition " + quoted(n), is_field, definitions, after);
    definitions[n] = p.expression();
  }

  std::vector<Sorted> sorted;
  for (const auto & text : equations)
  {
    const std::string what = "the equation " + quoted(text);
    Parser p(text, what, is_field, definitions, later);
    Evaluator ev(scope, what);
    sorted.push_back(sortEquation(ev, text, p.equation(), what));
  }

  // Match the equations to the fields.
  const std::size_t count = scope.fields.size();
  std::vector<int> field_of(count, -1); // equation -> field index
  std::vector<int> equation_of(count, -1);
  const auto index = [&](const std::string & field)
  {
    for (std::size_t k = 0; k < count; ++k)
      if (scope.fields[k].first == field)
        return static_cast<int>(k);
    return -1;
  };
  const auto assign = [&](std::size_t e, int f)
  {
    if (equation_of[static_cast<std::size_t>(f)] != -1)
      throw InputError(
          "The equations " +
          quoted(sorted[static_cast<std::size_t>(equation_of[static_cast<std::size_t>(f)])].text) +
          " and " + quoted(sorted[e].text) + " both govern " +
          scope.fields[static_cast<std::size_t>(f)].first +
          ". Give the field of each equation as a key: equations={\"field\": \"equation\", ...}.");
    field_of[e] = f;
    equation_of[static_cast<std::size_t>(f)] = static_cast<int>(e);
  };
  for (std::size_t e = 0; e < count; ++e)
    if (!keys.empty() && !keys[e].empty())
    {
      const int f = index(keys[e]);
      if (f < 0)
        throw InputError("The key " + quoted(keys[e]) + " of the equation " +
                         quoted(sorted[e].text) + " is not a field.");
      assign(e, f);
    }
  for (std::size_t e = 0; e < count; ++e)
    if (field_of[e] == -1 && !sorted[e].time_field.empty())
      assign(e, index(sorted[e].time_field));
  for (bool changed = true; changed;)
  {
    changed = false;
    for (std::size_t e = 0; e < count; ++e)
    {
      if (field_of[e] != -1)
        continue;
      std::vector<int> candidates;
      for (const auto & f : sorted[e].flux_fields)
        if (equation_of[static_cast<std::size_t>(index(f))] == -1)
          candidates.push_back(index(f));
      if (candidates.size() == 1)
      {
        assign(e, candidates[0]);
        changed = true;
      }
    }
  }
  std::vector<std::size_t> open_equations;
  std::vector<int> open_fields;
  for (std::size_t e = 0; e < count; ++e)
    if (field_of[e] == -1)
      open_equations.push_back(e);
  for (std::size_t f = 0; f < count; ++f)
    if (equation_of[f] == -1)
      open_fields.push_back(static_cast<int>(f));
  if (open_equations.size() == 1)
    assign(open_equations[0], open_fields[0]);
  else if (!open_equations.empty())
  {
    std::string list;
    for (auto e : open_equations)
      list += (list.empty() ? "" : ", ") + quoted(sorted[e].text);
    std::string fields;
    for (int f : open_fields)
      fields += (fields.empty() ? "" : ", ") + scope.fields[static_cast<std::size_t>(f)].first;
    throw InputError("The equations " + list + " could govern any of the fields " + fields +
                     ". Give the field of each equation as a key: equations={\"field\": "
                     "\"equation\", ...}.");
  }

  // One scalar equation per scalar unknown, in the order of the fields.
  Evaluator ev(scope, "the equations");
  std::vector<ScalarEquation> out;
  for (std::size_t f = 0; f < count; ++f)
  {
    const auto & [field, rank] = scope.fields[f];
    const Sorted & s = sorted[static_cast<std::size_t>(equation_of[f])];
    if (s.rank != rank)
      throw InputError("The equation " + quoted(s.text) + " is " + rankName(s.rank) +
                       " equation, and it governs " + field + ", which is " + rankName(rank) +
                       " field.");
    if (!s.time_field.empty() && s.time_field != field)
      throw InputError("The equation " + quoted(s.text) + " governs " + field +
                       " and has the time derivative of " + s.time_field +
                       "; the time derivative of a field belongs to its own equation.");
    const std::size_t components = rank == 0 ? 1 : static_cast<std::size_t>(scope.dimension);
    for (std::size_t c = 0; c < components; ++c)
    {
      ScalarEquation q;
      q.field = rank == 0 ? field : ev.component(field, static_cast<int>(c));
      q.equation = s.text;
      bool any_flux = false;
      for (const auto & g : s.flux[c])
        any_flux = any_flux || !isNumber(g, 0.0);
      if (any_flux)
        for (const auto & g : s.flux[c])
          q.flux.push_back(print(g));
      if (!isNumber(s.rest[c], 0.0))
        q.source = print(neg(s.rest[c]));
      if (!isNumber(s.time1[c], 0.0))
        q.time_coeff = print(s.time1[c]);
      if (!isNumber(s.time2[c], 0.0))
        q.second_time_coeff = print(s.time2[c]);
      out.push_back(q);
    }
  }
  return out;
}

std::vector<ScalarCondition>
reduceCondition(const EquationScope & scope,
                const std::vector<ScalarEquation> & reduced,
                const std::string & text)
{
  const std::string what = "the boundary condition " + quoted(text);
  Evaluator ev(scope, what);
  const auto is_field = [&](const std::string & n) { return ev.isField(n); };
  std::map<std::string, AstPtr> definitions;
  std::set<std::string> later;
  for (const auto & [n, t] : scope.definitions)
  {
    Parser p(t, "the definition " + quoted(n), is_field, definitions, later);
    definitions[n] = p.expression();
  }
  Parser p(text, what, is_field, definitions, later);
  const auto [lhs, rhs] = p.equation();

  // u = value, or U = [values].
  if (lhs->kind == Ast::Name && ev.isField(lhs->name))
  {
    const Value field = ev.eval(lhs);
    const Value value = ev.eval(rhs);
    if (field.rank != value.rank)
      ev.fail(std::string("the field is ") + rankName(field.rank) + " and the value " +
              rankName(value.rank));
    std::vector<ScalarCondition> out;
    for (std::size_t k = 0; k < field.c.size(); ++k)
    {
      if (hasGradient(value.c[k]))
        ev.fail("the value of a field on a boundary may not depend on derivatives; a condition "
                "on the flux is written with the flux on the left");
      out.push_back({field.c[k]->name, true, print(value.c[k])});
    }
    return out;
  }

  // A flux condition: the field it belongs to.
  std::string field;
  std::function<void(const AstPtr &)> find = [&](const AstPtr & a)
  {
    if (a->kind == Ast::Derivative && a->coordinate == "n")
    {
      if (!field.empty() && field != a->name)
        ev.fail("a flux condition is the flux of one field");
      field = a->name;
    }
    for (const auto & x : a->args)
      find(x);
  };
  find(lhs);
  const bool named = lhs->kind == Ast::Name &&
                     std::find(scope.flux_names.begin(), scope.flux_names.end(), lhs->name) !=
                         scope.flux_names.end();
  if (field.empty() && !named)
    ev.fail("the left side is a field (a value on the boundary), the normal flux written with "
            "d<field>/dn, or one of the names of the normal flux (" +
            [&]
            {
              std::string s;
              for (const auto & n : scope.flux_names)
                s += (s.empty() ? "" : ", ") + n;
              return s;
            }() +
            ")");
  if (named)
  {
    // The field is told by the kind of the value: a vector is the flux of the vector field, a
    // scalar that of the scalar field, among the fields whose equations have a flux.
    const int rank = ev.eval(rhs).rank;
    std::vector<std::string> candidates;
    for (const auto & [name, kind] : scope.fields)
    {
      if (kind != rank)
        continue;
      for (const auto & q : reduced)
        if ((q.field == name || ev.fieldOf(q.field) == name) && !q.flux.empty())
        {
          candidates.push_back(name);
          break;
        }
    }
    if (candidates.size() != 1)
    {
      std::string list;
      for (const auto & c : candidates)
        list += (list.empty() ? "" : ", ") + c;
      ev.fail("'" + lhs->name + "' with " + rankName(rank) +
              (candidates.empty()
                   ? std::string(" value is the flux of no field with a divergence term")
                   : " value could be the flux of " + list) +
              "; write the flux out with d<field>/dn");
    }
    field = candidates[0];
  }
  std::vector<const ScalarEquation *> equations;
  for (const auto & q : reduced)
    if (q.field == field || ev.fieldOf(q.field) == field)
      equations.push_back(&q);
  if (equations.empty() || equations[0]->flux.empty())
    ev.fail("the equation of " + field + " has no divergence term, so it has no flux to prescribe");
  const Value value = ev.eval(rhs);
  if (static_cast<std::size_t>(value.c.size()) != equations.size())
    ev.fail("the value has " + std::to_string(value.c.size()) + " component(s) and the field " +
            field + " " + std::to_string(equations.size()));
  if (!named)
  {
    // The flux written out must be the normal flux of the equation, n . Gamma.
    const Value left = ev.eval(lhs);
    if (left.rank || equations.size() != 1)
      ev.fail("a flux written with d<field>/dn is the flux of a scalar field");
    SymPtr normal = num(0.0);
    for (std::size_t j = 0; j < equations[0]->flux.size(); ++j)
    {
      std::set<std::string> none;
      std::map<std::string, AstPtr> no_definitions;
      Parser q(equations[0]->flux[j], what, is_field, no_definitions, none);
      normal = add(normal,
                   mul(leaf(Sym::Normal, "", static_cast<int>(j)), ev.eval(q.expression()).c[0]));
    }
    for (int trial = 0; trial < 3; ++trial)
    {
      const auto sample = [&](const Sym & s)
      {
        const std::string key =
            std::to_string(s.kind) + s.name + std::to_string(s.axis) + std::to_string(trial);
        return 0.5 + static_cast<double>(std::hash<std::string>{}(key) % 1000) / 1000.0;
      };
      const double a = evaluate(left.c[0], sample);
      const double b = evaluate(normal, sample);
      if (!(std::abs(a - b) <= 1e-9 * std::max({1.0, std::abs(a), std::abs(b)})))
        ev.fail("the left side is not the normal flux of the equation of " + field +
                ", which is n . Gamma = " + print(normal) +
                " (n is the outward normal); write that, or the name " + scope.flux_names[0]);
    }
  }
  std::vector<ScalarCondition> out;
  for (std::size_t k = 0; k < equations.size(); ++k)
    out.push_back({equations[k]->field, false, print(value.c[k])});
  return out;
}

} // namespace dualmesh
