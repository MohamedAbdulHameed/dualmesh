// SPDX-License-Identifier: LGPL-2.1-or-later
//
// A function of (x, y, z, t) given as text, compiled once and evaluated in
// C++.  This is the counterpart of MOOSE's ParsedFunction.
//
// Why compile it here rather than evaluate it in Python?  A coefficient, a
// source or a boundary value is evaluated at every quadrature point of every
// element on every Newton iteration.  Calling back into Python for each of
// those is slow, and because it needs the global interpreter lock it also
// forces the whole assembly onto one thread.  An expression compiled to a
// short sequence of stack operations costs a few tens of nanoseconds, is safe
// to call from any number of threads, and leaves the threading intact.
//
// The grammar, from loosest to tightest binding:
//
//   expression  := comparison
//   comparison  := sum [ ('<' | '>' | '<=' | '>=' | '==' | '!=') sum ]
//   sum         := product { ('+' | '-') product }
//   product     := unary { ('*' | '/') unary }
//   unary       := ('-' | '+') unary | power
//   power       := primary [ ('^' | '**') unary ]
//   primary     := number | name | name '(' arguments ')' | '(' expression ')'
//
// so exponentiation is right-associative and binds tighter than a unary minus
// (-x^2 is -(x^2)), as in mathematics.  The names in scope are the variables
// x, y, z and t; the constants pi and e (also E); and the functions sin, cos,
// tan, asin, acos, atan, sinh, cosh, tanh, exp, log, log10, sqrt, abs (also
// Abs and fabs), floor, ceil, erf, sign, atan2, pow, hypot, min, max, and
// if(condition, a, b).  A comparison is 1 when true and 0 when false.  The
// spellings E, Abs and ** are accepted so that the text printed by SymPy can be
// used unchanged, which is what the manufactured-solution tools rely on.
#pragma once

#include "dualmesh/core/Function.h"

#include <string>
#include <vector>

namespace dualmesh
{

class ParsedFunction : public Function
{
public:
  /// Compile @p expression.  Throws InputError, naming the position and the
  /// problem, when the text is not a valid expression.
  explicit ParsedFunction(std::string expression);

  double value(const Point & x, double t) const override;
  std::string description() const override { return _text; }
  const std::string & expression() const { return _text; }

  /// True when the text compiles; the reason is written to @p why otherwise.
  static bool isExpression(const std::string & text, std::string * why = nullptr);

  /// The operations of the compiled program.  Public only so that the
  /// compiler, which lives in the implementation file, can build them.
  enum class Op : unsigned char
  {
    Constant,
    Variable,
    Negate,
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
    Equal,
    NotEqual,
    Function1,
    Function2,
    Select
  };
  struct Instruction
  {
    Op op;
    int argument = 0; ///< variable index or function index
    double value = 0; ///< constant
  };

private:
  std::string _text;
  std::vector<Instruction> _program;
  int _max_depth = 0;
};

} // namespace dualmesh
