// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Partial differential equations and boundary conditions written as text, as they are printed:
//
//   "rho*cp*dT/dt = div(k*grad(T)) + q"
//   "rho*(dU/dt + dot(grad(U), U)) = div(-p*I + mu*(grad(U) + transpose(grad(U))))"
//   "-k*dT/dn = h*(T - 293)"
//
// An equation is reduced to the general form of each of its scalar components,
//
//   m d2u/dt2 + d du/dt + div(Gamma) = f ,
//
// with the flux Gamma, the source f and the coefficients d and m written in the scalar expression
// language of the parsed objects (grad_x(u), x, t, ...), which compiles them once and
// differentiates them automatically.  The reduction is symbolic: vector and tensor operators are
// expanded component by component, and the derivatives of expressions follow the chain rule.
//
// The terms are sorted by these rules.
//   * Everything inside div(...) whose argument contains a derivative of a field is the flux
//     Gamma, integrated by parts, so that n . Gamma is what a flux condition prescribes; such a
//     term may be multiplied only by a factor that is constant in space.  laplacian(u) is
//     div(grad(u)).
//   * A div(...) of an argument without derivatives (div(U) of a velocity, for instance) is a
//     first-order term and is expanded in place.
//   * du/dt gives d and d2u/dt2 gives m, with any factor that multiplies them.
//   * Everything else is moved to the right side and is the source f.
//
// The names: the fields (a vector field U has the components U_x, U_y, U_z, or U_r, U_z in
// axisymmetric coordinates); x, y, z (r, z in axisymmetric coordinates) and t; the constants pi,
// e and eigenvalue; the definitions of the scope, in order; and any other name, which is passed on
// unchanged (a registered function, a property, an element field) and may not be differentiated.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace dualmesh
{

/// What the equations of one physics may use.
struct EquationScope
{
  /// Space dimension of the mesh: the number of components of a vector.
  int dimension = 2;
  /// Axisymmetric coordinates: r is x and z is y of the mesh.
  bool axisymmetric = false;
  /// The fields, in the order of the unknowns: name and rank (0 scalar, 1 vector).
  std::vector<std::pair<std::string, int>> fields;
  /// Named constants (numbers), which stay names in the output.
  std::vector<std::string> constants;
  /// Named expressions, in order: name and text.  Each may use the ones before it.
  std::vector<std::pair<std::string, std::string>> definitions;
  /// Names of the normal flux n . Gamma in boundary conditions (normal_flux, and a physics' own
  /// name such as heat_flux).
  std::vector<std::string> flux_names{"normal_flux"};
};

/// One scalar equation m d2u/dt2 + d du/dt + div(Gamma) = f, in the scalar expression language.
struct ScalarEquation
{
  std::string field;             ///< the scalar unknown (u, or a component such as U_x)
  std::string equation;          ///< the text the user wrote, for messages and reports
  std::vector<std::string> flux; ///< Gamma, one expression per dimension; empty when none
  std::string source;            ///< f; empty when none
  std::string time_coeff;        ///< d; empty when none
  std::string second_time_coeff; ///< m; empty when none
};

/// Reduce @p equations to scalar equations, one per scalar unknown, in the order of the fields of
/// @p scope.  @p keys is empty, or names the field of each equation (an empty key leaves that one
/// to the matching): an equation is matched to the field whose time derivative it has, then to
/// the field whose second derivative it has, and a field left over takes the equation left over.
/// Throws InputError, quoting the equation, when an equation cannot be read or matched.
std::vector<ScalarEquation> reduceEquations(const EquationScope & scope,
                                            const std::vector<std::string> & equations,
                                            const std::vector<std::string> & keys = {});

/// One scalar boundary condition: u = value (Dirichlet) or n . Gamma = value (a flux condition,
/// whose value may depend on the fields).
struct ScalarCondition
{
  std::string field;
  bool dirichlet = false;
  std::string value;
};

/// Read the boundary condition @p text ("u = 0", "-k*du/dn = h*(u - 1)", "heat_flux = 5000") of
/// the equations @p reduced.  A flux written out with du/dn must be the normal flux of the
/// equation of u; it is compared with n . Gamma at sample points.
std::vector<ScalarCondition> reduceCondition(const EquationScope & scope,
                                             const std::vector<ScalarEquation> & reduced,
                                             const std::string & text);

} // namespace dualmesh
