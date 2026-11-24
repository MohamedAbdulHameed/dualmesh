// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Equations written on the undeformed mesh for a body that deforms.
//
// With finite_strain_stress (or small_strain_stress) the mesh stays where it
// was built, and x = X + u(X) is the deformed position of the material point
// X.  An equation that holds on the deformed body is carried back to the
// undeformed one with the deformation gradient F = dx/dX and its determinant
// J = det F (the volume ratio dv/dV):
//
//  * an area element: n da = J F^-T N dA (Nanson's relation), so a flux q per
//    unit deformed area is q |J F^-T N| per unit undeformed area;
//  * a gradient: grad_x T = F^-T Grad_X T, so the heat flux k grad_x T through
//    the deformed body is, per unit undeformed area, J k F^-1 F^-T Grad_X T
//    (the conductivity tensor J k C^-1, with C = F^T F).
//
// F is read from the nine-component property 'deformation_gradient'
// (row by row).  Its first dim rows and columns belong to the mesh directions
// (x, y, z; or r, z in axisymmetric problems), and the others to the
// directions out of the mesh (the hoop direction, and the axial direction of
// a one-dimensional rod).
#pragma once

#include "dualmesh/base/Problem.h"
#include "dualmesh/base/QpContext.h"
#include "dualmesh/core/InputParameters.h"

#include <string>

namespace dualmesh
{
namespace deformation
{

/// Declare the parameter that names the deformation gradient property.
inline void
addParameter(InputParameters & p, const std::string & what)
{
  p.addOptional("deformation_gradient_property",
                ParameterKind::String,
                std::string(""),
                "Name of the deformation gradient property (nine components, as "
                "declared by finite_strain_stress and small_strain_stress as "
                "'deformation_gradient'). When it is given, " +
                    what +
                    " Leave it empty (the default) for a body that does not deform, or when "
                    "the change of shape may be neglected, as at small strain.");
}

/// The id of the property named by 'deformation_gradient_property', or -1
/// when none is named.  The property must have nine components.
inline int
propertyId(const Problem & problem, const InputParameters & p, const std::string & object)
{
  const std::string name = p.getString("deformation_gradient_property");
  if (name.empty())
    return -1;
  const auto & registry = problem.propertyRegistry();
  if (registry.components(name) != 9)
    throw InputError("'" + object + "': the deformation gradient property '" + name +
                     "' must have nine components (it is declared as 'deformation_gradient' "
                     "by finite_strain_stress and small_strain_stress).");
  return registry.id(name);
}

/// The cofactor J F^-T of the 3x3 matrix F, read from property @p prop.
inline void
cofactor(const QpContext & ctx, int prop, ADReal cof[3][3])
{
  const auto F = [&](int i, int j) -> const ADReal & { return ctx.property(prop, 3 * i + j); };
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
    {
      const int a = (i + 1) % 3, b = (i + 2) % 3, c = (j + 1) % 3, d = (j + 2) % 3;
      cof[i][j] = F(a, c) * F(b, d) - F(a, d) * F(b, c);
    }
}

/// da/dA = |J F^-T N| at a boundary point with undeformed unit normal N.
inline ADReal
areaRatio(const QpContext & ctx, int prop)
{
  ADReal cof[3][3];
  cofactor(ctx, prop, cof);
  ADReal sum(0.0);
  for (int i = 0; i < 3; ++i)
  {
    ADReal n(0.0);
    for (int K = 0; K < 3; ++K)
      if (ctx.normal[K] != 0.0)
        n = n + cof[i][K] * ctx.normal[K];
    sum = sum + n * n;
  }
  return sqrt(sum);
}

/// The flux J k F^-1 F^-T g of a gradient g taken on the undeformed mesh
/// (its first ctx.dim components), written into flux[0 .. dim).
inline void
pullBackFlux(
    const QpContext & ctx, int prop, const ADReal & k, const ADVector3 & g, ADVector3 & flux)
{
  // J F^-1 F^-T = cof^T cof / J, with cof = J F^-T.
  ADReal cof[3][3];
  cofactor(ctx, prop, cof);
  const auto F = [&](int i, int j) -> const ADReal & { return ctx.property(prop, 3 * i + j); };
  ADReal J(0.0);
  for (int j = 0; j < 3; ++j)
    J = J + F(0, j) * cof[0][j];
  // w = cof g (only the mesh directions of g are nonzero), then flux = cof^T w k / J.
  ADReal w[3];
  for (int i = 0; i < 3; ++i)
  {
    w[i] = ADReal(0.0);
    for (int b = 0; b < ctx.dim; ++b)
      w[i] = w[i] + cof[i][b] * g[b];
  }
  const ADReal scale = k / J;
  for (int a = 0; a < ctx.dim; ++a)
  {
    ADReal s(0.0);
    for (int i = 0; i < 3; ++i)
      s = s + cof[i][a] * w[i];
    flux[a] = scale * s;
  }
}

} // namespace deformation
} // namespace dualmesh
