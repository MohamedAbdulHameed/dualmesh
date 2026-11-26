// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/linalg/SaddlePointBlocks.h"

namespace dualmesh
{

std::optional<SchurApproximation>
schurApproximation(const std::string & name)
{
  if (name == "pressure_mass")
    return SchurApproximation::PressureMass;
  if (name == "pcd")
    return SchurApproximation::ConvectionDiffusion;
  if (name == "lsc")
    return SchurApproximation::LeastSquaresCommutator;
  return std::nullopt;
}

} // namespace dualmesh
