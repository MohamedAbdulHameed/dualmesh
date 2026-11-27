// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The flow system of the fluid module (FlowSystem.h): the block preconditioners of a flow, the
// level of the pressure of an enclosed flow, the Stokes start of a steady solve and the projection
// of the initial velocity of a transient run.
#pragma once

#include "dualmesh/base/FlowSystem.h"
#include "dualmesh/core/Types.h"

#include <memory>

namespace dualmesh
{

class MassEquation;
struct FlowCoefficients;

/// A new flow system of the fluid module, which registerFluidObjects registers.
std::unique_ptr<FlowSystem> makeFluidSystem();

/// The flow coefficients of the mass equation @p constraint at the point @p x at the current time
/// of @p problem, in element @p element when one is given, so that a fluid of several blocks gives
/// those of its block.
FlowCoefficients flowCoefficientsAt(const Problem & problem,
                                    const MassEquation & constraint,
                                    const Point & x,
                                    Index element = -1);

} // namespace dualmesh
