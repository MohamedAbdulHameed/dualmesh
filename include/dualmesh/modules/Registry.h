// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The registration of the modules: every module adds its object types, and the systems of its
// physics, to the Factory of the base library.  A program calls registerAllObjects once, before it
// makes a Problem: the Python module does it when it is imported, and the C++ tests in main.
#pragma once

#include "dualmesh/base/Factory.h"

namespace dualmesh
{

/// Register the objects of every module in Factory::instance().  A second call does nothing.
void registerAllObjects();

/// The registration function of each module, which registerAllObjects calls.
void registerFrameworkObjects(Factory & f);
void registerHeatTransferObjects(Factory & f);
/// Solid mechanics: continuum elasticity and, through registerStructuralMemberObjects, the beams
/// and plates, and the solid system (Factory::setSolidSystem).
void registerSolidMechanicsObjects(Factory & f);
/// Fluids: the flow objects and the flow system (Factory::setFlowSystem).
void registerFluidObjects(Factory & f);
void registerParsedObjects(Factory & f);
/// The internal states of a balance law (parsed_state).
void registerParsedStateObjects(Factory & f);

} // namespace dualmesh
