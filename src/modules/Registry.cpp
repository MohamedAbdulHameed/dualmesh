// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/modules/Registry.h"

namespace dualmesh
{

void
registerAllObjects()
{
  static const bool registered = []
  {
    Factory & f = Factory::instance();
    registerFrameworkObjects(f);
    registerHeatTransferObjects(f);
    registerSolidMechanicsObjects(f);
    registerFluidObjects(f);
    registerParsedObjects(f);
    registerParsedStateObjects(f);
    return true;
  }();
  (void) registered;
}

} // namespace dualmesh
