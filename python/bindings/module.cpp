// SPDX-License-Identifier: LGPL-2.1-or-later
//
// pybind11 bindings: the whole framework is usable from Python, including
// user-defined kernels, boundary conditions, and property objects written in Python.
// Each subsystem adds its part of the module (bindings.h), in an order that registers every
// class before a default argument or a base class uses it.
#include "bindings.h"
#include "dualmesh/modules/Registry.h"

PYBIND11_MODULE(_core, m)
{
  m.doc() = "dualmesh: dual mesh control domain method for computational mechanics (core)";

  py::register_exception<dualmesh::InputError>(m, "InputError", PyExc_ValueError);

  dualmesh::registerAllObjects();

  dualmesh::bindings::bindFunctions(m);
  dualmesh::bindings::bindMesh(m);
  dualmesh::bindings::bindObjects(m);
  dualmesh::bindings::bindProblem(m);
  dualmesh::bindings::bindParallel(m);
}
