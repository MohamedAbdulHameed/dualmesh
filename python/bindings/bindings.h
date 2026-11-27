// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The parts of the pybind11 bindings: one function per subsystem, each adding its classes and
// functions to the module, and the conversions of Python values that they share.  The order of
// the calls in module.cpp registers every class before a default argument or a base class uses it.
#pragma once

#include "dualmesh/base/Problem.h"
#include "dualmesh/core/Function.h"
#include "dualmesh/core/InputParameters.h"

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

namespace py = pybind11;

namespace dualmesh::bindings
{

/// The automatic differentiation numbers, the compiled expressions and the functions.
void bindFunctions(py::module_ & m);
/// The mesh, the element types, the mesh generators and the mesh hierarchies.
void bindMesh(py::module_ & m);
/// The input parameters, the state at an integration point, the objects and their registry.
void bindObjects(py::module_ & m);
/// The solver options and results, the projection time integration and the problem.
void bindProblem(py::module_ & m);
/// The build information, MPI, the mesh partition and the distributed problem.
void bindParallel(py::module_ & m);

/// A function from a Python value: a compiled function, a Python callable f(x, y, z, t) or a
/// number.
FunctionPtr toFunction(const py::object & o);
/// Convert a Python value into a ParameterValue.
ParameterValue toParameter(const py::handle & v);
/// The parameters @p p with the values of @p kwargs set.
InputParameters paramsFromKwargs(InputParameters p, const py::kwargs & kwargs);
/// The method of a name: 'FEM' or 'DMCDM', in either case.
Method methodFromString(const std::string & given);
/// The coordinate system of a name: 'cartesian', 'axisymmetric' or 'spherical'.
CoordinateSystem coordFromString(const std::string & s);

/// A one-dimensional NumPy array that holds a copy of the @p size values at @p data.
template <class T>
py::array_t<T>
toArray(py::ssize_t size, const T * data)
{
  return py::array_t<T>(size, data);
}

/// A one-dimensional NumPy array that holds a copy of @p values.
template <class T>
py::array_t<T>
toArray(const std::vector<T> & values)
{
  return toArray(static_cast<py::ssize_t>(values.size()), values.data());
}

/// A one-dimensional NumPy array that holds a copy of @p values.
inline py::array_t<double>
toArray(const Vector & values)
{
  return toArray(values.size(), values.data());
}

} // namespace dualmesh::bindings
