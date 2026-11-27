// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The definition of a problem: its variables, functions and objects, its values and element
// fields, and the context of an integration point, with the history and the properties there.
#include "dualmesh/base/Problem.h"
#include "dualmesh/core/ParsedFunction.h"

#include <algorithm>
#include <ranges>
#include <set>
#include <sstream>
#include <utility>

namespace dualmesh
{

Problem::Problem(std::shared_ptr<Mesh> mesh, Method method, CoordinateSystem coord)
    : _mesh(std::move(mesh)), _method(method), _coord(coord)
{
  // The systems of the physics come from the modules, which register them with their objects.
  _flow = Factory::instance().makeFlowSystem();
  _solid = Factory::instance().solidSystem();
  if (!_flow || !_solid)
    throw std::logic_error("dualmesh: the modules are not registered. Call registerAllObjects() "
                           "(dualmesh/modules/Registry.h) before making a Problem.");
  if (!_mesh)
    throw InputError("Problem needs a mesh.");
  if (_mesh->numElements() == 0)
    throw InputError("The mesh has no elements, so there is nothing to solve on. A mesh built node "
                     "by node needs its elements too (add_element, or mesh_from_arrays with "
                     "cells).");
  if (_coord == CoordinateSystem::Axisymmetric || _coord == CoordinateSystem::SphericalRadial)
  {
    // The radius is x: a mesh that reaches x < 0 has no meaning in these coordinates, and the
    // factor 2 pi r of the integrals turns negative there.
    const auto box = _mesh->boundingBox();
    double size = 0.0;
    for (int k = 0; k < 3; ++k)
      size = std::max(size, box.second[k] - box.first[k]);
    if (box.first[0] < -1e-12 * size)
    {
      std::ostringstream os;
      os << "In " << (_coord == CoordinateSystem::Axisymmetric ? "axisymmetric" : "spherical")
         << " coordinates the radius is x, which must not be negative, and this mesh reaches x = "
         << box.first[0] << ". Place the axis at x = 0 and the body at x >= 0.";
      throw InputError(os.str());
    }
  }
  if (_coord == CoordinateSystem::SphericalRadial && _mesh->dimension() != 1)
    throw InputError("The spherical radial coordinate system requires a one-dimensional mesh.");
  if (_coord == CoordinateSystem::Axisymmetric && _mesh->dimension() == 3)
    throw InputError(
        "The axisymmetric coordinate system requires a 1D (radial) or 2D (r, z) mesh.");
  setMethod(method);
}

void
Problem::setMethod(Method method)
{
  if (_initialized)
    throw InputError("The method of a problem is chosen before it is initialised.");
  _method = method;
  // The dual mesh control domain method integrates over the control domains of the dual mesh, so
  // every element has to carry one.  The finite element method needs only the element itself.
  if (isDual())
  {
    std::set<ElementType> refused;
    for (const auto & el : _mesh->elements())
      if (!ReferenceElement::get(el.type).supportsDualMesh())
        refused.insert(el.type);
    if (!refused.empty())
    {
      const ElementType t = *refused.begin();
      std::ostringstream os;
      os << "The mesh contains " << elementTypeName(t)
         << " elements, which cannot be used with the dual mesh control domain method because "
         << ReferenceElement::get(t).dualMeshLimitation() << ". " << elementTypeName(t)
         << " elements work with the finite element method (method='FEM').";
      throw InputError(os.str());
    }
  }
}

void
Problem::setVariableUnit(const std::string & name, const std::string & unit)
{
  for (auto & v : _vars)
    if (v.name == name)
    {
      v.unit = unit;
      return;
    }
  throw InputError("Variable '" + name + "' does not exist.");
}

void
Problem::extendVariableBlocks(const std::string & name, const std::vector<std::string> & blocks)
{
  Variable & v = _vars[variableIndex(name)];
  // An empty set is every block: it stays so, or becomes so.
  if (v.blocks.empty())
    return;
  if (blocks.empty())
    v.blocks.clear();
  else
    for (const auto & b : blocks)
      v.blocks.insert(_mesh->blockId(b));
  _initialized = false;
}

int
Problem::addVariable(const std::string & name,
                     const std::vector<std::string> & blocks,
                     FunctionPtr ic,
                     double scaling,
                     VariableOrder order)
{
  if (hasVariable(name))
    throw InputError("Variable '" + name + "' already exists.");
  if (_by_name.count(name) || _functions.count(name))
    throw InputError("The name '" + name + "' is already used.");
  Variable v;
  v.name = name;
  v.index = numVariables();
  for (const auto & b : blocks)
    v.blocks.insert(_mesh->blockId(b));
  v.initial_condition = ic ? ic : std::make_shared<ConstantFunction>(0.0);
  v.scaling = scaling;
  v.order = order;
  _vars.push_back(v);
  buildOrderLayout();
  // Automatic differentiation seeds one derivative slot per local degree of
  // freedom (one per node and variable, the corners only for a first-order
  // variable), so the element with the most of them sets the limit.
  int max_local = 1;
  ElementType largest = ElementType::Edge2;
  for (const auto & el : _mesh->elements())
    if (numLocalDofs(el) > max_local)
    {
      max_local = numLocalDofs(el);
      largest = el.type;
    }
  if (max_local > kMaxDerivatives)
  {
    std::ostringstream os;
    os << "Too many variables for this mesh. The largest element is " << elementTypeName(largest)
       << " with " << elementNumNodes(largest) << " nodes, which carries " << max_local
       << " degrees of freedom for these " << numVariables()
       << " variables, and automatic differentiation seeds one slot per local degree of freedom, "
       << "within a limit of " << kMaxDerivatives << " slots. Every further variable adds "
       << elementNumNodes(largest) << " slots on this element ("
       << elementNumNodes(elementCornerType(largest))
       << " for a first-order variable, which lives on the corners only). Rebuild with "
       << "-DDUALMESH_MAX_AD_DERIVATIVES=<n>, with n at least the total the problem needs "
       << "(here at least " << max_local << "), to raise it.";
    _vars.pop_back();
    buildOrderLayout();
    throw InputError(os.str());
  }
  _initialized = false;
  _dependent_of.clear();
  // Grow the solution, keeping existing values.
  Vector U = Vector::Zero(numDofs());
  const int nold = numVariables() - 1;
  if (nold > 0 && _U.size() == _mesh->numNodes() * nold)
    for (Index n = 0; n < _mesh->numNodes(); ++n)
      for (int k = 0; k < nold; ++k)
        U[n * numVariables() + k] = _U[n * nold + k];
  _U = U;
  for (Index n = 0; n < _mesh->numNodes(); ++n)
    _U[dof(n, v.index)] = v.initial_condition->value(_mesh->node(n), _time);
  updateDependentDofs(_U);
  return v.index;
}

void
Problem::buildOrderLayout()
{
  const int nv = numVariables();
  bool any_first = false;
  for (const auto & v : _vars)
    any_first = any_first || v.order == VariableOrder::First;
  bool any_quadratic = false;
  for (const auto & el : _mesh->elements())
    any_quadratic = any_quadratic || el.numCorners() < el.numNodes();
  _mixed_order = any_first && any_quadratic;
  _full_order_rank.assign(nv, -1);
  _num_full_order = 0;
  for (int v = 0; v < nv; ++v)
    if (!_mixed_order || _vars[v].order == VariableOrder::Mesh)
      _full_order_rank[v] = _num_full_order++;
}

void
Problem::updateDependentDofs(Vector & U) const
{
  for (std::size_t d = 0; d < _dependent_of.size(); ++d)
    if (!_dependent_of[d].empty())
    {
      double value = 0.0;
      for (const auto & [p, w] : _dependent_of[d])
        value += w * U[p];
      U[static_cast<Index>(d)] = value;
    }
}

void
Problem::addFunction(const std::string & name, FunctionPtr f)
{
  // Objects resolve the functions they name when they are set up, so a function is never replaced:
  // an object that resolved the old one would keep reading it.
  if (_functions.count(name))
    throw InputError("The name '" + name +
                     "' is already a registered function, and replacing it would leave the "
                     "objects that resolved it reading the old one. Register the function or the "
                     "parameter under a new name.");
  if (hasVariable(name))
    throw InputError("The name '" + name +
                     "' is already a variable. Register the function or the parameter under a "
                     "new name.");
  if (_by_name.count(name))
    throw InputError("The name '" + name +
                     "' is already the name of an object. Register the function or the parameter "
                     "under a new name.");
  _functions[name] = std::move(f);
}

int
Problem::variableIndex(const std::string & name) const
{
  for (const auto & v : _vars)
    if (v.name == name)
      return v.index;
  std::ostringstream os;
  std::vector<std::string> names;
  for (const auto & v : _vars)
    names.push_back(v.name);
  os << "Unknown variable '" << name << "'." << didYouMean(name, names) << " Variables:";
  for (const auto & n : names)
    os << " " << n;
  throw InputError(os.str());
}

bool
Problem::hasVariable(const std::string & name) const
{
  for (const auto & v : _vars)
    if (v.name == name)
      return true;
  return false;
}

FunctionPtr
Problem::function(const std::string & name) const
{
  auto it = _functions.find(name);
  if (it != _functions.end())
    return it->second;
  // Not a registered name: the text may itself be an expression in x, y, z
  // and t, as in value = "sin(pi*x) * exp(-t)".  That saves registering a
  // function for a one-off coefficient, and it is compiled to C++, so it costs
  // nothing at assembly time and keeps the assembly threaded.
  std::string why;
  if (ParsedFunction::isExpression(name, &why))
    return std::make_shared<ParsedFunction>(name);
  std::ostringstream os;
  os << "'" << name << "' is neither a registered function nor an expression in x, y, z and t ("
     << why << ").";
  if (!_functions.empty())
  {
    os << " Registered functions:";
    for (const auto & [n, _] : _functions)
      os << " " << n;
  }
  throw InputError(os.str());
}

std::shared_ptr<Object>
Problem::addObject(const std::string & type, const std::string & name_in, InputParameters params)
{
  const auto & f = Factory::instance();
  std::string name = name_in;
  if (name.empty())
  {
    // A name is taken when an object has it, or when the objects of a list of variables carry it
    // as their prefix (name_variable).
    const auto taken = [this](const std::string & n)
    {
      if (_by_name.count(n))
        return true;
      const auto it = _by_name.lower_bound(n + "_");
      return it != _by_name.end() && it->first.compare(0, n.size() + 1, n + "_") == 0;
    };
    int i = 0;
    do
      name = type + "_" + std::to_string(i++);
    while (taken(name));
  }
  const auto category = f.category(type);
  // A model kernel added without 'variable' is added once for every variable of the model.
  const auto & model = params.modelVariables();
  if (!model.empty() && !params.isSetByUser("variable"))
  {
    std::vector<std::string> variables;
    for (const auto & key : model)
    {
      if (!params.isSet(key))
        continue;
      const auto & raw = params.getRaw(key);
      if (const auto * one = std::get_if<std::string>(&raw))
      {
        if (!one->empty())
          variables.push_back(*one);
      }
      else if (const auto * list = std::get_if<std::vector<std::string>>(&raw))
        for (const auto & v : *list)
          if (!v.empty())
            variables.push_back(v);
    }
    if (variables.empty())
      throw InputError(type + " '" + name + "': give the variables of the model.");
    std::shared_ptr<Object> last;
    for (const auto & variable : variables)
    {
      InputParameters copy = params;
      copy.set("variable", variable);
      last = addObject(type, name + "_" + variable, std::move(copy));
    }
    return last;
  }
  if (params.normalComponent())
    return addNormalComponentCondition(type, name, std::move(params));
  // An object given a list of variables is added once for each of them.
  if (params.has("variable") && params.isSetByUser("variable"))
  {
    const auto & raw = params.getRaw("variable");
    if (const auto * list = std::get_if<std::vector<std::string>>(&raw); list && list->size() > 1)
    {
      std::shared_ptr<Object> last;
      for (const auto & variable : *list)
      {
        InputParameters copy = params;
        copy.set("variable", variable);
        last = addObject(type, name + "_" + variable, std::move(copy));
      }
      return last;
    }
  }
  if (type == "Dirichlet")
  {
    // One value for one variable on one boundary: a second Dirichlet condition there would
    // silently replace the first.  Boundaries that only meet (a corner of two side sets) may carry
    // conditions of their own.
    auto boundaries = params.getStringList("boundary");
    std::sort(boundaries.begin(), boundaries.end());
    std::string key = params.getStringList("variable").front() + " on";
    for (const auto & b : boundaries)
      key += " " + b;
    if (!_dirichlet_keys.insert(key).second)
      throw InputError("Dirichlet '" + name + "': the value of " + key +
                       " is already prescribed by another condition. Give one value per "
                       "variable and boundary.");
  }
  auto obj = f.create(type, name, std::move(params));
  switch (category)
  {
  case ObjectCategory::Kernel:
    addKernel(std::static_pointer_cast<Kernel>(obj));
    break;
  case ObjectCategory::BoundaryCondition:
    addIntegratedBC(std::static_pointer_cast<IntegratedBC>(obj));
    break;
  case ObjectCategory::NodalBC:
    addNodalBC(std::static_pointer_cast<NodalBC>(obj));
    break;
  case ObjectCategory::NodalLoad:
    addNodalLoad(std::static_pointer_cast<NodalLoad>(obj));
    break;
  case ObjectCategory::Property:
    addProperty(std::static_pointer_cast<Property>(obj));
    break;
  case ObjectCategory::Constraint:
    addConstraint(std::static_pointer_cast<Constraint>(obj));
    break;
  }
  return obj;
}

std::shared_ptr<Object>
Problem::addNormalComponentCondition(const std::string & type,
                                     const std::string & name,
                                     InputParameters params)
{
  // The plane of the side set is normal to the coordinate axis along which
  // all its nodes share one coordinate.  The condition is the zero value of
  // the displacement along that axis.
  const auto displacements =
      params.isSet("variable") ? params.getStringList("variable") : std::vector<std::string>{};
  if (displacements.size() < 2)
    throw InputError("Boundary condition '" + name +
                     "': symmetry needs 'variable', the displacement variables in the order of "
                     "the coordinate axes.");
  const auto boundaries = params.getStringList("boundary");
  std::vector<Index> nodes;
  for (const auto & b : boundaries)
  {
    if (!_mesh->hasBoundary(b))
      throw InputError("Boundary condition '" + name + "': the mesh has no boundary '" + b + "'.");
    const auto list = _mesh->boundaryNodes(b);
    nodes.insert(nodes.end(), list.begin(), list.end());
  }
  const auto box = _mesh->boundingBox();
  double size = 0.0;
  for (int k = 0; k < 3; ++k)
    size = std::max(size, box.second[k] - box.first[k]);
  const double tolerance = 1e-8 * std::max(size, 1e-300);
  const int dim = std::min<int>(_mesh->spaceDimension(), static_cast<int>(displacements.size()));
  // The plane of symmetry is normal to the one axis along which every node of the boundary has
  // the same coordinate.  A boundary constant along two axes (one node, or nodes on a line in
  // three dimensions) does not determine the plane.
  int axis = nodes.empty() ? 0 : -1;
  int constant_axes = 0;
  for (int k = 0; k < dim && !nodes.empty(); ++k)
  {
    double lo = _mesh->node(nodes[0])[k], hi = lo;
    for (Index n : nodes)
    {
      lo = std::min(lo, _mesh->node(n)[k]);
      hi = std::max(hi, _mesh->node(n)[k]);
    }
    if (hi - lo <= tolerance)
    {
      if (axis < 0)
        axis = k;
      ++constant_axes;
    }
  }
  if (constant_axes > 1)
    throw InputError("Boundary condition '" + name +
                     "': symmetry needs a boundary that determines its plane, and the nodes of "
                     "this one have the same coordinate along " +
                     std::to_string(constant_axes) +
                     " axes (a single node, or a line in three dimensions). Give a side set or "
                     "a node set that spans the plane of symmetry.");
  if (axis < 0)
    throw InputError("Boundary condition '" + name +
                     "': symmetry needs a side set in a plane normal to a "
                     "coordinate axis (x, y or z constant on all its nodes), and this one is not. "
                     "An inclined symmetry plane is not available. Rotate the mesh so that the "
                     "plane is normal to an axis.");
  InputParameters copy = params;
  copy.set("variable", displacements[axis]);
  const std::string object_name = name + "_" + displacements[axis];
  auto obj = Factory::instance().create(type, object_name, std::move(copy));
  addNodalBC(std::static_pointer_cast<NodalBC>(obj));
  return obj;
}

namespace
{
template <typename T>
void
registerName(std::map<std::string, std::shared_ptr<Object>> & names, const std::shared_ptr<T> & obj)
{
  if (!obj)
    throw InputError("Cannot add an empty object.");
  if (names.count(obj->name()))
    throw InputError("An object named '" + obj->name() + "' already exists.");
  names[obj->name()] = obj;
}
} // namespace

void
Problem::addKernel(std::shared_ptr<Kernel> k)
{
  registerName(_by_name, k);
  _kernels.push_back(std::move(k));
  _initialized = false;
}
void
Problem::addIntegratedBC(std::shared_ptr<IntegratedBC> bc)
{
  registerName(_by_name, bc);
  _ibcs.push_back(std::move(bc));
  _initialized = false;
}
void
Problem::addNodalBC(std::shared_ptr<NodalBC> bc)
{
  registerName(_by_name, bc);
  _nbcs.push_back(std::move(bc));
  _initialized = false;
}
void
Problem::addNodalLoad(std::shared_ptr<NodalLoad> load)
{
  registerName(_by_name, load);
  _loads.push_back(std::move(load));
  _initialized = false;
}

void
Problem::setGhostElements(Index first)
{
  _num_integrated_elements = first;
  // The nodes added with the ghost elements take their initial values; the values of the other nodes are kept.
  const int nv = numVariables();
  const Eigen::Index old_size = _U.size();
  Vector U = Vector::Zero(_mesh->numNodes() * nv);
  const Eigen::Index kept = std::min<Eigen::Index>(old_size, U.size());
  U.head(kept) = _U.head(kept);
  for (const auto & v : _vars)
    if (v.initial_condition)
      for (Index n = old_size / std::max(nv, 1); n < _mesh->numNodes(); ++n)
        U[dof(n, v.index)] = v.initial_condition->value(_mesh->node(n), _time);
  _U = U;
  _initialized = false;
}

void
Problem::addInterfaceCandidates(const std::string & sideset, const std::vector<Side> & sides)
{
  auto & list = _interface_candidates[sideset];
  list.insert(list.end(), sides.begin(), sides.end());
  _initialized = false;
}

void
Problem::addConstraint(std::shared_ptr<Constraint> constraint)
{
  registerName(_by_name, constraint);
  _constraints.push_back(std::move(constraint));
  _initialized = false;
}

void
Problem::addProperty(std::shared_ptr<Property> m)
{
  registerName(_by_name, m);
  _property_objects.push_back(std::move(m));
  _initialized = false;
}

std::shared_ptr<Object>
Problem::object(const std::string & name) const
{
  auto it = _by_name.find(name);
  if (it == _by_name.end())
    throw InputError("Unknown object '" + name + "'.");
  return it->second;
}

std::vector<std::string>
Problem::objectNames() const
{
  std::vector<std::string> out;
  for (const auto & [n, _] : _by_name)
    out.push_back(n);
  return out;
}

void
Problem::setEigenvalueStudy(bool on)
{
  if (on == _eigenvalue_study)
    return;
  _eigenvalue_study = on;
  if (_initialized)
    buildGroups();
}

void
Problem::overrideActiveDofs(const std::vector<char> & active)
{
  initialize();
  if (static_cast<Index>(active.size()) != numDofs())
    throw InputError("overrideActiveDofs: expected one flag per degree of freedom.");
  _active_dof = active;
}

void
Problem::applyInitialConditions()
{
  for (const auto & v : _vars)
    for (Index n = 0; n < _mesh->numNodes(); ++n)
      _U[dof(n, v.index)] = v.initial_condition->value(_mesh->node(n), _time);
  updateDependentDofs(_U);
}

std::vector<double>
Problem::values(const std::string & var) const
{
  const int v = variableIndex(var);
  std::vector<double> out(_mesh->numNodes());
  for (Index n = 0; n < _mesh->numNodes(); ++n)
    out[n] = _U[dof(n, v)];
  return out;
}

void
Problem::setValues(const std::string & var, const std::vector<double> & vals)
{
  const int v = variableIndex(var);
  if (static_cast<Index>(vals.size()) != _mesh->numNodes())
    throw InputError("setValues: expected one value per degree of freedom entity.");
  for (Index n = 0; n < _mesh->numNodes(); ++n)
    _U[dof(n, v)] = vals[n];
  updateDependentDofs(_U);
}

void
Problem::setElementField(const std::string & name, const std::vector<double> & values)
{
  if (static_cast<Index>(values.size()) != _mesh->numElements())
    throw InputError("Element field '" + name + "' needs one value per element (" +
                     std::to_string(_mesh->numElements()) + "), got " +
                     std::to_string(values.size()) + ".");
  auto & field = _element_fields[name];
  field.assign(values.begin(), values.end());
}

const std::vector<double> &
Problem::elementField(const std::string & name)
{
  auto it = _element_fields.find(name);
  if (it == _element_fields.end())
    it = _element_fields.emplace(name, std::vector<double>(_mesh->numElements(), 0.0)).first;
  return it->second;
}

std::vector<std::string>
Problem::elementFieldNames() const
{
  std::vector<std::string> out;
  for (const auto & name : std::views::keys(_element_fields))
    out.push_back(name);
  return out;
}

void
Problem::fillShifts(QpContext & ctx,
                    const Element & el,
                    const MappedPoint & fp,
                    const std::vector<double> & rate_local,
                    const std::vector<double> & accel_local) const
{
  // The shifts of the time derivatives of the second-order variables, interpolated as the
  // variables are (QpContext::rate and acceleration).
  const int nv = numVariables();
  const int dim = _mesh->spaceDimension();
  for (int v = 0; v < nv; ++v)
  {
    if (!ctx.secondOrder(v))
      continue;
    const bool corner = _mixed_order && usesCornerShape(el, v);
    const double * N = corner ? fp.corner_shape : fp.N;
    const Point * dN = corner ? fp.corner_gradient : fp.dN;
    const int count = corner ? fp.num_corners : el.numNodes();
    const auto rate = [&](int k) { return rate_local[localDofIndex(el, k, v)]; };
    const auto accel = [&](int k) { return accel_local[localDofIndex(el, k, v)]; };
    ctx.rate_shift[v] = interpolate(N, dN, count, dim, rate, ctx.grad_rate_shift[v]);
    ctx.accel_shift[v] = interpolate(N, count, accel);
  }
}

Problem::Shape
Problem::variableShape(const Element & el, const MappedPoint & fp, int v) const
{
  if (!(_mixed_order && usesCornerShape(el, v)))
    return {fp.N, fp.dN, el.numNodes()};
  if (fp.num_corners == 0)
    throw std::logic_error("dualmesh: fillContext needs mapCornerShape for a first-order "
                           "variable on a quadratic element.");
  return {fp.corner_shape, fp.corner_gradient, fp.num_corners};
}

void
Problem::fillContext(QpContext & ctx,
                     const Element & el,
                     const MappedPoint & fp,
                     const std::vector<double> & cur_local,
                     const std::vector<double> * lag_local,
                     const std::vector<double> * old_local,
                     int ld) const
{
  // ld > 0: seed derivatives, one per local degree of freedom (see
  // localDofIndex); ld == 0: values only.  The local vectors are in the same
  // compact layout.
  const int nv = numVariables();
  const int dim = _mesh->spaceDimension();
  ctx.x_field = fp.x;
  ctx.u.resize(nv);
  ctx.grad_u.resize(nv);
  for (int v = 0; v < nv; ++v)
  {
    const Shape shape = variableShape(el, fp, v);
    const auto current = [&](int k) { return cur_local[localIndex(el, k, v)]; };
    Point g;
    const double val = interpolate(shape.N, shape.dN, shape.count, dim, current, g);
    ADReal & u = ctx.u[v];
    ADVector3 & gr = ctx.grad_u[v];
    u = ADReal::withZeroDerivatives(val, ld);
    for (int d = 0; d < 3; ++d)
      gr[d] = ADReal::withZeroDerivatives(g[d], d < dim ? ld : 0);
    if (ld > 0)
      for (int k = 0; k < shape.count; ++k)
      {
        const int i = localIndex(el, k, v);
        u.setDerivative(i, shape.N[k]);
        for (int d = 0; d < dim; ++d)
          gr[d].setDerivative(i, shape.dN[k][d]);
      }
  }
  // The value of variable v from the local values @p local, and its gradient @p g.
  const auto interpolateLocal = [&](const std::vector<double> & local, int v, Point & g)
  {
    const Shape shape = variableShape(el, fp, v);
    const auto nodal = [&](int k) { return local[localIndex(el, k, v)]; };
    return interpolate(shape.N, shape.dN, shape.count, dim, nodal, g);
  };
  if (lag_local)
  {
    ctx.u_lag.resize(nv);
    ctx.grad_u_lag.resize(nv);
    for (int v = 0; v < nv; ++v)
    {
      Point g;
      ctx.u_lag[v] = ADReal(interpolateLocal(*lag_local, v, g));
      ctx.grad_u_lag[v] = {ADReal(g[0]), ADReal(g[1]), ADReal(g[2])};
    }
  }
  else if (ctx.mode == LinearizationMode::Picard)
  {
    // Only Picard iteration reads the lagged values (QpContext::coefficientValue).
    ctx.u_lag = ctx.u;
    ctx.grad_u_lag = ctx.grad_u;
  }
  fillLaplacian(ctx, el, fp, cur_local, old_local, ld);
  ctx.u_old.assign(nv, 0.0);
  ctx.grad_u_old.assign(nv, {0, 0, 0});
  for (int v = 0; v < nv; ++v)
  {
    if (old_local)
      ctx.u_old[v] = interpolateLocal(*old_local, v, ctx.grad_u_old[v]);
    else
    {
      ctx.u_old[v] = ctx.u[v].value();
      for (int d = 0; d < 3; ++d)
        ctx.grad_u_old[v][d] = ctx.grad_u[v][d].value();
    }
  }
}

void
Problem::fillLaplacian(QpContext & ctx,
                       const Element & el,
                       const MappedPoint & fp,
                       const std::vector<double> & cur_local,
                       const std::vector<double> * old_local,
                       int ld) const
{
  // The Laplacian of the variables of a quadratic element, in the coordinates of the problem: the
  // Cartesian one, plus the hoop term (n/r) du/dr of the divergence of the gradient, with n the
  // hoop factor (1 axisymmetric, 2 spherical).  A linear element has none.
  ctx.laplacian.clear();
  ctx.laplacian_old.clear();
  const int dim = _mesh->spaceDimension();
  if (!(_needs_laplacian && ReferenceElement::get(el.type).order() == 2 &&
        ReferenceElement::get(el.type).dimension() == dim))
    return;
  const int nv = numVariables();
  const int num_nodes = el.numNodes();
  double L[kMaxElementNodes];
  mapLaplacian(*_mesh, el, fp, L);
  const double r = fp.x[0];
  const double radial = hoopOverRadius(_coord, r);
  ctx.laplacian.assign(nv, ADReal(0.0));
  ctx.laplacian_old.assign(nv, 0.0);
  for (int v = 0; v < nv; ++v)
  {
    if (_mixed_order && usesCornerShape(el, v))
      continue;
    double value = 0.0, old = 0.0;
    for (int k = 0; k < num_nodes; ++k)
    {
      const double c = L[k] + radial * fp.dN[k][0];
      value += c * cur_local[localIndex(el, k, v)];
      if (old_local)
        old += c * (*old_local)[localIndex(el, k, v)];
    }
    ADReal & lap = ctx.laplacian[v];
    lap = ADReal::withZeroDerivatives(value, ld);
    if (ld > 0)
      for (int k = 0; k < num_nodes; ++k)
        lap.setDerivative(localIndex(el, k, v), L[k] + radial * fp.dN[k][0]);
    ctx.laplacian_old[v] = old_local ? old : value;
  }
}

void
Problem::bindState(QpContext & ctx) const
{
  thread_local std::vector<double> scratch_old, scratch_new;
  if (_state_record == 0 || ctx.state_owner < 0)
  {
    ctx.state_old = nullptr;
    ctx.state_new = nullptr;
    return;
  }
  const std::size_t R = static_cast<std::size_t>(_state_record);
  if (ctx.state_key == kCentroidStateKey)
  {
    // Output at a centroid: the average committed history of the element's
    // volume points.
    scratch_old.assign(R, 0.0);
    scratch_new.assign(R, 0.0);
    if (static_cast<std::size_t>(ctx.state_owner) < _state.size())
    {
      const auto & o = _state[ctx.state_owner];
      int count = 0;
      for (std::size_t i = 0; i < o.keys.size(); ++i)
        if ((o.keys[i] >> 56) == 0)
        {
          for (std::size_t k = 0; k < R; ++k)
            scratch_old[k] += o.committed[i * R + k];
          ++count;
        }
      if (count > 0)
        for (double & v : scratch_old)
          v /= count;
    }
    ctx.state_old = scratch_old.data();
    ctx.state_new = scratch_new.data();
    return;
  }
  auto & o = _state.at(static_cast<std::size_t>(ctx.state_owner));
  std::size_t i = 0;
  while (i < o.keys.size() && o.keys[i] != ctx.state_key)
    ++i;
  if (i == o.keys.size())
  {
    o.keys.push_back(ctx.state_key);
    o.start.resize(o.start.size() + R, 0.0);
    o.trial.resize(o.trial.size() + R, 0.0);
    o.committed.resize(o.committed.size() + R, 0.0);
  }
  ctx.state_old = o.start.data() + i * R;
  ctx.state_new = o.trial.data() + i * R;
}

void
Problem::commitState()
{
  for (auto & o : _state)
  {
    o.committed = o.trial;
    o.start = o.trial;
  }
  for (auto & bc : _ibcs)
    bc->commitHistory();
}

void
Problem::startStateFromCommitted()
{
  for (auto & o : _state)
    o.start = o.committed;
  for (auto & bc : _ibcs)
    bc->startHistory(true);
}

void
Problem::startStateFromTrial()
{
  for (auto & o : _state)
    o.start = o.trial;
  for (auto & bc : _ibcs)
    bc->startHistory(false);
}

Problem::StateSnapshot
Problem::snapshotState() const
{
  StateSnapshot s;
  for (const auto & o : _state)
  {
    s.keys.push_back(o.keys);
    s.values.push_back(o.committed);
  }
  return s;
}

void
Problem::restoreState(const StateSnapshot & s)
{
  if (s.keys.size() != _state.size())
    throw InputError("restoreState: the snapshot belongs to a different problem.");
  for (std::size_t i = 0; i < _state.size(); ++i)
  {
    auto & o = _state[i];
    o.keys = s.keys[i];
    o.committed = s.values[i];
    o.start = o.committed;
    o.trial = o.committed;
  }
}

void
Problem::evaluateProperties(QpContext & ctx) const
{
  if (_property_objects.empty())
    return;
  bindState(ctx);
  ctx.properties.assign(_props.size(), ADReal(0.0));
  for (const auto & m : _property_objects)
    if (m->activeOnBlock(ctx.block))
    {
      m->computeProperties(ctx);
      m->applyPropertyFactors(ctx);
    }
}

} // namespace dualmesh
