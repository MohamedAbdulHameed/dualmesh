// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/base/Problem.h"
#include "dualmesh/core/ParsedFunction.h"

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>

namespace dualmesh
{

Problem::Problem(std::shared_ptr<Mesh> mesh, Method method, CoordinateSystem coord)
    : _mesh(std::move(mesh)), _method(method), _coord(coord)
{
  if (!_mesh)
    throw InputError("Problem needs a mesh.");
  if (_coord == CoordinateSystem::SphericalRadial && _mesh->dimension() != 1)
    throw InputError("The spherical radial coordinate system requires a one-dimensional mesh.");
  if (_coord == CoordinateSystem::Axisymmetric && _mesh->dimension() == 3)
    throw InputError(
        "The axisymmetric coordinate system requires a 1D (radial) or 2D (r, z) mesh.");
  // The dual mesh control domain method and the vertex-centred finite volume
  // method integrate over the control domains of the dual mesh, so every
  // element has to carry one.  The finite element method and the cell-centred
  // finite volume method need only the element itself.  Checking here, once,
  // is what lets every element type be offered to the methods that can use it
  // without any of them having to test for the others.
  if (_method == Method::DualMesh || _method == Method::FiniteVolumeVertex)
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
         << " elements, which cannot be used with the "
         << (_method == Method::DualMesh ? "dual mesh control domain method"
                                         : "vertex-centred finite volume method, which "
                                           "integrates over the same control domains,")
         << " because " << ReferenceElement::get(t).dualMeshLimitation() << ". "
         << elementTypeName(t)
         << " elements work with the finite element method (method='fem') and the "
            "cell-centred finite volume method (method='zfvm').";
      throw InputError(os.str());
    }
  }
  if (_method == Method::FiniteVolumeCell)
    _cells = std::make_shared<CellMesh>(*_mesh);
}

const CellMesh &
Problem::cellMesh() const
{
  if (!_cells)
    throw InputError("cellMesh() is only available for the cell-centred finite volume method.");
  return *_cells;
}

Index
Problem::numEntities() const
{
  return _cells ? _cells->numEntities() : _mesh->numNodes();
}

const Point &
Problem::entityPoint(Index i) const
{
  return _cells ? _cells->entityPoint(i) : _mesh->node(i);
}

std::vector<Index>
Problem::boundaryEntities(const std::string & name) const
{
  if (_cells)
  {
    if (!_mesh->hasBoundary(name))
      throw InputError("The cell-centred finite volume method needs a side set; '" + name +
                       "' is not one. Node sets have no degrees of freedom in this method.");
    return _cells->boundaryEntities(*_mesh, name);
  }
  return _mesh->boundaryNodes(name);
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
  // Grow the solution, keeping existing values.
  Vector U = Vector::Zero(numDofs());
  const int nold = numVariables() - 1;
  if (nold > 0 && _U.size() == numEntities() * nold)
    for (Index n = 0; n < numEntities(); ++n)
      for (int k = 0; k < nold; ++k)
        U[n * numVariables() + k] = _U[n * nold + k];
  _U = U;
  for (Index n = 0; n < numEntities(); ++n)
    _U[dof(n, v.index)] = v.initial_condition->value(entityPoint(n), _time);
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
  interpolateFirstOrderVariables(U);
  for (std::size_t d = 0; d < _primary_of.size(); ++d)
    if (_primary_of[d] >= 0)
      U[static_cast<Index>(d)] = U[_primary_of[d]];
}

void
Problem::interpolateFirstOrderVariables(Vector & U) const
{
  if (!_mixed_order || _cells)
    return;
  const int nv = numVariables();
  double N[kMaxElementNodes];
  Point dN[kMaxElementNodes];
  for (const auto & el : _mesh->elements())
  {
    const int corners = el.numCorners();
    const int nn = el.numNodes();
    if (corners == nn)
      continue;
    const auto & quadratic = ReferenceElement::get(el.type);
    const auto & linear = ReferenceElement::get(elementCornerType(el.type));
    for (int k = corners; k < nn; ++k)
    {
      // The linear shape functions at the reference position of node k.  A
      // node shared by several elements gets the same value from each,
      // because the linear interpolant is continuous.
      linear.shape(quadratic.node(k), N, dN);
      for (int v = 0; v < nv; ++v)
      {
        if (_full_order_rank[v] >= 0)
          continue;
        double value = 0.0;
        for (int a = 0; a < corners; ++a)
          value += N[a] * U[dof(el.nodes[a], v)];
        U[dof(el.nodes[k], v)] = value;
      }
    }
  }
}

void
Problem::addFunction(const std::string & name, FunctionPtr f)
{
  if (hasVariable(name) || _by_name.count(name))
    throw InputError("The name '" + name + "' is already used.");
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
    int i = 0;
    do
      name = type + "_" + std::to_string(i++);
    while (_by_name.count(name));
  }
  // A boundary condition given a list of variables is created once per
  // variable.  fixed_constraint names its list 'displacements'.
  const auto category = f.category(type);
  if (type == "symmetry_boundary_condition")
    return addSymmetryCondition(name, std::move(params));
  if (category == ObjectCategory::BoundaryCondition || category == ObjectCategory::NodalBC)
  {
    std::vector<std::string> keys;
    for (const char * key : {"variables", "displacements"})
      if ((std::string(key) == "variables" || type == "fixed_constraint") && params.has(key) &&
          params.isSetByUser(key) && !params.getStringList(key).empty())
        keys.push_back(key);
    if (keys.size() > 1)
      throw InputError(
          "Boundary condition '" + name +
          "': 'variables' and 'displacements' define the same thing. Give only one of them.");
    if (!keys.empty())
    {
      if (params.isSetByUser("variable") && !params.getString("variable").empty())
        throw InputError("Boundary condition '" + name + "': 'variable' and '" + keys[0] +
                         "' define the same thing. Give only one of them.");
      const auto variables = params.getStringList(keys[0]);
      std::shared_ptr<Object> last;
      for (const auto & variable : variables)
      {
        InputParameters copy = params;
        copy.set(keys[0], std::vector<std::string>{});
        copy.set("variable", variable);
        last = addObject(type, name + "_" + variable, std::move(copy));
      }
      return last;
    }
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
Problem::addSymmetryCondition(const std::string & name, InputParameters params)
{
  // The plane of the side set is normal to the coordinate axis along which
  // all its nodes share one coordinate.  The condition is the zero value of
  // the displacement along that axis.
  const auto displacements = params.getStringList("displacements");
  if (displacements.empty())
    throw InputError("Boundary condition '" + name +
                     "': symmetry_boundary_condition needs 'displacements', the displacement "
                     "variables in the order of the coordinate axes.");
  if (params.isSetByUser("variable") ||
      (params.has("variables") && params.isSetByUser("variables")))
    throw InputError("Boundary condition '" + name +
                     "': symmetry_boundary_condition takes 'displacements' only. 'variable' and "
                     "'variables' define the same thing and must not be given with it.");
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
  const int dim = std::min<int>(_mesh->dimension(), static_cast<int>(displacements.size()));
  int axis = nodes.empty() ? 0 : -1;
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
      axis = k;
      break;
    }
  }
  if (axis < 0)
    throw InputError("Boundary condition '" + name +
                     "': symmetry_boundary_condition needs a side set in a plane normal to a "
                     "coordinate axis (x, y or z constant on all its nodes), and this one is not. "
                     "An inclined symmetry plane is not available. Rotate the mesh so that the "
                     "plane is normal to an axis.");
  InputParameters copy = params;
  copy.set("displacements", std::vector<std::string>{});
  copy.set("variable", displacements[axis]);
  const std::string object_name = name + "_" + displacements[axis];
  auto obj =
      Factory::instance().create("symmetry_boundary_condition", object_name, std::move(copy));
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

bool
Problem::integratesFace(const CellFace & f) const
{
  const Index integrated = numIntegratedElements();
  const bool owner = f.owner < integrated;
  if (f.neighbor < 0)
    return owner;
  const bool neighbor = f.neighbor < integrated;
  if (owner == neighbor)
    return owner;
  if (_global_element_numbers.empty())
    throw std::logic_error("Problem: a face between an integrated cell and a ghost cell needs the "
                           "global numbers of the elements.");
  const Index here = _global_element_numbers[owner ? f.owner : f.neighbor];
  const Index there = _global_element_numbers[owner ? f.neighbor : f.owner];
  return here < there;
}

bool
Problem::integratedEntity(Index i) const
{
  if (!_cells)
    return true;
  const Index integrated = numIntegratedElements();
  if (_cells->isCell(i))
    return i < integrated;
  return _cells->faces()[_cells->faceOfBoundaryEntity(i)].owner < integrated;
}

void
Problem::setGhostElements(Index first)
{
  _num_integrated_elements = first;
  // The cells of elements added to the mesh since the problem was made.
  if (_cells && _cells->numCells() != _mesh->numElements())
    _cells = std::make_shared<CellMesh>(*_mesh);
  // The nodes added with the ghost elements take their initial values; the values of the other nodes are kept.
  const int nv = numVariables();
  const Index old_size = _U.size();
  Vector U = Vector::Zero(numEntities() * nv);
  U.head(std::min(old_size, U.size())) = _U.head(std::min(old_size, U.size()));
  for (const auto & v : _vars)
    if (v.initial_condition)
      for (Index n = old_size / std::max(nv, 1); n < numEntities(); ++n)
        U[dof(n, v.index)] = v.initial_condition->value(entityPoint(n), _time);
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
Problem::buildConstraints()
{
  _primary_of.clear();
  if (_constraints.empty())
    return;
  if (_cells)
    throw InputError("'" + _constraints.front()->name() +
                     "': the cell-centered finite volume method (zfvm) has its unknowns at cells, "
                     "and constraints join nodes; use fem, hfvm or dmcdm.");
  if (_constraints_outside)
    return;
  // Every pair joins two degrees of freedom into one unknown (a union-find), so that a node joined in several directions, such as a corner of a doubly periodic domain, ends in one group.
  const Index n = numDofs();
  std::vector<Index> parent(static_cast<std::size_t>(n));
  for (Index d = 0; d < n; ++d)
    parent[d] = d;
  const auto root = [&](Index d)
  {
    while (parent[d] != d)
      d = parent[d] = parent[parent[d]];
    return d;
  };
  std::vector<char> dependent(static_cast<std::size_t>(n), 0);
  for (const auto & c : _constraints)
  {
    c->initialSetup(*this);
    for (const auto & [secondary, primary] : c->pairs())
    {
      dependent[secondary] = 1;
      const Index a = root(secondary), b = root(primary);
      if (a != b)
        parent[std::max(a, b)] = std::min(a, b);
    }
  }
  // Each group is represented by its first member that is a primary only; a group whose every member is a secondary was joined in a loop.
  std::vector<Index> representative(static_cast<std::size_t>(n), -1);
  for (Index d = 0; d < n; ++d)
    if (!dependent[d] && representative[root(d)] < 0)
      representative[root(d)] = d;
  _primary_of.assign(static_cast<std::size_t>(n), -1);
  for (Index d = 0; d < n; ++d)
  {
    const Index r = root(d);
    if (r == d && parent[d] == d && !dependent[d])
      continue;
    if (representative[r] < 0)
      throw InputError("The constraints make a node a copy of itself through a loop; one node of "
                       "each loop must stay a primary.");
    if (representative[r] != d)
      _primary_of[d] = representative[r];
  }
  // A dependent degree of freedom has no equation of its own, so it cannot carry a prescribed value that its primary does not have.
  std::vector<char> prescribed(static_cast<std::size_t>(n), 0);
  for (const auto & bc : _nbcs)
    for (Index node : bc->nodes())
      prescribed[dof(node, bc->variable())] = 1;
  for (Index d = 0; d < n; ++d)
    if (_primary_of[d] >= 0 && prescribed[d] && !prescribed[_primary_of[d]])
      throw InputError("A node that a constraint makes a copy of another carries a prescribed "
                       "value that its primary node does not; prescribe the value on both "
                       "boundaries, or on neither.");
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
Problem::initialize()
{
  if (_initialized)
    return;
  _singularity_checked = false;
  if (_vars.empty())
    throw InputError("The problem has no variables.");
  if (_kernels.empty())
    throw InputError("The problem has no kernels.");
  buildOrderLayout();
  if (_mixed_order)
  {
    std::string first;
    for (const auto & v : _vars)
      if (v.order == VariableOrder::First)
        first += (first.empty() ? "'" : ", '") + v.name + "'";
    if (_method != Method::FiniteElement)
      throw InputError(
          "The variable " + first +
          " is declared first order on a mesh with quadratic elements. Mixed-order "
          "interpolation (the Taylor-Hood element) is a finite element construction and needs "
          "method='fem'. The dual mesh and finite volume methods interpolate every variable at "
          "the order of the mesh; for them, declare the variable with the mesh order.");
    for (const auto & el : _mesh->elements())
      if (el.numCorners() < el.numNodes() &&
          ReferenceElement::get(elementCornerType(el.type)).numNodes() > MappedPoint::kMaxCorners)
        throw InputError("Internal error: an element has more corners than MappedPoint holds.");
  }
  _props = PropertyRegistry();
  for (auto & m : _property_objects)
  {
    m->initialSetup(*this);
    m->declareProperties(_props);
    m->setupPropertyFactors(_props);
  }
  for (auto & k : _kernels)
    k->initialSetup(*this);
  for (auto & b : _ibcs)
    b->initialSetup(*this);
  if (_mixed_order)
    for (const auto & b : _ibcs)
      if (const auto * ib = dynamic_cast<const InterfaceBC *>(b.get()))
        for (int v : ib->coupledVariables())
          if (_full_order_rank[v] < 0)
            throw InputError("'" + ib->name() + "' couples the first-order variable '" +
                             _vars[v].name +
                             "' across an interface, which is not supported yet. Interface "
                             "conditions work with variables of the mesh order.");
  for (auto & b : _nbcs)
    b->initialSetup(*this);
  for (auto & l : _loads)
    l->initialSetup(*this);
  buildConstraints();
  // Check that required properties exist.
  for (auto & k : _kernels)
    for (const auto & p : k->requiredProperties())
      _props.id(p);
  // The history records: every stateful property object gets its slice of the
  // record, and every owner an (initially empty) list of points.
  _state_record = 0;
  for (auto & m : _property_objects)
  {
    m->setStateOffset(_state_record);
    _state_record += m->stateSize();
  }
  _state[0].assign(_state_record > 0 ? _mesh->numElements() : 0, OwnerState{});
  _state[1].assign(
      _state_record > 0 && _method == Method::FiniteVolumeCell ? cellMesh().faces().size() : 0,
      OwnerState{});
  buildGroups();
  buildInterfacePairs();
  markActiveDofs();
  // The mesh caches the list of exterior sides and the boundary node markers
  // the first time they are asked for.  Build them here, before any threaded
  // loop, so that no two threads try to fill them at the same time.
  _mesh->exteriorSides();
  _mesh->boundaryNodeMarkers();
  // Reference element definitions are built on first use and cached in a
  // function-local static; touch every type present in the mesh for the same
  // reason.
  for (Index e = 0; e < _mesh->numElements(); ++e)
    ReferenceElement::get(_mesh->element(e).type);
  // Threading is only used when every object and every function can be called
  // from several threads at once.  Anything defined in Python cannot, because
  // the interpreter is protected by the global interpreter lock.
  _thread_safe = true;
  for (const auto & [n, obj] : _by_name)
  {
    (void) n;
    if (!obj->threadSafe())
      _thread_safe = false;
  }
  for (const auto & [n, f] : _functions)
  {
    (void) n;
    if (f && !f->threadSafe())
      _thread_safe = false;
  }
  for (const auto & v : _vars)
    if (v.initial_condition && !v.initial_condition->threadSafe())
      _thread_safe = false;
  _initialized = true;
}

void
Problem::buildGroups()
{
  _groups.clear();
  for (const auto & k : _kernels)
  {
    auto it = std::find_if(
        _groups.begin(), _groups.end(), [&](const Group & g) { return g.spec == k->quadrature(); });
    if (it == _groups.end())
    {
      _groups.push_back({k->quadrature(), {}});
      it = _groups.end() - 1;
    }
    it->kernels.push_back(k.get());
  }
}

void
Problem::markActiveDofs()
{
  const int nv = numVariables();
  _active_dof.assign(numDofs(), 0);
  if (_cells)
  {
    for (Index c = 0; c < _cells->numCells(); ++c)
      for (int v = 0; v < nv; ++v)
      {
        const int block = _cells->cellBlock(c);
        const auto & vb = _vars[v].blocks;
        if (!vb.empty() && !vb.count(block))
          continue;
        bool has = false;
        for (const auto & k : _kernels)
          has = has || (k->variable() == v && k->activeOnBlock(block));
        if (!has)
          continue;
        _active_dof[dof(c, v)] = 1;
        for (int fi : _cells->cellFaces(c))
          if (_cells->faces()[fi].boundary_entity >= 0)
            _active_dof[dof(_cells->faces()[fi].boundary_entity, v)] = 1;
      }
    return;
  }
  for (Index e = 0; e < _mesh->numElements(); ++e)
  {
    const auto & el = _mesh->element(e);
    for (int v = 0; v < nv; ++v)
    {
      const auto & vb = _vars[v].blocks;
      if (!vb.empty() && !vb.count(el.block))
        continue;
      bool has = false;
      for (const auto & k : _kernels)
        has = has || (k->variable() == v && k->activeOnBlock(el.block));
      if (!has)
        continue;
      for (int a = 0; a < numLocalNodes(el, v); ++a)
        _active_dof[dof(el.nodes[a], v)] = 1;
    }
  }
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
    for (Index n = 0; n < numEntities(); ++n)
      _U[dof(n, v.index)] = v.initial_condition->value(entityPoint(n), _time);
  updateDependentDofs(_U);
}

std::vector<double>
Problem::values(const std::string & var) const
{
  const int v = variableIndex(var);
  std::vector<double> out(numEntities());
  for (Index n = 0; n < numEntities(); ++n)
    out[n] = _U[dof(n, v)];
  return out;
}

void
Problem::setValues(const std::string & var, const std::vector<double> & vals)
{
  const int v = variableIndex(var);
  if (static_cast<Index>(vals.size()) != numEntities())
    throw InputError("setValues: expected one value per degree of freedom entity.");
  for (Index n = 0; n < numEntities(); ++n)
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
  for (const auto & [name, values] : _element_fields)
  {
    (void) values;
    out.push_back(name);
  }
  return out;
}

double
Problem::coordFactor(const Point & x) const
{
  switch (_coord)
  {
  case CoordinateSystem::Cartesian:
    return 1.0;
  case CoordinateSystem::Axisymmetric:
    return 2.0 * M_PI * x[0];
  case CoordinateSystem::SphericalRadial:
    return 4.0 * M_PI * x[0] * x[0];
  }
  return 1.0;
}

double
Problem::boundaryEntityFactor(const Point & x) const
{
  const double c = coordFactor(x);
  return c != 0.0 ? c : 1.0;
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
  const int dim = _mesh->dimension();
  ctx.u.resize(nv);
  ctx.grad_u.resize(nv);
  // The shape functions a variable is interpolated with: the element's own,
  // or those of its corners for a first-order variable on a quadratic element.
  const int num_nodes = el.numNodes();
  const auto shapeOf = [&](int v, const double *& N, const Point *& dN, int & count)
  {
    if (_mixed_order && usesCornerShape(el, v))
    {
      if (fp.num_corners == 0)
        throw std::logic_error("dualmesh: fillContext needs mapCornerShape for a first-order "
                               "variable on a quadratic element.");
      N = fp.corner_shape;
      dN = fp.corner_gradient;
      count = fp.num_corners;
    }
    else
    {
      N = fp.N;
      dN = fp.dN;
      count = num_nodes;
    }
  };
  // The local index of variable v at local node k.  Without mixed orders it
  // is k * nv + v, computed inline; with them, localDofIndex.
  const bool mixed = _mixed_order;
  const auto index = [&](int k, int v) { return mixed ? localDofIndex(el, k, v) : k * nv + v; };
  for (int v = 0; v < nv; ++v)
  {
    const double * N;
    const Point * dN;
    int count;
    shapeOf(v, N, dN, count);
    double val = 0;
    Point g{0, 0, 0};
    for (int k = 0; k < count; ++k)
    {
      const double Uk = cur_local[index(k, v)];
      val += N[k] * Uk;
      for (int d = 0; d < dim; ++d)
        g[d] += dN[k][d] * Uk;
    }
    ADReal & u = ctx.u[v];
    ADVector3 & gr = ctx.grad_u[v];
    u = ADReal::withZeroDerivatives(val, ld);
    for (int d = 0; d < 3; ++d)
      gr[d] = ADReal::withZeroDerivatives(g[d], d < dim ? ld : 0);
    if (ld > 0)
      for (int k = 0; k < count; ++k)
      {
        const int i = index(k, v);
        u.setDerivative(i, N[k]);
        for (int d = 0; d < dim; ++d)
          gr[d].setDerivative(i, dN[k][d]);
      }
  }
  const auto interpolate = [&](const std::vector<double> & loc, int v, double & val, Point & g)
  {
    const double * N;
    const Point * dN;
    int count;
    shapeOf(v, N, dN, count);
    val = 0;
    g = {0, 0, 0};
    for (int k = 0; k < count; ++k)
    {
      const double Uk = loc[index(k, v)];
      val += N[k] * Uk;
      for (int d = 0; d < dim; ++d)
        g[d] += dN[k][d] * Uk;
    }
  };
  if (lag_local)
  {
    ctx.u_lag.resize(nv);
    ctx.grad_u_lag.resize(nv);
    for (int v = 0; v < nv; ++v)
    {
      double val;
      Point g;
      interpolate(*lag_local, v, val, g);
      ctx.u_lag[v] = ADReal(val);
      ctx.grad_u_lag[v] = {ADReal(g[0]), ADReal(g[1]), ADReal(g[2])};
    }
  }
  else
  {
    ctx.u_lag = ctx.u;
    ctx.grad_u_lag = ctx.grad_u;
  }
  ctx.u_old.assign(nv, 0.0);
  ctx.grad_u_old.assign(nv, {0, 0, 0});
  for (int v = 0; v < nv; ++v)
  {
    if (old_local)
      interpolate(*old_local, v, ctx.u_old[v], ctx.grad_u_old[v]);
    else
    {
      ctx.u_old[v] = ctx.u[v].value();
      for (int d = 0; d < 3; ++d)
        ctx.grad_u_old[v][d] = ctx.grad_u[v][d].value();
    }
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
  auto & owners = _state[ctx.state_domain];
  const std::size_t R = static_cast<std::size_t>(_state_record);
  if (ctx.state_key == kCentroidStateKey)
  {
    // Output at a centroid: the average committed history of the element's
    // volume points.
    scratch_old.assign(R, 0.0);
    scratch_new.assign(R, 0.0);
    if (static_cast<std::size_t>(ctx.state_owner) < owners.size())
    {
      const auto & o = owners[ctx.state_owner];
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
  auto & o = owners.at(static_cast<std::size_t>(ctx.state_owner));
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
  for (auto & owners : _state)
    for (auto & o : owners)
    {
      o.committed = o.trial;
      o.start = o.trial;
    }
}

void
Problem::startStateFromCommitted()
{
  for (auto & owners : _state)
    for (auto & o : owners)
      o.start = o.committed;
}

void
Problem::startStateFromTrial()
{
  for (auto & owners : _state)
    for (auto & o : owners)
      o.start = o.trial;
}

Problem::StateSnapshot
Problem::snapshotState() const
{
  StateSnapshot s;
  for (int d = 0; d < 2; ++d)
    for (const auto & o : _state[d])
    {
      s.keys[d].push_back(o.keys);
      s.values[d].push_back(o.committed);
    }
  return s;
}

void
Problem::restoreState(const StateSnapshot & s)
{
  for (int d = 0; d < 2; ++d)
  {
    if (s.keys[d].size() != _state[d].size())
      throw InputError("restoreState: the snapshot belongs to a different problem.");
    for (std::size_t i = 0; i < _state[d].size(); ++i)
    {
      auto & o = _state[d][i];
      o.keys = s.keys[d][i];
      o.committed = s.values[d][i];
      o.start = o.committed;
      o.trial = o.committed;
    }
  }
}

std::vector<std::vector<double>>
Problem::stateAtElements() const
{
  std::vector<std::vector<double>> out(_mesh->numElements());
  QpContext ctx;
  for (Index e = 0; e < _mesh->numElements(); ++e)
  {
    ctx.state_domain = 0;
    ctx.state_owner = e;
    ctx.state_key = kCentroidStateKey;
    bindState(ctx);
    if (ctx.state_old)
      out[e].assign(ctx.state_old, ctx.state_old + _state_record);
  }
  return out;
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
