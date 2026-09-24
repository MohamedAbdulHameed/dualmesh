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

int
Problem::addVariable(const std::string & name,
                     const std::vector<std::string> & blocks,
                     FunctionPtr ic,
                     double scaling)
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
  _vars.push_back(v);
  // Automatic differentiation seeds one derivative slot per element node and
  // variable, so the largest element in the mesh sets the limit.
  int max_nodes = 1;
  ElementType largest = ElementType::Edge2;
  for (const auto & el : _mesh->elements())
    if (el.numNodes() > max_nodes)
    {
      max_nodes = el.numNodes();
      largest = el.type;
    }
  if (numVariables() * max_nodes > kMaxDerivatives)
  {
    std::ostringstream os;
    os << "Too many variables for this mesh. The largest element is " << elementTypeName(largest)
       << " with " << max_nodes
       << " nodes, and automatic differentiation seeds one slot per node and variable, so "
       << "at most " << kMaxDerivatives / max_nodes << " variable"
       << (kMaxDerivatives / max_nodes == 1 ? "" : "s") << " fit in the limit of "
       << kMaxDerivatives
       << " slots. Rebuild with -DDUALMESH_MAX_AD_DERIVATIVES=" << numVariables() * max_nodes
       << " to raise it.";
    _vars.pop_back();
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
  return v.index;
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
  auto obj = f.create(type, name, std::move(params));
  switch (f.category(type))
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
  case ObjectCategory::Material:
    addMaterial(std::static_pointer_cast<Material>(obj));
    break;
  }
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
Problem::addMaterial(std::shared_ptr<Material> m)
{
  registerName(_by_name, m);
  _materials.push_back(std::move(m));
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
  if (_vars.empty())
    throw InputError("The problem has no variables.");
  if (_kernels.empty())
    throw InputError("The problem has no kernels.");
  _props = MaterialPropertyRegistry();
  for (auto & m : _materials)
  {
    m->initialSetup(*this);
    m->declareProperties(_props);
  }
  for (auto & k : _kernels)
    k->initialSetup(*this);
  for (auto & b : _ibcs)
    b->initialSetup(*this);
  for (auto & b : _nbcs)
    b->initialSetup(*this);
  for (auto & l : _loads)
    l->initialSetup(*this);
  // Check that required material properties exist.
  for (auto & k : _kernels)
    for (const auto & p : k->requiredProperties())
      _props.id(p);
  buildGroups();
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
      for (int a = 0; a < el.numNodes(); ++a)
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

void
Problem::fillContext(QpContext & ctx,
                     const Element & el,
                     const MappedPoint & fp,
                     const std::vector<double> & cur_local,
                     const std::vector<double> * lag_local,
                     const std::vector<double> * old_local,
                     int ld) const
{
  // ld > 0: seed derivatives (local dof index k*nvar+v); ld == 0: values only.
  const int nv = numVariables();
  const int nn = el.numNodes();
  const int dim = _mesh->dimension();
  ctx.u.resize(nv);
  ctx.grad_u.resize(nv);
  for (int v = 0; v < nv; ++v)
  {
    double val = 0;
    Point g{0, 0, 0};
    for (int k = 0; k < nn; ++k)
    {
      const double Uk = cur_local[k * nv + v];
      val += fp.N[k] * Uk;
      for (int d = 0; d < dim; ++d)
        g[d] += fp.dN[k][d] * Uk;
    }
    ADReal u = ADReal::withZeroDerivatives(val, ld);
    ADVector3 gr;
    for (int d = 0; d < 3; ++d)
      gr[d] = ADReal::withZeroDerivatives(g[d], d < dim ? ld : 0);
    if (ld > 0)
      for (int k = 0; k < nn; ++k)
      {
        u.setDerivative(k * nv + v, fp.N[k]);
        for (int d = 0; d < dim; ++d)
          gr[d].setDerivative(k * nv + v, fp.dN[k][d]);
      }
    ctx.u[v] = u;
    ctx.grad_u[v] = gr;
  }
  const auto interp = [&](const std::vector<double> & loc,
                          std::vector<ADReal> & vals,
                          std::vector<ADVector3> & grads)
  {
    vals.resize(nv);
    grads.resize(nv);
    for (int v = 0; v < nv; ++v)
    {
      double val = 0;
      Point g{0, 0, 0};
      for (int k = 0; k < nn; ++k)
      {
        val += fp.N[k] * loc[k * nv + v];
        for (int d = 0; d < dim; ++d)
          g[d] += fp.dN[k][d] * loc[k * nv + v];
      }
      vals[v] = ADReal(val);
      grads[v] = {ADReal(g[0]), ADReal(g[1]), ADReal(g[2])};
    }
  };
  if (lag_local)
    interp(*lag_local, ctx.u_lag, ctx.grad_u_lag);
  else
  {
    ctx.u_lag = ctx.u;
    ctx.grad_u_lag = ctx.grad_u;
  }
  if (old_local)
  {
    ctx.u_old.assign(nv, 0.0);
    ctx.grad_u_old.assign(nv, {0, 0, 0});
    for (int v = 0; v < nv; ++v)
      for (int k = 0; k < nn; ++k)
      {
        ctx.u_old[v] += fp.N[k] * (*old_local)[k * nv + v];
        for (int d = 0; d < dim; ++d)
          ctx.grad_u_old[v][d] += fp.dN[k][d] * (*old_local)[k * nv + v];
      }
  }
  else
  {
    ctx.u_old.assign(nv, 0.0);
    ctx.grad_u_old.assign(nv, {0, 0, 0});
    for (int v = 0; v < nv; ++v)
    {
      ctx.u_old[v] = ctx.u[v].value();
      for (int d = 0; d < 3; ++d)
        ctx.grad_u_old[v][d] = ctx.grad_u[v][d].value();
    }
  }
}

void
Problem::computeMaterials(QpContext & ctx) const
{
  if (_materials.empty())
    return;
  ctx.properties.assign(_props.size(), ADReal(0.0));
  for (const auto & m : _materials)
    if (m->activeOnBlock(ctx.block))
      m->computeProperties(ctx);
}

} // namespace dualmesh
