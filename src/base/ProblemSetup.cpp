// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The setup of a problem before its first solve (initialize and its stages): the dependent
// degrees of freedom of periodic copies, hanging nodes and mixed orders, the polygon cells of
// DMCDM, the node gradients, the kernel groups, the active degrees of freedom and the check of
// the mesh.
#include "dualmesh/base/Problem.h"

#include <Eigen/LU>

#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <optional>
#include <ranges>
#include <set>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace dualmesh
{

std::unordered_map<Index, Index>
joinedRepresentatives(const std::vector<std::pair<Index, Index>> & pairs)
{
  // The ids that the pairs join, each in a set of its own slot, and whether each is a secondary.
  DisjointSet sets;
  std::unordered_map<Index, Index> slot_of;
  std::vector<Index> id_of;
  std::vector<char> secondary;
  const auto slot = [&](Index g)
  {
    const auto [it, added] = slot_of.emplace(g, static_cast<Index>(id_of.size()));
    if (added)
    {
      sets.add();
      id_of.push_back(g);
      secondary.push_back(0);
    }
    return it->second;
  };
  for (const auto & [s, p] : pairs)
  {
    if (s == p)
      continue;
    const Index a = slot(s), b = slot(p);
    secondary[a] = 1;
    sets.join(a, b);
  }
  // The representative of each group, by the slot of its root: the slot of its smallest id that
  // is a primary only (-1 for none).
  const Index n = static_cast<Index>(id_of.size());
  std::vector<Index> representative(static_cast<std::size_t>(n), -1);
  for (Index i = 0; i < n; ++i)
    if (!secondary[i])
    {
      Index & r = representative[sets.root(i)];
      if (r < 0 || id_of[i] < id_of[r])
        r = i;
    }
  std::unordered_map<Index, Index> joined;
  for (Index i = 0; i < n; ++i)
  {
    const Index r = representative[sets.root(i)];
    if (r < 0)
      throw InputError("The constraints make a node a copy of itself through a loop. One node of "
                       "each loop must stay a primary.");
    if (r != i)
      joined[id_of[i]] = id_of[r];
  }
  return joined;
}

void
Problem::initialize()
{
  if (_initialized)
    return;
  _singularity_checked = false;
  _flow->resetPressureLevels();
  _singularity_pattern_rows = -1;
  _singularity_pattern_nonzeros = -1;
  if (_vars.empty())
    throw InputError("The problem has no variables.");
  if (_kernels.empty())
    throw InputError("The problem has no kernels.");
  buildOrderLayout();
  checkMixedOrder();
  _props = PropertyRegistry();
  for (auto & m : _property_objects)
  {
    m->initialSetup(*this);
    m->declareProperties(_props);
    m->setupPropertyFactors(_props);
  }
  for (auto & k : _kernels)
    k->initialSetup(*this);
  classifyEquations();
  setupNodeGradients();
  for (auto & b : _ibcs)
    b->initialSetup(*this);
  checkInterfaceOrders();
  for (auto & b : _nbcs)
    b->initialSetup(*this);
  for (auto & l : _loads)
    l->initialSetup(*this);
  buildDependentDofs();
  // Check that required properties exist.
  for (auto & k : _kernels)
    for (const auto & p : k->requiredProperties())
      _props.id(p);
  allocateState();
  buildGroups();
  buildInterfacePairs();
  markActiveDofs();
  warmCaches();
  checkMeshInversions();
  _thread_safe = everythingThreadSafe();
  _initialized = true;
}

void
Problem::checkMixedOrder() const
{
  if (!_mixed_order)
    return;
  std::string first;
  for (const auto & v : _vars)
    if (v.order == VariableOrder::First)
      first += (first.empty() ? "'" : ", '") + v.name + "'";
  if (isDual())
    throw InputError(
        "The variable " + first +
        " is declared first order on a mesh with quadratic elements. Mixed-order "
        "interpolation (the Taylor-Hood element) is a finite element construction and needs "
        "method='FEM'. The dual mesh control domain method interpolates every variable at "
        "the order of the mesh; for it, declare the variable with the mesh order.");
  for (const auto & el : _mesh->elements())
    if (el.numCorners() < el.numNodes() &&
        ReferenceElement::get(elementCornerType(el.type)).numNodes() > MappedPoint::kMaxCorners)
      throw InputError("Internal error: an element has more corners than MappedPoint holds.");
}

void
Problem::classifyEquations()
{
  _second_order_variables.assign(_vars.size(), 0);
  for (const auto & k : _kernels)
    if (const auto * storage = dynamic_cast<const StorageKernel *>(k.get());
        storage && storage->isSecondOrder())
      _second_order_variables[static_cast<std::size_t>(k->variable())] = 1;
  // The equations without a time derivative, which a step of the theta method takes at the new
  // time only.
  _new_time_only.assign(_vars.size(), 1);
  for (const auto & k : _kernels)
    if (k->isTimeKernel())
      _new_time_only[static_cast<std::size_t>(k->variable())] = 0;
  _needs_laplacian = false;
  for (const auto & k : _kernels)
    _needs_laplacian = _needs_laplacian || k->needsLaplacian();
}

void
Problem::setupNodeGradients()
{
  _gradient_offsets.clear();
  if (!isDual())
    return;
  for (const auto & k : _kernels)
    if (const auto * advection = dynamic_cast<const AdvectiveKernel *>(k.get());
        advection && advection->needsSideGradients())
    {
      if (_parallel)
        throw InputError(
            "'" + k->name() +
            "': the linear upwind and limited schemes of DMCDM reconstruct the gradient at a "
            "node from the nodes around it, which a part of a distributed mesh does not hold "
            "at its interfaces. Use them on one process, or the upwind scheme.");
      buildNodeGradients();
      return;
    }
}

void
Problem::checkInterfaceOrders() const
{
  if (!_mixed_order)
    return;
  for (const auto & b : _ibcs)
    if (const auto * ib = dynamic_cast<const InterfaceBC *>(b.get()))
      for (int v : ib->coupledVariables())
        if (_full_order_rank[v] < 0)
          throw InputError("'" + ib->name() + "' couples the first-order variable '" +
                           _vars[v].name +
                           "' across an interface, which is not supported. Interface "
                           "conditions work with variables of the mesh order.");
}

void
Problem::allocateState()
{
  // Every stateful property object gets its slice of the record, and every owner an (initially
  // empty) list of points.
  _state_record = 0;
  for (auto & m : _property_objects)
  {
    m->setStateOffset(_state_record);
    _state_record += m->stateSize();
  }
  _state.assign(_state_record > 0 ? _mesh->numElements() : 0, OwnerState{});
}

void
Problem::warmCaches() const
{
  // The mesh caches the list of exterior sides and the boundary node markers
  // the first time they are asked for.
  _mesh->exteriorSides();
  _mesh->boundaryNodeMarkers();
  // Reference element definitions are built on first use and cached in a
  // function-local static.
  for (Index e = 0; e < _mesh->numElements(); ++e)
    ReferenceElement::get(_mesh->element(e).type);
}

bool
Problem::everythingThreadSafe() const
{
  for (const auto & object : std::views::values(_by_name))
    if (!object->threadSafe())
      return false;
  for (const auto & f : std::views::values(_functions))
    if (f && !f->threadSafe())
      return false;
  for (const auto & v : _vars)
    if (v.initial_condition && !v.initial_condition->threadSafe())
      return false;
  return true;
}

void
Problem::buildGroups()
{
  _groups.clear();
  const auto groupOf = [&](const QuadratureSpec & spec) -> Group &
  {
    auto it = std::find_if(
        _groups.begin(), _groups.end(), [&](const Group & g) { return g.spec == spec; });
    if (it != _groups.end())
      return *it;
    _groups.push_back({spec, {}, {}});
    return _groups.back();
  };
  // The groups of the rules of time stepping come first, in every study, so that the index of
  // a group, which keys the history states of its points (stateKey), is the same in each.
  for (const auto & k : _kernels)
    groupOf(k->quadrature());
  for (const auto & k : _kernels)
    groupOf(studyQuadrature(*k)).add(k.get());
}

void
Problem::buildDependentDofs()
{
  const Index n = numDofs();
  const int nv = numVariables();
  _dependent_of.clear();
  _fold_dependents = false;
  _polygon_of.clear();
  _polygon_cells.clear();
  // What each dependent degree of freedom depends on directly, before chains are resolved.
  std::vector<std::vector<std::pair<Index, double>>> on(static_cast<std::size_t>(n));
  bool any = false;
  // A first-order variable has no unknown at a non-corner node of a quadratic element: its slot
  // there is the linear interpolant of the corners, the same from every element that holds the
  // node, because the interpolant is continuous.
  if (_mixed_order)
  {
    double N[kMaxElementNodes];
    Point dN[kMaxElementNodes];
    for (const auto & el : _mesh->elements())
    {
      const int corners = el.numCorners();
      if (corners == el.numNodes())
        continue;
      const auto & quadratic = ReferenceElement::get(el.type);
      const auto & linear = ReferenceElement::get(elementCornerType(el.type));
      for (int k = corners; k < el.numNodes(); ++k)
      {
        linear.shape(quadratic.node(k), N, dN);
        for (int v = 0; v < nv; ++v)
        {
          auto & list = on[static_cast<std::size_t>(dof(el.nodes[k], v))];
          if (_full_order_rank[v] >= 0 || !list.empty())
            continue;
          for (int a = 0; a < corners; ++a)
            if (N[a] != 0.0)
              list.push_back({dof(el.nodes[a], v), N[a]});
          any = true;
        }
      }
    }
  }
  // A periodic copy takes the value of its group's representative.  A distributed run joins the
  // copies in its gather-scatter instead (DistributedProblem::joinConstraints).
  std::unordered_map<Index, Index> representative;
  if (!_parallel)
  {
    std::vector<std::pair<Index, Index>> pairs;
    for (const auto & c : _constraints)
    {
      c->initialSetup(*this);
      pairs.insert(pairs.end(), c->pairs().begin(), c->pairs().end());
    }
    representative = joinedRepresentatives(pairs);
    for (const auto & [d, r] : representative)
      if (on[static_cast<std::size_t>(d)].empty())
      {
        on[static_cast<std::size_t>(d)] = {{r, 1.0}};
        any = _fold_dependents = true;
      }
  }
  // Under FEM a hanging node is the weighted sum of the nodes of the side it lies on.  A periodic
  // group with a hanging member hangs as a whole: its representative takes the weights, its own
  // first, and every copy follows the representative.  DMCDM keeps a hanging node as an unknown
  // of its own, in a polygon cell.
  const auto & hanging = _mesh->nodeConstraints();
  if (!hanging.empty() && isDual())
    buildPolygonCells();
  else
    for (const bool copies : {false, true})
      for (const auto & c : hanging)
        for (int v = 0; v < nv; ++v)
        {
          const Index d = dof(c.node, v);
          const auto joined = representative.find(d);
          const Index target = joined == representative.end() ? d : joined->second;
          auto & list = on[static_cast<std::size_t>(target)];
          if ((target != d) != copies || !list.empty())
            continue;
          for (const auto & [p, w] : c.on)
            list.push_back({dof(p, v), w});
          any = _fold_dependents = true;
        }
  if (!any)
    return;
  // Resolve chains (a node hanging on a hanging node or on a periodic copy, the copy of a hanging
  // node) until every dependent degree of freedom is a sum over independent ones.
  _dependent_of.assign(static_cast<std::size_t>(n), {});
  std::vector<char> state(static_cast<std::size_t>(n), 0);
  const std::function<const std::vector<std::pair<Index, double>> &(Index)> resolve =
      [&](Index d) -> const std::vector<std::pair<Index, double>> &
  {
    auto & out = _dependent_of[static_cast<std::size_t>(d)];
    if (state[static_cast<std::size_t>(d)] == 2)
      return out;
    if (state[static_cast<std::size_t>(d)] == 1)
      throw InputError("The periodic boundaries and the hanging nodes of the mesh make a node "
                       "depend on itself through a loop.");
    state[static_cast<std::size_t>(d)] = 1;
    std::map<Index, double> sum;
    for (const auto & [p, w] : on[static_cast<std::size_t>(d)])
      if (on[static_cast<std::size_t>(p)].empty())
        sum[p] += w;
      else
        for (const auto & [q, u] : resolve(p))
          sum[q] += w * u;
    out.assign(sum.begin(), sum.end());
    state[static_cast<std::size_t>(d)] = 2;
    return out;
  };
  for (Index d = 0; d < n; ++d)
    if (!on[static_cast<std::size_t>(d)].empty())
      resolve(d);
  // A dependent degree of freedom has no equation of its own, so it can be prescribed only where
  // every degree of freedom it depends on is.
  const std::vector<char> prescribed = prescribedDofs();
  for (Index d = 0; d < n; ++d)
    if (prescribed[static_cast<std::size_t>(d)])
      for (const auto & [p, w] : _dependent_of[static_cast<std::size_t>(d)])
        if (!prescribed[static_cast<std::size_t>(p)])
          throw InputError("A periodic copy or a hanging node carries a prescribed value that its "
                           "primary node does not. Prescribe the value on both periodic "
                           "boundaries or on neither, and on the whole side a node hangs on.");
  updateDependentDofs(_U);
}

namespace
{
/// The hanging nodes of a mesh: the one in the middle of each edge, by the end nodes of the edge
/// in increasing order, and the one at the centre of each face, by its corners in increasing
/// order.
struct HangingNodes
{
  std::map<std::pair<Index, Index>, Index> on_edge;
  std::map<std::vector<Index>, Index> on_face;
};

HangingNodes
hangingNodes(const Mesh & mesh)
{
  HangingNodes hanging;
  for (const auto & c : mesh.nodeConstraints())
  {
    std::vector<Index> on;
    for (const auto & [node, weight] : c.on)
      on.push_back(node);
    std::sort(on.begin(), on.end());
    if (on.size() == 2)
      hanging.on_edge[{on[0], on[1]}] = c.node;
    else if (on.size() == 4)
      hanging.on_face[on] = c.node;
  }
  return hanging;
}

/// The leaf edges of a Quad4 cell, or the leaf faces of a Hex8 cell, each the ring of its slots,
/// with their centres.
struct Leaves
{
  std::vector<std::array<int, 2>> edges;
  std::vector<std::vector<int>> faces;
  std::vector<Point> face_centres;
};

/// The polygon cell of a Quad4 or Hex8 element with nodes hanging on its sides: its nodes, the
/// element's and then the hanging ones, which the cell calls its slots, and its dual mesh, as
/// ReferenceElement builds that of a conforming cell.
class PolygonBuilder
{
public:
  PolygonBuilder(const Element & el, const HangingNodes & hanging)
      : _el(el), _ref(ReferenceElement::get(el.type)), _hanging(hanging)
  {
    for (int k = 0; k < _ref.numNodes(); ++k)
    {
      _cell.nodes.push_back(el.nodes[k]);
      _cell.xi.push_back(_ref.node(k));
    }
  }

  /// The polygon cell, or nothing when no node hangs on the sides of the element.
  std::optional<PolygonCell> build()
  {
    const bool quad = _el.type == ElementType::Quad4;
    const Leaves leaves = quad ? quadLeaves() : hexLeaves();
    if (!_has_hanging)
      return std::nullopt;
    _cell.control.assign(_cell.nodes.size(), {});
    if (quad)
      quadDual(leaves);
    else
      hexDual(leaves);
    return std::move(_cell);
  }

private:
  /// The slot of the hanging node @p node at the reference point @p xi, made the first time it is
  /// met.
  int slotOf(Index node, const Point & xi)
  {
    _has_hanging = true;
    for (std::size_t k = 0; k < _cell.nodes.size(); ++k)
      if (_cell.nodes[k] == node)
        return static_cast<int>(k);
    _cell.nodes.push_back(node);
    _cell.xi.push_back(xi);
    return static_cast<int>(_cell.nodes.size() - 1);
  }

  /// The slot of the node hanging in the middle of the edge between the local nodes @p i and
  /// @p j, or -1 for none.
  int middleOf(int i, int j)
  {
    const auto found = _hanging.on_edge.find(std::minmax(_el.nodes[i], _el.nodes[j]));
    return found == _hanging.on_edge.end()
               ? -1
               : slotOf(found->second, 0.5 * (_ref.node(i) + _ref.node(j)));
  }

  /// The reference point of slot @p slot.
  Point at(int slot) const { return _cell.xi[static_cast<std::size_t>(slot)]; }

  /// The sides of Quad4 go round the cell, and a side with a hanging node is two leaf edges.
  Leaves quadLeaves()
  {
    Leaves leaves;
    for (int side = 0; side < _ref.numSides(); ++side)
    {
      const int i = _ref.sideNodes(side)[0], j = _ref.sideNodes(side)[1];
      const int m = middleOf(i, j);
      if (m < 0)
        leaves.edges.push_back({i, j});
      else
      {
        leaves.edges.push_back({i, m});
        leaves.edges.push_back({m, j});
      }
    }
    return leaves;
  }

  /// A side of Hex8 whose centre hangs is four leaf faces, and another is one, with the hanging
  /// middles of its edges among its corners.
  Leaves hexLeaves()
  {
    Leaves leaves;
    for (int side = 0; side < _ref.numSides(); ++side)
    {
      const std::vector<int> & corners = _ref.sideNodes(side);
      std::vector<Index> sorted;
      Point face_centre{0, 0, 0};
      for (int k : corners)
      {
        sorted.push_back(_el.nodes[k]);
        face_centre = face_centre + 0.25 * _ref.node(k);
      }
      std::sort(sorted.begin(), sorted.end());
      const auto centre = _hanging.on_face.find(sorted);
      if (centre == _hanging.on_face.end())
      {
        std::vector<int> ring;
        for (std::size_t k = 0; k < corners.size(); ++k)
        {
          ring.push_back(corners[k]);
          const int m = middleOf(corners[k], corners[(k + 1) % corners.size()]);
          if (m >= 0)
            ring.push_back(m);
        }
        leaves.faces.push_back(ring);
        leaves.face_centres.push_back(face_centre);
        continue;
      }
      const int c = slotOf(centre->second, face_centre);
      for (std::size_t k = 0; k < corners.size(); ++k)
      {
        const int v = corners[k];
        const int next = middleOf(v, corners[(k + 1) % corners.size()]);
        const int previous = middleOf(corners[(k + corners.size() - 1) % corners.size()], v);
        if (next < 0 || previous < 0)
          throw std::logic_error("dualmesh: a hanging face centre without its edge middles.");
        const std::vector<int> ring{v, next, c, previous};
        Point sub_centre{0, 0, 0};
        for (int slot : ring)
          sub_centre = sub_centre + 0.25 * at(slot);
        leaves.faces.push_back(ring);
        leaves.face_centres.push_back(sub_centre);
      }
    }
    return leaves;
  }

  /// Each slot of a Quad4 cell ends two leaf edges, whose middles bound its quadrilateral with
  /// the centre.
  void quadDual(const Leaves & leaves)
  {
    const Point & centre = _ref.centroid();
    std::vector<std::vector<Point>> middles(_cell.nodes.size());
    for (const auto & [p, q] : leaves.edges)
    {
      const Point middle = 0.5 * (at(p) + at(q));
      middles[static_cast<std::size_t>(p)].push_back(middle);
      middles[static_cast<std::size_t>(q)].push_back(middle);
      DualFace face{p, q, {}};
      face.patch.dim = 1;
      face.patch.corners = {middle, centre};
      _cell.faces.push_back(face);
    }
    for (std::size_t k = 0; k < _cell.nodes.size(); ++k)
    {
      Patch patch;
      patch.dim = 2;
      patch.corners = {_cell.xi[k], middles[k][0], centre, middles[k][1]};
      _cell.control[k] = {patch};
    }
  }

  /// The control volume of each slot of a Hex8 cell is a pyramid over the part of each leaf face
  /// nearest it, and two slots of a leaf face share a triangle.
  void hexDual(const Leaves & leaves)
  {
    const Point & centre = _ref.centroid();
    for (std::size_t f = 0; f < leaves.faces.size(); ++f)
    {
      const auto & ring = leaves.faces[f];
      const Point & face_centre = leaves.face_centres[f];
      for (std::size_t i = 0; i < ring.size(); ++i)
      {
        const int p = ring[i], q = ring[(i + 1) % ring.size()];
        const int r = ring[(i + ring.size() - 1) % ring.size()];
        const Point next = 0.5 * (at(p) + at(q)), previous = 0.5 * (at(r) + at(p));
        // The pyramid over the part of the leaf face nearest p, a cube collapsed onto the centre.
        Patch patch;
        patch.dim = 3;
        patch.extra_points = 1;
        patch.corners = {at(p), next, face_centre, previous, centre, centre, centre, centre};
        _cell.control[static_cast<std::size_t>(p)].push_back(patch);
        // The triangle between p and q, a square collapsed onto the centre.
        DualFace face{p, q, {}};
        face.patch.dim = 2;
        face.patch.extra_points = 1;
        face.patch.corners = {next, face_centre, centre, centre};
        _cell.faces.push_back(face);
      }
    }
  }

  const Element & _el;
  const ReferenceElement & _ref;
  const HangingNodes & _hanging;
  PolygonCell _cell;
  bool _has_hanging = false;
};
} // namespace

void
Problem::buildPolygonCells()
{
  const HangingNodes hanging = hangingNodes(*_mesh);
  _polygon_of.assign(static_cast<std::size_t>(_mesh->numElements()), -1);
  for (Index e = 0; e < _mesh->numElements(); ++e)
  {
    const Element & el = _mesh->element(e);
    if (el.type != ElementType::Quad4 && el.type != ElementType::Hex8)
      continue;
    std::optional<PolygonCell> cell = PolygonBuilder(el, hanging).build();
    if (!cell)
      continue;
    // Automatic differentiation seeds a slot for every variable at every node of the cell, its
    // hanging nodes included.
    const int slots = static_cast<int>(cell->nodes.size()) * numVariables();
    if (slots > kMaxDerivatives)
      throw InputError("A cell with hanging nodes on its sides carries " + std::to_string(slots) +
                       " degrees of freedom, and automatic differentiation seeds one slot per "
                       "local degree of freedom, within a limit of " +
                       std::to_string(kMaxDerivatives) +
                       ". Rebuild with -DDUALMESH_MAX_AD_DERIVATIVES=<n>, with n at least " +
                       std::to_string(slots) + ", to raise it.");
    _polygon_of[static_cast<std::size_t>(e)] = static_cast<int>(_polygon_cells.size());
    _polygon_cells.push_back(std::move(*cell));
  }
}

void
Problem::buildNodeGradients()
{
  const Index nn = _mesh->numNodes();
  const int dim = _mesh->dimension();
  if (dim != _mesh->spaceDimension())
    throw InputError("The linear upwind and limited schemes of DMCDM need a "
                     "mesh of the dimension of its space (not a surface or a line in space).");
  std::vector<std::set<Index>> around(static_cast<std::size_t>(nn));
  for (const auto & el : _mesh->elements())
    for (int i = 0; i < el.numNodes(); ++i)
      for (int j = 0; j < el.numNodes(); ++j)
        if (i != j)
          around[static_cast<std::size_t>(el.nodes[i])].insert(el.nodes[j]);
  _gradient_offsets.assign(static_cast<std::size_t>(nn) + 1, 0);
  _gradient_nodes.clear();
  _gradient_weights.clear();
  for (Index a = 0; a < nn; ++a)
  {
    const Point xa = _mesh->node(a);
    Eigen::Matrix3d G = Eigen::Matrix3d::Zero();
    for (Index b : around[static_cast<std::size_t>(a)])
    {
      const Point d = _mesh->node(b) - xa;
      for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
          G(i, j) += d[i] * d[j];
    }
    const Eigen::MatrixXd Gd = G.topLeftCorner(dim, dim);
    const Eigen::FullPivLU<Eigen::MatrixXd> lu(Gd);
    if (!lu.isInvertible())
      throw InputError("The gradient at node " + std::to_string(a) +
                       " cannot be reconstructed from the nodes around it (they do not span the "
                       "space), which the linear upwind and limited schemes need.");
    const Eigen::MatrixXd Ginv = lu.inverse();
    for (Index b : around[static_cast<std::size_t>(a)])
    {
      const Point d = _mesh->node(b) - xa;
      Point w{0, 0, 0};
      for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
          w[i] += Ginv(i, j) * d[j];
      _gradient_nodes.push_back(b);
      _gradient_weights.push_back(w);
    }
    _gradient_offsets[static_cast<std::size_t>(a) + 1] = static_cast<Index>(_gradient_nodes.size());
  }
}

void
Problem::markActiveDofs()
{
  const int nv = numVariables();
  _active_dof.assign(numDofs(), 0);
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
  for (const auto & bc : _ibcs)
    for (Index d : bc->multiplierDofs(*this))
      _active_dof[d] = 1;
}

void
Problem::checkMeshInversions()
{
  // The measure of a volume element enters every integral as |det J|, so a
  // mesh whose winding is reversed is integrated with a positive measure and
  // returns a wrong answer without a warning.  The generators and the mesh
  // readers fix the orientation themselves; a mesh built node by node may not,
  // and this is where that is caught.  The check runs once, at initialize(),
  // at the centroid and half way from it to every node: points inside every
  // element, where the integration points lie, and away from the apex of a
  // pyramid, whose map is degenerate there by construction.  A surface or a
  // curve in space is exempt: its measure is a square root, and the side of
  // its normal is a convention that fixOrientation sets.
  const int space = _mesh->spaceDimension();
  MappedPoint mp;
  for (Index e = 0; e < _mesh->numElements(); ++e)
  {
    const Element & el = _mesh->element(e);
    const auto & ref = ReferenceElement::get(el.type);
    if (ref.dimension() != space)
      continue;
    const Point & c = ref.centroid();
    bool inverted = false;
    for (int n = -1; !inverted && n < el.numNodes(); ++n)
    {
      const Point xi = n < 0 ? c : 0.5 * (c + ref.node(n));
      try
      {
        mapPoint(*_mesh, el, xi, mp);
      }
      catch (const std::runtime_error &)
      {
        std::ostringstream os;
        os << "Element " << e << " of block " << el.block << " is degenerate: its Jacobian "
           << "determinant is zero inside it (nodes on one line or in one plane, or a repeated "
           << "node). Correct the mesh where it is generated.";
        throw InputError(os.str());
      }
      inverted = mp.detJ < 0.0;
    }
    if (!inverted)
      continue;
    const Point & x = _mesh->node(el.nodes[0]);
    std::ostringstream os;
    os << "Element " << e << " of block " << el.block << " (a " << el.numNodes()
       << "-node element whose first node is at (" << x[0] << ", " << x[1] << ", " << x[2]
       << ")) is inverted: its Jacobian determinant is negative, so its "
          "integrals would be computed with a positive measure and the answer would be wrong. "
          "Reverse the winding of its nodes, or call fix_orientation() on the mesh (the "
          "generators and read_mesh do this themselves; a mesh built node by node may need it).";
    throw InputError(os.str());
  }
}

} // namespace dualmesh
