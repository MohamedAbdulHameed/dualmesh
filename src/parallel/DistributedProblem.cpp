// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/parallel/DistributedProblem.h"
#include "dualmesh/linalg/IncompleteLU.h"

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

namespace dualmesh
{

DistributedProblem::DistributedProblem(const Mesh & global_mesh,
                                       Method method,
                                       CoordinateSystem coord,
                                       const DistributedOptions & options)
    : _comm(Communicator::world()), _options(options)
{
  // The decomposition below identifies a degree of freedom by the mesh node it
  // sits on: ownership, the exchange of shared values and the inner product
  // are all indexed by node.  The cell-centred finite volume method puts its
  // unknowns at cell centroids and at boundary face centroids instead, so none
  // of that indexing applies to it, and it would additionally need a layer of
  // ghost cells so that a face on the partition boundary is not mistaken for a
  // face on the boundary of the domain.  Neither is implemented, so the
  // combination is refused rather than allowed to return a wrong answer.
  if (method == Method::FiniteVolumeCell)
    throw InputError(
        "The cell-centred finite volume method (zfvm) cannot be run with the distributed "
        "solver. Its unknowns sit at cell and boundary face centroids rather than at mesh "
        "nodes, and the distributed decomposition identifies unknowns by node; it would also "
        "need a layer of ghost cells across each partition boundary, which is not "
        "implemented. Use the dual mesh (dmcdm), finite element (fem) or vertex-centred "
        "finite volume (hfvm) method for a distributed run, or run zfvm in one process with "
        "threads.");
  _num_global_nodes = global_mesh.numNodes();
  _partition = partitionMesh(global_mesh, _comm.size(), options.partitioner);
  const auto elements = _partition.elementsOf(_comm.rank());
  _mesh = std::make_shared<Mesh>(subMesh(global_mesh, elements, _local_to_global));
  _local = std::make_unique<Problem>(_mesh, method, coord);

  _owned_node.assign(_local_to_global.size(), 0);
  std::vector<std::vector<Index>> shared(_comm.size());
  for (std::size_t i = 0; i < _local_to_global.size(); ++i)
  {
    const Index g = _local_to_global[i];
    _owned_node[i] = _partition.node_owner[g] == _comm.rank() ? 1 : 0;
    for (int p : _partition.node_parts[g])
      if (p != _comm.rank())
        shared[p].push_back(static_cast<Index>(i));
  }
  // Both sides of an exchange must agree on the order of the shared nodes.
  // Sorting by global node index achieves that without any communication.
  for (auto & list : shared)
    std::sort(list.begin(),
              list.end(),
              [&](Index a, Index b) { return _local_to_global[a] < _local_to_global[b]; });
  _shared_nodes = std::move(shared);
  _send_buffer.assign(_comm.size(), {});
  _recv_buffer.assign(_comm.size(), {});
  if (_options.overlap < 0)
    throw InputError("The overlap of the Schwarz subdomains must be zero or positive.");
  if (_options.preconditioner != "jacobi" && _comm.size() > 1)
    buildOverlap(global_mesh, elements);
}

void
DistributedProblem::buildOverlap(const Mesh & global_mesh, const std::vector<Index> & elements)
{
  // Every rank holds the whole mesh and the whole partition, so each can grow
  // its own subdomain without communicating: starting from its elements, add
  // every element that touches a node of the current set, `overlap` times.
  const Index num_nodes = global_mesh.numNodes();
  const Index num_elements = global_mesh.numElements();
  std::vector<Index> start(num_nodes + 1, 0);
  const auto nodesOf = [&](Index e)
  {
    const auto & el = global_mesh.element(e);
    return std::make_pair(el.nodes.data(), el.nodes.data() + el.numNodes());
  };
  for (Index e = 0; e < num_elements; ++e)
    for (auto [g, end] = nodesOf(e); g != end; ++g)
      ++start[*g + 1];
  for (Index g = 0; g < num_nodes; ++g)
    start[g + 1] += start[g];
  std::vector<Index> incident(start.back());
  {
    std::vector<Index> fill(start.begin(), start.end() - 1);
    for (Index e = 0; e < num_elements; ++e)
      for (auto [g, end] = nodesOf(e); g != end; ++g)
        incident[fill[*g]++] = e;
  }
  std::vector<char> in_subdomain(num_elements, 0);
  for (Index e : elements)
    in_subdomain[e] = 1;
  std::vector<char> node_in(num_nodes, 0);
  const auto markNodes = [&]
  {
    for (Index e = 0; e < num_elements; ++e)
      if (in_subdomain[e])
        for (auto [g, end] = nodesOf(e); g != end; ++g)
          node_in[*g] = 1;
  };
  markNodes();
  for (int layer = 0; layer < _options.overlap; ++layer)
  {
    for (Index g = 0; g < num_nodes; ++g)
      if (node_in[g])
        for (Index k = start[g]; k < start[g + 1]; ++k)
          in_subdomain[incident[k]] = 1;
    markNodes();
  }

  // Subdomain numbering: the local nodes first, in local order, then the
  // ghosts in increasing global order.
  const int me = _comm.rank();
  const int ranks = _comm.size();
  _subdomain_index.clear();
  for (std::size_t i = 0; i < _local_to_global.size(); ++i)
    _subdomain_index[_local_to_global[i]] = static_cast<Index>(i);
  _ghost_global.clear();
  for (Index g = 0; g < num_nodes; ++g)
    if (node_in[g] && !_subdomain_index.count(g))
    {
      _subdomain_index[g] = static_cast<Index>(_local_to_global.size() + _ghost_global.size());
      _ghost_global.push_back(g);
    }

  // Tell every rank which of the nodes it holds lie in this subdomain; it
  // will send the matrix entries between them.  A node is held by the ranks
  // whose elements touch it.
  std::vector<std::vector<Index>> request(ranks), reply;
  for (Index g = 0; g < num_nodes; ++g)
    if (node_in[g])
      for (int r : _partition.node_parts[g])
        if (r != me)
          request[r].push_back(g);
  _comm.exchange(request, reply);
  std::unordered_map<Index, Index> local_of;
  for (std::size_t i = 0; i < _local_to_global.size(); ++i)
    local_of[_local_to_global[i]] = static_cast<Index>(i);
  _overlap_send_nodes.assign(ranks, {});
  for (int r = 0; r < ranks; ++r)
    for (Index g : reply[r])
      _overlap_send_nodes[r].push_back(local_of.at(g));

  // Ask the owner of every ghost node for its values.
  std::vector<std::vector<Index>> wanted(ranks), asked;
  _ghost_recv_slots.assign(ranks, {});
  for (Index g : _ghost_global)
  {
    const int owner = _partition.node_owner[g];
    wanted[owner].push_back(g);
    _ghost_recv_slots[owner].push_back(_subdomain_index.at(g));
  }
  _comm.exchange(wanted, asked);
  _ghost_send_nodes.assign(ranks, {});
  for (int r = 0; r < ranks; ++r)
    for (Index g : asked[r])
      _ghost_send_nodes[r].push_back(local_of.at(g));
}

void
DistributedProblem::gatherGhosts(const Vector & local, Vector & subdomain) const
{
  const int nv = _local->numVariables();
  const Index n = local.size();
  subdomain.resize(n + static_cast<Index>(_ghost_global.size()) * nv);
  subdomain.head(n) = local;
  std::vector<std::vector<double>> send(_comm.size()), recv;
  for (int r = 0; r < _comm.size(); ++r)
    for (Index i : _ghost_send_nodes[r])
      for (int k = 0; k < nv; ++k)
        send[r].push_back(local[_local->dof(i, k)]);
  _comm.exchange(send, recv);
  for (int r = 0; r < _comm.size(); ++r)
  {
    std::size_t at = 0;
    for (Index slot : _ghost_recv_slots[r])
      for (int k = 0; k < nv; ++k)
        subdomain[slot * nv + k] = recv[r][at++];
  }
}

void
DistributedProblem::addGhostsToOwners(const Vector & subdomain, Vector & local) const
{
  const int nv = _local->numVariables();
  std::vector<std::vector<double>> send(_comm.size()), recv;
  for (int r = 0; r < _comm.size(); ++r)
    for (Index slot : _ghost_recv_slots[r])
      for (int k = 0; k < nv; ++k)
        send[r].push_back(subdomain[slot * nv + k]);
  _comm.exchange(send, recv);
  for (int r = 0; r < _comm.size(); ++r)
  {
    std::size_t at = 0;
    for (Index i : _ghost_send_nodes[r])
      for (int k = 0; k < nv; ++k)
        local[_local->dof(i, k)] += recv[r][at++];
  }
}

SparseMatrix
DistributedProblem::subdomainMatrix(const SparseMatrix & A) const
{
  // A = sum_r A_r, so the entry (i, j) of R_d A R_d^T is the sum of the
  // entries (i, j) of the local matrices of every rank that holds both nodes.
  // Each rank sends every neighbour the entries between the nodes of that
  // neighbour's subdomain, identified by global degree of freedom.
  const int nv = _local->numVariables();
  const int ranks = _comm.size();
  const Index local_nodes = static_cast<Index>(_local_to_global.size());
  const Index size = (local_nodes + static_cast<Index>(_ghost_global.size())) * nv;
  std::vector<Eigen::Triplet<double>> entries;
  entries.reserve(static_cast<std::size_t>(A.nonZeros()) * 2);
  for (Index col = 0; col < A.outerSize(); ++col)
    for (SparseMatrix::InnerIterator it(A, col); it; ++it)
      entries.emplace_back(it.row(), it.col(), it.value());

  std::vector<std::vector<Index>> send_index(ranks), recv_index;
  std::vector<std::vector<double>> send_value(ranks), recv_value;
  std::vector<char> in_set(local_nodes, 0);
  for (int r = 0; r < ranks; ++r)
  {
    if (_overlap_send_nodes[r].empty())
      continue;
    for (Index i : _overlap_send_nodes[r])
      in_set[i] = 1;
    for (Index col = 0; col < A.outerSize(); ++col)
    {
      if (!in_set[col / nv])
        continue;
      for (SparseMatrix::InnerIterator it(A, col); it; ++it)
        if (in_set[it.row() / nv])
        {
          send_index[r].push_back(_local_to_global[it.row() / nv] * nv + it.row() % nv);
          send_index[r].push_back(_local_to_global[col / nv] * nv + col % nv);
          send_value[r].push_back(it.value());
        }
    }
    for (Index i : _overlap_send_nodes[r])
      in_set[i] = 0;
  }
  _comm.exchange(send_index, recv_index);
  _comm.exchange(send_value, recv_value);
  for (int r = 0; r < ranks; ++r)
    for (std::size_t k = 0; k < recv_value[r].size(); ++k)
    {
      const Index gi = recv_index[r][2 * k], gj = recv_index[r][2 * k + 1];
      const Index i = _subdomain_index.at(gi / nv) * nv + gi % nv;
      const Index j = _subdomain_index.at(gj / nv) * nv + gj % nv;
      entries.emplace_back(i, j, recv_value[r][k]);
    }
  // A stored diagonal in every row, so that the incomplete factorisation can
  // run on the pattern.
  for (Index i = 0; i < size; ++i)
    entries.emplace_back(i, i, 0.0);
  SparseMatrix Ad(size, size);
  Ad.setFromTriplets(entries.begin(), entries.end());
  Ad.makeCompressed();
  return Ad;
}

void
DistributedProblem::prepare()
{
  if (_prepared)
    return;
  _local->initialize();
  // A degree of freedom that no local element touches may still be touched by
  // an element on another rank, so the mask of active degrees of freedom has
  // to be agreed on globally before it is used to replace equations.
  Vector active(_local->numDofs());
  for (Index i = 0; i < _local->numDofs(); ++i)
    active[i] = _local->activeDofs()[i] ? 1.0 : 0.0;
  addAcrossRanks(active);
  std::vector<char> global_active(_local->numDofs(), 0);
  for (Index i = 0; i < _local->numDofs(); ++i)
    global_active[i] = active[i] > 0.5 ? 1 : 0;
  _local->overrideActiveDofs(global_active);
  // Concentrated loads are attached to a node, not to an element, so only the
  // owning rank may apply them.
  const int nv = _local->numVariables();
  std::vector<char> owned_entities(_local->numEntities(), 1);
  if (!_local->isCellCentered())
    for (Index i = 0; i < _local->numEntities(); ++i)
      owned_entities[i] = _owned_node[i];
  _local->setOwnedEntities(owned_entities);
  (void) nv;
  // A concentrated load given by coordinates is resolved to the nearest node,
  // and each process only sees its own part of the mesh, so every one of them
  // would find a nearest node of its own.  The process whose node is globally
  // nearest keeps the load; a tie (the point sits on a shared node) is broken
  // by the ownership mask above.
  for (const auto & load : _local->nodalLoads())
  {
    const auto & points = load->pointNodes();
    if (points.empty())
      continue;
    std::vector<double> distance;
    distance.reserve(points.size());
    for (const auto & [node, d] : points)
    {
      (void) node;
      distance.push_back(d);
    }
    std::vector<double> best = distance;
    _comm.minInPlace(best);
    std::vector<char> keep(points.size(), 0);
    for (std::size_t i = 0; i < points.size(); ++i)
      keep[i] = distance[i] <= best[i] * (1.0 + 1e-12) + 1e-12 ? 1 : 0;
    load->keepPoints(keep);
  }
  _num_global_dofs = _comm.sum(numOwnedDofs());
  _prepared = true;
}

Index
DistributedProblem::numOwnedDofs() const
{
  Index count = 0;
  const int nv = _local->numVariables();
  for (std::size_t i = 0; i < _owned_node.size(); ++i)
    if (_owned_node[i])
      count += nv;
  return count;
}

void
DistributedProblem::addAcrossRanks(Vector & v) const
{
  if (_comm.size() == 1)
    return;
  const int nv = _local->numVariables();
  for (int r = 0; r < _comm.size(); ++r)
  {
    auto & buffer = _send_buffer[r];
    buffer.clear();
    buffer.reserve(_shared_nodes[r].size() * nv);
    for (Index n : _shared_nodes[r])
      for (int k = 0; k < nv; ++k)
        buffer.push_back(v[_local->dof(n, k)]);
  }
  _comm.exchange(_send_buffer, _recv_buffer);
  for (int r = 0; r < _comm.size(); ++r)
  {
    const auto & buffer = _recv_buffer[r];
    if (buffer.empty())
      continue;
    std::size_t at = 0;
    for (Index n : _shared_nodes[r])
      for (int k = 0; k < nv; ++k)
        v[_local->dof(n, k)] += buffer[at++];
  }
}

void
DistributedProblem::copyFromOwners(Vector & v) const
{
  if (_comm.size() == 1)
    return;
  const int nv = _local->numVariables();
  for (int r = 0; r < _comm.size(); ++r)
  {
    auto & buffer = _send_buffer[r];
    buffer.clear();
    for (Index n : _shared_nodes[r])
      for (int k = 0; k < nv; ++k)
        buffer.push_back(_owned_node[n] ? v[_local->dof(n, k)] : 0.0);
  }
  _comm.exchange(_send_buffer, _recv_buffer);
  for (int r = 0; r < _comm.size(); ++r)
  {
    const auto & buffer = _recv_buffer[r];
    if (buffer.empty())
      continue;
    std::size_t at = 0;
    for (Index n : _shared_nodes[r])
    {
      // A node may be shared by three or more ranks, and only the one that
      // owns it may overwrite the value; the others send a placeholder that
      // has to be ignored.
      const bool from_owner = _partition.node_owner[_local_to_global[n]] == r;
      for (int k = 0; k < nv; ++k)
      {
        const double value = buffer[at++];
        if (!_owned_node[n] && from_owner)
          v[_local->dof(n, k)] = value;
      }
    }
  }
}

double
DistributedProblem::dot(const Vector & a, const Vector & b) const
{
  const int nv = _local->numVariables();
  double local = 0;
  for (std::size_t i = 0; i < _owned_node.size(); ++i)
  {
    if (!_owned_node[i])
      continue;
    for (int k = 0; k < nv; ++k)
    {
      const Index d = _local->dof(static_cast<Index>(i), k);
      local += a[d] * b[d];
    }
  }
  return _comm.sum(local);
}

namespace
{
/// The subdomain part of the preconditioner: every rank factorizes its own
/// local matrix incompletely.  In the *restricted* form the correction is
/// trimmed to the degrees of freedom the rank owns before the corrections are
/// added together, which avoids counting the interface twice and converges
/// faster than the classical form (Cai and Sarkis, 1999).
struct LocalPreconditioner
{
  enum class Kind
  {
    Jacobi,
    Schwarz,
    TwoLevelSchwarz
  };
  Kind kind = Kind::TwoLevelSchwarz;
  Vector inverse_diagonal;
  /// The factors of the subdomain matrix: incomplete or exact.
  IncompleteLU0 ilu;
  Eigen::SparseLU<SparseMatrix, Eigen::COLAMDOrdering<int>> lu;
  bool exact = false;
  /// Dense inverse of the coarse operator, one unknown per subdomain.
  Eigen::MatrixXd coarse;
  bool have_coarse = false;
};
} // namespace

Vector
DistributedProblem::solveLinearSystem(const SparseMatrix & A,
                                      const Vector & b,
                                      int * iterations) const
{
  // b is consistent (the same at every rank that holds a shared degree of
  // freedom) and so is every vector produced below.
  const Index n = b.size();
  const int nv = _local->numVariables();
  const int ranks = _comm.size();
  LocalPreconditioner pc;
  if (_options.preconditioner == "jacobi")
    pc.kind = LocalPreconditioner::Kind::Jacobi;
  else if (_options.preconditioner == "additive_schwarz")
    pc.kind = LocalPreconditioner::Kind::Schwarz;
  else if (_options.preconditioner == "two_level_schwarz")
    pc.kind = LocalPreconditioner::Kind::TwoLevelSchwarz;
  else
    throw InputError("Unknown preconditioner '" + _options.preconditioner +
                     "' (use jacobi, additive_schwarz, or two_level_schwarz).");

  if (pc.kind == LocalPreconditioner::Kind::Jacobi)
  {
    Vector diagonal = A.diagonal();
    addAcrossRanks(diagonal);
    pc.inverse_diagonal.resize(n);
    for (Index i = 0; i < n; ++i)
      pc.inverse_diagonal[i] = diagonal[i] != 0.0 ? 1.0 / diagonal[i] : 1.0;
  }
  else if (ranks == 1)
  {
    // One rank: the subdomain is the whole problem.
    pc.exact = _options.subdomain_solver == "lu";
    if (pc.exact)
      pc.lu.compute(A);
    else
      pc.ilu.compute(A);
  }
  else
  {
    const SparseMatrix Ad = subdomainMatrix(A);
    pc.exact = _options.subdomain_solver == "lu";
    if (pc.exact)
      pc.lu.compute(Ad);
    else
      pc.ilu.compute(Ad);
  }
  if (pc.kind != LocalPreconditioner::Kind::Jacobi)
  {
    if (_options.subdomain_solver != "lu" && _options.subdomain_solver != "ilu")
      throw InputError("Unknown subdomain solver '" + _options.subdomain_solver +
                       "' (use ilu or lu).");
    const bool failed = pc.exact ? pc.lu.info() != Eigen::Success : pc.ilu.info() != Eigen::Success;
    if (_comm.any(failed))
      throw std::runtime_error("dualmesh: the factorisation of a Schwarz subdomain failed; try "
                               "subdomain_solver = 'lu', or preconditioner = 'jacobi'.");
  }

  // The coarse space of the two-level method (Nicolaides, SIAM J. Numer.
  // Anal. 24 (1987) 355-365): one basis function per subdomain and variable,
  // equal to one on the degrees of freedom the subdomain owns and zero
  // elsewhere.  A one-level Schwarz preconditioner moves information only one
  // subdomain further per application, so its iteration count grows as
  // subdomains are added; the coarse problem couples all of them at once.
  // Degrees of freedom whose equation has been replaced by the identity
  // (prescribed values, or no kernel) are left out of the basis: their rows
  // would add to the coarse matrix an identity that has nothing to do with
  // the operator.
  const int coarse_size = ranks * nv;
  std::vector<int> coarse_index(n, -1);
  if (pc.kind == LocalPreconditioner::Kind::TwoLevelSchwarz && ranks > 1)
  {
    std::vector<char> coupled(n, 0);
    for (Index col = 0; col < A.outerSize(); ++col)
      for (SparseMatrix::InnerIterator it(A, col); it; ++it)
        if (it.row() != col && it.value() != 0.0)
          coupled[it.row()] = 1;
    for (std::size_t i = 0; i < _owned_node.size(); ++i)
    {
      const int owner = _partition.node_owner[_local_to_global[i]];
      for (int k = 0; k < nv; ++k)
      {
        const Index d = _local->dof(static_cast<Index>(i), k);
        coarse_index[d] = coupled[d] ? owner * nv + k : -1;
      }
    }
    // A shared degree of freedom must be in the basis on every rank or on
    // none; a rank whose elements do not couple it would otherwise disagree.
    Vector in_basis(n);
    for (Index d = 0; d < n; ++d)
      in_basis[d] = coarse_index[d] >= 0 ? 1.0 : 0.0;
    addAcrossRanks(in_basis);
    for (std::size_t i = 0; i < _owned_node.size(); ++i)
    {
      const int owner = _partition.node_owner[_local_to_global[i]];
      for (int k = 0; k < nv; ++k)
      {
        const Index d = _local->dof(static_cast<Index>(i), k);
        coarse_index[d] = in_basis[d] > 0.5 ? owner * nv + k : -1;
      }
    }
    // A0 = R0 A R0^T, assembled from the local matrices: A = sum_r A_r, so
    // every rank adds the contribution of its own entries and the small dense
    // matrix is then summed over all ranks.
    std::vector<double> flat(static_cast<std::size_t>(coarse_size) * coarse_size, 0.0);
    for (Index col = 0; col < A.outerSize(); ++col)
    {
      if (coarse_index[col] < 0)
        continue;
      for (SparseMatrix::InnerIterator it(A, col); it; ++it)
        if (coarse_index[it.row()] >= 0)
          flat[static_cast<std::size_t>(coarse_index[it.row()]) * coarse_size +
               coarse_index[col]] += it.value();
    }
    _comm.sumInPlace(flat);
    Eigen::MatrixXd A0(coarse_size, coarse_size);
    for (int i = 0; i < coarse_size; ++i)
      for (int j = 0; j < coarse_size; ++j)
        A0(i, j) = flat[static_cast<std::size_t>(i) * coarse_size + j];
    // A subdomain with no free degree of freedom of some variable gives an
    // empty row and column; the identity there keeps A0 invertible and leaves
    // the other unknowns alone.
    for (int i = 0; i < coarse_size; ++i)
      if (A0.row(i).cwiseAbs().maxCoeff() == 0.0)
        A0(i, i) = 1.0;
    Eigen::FullPivLU<Eigen::MatrixXd> lu(A0);
    if (lu.isInvertible())
    {
      pc.coarse = lu.inverse();
      pc.have_coarse = true;
    }
  }

  const auto applyMatrix = [&](const Vector & x)
  {
    Vector y = A * x;
    addAcrossRanks(y);
    return y;
  };
  // The one-level part: restricted additive Schwarz, or, for the conjugate
  // gradient method, which needs a symmetric preconditioner, the classical
  // (unrestricted) additive Schwarz method.  With an exact or ILU(0)
  // subdomain solve of a symmetric matrix the classical form is symmetric;
  // the restricted form never is.
  const bool symmetric = _options.linear_solver == "cg";
  const auto applySchwarz = [&](const Vector & r)
  {
    // R_d r: the residual on the whole subdomain, the ghost values coming
    // from their owners.
    Vector rd;
    if (ranks > 1)
      gatherGhosts(r, rd);
    else
      rd = r;
    Vector zd = pc.exact ? Vector(pc.lu.solve(rd)) : pc.ilu.solve(rd);
    Vector z = zd.head(n);
    if (!symmetric)
    {
      // Restricted: keep only the owned part of each subdomain's correction,
      // then add the corrections together.
      for (std::size_t i = 0; i < _owned_node.size(); ++i)
        if (!_owned_node[i])
          for (int k = 0; k < nv; ++k)
            z[_local->dof(static_cast<Index>(i), k)] = 0.0;
      addAcrossRanks(z);
      return z;
    }
    // Classical: add every subdomain's whole correction, sum_d R_d^T z_d.
    // The local nodes are summed among the ranks that hold them, the ghost
    // parts are added to their owners, and the owners' totals are copied back
    // to every rank holding the node.
    addAcrossRanks(z);
    if (ranks > 1)
      addGhostsToOwners(zd, z);
    copyFromOwners(z);
    return z;
  };
  // The coarse correction R0^T A0^{-1} R0 r.  R0 r sums the residual over the
  // owned degrees of freedom of each subdomain; every rank fills its own
  // entries and one reduction makes the short vector known everywhere.
  const auto applyCoarse = [&](const Vector & r)
  {
    std::vector<double> coarse_rhs(coarse_size, 0.0);
    for (std::size_t i = 0; i < _owned_node.size(); ++i)
    {
      if (!_owned_node[i])
        continue;
      for (int k = 0; k < nv; ++k)
      {
        const Index d = _local->dof(static_cast<Index>(i), k);
        if (coarse_index[d] >= 0)
          coarse_rhs[coarse_index[d]] += r[d];
      }
    }
    _comm.sumInPlace(coarse_rhs);
    const Eigen::VectorXd y =
        pc.coarse * Eigen::Map<const Eigen::VectorXd>(coarse_rhs.data(), coarse_size);
    Vector z = Vector::Zero(n);
    for (Index d = 0; d < n; ++d)
      if (coarse_index[d] >= 0)
        z[d] = y[coarse_index[d]];
    return z;
  };
  const auto applyPreconditioner = [&](const Vector & r)
  {
    if (pc.kind == LocalPreconditioner::Kind::Jacobi)
      return Vector(pc.inverse_diagonal.cwiseProduct(r));
    if (!pc.have_coarse)
      return applySchwarz(r);
    if (symmetric)
      return Vector(applyCoarse(r) + applySchwarz(r));
    // The two levels are combined multiplicatively, coarse first:
    //
    //   z0 = Q r ,   z = z0 + M^{-1} (r - A z0) ,   Q = R0^T A0^{-1} R0 ,
    //
    // that is, z = (M^{-1} P + Q) r with P = I - A Q, the operator called
    // A-DEF1 by Tang, Nabben, Vuik and Erlangga (J. Sci. Comput. 39 (2009)
    // 340-370).  Adding the two corrections instead (their P_AD) was measured
    // here to be worse than the one-level method: the coarse and the
    // restricted local corrections overlap and the sum overshoots.  The extra
    // matrix-vector product is cheap next to the subdomain solve.  The
    // conjugate gradient method needs a symmetric operator, so for it the
    // additive form is used.
    const Vector z0 = applyCoarse(r);
    return Vector(z0 + applySchwarz(r - applyMatrix(z0)));
  };

  Vector x = Vector::Zero(n);
  Vector r = b;
  const double b_norm = norm(b);
  if (b_norm == 0.0)
    return x;
  const double target = _options.linear_tolerance * b_norm;

  if (_options.linear_solver == "cg")
  {
    Vector z = applyPreconditioner(r);
    Vector p = z;
    double rz = dot(r, z);
    for (int it = 0; it < _options.linear_max_iterations; ++it)
    {
      if (iterations)
        ++(*iterations);
      Vector Ap = applyMatrix(p);
      const double pAp = dot(p, Ap);
      if (pAp == 0.0)
        break;
      const double alpha = rz / pAp;
      x += alpha * p;
      r -= alpha * Ap;
      if (norm(r) <= target)
        return x;
      z = applyPreconditioner(r);
      const double rz_new = dot(r, z);
      p = z + (rz_new / rz) * p;
      rz = rz_new;
    }
    if (norm(r) > target)
      throw std::runtime_error(
          "dualmesh: the distributed conjugate gradient method did not converge. It requires a "
          "symmetric positive definite matrix, which the Galerkin finite element method gives "
          "but the dual mesh and finite volume methods do not; use linear_solver = 'bicgstab' "
          "with those.");
    return x;
  }
  if (_options.linear_solver == "bicgstab")
  {
    // BiCGSTAB keeps a fixed shadow residual r0 and builds its iterates from
    // the bi-orthogonality of r against it.  When that bi-orthogonality is
    // lost the scalar rho = (r0, r) collapses towards zero and the method
    // breaks down: the next beta is meaningless and the iteration stalls or
    // diverges.  The standard remedy is to restart, taking the current
    // residual as the new shadow vector, which throws away the Krylov space
    // built so far but keeps the iterate reached with it.  Restarting is what
    // makes the method usable with a strongly non-symmetric preconditioner
    // such as restricted additive Schwarz on many subdomains, where breakdown
    // is otherwise common.
    Vector r0 = r;
    Vector p = Vector::Zero(n), v = Vector::Zero(n);
    double rho = 1, alpha = 1, omega = 1;
    const double eps = std::numeric_limits<double>::epsilon();
    int restarts = 0;
    const int max_restarts = 20;
    for (int it = 0; it < _options.linear_max_iterations; ++it)
    {
      const double rho_new = dot(r0, r);
      // Breakdown is a near-zero rho, not an exactly zero one; comparing
      // against the product of the norms is what makes the test scale free.
      if (std::abs(rho_new) <= eps * norm(r0) * norm(r))
      {
        if (++restarts > max_restarts)
          break;
        r0 = r;
        p.setZero();
        v.setZero();
        rho = alpha = omega = 1;
        continue;
      }
      const double beta = (rho_new / rho) * (alpha / omega);
      p = r + beta * (p - omega * v);
      if (iterations)
        ++(*iterations);
      Vector y = applyPreconditioner(p);
      v = applyMatrix(y);
      const double r0v = dot(r0, v);
      if (std::abs(r0v) <= eps * norm(r0) * norm(v))
      {
        if (++restarts > max_restarts)
          break;
        r0 = r;
        p.setZero();
        v.setZero();
        rho = alpha = omega = 1;
        continue;
      }
      alpha = rho_new / r0v;
      Vector s = r - alpha * v;
      x += alpha * y;
      if (norm(s) <= target)
        return x;
      Vector z = applyPreconditioner(s);
      Vector t = applyMatrix(z);
      const double tt = dot(t, t);
      omega = tt != 0.0 ? dot(t, s) / tt : 0.0;
      x += omega * z;
      r = s - omega * t;
      rho = rho_new;
      if (norm(r) <= target)
        return x;
      if (omega == 0.0)
      {
        // With omega zero the update above did nothing in the second half
        // step; restart rather than divide by it on the next pass.
        if (++restarts > max_restarts)
          break;
        r0 = r;
        p.setZero();
        v.setZero();
        rho = alpha = omega = 1;
      }
    }
    if (norm(r) > target)
    {
      std::ostringstream os;
      os << "dualmesh: the distributed BiCGSTAB did not converge in "
         << _options.linear_max_iterations << " iterations (" << restarts << " restart"
         << (restarts == 1 ? "" : "s") << "), reaching a relative residual of "
         << norm(r) / target * _options.linear_tolerance
         << ". Try a larger overlap, subdomain_solver = 'lu', preconditioner = "
            "'two_level_schwarz' if another was chosen, or a looser linear_tolerance.";
      throw std::runtime_error(os.str());
    }
    return x;
  }
  throw InputError("Unknown distributed linear solver '" + _options.linear_solver +
                   "' (use cg or bicgstab).");
}

SolveResult
DistributedProblem::nonlinearSolve(const SolverOptions & o,
                                   Problem::AssemblyOptions base,
                                   const Vector * steady_old)
{
  prepare();
  const bool picard = o.nonlinear_solver == "picard";
  base.mode = picard ? LinearizationMode::Picard : LinearizationMode::Newton;
  std::vector<double> factors = o.load_factors;
  if (factors.empty())
    factors = {1.0};

  SolveResult result;
  Vector R;
  SparseMatrix J;
  Vector & U = _local->solution();
  int step_index = 0;
  for (double lf : factors)
  {
    ++step_index;
    base.load_factor = lf;
    _local->applyDirichlet(U, lf);
    copyFromOwners(U);
    double r0 = -1;
    bool converged = false;
    for (int it = 1; it <= o.max_iterations + 1; ++it)
    {
      Vector lagU = U;
      base.lagged = &lagU;
      _local->assemble(U, base, R, &J);
      if (steady_old)
        R += *steady_old;
      addAcrossRanks(R);
      _local->dirichletRows(U, lf, R, &J);
      const double rn = norm(R);
      if (r0 < 0)
        r0 = rn;
      IterationRecord record{step_index, lf, it, rn, 0.0};
      if (rn <= o.absolute_tolerance || (it > 1 && rn <= o.relative_tolerance * r0) ||
          (o.nonlinear_solver == "linear" && it > 1))
      {
        converged = true;
        result.history.push_back(record);
        if (o.verbose && _comm.isRoot())
          std::cout << "  load " << lf << " iteration " << it << " |R| = " << rn
                    << " (converged)\n";
        break;
      }
      if (it > o.max_iterations)
      {
        result.history.push_back(record);
        break;
      }
      Vector delta = solveLinearSystem(J, -R, &result.linear_iterations);
      copyFromOwners(delta);
      Vector Unew = U + delta;
      if (o.relaxation > 0 && it > 1)
        Unew = (1.0 - o.relaxation) * Unew + o.relaxation * U;
      const double un = norm(Unew);
      record.step_norm = norm(Vector(Unew - U)) / (un > 0 ? un : 1.0);
      U = Unew;
      result.history.push_back(record);
      ++result.total_iterations;
      if (o.verbose && _comm.isRoot())
        std::cout << "  load " << lf << " iteration " << it << " |R| = " << std::scientific
                  << std::setprecision(3) << rn << " |dU|/|U| = " << record.step_norm << "\n";
      if (o.nonlinear_solver == "linear" || (it > 1 && record.step_norm <= o.step_tolerance))
      {
        converged = true;
        break;
      }
    }
    if (!converged)
    {
      result.converged = false;
      if (o.error_on_divergence)
      {
        std::ostringstream os;
        os << "dualmesh: the distributed nonlinear solve did not converge at load factor " << lf
           << " within " << o.max_iterations << " iterations.";
        throw std::runtime_error(os.str());
      }
      return result;
    }
  }
  result.converged = true;
  return result;
}

SolveResult
DistributedProblem::solveSteady(const SolverOptions & options)
{
  prepare();
  Problem::AssemblyOptions base;
  base.include_time_kernels = false;
  base.theta = 1.0;
  return nonlinearSolve(options, base, nullptr);
}

SolveResult
DistributedProblem::solveTransient(const TransientOptions & tr, const SolverOptions & options)
{
  prepare();
  if (tr.dt <= 0)
    throw InputError("Transient: the time step must be positive.");
  SolverOptions attempt = options;
  attempt.error_on_divergence = false;
  Vector & U = _local->solution();
  _local->setTime(tr.start_time);
  _local->applyDirichlet(U, 1.0);
  copyFromOwners(U);
  double time = tr.start_time;
  const auto write = [&](int step)
  {
    if (tr.output_interval > 0 && step % tr.output_interval == 0 && !tr.output_file_base.empty())
    {
      std::ostringstream os;
      os << tr.output_file_base << "_" << std::setw(5) << std::setfill('0') << step;
      writeVTU(os.str());
    }
  };
  write(0);
  // Every rank runs the same controller on the same numbers, so they all take
  // the same steps and accept or reject together: the residual norms that the
  // decision rests on are global reductions, and the error estimate is formed
  // from the global norm as well.
  return runTransient(
      tr,
      options,
      U,
      time,
      [&](const Vector & old, double dt)
      {
        _local->setTime(time);
        Vector old_residual;
        if (tr.theta < 1.0)
        {
          Problem::AssemblyOptions ao;
          ao.include_time_kernels = false;
          ao.theta = 1.0 - tr.theta;
          ao.dt = dt;
          // The old part of the theta method at the old time; see
          // Problem::takeTimeStep.
          _local->setTime(time - dt);
          _local->assemble(old, ao, old_residual, nullptr);
          _local->setTime(time);
        }
        Problem::AssemblyOptions base;
        base.old = &old;
        base.dt = dt;
        base.theta = tr.theta;
        base.include_time_kernels = true;
        base.include_steady_terms = tr.theta > 0;
        return nonlinearSolve(attempt, base, tr.theta < 1.0 ? &old_residual : nullptr);
      },
      [&](int step, double) { write(step); },
      _comm.isRoot());
}

std::vector<double>
DistributedProblem::gatheredValues(const std::string & variable) const
{
  const auto local_values = _local->values(variable);
  std::vector<double> global(_num_global_nodes, 0.0);
  for (std::size_t i = 0; i < _local_to_global.size(); ++i)
    if (_owned_node[i])
      global[_local_to_global[i]] = local_values[i];
  _comm.sumInPlace(global);
  return global;
}

void
DistributedProblem::writeVTU(const std::string & base,
                             const std::vector<std::string> & cell_properties) const
{
  if (_comm.size() == 1)
  {
    _local->writeVTU(base + ".vtu", cell_properties);
    return;
  }
  std::ostringstream piece;
  piece << base << "_" << std::setw(4) << std::setfill('0') << _comm.rank() << ".vtu";
  _local->writeVTU(piece.str(), cell_properties);
  _comm.barrier();
  if (!_comm.isRoot())
    return;
  // The ".pvtu" index lists the pieces; ParaView and VisIt open it as one
  // data set.
  std::ofstream f(base + ".pvtu");
  if (!f)
    throw InputError("Cannot open '" + base + ".pvtu' for writing.");
  f << "<?xml version=\"1.0\"?>\n<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" "
       "byte_order=\"LittleEndian\">\n<PUnstructuredGrid GhostLevel=\"0\">\n";
  f << "<PPoints><PDataArray type=\"Float64\" NumberOfComponents=\"3\"/></PPoints>\n";
  f << "<PPointData>\n";
  for (int v = 0; v < _local->numVariables(); ++v)
    f << "<PDataArray type=\"Float64\" Name=\"" << _local->variable(v).name << "\"/>\n";
  f << "</PPointData>\n<PCellData>\n<PDataArray type=\"Int32\" Name=\"block\"/>\n";
  for (const auto & p : cell_properties)
    f << "<PDataArray type=\"Float64\" Name=\"" << p << "\"/>\n";
  f << "</PCellData>\n";
  for (int r = 0; r < _comm.size(); ++r)
  {
    std::ostringstream name;
    name << base << "_" << std::setw(4) << std::setfill('0') << r << ".vtu";
    // Strip any directory, because the index and the pieces sit side by side.
    std::string file = name.str();
    const auto slash = file.find_last_of('/');
    if (slash != std::string::npos)
      file = file.substr(slash + 1);
    f << "<Piece Source=\"" << file << "\"/>\n";
  }
  f << "</PUnstructuredGrid>\n</VTKFile>\n";
}

std::string
DistributedProblem::summary() const
{
  const auto [largest, smallest] = _partition.partSizes();
  std::ostringstream os;
  os << "distributed problem: " << _comm.size() << " rank" << (_comm.size() == 1 ? "" : "s")
     << ", partitioner " << _options.partitioner << (haveMetis() ? "" : " (no METIS in this build)")
     << ", elements per rank " << smallest << " to " << largest << ", rank " << _comm.rank()
     << " holds " << _mesh->numElements() << " elements and " << _mesh->numNodes()
     << " nodes of which " << std::count(_owned_node.begin(), _owned_node.end(), 1) << " are owned";
  return os.str();
}

} // namespace dualmesh
