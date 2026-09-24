// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Distributed-memory (MPI) solver.
//
// How it works
// ------------
// The mesh is partitioned into one group of elements per rank.  Each rank
// builds the sub-mesh of its own elements, together with every node those
// elements touch, and then defines exactly the same variables, kernels,
// materials and boundary conditions on that sub-mesh as a serial run would.
// The rank-local object is an ordinary Problem, so every discretization, every
// physics module and the automatic differentiation all work unchanged.
//
// Because every element belongs to exactly one rank, the sum of the local
// residuals is the global residual:
//
//     R(U) = sum_r R_r(U) ,      J(U) = sum_r J_r(U) ,
//
// where R_r and J_r collect the contributions of rank r's elements only.  A
// node in the interior of a partition gets its whole control domain from one
// rank; a node on a partition boundary gets a piece of its control domain from
// each rank that touches it.  One exchange with the neighbouring ranks adds
// those pieces together, after which every rank that holds a node has the same,
// complete value there.  Vectors kept in that state are called *consistent*.
//
// The linear system is never assembled as one distributed matrix.  Instead the
// Krylov solvers use the local matrices directly:
//
//     J x = sum_r ( J_r x )   followed by one exchange,
//
// which is exact, needs no global numbering of the matrix entries, and keeps
// the communication to the partition boundary.  Inner products count every
// degree of freedom once, on the rank that owns it, and are then reduced over
// all ranks.  Two kinds of preconditioner are available: the diagonal of the
// global matrix (Jacobi), and restricted additive Schwarz on overlapping
// subdomains, with an optional coarse level of one unknown per subdomain.
// The Schwarz subdomain of a rank is its own elements extended by a few layers
// of its neighbours' elements; its matrix is the fully assembled global matrix
// restricted to the subdomain, which the ranks build by sending each other the
// entries of their local matrices that fall inside a neighbour's subdomain.
//
// Without MPI, or with MPI on a single rank, all of this reduces to the serial
// algorithm, and the answer is identical.
#pragma once

#include "dualmesh/base/Problem.h"
#include "dualmesh/parallel/Communicator.h"
#include "dualmesh/parallel/Partition.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace dualmesh
{

struct DistributedOptions
{
  /// "recursive_coordinate_bisection", "graph", or "metis".
  std::string partitioner = "graph";
  /// "bicgstab" (the default; works for any matrix) or "cg" (only for a
  /// symmetric positive definite matrix, which the Galerkin finite element
  /// method gives but the dual mesh and finite volume methods do not).
  std::string linear_solver = "bicgstab";
  /// The preconditioner for the distributed Krylov solver.
  ///
  /// "jacobi" divides by the diagonal of the global matrix.  It is the same
  /// operator whatever the partition, so its iteration count does not depend
  /// on the number of ranks, but that count is large (it grows like the mesh
  /// size in one direction).
  ///
  /// "additive_schwarz", the default, is the restricted additive Schwarz
  /// method of Cai and Sarkis (1999) on overlapping subdomains.  Every rank's
  /// subdomain is its own elements extended by `overlap` layers of the
  /// neighbours' elements; the matrix of the subdomain is the global matrix
  /// restricted to it, R_d A R_d^T, with every row fully assembled (the
  /// ranks exchange the entries they hold); the subdomain problem is solved
  /// approximately (`subdomain_solver`), and each rank keeps the correction
  /// only on the degrees of freedom it owns.  "two_level_schwarz" adds a
  /// coarse correction with one unknown per subdomain and variable, which is
  /// what keeps the iteration count from growing with the number of ranks.
  std::string preconditioner = "two_level_schwarz";
  /// Layers of elements by which each subdomain is extended into its
  /// neighbours for the Schwarz preconditioners.  Zero gives non-overlapping
  /// subdomains (block Jacobi with fully assembled interface rows).
  int overlap = 1;
  /// How the subdomain problems of the Schwarz preconditioners are solved:
  /// "ilu" (incomplete LU without fill, cheap) or "lu" (exact sparse LU).
  std::string subdomain_solver = "ilu";
  double linear_tolerance = 1e-10;
  int linear_max_iterations = 5000;
  bool verbose = false;
};

class DistributedProblem
{
public:
  /// @param global_mesh the whole mesh, which every rank reads.  Each rank
  ///        keeps only its own part after construction.
  DistributedProblem(const Mesh & global_mesh,
                     Method method = Method::DualMesh,
                     CoordinateSystem coord = CoordinateSystem::Cartesian,
                     const DistributedOptions & options = {});

  /// The rank-local problem.  Define variables and objects on it exactly as in
  /// a serial run, then call solveSteady() or solveTransient() here rather than
  /// on the local problem.
  Problem & local() { return *_local; }
  const Problem & local() const { return *_local; }
  const Communicator & communicator() const { return _comm; }
  const MeshPartition & partition() const { return _partition; }
  int rank() const { return _comm.rank(); }
  int numRanks() const { return _comm.size(); }

  /// Global node index of a local node.
  Index globalNode(Index local_node) const { return _local_to_global[local_node]; }
  /// Whether this rank owns a local node (and therefore its degrees of freedom).
  bool ownsNode(Index local_node) const { return _owned_node[local_node] != 0; }
  /// Number of degrees of freedom this rank owns.
  Index numOwnedDofs() const;
  /// Number of degrees of freedom of the whole problem.  The total is computed
  /// once, during setup, so that this is a local query: reading it on one rank
  /// only cannot deadlock.  It is available after the problem has been given
  /// its variables and is zero before that.
  Index numGlobalDofs() const { return _num_global_dofs; }

  /// Make a local vector consistent: add the contributions of every rank that
  /// touches a node, so that all of them end up with the same complete value.
  void addAcrossRanks(Vector & v) const;
  /// Replace shared values by the owner's value.  Used after an operation that
  /// is not guaranteed to give the same answer on every rank.
  void copyFromOwners(Vector & v) const;
  /// Inner product that counts every degree of freedom exactly once.
  double dot(const Vector & a, const Vector & b) const;
  double norm(const Vector & a) const { return std::sqrt(dot(a, a)); }

  SolveResult solveSteady(const SolverOptions & options = {});
  SolveResult solveTransient(const TransientOptions & transient,
                             const SolverOptions & options = {});

  /// Values of a variable at every node of the *global* mesh, assembled on
  /// every rank.  Convenient for testing and for small problems; it needs one
  /// vector of the global size per rank, so it is not meant for large runs.
  std::vector<double> gatheredValues(const std::string & variable) const;

  /// Write one ".vtu" file per rank plus a ".pvtu" index that ParaView and
  /// VisIt open as a single data set.  @p base is the file name without an
  /// extension.
  void writeVTU(const std::string & base,
                const std::vector<std::string> & cell_properties = {}) const;

  /// A one-line description of the partition, for logs.
  std::string summary() const;

private:
  Vector solveLinearSystem(const SparseMatrix & A, const Vector & b, int * iterations) const;
  SolveResult nonlinearSolve(const SolverOptions & options,
                             Problem::AssemblyOptions base,
                             const Vector * steady_old_residual);

  Communicator & _comm;
  DistributedOptions _options;
  MeshPartition _partition;
  std::shared_ptr<Mesh> _mesh; ///< the local sub-mesh
  std::unique_ptr<Problem> _local;
  std::vector<Index> _local_to_global;
  std::vector<char> _owned_node;
  Index _num_global_nodes = 0;
  Index _num_global_dofs = 0;
  /// For every other rank, the local nodes shared with it, sorted by global
  /// node index so that both sides agree on the order without communicating.
  std::vector<std::vector<Index>> _shared_nodes;

  // ---- overlapping subdomain of the Schwarz preconditioners ---------------
  // The subdomain is numbered by extending the local numbering: the local
  // nodes keep their indices and the ghost nodes (in the overlap but not
  // touched by a local element) follow them.
  /// Global index of every ghost node, in subdomain order.
  std::vector<Index> _ghost_global;
  /// Subdomain index of every node of the subdomain, by global index.
  std::unordered_map<Index, Index> _subdomain_index;
  /// For every rank, the local nodes that lie in that rank's subdomain; the
  /// matrix entries between them are sent to it.
  std::vector<std::vector<Index>> _overlap_send_nodes;
  /// For every rank, the local (owned) nodes whose values it needs as ghosts,
  /// in the order it asked for them ...
  std::vector<std::vector<Index>> _ghost_send_nodes;
  /// ... and, for every rank, the subdomain indices of the ghost nodes whose
  /// values it sends here, in the same order.
  std::vector<std::vector<Index>> _ghost_recv_slots;
  void buildOverlap(const Mesh & global_mesh, const std::vector<Index> & elements);
  /// Fill the ghost part of a subdomain vector from the owners.
  void gatherGhosts(const Vector & local, Vector & subdomain) const;
  /// The transpose of gatherGhosts: add the ghost part of a subdomain vector
  /// to the owners' entries of @p local.
  void addGhostsToOwners(const Vector & subdomain, Vector & local) const;
  /// R_d A R_d^T, the global matrix restricted to this rank's subdomain.
  SparseMatrix subdomainMatrix(const SparseMatrix & A) const;
  mutable std::vector<std::vector<double>> _send_buffer, _recv_buffer;
  bool _prepared = false;
  void prepare();
};

} // namespace dualmesh
