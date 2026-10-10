// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Distributed-memory (MPI) solver.
//
// How it works
// ------------
// The mesh is divided into one group of elements per rank.
// Each rank holds the sub-mesh of its own elements, every node those elements touch and the global index of each node (a LocalMesh), and defines on it exactly the same variables, kernels, property objects and boundary conditions as a serial run would.
// No rank needs the whole mesh: a gather-scatter (GatherScatter.h) finds from the global node indices which ranks share each node and which one owns it, and every later exchange goes to the neighboring ranks only.
// A whole mesh can also be given on the root rank, which partitions it and sends every rank its part.
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
// The cell-centered finite volume method has its unknowns at the cells and at the faces of the boundary of the domain, and integrates over the faces.
// Every part then also holds two layers of ghost cells, the cells of other ranks next to its own, so that the gradients of the cells on both sides of a face between two parts are reconstructed from all their neighbors.
// Each face is integrated by one rank, the one whose cell on it has the smaller global number, and the gather-scatter adds the pieces at the cells and faces exactly as it adds them at shared nodes.
//
// The linear system is one distributed PETSc matrix: every rank adds the entries of its local matrix in a global numbering of the degrees of freedom, in which each rank owns a contiguous range.
// PETSc solves it with its Krylov methods and preconditioners: by default algebraic multigrid (hypre BoomerAMG) inside GMRES, and the parallel direct solver MUMPS when multigrid fails or the system is a saddle point problem; petsc_options chooses any other.
// Inner products count every degree of freedom once, on the rank that owns it, and are then reduced over all ranks.
//
// Without MPI, or with MPI on a single rank, all of this reduces to the serial
// algorithm, and the answer is identical.
#pragma once

#include "dualmesh/base/Problem.h"
#include "dualmesh/linalg/PetscSolver.h"
#include "dualmesh/parallel/Communicator.h"
#include "dualmesh/parallel/GatherScatter.h"
#include "dualmesh/parallel/MeshDistribution.h"

#include <memory>
#include <string>
#include <vector>

namespace dualmesh
{

struct DistributedOptions
{
  /// The library that exchanges the values at the nodes that several ranks hold: "petsc" (PETSc's star forests) or "gslib" (the gather-scatter of Nek5000, which times pairwise exchanges, a crystal router and an all-reduce at setup and keeps the fastest).
  std::string gather_scatter = "petsc";
  /// PETSc's partitioner of the elements: "automatic" (PT-Scotch, else ParMETIS), "ptscotch", "parmetis" or "simple".
  std::string partitioner = "automatic";
  double linear_tolerance = 1e-10;
  int linear_max_iterations = 5000;
  /// PETSc's option string, for example "-ksp_type cg -pc_type gamg" or "-ksp_type gmres -pc_type asm -sub_pc_type ilu".
  /// Empty selects the automatic choice: GMRES with hypre BoomerAMG, and MUMPS when that fails or the system has a zero pressure block (the Taylor-Hood element).
  std::string petsc_options;
  bool verbose = false;
};

class DistributedProblem
{
public:
  /// @param part this rank's part of the mesh.
  ///        No rank needs the whole mesh: the ranks that share nodes and the owners are found from the global node indices alone.
  DistributedProblem(LocalMesh part,
                     Method method = Method::DualMesh,
                     CoordinateSystem coord = CoordinateSystem::Cartesian,
                     const DistributedOptions & options = {});
  /// @param global_mesh the whole mesh, read on the root rank only; the other ranks may pass an empty mesh.
  ///        The root partitions it with options.partitioner and sends every rank its part (distributeMesh).
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
  int rank() const { return _comm.rank(); }
  int numRanks() const { return _comm.size(); }

  /// Global index of a local node, or, for the cell-centered method, of a local cell or boundary face (the elements first, then the boundary faces; -1 for an outer face of the outermost ghost cells, which only this rank has).
  Index globalNode(Index local_node) const { return _local_to_global[local_node]; }
  /// Whether this rank owns a local node, or cell or boundary face for the cell-centered method (and therefore its degrees of freedom).
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
  /// For the cell-centered method, the outer faces of the outermost ghost cells, which only this rank has, take the values of their cells.
  void copyFromOwners(Vector & v) const;
  /// Inner product that counts every degree of freedom exactly once.
  double dot(const Vector & a, const Vector & b) const;

  // ---- one value per entity, for a solver that works on the local problem directly -------------
  /// Set up the distributed problem: the solves call it, and a solver that works on the local problem directly calls it first.
  void prepare();
  /// addAcrossRanks and copyFromOwners for @p width values per local entity (a node, or a cell and a boundary face of the cell-centered method).
  void addAcrossRanks(double * values, int width) const { _gs.sum(values, width); }
  void copyFromOwners(double * values, int width) const { _gs.copyFromOwners(values, width); }
  /// The global number of every local entity, in a numbering in which each rank numbers its owned entities contiguously from firstOwnedEntity(); a copy has its owner's number, and an entity that only this rank has -1.
  const std::vector<Index> & globalEntityNumbers() const { return _global_entity; }
  Index firstOwnedEntity() const { return _first_owned_entity; }
  Index numOwnedEntities() const;
  Index numGlobalEntities() const { return _num_global_entities; }
  /// Write the prescribed values at the time of the local problem into @p U on every rank that holds them.
  void applyDirichlet(Vector & U, double load_factor = 1.0) const;
  double norm(const Vector & a) const { return std::sqrt(dot(a, a)); }

  /// Change the settings of the linear solver: everything in @p options except the partitioner, which is used at construction.
  void setLinearSolver(const DistributedOptions & options);

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
                const std::vector<std::string> & cell_properties = {},
                const std::vector<std::string> & fields = {}) const;

  /// The sum of the reactions (the residual of the converged solution) of @p variable over the nodes of @p boundary, each node counted once, on the rank that owns it.
  double totalReaction(const std::string & variable, const std::string & boundary) const;

  /// A one-line description of the partition, for logs.
  std::string summary() const;

private:
  Vector solveLinearSystem(const SparseMatrix & A,
                           const Vector & b,
                           const SolverOptions & o,
                           int * iterations) const;
  /// The global system of this rank: its entries of the matrix and its owned entries of the right-hand side, in the global numbering.
  struct GlobalSystem
  {
    std::vector<petsc::GlobalEntry> entries;
    Vector b_owned;
    std::vector<char> owned_dof;
  };
  GlobalSystem globalSystem(const SparseMatrix & A, const Vector & b) const;
  /// Solve with the PETSc options @p options; when @p must_converge is false, a failure returns an empty vector instead of an error.
  Vector solveWithPetsc(const SparseMatrix & A,
                        const Vector & b,
                        const std::string & options,
                        int * iterations,
                        bool must_converge = true) const;
  SolveResult nonlinearSolve(const SolverOptions & options,
                             Problem::AssemblyOptions base,
                             const Vector * steady_old_residual);

  Communicator & _comm;
  DistributedOptions _options;
  std::shared_ptr<Mesh> _mesh; ///< the local sub-mesh
  std::unique_ptr<Problem> _local;
  std::vector<Index> _local_to_global;
  /// The global index of every local element, the ghost elements included (empty for a part built without it).
  std::vector<Index> _global_elements;
  /// The number of elements of this rank's part; the ghost elements follow them.
  Index _num_owned_elements = 0;
  /// Whether the unknowns sit at cells and boundary faces (the cell-centered finite volume method) rather than at nodes.
  bool _cell_centered = false;
  std::vector<char> _owned_node;
  Index _num_global_nodes = 0;
  /// The smallest and the largest number of elements on a rank, for summary().
  Index _fewest_elements = 0, _most_elements = 0;
  Index _num_global_dofs = 0;
  /// The global number of every local degree of
  /// freedom, with each rank's owned degrees of freedom numbered
  /// contiguously from _first_owned_dof in the order of the local numbering.
  std::vector<Index> _global_dof;
  Index _first_owned_dof = 0;
  /// The exchanges of the values at the nodes that several ranks hold.
  GatherScatter _gs;

  /// Number the cells and boundary faces of the cell-centered method, the entities of its unknowns, and set up the gather-scatter on them: the process that integrates a cell owns it and its boundary faces, and the ghost cells and their faces are copies.
  void numberCellEntities();
  /// Join the nodes that periodic constraints make one: every rank pairs the nodes of the periodic boundaries of all ranks, and the gather-scatter and the global numbering then treat a node and its primary as one entity.
  void joinConstraints();
  /// Give every rank that holds a primary side of an interface condition the secondary elements it lacks, as ghost elements that it reads but does not integrate, so that every interface point pairs with the secondary side as in a serial run.
  void fetchInterfaceGhosts();
  bool _has_ghosts = false;
  /// Mark the copies of a prescribed node that only another rank knows as prescribed, after the point conditions have been resolved.
  void agreePrescribedValues();
  /// The degrees of freedom that this rank's nodal boundary conditions prescribe.
  std::vector<char> localPrescribed() const;
  /// Give the copies of a prescribed node that only another rank knows as prescribed the prescribed value.
  void shareDirichletValues(Vector & U) const;
  /// Whether this rank owns each local node as a node of the mesh, before constraints join nodes: the rank that reports and loads it.
  std::vector<char> _owns_node;
  /// Per local degree of freedom, whether only another rank prescribes its value (a copy of a prescribed node).
  std::vector<char> _prescribed_elsewhere;
  /// Complete the boundary node sets: a node on a side set held by this rank, whose side belongs to another rank, joins the node set of that side set.
  void completeBoundaryNodes();
  bool _prepared = false;
  std::vector<Index> _global_entity;
  Index _first_owned_entity = 0, _num_global_entities = 0;
};

} // namespace dualmesh
