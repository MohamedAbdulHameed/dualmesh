// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/linalg/PetscSolver.h"

#include <algorithm>
#include <stdexcept>

#ifndef DUALMESH_HAVE_PETSC
#include <Eigen/SparseLU>
#endif

#ifdef DUALMESH_HAVE_PETSC
#include "dualmesh/parallel/Communicator.h"

#include <petscksp.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>
#endif

namespace dualmesh
{
namespace petsc
{

SchurBlocks
globalSchurBlocks(const SaddlePointBlocks & blocks,
                  const std::vector<Index> & global_dof,
                  const std::vector<char> & owned)
{
  SchurBlocks g;
  g.approximation = blocks.approximation;
  g.sign = blocks.sign;
  for (Index d : blocks.other_dofs)
    if (owned[d])
      g.other_rows.push_back(global_dof[d]);
  for (Index d : blocks.pressure_dofs)
    if (owned[d])
    {
      g.pressure_rows.push_back(global_dof[d]);
      if (blocks.constrained[d])
        g.constrained_pressure_rows.push_back(global_dof[d]);
    }
  for (auto * rows : {&g.other_rows, &g.pressure_rows, &g.constrained_pressure_rows})
  {
    std::sort(rows->begin(), rows->end());
    rows->erase(std::unique(rows->begin(), rows->end()), rows->end());
  }
  const auto global = [&](const std::vector<Eigen::Triplet<double>> & local)
  {
    std::vector<GlobalEntry> entries;
    entries.reserve(local.size());
    for (const auto & e : local)
      entries.push_back({global_dof[e.row()], global_dof[e.col()], e.value()});
    return entries;
  };
  g.mass = global(blocks.mass);
  g.stabilization = global(blocks.stabilization);
  g.commutator_stabilization = global(blocks.commutator_stabilization);
  g.convection_diffusion = global(blocks.convection_diffusion);
  for (Index d = 0; d < blocks.lumped_mass.size(); ++d)
    if (blocks.lumped_mass[d] != 0.0)
      g.lumped_mass.push_back({global_dof[d], global_dof[d], blocks.lumped_mass[d]});
  return g;
}

#ifndef DUALMESH_HAVE_PETSC

bool
available()
{
  return false;
}

std::string
version()
{
  return "";
}

void
initialize()
{
}

void
finalize()
{
}

void
check(int, const char *)
{
}

namespace
{
[[noreturn]] void
missing()
{
  throw std::runtime_error(
      "dualmesh: linear_solver='PETSc' needs a build with PETSc. Install PETSc (for example "
      "the libpetsc-real-dev package, or 'pip install petsc') and configure with "
      "-DDUALMESH_ENABLE_PETSC=ON.");
}
} // namespace

Vector
solve(const SparseMatrix &, const Vector &, const Settings &, Result &)
{
  missing();
}

Vector
solveDistributed(
    Index, Index, const std::vector<GlobalEntry> &, const Vector &, const Settings &, Result &)
{
  missing();
}

struct LinearSolver::Impl
{
  Eigen::SparseLU<SparseMatrix> lu;
  Index size = 0;
  std::vector<GlobalEntry> entries;

  void factor()
  {
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(entries.size());
    for (const auto & e : entries)
      triplets.emplace_back(e.row, e.col, e.value);
    SparseMatrix A(size, size);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();
    lu.compute(A);
    if (lu.info() != Eigen::Success)
      throw std::runtime_error("dualmesh: the sparse LU factorization of a linear system failed.");
  }
};

LinearSolver::LinearSolver(bool distributed,
                           Index global_size,
                           Index first_row,
                           Index num_owned,
                           const std::vector<GlobalEntry> & entries,
                           const Settings &)
    : _impl(std::make_unique<Impl>())
{
  // Without PETSc there is one process, which holds the whole system.
  (void) distributed;
  (void) first_row;
  (void) num_owned;
  _impl->size = global_size;
  _impl->entries = entries;
  _impl->factor();
}

LinearSolver::~LinearSolver() = default;

void
LinearSolver::setValues(const std::vector<double> & values)
{
  if (values.size() != _impl->entries.size())
    throw std::invalid_argument("dualmesh: new values for a different number of entries.");
  for (std::size_t k = 0; k < values.size(); ++k)
    _impl->entries[k].value = values[k];
  _impl->factor();
}

Vector
LinearSolver::solve(const Vector & b_owned, const Vector *, Result & result) const
{
  Vector x = _impl->lu.solve(b_owned);
  result.iterations = 1;
  result.converged = _impl->lu.info() == Eigen::Success;
  result.reason = result.converged ? "sparse LU" : "sparse LU failed";
  return x;
}

#else

void
check(int code, const char * what)
{
  if (code != 0)
  {
    const char * text = nullptr;
    PetscErrorMessage(code, &text, nullptr);
    throw std::runtime_error(std::string("dualmesh: PETSc failed in ") + what + ": " +
                             (text ? text : "unknown error"));
  }
}

namespace
{
bool g_we_initialized = false;

#define DM_PETSC(call) check((call), #call)

/// The options of one solve, in a private database made the default while the
/// solver is set up, so that they apply to this solve only.
class ScopedOptions
{
public:
  explicit ScopedOptions(const std::string & text)
  {
    DM_PETSC(PetscOptionsCreate(&_options));
    if (!text.empty())
      DM_PETSC(PetscOptionsInsertString(_options, text.c_str()));
    DM_PETSC(PetscOptionsPush(_options));
  }
  ~ScopedOptions()
  {
    PetscOptionsPop();
    PetscOptionsDestroy(&_options);
  }
  ScopedOptions(const ScopedOptions &) = delete;
  ScopedOptions & operator=(const ScopedOptions &) = delete;

private:
  PetscOptions _options = nullptr;
};

std::string
defaultOptions(bool parallel)
{
  // MUMPS pivots during the factorisation, so it also factorises saddle point
  // systems whose diagonal has zeros (the pressure block of the Taylor-Hood
  // element); PETSc's own LU does not pivot and stops at the first zero pivot.
#ifdef PETSC_HAVE_MUMPS
  (void) parallel;
  return "-ksp_type gmres -pc_type lu -pc_factor_mat_solver_type mumps";
#else
  if (!parallel)
    return "-ksp_type gmres -pc_type lu";
  return "-ksp_type gmres -pc_type bjacobi -sub_pc_type ilu";
#endif
}

/// Check a solve that PETSc reports converged by its true residual |b - A x|.  PETSc measures
/// the preconditioned residual by default, which a preconditioner that fails without an error
/// (algebraic multigrid on rows whose diagonal is zero) makes zero.  A true residual above
/// sqrt(rtol) |b| marks the solve as failed, so that the caller falls back or stops.
void
checkTrueResidual(Mat A, Vec b, Vec x, double relative_tolerance, Result & result)
{
  if (!result.converged)
    return;
  Vec r;
  DM_PETSC(VecDuplicate(b, &r));
  DM_PETSC(MatMult(A, x, r));
  DM_PETSC(VecAYPX(r, -1.0, b));
  PetscReal r_norm = 0, b_norm = 0;
  DM_PETSC(VecNorm(r, NORM_2, &r_norm));
  DM_PETSC(VecNorm(b, NORM_2, &b_norm));
  VecDestroy(&r);
  const double limit = std::sqrt(std::max(relative_tolerance, 1e-16)) * b_norm;
  if (b_norm > 0 && !(r_norm <= limit))
  {
    std::ostringstream os;
    os << "a true residual of " << r_norm / b_norm << " times the right-hand side, although "
       << "PETSc reported " << result.reason;
    result.converged = false;
    result.reason = os.str();
  }
}

/// The options of one V-cycle of algebraic multigrid on the solver with options prefix
/// @p prefix: hypre BoomerAMG with the coarsening and interpolation that keep the cost of a cycle
/// low in three dimensions (HMIS coarsening, De Sterck, Yang and Heys 2006, aggressive on the
/// first level, and extended+i interpolation with at most four entries per row, De Sterck,
/// Falgout, Nolting and Yang 2008), or PETSc's own GAMG when PETSc has no hypre.  Measured on a
/// three-dimensional Taylor-Hood cavity, these settings make lsc three to four times faster than
/// BoomerAMG's defaults, at twice the iterations.
std::string
amgOptions(const std::string & prefix)
{
#ifdef PETSC_HAVE_HYPRE
  return " -" + prefix + "pc_type hypre -" + prefix + "pc_hypre_boomeramg_coarsen_type HMIS -" +
         prefix + "pc_hypre_boomeramg_interp_type ext+i -" + prefix +
         "pc_hypre_boomeramg_P_max 4 -" + prefix + "pc_hypre_boomeramg_agg_nl 1";
#else
  return " -" + prefix + "pc_type gamg";
#endif
}

/// The default options of a Schur complement field split; the user's options follow them and win.
std::string
schurOptions(SchurApproximation approximation)
{
  std::string o = "-ksp_type fgmres -ksp_gmres_restart 200 -pc_type fieldsplit "
                  "-pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type upper "
                  "-fieldsplit_0_ksp_type preonly" +
                  amgOptions("fieldsplit_0_") + " -fieldsplit_1_ksp_type preonly";
  if (approximation == SchurApproximation::PressureMass)
    return o + " -pc_fieldsplit_schur_precondition user" + amgOptions("fieldsplit_1_");
  return o + " -pc_fieldsplit_schur_precondition self -fieldsplit_1_laplacian_ksp_type preonly" +
         amgOptions("fieldsplit_1_laplacian_");
}

/// The default options of a field split by physics, block "0", "1", ... one after the other.
std::string
physicsBlockOptions(std::size_t blocks)
{
  std::string o = "-ksp_type fgmres -ksp_gmres_restart 200 -pc_type fieldsplit "
                  "-pc_fieldsplit_type multiplicative";
  for (std::size_t b = 0; b < blocks; ++b)
    o += " -fieldsplit_" + std::to_string(b) + "_ksp_type preonly -fieldsplit_" +
         std::to_string(b) + "_pc_type gamg";
  return o;
}

/// An index set of the given global rows of this process.
IS
indexSet(MPI_Comm comm, const std::vector<Index> & rows)
{
  std::vector<PetscInt> indices(rows.begin(), rows.end());
  IS set;
  DM_PETSC(ISCreateGeneral(
      comm, static_cast<PetscInt>(indices.size()), indices.data(), PETSC_COPY_VALUES, &set));
  return set;
}

/// A matrix of the layout of @p like from entries in the global numbering, summed, with an
/// explicit diagonal on every owned row: zero, or one on the rows of @p ones.
Mat
assembledMatrix(Mat like, const std::vector<GlobalEntry> & entries, const std::vector<Index> & ones)
{
  MPI_Comm comm;
  DM_PETSC(PetscObjectGetComm(reinterpret_cast<PetscObject>(like), &comm));
  PetscInt lo = 0, hi = 0, size = 0;
  DM_PETSC(MatGetOwnershipRange(like, &lo, &hi));
  DM_PETSC(MatGetSize(like, &size, nullptr));
  std::vector<PetscInt> rows, cols;
  std::vector<PetscScalar> values;
  const auto add = [&](PetscInt r, PetscInt c, double v)
  {
    rows.push_back(r);
    cols.push_back(c);
    values.push_back(v);
  };
  for (const auto & e : entries)
    add(static_cast<PetscInt>(e.row), static_cast<PetscInt>(e.col), e.value);
  for (PetscInt i = lo; i < hi; ++i)
    add(i, i, 0.0);
  for (Index r : ones)
    add(static_cast<PetscInt>(r), static_cast<PetscInt>(r), 1.0);
  Mat M;
  DM_PETSC(MatCreate(comm, &M));
  DM_PETSC(MatSetSizes(M, hi - lo, hi - lo, size, size));
  DM_PETSC(MatSetType(M, MATAIJ));
  DM_PETSC(
      MatSetPreallocationCOO(M, static_cast<PetscCount>(rows.size()), rows.data(), cols.data()));
  DM_PETSC(MatSetValuesCOO(M, values.data(), ADD_VALUES));
  return M;
}

/// The submatrix of @p full on the rows and columns of @p set; @p full is destroyed.
Mat
restricted(Mat full, IS set)
{
  Mat part;
  DM_PETSC(MatCreateSubMatrix(full, set, set, MAT_INITIAL_MATRIX, &part));
  MatDestroy(&full);
  return part;
}

/// The vector of the layout of @p like whose entries are the sums of the values of @p entries at their rows.
Vec
assembledVector(Mat like, const std::vector<GlobalEntry> & entries)
{
  Vec v;
  DM_PETSC(MatCreateVecs(like, nullptr, &v));
  DM_PETSC(VecSet(v, 0.0));
  for (const auto & e : entries)
    DM_PETSC(VecSetValue(v, static_cast<PetscInt>(e.row), e.value, ADD_VALUES));
  DM_PETSC(VecAssemblyBegin(v));
  DM_PETSC(VecAssemblyEnd(v));
  return v;
}

/// The part of @p v on the rows of @p set, as a vector of its own.
Vec
subvector(Vec v, IS set)
{
  Vec part, copy;
  DM_PETSC(VecGetSubVector(v, set, &part));
  DM_PETSC(VecDuplicate(part, &copy));
  DM_PETSC(VecCopy(part, copy));
  DM_PETSC(VecRestoreSubVector(v, set, &part));
  return copy;
}

/// Replace every positive entry of @p v by its reciprocal and every other by @p otherwise.
void
invert(Vec v, double otherwise)
{
  PetscScalar * a = nullptr;
  PetscInt n = 0;
  DM_PETSC(VecGetLocalSize(v, &n));
  DM_PETSC(VecGetArray(v, &a));
  for (PetscInt i = 0; i < n; ++i)
    a[i] = a[i] > 0.0 ? 1.0 / a[i] : otherwise;
  DM_PETSC(VecRestoreArray(v, &a));
}

/// The columns of @p modes (one row per owned row of @p like) as vectors, orthonormalised by the
/// modified Gram-Schmidt process; a column that depends on the earlier ones is dropped.
/// With @p set, the vectors are restricted to its rows first.
std::vector<Vec>
orthonormalModes(Mat like, const Eigen::MatrixXd & modes, IS set = nullptr)
{
  std::vector<Vec> basis;
  for (Index c = 0; c < modes.cols(); ++c)
  {
    Vec v;
    DM_PETSC(MatCreateVecs(like, nullptr, &v));
    PetscScalar * a = nullptr;
    DM_PETSC(VecGetArray(v, &a));
    for (Index i = 0; i < modes.rows(); ++i)
      a[i] = modes(i, c);
    DM_PETSC(VecRestoreArray(v, &a));
    if (set)
    {
      Vec part = subvector(v, set);
      VecDestroy(&v);
      v = part;
    }
    PetscReal before = 0;
    DM_PETSC(VecNorm(v, NORM_2, &before));
    for (Vec u : basis)
    {
      PetscScalar d = 0;
      DM_PETSC(VecDot(v, u, &d));
      DM_PETSC(VecAXPY(v, -d, u));
    }
    PetscReal after = 0;
    DM_PETSC(VecNorm(v, NORM_2, &after));
    if (!(after > 1e-10 * before) || before == 0.0)
    {
      VecDestroy(&v);
      continue;
    }
    DM_PETSC(VecScale(v, 1.0 / after));
    basis.push_back(v);
  }
  return basis;
}

/// The near null space spanned by @p modes, restricted to @p set when it is given.
MatNullSpace
nearNullSpace(Mat like, const Eigen::MatrixXd & modes, IS set = nullptr)
{
  MPI_Comm comm;
  DM_PETSC(PetscObjectGetComm(reinterpret_cast<PetscObject>(like), &comm));
  std::vector<Vec> basis = orthonormalModes(like, modes, set);
  MatNullSpace space;
  DM_PETSC(MatNullSpaceCreate(
      comm, PETSC_FALSE, static_cast<PetscInt>(basis.size()), basis.data(), &space));
  for (Vec & v : basis)
    VecDestroy(&v);
  return space;
}

/// The Schur complement preconditioner of pcd and lsc (SaddlePointSolver.h), applied by a shell.
struct SchurShell
{
  SchurApproximation approximation = SchurApproximation::ConvectionDiffusion;
  double sign = 1.0;
  Mat gradient = nullptr, momentum = nullptr, laplacian = nullptr;
  Mat convection_diffusion = nullptr, commutator_stabilization = nullptr;
  KSP laplacian_solver = nullptr;
  Vec inverse_velocity_mass = nullptr, inverse_pressure_mass = nullptr;
  Vec prescribed = nullptr, free = nullptr;
  Vec p1 = nullptr, p2 = nullptr, u1 = nullptr, u2 = nullptr;

  ~SchurShell()
  {
    if (!PetscInitializeCalled || PetscFinalizeCalled)
      return;
    KSPDestroy(&laplacian_solver);
    for (Mat * m :
         {&gradient, &momentum, &laplacian, &convection_diffusion, &commutator_stabilization})
      MatDestroy(m);
    for (Vec * v :
         {&inverse_velocity_mass, &inverse_pressure_mass, &prescribed, &free, &p1, &p2, &u1, &u2})
      VecDestroy(v);
  }
};

PetscErrorCode
applySchurShell(PC pc, Vec r, Vec z)
{
  SchurShell * s = nullptr;
  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, &s));
  if (s->approximation == SchurApproximation::ConvectionDiffusion)
  {
    // z = A_p^{-1} F_p Q_p^{-1} r
    PetscCall(VecPointwiseMult(s->p1, s->inverse_pressure_mass, r));
    PetscCall(MatMult(s->convection_diffusion, s->p1, s->p2));
    PetscCall(KSPSolve(s->laplacian_solver, s->p2, z));
  }
  else
  {
    // z = A_p^{-1} (G^T Q_v^{-1} F Q_v^{-1} G + C_2) A_p^{-1} r
    PetscCall(KSPSolve(s->laplacian_solver, r, s->p1));
    PetscCall(MatMult(s->gradient, s->p1, s->u1));
    PetscCall(VecPointwiseMult(s->u1, s->u1, s->inverse_velocity_mass));
    PetscCall(MatMult(s->momentum, s->u1, s->u2));
    PetscCall(VecPointwiseMult(s->u2, s->u2, s->inverse_velocity_mass));
    PetscCall(MatMultTranspose(s->gradient, s->u2, s->p2));
    PetscCall(MatMultAdd(s->commutator_stabilization, s->p1, s->p2, s->p2));
    PetscCall(KSPSolve(s->laplacian_solver, s->p2, z));
  }
  PetscCall(VecScale(z, s->sign));
  // A prescribed pressure is a row of the identity, whose Schur complement is 1.
  PetscCall(VecPointwiseMult(z, z, s->free));
  PetscCall(VecPointwiseMult(s->p1, s->prescribed, r));
  PetscCall(VecAXPY(z, 1.0, s->p1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Build the shell of pcd or lsc for the Schur split of @p ksp, set up, whose matrix is @p A.
std::unique_ptr<SchurShell>
makeSchurShell(KSP ksp, Mat A, IS other_set, IS pressure_set, const SchurBlocks & blocks)
{
  auto s = std::make_unique<SchurShell>();
  s->approximation = blocks.approximation;
  s->sign = blocks.sign;
  MPI_Comm comm;
  DM_PETSC(PetscObjectGetComm(reinterpret_cast<PetscObject>(A), &comm));
  DM_PETSC(MatCreateSubMatrix(A, other_set, pressure_set, MAT_INITIAL_MATRIX, &s->gradient));
  Vec lumped = assembledVector(A, blocks.lumped_mass);
  s->inverse_velocity_mass = subvector(lumped, other_set);
  invert(s->inverse_velocity_mass, 0.0);
  s->inverse_pressure_mass = subvector(lumped, pressure_set);
  invert(s->inverse_pressure_mass, 1.0);
  VecDestroy(&lumped);
  // A_p = G^T Q_v^{-1} G + C_1, with ones at the prescribed pressures.
  Mat scaled;
  DM_PETSC(MatDuplicate(s->gradient, MAT_COPY_VALUES, &scaled));
  DM_PETSC(MatDiagonalScale(scaled, s->inverse_velocity_mass, nullptr));
  DM_PETSC(
      MatTransposeMatMult(s->gradient, scaled, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &s->laplacian));
  MatDestroy(&scaled);
  Mat stabilization = restricted(
      assembledMatrix(A, blocks.stabilization, blocks.constrained_pressure_rows), pressure_set);
  DM_PETSC(MatAXPY(s->laplacian, 1.0, stabilization, DIFFERENT_NONZERO_PATTERN));
  MatDestroy(&stabilization);
  if (blocks.approximation == SchurApproximation::ConvectionDiffusion)
    s->convection_diffusion = restricted(
        assembledMatrix(A, blocks.convection_diffusion, blocks.constrained_pressure_rows),
        pressure_set);
  else
  {
    DM_PETSC(MatCreateSubMatrix(A, other_set, other_set, MAT_INITIAL_MATRIX, &s->momentum));
    s->commutator_stabilization =
        restricted(assembledMatrix(A, blocks.commutator_stabilization, {}), pressure_set);
  }
  std::vector<GlobalEntry> ones;
  for (Index r : blocks.constrained_pressure_rows)
    ones.push_back({r, r, 1.0});
  Vec prescribed = assembledVector(A, ones);
  s->prescribed = subvector(prescribed, pressure_set);
  VecDestroy(&prescribed);
  DM_PETSC(VecDuplicate(s->prescribed, &s->free));
  DM_PETSC(VecSet(s->free, 1.0));
  DM_PETSC(VecAXPY(s->free, -1.0, s->prescribed));
  DM_PETSC(MatCreateVecs(s->gradient, &s->p1, &s->u1));
  DM_PETSC(VecDuplicate(s->p1, &s->p2));
  DM_PETSC(VecDuplicate(s->u1, &s->u2));

  DM_PETSC(KSPCreate(comm, &s->laplacian_solver));
  DM_PETSC(KSPSetOptionsPrefix(s->laplacian_solver, "fieldsplit_1_laplacian_"));
  DM_PETSC(KSPSetOperators(s->laplacian_solver, s->laplacian, s->laplacian));
  DM_PETSC(KSPSetFromOptions(s->laplacian_solver));
  DM_PETSC(KSPSetUp(s->laplacian_solver));

  PC pc;
  DM_PETSC(KSPGetPC(ksp, &pc));
  PetscInt count = 0;
  KSP * sub = nullptr;
  DM_PETSC(PCFieldSplitSchurGetSubKSP(pc, &count, &sub));
  KSP schur_solver = sub[1];
  DM_PETSC(PetscFree(sub));
  PC shell;
  DM_PETSC(KSPGetPC(schur_solver, &shell));
  DM_PETSC(PCSetType(shell, PCSHELL));
  DM_PETSC(PCShellSetContext(shell, s.get()));
  DM_PETSC(PCShellSetApply(shell, applySchurShell));
  DM_PETSC(PCShellSetName(shell,
                          blocks.approximation == SchurApproximation::ConvectionDiffusion
                              ? "dualmesh pressure convection-diffusion"
                              : "dualmesh least squares commutator"));
  return s;
}

/// Whether the preconditioner of @p ksp is a field split of the Schur complement type.
bool
isSchurSplit(KSP ksp)
{
  PC pc;
  DM_PETSC(KSPGetPC(ksp, &pc));
  PetscBool fieldsplit = PETSC_FALSE;
  DM_PETSC(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(pc), PCFIELDSPLIT, &fieldsplit));
  if (!fieldsplit)
    return false;
  PCCompositeType type;
  DM_PETSC(PCFieldSplitGetType(pc, &type));
  return type == PC_COMPOSITE_SCHUR;
}

} // namespace

bool
available()
{
  return true;
}

std::string
version()
{
  std::ostringstream s;
  s << PETSC_VERSION_MAJOR << "." << PETSC_VERSION_MINOR << "." << PETSC_VERSION_SUBMINOR;
  return s.str();
}

void
initialize()
{
  PetscBool done = PETSC_FALSE;
  PetscInitialized(&done);
  if (done)
    return;
  // Start MPI the way the rest of the library does, so that PETSc finds it
  // running and leaves its finalization to us.
  (void) Communicator::world();
  DM_PETSC(PetscInitializeNoArguments());
  g_we_initialized = true;
  // atexit handlers run in the reverse order of registration, and the
  // Communicator registered the finalization of MPI earlier, so PETSc is
  // finalized first, as it must be.
  std::atexit(finalize);
}

void
finalize()
{
  PetscBool done = PETSC_FALSE;
  PetscFinalized(&done);
  if (g_we_initialized && !done)
    PetscFinalize();
}

Vector
solve(const SparseMatrix & A_in, const Vector & b, const Settings & settings, Result & result)
{
  initialize();
  const Index n = A_in.rows();
  if (A_in.cols() != n || b.size() != n)
    throw std::invalid_argument("dualmesh: petsc::solve needs a square matrix and a matching "
                                "right-hand side.");
  std::vector<GlobalEntry> entries;
  entries.reserve(static_cast<std::size_t>(A_in.nonZeros()));
  for (Index col = 0; col < A_in.outerSize(); ++col)
    for (SparseMatrix::InnerIterator it(A_in, col); it; ++it)
      entries.push_back({it.row(), it.col(), it.value()});
  const LinearSolver solver(false, n, 0, n, entries, settings);
  return solver.solve(b, nullptr, result);
}

struct LinearSolver::Impl
{
  MPI_Comm comm = PETSC_COMM_SELF;
  Index n_local = 0;
  Mat matrix = nullptr;
  KSP ksp = nullptr;
  Vec b = nullptr, x = nullptr, r = nullptr, w = nullptr;
  double relative_tolerance = 1e-12;
  /// The projected earlier solutions, orthonormal in the energy norm, and their products with the matrix.
  int capacity = 0;
  bool symmetric = true;
  std::size_t num_entries = 0;
  std::vector<Vec> basis, images;
  /// The splits of a field split preconditioner, the Schur complement
  /// preconditioner of pressure_mass, and the shell of pcd and lsc.
  std::vector<IS> splits;
  Mat schur_preconditioner = nullptr;
  std::unique_ptr<SchurShell> shell;
  /// The options of this solver, made the default database again for every solve: the
  /// preconditioners of the blocks are set up at the first solve, and again after setValues, and
  /// read their options then.
  std::string options;

  ~Impl()
  {
    // A solver may outlive PETSc at the exit of a program; there is then nothing left to free.
    if (!PetscInitializeCalled || PetscFinalizeCalled)
    {
      (void) shell.release();
      return;
    }
    for (auto * list : {&basis, &images})
      for (Vec & v : *list)
        VecDestroy(&v);
    VecDestroy(&b);
    VecDestroy(&x);
    VecDestroy(&r);
    VecDestroy(&w);
    KSPDestroy(&ksp);
    shell.reset();
    MatDestroy(&schur_preconditioner);
    for (IS & set : splits)
      ISDestroy(&set);
    MatDestroy(&matrix);
  }

  /// Add the direction @p v (with its product @p av with the matrix) to the basis, orthogonalized against it and normalized in the energy norm; a full basis starts again from @p restart.
  void remember(Vec v, Vec av, Vec restart)
  {
    if (static_cast<int>(basis.size()) == capacity)
    {
      for (auto * list : {&basis, &images})
        for (Vec & u : *list)
          VecDestroy(&u);
      basis.clear();
      images.clear();
      v = restart;
      DM_PETSC(MatMult(matrix, restart, w));
      av = w;
    }
    Vec direction, image;
    DM_PETSC(VecDuplicate(v, &direction));
    DM_PETSC(VecDuplicate(v, &image));
    DM_PETSC(VecCopy(v, direction));
    DM_PETSC(VecCopy(av, image));
    // A symmetric matrix keeps the directions orthonormal in its energy norm, any other keeps their images orthonormal, so that the start minimizes the residual.
    for (std::size_t i = 0; i < basis.size(); ++i)
    {
      PetscScalar c = 0;
      DM_PETSC(VecDot(images[i], symmetric ? direction : image, &c));
      DM_PETSC(VecAXPY(direction, -c, basis[i]));
      DM_PETSC(VecAXPY(image, -c, images[i]));
    }
    PetscScalar energy = 0;
    DM_PETSC(VecDot(symmetric ? direction : image, image, &energy));
    if (!(energy > 1e-300))
    {
      VecDestroy(&direction);
      VecDestroy(&image);
      return;
    }
    DM_PETSC(VecScale(direction, 1.0 / std::sqrt(energy)));
    DM_PETSC(VecScale(image, 1.0 / std::sqrt(energy)));
    basis.push_back(direction);
    images.push_back(image);
  }
};

LinearSolver::LinearSolver(bool distributed,
                           Index global_size,
                           Index first_row,
                           Index num_owned,
                           const std::vector<GlobalEntry> & entries,
                           const Settings & settings)
    : _impl(std::make_unique<Impl>())
{
  initialize();
  Impl & s = *_impl;
  s.comm = distributed ? PETSC_COMM_WORLD : PETSC_COMM_SELF;
  int size = 1;
  MPI_Comm_size(s.comm, &size);
  const Index n_local = num_owned;
  s.n_local = n_local;

  DM_PETSC(MatCreate(s.comm, &s.matrix));
  DM_PETSC(MatSetSizes(s.matrix, n_local, n_local, global_size, global_size));
  DM_PETSC(MatSetType(s.matrix, MATAIJ));
  if (settings.block_size > 1)
    DM_PETSC(MatSetBlockSize(s.matrix, settings.block_size));
  // Coordinate (COO) assembly: every process lists the entries it computed,
  // wherever their rows live, and PETSc sends them to the owners and sums
  // repeated positions.  This is exactly the element-by-element sum of the
  // partitioned assembly.
  // An explicit zero on the diagonal of every owned row, which PETSc's
  // factorisations and field splits need in the sparsity pattern (see solve).
  const std::size_t num_entries = entries.size() + static_cast<std::size_t>(n_local);
  std::vector<PetscInt> rows(num_entries), cols(num_entries);
  std::vector<PetscScalar> values(num_entries);
  for (std::size_t k = 0; k < entries.size(); ++k)
  {
    rows[k] = static_cast<PetscInt>(entries[k].row);
    cols[k] = static_cast<PetscInt>(entries[k].col);
    values[k] = entries[k].value;
  }
  for (Index i = 0; i < n_local; ++i)
  {
    const std::size_t k = entries.size() + static_cast<std::size_t>(i);
    rows[k] = cols[k] = static_cast<PetscInt>(first_row + i);
    values[k] = 0.0;
  }
  DM_PETSC(MatSetPreallocationCOO(
      s.matrix, static_cast<PetscCount>(num_entries), rows.data(), cols.data()));
  DM_PETSC(MatSetValuesCOO(s.matrix, values.data(), ADD_VALUES));
  s.num_entries = entries.size();
  DM_PETSC(MatCreateVecs(s.matrix, &s.x, &s.b));
  DM_PETSC(VecDuplicate(s.b, &s.r));
  DM_PETSC(VecDuplicate(s.b, &s.w));
  s.relative_tolerance = settings.relative_tolerance;
  s.capacity = std::max(0, settings.projection_vectors);
  s.symmetric = settings.symmetric;
  PetscInt lo = 0, hi = 0;
  DM_PETSC(VecGetOwnershipRange(s.b, &lo, &hi));
  if (static_cast<Index>(lo) != first_row || static_cast<Index>(hi - lo) != n_local)
    throw std::logic_error("dualmesh: PETSc chose a different row distribution.");

  if (settings.schur && !settings.physics_blocks.empty())
    throw std::logic_error("dualmesh: a Schur complement split and a split by physics at once.");
  if (settings.near_nullspace.cols() > 0)
  {
    if (settings.near_nullspace.rows() != n_local)
      throw std::invalid_argument("dualmesh: the near null space needs one row per owned row.");
    MatNullSpace space = nearNullSpace(s.matrix, settings.near_nullspace);
    DM_PETSC(MatSetNearNullSpace(s.matrix, space));
    MatNullSpaceDestroy(&space);
  }

  // The options are read, and the preconditioner is built, while this solver's options are the
  // default database.  The options of a block preconditioner refine its defaults (the later
  // value of an option wins); otherwise they replace the default.
  std::string text = settings.options.empty() ? defaultOptions(size > 1) : settings.options;
  if (settings.schur)
    text = schurOptions(settings.schur->approximation) + " " + settings.options;
  else if (!settings.physics_blocks.empty())
    text = physicsBlockOptions(settings.physics_blocks.size()) + " " + settings.options;
  s.options = text;
  ScopedOptions scope(text);
  DM_PETSC(KSPCreate(s.comm, &s.ksp));
  DM_PETSC(KSPSetOperators(s.ksp, s.matrix, s.matrix));
  DM_PETSC(KSPSetTolerances(s.ksp,
                            settings.relative_tolerance,
                            settings.absolute_tolerance,
                            PETSC_DEFAULT,
                            static_cast<PetscInt>(settings.max_iterations)));
  PC pc;
  DM_PETSC(KSPGetPC(s.ksp, &pc));
  if (settings.schur)
  {
    DM_PETSC(PCSetType(pc, PCFIELDSPLIT));
    s.splits = {indexSet(s.comm, settings.schur->other_rows),
                indexSet(s.comm, settings.schur->pressure_rows)};
    DM_PETSC(PCFieldSplitSetIS(pc, "0", s.splits[0]));
    DM_PETSC(PCFieldSplitSetIS(pc, "1", s.splits[1]));
  }
  for (std::size_t k = 0; k < settings.physics_blocks.size(); ++k)
  {
    if (k == 0)
      DM_PETSC(PCSetType(pc, PCFIELDSPLIT));
    IS set = indexSet(s.comm, settings.physics_blocks[k]);
    s.splits.push_back(set);
    // The field split gives the near null space composed with a block's rows to its matrix.
    if (settings.near_nullspace.cols() > 0)
    {
      MatNullSpace space = nearNullSpace(s.matrix, settings.near_nullspace, set);
      DM_PETSC(PetscObjectCompose(reinterpret_cast<PetscObject>(set),
                                  "nearnullspace",
                                  reinterpret_cast<PetscObject>(space)));
      MatNullSpaceDestroy(&space);
    }
    DM_PETSC(PCFieldSplitSetIS(pc, std::to_string(k).c_str(), set));
  }
  DM_PETSC(KSPSetFromOptions(s.ksp));
  const bool schur_split = settings.schur && isSchurSplit(s.ksp);
  if (schur_split && settings.schur->approximation == SchurApproximation::PressureMass)
  {
    // sign (1/mu) M_p plus the symmetric part of the pressure block of the matrix, whose
    // prescribed rows are rows of the identity already.
    Mat block;
    DM_PETSC(MatCreateSubMatrix(s.matrix, s.splits[1], s.splits[1], MAT_INITIAL_MATRIX, &block));
    DM_PETSC(MatTranspose(block, MAT_INITIAL_MATRIX, &s.schur_preconditioner));
    DM_PETSC(MatAXPY(s.schur_preconditioner, 1.0, block, DIFFERENT_NONZERO_PATTERN));
    DM_PETSC(MatScale(s.schur_preconditioner, 0.5));
    MatDestroy(&block);
    Mat mass = restricted(assembledMatrix(s.matrix, settings.schur->mass, {}), s.splits[1]);
    DM_PETSC(
        MatAXPY(s.schur_preconditioner, settings.schur->sign, mass, DIFFERENT_NONZERO_PATTERN));
    MatDestroy(&mass);
    DM_PETSC(PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_USER, s.schur_preconditioner));
  }
  DM_PETSC(KSPSetUp(s.ksp));
  if (schur_split && settings.schur->approximation != SchurApproximation::PressureMass)
    s.shell = makeSchurShell(s.ksp, s.matrix, s.splits[0], s.splits[1], *settings.schur);
}

LinearSolver::~LinearSolver() = default;

void
LinearSolver::setValues(const std::vector<double> & values)
{
  Impl & s = *_impl;
  if (values.size() != s.num_entries)
    throw std::invalid_argument("dualmesh: new values for a different number of entries.");
  if (s.shell || s.schur_preconditioner)
    throw std::logic_error("dualmesh: the operators of a Schur complement split belong to the "
                           "first values of the matrix; make a new solver.");
  // The zeros on the diagonal of every owned row follow the entries, as in the constructor.
  std::vector<PetscScalar> all(values.begin(), values.end());
  all.resize(values.size() + static_cast<std::size_t>(s.n_local), 0.0);
  DM_PETSC(MatSetValuesCOO(s.matrix, all.data(), INSERT_VALUES));
  // The same operator object with new values: the preconditioner is rebuilt at the next solve.
  DM_PETSC(KSPSetOperators(s.ksp, s.matrix, s.matrix));
  // The projected solutions belong to the old matrix.
  for (auto * list : {&s.basis, &s.images})
    for (Vec & u : *list)
      VecDestroy(&u);
  s.basis.clear();
  s.images.clear();
}

Vector
LinearSolver::solve(const Vector & b_owned, const Vector * guess, Result & result) const
{
  Impl & s = *_impl;
  if (b_owned.size() != s.n_local || (guess && guess->size() != s.n_local))
    throw std::invalid_argument("dualmesh: a right-hand side of the wrong size.");
  const ScopedOptions scope(s.options);
  PetscScalar * data = nullptr;
  DM_PETSC(VecGetArray(s.b, &data));
  std::copy(b_owned.data(), b_owned.data() + s.n_local, data);
  DM_PETSC(VecRestoreArray(s.b, &data));
  bool solved = true;
  if (s.capacity > 0)
  {
    // Start from the combination of the earlier solutions closest to the solution, xbar = sum_i c_i q_i, in the energy norm of a symmetric matrix (c_i = q_i . b) or with the smallest residual (c_i = A q_i . b), and solve A dx = b - A xbar to the accuracy asked of the whole solve.
    PetscReal b_norm = 0;
    DM_PETSC(VecNorm(s.b, NORM_2, &b_norm));
    Vec start;
    DM_PETSC(VecDuplicate(s.b, &start));
    DM_PETSC(VecSet(start, 0.0));
    DM_PETSC(VecCopy(s.b, s.r));
    for (std::size_t i = 0; i < s.basis.size(); ++i)
    {
      PetscScalar c = 0;
      DM_PETSC(VecDot(s.symmetric ? s.basis[i] : s.images[i], s.b, &c));
      DM_PETSC(VecAXPY(start, c, s.basis[i]));
      DM_PETSC(VecAXPY(s.r, -c, s.images[i]));
    }
    PetscReal r_norm = 0;
    DM_PETSC(VecNorm(s.r, NORM_2, &r_norm));
    const double tolerance =
        r_norm > 0 ? std::min(0.5, s.relative_tolerance * b_norm / r_norm) : 0.5;
    DM_PETSC(KSPSetTolerances(s.ksp, tolerance, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    DM_PETSC(VecSet(s.x, 0.0));
    DM_PETSC(KSPSetInitialGuessNonzero(s.ksp, PETSC_FALSE));
    solved = r_norm > 0;
    if (solved)
      DM_PETSC(KSPSolve(s.ksp, s.r, s.x));
    // x holds the remainder dx: remember it, and form the solution.
    Vec image;
    DM_PETSC(VecDuplicate(s.b, &image));
    DM_PETSC(MatMult(s.matrix, s.x, image));
    DM_PETSC(VecAXPY(start, 1.0, s.x));
    if (r_norm > 0)
      s.remember(s.x, image, start);
    DM_PETSC(VecCopy(start, s.x));
    VecDestroy(&image);
    VecDestroy(&start);
  }
  else
  {
    DM_PETSC(VecGetArray(s.x, &data));
    if (guess)
      std::copy(guess->data(), guess->data() + s.n_local, data);
    else
      std::fill(data, data + s.n_local, 0.0);
    DM_PETSC(VecRestoreArray(s.x, &data));
    DM_PETSC(KSPSetInitialGuessNonzero(s.ksp, guess ? PETSC_TRUE : PETSC_FALSE));
    DM_PETSC(KSPSolve(s.ksp, s.b, s.x));
  }
  PetscInt its = 0;
  KSPConvergedReason reason;
  DM_PETSC(KSPGetIterationNumber(s.ksp, &its));
  DM_PETSC(KSPGetConvergedReason(s.ksp, &reason));
  const char * text = nullptr;
  DM_PETSC(KSPGetConvergedReasonString(s.ksp, &text));
  result.iterations = solved ? static_cast<int>(its) : 0;
  result.converged = !solved || reason > 0;
  result.reason = solved ? (text ? text : "") : "the projection of the earlier solutions is exact";
  checkTrueResidual(s.matrix, s.b, s.x, s.relative_tolerance, result);
  Vector x(s.n_local);
  const PetscScalar * xs = nullptr;
  DM_PETSC(VecGetArrayRead(s.x, &xs));
  std::copy(xs, xs + s.n_local, x.data());
  DM_PETSC(VecRestoreArrayRead(s.x, &xs));
  return x;
}

Vector
solveDistributed(Index global_size,
                 Index first_row,
                 const std::vector<GlobalEntry> & entries,
                 const Vector & b_owned,
                 const Settings & settings,
                 Result & result)
{
  initialize();
  const LinearSolver solver(true, global_size, first_row, b_owned.size(), entries, settings);
  return solver.solve(b_owned, nullptr, result);
}

#endif

} // namespace petsc
} // namespace dualmesh
