// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/linalg/PetscSolver.h"

#include <stdexcept>

#ifdef DUALMESH_HAVE_PETSC
#include "dualmesh/parallel/Communicator.h"

#include <petscksp.h>

#include <algorithm>
#include <cstdlib>
#include <sstream>
#endif

namespace dualmesh
{
namespace petsc
{

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

namespace
{
[[noreturn]] void
missing()
{
  throw std::runtime_error(
      "dualmesh: linear_solver='petsc' needs a build with PETSc. Install PETSc (for example "
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

#else

namespace
{
bool g_we_initialized = false;

void
check(PetscErrorCode code, const char * what)
{
  if (code != 0)
  {
    const char * text = nullptr;
    PetscErrorMessage(code, &text, nullptr);
    throw std::runtime_error(std::string("dualmesh: PETSc failed in ") + what + ": " +
                             (text ? text : "unknown error"));
  }
}

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

/// The field split options of the pressure mass matrix Schur preconditioner.
std::string
saddlePointOptions()
{
  std::string solver = "";
#ifdef PETSC_HAVE_MUMPS
  solver = " -fieldsplit_0_pc_factor_mat_solver_type mumps";
#endif
  return "-ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_type schur "
         "-pc_fieldsplit_schur_fact_type upper -fieldsplit_0_ksp_type preonly "
         "-fieldsplit_0_pc_type lu" +
         solver + " -fieldsplit_1_ksp_type preonly -fieldsplit_1_pc_type lu";
}

/// Convert a sequential Eigen matrix to a PETSc SeqAIJ matrix.
Mat
toSequentialPetsc(const SparseMatrix & A_in)
{
  Eigen::SparseMatrix<double, Eigen::RowMajor> A = A_in;
  A.makeCompressed();
  const Index n = A.rows();
  Mat M;
  DM_PETSC(MatCreate(PETSC_COMM_SELF, &M));
  DM_PETSC(MatSetSizes(M, n, A.cols(), n, A.cols()));
  DM_PETSC(MatSetType(M, MATSEQAIJ));
  std::vector<PetscInt> nnz(n);
  for (Index i = 0; i < n; ++i)
    nnz[i] = static_cast<PetscInt>(A.outerIndexPtr()[i + 1] - A.outerIndexPtr()[i]);
  DM_PETSC(MatSeqAIJSetPreallocation(M, 0, nnz.data()));
  std::vector<PetscInt> cols;
  for (Index i = 0; i < n; ++i)
  {
    const auto begin = A.outerIndexPtr()[i];
    const auto end = A.outerIndexPtr()[i + 1];
    cols.assign(A.innerIndexPtr() + begin, A.innerIndexPtr() + end);
    const PetscInt row = static_cast<PetscInt>(i);
    DM_PETSC(MatSetValues(M,
                          1,
                          &row,
                          static_cast<PetscInt>(end - begin),
                          cols.data(),
                          A.valuePtr() + begin,
                          INSERT_VALUES));
  }
  DM_PETSC(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
  DM_PETSC(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));
  return M;
}

/// Solve with a configured matrix and right-hand side, and destroy the solver.
void
runKsp(MPI_Comm comm, Mat A, Vec b, Vec x, const Settings & s, bool parallel, Result & result)
{
  const SaddlePointBlocks * saddle = s.saddle_point;
  if (saddle && parallel)
    throw std::runtime_error("dualmesh: the pressure mass matrix Schur preconditioner is "
                             "available for serial solves only so far.");
  const std::string base = saddle ? saddlePointOptions() : defaultOptions(parallel);
  // With the saddle point preconditioner the user's options refine the
  // defaults (the later value of an option wins); otherwise they replace them.
  ScopedOptions scope(saddle ? base + " " + s.options : (s.options.empty() ? base : s.options));
  KSP ksp;
  DM_PETSC(KSPCreate(comm, &ksp));
  DM_PETSC(KSPSetOperators(ksp, A, A));
  DM_PETSC(KSPSetTolerances(ksp,
                            s.relative_tolerance,
                            s.absolute_tolerance,
                            PETSC_DEFAULT,
                            static_cast<PetscInt>(s.max_iterations)));
  IS other_set = nullptr, pressure_set = nullptr;
  Mat schur = nullptr;
  if (saddle)
  {
    PC pc;
    DM_PETSC(KSPGetPC(ksp, &pc));
    DM_PETSC(PCSetType(pc, PCFIELDSPLIT));
    std::vector<PetscInt> other(saddle->other_dofs.begin(), saddle->other_dofs.end());
    std::vector<PetscInt> pressure(saddle->pressure_dofs.begin(), saddle->pressure_dofs.end());
    DM_PETSC(ISCreateGeneral(
        comm, static_cast<PetscInt>(other.size()), other.data(), PETSC_COPY_VALUES, &other_set));
    DM_PETSC(ISCreateGeneral(comm,
                             static_cast<PetscInt>(pressure.size()),
                             pressure.data(),
                             PETSC_COPY_VALUES,
                             &pressure_set));
    DM_PETSC(PCFieldSplitSetIS(pc, "0", other_set));
    DM_PETSC(PCFieldSplitSetIS(pc, "1", pressure_set));
    schur = toSequentialPetsc(saddle->schur_approximation);
  }
  DM_PETSC(KSPSetFromOptions(ksp));
  if (saddle)
  {
    PC pc;
    DM_PETSC(KSPGetPC(ksp, &pc));
    DM_PETSC(PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_USER, schur));
  }
  DM_PETSC(KSPSolve(ksp, b, x));
  PetscInt its = 0;
  KSPConvergedReason reason;
  DM_PETSC(KSPGetIterationNumber(ksp, &its));
  DM_PETSC(KSPGetConvergedReason(ksp, &reason));
  const char * text = nullptr;
  DM_PETSC(KSPGetConvergedReasonString(ksp, &text));
  result.iterations = static_cast<int>(its);
  result.converged = reason > 0;
  result.reason = text ? text : "";
  DM_PETSC(KSPDestroy(&ksp));
  if (schur)
    MatDestroy(&schur);
  if (other_set)
    ISDestroy(&other_set);
  if (pressure_set)
    ISDestroy(&pressure_set);
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
  Eigen::SparseMatrix<double, Eigen::RowMajor> A = A_in;
  A.makeCompressed();

  Mat M;
  DM_PETSC(MatCreate(PETSC_COMM_SELF, &M));
  DM_PETSC(MatSetSizes(M, n, n, n, n));
  DM_PETSC(MatSetType(M, MATSEQAIJ));
  if (settings.block_size > 1 && n % settings.block_size == 0)
    DM_PETSC(MatSetBlockSize(M, settings.block_size));
  // Every diagonal entry is stored, as an explicit zero where the matrix has
  // none: PETSc's factorisations and field splits need the diagonal in the
  // sparsity pattern, and a saddle point system (the pressure rows of the
  // Taylor-Hood element) has no diagonal of its own.
  std::vector<PetscInt> nnz(n);
  std::vector<char> has_diagonal(n, 0);
  for (Index i = 0; i < n; ++i)
  {
    const auto begin = A.outerIndexPtr()[i];
    const auto end = A.outerIndexPtr()[i + 1];
    for (auto k = begin; k < end; ++k)
      if (A.innerIndexPtr()[k] == i)
        has_diagonal[i] = 1;
    nnz[i] = static_cast<PetscInt>(end - begin) + (has_diagonal[i] ? 0 : 1);
  }
  DM_PETSC(MatSeqAIJSetPreallocation(M, 0, nnz.data()));
  for (Index i = 0; i < n; ++i)
    if (!has_diagonal[i])
    {
      const PetscInt row = static_cast<PetscInt>(i);
      const PetscScalar zero = 0.0;
      DM_PETSC(MatSetValues(M, 1, &row, 1, &row, &zero, INSERT_VALUES));
    }
  std::vector<PetscInt> cols;
  for (Index i = 0; i < n; ++i)
  {
    const auto begin = A.outerIndexPtr()[i];
    const auto end = A.outerIndexPtr()[i + 1];
    cols.assign(A.innerIndexPtr() + begin, A.innerIndexPtr() + end);
    const PetscInt row = static_cast<PetscInt>(i);
    DM_PETSC(MatSetValues(M,
                          1,
                          &row,
                          static_cast<PetscInt>(end - begin),
                          cols.data(),
                          A.valuePtr() + begin,
                          INSERT_VALUES));
  }
  DM_PETSC(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
  DM_PETSC(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));

  Vec vb, vx;
  DM_PETSC(MatCreateVecs(M, &vx, &vb));
  PetscScalar * data = nullptr;
  DM_PETSC(VecGetArray(vb, &data));
  std::copy(b.data(), b.data() + n, data);
  DM_PETSC(VecRestoreArray(vb, &data));

  runKsp(PETSC_COMM_SELF, M, vb, vx, settings, false, result);

  Vector x(n);
  const PetscScalar * xs = nullptr;
  DM_PETSC(VecGetArrayRead(vx, &xs));
  std::copy(xs, xs + n, x.data());
  DM_PETSC(VecRestoreArrayRead(vx, &xs));
  VecDestroy(&vb);
  VecDestroy(&vx);
  MatDestroy(&M);
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
  const Index n_local = b_owned.size();

  Mat M;
  DM_PETSC(MatCreate(PETSC_COMM_WORLD, &M));
  DM_PETSC(MatSetSizes(M, n_local, n_local, global_size, global_size));
  DM_PETSC(MatSetType(M, MATAIJ));
  if (settings.block_size > 1)
    DM_PETSC(MatSetBlockSize(M, settings.block_size));
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
  DM_PETSC(
      MatSetPreallocationCOO(M, static_cast<PetscCount>(num_entries), rows.data(), cols.data()));
  DM_PETSC(MatSetValuesCOO(M, values.data(), ADD_VALUES));

  Vec vb, vx;
  DM_PETSC(MatCreateVecs(M, &vx, &vb));
  PetscInt lo = 0, hi = 0;
  DM_PETSC(VecGetOwnershipRange(vb, &lo, &hi));
  if (static_cast<Index>(lo) != first_row || static_cast<Index>(hi - lo) != n_local)
    throw std::logic_error("dualmesh: PETSc chose a different row distribution.");
  PetscScalar * data = nullptr;
  DM_PETSC(VecGetArray(vb, &data));
  std::copy(b_owned.data(), b_owned.data() + n_local, data);
  DM_PETSC(VecRestoreArray(vb, &data));

  int size = 1;
  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  runKsp(PETSC_COMM_WORLD, M, vb, vx, settings, size > 1, result);

  Vector x(n_local);
  const PetscScalar * xs = nullptr;
  DM_PETSC(VecGetArrayRead(vx, &xs));
  std::copy(xs, xs + n_local, x.data());
  DM_PETSC(VecRestoreArrayRead(vx, &xs));
  VecDestroy(&vb);
  VecDestroy(&vx);
  MatDestroy(&M);
  return x;
}

#endif

} // namespace petsc
} // namespace dualmesh
