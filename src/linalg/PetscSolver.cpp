// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/linalg/PetscSolver.h"

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

  ~Impl()
  {
    // A solver may outlive PETSc at the exit of a program; there is then nothing left to free.
    if (!PetscInitializeCalled || PetscFinalizeCalled)
      return;
    for (auto * list : {&basis, &images})
      for (Vec & v : *list)
        VecDestroy(&v);
    VecDestroy(&b);
    VecDestroy(&x);
    VecDestroy(&r);
    VecDestroy(&w);
    KSPDestroy(&ksp);
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

  // The options are read, and the preconditioner is built, while this solver's options are the default database.
  ScopedOptions scope(settings.options.empty() ? defaultOptions(size > 1) : settings.options);
  DM_PETSC(KSPCreate(s.comm, &s.ksp));
  DM_PETSC(KSPSetOperators(s.ksp, s.matrix, s.matrix));
  DM_PETSC(KSPSetTolerances(s.ksp,
                            settings.relative_tolerance,
                            settings.absolute_tolerance,
                            PETSC_DEFAULT,
                            static_cast<PetscInt>(settings.max_iterations)));
  DM_PETSC(KSPSetFromOptions(s.ksp));
  DM_PETSC(KSPSetUp(s.ksp));
}

LinearSolver::~LinearSolver() = default;

void
LinearSolver::setValues(const std::vector<double> & values)
{
  Impl & s = *_impl;
  if (values.size() != s.num_entries)
    throw std::invalid_argument("dualmesh: new values for a different number of entries.");
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
