// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The incomplete LU factorisation with no fill, ILU(0), as a preconditioner
// for Eigen's Krylov solvers.
//
// ILU(0) computes factors L (unit lower triangular) and U (upper triangular)
// that have exactly the sparsity pattern of A and whose product matches A on
// that pattern: (LU)_{ij} = A_{ij} wherever A_{ij} is stored (Saad, Iterative
// Methods for Sparse Linear Systems, 2nd ed., SIAM 2003, section 10.3.2).
// The entries that an exact factorisation would create outside the pattern,
// the fill, are dropped.  The factorisation therefore costs about as much as
// a few matrix-vector products and needs no more memory than A, which is why
// it is the default preconditioner for systems too large to factorise
// exactly.  Eigen provides the threshold variant ILUT but not ILU(0); ILUT
// with a large fill factor approaches a complete factorisation and loses the
// point of an incomplete one on three-dimensional meshes.
#pragma once

#include <Eigen/Sparse>

#include <vector>

namespace dualmesh
{

class IncompleteLU0
{
public:
  using Scalar = double;
  using StorageIndex = int;
  using Vector = Eigen::VectorXd;
  enum
  {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic
  };

  IncompleteLU0() = default;
  template <typename MatrixType> explicit IncompleteLU0(const MatrixType & A) { compute(A); }

  /// Factorise @p A (any Eigen sparse matrix; it is copied to row-major).
  template <typename MatrixType> IncompleteLU0 & compute(const MatrixType & A)
  {
    factorizeRowMajor(Eigen::SparseMatrix<double, Eigen::RowMajor, int>(A));
    return *this;
  }
  template <typename MatrixType> IncompleteLU0 & analyzePattern(const MatrixType &)
  {
    return *this;
  }
  template <typename MatrixType> IncompleteLU0 & factorize(const MatrixType & A)
  {
    return compute(A);
  }

  /// z = (LU)^{-1} r, by one forward and one backward substitution.
  template <typename Rhs> Vector solve(const Eigen::MatrixBase<Rhs> & r) const
  {
    Vector z = r;
    apply(z);
    return z;
  }
  /// In place: z <- (LU)^{-1} z.
  void apply(Vector & z) const;

  Eigen::ComputationInfo info() const { return _info; }
  Eigen::Index rows() const { return _n; }
  Eigen::Index cols() const { return _n; }
  /// The number of pivots that were too small and were replaced; a nonzero
  /// count means the preconditioner is a poorer approximation than usual.
  int replacedPivots() const { return _replaced; }

private:
  void factorizeRowMajor(Eigen::SparseMatrix<double, Eigen::RowMajor, int> A);

  Eigen::Index _n = 0;
  std::vector<int> _ptr, _col, _diag;
  std::vector<double> _val;
  int _replaced = 0;
  Eigen::ComputationInfo _info = Eigen::Success;
};

} // namespace dualmesh
