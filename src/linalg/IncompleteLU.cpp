// SPDX-License-Identifier: LGPL-2.1-or-later
#include "dualmesh/linalg/IncompleteLU.h"

#include <cmath>

namespace dualmesh
{

void
IncompleteLU0::factorizeRowMajor(Eigen::SparseMatrix<double, Eigen::RowMajor, int> A)
{
  A.makeCompressed();
  _n = A.rows();
  const int n = static_cast<int>(_n);
  _ptr.assign(A.outerIndexPtr(), A.outerIndexPtr() + n + 1);
  _col.assign(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros());
  _val.assign(A.valuePtr(), A.valuePtr() + A.nonZeros());
  _diag.assign(n, -1);
  _replaced = 0;
  _info = Eigen::Success;

  // A pivot is replaced when it is this small relative to its row, so that
  // a singular or nearly singular leading block does not stop the solver;
  // the preconditioner is then only an approximation, which a Krylov method
  // tolerates.
  double largest = 0;
  for (double v : _val)
    largest = std::max(largest, std::abs(v));
  const double tiny = 1e-14 * (largest > 0 ? largest : 1.0);

  // Row i is eliminated with the rows k < i already factorised (the IKJ
  // ordering of Saad's section 10.3).  position[j] is the index of entry
  // (i, j) in the storage of row i, or -1 when (i, j) is outside the pattern.
  std::vector<int> position(n, -1);
  for (int i = 0; i < n; ++i)
  {
    for (int p = _ptr[i]; p < _ptr[i + 1]; ++p)
    {
      position[_col[p]] = p;
      if (_col[p] == i)
        _diag[i] = p;
    }
    for (int p = _ptr[i]; p < _ptr[i + 1] && _col[p] < i; ++p)
    {
      const int k = _col[p];
      const double pivot = _val[_diag[k]];
      const double lik = _val[p] / pivot;
      _val[p] = lik;
      for (int q = _diag[k] + 1; q < _ptr[k + 1]; ++q)
      {
        const int at = position[_col[q]];
        if (at >= 0)
          _val[at] -= lik * _val[q];
      }
    }
    if (_diag[i] < 0)
    {
      // No stored diagonal: the factorisation cannot proceed on the pattern.
      _info = Eigen::NumericalIssue;
      for (int p = _ptr[i]; p < _ptr[i + 1]; ++p)
        position[_col[p]] = -1;
      return;
    }
    double & d = _val[_diag[i]];
    if (std::abs(d) < tiny)
    {
      d = d < 0 ? -tiny : tiny;
      ++_replaced;
    }
    for (int p = _ptr[i]; p < _ptr[i + 1]; ++p)
      position[_col[p]] = -1;
  }
}

void
IncompleteLU0::apply(Vector & z) const
{
  const int n = static_cast<int>(_n);
  // L y = r, L unit lower triangular.
  for (int i = 0; i < n; ++i)
  {
    double s = z[i];
    for (int p = _ptr[i]; p < _diag[i]; ++p)
      s -= _val[p] * z[_col[p]];
    z[i] = s;
  }
  // U z = y.
  for (int i = n - 1; i >= 0; --i)
  {
    double s = z[i];
    for (int p = _diag[i] + 1; p < _ptr[i + 1]; ++p)
      s -= _val[p] * z[_col[p]];
    z[i] = s / _val[_diag[i]];
  }
}

} // namespace dualmesh
