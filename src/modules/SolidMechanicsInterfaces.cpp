// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Solid mechanics module, continuum: the conditions that join two surfaces or hold one by a
// constraint, mortar contact with Coulomb friction, the cohesive zone, and the prescribed normal
// displacement.  Each assembles its own equations (IntegratedBC::assembleConstraint).
#include "dualmesh/modules/SolidMechanics.h"

#include "dualmesh/base/Checkpoint.h"
#include "dualmesh/base/FieldExpression.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Problem.h"
#include "dualmesh/core/Coordinates.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace dualmesh::solid_mechanics
{

namespace
{

/// The sides of the surfaces 'names', each as its two nodes, for the mortar constraints. With
/// 'orient', the nodes of a side are ordered so that (t_y, -t_x) is its outward normal. With
/// 'sides', the sides themselves, in the same order.
std::vector<std::array<Index, 2>>
mortarEdges(const Mesh & mesh,
            const std::vector<std::string> & names,
            bool orient,
            const std::string & who,
            std::vector<Side> * sides = nullptr)
{
  std::vector<std::array<Index, 2>> out;
  for (const auto & boundary : names)
    for (const Side & side : mesh.sideset(boundary))
    {
      const auto nodes = mesh.sideNodes(side);
      if (nodes.size() != 2)
        throw InputError("'" + who +
                         "': a mortar surface takes linear elements (two nodes per side).");
      std::array<Index, 2> e{nodes[0], nodes[1]};
      if (orient)
      {
        const Point a = mesh.node(e[0]), b = mesh.node(e[1]);
        const Point c = mesh.elementCentroid(side.first);
        const double nx = b[1] - a[1], ny = -(b[0] - a[0]);
        if (nx * (0.5 * (a[0] + b[0]) - c[0]) + ny * (0.5 * (a[1] + b[1]) - c[1]) < 0.0)
          std::swap(e[0], e[1]);
      }
      out.push_back(e);
      if (sides)
        sides->push_back(side);
    }
  return out;
}

/// Contact between two surfaces of a two-dimensional problem by the mortar method, with
/// Lagrange multipliers: the contact pressure (and, with friction, the tangential traction) at
/// the nodes of the secondary surface.  The weighted gap of node j, g_j = int N_j (x_p - x_s) . n
/// over the secondary surface with the primary point x_p projected along the secondary normal n,
/// and the pressure satisfy g >= 0, lambda >= 0, lambda g = 0, written as the complementarity
/// function lambda - max(0, lambda - c g_j / D_j) = 0 (D_j = int N_j), solved by Newton's
/// method with the active set.  Coulomb friction is the complementarity function of Hueber,
/// Stadler and Wohlmuth on the weighted slip.  The scale c changes the convergence only, not
/// the solution.  The pressure acts on both surfaces, -lambda n on the secondary and +lambda n
/// on the primary, so the forces balance exactly.
class MortarContact : public IntegratedBC
{
public:
  static InputParameters validParams()
  {
    InputParameters p = IntegratedBC::validParams();
    p.setClassDescription(
        "Mortar contact between two surfaces of a two-dimensional problem, plane or "
        "axisymmetric, with linear elements on both surfaces. 'variable' is the contact pressure "
        "p_c (Pa), a Lagrange multiplier on the nodes of the secondary surface 'boundary', and "
        "'primary_boundary' is the other surface. At each secondary node the weighted gap g (m) "
        "and the pressure satisfy g >= 0, p_c >= 0 and p_c g = 0, which Newton's method solves "
        "with an active set. So the surfaces do not penetrate in the weighted sense of the "
        "mortar method, and no penalty is needed. With a friction_coefficient, 'tangential_traction' "
        "adds Coulomb friction. The pressure acts on both surfaces with opposite signs, so the "
        "contact forces balance.");
    p.addRequired("primary_boundary", ParameterKind::String, "The primary surface.");
    addDisplacementsParam(p, "The displacements of both bodies.");
    p.addOptional("tangential_traction",
                  ParameterKind::String,
                  std::string(""),
                  "The variable of the tangential traction (Pa) of friction, a second Lagrange "
                  "multiplier on the secondary nodes. Required with a friction_coefficient. "
                  "Default none: frictionless contact.");
    p.addOptional("friction_coefficient",
                  ParameterKind::Real,
                  0.0,
                  "The Coulomb friction coefficient (dimensionless). Default 0: frictionless.");
    p.setRange("friction_coefficient", ParameterRange::NonNegative);
    p.addRequired("stiffness_scale",
                  ParameterKind::Real,
                  "The scale c (Pa/m) of the complementarity functions, of the size of Young's "
                  "modulus over the element size. It sets the convergence of Newton's method and "
                  "leaves the solution unchanged.");
    p.setRange("stiffness_scale", ParameterRange::Positive);
    p.addOptional("thickness",
                  ParameterKind::Real,
                  1.0,
                  "The out-of-plane thickness (m) of a plane problem, as stress_divergence takes "
                  "it. Default 1, the default of stress_divergence.");
    p.setRange("thickness", ParameterRange::Positive);
    p.addOptional("large_deformation",
                  ParameterKind::Boolean,
                  false,
                  "Measure and orient the contact on the deformed surfaces, for bodies at finite "
                  "strain. Default false: the measure and the normal of the undeformed surface, "
                  "as a small-strain solid measures its forces, with the points projected at "
                  "their current positions, so that the surfaces may slide.");
    return p;
  }
  explicit MortarContact(const InputParameters & p)
      : IntegratedBC(p), _mu(p.getReal("friction_coefficient")), _c(p.getReal("stiffness_scale")),
        _thickness(p.getReal("thickness")), _large(p.getBool("large_deformation"))
  {
    if (_mu > 0.0 && p.getString("tangential_traction").empty())
      throw InputError("'" + name() + "': friction needs the variable 'tangential_traction'.");
  }
  bool isConstraint() const override { return true; }
  ADReal computeBoundaryFlux(const QpContext &) const override { return ADReal(0.0); }
  std::vector<std::string> constraintBoundaries() const override
  {
    std::vector<std::string> names = _boundaries;
    names.push_back(_params.getString("primary_boundary"));
    return names;
  }

  void initialSetup(Problem & problem) override
  {
    IntegratedBC::initialSetup(problem);
    const Mesh & mesh = problem.mesh();
    if (mesh.spaceDimension() != 2)
      throw InputError("'" + name() +
                       "': mortar contact is available in two dimensions "
                       "only.");
    _coord = problem.coordinateSystem();
    _disp.clear();
    for (const auto & d : _params.getStringList("displacements"))
      _disp.push_back(problem.variableIndex(d));
    _tau = _params.getString("tangential_traction").empty()
               ? -1
               : problem.variableIndex(_params.getString("tangential_traction"));
    _secondary = mortarEdges(mesh, _boundaries, true, name());
    _primary = mortarEdges(mesh, {_params.getString("primary_boundary")}, false, name());
    std::set<Index> on;
    for (const auto & e : _secondary)
      on.insert(e.begin(), e.end());
    _secondary_nodes.assign(on.begin(), on.end());
    _multipliers.clear();
    for (Index n : _secondary_nodes)
    {
      _multipliers.push_back(problem.dof(n, _var));
      if (_tau >= 0)
        _multipliers.push_back(problem.dof(n, _tau));
    }
  }
  std::vector<Index> multiplierDofs(const Problem &) const override { return _multipliers; }

  void assembleConstraint(const Problem & problem,
                          const Eigen::VectorXd & U,
                          const Eigen::VectorXd * old,
                          Eigen::VectorXd & R,
                          Eigen::VectorXd * reference,
                          std::vector<Eigen::Triplet<double>> * triplets) const override
  {
    std::map<Index, NodeSums> sums;
    for (const auto & se : _secondary)
    {
      // The primary sides that may face the secondary side: those within half its length.
      const double length =
          std::hypot(position(problem, U, se[1], 0) - position(problem, U, se[0], 0),
                     position(problem, U, se[1], 1) - position(problem, U, se[0], 1));
      std::array<double, 4> sb;
      bounds(problem, U, se, 0.5 * length, sb);
      for (const auto & pe : _primary)
      {
        std::array<double, 4> pb;
        bounds(problem, U, pe, 0.0, pb);
        if (pb[1] < sb[0] || pb[0] > sb[1] || pb[3] < sb[2] || pb[2] > sb[3])
          continue;
        Pair pair = makePair(problem, U, se, pe);
        ADReal lo, hi;
        if (!overlapSegment(pair, lo, hi))
          continue;
        for (double gauss : kGauss)
          addQuadraturePoint(pair, lo, hi, gauss, old);
        scatter(pair, R, reference, triplets);
        addToSums(pair, se, sums);
      }
    }
    for (Index n : _secondary_nodes)
    {
      const auto it = sums.find(n);
      if (it == sums.end() || it->second.D <= 0.0)
        freeNode(problem, U, n, R, reference, triplets);
      else
        contactNode(problem, U, n, it->second, R, reference, triplets);
    }
  }

private:
  /// The Gauss points of two-point quadrature on [-1, 1].
  static inline const double kGauss[2] = {-1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)};

  /// The weighted gap, the measure and the weighted slip of a secondary node, with their
  /// derivatives by global unknown.
  struct NodeSums
  {
    double g = 0, D = 0, s = 0;
    std::map<Index, double> dg, dD, ds;
  };

  /// A secondary side A B and a primary side C D that may face each other.
  struct Pair
  {
    /// The local unknowns: the multipliers of A and B (pressures, then tractions with friction),
    /// then the displacements of A, B, C and D from the place first_u.
    std::vector<Index> dofs;
    int first_u = 0;
    /// The positions and the displacements of A, B, C and D.
    std::array<std::array<ADReal, 2>, 4> x, u;
    /// The pressure and the tangential traction at A and B.
    std::array<ADReal, 2> pressure, traction;
    /// The tangent B - A, its square length, the measure of the side and its unit normal.
    ADReal t0, t1, L2, L, n0, n1;
    /// The undeformed positions of A and B.
    Point a, b;
    /// The residual of the local unknowns (the displacements take minus the contact forces), and
    /// the weighted gap, the measure and the weighted slip of A and B.
    std::vector<ADReal> R;
    std::array<ADReal, 2> g, D, s;
  };

  /// The current position of node @p n along the axis @p d.
  double position(const Problem & problem, const Eigen::VectorXd & U, Index n, int d) const
  {
    return problem.mesh().node(n)[d] + U[problem.dof(n, _disp[d])];
  }

  /// The box (x_min, x_max, y_min, y_max) of the current side @p e, grown by @p grow.
  void bounds(const Problem & problem,
              const Eigen::VectorXd & U,
              const std::array<Index, 2> & e,
              double grow,
              std::array<double, 4> & b) const
  {
    for (int d = 0; d < 2; ++d)
    {
      const double p0 = position(problem, U, e[0], d), p1 = position(problem, U, e[1], d);
      b[2 * d] = std::min(p0, p1) - grow;
      b[2 * d + 1] = std::max(p0, p1) + grow;
    }
  }

  /// The pair of the secondary side @p se and the primary side @p pe, with the derivatives of
  /// its local unknowns.
  Pair makePair(const Problem & problem,
                const Eigen::VectorXd & U,
                const std::array<Index, 2> & se,
                const std::array<Index, 2> & pe) const
  {
    const Mesh & mesh = problem.mesh();
    const bool friction = _tau >= 0;
    Pair p;
    const std::array<Index, 4> nodes{se[0], se[1], pe[0], pe[1]};
    for (int k = 0; k < 2; ++k)
      p.dofs.push_back(problem.dof(se[k], _var));
    if (friction)
      for (int k = 0; k < 2; ++k)
        p.dofs.push_back(problem.dof(se[k], _tau));
    p.first_u = static_cast<int>(p.dofs.size());
    for (Index n : nodes)
      for (int d = 0; d < 2; ++d)
        p.dofs.push_back(problem.dof(n, _disp[d]));
    const int nd = static_cast<int>(p.dofs.size());
    const auto unknown = [&](int slot) { return ADReal::independent(U[p.dofs[slot]], slot, nd); };
    for (int k = 0; k < 4; ++k)
      for (int d = 0; d < 2; ++d)
      {
        p.u[k][d] = unknown(p.first_u + 2 * k + d);
        p.x[k][d] = mesh.node(nodes[k])[d] + p.u[k][d];
      }
    for (int k = 0; k < 2; ++k)
    {
      p.pressure[k] = unknown(k);
      p.traction[k] = friction ? unknown(2 + k) : ADReal(0.0);
    }
    p.t0 = p.x[1][0] - p.x[0][0];
    p.t1 = p.x[1][1] - p.x[0][1];
    p.L2 = p.t0 * p.t0 + p.t1 * p.t1;
    // The measure and the normal: of the deformed surface, or of the undeformed one.
    p.a = mesh.node(se[0]);
    p.b = mesh.node(se[1]);
    p.L = _large ? sqrt(p.L2) : ADReal(std::hypot(p.b[0] - p.a[0], p.b[1] - p.a[1]));
    p.n0 = _large ? p.t1 / p.L : ADReal((p.b[1] - p.a[1]) / p.L.value());
    p.n1 = _large ? -p.t0 / p.L : ADReal(-(p.b[0] - p.a[0]) / p.L.value());
    p.R.assign(static_cast<std::size_t>(nd), ADReal(0.0));
    return p;
  }

  /// The part [lo, hi] of the secondary side, in its coordinate xi from 0 at A to 1 at B, that
  /// the primary side projects onto, or false where the overlap is empty.
  static bool overlapSegment(const Pair & p, ADReal & lo, ADReal & hi)
  {
    const auto along = [&](int k)
    { return ((p.x[k][0] - p.x[0][0]) * p.t0 + (p.x[k][1] - p.x[0][1]) * p.t1) / p.L2; };
    const ADReal xiC = along(2), xiD = along(3);
    lo = xiC.value() < xiD.value() ? xiC : xiD;
    hi = xiC.value() < xiD.value() ? xiD : xiC;
    if (lo.value() < 0.0)
      lo = ADReal(0.0);
    if (hi.value() > 1.0)
      hi = ADReal(1.0);
    return !(hi.value() - lo.value() <= 1e-12);
  }

  /// The terms of the quadrature point @p gauss of the overlap [lo, hi]: the contact forces on
  /// the displacements and the weighted gap, measure and slip of the secondary nodes.
  void addQuadraturePoint(Pair & p,
                          const ADReal & lo,
                          const ADReal & hi,
                          double gauss,
                          const Eigen::VectorXd * old) const
  {
    const auto & x = p.x;
    const ADReal xi = lo + (hi - lo) * (1.0 + gauss) / 2.0;
    const ADReal xs0 = x[0][0] + xi * p.t0, xs1 = x[0][1] + xi * p.t1;
    const ADReal pm0 = x[3][0] - x[2][0], pm1 = x[3][1] - x[2][1];
    const ADReal eta =
        ((xs0 - x[2][0]) * p.t0 + (xs1 - x[2][1]) * p.t1) / (pm0 * p.t0 + pm1 * p.t1);
    const ADReal xm0 = x[2][0] + eta * pm0, xm1 = x[2][1] + eta * pm1;
    const ADReal gap = (xm0 - xs0) * p.n0 + (xm1 - xs1) * p.n1;
    ADReal w = (hi - lo) / 2.0 * p.L * _thickness;
    w = w *
        coordinateFactor(_coord, _large ? xs0 : ADReal(p.a[0] + xi.value() * (p.b[0] - p.a[0])));
    const std::array<ADReal, 2> Ns{1.0 - xi, xi}, Nm{1.0 - eta, eta};
    const ADReal pressure = p.pressure[0] * Ns[0] + p.pressure[1] * Ns[1];
    const ADReal traction = p.traction[0] * Ns[0] + p.traction[1] * Ns[1];
    // The force on the secondary surface is -(lambda n + tau t), on the primary its opposite.
    // The residual takes minus the force.  The tangent of the same surface as the normal:
    // (-n1, n0).
    const ADReal f0 = (pressure * p.n0 - traction * p.n1) * w;
    const ADReal f1 = (pressure * p.n1 + traction * p.n0) * w;
    const int first_u = p.first_u;
    for (int k = 0; k < 2; ++k)
    {
      p.R[first_u + 2 * k] += Ns[k] * f0;
      p.R[first_u + 2 * k + 1] += Ns[k] * f1;
      p.R[first_u + 2 * (k + 2)] -= Nm[k] * f0;
      p.R[first_u + 2 * (k + 2) + 1] -= Nm[k] * f1;
    }
    const ADReal slip = _tau >= 0 ? slipAt(p, Ns, Nm, old) : ADReal(0.0);
    for (int k = 0; k < 2; ++k)
    {
      p.g[k] += Ns[k] * gap * w;
      p.D[k] += Ns[k] * w;
      p.s[k] += Ns[k] * slip * w;
    }
  }

  /// The slip of the secondary point of the shape functions @p Ns relative to the primary point
  /// of @p Nm, along the tangent, since the start of the step.
  static ADReal slipAt(const Pair & p,
                       const std::array<ADReal, 2> & Ns,
                       const std::array<ADReal, 2> & Nm,
                       const Eigen::VectorXd * old)
  {
    ADReal slip(0.0);
    for (int d = 0; d < 2; ++d)
    {
      const auto start = [&](int k) { return old ? (*old)[p.dofs[p.first_u + 2 * k + d]] : 0.0; };
      const ADReal us = Ns[0] * (p.u[0][d] - start(0)) + Ns[1] * (p.u[1][d] - start(1));
      const ADReal um = Nm[0] * (p.u[2][d] - start(2)) + Nm[1] * (p.u[3][d] - start(3));
      slip += (us - um) * (d == 0 ? -p.n1 : p.n0);
    }
    return slip;
  }

  /// Add the residual of the displacements of @p p to the global residual and matrix.
  static void scatter(const Pair & p,
                      Eigen::VectorXd & R,
                      Eigen::VectorXd * reference,
                      std::vector<Eigen::Triplet<double>> * triplets)
  {
    const int nd = static_cast<int>(p.dofs.size());
    for (int i = p.first_u; i < nd; ++i)
    {
      R[p.dofs[i]] += p.R[i].value();
      if (reference)
        (*reference)[p.dofs[i]] += std::abs(p.R[i].value());
      if (triplets)
        for (int j = 0; j < nd; ++j)
          if (p.R[i].derivative(j) != 0.0)
            triplets->emplace_back(p.dofs[i], p.dofs[j], p.R[i].derivative(j));
    }
  }

  /// Add the weighted gap, measure and slip of @p p to those of its secondary nodes @p se.
  static void
  addToSums(const Pair & p, const std::array<Index, 2> & se, std::map<Index, NodeSums> & sums)
  {
    const int nd = static_cast<int>(p.dofs.size());
    for (int k = 0; k < 2; ++k)
    {
      NodeSums & sum = sums[se[k]];
      sum.g += p.g[k].value();
      sum.D += p.D[k].value();
      sum.s += p.s[k].value();
      for (int j = 0; j < nd; ++j)
      {
        if (p.g[k].derivative(j) != 0.0)
          sum.dg[p.dofs[j]] += p.g[k].derivative(j);
        if (p.D[k].derivative(j) != 0.0)
          sum.dD[p.dofs[j]] += p.D[k].derivative(j);
        if (p.s[k].derivative(j) != 0.0)
          sum.ds[p.dofs[j]] += p.s[k].derivative(j);
      }
    }
  }

  /// Add the row @p row with the derivatives @p derivatives times @p factor to the matrix.
  static void addRow(std::vector<Eigen::Triplet<double>> * triplets,
                     Index row,
                     const std::map<Index, double> & derivatives,
                     double factor)
  {
    if (triplets && factor != 0.0)
      for (const auto & [column, value] : derivatives)
        triplets->emplace_back(row, column, factor * value);
  }

  /// The derivatives d(v / D) = dv / D - v dD / D^2 of the value per length of @p v with the
  /// derivatives @p dv, at the node of @p sum.
  static std::map<Index, double>
  perLength(double v, const std::map<Index, double> & dv, const NodeSums & sum)
  {
    std::map<Index, double> out;
    for (const auto & [k, d] : dv)
      out[k] += d / sum.D;
    for (const auto & [k, d] : sum.dD)
      out[k] -= v * d / (sum.D * sum.D);
    return out;
  }

  /// A secondary node that no primary surface faces: no pressure, no traction.
  void freeNode(const Problem & problem,
                const Eigen::VectorXd & U,
                Index n,
                Eigen::VectorXd & R,
                Eigen::VectorXd * reference,
                std::vector<Eigen::Triplet<double>> * triplets) const
  {
    for (int v : {_var, _tau})
    {
      if (v < 0)
        continue;
      const Index row = problem.dof(n, v);
      R[row] += U[row];
      if (reference)
        (*reference)[row] += std::abs(U[row]);
      if (triplets)
        triplets->emplace_back(row, row, 1.0);
    }
  }

  /// The complementarity function of the pressure at a secondary node that a primary surface
  /// faces, and with friction that of the tangential traction.
  void contactNode(const Problem & problem,
                   const Eigen::VectorXd & U,
                   Index n,
                   const NodeSums & sum,
                   Eigen::VectorXd & R,
                   Eigen::VectorXd * reference,
                   std::vector<Eigen::Triplet<double>> * triplets) const
  {
    const Index rl = problem.dof(n, _var);
    const double lambda = U[rl];
    // The gap per length, gt = g / D.
    const double gt = sum.g / sum.D;
    const std::map<Index, double> dgt = perLength(sum.g, sum.dg, sum);
    // A node that touches is active, so that two faces in contact at the start are held
    // together from the first iterate: a gap of the round-off of the positions (1e-12 of the
    // element size, D) counts as closed.
    const double trial = lambda - _c * gt;
    const bool active = trial >= -_c * 1e-12 * sum.D;
    if (active)
    {
      R[rl] += _c * gt;
      addRow(triplets, rl, dgt, _c);
    }
    else
    {
      R[rl] += lambda;
      if (triplets)
        triplets->emplace_back(rl, rl, 1.0);
    }
    if (reference)
      (*reference)[rl] += std::abs(lambda) + std::abs(_c * gt);
    if (_tau >= 0)
      frictionCondition(problem.dof(n, _tau),
                        rl,
                        U,
                        sum,
                        active ? _mu * trial : 0.0,
                        active,
                        dgt,
                        R,
                        reference,
                        triplets);
  }

  /// Coulomb friction at a secondary node: C = max(b, |z|) tau - b z, with b = mu max(0, lambda -
  /// c gt), given as @p b, and z = tau + c st, st the slip per length.
  void frictionCondition(Index rt,
                         Index rl,
                         const Eigen::VectorXd & U,
                         const NodeSums & sum,
                         double b,
                         bool active,
                         const std::map<Index, double> & dgt,
                         Eigen::VectorXd & R,
                         Eigen::VectorXd * reference,
                         std::vector<Eigen::Triplet<double>> * triplets) const
  {
    const double tau = U[rt];
    const double st = sum.s / sum.D;
    const std::map<Index, double> dst = perLength(sum.s, sum.ds, sum);
    // db = mu (d lambda - c d gt) where the node is active.
    std::map<Index, double> db;
    if (active)
    {
      db[rl] += _mu;
      for (const auto & [k, v] : dgt)
        db[k] -= _mu * _c * v;
    }
    const double z = tau + _c * st;
    std::map<Index, double> dz = {{rt, 1.0}};
    for (const auto & [k, v] : dst)
      dz[k] += _c * v;
    double C;
    std::map<Index, double> dC;
    if (b <= 0.0)
    {
      C = tau;
      dC[rt] = 1.0;
    }
    else if (std::abs(z) <= b)
    {
      // Stick: C = b tau - b z = -b c st.
      C = -b * _c * st;
      for (const auto & [k, v] : db)
        dC[k] -= _c * st * v;
      for (const auto & [k, v] : dst)
        dC[k] -= b * _c * v;
    }
    else
    {
      // Slip: C = |z| tau - b z.
      const double sign = z > 0.0 ? 1.0 : -1.0;
      C = std::abs(z) * tau - b * z;
      dC[rt] += std::abs(z);
      for (const auto & [k, v] : dz)
        dC[k] += sign * tau * v - b * v;
      for (const auto & [k, v] : db)
        dC[k] -= z * v;
    }
    R[rt] += C;
    if (reference)
      (*reference)[rt] += std::abs(std::max(b, std::abs(z)) * tau) + std::abs(b * z);
    addRow(triplets, rt, dC, 1.0);
  }

  double _mu, _c, _thickness;
  bool _large;
  CoordinateSystem _coord = CoordinateSystem::Cartesian;
  int _tau = -1;
  std::vector<int> _disp;
  std::vector<std::array<Index, 2>> _secondary, _primary;
  std::vector<Index> _secondary_nodes, _multipliers;
};

/// A cohesive zone between two surfaces of a two-dimensional problem that coincide at the start:
/// a bilinear traction-separation law with damage that does not heal.  The points of the two
/// surfaces are paired once, on the undeformed surfaces, as a bonded interface pairs its material
/// points.  The separation of a pair is delta = x_p - x_s, with the normal part g_n = delta . n
/// and the tangential part g_t = delta . t.  The traction on the secondary surface is
/// K (1 - D) delta, except that a closing normal part (g_n < 0) keeps the full stiffness K, so
/// that the surfaces do not pass through each other, and the primary surface takes the opposite.  The
/// damage D grows with the largest effective opening kappa = max sqrt(<g_n>^2 + g_t^2) reached:
/// D = 0 up to delta_0 = strength / K, D = delta_f (kappa - delta_0) / (kappa (delta_f - delta_0))
/// up to delta_f = 2 toughness / strength, and D = 1 beyond, so that the traction peaks at the
/// strength and the work of separation per area is the toughness.  The largest opening of each
/// pair is a history that a checkpoint keeps (CheckpointHistory), by the element of the secondary
/// side, the side and the place of the pair along it.
class CohesiveZone : public IntegratedBC, public CheckpointHistory
{
public:
  static InputParameters validParams()
  {
    InputParameters p = IntegratedBC::validParams();
    p.setClassDescription(
        "A cohesive zone between the surface 'boundary' (secondary) and 'primary_boundary' of a "
        "two-dimensional problem, plane or axisymmetric, which coincide at the start. The "
        "traction on the secondary surface is K (1 - D) delta, with K the 'stiffness', delta "
        "the separation of the two surfaces and D the damage, and the primary surface takes the "
        "opposite. A closing normal separation keeps the full stiffness K, so that the surfaces "
        "do not pass through each other. The bilinear traction-separation law, normal and "
        "tangential, makes the traction peak at the 'strength', and the work of separation per "
        "unit area is the 'toughness'. The damage grows with the largest opening reached and "
        "does not heal. 'variable' is one of the displacements, and one object acts on all of "
        "them.");
    p.addRequired("primary_boundary", ParameterKind::String, "The primary surface.");
    addDisplacementsParam(p, "The displacements of both bodies.");
    p.addRequired("strength", ParameterKind::Real, "The largest traction (Pa).");
    p.setRange("strength", ParameterRange::Positive);
    p.addRequired(
        "toughness", ParameterKind::Real, "The work of separation per unit area (J/m^2).");
    p.setRange("toughness", ParameterRange::Positive);
    p.addRequired("stiffness",
                  ParameterKind::Real,
                  "The stiffness K (Pa/m) of the undamaged zone, large against the stiffness of "
                  "the bodies, of the size of 50 times Young's modulus over the element size. The "
                  "zone then opens by strength/K, a small fraction of an element, before it "
                  "softens. The opening at failure, 2 toughness/strength, must exceed "
                  "strength/K.");
    p.setRange("stiffness", ParameterRange::Positive);
    p.addOptional("thickness",
                  ParameterKind::Real,
                  1.0,
                  "The out-of-plane thickness (m) of a plane problem, as stress_divergence takes "
                  "it. Default 1, the default of stress_divergence.");
    p.setRange("thickness", ParameterRange::Positive);
    p.addOptional("large_deformation",
                  ParameterKind::Boolean,
                  false,
                  "Take the normal of the deformed secondary surface, for bodies at finite strain. "
                  "Default false: the normal of the undeformed surface.");
    return p;
  }
  explicit CohesiveZone(const InputParameters & p)
      : IntegratedBC(p), _K(p.getReal("stiffness")), _thickness(p.getReal("thickness")),
        _large(p.getBool("large_deformation"))
  {
    const double strength = p.getReal("strength");
    _d0 = strength / _K;
    _df = 2.0 * p.getReal("toughness") / strength;
    if (_df <= _d0)
      throw InputError("'" + name() +
                       "': the zone would fail before it softens: the opening at "
                       "failure, 2 toughness / strength, must exceed the elastic "
                       "opening, strength / stiffness.");
  }
  bool isConstraint() const override { return true; }
  ADReal computeBoundaryFlux(const QpContext &) const override { return ADReal(0.0); }
  std::vector<std::string> constraintBoundaries() const override
  {
    std::vector<std::string> names = _boundaries;
    names.push_back(_params.getString("primary_boundary"));
    return names;
  }

  void initialSetup(Problem & problem) override
  {
    IntegratedBC::initialSetup(problem);
    const Mesh & mesh = problem.mesh();
    if (mesh.spaceDimension() != 2)
      throw InputError("'" + name() +
                       "': cohesive zones are available in two dimensions "
                       "only.");
    const CoordinateSystem coord = problem.coordinateSystem();
    _disp.clear();
    for (const auto & d : _params.getStringList("displacements"))
      _disp.push_back(problem.variableIndex(d));
    // The pairs: the two Gauss points of each overlap of a secondary and a primary side, on the
    // undeformed surfaces.
    static const double gauss[2] = {-1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)};
    _pairs.clear();
    std::vector<Side> sides;
    const auto secondary = mortarEdges(mesh, _boundaries, true, name(), &sides);
    for (std::size_t s = 0; s < secondary.size(); ++s)
    {
      const auto & se = secondary[s];
      const std::size_t first_pair = _pairs.size();
      const Point a = mesh.node(se[0]), b = mesh.node(se[1]);
      const double t0 = b[0] - a[0], t1 = b[1] - a[1], L = std::hypot(t0, t1);
      const double n0 = t1 / L, n1 = -t0 / L;
      const auto along = [&](const Point & x)
      { return ((x[0] - a[0]) * t0 + (x[1] - a[1]) * t1) / (L * L); };
      const auto off = [&](const Point & x) { return (x[0] - a[0]) * n0 + (x[1] - a[1]) * n1; };
      double covered = 0.0;
      for (const auto & pe :
           mortarEdges(mesh, {_params.getString("primary_boundary")}, false, name()))
      {
        const Point c = mesh.node(pe[0]), d = mesh.node(pe[1]);
        if (std::abs(off(c)) > 1e-8 * L || std::abs(off(d)) > 1e-8 * L)
          continue;
        const double xiC = along(c), xiD = along(d);
        const double lo = std::max(0.0, std::min(xiC, xiD)), hi = std::min(1.0, std::max(xiC, xiD));
        if (hi - lo <= 1e-12)
          continue;
        covered += hi - lo;
        for (int q = 0; q < 2; ++q)
        {
          const double xi = lo + (hi - lo) * (1.0 + gauss[q]) / 2.0;
          const double eta = (xi - xiC) / (xiD - xiC);
          double w = (hi - lo) / 2.0 * L * _thickness;
          w *= coordinateFactor(coord, a[0] + xi * t0);
          std::array<double, 2> offset;
          for (int k = 0; k < 2; ++k)
            offset[k] = (1.0 - eta) * c[k] + eta * d[k] - (1.0 - xi) * a[k] - xi * b[k];
          _pairs.push_back({{se[0], se[1], pe[0], pe[1]},
                            {1.0 - xi, xi},
                            {1.0 - eta, eta},
                            offset,
                            w,
                            {n0, n1},
                            sides[s].first,
                            0});
        }
      }
      // The pairs of the side in their order along it, which the numbering of the primary sides
      // does not change, so that the key of a pair is the same in every partition of the mesh.
      std::sort(_pairs.begin() + static_cast<std::ptrdiff_t>(first_pair),
                _pairs.end(),
                [](const Pair & left, const Pair & right) { return left.Ns[1] < right.Ns[1]; });
      for (std::size_t k = first_pair; k < _pairs.size(); ++k)
        _pairs[k].key = (static_cast<std::uint64_t>(sides[s].second) << 32) |
                        static_cast<std::uint64_t>(k - first_pair);
      if (std::abs(covered - 1.0) > 1e-8)
        throw InputError("'" + name() +
                         "': the surfaces of a cohesive zone must coincide at the "
                         "start. Part of the secondary side at (" +
                         std::to_string(a[0]) + ", " + std::to_string(a[1]) +
                         ") faces no primary side.");
    }
    _start.assign(_pairs.size(), 0.0);
    _trial = _start;
    _committed = _start;
  }

  void assembleConstraint(const Problem & problem,
                          const Eigen::VectorXd & U,
                          const Eigen::VectorXd *,
                          Eigen::VectorXd & R,
                          Eigen::VectorXd * reference,
                          std::vector<Eigen::Triplet<double>> * triplets) const override
  {
    const Mesh & mesh = problem.mesh();
    for (std::size_t i = 0; i < _pairs.size(); ++i)
    {
      const Pair & pair = _pairs[i];
      // The local unknowns: the displacements of A, B (secondary) and C, D (primary).
      std::array<Index, 8> dofs;
      for (int k = 0; k < 4; ++k)
        for (int d = 0; d < 2; ++d)
          dofs[2 * k + d] = problem.dof(pair.nodes[k], _disp[d]);
      std::array<std::array<ADReal, 2>, 4> u;
      for (int k = 0; k < 4; ++k)
        for (int d = 0; d < 2; ++d)
          u[k][d] = ADReal::independent(U[dofs[2 * k + d]], 2 * k + d, 8);
      // The separation from the displacements and the offset at the start, not from the
      // positions, whose round-off the stiffness would magnify.
      std::array<ADReal, 2> delta, n{ADReal(pair.normal[0]), ADReal(pair.normal[1])};
      for (int d = 0; d < 2; ++d)
        delta[d] = pair.offset[d] + pair.Nm[0] * u[2][d] + pair.Nm[1] * u[3][d] -
                   pair.Ns[0] * u[0][d] - pair.Ns[1] * u[1][d];
      if (_large)
      {
        const auto x = [&](int k, int d) { return mesh.node(pair.nodes[k])[d] + u[k][d]; };
        const ADReal t0 = x(1, 0) - x(0, 0), t1 = x(1, 1) - x(0, 1);
        const ADReal L = sqrt(t0 * t0 + t1 * t1);
        n = {t1 / L, -t0 / L};
      }
      const ADReal gn = delta[0] * n[0] + delta[1] * n[1];
      const ADReal gt = -delta[0] * n[1] + delta[1] * n[0];
      // The largest effective opening, which a closing normal part does not add to.
      const double open = std::max(gn.value(), 0.0);
      const double reach = std::sqrt(open * open + gt.value() * gt.value());
      ADReal kappa(_start[i]);
      if (reach > _start[i])
        kappa = sqrt((gn.value() > 0.0 ? gn * gn : ADReal(0.0)) + gt * gt);
      _trial[i] = kappa.value();
      ADReal D(0.0);
      if (kappa.value() >= _df)
        D = ADReal(1.0);
      else if (kappa.value() > _d0)
        D = _df * (kappa - _d0) / (kappa * (_df - _d0));
      const ADReal kn = gn.value() < 0.0 ? ADReal(_K) : _K * (1.0 - D), kt = _K * (1.0 - D);
      // The force per area on the secondary surface.  The residual takes minus it, and the
      // primary surface the opposite.
      const std::array<ADReal, 2> f{kn * gn * n[0] - kt * gt * n[1],
                                    kn * gn * n[1] + kt * gt * n[0]};
      for (int k = 0; k < 4; ++k)
        for (int d = 0; d < 2; ++d)
        {
          const ADReal r = (k < 2 ? -pair.Ns[k] : pair.Nm[k - 2]) * f[d] * pair.weight;
          const Index row = dofs[2 * k + d];
          R[row] += r.value();
          if (reference)
            (*reference)[row] += std::abs(r.value());
          if (triplets)
            for (int j = 0; j < 8; ++j)
              if (r.derivative(j) != 0.0)
                triplets->emplace_back(row, dofs[j], r.derivative(j));
        }
    }
  }
  void commitHistory() override
  {
    _committed = _trial;
    _start = _trial;
  }
  void startHistory(bool from_committed) override { _start = from_committed ? _committed : _trial; }

  Problem::StateSnapshot committedHistory(const Problem & problem) const override
  {
    Problem::StateSnapshot history;
    const auto elements = static_cast<std::size_t>(problem.mesh().numElements());
    history.keys.resize(elements);
    history.values.resize(elements);
    for (std::size_t i = 0; i < _pairs.size(); ++i)
    {
      const auto e = static_cast<std::size_t>(_pairs[i].element);
      history.keys[e].push_back(_pairs[i].key);
      history.values[e].push_back(_committed[i]);
    }
    return history;
  }
  std::string restoreHistory(const Problem &, const Problem::StateSnapshot & history) override
  {
    for (std::size_t i = 0; i < _pairs.size(); ++i)
    {
      const auto e = static_cast<std::size_t>(_pairs[i].element);
      const auto & keys = history.keys[e];
      const auto at = std::find(keys.begin(), keys.end(), _pairs[i].key);
      if (at == keys.end() || history.values[e].size() != keys.size())
        return "'" + name() +
               "': the checkpoint holds no largest opening of a point of this "
               "cohesive zone; a checkpoint restarts the problem that wrote it.";
      _committed[i] = history.values[e][static_cast<std::size_t>(at - keys.begin())];
    }
    _start = _committed;
    _trial = _committed;
    return "";
  }

private:
  struct Pair
  {
    std::array<Index, 4> nodes;
    std::array<double, 2> Ns, Nm, offset;
    double weight;
    std::array<double, 2> normal;
    /// The element of the secondary side, and the key of the pair in its history: the side of
    /// the element in the upper 32 bits and the place of the pair along the side in the lower.
    Index element;
    std::uint64_t key;
  };
  double _K, _thickness, _d0, _df;
  bool _large;
  std::vector<int> _disp;
  std::vector<Pair> _pairs;
  // The largest effective opening of each pair: at the start of the step, reached by the last
  // assembly, and accepted.
  std::vector<double> _start;
  mutable std::vector<double> _trial;
  std::vector<double> _committed;
};

/// A prescribed normal displacement n . u = g of a boundary, the tangential displacements free:
/// a roller on an inclined or curved face.  At each node of the boundary the normal force lambda
/// (N), a Lagrange multiplier, acts on the body along n, and the node's equation is
/// c (n . u - g) = 0, so the condition is exact.  The normal of a node is the mean of the normals
/// of its sides, weighted by their areas.  A node whose free displacements cannot move along n
/// (a corner that other conditions hold) has no multiplier: lambda = 0 there.
class NormalDisplacementBC : public IntegratedBC
{
public:
  static InputParameters validParams()
  {
    InputParameters p = IntegratedBC::validParams();
    p.setClassDescription(
        "A prescribed normal displacement n . u = g on the boundary, with n the outward normal, "
        "u the displacement and g the 'value', and the tangential displacements free: a roller "
        "on an inclined or curved face. The condition holds exactly at each node of the "
        "boundary through a Lagrange multiplier, 'variable', the normal force (N) on the body "
        "there. The normal of a node is the area-weighted mean of the normals of its sides. At a "
        "node that other conditions hold along n, the normal force is zero.");
    addDisplacementsParam(p, "The displacements of the body, one per dimension of space.");
    p.addOptional("value",
                  ParameterKind::Function,
                  0.0,
                  "The normal displacement g (m) along the outward normal, a constant or an "
                  "expression of x, y, z and t. Default 0: a roller that holds the boundary in "
                  "place along its normal.");
    p.addRequired("stiffness_scale",
                  ParameterKind::Real,
                  "The scale c (N/m) of the equation of a node, of the size of the stiffness of a "
                  "node. It sets the conditioning and leaves the solution unchanged.");
    p.setRange("stiffness_scale", ParameterRange::Positive);
    return p;
  }
  explicit NormalDisplacementBC(const InputParameters & p)
      : IntegratedBC(p), _c(p.getReal("stiffness_scale"))
  {
  }
  bool isConstraint() const override { return true; }
  ADReal computeBoundaryFlux(const QpContext &) const override { return ADReal(0.0); }
  std::vector<std::string> constraintBoundaries() const override { return _boundaries; }
  std::vector<Index> multiplierDofs(const Problem &) const override { return _multipliers; }

  void initialSetup(Problem & problem) override
  {
    IntegratedBC::initialSetup(problem);
    const Mesh & mesh = problem.mesh();
    _disp.clear();
    for (const auto & d : _params.getStringList("displacements"))
      _disp.push_back(problem.variableIndex(d));
    if (static_cast<int>(_disp.size()) != mesh.spaceDimension())
      throw InputError("'" + name() + "': give one displacement per dimension of space.");
    _g.setup(problem, _params, "value", name());
    std::map<Index, Point> normals;
    QuadratureSpec spec;
    spec.points = 3;
    std::vector<IntegrationPoint> points;
    MappedPoint mp;
    for (const auto & boundary : _boundaries)
      for (const Side & side : mesh.sideset(boundary))
      {
        const Element & el = mesh.element(side.first);
        buildSidePoints(mesh, side, false, spec, points);
        for (const auto & ip : points)
        {
          mapPoint(mesh, el, ip.xi, mp);
          for (Index n : mesh.sideNodes(side))
          {
            const int a =
                static_cast<int>(std::find(el.nodes.begin(), el.nodes.begin() + el.numNodes(), n) -
                                 el.nodes.begin());
            for (int d = 0; d < 3; ++d)
              normals[n][d] += mp.N[a] * ip.area[d];
          }
        }
      }
    _nodes.clear();
    _multipliers.clear();
    for (auto & [n, v] : normals)
    {
      const double length = std::sqrt(dot(v, v));
      for (int d = 0; d < 3; ++d)
        v[d] /= length;
      _nodes.push_back({n, v});
      _multipliers.push_back(problem.dof(n, _var));
    }
  }

  void assembleConstraint(const Problem & problem,
                          const Eigen::VectorXd & U,
                          const Eigen::VectorXd *,
                          Eigen::VectorXd & R,
                          Eigen::VectorXd * reference,
                          std::vector<Eigen::Triplet<double>> * triplets) const override
  {
    const std::vector<char> held = problem.constrainedDofs();
    const Mesh & mesh = problem.mesh();
    QpContext ctx;
    ctx.time = problem.time();
    for (const auto & [n, normal] : _nodes)
    {
      const Index row = problem.dof(n, _var);
      const double lambda = U[row];
      // Whether a free displacement of the node can move along n.
      double free = 0.0;
      for (std::size_t d = 0; d < _disp.size(); ++d)
        if (!held[static_cast<std::size_t>(problem.dof(n, _disp[d]))])
          free += normal[d] * normal[d];
      if (free < 1e-12)
      {
        R[row] += lambda;
        if (reference)
          (*reference)[row] += std::abs(lambda);
        if (triplets)
          triplets->emplace_back(row, row, 1.0);
        continue;
      }
      ctx.x = mesh.node(n);
      double gap = -_g.value(ctx).value();
      for (std::size_t d = 0; d < _disp.size(); ++d)
      {
        const Index column = problem.dof(n, _disp[d]);
        gap += normal[d] * U[column];
        // The force lambda n on the body.  The residual takes minus it.
        R[column] -= lambda * normal[d];
        if (reference)
          (*reference)[column] += std::abs(lambda * normal[d]);
        if (triplets)
        {
          triplets->emplace_back(column, row, -normal[d]);
          triplets->emplace_back(row, column, _c * normal[d]);
        }
      }
      R[row] += _c * gap;
      if (reference)
        (*reference)[row] += std::abs(lambda) + std::abs(_c * gap);
    }
  }

private:
  double _c;
  std::vector<int> _disp;
  FieldCoefficient _g;
  std::vector<std::pair<Index, Point>> _nodes;
  std::vector<Index> _multipliers;
};

} // namespace

void
registerInterfaceObjects(Factory & f)
{
  const std::string m = "solid_mechanics";
  f.add<MortarContact>("mortar_contact", ObjectCategory::BoundaryCondition, m);
  f.add<CohesiveZone>("cohesive_zone", ObjectCategory::BoundaryCondition, m);
  f.add<NormalDisplacementBC>("normal_displacement", ObjectCategory::BoundaryCondition, m);
}

} // namespace dualmesh::solid_mechanics
