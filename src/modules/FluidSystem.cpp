// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The flow system of the fluid module (FlowSystem.h).
#include "dualmesh/modules/FluidSystem.h"
#include "dualmesh/base/Kernel.h"
#include "dualmesh/base/Problem.h"
#include "dualmesh/fe/Assembly.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace dualmesh
{

namespace
{

/// The refusal of an enclosed flow whose pressure level changes the flow.
const char * const enclosedLevelMessage =
    "The pressure of this enclosed flow enters its equations (through a viscosity, a density or a "
    "source that depends on it), so its level changes the flow, and no boundary fixes it. Give "
    "the pressure at a point, flow.add_boundary_condition(\"pressure = 0\", point=[...]), or a "
    "boundary with a pressure or traction condition.";

/// The context of a point of the flow of @p problem at its current time, whose position, element
/// and block the caller sets.
QpContext
flowContext(const Problem & problem)
{
  QpContext context;
  context.dim = problem.mesh().spaceDimension();
  context.time = problem.time();
  return context;
}

/// Split the unknowns of @p problem into those of the pressure variable @p p and the others.
void
partitionDofs(const Problem & problem, int p, SaddlePointBlocks & blocks)
{
  const int nv = problem.numVariables();
  for (Index d = 0; d < problem.numDofs(); ++d)
    if (d % nv == p)
      blocks.pressure_dofs.push_back(d);
    else
      blocks.other_dofs.push_back(d);
}

/// Set the sign of S = C - D F^{-1} G relative to G^T F^{-1} G from the coupling blocks of the
/// Jacobian @p J: the Frobenius product <G, D^T> is negative when G = -D^T, and then -D F^{-1} G
/// is G^T F^{-1} G.  It is taken over the velocities only: the other unknowns that the flow
/// couples to the pressure (the temperature) have other units, and would make the sign depend on
/// them.
void
setSchurSign(const Problem & problem,
             const MassEquation & constraint,
             const SparseMatrix & J,
             SaddlePointBlocks & blocks)
{
  const int nv = problem.numVariables();
  const FlowCoefficients velocities =
      flowCoefficientsAt(problem, constraint, problem.mesh().node(0));
  std::vector<char> is_velocity(static_cast<std::size_t>(nv), 0);
  for (int d = 0; d < velocities.num_velocities; ++d)
    is_velocity[static_cast<std::size_t>(velocities.velocities[d])] = 1;
  double product = 0.0;
  for (Index d : blocks.pressure_dofs)
    for (SparseMatrix::InnerIterator it(J, d); it; ++it)
      if (is_velocity[static_cast<std::size_t>(it.row() % nv)])
        product += it.value() * J.coeff(d, it.row());
  blocks.coupling = product;
  blocks.sign = product < 0.0 ? 1 : -1;
}

/// What the blocks of one approximation of the Schur complement take: the approximation, whether
/// it is pcd or lsc, and the time step dt and the weight theta of the steady terms of the
/// equations whose Jacobian is solved (zero and one for a steady solve).
struct ApproximationTerms
{
  SchurApproximation approximation = SchurApproximation::PressureMass;
  bool pcd = false, lsc = false;
  double dt = 0.0, theta = 1.0;
};

/// The integrals over one element that the approximations of the Schur complement take: on the
/// pressure nodes, the mass M_k, the integrals m_k of the shape functions, (1/mu) M_k and F_p,
/// and the diagonal of the velocity mass matrix on all nodes.  The measure of the element, with
/// and without the coordinate factor, and the integrals of the viscosity and of the density of
/// the time derivative, with the flow coefficients at its last point.
struct ElementIntegrals
{
  /// The number of pressure nodes and of all nodes.
  int np = 0, nn = 0;
  double M[kMaxElementNodes][kMaxElementNodes] = {};
  double Mmu[kMaxElementNodes][kMaxElementNodes] = {};
  double F[kMaxElementNodes][kMaxElementNodes] = {};
  double m[kMaxElementNodes] = {}, velocity_diagonal[kMaxElementNodes] = {};
  double volume = 0.0, area = 0.0, viscosity = 0.0, time_density = 0.0;
  FlowCoefficients c;
};

/// The quadrature points of an element and their geometry, kept from one element to the next.
struct ElementScratch
{
  std::vector<IntegrationPoint> points;
  std::vector<MappedPoint> mapped;
  MappedPoint geo;
  QpContext context;
};

/// Add the point of weight @p w to the matrices of the pressure nodes of @p I, with the pressure
/// shape functions @p N and their gradients @p dN there, the current velocity @p wind and the flow
/// coefficients I.c of the point.
void
addPressureMatrices(const double * N,
                    const Point * dN,
                    double w,
                    const Point & wind,
                    const ApproximationTerms & terms,
                    ElementIntegrals & I)
{
  const FlowCoefficients & c = I.c;
  const double mass_rate = terms.dt > 0.0 ? c.time_density / terms.dt : 0.0;
  for (int a = 0; a < I.np; ++a)
  {
    I.m[a] += w * N[a];
    for (int b = 0; b < I.np; ++b)
    {
      I.M[a][b] += w * N[a] * N[b];
      I.Mmu[a][b] += w * N[a] * N[b] / c.viscosity;
      if (terms.pcd)
      {
        double diffusion = 0.0, convection = 0.0;
        for (int d = 0; d < 3; ++d)
        {
          diffusion += dN[a][d] * dN[b][d];
          convection += wind[d] * dN[b][d];
        }
        I.F[a][b] += w * (mass_rate * N[a] * N[b] +
                          terms.theta * (c.viscosity * diffusion + c.density * convection * N[a]));
      }
    }
  }
}

/// The ElementIntegrals of element @p e of the flow whose mass balance is @p constraint, with the
/// current velocity as the wind of F_p.  @p refusal receives a message where the viscosity is not
/// positive.
ElementIntegrals
integrateElement(const Problem & problem,
                 const MassEquation & constraint,
                 Index e,
                 const ApproximationTerms & terms,
                 ElementScratch & scratch,
                 std::string & refusal)
{
  const Mesh & mesh = problem.mesh();
  const Vector & U = problem.solution();
  const int p = constraint.variable();
  const Element & el = mesh.element(e);
  QuadratureSpec spec;
  spec.kind = QuadratureSpec::Kind::Gauss;
  spec.points = ReferenceElement::get(el.type).order() + 2;
  buildElementPoints(mesh, e, false, PointSet::Volume, spec, scratch.points, &scratch.mapped);
  const bool corner = problem.usesCornerShape(el, p);
  ElementIntegrals I;
  I.np = corner ? el.numCorners() : el.numNodes();
  I.nn = el.numNodes();
  const int nn = I.nn;
  MappedPoint & geo = scratch.geo;
  QpContext & context = scratch.context;
  FlowCoefficients & c = I.c;
  for (std::size_t q = 0; q < scratch.points.size(); ++q)
  {
    if (q < scratch.mapped.size())
      geo = scratch.mapped[q];
    else
      mapPoint(mesh, el, scratch.points[q].xi, geo);
    if (corner)
      mapCornerShape(el, geo);
    const double * N = corner ? geo.corner_shape : geo.N;
    const Point * dN = corner ? geo.corner_gradient : geo.dN;
    context.x = geo.x;
    context.element = e;
    context.block = el.block;
    constraint.flowCoefficients(context, c);
    if (!(c.viscosity > 0.0))
      refusal = "The block preconditioners of the flow (pressure_mass, pcd, lsc) scale "
                "their operators by the dynamic_viscosity, which must be positive "
                "everywhere.";
    const double w = scratch.points[q].weight * problem.coordFactor(geo.x);
    // The measure of the element itself, without the factor of the
    // coordinate system, gives its size.
    I.area += scratch.points[q].weight;
    I.volume += w;
    I.viscosity += w * c.viscosity;
    I.time_density += w * c.time_density;
    // The current velocity, the wind of the convective term.
    Point wind{0.0, 0.0, 0.0};
    for (int d = 0; d < c.num_velocities; ++d)
      for (int a = 0; a < nn; ++a)
        wind[d] += geo.N[a] * U[problem.dof(el.nodes[a], c.velocities[d])];
    for (int a = 0; a < nn; ++a)
      I.velocity_diagonal[a] += w * geo.N[a] * geo.N[a];
    addPressureMatrices(N, dN, w, wind, terms, I);
  }
  return I;
}

/// Add the lumped masses of element @p el to @p lumped_mass: the diagonal of its mass matrices
/// scaled to the element measure (Hinton, Rock and Zienkiewicz), which is positive for quadratic
/// elements too.
void
addLumpedMass(const Problem & problem,
              const Element & el,
              int p,
              const ElementIntegrals & I,
              Eigen::VectorXd & lumped_mass)
{
  double pressure_trace = 0.0, velocity_trace = 0.0;
  for (int a = 0; a < I.np; ++a)
    pressure_trace += I.M[a][a];
  for (int a = 0; a < I.nn; ++a)
    velocity_trace += I.velocity_diagonal[a];
  for (int a = 0; a < I.np; ++a)
    lumped_mass[problem.dof(el.nodes[a], p)] += I.M[a][a] * I.volume / pressure_trace;
  for (int d = 0; d < I.c.num_velocities; ++d)
    for (int a = 0; a < I.nn; ++a)
      lumped_mass[problem.dof(el.nodes[a], I.c.velocities[d])] +=
          I.velocity_diagonal[a] * I.volume / velocity_trace;
}

/// The factor of the stabilization C_1 of pcd over that of lsc.  pcd applies A_p^{-1} to
/// F_p Q_p^{-1} r, which holds the spurious modes at full size, and needs C_1 a hundred times
/// larger than lsc does (see SaddlePointBlocks.h).
constexpr double kPcdStabilizationFactor = 100.0;

/// Add the entries of element @p el to the pressure blocks of @p blocks that the approximation
/// takes, with the stabilisation of A_p and of the middle factor of lsc for an equal-order
/// element: the L2 distance from the element mean, scaled.
void
addPressureBlocks(const Problem & problem,
                  const Element & el,
                  int p,
                  const ElementIntegrals & I,
                  const ApproximationTerms & terms,
                  SaddlePointBlocks & blocks)
{
  const bool pcd = terms.pcd, lsc = terms.lsc;
  const bool stabilized = I.c.stabilized;
  const double h2 = std::pow(I.area, 2.0 / problem.mesh().dimension());
  const double viscosity = I.viscosity / I.volume;
  const double time_density = I.time_density / I.volume;
  const double c1 = (pcd ? kPcdStabilizationFactor : 1.0) / h2;
  const double c2 =
      terms.theta * viscosity / (h2 * h2) + (terms.dt > 0.0 ? time_density / (terms.dt * h2) : 0.0);
  for (int a = 0; a < I.np; ++a)
    for (int b = 0; b < I.np; ++b)
    {
      const Index da = problem.dof(el.nodes[a], p), db = problem.dof(el.nodes[b], p);
      if (blocks.constrained[da] || blocks.constrained[db])
        continue;
      if (terms.approximation == SchurApproximation::PressureMass)
        blocks.mass.emplace_back(da, db, I.Mmu[a][b]);
      if (pcd)
        blocks.convection_diffusion.emplace_back(da, db, I.F[a][b]);
      const double distance = I.M[a][b] - I.m[a] * I.m[b] / I.volume;
      if ((pcd || lsc) && stabilized)
        blocks.stabilization.emplace_back(da, db, c1 * distance);
      if (lsc && stabilized)
        blocks.commutator_stabilization.emplace_back(da, db, c2 * distance);
    }
}

/// Add to @p blocks the lumped masses and the pressure blocks of every element of the flow whose
/// mass balance is @p constraint.
void
addElementBlocks(const Problem & problem,
                 const MassEquation & constraint,
                 const ApproximationTerms & terms,
                 SaddlePointBlocks & blocks)
{
  const Mesh & mesh = problem.mesh();
  const int p = constraint.variable();
  blocks.lumped_mass = Eigen::VectorXd::Zero(problem.numDofs());
  ElementScratch scratch;
  scratch.context = flowContext(problem);
  for (Index e = 0; e < problem.numIntegratedElements(); ++e)
  {
    const Element & el = mesh.element(e);
    if (!constraint.activeOnBlock(el.block))
      continue;
    const auto & vb = problem.variable(p).blocks;
    if (!vb.empty() && !vb.count(el.block))
      continue;
    const ElementIntegrals integrals =
        integrateElement(problem, constraint, e, terms, scratch, blocks.refusal);
    addLumpedMass(problem, el, p, integrals, blocks.lumped_mass);
    addPressureBlocks(problem, el, p, integrals, terms, blocks);
  }
}

class FluidSystem : public FlowSystem
{
public:
  bool atRest(const Problem & problem) const override;
  SaddlePointBlocks saddlePointBlocks(const Problem & problem,
                                      const SparseMatrix & J,
                                      SchurApproximation approximation) const override;
  void resetPressureLevels() override
  {
    _levels_checked = false;
    _levels.clear();
  }
  bool holdPressureLevels(const Problem & problem, const SparseMatrix & J) override;
  std::vector<Index> heldPressureDofs() const override
  {
    std::vector<Index> held;
    for (const auto & level : _levels)
      if (level.dof >= 0)
        held.push_back(level.dof);
    return held;
  }
  void notePressureLevelFlux(const Problem & problem) override;
  void settlePressureLevels(Problem & problem, Vector & R, const ResidualAt & residual) override;
  void addStartProjection(std::function<double(Vector &)> project) override
  {
    _start_projections.push_back(std::move(project));
  }
  std::optional<double> projectInitialVelocity(Problem & problem) override;

private:
  /// Add to the F_p of @p blocks the Robin condition at the inflow of the flow whose mass balance
  /// is @p constraint (see SaddlePointBlocks.h).
  void addInflowRobin(const Problem & problem,
                      const MassEquation & constraint,
                      double theta,
                      SaddlePointBlocks & blocks) const;
  /// Shift each held pressure of @p U to a mean of zero over its fluid, and return the shifts.
  std::vector<double> shiftPressureLevels(const Problem & problem, Vector & U) const;
  /// Add to each held pressure of @p U a test shift of the size of the pressure in its equations:
  /// the larger of its largest value and, for each equation, the reactions of that equation
  /// @p forces over the size of the constant pressure in it.  Return the shifts.
  std::vector<double> testShiftPressureLevels(const Problem & problem,
                                              Vector & U,
                                              const std::vector<double> & forces) const;
  /// The size, in each equation, of the change that the pressure shifts @p shifts make: the sum
  /// over the held pressures of |shift| times the size of the constant pressure in that equation.
  std::vector<double> pressureLevelChange(const Problem & problem,
                                          const std::vector<double> & shifts) const;
  /// Warn once when the mass equation of a held unknown, at the converged @p R, carries a net
  /// flux of the boundary: data that an incompressible flow cannot satisfy.
  void checkPressureLevelBalance(const Problem & problem, const Vector & R);

  /// A pressure whose level is free: the variable, the unknown held for it (-1 on a process
  /// that does not have it), the norm of |J| times its constant mode, the largest norm of its
  /// mass residual at the start of a solve (the scale of the boundary flux), the weights of its
  /// mean (nodeWeights), and whether the net flux was reported.
  struct PressureLevel
  {
    int variable = -1;
    Index dof = -1;
    /// The size of the constant pressure in each equation, the norm of |J| 1_p over its unknowns.
    std::vector<double> scales;
    double flux = 0.0;
    Vector weight;
    bool warned = false;
  };
  bool _levels_checked = false;
  std::vector<PressureLevel> _levels;
  std::vector<std::function<double(Vector &)>> _start_projections;
};

bool
FluidSystem::atRest(const Problem & problem) const
{
  const int nv = problem.numVariables();
  std::vector<char> inertial(static_cast<std::size_t>(nv), 0);
  bool any = false;
  for (const auto & k : problem.kernels())
    if (const auto * advection = dynamic_cast<const AdvectiveKernel *>(k.get());
        advection && advection->isInertia())
    {
      inertial[static_cast<std::size_t>(k->variable())] = 1;
      any = true;
    }
  if (!any)
    return false;
  const std::vector<char> fixed = problem.constrainedDofs();
  const Vector & U = problem.solution();
  bool moving = false;
  for (Index i = 0; i < problem.numDofs() && !moving; ++i)
    moving = inertial[static_cast<std::size_t>(i % nv)] && !fixed[i] && U[i] != 0.0;
  return !problem.parallel().any(moving);
}

SaddlePointBlocks
FluidSystem::saddlePointBlocks(const Problem & problem,
                               const SparseMatrix & J,
                               SchurApproximation approximation) const
{
  const MassEquation * constraint = problem.flowConstraint();
  if (!constraint)
    throw InputError("The preconditioners pressure_mass, pcd and lsc are for pressure-velocity "
                     "flow, and this problem has no mass conservation equation "
                     "(mass_conservation). Use another preconditioner.");
  SaddlePointBlocks blocks;
  blocks.approximation = approximation;
  blocks.pressure_variable = problem.variable(constraint->variable()).name;
  blocks.constrained = problem.constrainedDofs();
  partitionDofs(problem, constraint->variable(), blocks);
  setSchurSign(problem, *constraint, J, blocks);
  ApproximationTerms terms;
  terms.approximation = approximation;
  terms.pcd = approximation == SchurApproximation::ConvectionDiffusion;
  terms.lsc = approximation == SchurApproximation::LeastSquaresCommutator;
  terms.dt = problem.linearTimeStep();
  terms.theta = problem.linearSteadyWeight();
  addElementBlocks(problem, *constraint, terms, blocks);
  if (terms.pcd)
    addInflowRobin(problem, *constraint, terms.theta, blocks);
  return blocks;
}

void
FluidSystem::addInflowRobin(const Problem & problem,
                            const MassEquation & constraint,
                            double theta,
                            SaddlePointBlocks & blocks) const
{
  // The Robin condition of F_p, mu dp/dn = rho (w . n) p, on the sides where
  // every velocity is prescribed: it adds theta rho max(0, -w . n) (p, q) on
  // the side, which is zero on a wall and acts at an inflow only.
  const Mesh & mesh = problem.mesh();
  const Vector & U = problem.solution();
  const int p = constraint.variable();
  QpContext context = flowContext(problem);
  FlowCoefficients c;
  std::vector<IntegrationPoint> points;
  MappedPoint side_geo;
  for (const Side & side : mesh.exteriorSides())
  {
    if (side.first >= problem.numIntegratedElements())
      continue;
    const Element & el = mesh.element(side.first);
    if (!constraint.activeOnBlock(el.block))
      continue;
    const auto & vb = problem.variable(p).blocks;
    if (!vb.empty() && !vb.count(el.block))
      continue;
    context.element = side.first;
    context.block = el.block;
    context.x = mesh.sideCentroid(side);
    constraint.flowCoefficients(context, c);
    bool prescribed = true;
    for (Index node : mesh.sideNodes(side))
      for (int d = 0; d < c.num_velocities; ++d)
        prescribed = prescribed && blocks.constrained[problem.dof(node, c.velocities[d])];
    if (!prescribed)
      continue;
    QuadratureSpec spec;
    spec.kind = QuadratureSpec::Kind::Gauss;
    spec.points = ReferenceElement::get(el.type).order() + 2;
    buildSidePoints(mesh, side, false, spec, points);
    const bool corner = problem.usesCornerShape(el, p);
    const int np = corner ? el.numCorners() : el.numNodes();
    for (const auto & ip : points)
    {
      mapPoint(mesh, el, ip.xi, side_geo);
      if (corner)
        mapCornerShape(el, side_geo);
      context.x = side_geo.x;
      constraint.flowCoefficients(context, c);
      double inflow = 0.0;
      for (int d = 0; d < c.num_velocities; ++d)
      {
        double w_d = 0.0;
        for (int a = 0; a < el.numNodes(); ++a)
          w_d += side_geo.N[a] * U[problem.dof(el.nodes[a], c.velocities[d])];
        inflow -= w_d * ip.area[d];
      }
      if (!(inflow > 0.0))
        continue;
      const double * N = corner ? side_geo.corner_shape : side_geo.N;
      const double w = theta * c.density * inflow * problem.coordFactor(side_geo.x);
      for (int a = 0; a < np; ++a)
        for (int b = 0; b < np; ++b)
        {
          const Index da = problem.dof(el.nodes[a], p), db = problem.dof(el.nodes[b], p);
          if (!blocks.constrained[da] && !blocks.constrained[db] && N[a] * N[b] != 0.0)
            blocks.convection_diffusion.emplace_back(da, db, w * N[a] * N[b]);
        }
    }
  }
}

bool
FluidSystem::holdPressureLevels(const Problem & problem, const SparseMatrix & J)
{
  if (_levels_checked)
    return false;
  const Mesh & mesh = problem.mesh();
  const Parallel & par = problem.parallel();
  const std::vector<char> fixed = problem.constrainedDofs();
  for (const auto & k : problem.kernels())
  {
    if (!dynamic_cast<const MassEquation *>(k.get()))
      continue;
    PressureLevel level;
    level.variable = k->variable();
    const int p = level.variable;
    std::vector<Index> free;
    for (Index n = 0; n < mesh.numNodes(); ++n)
      if (!fixed[problem.dof(n, p)])
        free.push_back(n);
    // The constant pressure, one at every free unknown, its image under J against that under |J|
    // for the scale, and the sum of the mass equations (J^T 1).
    Vector e = Vector::Zero(problem.numDofs());
    for (Index n : free)
      e[problem.dof(n, p)] = 1.0;
    const SparseMatrix size = J.cwiseAbs();
    Vector image = J * e, scale = size * e;
    Vector rows = J.transpose() * e, row_scale = size.transpose() * e;
    for (Vector * v : {&image, &scale, &rows, &row_scale})
      par.addAtNodes(v->data(), problem.numVariables());
    // Each equation is compared with itself, in its own units: the heat that a flow carries
    // depends on the pressure through the conserved velocity, in numbers that would hide the
    // momentum equations.
    level.scales = problem.equationNorms(scale);
    if (std::none_of(level.scales.begin(), level.scales.end(), [](double s) { return s > 0.0; }))
      continue;
    const std::vector<double> images = problem.equationNorms(image);
    bool changes = false;
    for (std::size_t v = 0; v < images.size(); ++v)
      changes = changes || images[v] > 1e-9 * level.scales[v];
    if (changes)
    {
      // The level changes the equations.  The mass equations of an enclosed flow sum to its
      // boundary flux whatever the solution (J^T 1 = 0, in the columns of every variable), so
      // they do not fix the level either.
      const std::vector<double> sums = problem.equationNorms(rows),
                                sum_scales = problem.equationNorms(row_scale);
      bool enclosed = true;
      for (std::size_t v = 0; v < sums.size(); ++v)
        enclosed = enclosed && sums[v] <= 1e-9 * sum_scales[v];
      if (enclosed)
      {
        _levels.clear();
        throw InputError(enclosedLevelMessage);
      }
      continue;
    }
    // Hold the unknown nearest the centre of the free unknowns: one in a corner slows the Krylov
    // solvers by about 40 % (the cavity with the pressure mass and pcd preconditioners).
    Point centre{0, 0, 0};
    for (Index n : free)
      centre = centre + mesh.node(n);
    const double count = par.sum(static_cast<double>(free.size()));
    for (int c = 0; c < 3; ++c)
      centre[c] = par.sum(centre[c]) / std::max(count, 1.0);
    std::vector<char> candidate(static_cast<std::size_t>(mesh.numNodes()), 0);
    for (Index n : free)
      candidate[static_cast<std::size_t>(n)] = par.globalDof(problem.dof(n, p)) >= 0 ? 1 : 0;
    double distance = 0.0;
    const Index held = problem.nearestNode(
        centre, [&](Index n) { return candidate[static_cast<std::size_t>(n)] != 0; }, &distance);
    if (!std::isfinite(distance))
      continue;
    if (held >= 0)
      level.dof = problem.dof(held, p);
    level.weight = problem.nodeWeights(p);
    _levels.push_back(std::move(level));
  }
  _levels_checked = true;
  return !_levels.empty();
}

std::vector<double>
FluidSystem::shiftPressureLevels(const Problem & problem, Vector & U) const
{
  const Parallel & par = problem.parallel();
  const Index nodes = problem.mesh().numNodes();
  const std::vector<char> & active = problem.activeDofs();
  std::vector<double> shifts;
  for (const auto & level : _levels)
  {
    double integral = 0.0;
    for (Index n = 0; n < nodes; ++n)
      integral += level.weight[n] * U[problem.dof(n, level.variable)];
    const double shift = -par.sum(integral) / par.sum(level.weight.sum());
    for (Index n = 0; n < nodes; ++n)
      if (active[problem.dof(n, level.variable)])
        U[problem.dof(n, level.variable)] += shift;
    shifts.push_back(shift);
  }
  problem.updateDependentDofs(U);
  return shifts;
}

std::vector<double>
FluidSystem::testShiftPressureLevels(const Problem & problem,
                                     Vector & U,
                                     const std::vector<double> & forces) const
{
  const Parallel & par = problem.parallel();
  const Index nodes = problem.mesh().numNodes();
  const std::vector<char> & active = problem.activeDofs();
  std::vector<double> shifts;
  for (const auto & level : _levels)
  {
    double largest = 0.0;
    for (Index n = 0; n < nodes; ++n)
      if (active[problem.dof(n, level.variable)])
        largest = std::max(largest, std::abs(U[problem.dof(n, level.variable)]));
    double shift = par.maximum(largest);
    for (std::size_t v = 0; v < forces.size(); ++v)
      if (level.scales[v] > 0.0)
        shift = std::max(shift, forces[v] / level.scales[v]);
    if (!(shift > 0.0))
      shift = 1.0;
    for (Index n = 0; n < nodes; ++n)
      if (active[problem.dof(n, level.variable)])
        U[problem.dof(n, level.variable)] += shift;
    shifts.push_back(shift);
  }
  problem.updateDependentDofs(U);
  return shifts;
}

std::vector<double>
FluidSystem::pressureLevelChange(const Problem & problem, const std::vector<double> & shifts) const
{
  std::vector<double> change(static_cast<std::size_t>(problem.numVariables()), 0.0);
  for (std::size_t l = 0; l < _levels.size(); ++l)
    for (std::size_t v = 0; v < change.size(); ++v)
      change[v] += std::abs(shifts[l]) * _levels[l].scales[v];
  return change;
}

void
FluidSystem::notePressureLevelFlux(const Problem & problem)
{
  const Parallel & par = problem.parallel();
  const std::vector<char> fixed = problem.constrainedDofs();
  const Vector & residual = problem.lastResidual();
  for (auto & level : _levels)
  {
    Vector flux = Vector::Zero(problem.numDofs());
    for (Index n = 0; n < problem.mesh().numNodes(); ++n)
      if (!fixed[problem.dof(n, level.variable)])
        flux[problem.dof(n, level.variable)] = residual[problem.dof(n, level.variable)];
    level.flux = std::max(level.flux, par.norm(flux));
  }
}

void
FluidSystem::checkPressureLevelBalance(const Problem & problem, const Vector & R)
{
  const Parallel & par = problem.parallel();
  for (auto & level : _levels)
  {
    const double imbalance = par.maximum(level.dof >= 0 ? std::abs(R[level.dof]) : 0.0);
    if (level.warned || !(level.flux > 0.0) || imbalance <= 1e-6 * level.flux)
      continue;
    level.warned = true;
    if (par.isRoot())
      std::fprintf(stderr,
                   "dualmesh: warning: the velocity prescribed on the boundary of the enclosed "
                   "flow of '%s' carries a net flux of %.3g (%.2g of its boundary flux). An "
                   "incompressible flow cannot take it in, so it goes to the mass equation of "
                   "the one pressure unknown that fixes the pressure level, and the pressure has "
                   "a spike there. Prescribe a velocity whose net flux is zero, or give the flow "
                   "an outlet.\n",
                   problem.variable(level.variable).name.c_str(),
                   imbalance,
                   imbalance / level.flux);
  }
}

void
FluidSystem::settlePressureLevels(Problem & problem, Vector & R, const ResidualAt & residual)
{
  // An enclosed flow reports its pressure with a mean of zero, and its reactions at the shifted
  // state.  A second shift, of the size of the pressure in the equations, checks that the
  // level is still free at the solution: a property that depends on the pressure makes it
  // matter, which the start at rest does not show.  The mass equations of an enclosed flow sum
  // to its boundary flux whatever the pressure, so every level then gives a solution of its
  // own, and the problem needs the level as data.  The last residual is that of the solution.
  if (_levels.empty())
    return;
  checkPressureLevelBalance(problem, problem.lastResidual());
  const std::vector<char> fixed = problem.constrainedDofs();
  Vector reactions = Vector::Zero(problem.numDofs());
  for (Index d = 0; d < problem.numDofs(); ++d)
    if (fixed[d])
      reactions[d] = problem.lastResidual()[d];
  const std::vector<double> forces = problem.equationNorms(reactions);
  // Each equation must stay as it is, to 1e-6 of the size that the shift would change it by
  // if the level mattered.  A shift too small to show that is replaced by a test shift.
  const Vector before = R;
  std::vector<double> shifts = shiftPressureLevels(problem, problem.solution());
  const bool shifted = std::any_of(shifts.begin(), shifts.end(), [](double s) { return s != 0.0; });
  if (shifted)
    residual(problem.solution(), R, false);
  Vector change = R - before;
  Vector test = problem.solution();
  const std::vector<double> test_shifts = testShiftPressureLevels(problem, test, forces);
  bool small = true;
  for (std::size_t l = 0; l < shifts.size(); ++l)
    small = small && std::abs(shifts[l]) < 1e-3 * test_shifts[l];
  if (small)
  {
    Vector at_test;
    residual(test, at_test, true);
    change = at_test - R;
    shifts = test_shifts;
  }
  const std::vector<double> changes = problem.equationNorms(change),
                            allowed = pressureLevelChange(problem, shifts);
  for (std::size_t v = 0; v < changes.size(); ++v)
    if (changes[v] > 1e-6 * allowed[v])
      throw InputError(enclosedLevelMessage);
}

std::optional<double>
FluidSystem::projectInitialVelocity(Problem & problem)
{
  problem.initialize();
  if (_start_projections.empty())
    return std::nullopt;
  problem.applyDirichlet(problem.solution(), 1.0);
  double correction = 0.0;
  for (const auto & project : _start_projections)
    correction = std::max(correction, project(problem.solution()));
  return correction;
}

} // namespace

std::unique_ptr<FlowSystem>
makeFluidSystem()
{
  return std::make_unique<FluidSystem>();
}

FlowCoefficients
flowCoefficientsAt(const Problem & problem,
                   const MassEquation & constraint,
                   const Point & x,
                   Index element)
{
  QpContext probe = flowContext(problem);
  probe.x = x;
  if (element >= 0)
  {
    probe.element = element;
    probe.block = problem.mesh().element(element).block;
  }
  FlowCoefficients c;
  constraint.flowCoefficients(probe, c);
  return c;
}

} // namespace dualmesh
