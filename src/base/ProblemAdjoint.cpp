// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Adjoint sensitivities: the derivative of an objective functional of the
// solution with respect to named parameters, at the cost of one adjoint
// solve plus one residual assembly per parameter.
#include "dualmesh/base/Console.h"
#include "dualmesh/base/Problem.h"
#include "dualmesh/core/Function.h"

#include <cmath>
#include <stdexcept>

namespace dualmesh
{

void
Problem::addParameter(const std::string & name, double value)
{
  if (_functions.count(name))
    throw InputError("add_parameter: the name '" + name +
                     "' is already a registered function, and replacing it would leave the "
                     "objects that resolved it reading the old one. Register a parameter under "
                     "a new name.");
  _functions[name] = std::make_shared<SettableFunction>(value);
  _initialized = false;
}

namespace
{
/// The settable constant of a registered name, or the error that explains
/// what a sensitivity parameter must be.
std::shared_ptr<SettableFunction>
settableParameter(const std::map<std::string, FunctionPtr> & functions, const std::string & name)
{
  const auto it = functions.find(name);
  if (it == functions.end())
    throw InputError("sensitivities: '" + name +
                     "' is not a registered function. A sensitivity parameter is a settable "
                     "constant, registered with add_parameter(name, value) and used by its "
                     "name where a coefficient, a source or a boundary value takes a function.");
  auto parameter = std::dynamic_pointer_cast<SettableFunction>(it->second);
  if (!parameter)
    throw InputError("sensitivities: the function '" + name +
                     "' is not a settable constant. Register it with add_parameter(name, "
                     "value) so that its value can be varied for the differentiation.");
  return parameter;
}
} // namespace

void
Problem::setParameter(const std::string & name, double value)
{
  settableParameter(_functions, name)->set(value);
}

double
Problem::parameterValue(const std::string & name) const
{
  return settableParameter(_functions, name)->get();
}

Vector
Problem::objectiveGradient(const Objective & objective) const
{
  const_cast<Problem *>(this)->initialize();
  if (!hasVariable(objective.variable))
    throw InputError("sensitivities: the objective names the variable '" + objective.variable +
                     "', which the problem does not have.");
  const int v = variableIndex(objective.variable);
  Vector gradient = Vector::Zero(numDofs());
  const bool corner = _mixed_order && _full_order_rank[v] < 0;

  if (objective.boundary.empty())
  {
    // The integral of the variable over the domain: g = int u dV, so
    // dg/dU at the node of a shape function N is int N dV, the same vector
    // a one-point source assembled on that variable would produce.
    if (_method == Method::FiniteVolumeCell)
    {
      // The unknowns are the cell values: g = sum_c u_c |V_c|.
      const std::vector<double> volumes = elementIntegrals("");
      const Index cells = cellMesh().numCells();
      for (Index c = 0; c < cells && c < static_cast<Index>(volumes.size()); ++c)
        gradient[dof(c, v)] = volumes[c];
      return gradient;
    }
    QuadratureSpec spec;
    spec.kind = QuadratureSpec::Kind::Gauss;
    std::vector<IntegrationPoint> points;
    std::vector<MappedPoint> mapped;
    for (Index e = 0; e < _mesh->numElements(); ++e)
    {
      const Element & el = _mesh->element(e);
      const auto & blocks = _vars[v].blocks;
      if (!blocks.empty() && !blocks.count(el.block))
        continue;
      const auto & ref = ReferenceElement::get(el.type);
      spec.points = ref.order() + 2;
      buildElementPoints(*_mesh, e, false, PointSet::Volume, spec, points, &mapped);
      const int count = corner ? el.numCorners() : el.numNodes();
      for (std::size_t q = 0; q < points.size(); ++q)
      {
        if (corner)
          mapCornerShape(el, mapped[q]);
        const double w = points[q].weight * coordFactor(mapped[q].x);
        for (int a = 0; a < count; ++a)
          gradient[dof(el.nodes[a], v)] +=
              w * (corner ? mapped[q].corner_shape[a] : mapped[q].N[a]);
      }
    }
    return gradient;
  }

  // The integral of the variable over a side set.
  if (_method == Method::FiniteVolumeCell)
    throw InputError("sensitivities: an objective over the boundary of the cell-centred finite "
                     "volume method is not available yet; integrate over the domain, or use "
                     "fem, dmcdm or hfvm for the boundary.");
  if (!_mesh->sidesets().count(objective.boundary))
    throw InputError("sensitivities: the objective names the boundary '" + objective.boundary +
                     "', which the mesh does not have.");
  QuadratureSpec spec;
  spec.kind = QuadratureSpec::Kind::Gauss;
  spec.points = 3;
  std::vector<IntegrationPoint> points;
  MappedPoint geo;
  for (const Side & side : _mesh->sideset(objective.boundary))
  {
    const Element & el = _mesh->element(side.first);
    const auto & blocks = _vars[v].blocks;
    if (!blocks.empty() && !blocks.count(el.block))
      continue;
    const auto & ref = ReferenceElement::get(el.type);
    spec.points = ref.order() + 2;
    buildSidePoints(*_mesh, side, false, spec, points);
    const int count = corner ? el.numCorners() : el.numNodes();
    for (const auto & ip : points)
    {
      mapPoint(*_mesh, el, ip.xi, geo);
      if (corner)
        mapCornerShape(el, geo);
      const double w = ip.weight * coordFactor(geo.x);
      for (int a = 0; a < count; ++a)
        gradient[dof(el.nodes[a], v)] += w * (corner ? geo.corner_shape[a] : geo.N[a]);
    }
  }
  return gradient;
}

double
Problem::objectiveValue(const Objective & objective) const
{
  return objectiveGradient(objective).dot(_U);
}

std::map<std::string, double>
Problem::sensitivities(const Objective & objective,
                       const std::vector<std::string> & parameters,
                       const SolverOptions & options)
{
  initialize();
  if (!_primary_of.empty())
    throw InputError("sensitivities: a problem with periodic boundary conditions is not "
                     "differentiated yet; the constraints tie the degrees of freedom of two "
                     "boundaries in a way the adjoint does not follow.");
  if (_state_record > 0)
    throw InputError("sensitivities: a problem with a material history (creep, a stateful property "
                     "object) is not differentiated yet; its solution depends on the whole "
                     "history, which the adjoint of the last state does not follow.");
  if (parameters.empty())
    throw InputError("sensitivities: no parameter was named. Give the names of settable "
                     "constants, registered with add_parameter(name, value).");
  // Every parameter must be a settable constant before any work is done.
  std::vector<std::shared_ptr<SettableFunction>> vars;
  for (const auto & name : parameters)
    vars.push_back(settableParameter(_functions, name));

  // The primal state: the residual and the exact Jacobian at the converged
  // solution, with the prescribed rows and columns of the identity.
  AssemblyOptions ao;
  ao.include_time_kernels = false;
  ao.theta = 1.0;
  ao.load_factor = 1.0;
  Vector R;
  SparseMatrix Jraw;
  assemble(_U, ao, R, &Jraw);
  Vector Rmod = R;
  SparseMatrix Jmod = Jraw;
  dirichletRows(_U, 1.0, Rmod, &Jmod);

  // The adjoint: J^T lambda = dg/dU.  Its matrix is that of a steady solve, and pcd, whose
  // F_p carries the convection of the forward equations, does not fit the transposed system.
  if (options.preconditioner == "pcd")
    throw InputError("sensitivities: the preconditioner 'PCD' builds the convection of the "
                     "forward equations, which the adjoint (transposed) system does not have. Use "
                     "'LSC' or 'pressure_mass', which take their operators from the matrix.");
  _linear_dt = 0.0;
  _linear_theta = 1.0;
  const Vector gradient = objectiveGradient(objective);
  const SparseMatrix transposed = Jmod.transpose();
  const Vector lambda = linearSolve(transposed, gradient, options, nullptr);

  // A parameter enters the solution through the equations of the free
  // degrees of freedom and, when it moves a prescribed value, through U
  // itself at the nodes that carry it:
  //
  //   dg/dp = dg/dU . dU/dp|prescribed - lambda . ( dR/dp + J dU/dp|prescribed ) ,
  //
  // with the second sum over the free rows only (the prescribed rows hold no
  // equation of the solution; their lambda entries carry dg/dU there, which
  // the first term has already counted).
  const std::vector<char> fixed = constrainedDofs();
  std::map<std::string, double> out;
  for (std::size_t k = 0; k < parameters.size(); ++k)
  {
    const std::string & name = parameters[k];
    const double p0 = vars[k]->get();
    const double h = 1.0e-6 * std::max(1.0, std::abs(p0));

    vars[k]->set(p0 + h);
    Vector Rp;
    assemble(_U, ao, Rp, nullptr);
    dirichletRows(_U, 1.0, Rp, nullptr);
    Vector Up(_U);
    applyDirichlet(Up, 1.0);

    vars[k]->set(p0 - h);
    Vector Rm;
    assemble(_U, ao, Rm, nullptr);
    dirichletRows(_U, 1.0, Rm, nullptr);
    Vector Um(_U);
    applyDirichlet(Um, 1.0);
    vars[k]->set(p0);

    const Vector dRdp = (Rp - Rm) / (2.0 * h);
    const Vector dUdp = (Up - Um) / (2.0 * h);
    const Vector JdU = Jraw * dUdp;
    double total = gradient.dot(dUdp); // dU/dp is zero on the free rows
    for (Index i = 0; i < numDofs(); ++i)
      if (!fixed[i])
        total -= lambda[i] * (dRdp[i] + JdU[i]);
    out[name] = total;
  }
  return out;
}

} // namespace dualmesh
