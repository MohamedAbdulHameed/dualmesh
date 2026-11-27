// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Solid mechanics module: the solid system (SolidSystem.h), which gives the linear solvers the
// rigid body modes of the solids.
#include "dualmesh/base/Problem.h"
#include "dualmesh/modules/SolidMechanics.h"

#include <algorithm>

namespace dualmesh
{

namespace
{

/// The rotations among the rigid body modes, by the pair (a, b) of the displacement components
/// they turn: u_a = -x_b and u_b = x_a, about z, x and y.
constexpr int kRotationPairs[3][2] = {{0, 1}, {1, 2}, {2, 0}};

/// The number of rotations among the rigid body modes of a group of @p d displacements in a
/// space of dimension @p dim: one in a plane, three in space, and none in coordinates that are
/// not Cartesian.
int
rotationCount(int d, int dim, bool cartesian)
{
  if (!cartesian)
    return 0;
  if (d == 2 && dim >= 2)
    return 1;
  return d == 3 ? 3 : 0;
}

class SolidMechanicsSystem : public SolidSystem
{
public:
  std::vector<std::vector<int>> displacementGroups(const Problem & problem) const override;
  Eigen::MatrixXd nearNullspace(const Problem & problem) const override;
};

std::vector<std::vector<int>>
SolidMechanicsSystem::displacementGroups(const Problem & problem) const
{
  const int nv = problem.numVariables();
  std::vector<std::vector<int>> groups;
  std::vector<char> grouped(nv, 0);
  for (const auto & material : problem.properties())
  {
    const auto & p = material->parameters();
    if (!p.has("displacements") || !p.isSet("displacements"))
      continue;
    std::vector<int> group;
    for (const auto & name : p.getStringList("displacements"))
      for (int v = 0; v < nv; ++v)
        if (problem.variable(v).name == name)
          group.push_back(v);
    if (group.empty() || std::any_of(group.begin(), group.end(), [&](int v) { return grouped[v]; }))
      continue;
    for (int v : group)
      grouped[v] = 1;
    groups.push_back(group);
  }
  return groups;
}

Eigen::MatrixXd
SolidMechanicsSystem::nearNullspace(const Problem & problem) const
{
  const Mesh & mesh = problem.mesh();
  const int nv = problem.numVariables();
  const Index entities = mesh.numNodes();
  const int dim = mesh.spaceDimension();
  // The displacement variables of the elastic property objects, in component order.
  const std::vector<std::vector<int>> groups = displacementGroups(problem);
  std::vector<char> grouped(nv, 0);
  for (const auto & group : groups)
    for (int v : group)
      grouped[v] = 1;
  const bool cartesian = problem.coordinateSystem() == CoordinateSystem::Cartesian;
  // A group has its translations and its rotations, and any other variable a constant.
  int columns = 0;
  for (const auto & group : groups)
  {
    const int d = static_cast<int>(group.size());
    columns += d + rotationCount(d, dim, cartesian);
  }
  for (int v = 0; v < nv; ++v)
    if (!grouped[v])
      ++columns;
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(problem.numDofs(), columns);
  // Rotations about the centroid of the entities, for a well-conditioned basis.
  Point centre{0.0, 0.0, 0.0};
  for (Index e = 0; e < entities; ++e)
    centre = centre + mesh.node(e);
  centre = (1.0 / std::max<Index>(entities, 1)) * centre;
  int column = 0;
  for (const auto & group : groups)
  {
    const int d = static_cast<int>(group.size());
    for (int c = 0; c < d; ++c, ++column)
      for (Index e = 0; e < entities; ++e)
        B(problem.dof(e, group[c]), column) = 1.0;
    const int rotations = rotationCount(d, dim, cartesian);
    for (Index e = 0; e < entities && rotations > 0; ++e)
    {
      const Point x = mesh.node(e) - centre;
      for (int k = 0; k < rotations; ++k)
      {
        const auto [a, b] = kRotationPairs[k];
        B(problem.dof(e, group[a]), column + k) = -x[b];
        B(problem.dof(e, group[b]), column + k) = x[a];
      }
    }
    column += rotations;
  }
  for (int v = 0; v < nv; ++v)
    if (!grouped[v])
    {
      for (Index e = 0; e < entities; ++e)
        B(problem.dof(e, v), column) = 1.0;
      ++column;
    }
  return B;
}

} // namespace

std::shared_ptr<const SolidSystem>
makeSolidMechanicsSystem()
{
  return std::make_shared<SolidMechanicsSystem>();
}

} // namespace dualmesh
