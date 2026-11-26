// SPDX-License-Identifier: LGPL-2.1-or-later
#ifdef DUALMESH_HAVE_PETSC

#include "Plex.h"
#include "dualmesh/core/InputParameters.h"
#include "dualmesh/linalg/PetscSolver.h"

#include <vector>

#define DM_PETSC(call) petsc::check((call), #call)

namespace dualmesh
{
namespace plex
{

namespace
{
DMPolytopeType
polytope(ElementType corner_type)
{
  switch (corner_type)
  {
  case ElementType::Edge2:
    return DM_POLYTOPE_SEGMENT;
  case ElementType::Tri3:
    return DM_POLYTOPE_TRIANGLE;
  case ElementType::Quad4:
    return DM_POLYTOPE_QUADRILATERAL;
  case ElementType::Tet4:
    return DM_POLYTOPE_TETRAHEDRON;
  case ElementType::Hex8:
    return DM_POLYTOPE_HEXAHEDRON;
  case ElementType::Wedge6:
    return DM_POLYTOPE_TRI_PRISM;
  case ElementType::Pyramid5:
    return DM_POLYTOPE_PYRAMID;
  default:
    throw std::logic_error("plex: the corner type of an element is a linear type.");
  }
}
} // namespace

DM
cornerTopology(const Mesh & mesh, MPI_Comm comm, bool holds)
{
  petsc::initialize();
  const PetscInt num_cells = holds ? static_cast<PetscInt>(mesh.numElements()) : 0;
  // The vertices are the corner nodes, numbered after the cells.
  std::vector<PetscInt> vertex_of(holds ? mesh.numNodes() : 0, -1);
  PetscInt num_vertices = 0;
  for (PetscInt e = 0; e < num_cells; ++e)
  {
    const Element & el = mesh.element(e);
    for (int k = 0; k < el.numCorners(); ++k)
      if (vertex_of[el.nodes[k]] < 0)
        vertex_of[el.nodes[k]] = num_cells + num_vertices++;
  }
  DM dm;
  DM_PETSC(DMCreate(comm, &dm));
  DM_PETSC(DMSetType(dm, DMPLEX));
  DM_PETSC(DMSetDimension(dm, mesh.dimension()));
  DM_PETSC(DMPlexSetChart(dm, 0, num_cells + num_vertices));
  for (PetscInt e = 0; e < num_cells; ++e)
    DM_PETSC(DMPlexSetConeSize(dm, e, mesh.element(e).numCorners()));
  DM_PETSC(DMSetUp(dm));
  std::vector<PetscInt> cone;
  for (PetscInt e = 0; e < num_cells; ++e)
  {
    const Element & el = mesh.element(e);
    cone.assign(static_cast<std::size_t>(el.numCorners()), 0);
    for (int k = 0; k < el.numCorners(); ++k)
      cone[k] = vertex_of[el.nodes[k]];
    DM_PETSC(DMPlexSetCone(dm, e, cone.data()));
  }
  DM_PETSC(DMPlexSymmetrize(dm));
  DM_PETSC(DMPlexStratify(dm));
  for (PetscInt e = 0; e < num_cells; ++e)
    DM_PETSC(DMPlexSetCellType(dm, e, polytope(mesh.element(e).cornerType())));
  for (PetscInt v = num_cells; v < num_cells + num_vertices; ++v)
    DM_PETSC(DMPlexSetCellType(dm, v, DM_POLYTOPE_POINT));
  return dm;
}

std::string
partitionerType(const std::string & name)
{
  std::vector<std::string> available;
#ifdef PETSC_HAVE_PTSCOTCH
  available.push_back("ptscotch");
#endif
#ifdef PETSC_HAVE_PARMETIS
  available.push_back("parmetis");
#endif
  available.push_back("simple");
  if (name == "automatic")
    return available.front();
  for (const auto & type : available)
    if (type == name)
      return type;
  std::string list = "automatic";
  for (const auto & type : available)
    list += ", " + type;
  throw InputError("Unknown or unavailable partitioner '" + name + "'. This build has: " + list +
                   ".");
}

} // namespace plex
} // namespace dualmesh

#endif
