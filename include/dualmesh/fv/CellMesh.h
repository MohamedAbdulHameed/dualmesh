// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Cell-centred finite volume mesh (the "zero-thickness control volume"
// formulation of Reddy, Chapter 3).  Every element of the primal mesh is one
// control volume with its degree of freedom at the cell centroid, and every
// boundary face carries one extra degree of freedom whose control volume has
// zero thickness.  The face list is the owner/neighbour structure that every
// finite volume code is built on.
#pragma once

#include "dualmesh/mesh/Mesh.h"

#include <map>
#include <string>
#include <vector>

namespace dualmesh
{

/// One face of the cell-centred finite volume mesh.
struct CellFace
{
  Index owner = -1;    ///< cell the area vector points away from
  Index neighbor = -1; ///< the cell on the other side, or -1 on the boundary
  /// Entity index of the zero-thickness boundary node (boundary faces only).
  Index boundary_entity = -1;
  /// Cell beyond `owner` along the inward normal, used by the second-order
  /// boundary gradient; -1 when there is none.
  Index second_cell = -1;
  Point centroid{0, 0, 0};
  Point area{0, 0, 0}; ///< outward from `owner`; its norm is the face measure
  double measure = 0.0;
  Side side{-1, -1}; ///< (element, local side) as seen from `owner`
};

class CellMesh
{
public:
  explicit CellMesh(const Mesh & mesh);

  Index numCells() const { return _num_cells; }
  Index numBoundaryFaces() const { return static_cast<Index>(_boundary_faces.size()); }
  /// Degrees of freedom live on entities: cells first, then boundary nodes.
  Index numEntities() const { return _num_cells + numBoundaryFaces(); }
  bool isCell(Index entity) const { return entity < _num_cells; }

  const Point & entityPoint(Index entity) const { return _entity_point[entity]; }
  double cellVolume(Index cell) const { return _cell_volume[cell]; }
  int cellBlock(Index cell) const { return _cell_block[cell]; }

  const std::vector<CellFace> & faces() const { return _faces; }
  /// Indices into faces() of the faces of one cell.
  const std::vector<int> & cellFaces(Index cell) const { return _cell_faces[cell]; }
  /// Index into faces() of the face carrying boundary entity @p entity.
  int faceOfBoundaryEntity(Index entity) const { return _boundary_faces[entity - _num_cells]; }

  /// Coefficients of the least-squares gradient of cell @p cell:
  ///
  ///     grad u |_c = sum_i c_i (U_i - U_c) ,
  ///
  /// where the sum runs over the neighbouring cells and the boundary nodes of
  /// the cell.  The coefficients come from the weighted least-squares fit
  /// min_g sum_i |d_i|^{-2} (g . d_i - (U_i - U_c))^2, which reproduces the
  /// gradient of any linear field exactly on any mesh, however distorted.
  /// Green-Gauss gradients do not, because the face value they need sits at
  /// the face centroid and the interpolated one does not.
  const std::vector<std::pair<Index, Point>> & gradientStencil(Index cell) const
  {
    return _lsq[cell];
  }

  /// Entities (zero-thickness boundary nodes) of a side set.
  std::vector<Index> boundaryEntities(const Mesh & mesh, const std::string & name) const;
  /// Faces of a side set.
  std::vector<int> boundaryFaceIndices(const Mesh & mesh, const std::string & name) const;

private:
  Index _num_cells = 0;
  std::vector<Point> _entity_point;
  std::vector<double> _cell_volume;
  std::vector<int> _cell_block;
  std::vector<CellFace> _faces;
  std::vector<std::vector<int>> _cell_faces;
  std::vector<int> _boundary_faces; ///< per boundary entity: index into _faces
  std::vector<std::vector<std::pair<Index, Point>>> _lsq;
  /// (element, local side) -> index into _faces.
  std::map<std::pair<Index, int>, int> _side_to_face;
};

} // namespace dualmesh
