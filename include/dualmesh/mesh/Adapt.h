// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Local, conforming mesh refinement.
//
// Why conforming?  A hanging node is a node of one element that lies in the
// interior of a face of its neighbour.  Finite element codes cope with these by
// constraining the hanging degree of freedom to the values of the nodes at the
// ends of the face it hangs on, so the interpolation stays continuous.  That
// device does not work for the dual mesh method, because the dual mesh is not
// an interpolation device: the control domain of a node is a region of space,
// and every point of the domain has to belong to exactly one control domain for
// the balance law to hold.  A hanging node owns a control domain on its own
// side of the face and nothing on the other, so the control domains stop
// tiling the domain, flux leaves one control domain without entering another,
// and the discretisation is no longer conservative - which is the one property
// the method exists to guarantee.
//
// dualmesh therefore refines conformingly, by bisecting the longest edge of a
// marked triangle and propagating the bisection to the neighbour that shares
// that edge until no hanging node is left.  This is the algorithm of
//
//   M.-C. Rivara, "Algorithms for refining triangular grids suitable for
//   adaptive and multigrid techniques", International Journal for Numerical
//   Methods in Engineering 20 (1984) 745-756.
//
// Rivara proves that the propagation terminates (each step bisects a strictly
// longer edge, and there are finitely many) and that the smallest angle of the
// refined mesh is bounded below by half the smallest angle of the original
// mesh, so repeated refinement cannot degenerate the elements.
//
// The algorithm is specific to triangles.  Quadrilateral, tetrahedral and
// hexahedral meshes are refined uniformly with Mesh::refined() instead; see the
// documentation for what is and is not available.
#pragma once

#include "dualmesh/mesh/Mesh.h"

#include <vector>

namespace dualmesh
{

/// Refine the elements flagged in @p marked (one entry per element, non-zero
/// to refine) by longest-edge bisection, and return the refined mesh.
///
/// More elements than those marked are bisected, because the mesh is kept
/// conforming.  Side sets and node sets are carried over: a child side that is
/// exterior and lies inside a parent side of a set joins that set.
///
/// @param parents optional output, one entry per element of the refined mesh,
///        holding the index in @p mesh of the element it came from.  This is
///        what a field defined per element is transferred with.
Mesh refineMarked(const Mesh & mesh,
                  const std::vector<char> & marked,
                  std::vector<Index> * parents = nullptr);

} // namespace dualmesh
