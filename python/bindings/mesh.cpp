// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Bindings of the mesh, the element types, the mesh generators and the mesh hierarchies of
// adaptive refinement.
#include "dualmesh/mesh/Mesh.h"
#include "bindings.h"
#include "dualmesh/fe/Assembly.h"
#include "dualmesh/mesh/Bisection.h"
#include "dualmesh/mesh/Subdivision.h"

#include <optional>
#include <set>
#include <string>
#include <vector>

namespace dualmesh::bindings
{
namespace
{

using MeshClass = py::class_<Mesh, std::shared_ptr<Mesh>>;

/// The construction of a mesh, its dimensions, and its nodes and elements.
void
defineNodesAndElements(MeshClass & cls)
{
  cls.def(py::init(
              [](int dimension, int space_dimension)
              {
                Mesh m(dimension);
                if (space_dimension)
                  m.setSpaceDimension(space_dimension);
                return m;
              }),
          py::arg("dimension") = 2,
          py::arg("space_dimension") = 0,
          "An empty mesh whose elements have the given dimension (1, 2 or 3).  "
          "space_dimension places it in a space of higher dimension: 3 for a surface "
          "(a shell) or a curve in space; by default the space has the dimension of the "
          "elements.")
      .def_property_readonly(
          "dimension", &Mesh::dimension, "The dimension of the elements, 1, 2 or 3.")
      .def_property("space_dimension",
                    &Mesh::spaceDimension,
                    &Mesh::setSpaceDimension,
                    "The dimension of the space the nodes live in: 3 for a surface in space, "
                    "otherwise the dimension of the elements.")
      .def_property_readonly("num_nodes", &Mesh::numNodes, "The number of nodes.")
      .def_property_readonly("num_elements", &Mesh::numElements, "The number of elements.")
      .def(
          "element_measures",
          [](const Mesh & mesh)
          {
            std::vector<double> out(mesh.numElements());
            for (Index e = 0; e < mesh.numElements(); ++e)
              out[e] = elementMeasure(mesh, e);
            return out;
          },
          "The length, area or volume of every element.")
      .def("add_node",
           &Mesh::addNode,
           py::arg("point"),
           "Add a node at (x, y, z) and return its index.")
      .def(
          "add_element",
          [](Mesh & mesh, const std::string & type, const std::vector<Index> & nodes, int block)
          { return mesh.addElement(elementTypeFromName(type), nodes, block); },
          py::arg("element_type"),
          py::arg("nodes"),
          py::arg("block") = 0,
          "Add an element of the named type (for instance 'Quad4') on the given nodes, "
          "numbered as VTK numbers them, in element block 'block'. Returns its index.")
      .def("node", &Mesh::node, py::arg("index"), "The coordinates (x, y, z) of a node.")
      .def(
          "points",
          [](const Mesh & mesh)
          {
            py::array_t<double> a(
                {static_cast<py::ssize_t>(mesh.numNodes()), static_cast<py::ssize_t>(3)});
            auto r = a.mutable_unchecked<2>();
            for (Index i = 0; i < mesh.numNodes(); ++i)
              for (int d = 0; d < 3; ++d)
                r(i, d) = mesh.node(i)[d];
            return a;
          },
          "The coordinates of every node, as an array of shape (num_nodes, 3).")
      .def(
          "cells",
          [](const Mesh & mesh)
          {
            py::list out;
            for (const auto & el : mesh.elements())
            {
              py::list conn;
              for (int k = 0; k < el.numNodes(); ++k)
                conn.append(el.nodes[k]);
              out.append(py::make_tuple(elementTypeName(el.type), conn, el.block));
            }
            return out;
          },
          "Every element as a tuple (type, node indices, block).")
      .def(
          "element_nodes",
          [](const Mesh & mesh, Index e)
          {
            std::vector<Index> out;
            for (int k = 0; k < mesh.element(e).numNodes(); ++k)
              out.push_back(mesh.element(e).nodes[k]);
            return out;
          },
          py::arg("element"),
          "The node indices of an element.")
      .def(
          "element_type",
          [](const Mesh & mesh, Index e) { return elementTypeName(mesh.element(e).type); },
          py::arg("element"),
          "The type of an element, for instance 'Hex8'.")
      .def(
          "element_types",
          [](const Mesh & mesh)
          {
            std::set<ElementType> types;
            for (const auto & el : mesh.elements())
              types.insert(el.type);
            std::set<std::string> out;
            for (ElementType t : types)
              out.insert(elementTypeName(t));
            return out;
          },
          "The names of the element types the mesh holds, for instance {'Quad4', 'Tri3'}.");
}

/// The element blocks, the side sets and the node sets of a mesh.
void
defineBlocksAndSets(MeshClass & cls)
{
  cls.def("set_block_name",
          &Mesh::setBlockName,
          py::arg("block"),
          py::arg("name"),
          "Give an element block a name, by which objects can then be restricted to it.")
      .def("block_ids", &Mesh::blockIds, "The ids of the element blocks present.")
      .def("block_names", &Mesh::blockNames, "The names of the named element blocks, by block id.")
      .def("add_sideset_from_faces",
           &Mesh::addSidesetFromFaces,
           py::arg("name"),
           py::arg("faces"),
           "Add a side set from faces given by their node indices (a list of node lists); "
           "each face is matched to the element side with those nodes.")
      .def("add_nodeset",
           &Mesh::addNodeset,
           py::arg("name"),
           py::arg("nodes"),
           "Add a node set from a list of node indices.")
      .def("alias_sideset",
           &Mesh::aliasSideset,
           py::arg("name"),
           py::arg("alias"),
           "Make 'alias' a second name for the existing side set 'name'.")
      .def(
          "add_sideset_by_predicate",
          [](Mesh & mesh, const std::string & name, const py::function & f)
          {
            mesh.addSidesetByPredicate(name,
                                       [f](const Point & p)
                                       {
                                         py::gil_scoped_acquire gil;
                                         return py::cast<bool>(f(p[0], p[1], p[2]));
                                       });
          },
          py::arg("name"),
          py::arg("predicate"),
          "Add a side set of the boundary sides whose centroid satisfies "
          "predicate(x, y, z).")
      .def(
          "add_nodeset_by_predicate",
          [](Mesh & mesh, const std::string & name, const py::function & f)
          {
            mesh.addNodesetByPredicate(name,
                                       [f](const Point & p)
                                       {
                                         py::gil_scoped_acquire gil;
                                         return py::cast<bool>(f(p[0], p[1], p[2]));
                                       });
          },
          py::arg("name"),
          py::arg("predicate"),
          "Add a node set of the nodes that satisfy predicate(x, y, z).");
}

/// The transformations of a mesh: moved nodes, orientation, refinement and promotion to second
/// order.
void
defineTransformations(MeshClass & cls)
{
  cls.def(
         "transform_nodes",
         [](Mesh & mesh, const py::function & f)
         {
           mesh.transformNodes(
               [f](const Point & p)
               {
                 py::gil_scoped_acquire gil;
                 auto r = py::cast<std::vector<double>>(f(p[0], p[1], p[2]));
                 Point out{0, 0, 0};
                 for (std::size_t i = 0; i < r.size() && i < 3; ++i)
                   out[i] = r[i];
                 return out;
               });
         },
         py::arg("map"),
         "Move every node: map(x, y, z) returns the new coordinates. Promote a mesh to "
         "second order before mapping it onto a curved domain, so that the mid-side nodes "
         "land on the curve.")
      .def(
          "fix_orientation",
          [](Mesh & mesh, std::optional<py::array_t<double>> preferred)
          {
            std::vector<Point> normals;
            if (preferred)
            {
              const auto a = preferred->unchecked<2>();
              if (a.shape(1) != 3)
                throw InputError("preferred must have shape (num_elements, 3).");
              for (py::ssize_t i = 0; i < a.shape(0); ++i)
                normals.push_back({a(i, 0), a(i, 1), a(i, 2)});
            }
            mesh.fixOrientation(normals);
          },
          py::arg("preferred") = py::none(),
          "Wind every element consistently.  A mesh that fills its space gets positive "
          "Jacobian determinants.  A surface in space is oriented so that the normals of "
          "neighbours agree (a surface with one side is refused); each connected part then "
          "takes the side most of ``preferred`` (one normal per element, shape "
          "(num_elements, 3)) agree with, or, without it, points outward when closed and "
          "has the largest component of its summed normal positive when open.  The readers "
          "call it.")
      .def("refined",
           &Mesh::refined,
           "A uniformly refined copy: every element is split into 2^dim children (a pyramid into "
           "six pyramids and four tetrahedra), and the "
           "side sets and node sets are carried over. Only linear meshes can be refined; "
           "refine first and promote to second order afterwards.")
      .def("second_order",
           &Mesh::secondOrder,
           py::arg("serendipity") = false,
           "A copy of this mesh with quadratic elements. Edge2 becomes Edge3, Tri3 becomes "
           "Tri6, Quad4 becomes Quad9, Tet4 becomes Tet10 and Hex8 becomes Hex27; with "
           "serendipity=True, Quad4 becomes Quad8 and Hex8 becomes Hex20 instead, which work "
           "with the finite element method only. The corner "
           "nodes keep their numbers and positions, so the geometry of the domain does not "
           "change; the added nodes sit at the midpoints of the edges, faces and cells they "
           "belong to. Side sets carry over unchanged and node sets gain the added nodes that "
           "lie between members.");
}

/// The boundary of a mesh and its side sets, and the summary of a mesh.
void
defineBoundaryQueries(MeshClass & cls)
{
  cls.def("add_bounding_box_sidesets",
          &Mesh::addBoundingBoxSidesets,
          py::arg("tolerance") = 1e-10,
          "Add side sets named left, right, bottom, top (and front, back in three "
          "dimensions) from the boundary sides that lie on the faces of the bounding box.")
      .def("boundary_nodes",
           &Mesh::boundaryNodes,
           py::arg("name"),
           "The nodes of a side set or a node set.")
      .def(
          "sideset",
          [](const Mesh & mesh, const std::string & name)
          {
            std::vector<std::pair<Index, int>> out;
            for (const Side & side : mesh.sideset(name))
              out.emplace_back(side.first, side.second);
            return out;
          },
          py::arg("name"),
          "The sides of a side set, as (element, local side) pairs.")
      .def(
          "exterior_sides",
          [](const Mesh & mesh)
          {
            std::vector<std::pair<Index, int>> out;
            for (const Side & side : mesh.exteriorSides())
              out.emplace_back(side.first, side.second);
            return out;
          },
          "All the sides on the exterior boundary of the domain, as (element, local side) pairs.")
      .def(
          "side_nodes",
          [](const Mesh & mesh, const std::vector<std::pair<Index, int>> & sides)
          {
            std::vector<std::vector<Index>> out;
            out.reserve(sides.size());
            for (const auto & side : sides)
              out.push_back(mesh.sideNodes({side.first, side.second}));
            return out;
          },
          py::arg("sides"),
          "The node indices of each side in a list of (element, local side) pairs. The corner "
          "nodes come first, in the order of the side's own element type.")
      .def(
          "sideset_polar_moment",
          [](const Mesh & mesh,
             const std::vector<std::string> & names,
             const Point & axis,
             const Point & center) { return sidesetPolarMoment(mesh, names, axis, center); },
          py::arg("names"),
          py::arg("axis"),
          py::arg("center"),
          "The polar moment int |k x (x - c)|^2 dA of the named side sets about the axis through "
          "'center' along the unit vector 'axis'.")
      .def(
          "sideset_measure",
          [](const Mesh & mesh, const std::vector<std::string> & names)
          { return sidesetMeasure(mesh, names); },
          py::arg("names"),
          "The total length (in two dimensions) or area (in three) of the named side sets, "
          "integrated with the isoparametric map of the elements. A side in several of the sets "
          "is counted once.")
      .def(
          "sideset_names",
          [](const Mesh & mesh)
          {
            std::vector<std::string> out;
            for (const auto & [n, s] : mesh.sidesets())
              out.push_back(n);
            return out;
          },
          "The names of the side sets.")
      .def(
          "nodeset_names",
          [](const Mesh & mesh)
          {
            std::vector<std::string> out;
            for (const auto & [n, s] : mesh.nodesets())
              out.push_back(n);
            return out;
          },
          "The names of the node sets.")
      .def("element_centroid",
           &Mesh::elementCentroid,
           py::arg("element"),
           "The mean of the coordinates of an element's nodes.")
      .def(
          "summary",
          &Mesh::summary,
          "A short report of the mesh: dimensions, counts, element types, blocks, side sets and node sets.")
      .def("__repr__", [](const Mesh & mesh) { return "<dualmesh.Mesh " + mesh.summary() + ">"; });
}

/// The element types, by name.
void
bindElementTypes(py::module_ & m)
{
  m.def(
      "element_dimension",
      [](const std::string & element_type)
      { return elementDimension(elementTypeFromName(element_type)); },
      py::arg("element_type"),
      "The dimension of an element type, 1, 2 or 3.");
  m.def(
      "element_is_quadratic",
      [](const std::string & element_type)
      { return elementIsQuadratic(elementTypeFromName(element_type)); },
      py::arg("element_type"),
      "Whether an element type carries mid-edge, mid-face or interior nodes.");
  m.def(
      "element_is_serendipity",
      [](const std::string & element_type)
      { return elementIsSerendipity(elementTypeFromName(element_type)); },
      py::arg("element_type"),
      "Whether an element type is a serendipity element (Quad8, Hex20).");
  m.def(
      "element_corner_type",
      [](const std::string & element_type)
      { return elementTypeName(elementCornerType(elementTypeFromName(element_type))); },
      py::arg("element_type"),
      "The linear element type with the same corners, for instance 'Quad4' for 'Quad9'.");
}

/// The mesh generators.
void
bindGenerators(py::module_ & m)
{
  m.def("generate_line_mesh", &generateLineMesh, py::arg("x"));
  m.def("generate_rectangle_mesh",
        &generateRectangleMesh,
        py::arg("x"),
        py::arg("y"),
        py::arg("element_type") = "Quad4",
        py::arg("diagonal") = "alternate");
  m.def("generate_box_mesh",
        &generateBoxMesh,
        py::arg("x"),
        py::arg("y"),
        py::arg("z"),
        py::arg("element_type") = "Hex8");
}

/// The mesh hierarchies of adaptive refinement: subdivision and bisection.
void
bindHierarchies(py::module_ & m)
{
  py::class_<MeshHierarchy>(m, "MeshHierarchy")
      .def(
          "mesh", [](const MeshHierarchy & h) { return h.mesh(); }, "The current mesh.")
      .def(
          "refine",
          [](MeshHierarchy & h, const std::vector<int> & marked, int max_level)
          { return h.refine(std::vector<char>(marked.begin(), marked.end()), max_level); },
          py::arg("marked"),
          py::arg("max_level"),
          "Refine the marked elements below max_level refinements, and those that the "
          "conformity or the balance needs; return whether the mesh changed.")
      .def(
          "coarsen",
          [](MeshHierarchy & h, const std::vector<int> & marked)
          { return h.coarsen(std::vector<char>(marked.begin(), marked.end())); },
          py::arg("marked"),
          "Coarsen every family whose elements are all marked; return whether the mesh changed.")
      .def(
          "adapt",
          [](MeshHierarchy & h,
             const std::vector<int> & refine,
             const std::vector<int> & coarsen,
             int max_level)
          {
            return h.adapt(std::vector<char>(refine.begin(), refine.end()),
                           std::vector<char>(coarsen.begin(), coarsen.end()),
                           max_level);
          },
          py::arg("refine"),
          py::arg("coarsen"),
          py::arg("max_level"),
          "Coarsen the families whose elements are all marked to coarsen and none to refine, "
          "then refine, in one change; return whether the mesh changed.")
      .def(
          "project",
          [](MeshHierarchy & h, const std::string & name, const py::function & projection)
          {
            h.project(
                name,
                [&](const Point & x)
                {
                  const auto moved =
                      projection(py::make_tuple(x[0], x[1], x[2])).cast<std::vector<double>>();
                  Point out{0, 0, 0};
                  for (std::size_t k = 0; k < moved.size() && k < 3; ++k)
                    out[static_cast<int>(k)] = moved[k];
                  return out;
                });
          },
          py::arg("name"),
          py::arg("projection"),
          "Move the nodes of a side set onto a curved boundary, except the hanging nodes: "
          "projection((x, y, z)) gives the point of each.")
      .def("levels", &MeshHierarchy::levels)
      .def("origins", &MeshHierarchy::origins)
      .def(
          "transfer",
          [](MeshHierarchy & h, const std::vector<double> & values, int width)
          {
            const Eigen::VectorXd after = h.transfer(
                Eigen::Map<const Eigen::VectorXd>(values.data(), static_cast<Index>(values.size())),
                width);
            return std::vector<double>(after.data(), after.data() + after.size());
          },
          py::arg("values"),
          py::arg("width"),
          "Values at the nodes of the mesh before the last change, width per node, carried to "
          "the current mesh.")
      .def(
          "restrict",
          [](MeshHierarchy & h,
             const std::vector<double> & values,
             int width,
             const std::vector<double> & measures)
          {
            const Eigen::VectorXd after = h.restrictValues(
                Eigen::Map<const Eigen::VectorXd>(values.data(), static_cast<Index>(values.size())),
                width,
                Eigen::Map<const Eigen::VectorXd>(measures.data(),
                                                  static_cast<Index>(measures.size())));
            return std::vector<double>(after.data(), after.data() + after.size());
          },
          py::arg("values"),
          py::arg("width"),
          py::arg("measures"),
          "Values at the nodes of the mesh before the last change, width per node, carried to "
          "the current mesh so that the sum of the values times the measures of the nodes "
          "before stays: a removed node gives its content to the nodes of its merged element.");
  py::class_<Subdivision, MeshHierarchy>(m, "Subdivision")
      .def(py::init<const Mesh &>(), py::arg("mesh"))
      .def(
          "hanging",
          [](const Subdivision & s)
          {
            py::list out;
            for (const auto & h : s.hanging())
              out.append(py::make_tuple(h.node, h.on));
            return out;
          },
          "The hanging nodes: (node, [(node, weight), ...]) for each.");
  py::class_<Bisection, MeshHierarchy>(m, "Bisection")
      .def(py::init<const Mesh &>(), py::arg("mesh"));
}

} // namespace

void
bindMesh(py::module_ & m)
{
  MeshClass cls(
      m,
      "Mesh",
      "A mesh of finite elements: nodes, elements of any of the fourteen supported types "
      "(which may be mixed), element blocks, side sets and node sets. Meshes are usually "
      "made by the generators (generate_rectangle_mesh and the others) or read from a file "
      "(read_mesh), but can be built node by node with add_node and add_element.");
  defineNodesAndElements(cls);
  defineBlocksAndSets(cls);
  defineTransformations(cls);
  defineBoundaryQueries(cls);
  bindElementTypes(m);
  bindGenerators(m);
  bindHierarchies(m);
}

} // namespace dualmesh::bindings
