// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Tests of the mesh partitioners and of the distributed solver.
//
// Build without MPI, this program checks the partitioners and runs the
// distributed solver on a single rank, where it must reproduce the serial
// answer exactly.  Built with MPI (-DDUALMESH_ENABLE_MPI=ON) and launched with
//
//     mpirun -n 4 dualmesh_parallel_tests
//
// it additionally checks that the answer does not depend on the number of
// ranks: every rank computes the reference serial solution on the whole mesh
// and compares it, node by node, with the distributed one.
#include "dualmesh/base/Problem.h"
#include "dualmesh/linalg/PetscSolver.h"
#include "dualmesh/modules/Projection.h"
#include "dualmesh/parallel/Checkpoint.h"
#include "dualmesh/parallel/DistributedProblem.h"
#include "dualmesh/parallel/GatherScatter.h"
#include "dualmesh/parallel/Partition.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace dualmesh;

namespace
{
int g_failures = 0;

void
check(bool condition, const std::string & what)
{
  if (!Communicator::world().isRoot())
    return;
  if (condition)
    std::cout << "[  OK  ] " << what << "\n";
  else
  {
    std::cout << "[ FAIL ] " << what << "\n";
    ++g_failures;
  }
}

std::vector<double>
grid(int n, double a = 0.0, double b = 1.0)
{
  std::vector<double> x(n + 1);
  for (int i = 0; i <= n; ++i)
    x[i] = a + (b - a) * i / n;
  return x;
}

InputParameters
params(const std::string & type, std::initializer_list<std::pair<std::string, ParameterValue>> kv)
{
  auto p = Factory::instance().validParams(type);
  for (const auto & [k, v] : kv)
    p.set(k, v);
  return p;
}

/// This rank's part of @p mesh as a rank holds it when it builds or receives its part instead of cutting it from the whole mesh: its elements, their nodes and its own boundary sides, with no node sets.
/// The distributed solver must then complete the boundary node sets and the side set measures by itself.
LocalMesh
bareLocalPart(const Mesh & mesh, const std::vector<Index> & elements)
{
  LocalMesh part;
  part.mesh = std::make_shared<Mesh>(mesh.dimension());
  std::vector<Index> local_of(mesh.numNodes(), -1);
  std::map<Index, Index> local_element;
  for (Index e : elements)
  {
    const auto & el = mesh.element(e);
    std::vector<Index> nodes;
    for (int k = 0; k < el.numNodes(); ++k)
    {
      const Index g = el.nodes[k];
      if (local_of[g] < 0)
      {
        local_of[g] = part.mesh->addNode(mesh.node(g));
        part.global_nodes.push_back(g);
      }
      nodes.push_back(local_of[g]);
    }
    local_element[e] = part.mesh->addElement(el.type, nodes, el.block);
  }
  for (const auto & [name, sides] : mesh.sidesets())
  {
    std::vector<Side> kept;
    for (const Side & side : sides)
      if (local_element.count(side.first))
        kept.push_back({local_element[side.first], side.second});
    part.mesh->addSideset(name, kept);
  }
  return part;
}

/// Quadrilaterals on the unit square whose interior nodes are moved off the grid, so that no line between two cell centers is normal to their face; the edges stay straight and form the side sets "left", "right", "bottom" and "top".
Mesh
distortedQuadrilaterals(int nx, int ny)
{
  Mesh mesh(2);
  const double pi = std::acos(-1.0);
  for (int j = 0; j <= ny; ++j)
    for (int i = 0; i <= nx; ++i)
    {
      const double X = static_cast<double>(i) / nx, Y = static_cast<double>(j) / ny;
      const double bump = 0.07 * std::sin(pi * X) * std::sin(pi * Y);
      mesh.addNode({X + bump * std::cos(1.3 + 2 * Y), Y + bump * std::sin(0.7 + 3 * X), 0.0});
    }
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i)
    {
      const Index a = j * (nx + 1) + i;
      mesh.addElement(ElementType::Quad4, {a, a + 1, a + nx + 2, a + nx + 1}, 0);
    }
  for (const auto & [name, axis, position] : std::vector<std::tuple<std::string, int, double>>{
           {"left", 0, 0.0}, {"right", 0, 1.0}, {"bottom", 1, 0.0}, {"top", 1, 1.0}})
  {
    const int k = axis;
    const double p = position;
    mesh.addSidesetByPredicate(name, [k, p](const Point & x) { return std::abs(x[k] - p) < 1e-9; });
  }
  return mesh;
}

/// Steady heat conduction with a source on the unit square, with a mixture of
/// prescribed temperature and convection boundary conditions, so that both
/// kinds of boundary term are exercised.
void
defineConduction(Problem & problem)
{
  problem.addVariable("temperature");
  problem.addObject(
      "heat_conduction",
      "conduction",
      params("heat_conduction",
             {{"variable", std::string("temperature")}, {"thermal_conductivity", 2.5}}));
  problem.addObject(
      "heat_source",
      "source",
      params("heat_source", {{"variable", std::string("temperature")}, {"heat_source", 40.0}}));
  problem.addObject("Dirichlet_boundary_condition",
                    "cold",
                    params("Dirichlet_boundary_condition",
                           {{"variable", std::string("temperature")},
                            {"boundary", std::vector<std::string>{"left", "bottom"}},
                            {"value", 20.0}}));
  problem.addObject("convective_heat_flux_boundary_condition",
                    "convection",
                    params("convective_heat_flux_boundary_condition",
                           {{"variable", std::string("temperature")},
                            {"boundary", std::vector<std::string>{"right", "top"}},
                            {"heat_transfer_coefficient", 15.0},
                            {"ambient_temperature", 5.0}}));
}
} // namespace

int
main()
{
  auto & comm = Communicator::world();
  if (comm.isRoot())
    std::cout << "dualmesh parallel tests: " << comm.size() << " rank"
              << (comm.size() == 1 ? "" : "s") << ", MPI "
              << (Communicator::haveMpi() ? "enabled" : "not built in") << ", METIS "
              << (petsc::available() ? "available" : "not built in") << "\n\n";

  // ---- gather-scatter ----------------------------------------------------
  // Rank r holds global index g when a fixed pseudo-random rule says so, and every index is held at least by rank g % P.
  // So indices are held by one rank up to all of them, and each rank gives its indices in a scrambled order.
  // The expected sums, owners and copies follow from the same rule, which every rank can evaluate for every rank.
  std::vector<GatherScatter::Library> libraries = {GatherScatter::Library::Petsc};
  if (GatherScatter::haveGslib())
    libraries.push_back(GatherScatter::Library::Gslib);
  for (const auto library : libraries)
  {
    const std::string tag = library == GatherScatter::Library::Gslib ? "gslib" : "petsc";
    const int ranks = comm.size();
    const Index num_global = 1000;
    const auto holds = [&](int r, Index g)
    { return g % ranks == r || ((g * 2654435761L + r * 40503L) >> 7) % 3 == 0; };
    const auto value = [](int r, Index g, int c) { return 1.0 + r + 0.001 * g + 10.0 * c; };
    std::vector<Index> ids;
    for (Index g = 0; g < num_global; ++g)
      if (holds(comm.rank(), g))
        ids.push_back(g);
    for (std::size_t i = 0; i < ids.size(); ++i)
      std::swap(ids[i], ids[(i * 7919) % ids.size()]);
    GatherScatter gs;
    gs.setup(comm, ids, library);
    const int width = 2;
    std::vector<double> summed(ids.size() * width), copied(ids.size() * width);
    for (std::size_t i = 0; i < ids.size(); ++i)
      for (int c = 0; c < width; ++c)
        summed[i * width + c] = copied[i * width + c] = value(comm.rank(), ids[i], c);
    gs.sum(summed.data(), width);
    gs.copyFromOwners(copied.data(), width);
    bool owners_right = true, sums_right = true, copies_right = true;
    for (std::size_t i = 0; i < ids.size(); ++i)
    {
      const Index g = ids[i];
      int first = -1;
      for (int r = ranks - 1; r >= 0; --r)
        if (holds(r, g))
          first = r;
      owners_right = owners_right && gs.owner(static_cast<Index>(i)) == first;
      for (int c = 0; c < width; ++c)
      {
        double expected = 0;
        for (int r = 0; r < ranks; ++r)
          if (holds(r, g))
            expected += value(r, g, c);
        sums_right = sums_right && std::abs(summed[i * width + c] - expected) < 1e-12 * expected;
        copies_right = copies_right && copied[i * width + c] == value(first, g, c);
      }
    }
    const auto everywhere = [&](bool ok) { return !comm.any(!ok); };
    check(everywhere(owners_right),
          "gather-scatter (" + tag + "): the owner of every index is its smallest holder");
    check(everywhere(sums_right),
          "gather-scatter (" + tag + "): the sum over the holders reaches every holder");
    check(everywhere(copies_right),
          "gather-scatter (" + tag + "): every holder receives the owner's value");
    check(comm.sum(gs.numOwned()) == num_global,
          "gather-scatter (" + tag + "): every index is owned exactly once");

    // The largest holder of every index claims it, and every rank adds entities of its own (negative indices), which no operation changes and no rank owns.
    const auto claims_of = [&](int r, Index g)
    {
      for (int q = ranks - 1; q > r; --q)
        if (holds(q, g))
          return false;
      return true;
    };
    std::vector<Index> with_own = ids;
    std::vector<char> claims;
    for (Index g : ids)
      claims.push_back(claims_of(comm.rank(), g) ? 1 : 0);
    for (int k = 0; k < 5; ++k)
    {
      with_own.insert(with_own.begin() + k * 3, -1);
      claims.insert(claims.begin() + k * 3, 1);
    }
    GatherScatter claimed;
    claimed.setup(comm, with_own, library, claims);
    std::vector<double> values(with_own.size());
    for (std::size_t i = 0; i < with_own.size(); ++i)
      values[i] = with_own[i] < 0 ? -7.0 : value(comm.rank(), with_own[i], 0);
    std::vector<double> own_sum = values;
    claimed.sum(own_sum.data(), 1);
    claimed.copyFromOwners(values.data(), 1);
    bool claimed_right = true, own_untouched = true;
    for (std::size_t i = 0; i < with_own.size(); ++i)
    {
      const Index g = with_own[i];
      if (g < 0)
      {
        own_untouched = own_untouched && values[i] == -7.0 && own_sum[i] == -7.0 &&
                        !claimed.owns(static_cast<Index>(i)) &&
                        claimed.owner(static_cast<Index>(i)) == -1;
        continue;
      }
      int last = -1;
      for (int r = 0; r < ranks; ++r)
        if (holds(r, g))
          last = r;
      claimed_right = claimed_right && claimed.owner(static_cast<Index>(i)) == last &&
                      values[i] == value(last, g, 0);
    }
    check(everywhere(claimed_right),
          "gather-scatter (" + tag + "): the claiming holder owns an index and gives its value");
    check(everywhere(own_untouched),
          "gather-scatter (" + tag +
              "): an entity of one rank alone is left unchanged and unowned");
  }

  // ---- partitioners ------------------------------------------------------
  {
    for (const char * type : {"Quad4", "Tri3"})
    {
      Mesh mesh = generateRectangleMesh(grid(20), grid(20), type);
      for (const auto & method : availablePartitioners())
        for (int parts : {2, 4, 7})
        {
          const auto p = partitionMesh(mesh, parts, method);
          const auto [largest, smallest] = p.partSizes();
          bool all_assigned = true;
          for (int e : p.element_part)
            all_assigned = all_assigned && e >= 0 && e < parts;
          // No part may be more than 25% larger than the average.
          const double average = static_cast<double>(mesh.numElements()) / parts;
          const bool balanced = largest <= 1.25 * average + 1;
          std::ostringstream name;
          name << "partitioner " << method << " into " << parts << " parts of a 20 x 20 " << type
               << " mesh (cut " << p.edgeCut(mesh) << " faces, sizes " << smallest << " to "
               << largest << ")";
          check(all_assigned && balanced && smallest > 0, name.str());
        }
    }
  }

  // ---- sub-meshes preserve the boundary ----------------------------------
  {
    Mesh mesh = generateRectangleMesh(grid(8), grid(8), "Quad4");
    // Any division of the elements will do: element e goes to part e mod 4.
    MeshPartition p;
    p.num_parts = 4;
    for (Index e = 0; e < mesh.numElements(); ++e)
      p.element_part.push_back(static_cast<int>(e % 4));
    Index total_sides = 0;
    std::vector<char> seen(mesh.numNodes(), 0);
    for (int r = 0; r < 4; ++r)
    {
      std::vector<Index> l2g;
      Mesh part = subMesh(mesh, p.elementsOf(r), l2g);
      for (const auto & name : {"left", "right", "bottom", "top"})
        total_sides += static_cast<Index>(part.sideset(name).size());
      for (Index g : l2g)
        seen[g] = 1;
    }
    Index reference = 0;
    for (const auto & name : {"left", "right", "bottom", "top"})
      reference += static_cast<Index>(mesh.sideset(name).size());
    check(total_sides == reference,
          "the sub-meshes of a partition hold every boundary side exactly once");
    check(std::count(seen.begin(), seen.end(), 1) == mesh.numNodes(),
          "the sub-meshes of a partition cover every node");
  }

  // ---- the root sends every rank its part ---------------------------------
  // Only the root holds the mesh; the parts must cover it exactly, keep its numbering, coordinates, blocks and boundaries, and solve as the whole mesh does.
  {
    Mesh whole = generateRectangleMesh(grid(11), grid(7, 0.0, 0.6), "Tri3");
    whole.addNodesetByPredicate("probe",
                                [](const Point & p) { return p[0] > 0.45 && p[0] < 0.55; });
    const Mesh empty(2);
    const LocalMesh part = distributeMesh(comm.isRoot() ? whole : empty, comm, "automatic");
    const Mesh & local = *part.mesh;
    bool coordinates_right = local.dimension() == whole.dimension();
    for (Index n = 0; n < local.numNodes(); ++n)
      for (int c = 0; c < 3; ++c)
        coordinates_right =
            coordinates_right && local.node(n)[c] == whole.node(part.global_nodes[n])[c];
    check(!comm.any(!coordinates_right),
          "distributed parts keep the global numbering and coordinates");
    check(comm.sum(local.numElements()) == whole.numElements(),
          "distributed parts hold every element exactly once");
    Index sides = 0, whole_sides = 0;
    for (const auto & [name, list] : whole.sidesets())
    {
      sides += static_cast<Index>(local.sideset(name).size());
      whole_sides += static_cast<Index>(list.size());
    }
    check(comm.sum(sides) == whole_sides,
          "distributed parts hold every boundary side exactly once");
    bool sets_right = true;
    for (const char * name : {"probe", "left", "top"})
    {
      std::vector<char> in_whole(whole.numNodes(), 0), in_part(local.numNodes(), 0);
      for (Index g : whole.boundaryNodes(name))
        in_whole[g] = 1;
      for (Index n : local.boundaryNodes(name))
        in_part[n] = 1;
      for (Index n = 0; n < local.numNodes(); ++n)
        sets_right = sets_right && in_part[n] == in_whole[part.global_nodes[n]];
    }
    check(!comm.any(!sets_right),
          "distributed parts hold the nodes of every node set and side set");

    auto reference_mesh = std::make_shared<Mesh>(whole);
    Problem reference(reference_mesh, Method::DualMesh);
    defineConduction(reference);
    reference.solveSteady();
    const auto expected = reference.values("temperature");
    DistributedOptions options;
    options.linear_tolerance = 1e-13;
    DistributedProblem distributed(
        comm.isRoot() ? whole : empty, Method::DualMesh, CoordinateSystem::Cartesian, options);
    defineConduction(distributed.local());
    distributed.solveSteady();
    const auto got = distributed.gatheredValues("temperature");
    check(got.size() == static_cast<std::size_t>(whole.numNodes()),
          "the gathered values hold one value per node of the whole mesh");
    double worst = 0;
    for (Index n = 0; n < whole.numNodes(); ++n)
      worst = std::max(worst, std::abs(got[n] - expected[n]));
    std::ostringstream label;
    label << "a mesh held by the root alone gives the serial solution (worst difference " << worst
          << ")";
    check(worst < 1e-8, label.str());
    bool refused = false;
    try
    {
      DistributedProblem too_many(comm.isRoot() ? generateRectangleMesh(grid(1), grid(1), "Quad4")
                                                : empty,
                                  Method::DualMesh,
                                  CoordinateSystem::Cartesian,
                                  options);
    }
    catch (const InputError &)
    {
      refused = true;
    }
    if (comm.size() > 1)
      check(!comm.any(!refused), "a mesh the root cannot partition stops every rank with an error");
  }

  // ---- every element type travels whole -----------------------------------
  // The partitioner sees only the corners of an element, and the element travels as the record of its cell, so every type, the quadratic ones included, must arrive complete.
  {
    struct Case
    {
      const char * name;
      Mesh mesh;
    };
    std::vector<Case> cases;
    cases.push_back({"Tri6", generateRectangleMesh(grid(6), grid(5), "Tri3").secondOrder()});
    cases.push_back({"Quad9", generateRectangleMesh(grid(6), grid(5), "Quad4").secondOrder()});
    cases.push_back({"Quad8", generateRectangleMesh(grid(6), grid(5), "Quad4").secondOrder(true)});
    cases.push_back({"Hex8", generateBoxMesh(grid(3), grid(3), grid(2), "Hex8")});
    cases.push_back({"Tet4", generateBoxMesh(grid(2), grid(2), grid(2), "Tet4")});
    cases.push_back({"Tet10", generateBoxMesh(grid(2), grid(2), grid(2), "Tet4").secondOrder()});
    cases.push_back({"Hex27", generateBoxMesh(grid(3), grid(2), grid(2), "Hex8").secondOrder()});
    cases.push_back(
        {"Hex20", generateBoxMesh(grid(3), grid(2), grid(2), "Hex8").secondOrder(true)});
    for (const auto & c : cases)
    {
      const Mesh empty(c.mesh.dimension());
      const LocalMesh part = distributeMesh(comm.isRoot() ? c.mesh : empty, comm, "automatic");
      const Mesh & local = *part.mesh;
      bool same = local.dimension() == c.mesh.dimension();
      for (Index n = 0; n < local.numNodes(); ++n)
        for (int k = 0; k < 3; ++k)
          same = same && local.node(n)[k] == c.mesh.node(part.global_nodes[n])[k];
      for (Index e = 0; e < local.numElements(); ++e)
        same = same && local.element(e).type == c.mesh.element(0).type;
      Index sides = 0, whole_sides = 0;
      for (const auto & [name, list] : c.mesh.sidesets())
      {
        sides += static_cast<Index>(local.sideset(name).size());
        whole_sides += static_cast<Index>(list.size());
      }
      check(!comm.any(!same) && comm.sum(local.numElements()) == c.mesh.numElements() &&
                comm.sum(sides) == whole_sides,
            std::string("a ") + c.name +
                " mesh arrives whole: numbering, coordinates, every element and side once");
    }
  }

  // ---- every element type travels whole -----------------------------------
  // The partitioner sees only the corners of an element, and the element travels as the record of its cell, so every type, the quadratic ones included, must arrive complete.
  {
    struct Case
    {
      const char * name;
      Mesh mesh;
    };
    std::vector<Case> cases;
    cases.push_back({"Tri6", generateRectangleMesh(grid(6), grid(5), "Tri3").secondOrder()});
    cases.push_back({"Quad9", generateRectangleMesh(grid(6), grid(5), "Quad4").secondOrder()});
    cases.push_back({"Quad8", generateRectangleMesh(grid(6), grid(5), "Quad4").secondOrder(true)});
    cases.push_back({"Hex8", generateBoxMesh(grid(3), grid(3), grid(2), "Hex8")});
    cases.push_back({"Tet4", generateBoxMesh(grid(2), grid(2), grid(2), "Tet4")});
    cases.push_back({"Tet10", generateBoxMesh(grid(2), grid(2), grid(2), "Tet4").secondOrder()});
    cases.push_back({"Hex27", generateBoxMesh(grid(3), grid(2), grid(2), "Hex8").secondOrder()});
    cases.push_back(
        {"Hex20", generateBoxMesh(grid(3), grid(2), grid(2), "Hex8").secondOrder(true)});
    for (const auto & c : cases)
    {
      const Mesh empty(c.mesh.dimension());
      const LocalMesh part = distributeMesh(comm.isRoot() ? c.mesh : empty, comm, "automatic");
      const Mesh & local = *part.mesh;
      bool same = local.dimension() == c.mesh.dimension();
      for (Index n = 0; n < local.numNodes(); ++n)
        for (int k = 0; k < 3; ++k)
          same = same && local.node(n)[k] == c.mesh.node(part.global_nodes[n])[k];
      for (Index e = 0; e < local.numElements(); ++e)
        same = same && local.element(e).type == c.mesh.element(0).type;
      Index sides = 0, whole_sides = 0;
      for (const auto & [name, list] : c.mesh.sidesets())
      {
        sides += static_cast<Index>(local.sideset(name).size());
        whole_sides += static_cast<Index>(list.size());
      }
      check(!comm.any(!same) && comm.sum(local.numElements()) == c.mesh.numElements() &&
                comm.sum(sides) == whole_sides,
            std::string("a ") + c.name +
                " mesh arrives whole: numbering, coordinates, every element and side once");
    }
  }

  // ---- the distributed solution does not depend on the partition ---------
  for (auto [name, method] :
       std::vector<std::pair<std::string, Method>>{{"dmcdm", Method::DualMesh},
                                                   {"fem", Method::FiniteElement},
                                                   {"hfvm", Method::FiniteVolumeVertex}})
  {
    Mesh mesh = generateRectangleMesh(grid(12), grid(12), "Quad4");

    // Reference: the whole problem on one process.
    auto reference_mesh = std::make_shared<Mesh>(mesh);
    Problem reference(reference_mesh, method);
    defineConduction(reference);
    reference.solveSteady();
    const auto expected = reference.values("temperature");

    // PETSc's Krylov methods and preconditioners: the automatic choice, restricted additive Schwarz with incomplete and exact subdomain solves, smoothed aggregation multigrid, and the parallel direct solver.
    const std::pair<const char *, const char *> variants[] = {
        {"automatic", ""},
        {"Schwarz, ILU", "-ksp_type gmres -pc_type asm -sub_pc_type ilu"},
        {"Schwarz overlap 2, LU", "-ksp_type gmres -pc_type asm -pc_asm_overlap 2 -sub_pc_type lu"},
        {"GAMG", "-ksp_type gmres -pc_type gamg"},
        {"MUMPS", "-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps"}};
    std::vector<std::string> gather_scatters = {"petsc"};
    if (GatherScatter::haveGslib())
      gather_scatters.push_back("gslib");
    std::vector<std::string> partitioners = availablePartitioners();
    if (partitioners.empty())
      partitioners.push_back("automatic");
    for (const auto & partitioner : partitioners)
      for (const auto & [solver, petsc_options] : variants)
        for (const auto & gather_scatter : gather_scatters)
        {
          const std::string preconditioner =
              std::string(solver) + ", gather-scatter " + gather_scatter;
          DistributedOptions options;
          options.partitioner = partitioner;
          options.gather_scatter = gather_scatter;
          options.petsc_options = petsc_options;
          options.linear_tolerance = 1e-13;
          DistributedProblem distributed(mesh, method, CoordinateSystem::Cartesian, options);
          defineConduction(distributed.local());
          distributed.solveSteady();
          const auto got = distributed.gatheredValues("temperature");
          check(got.size() == static_cast<std::size_t>(mesh.numNodes()),
                "the gathered values hold one value per node of the whole mesh");
          double worst = 0;
          for (Index n = 0; n < mesh.numNodes(); ++n)
            worst = std::max(worst, std::abs(got[n] - expected[n]));
          std::ostringstream label;
          label << name << ": distributed solution matches the serial one (" << partitioner << ", "
                << preconditioner << ", worst difference " << worst << ")";
          check(worst < 1e-8, label.str());
        }
  }

  // ---- a concentrated load is applied exactly once -----------------------
  {
    Mesh mesh = generateRectangleMesh(grid(10), grid(10), "Quad4");
    const auto define = [](Problem & problem)
    {
      problem.addVariable("u");
      problem.addObject(
          "diffusion", "diffusion", params("diffusion", {{"variable", std::string("u")}}));
      problem.addObject("point_source",
                        "load",
                        params("point_source",
                               {{"variable", std::string("u")},
                                {"value", 3.0},
                                {"points", std::vector<double>{0.5, 0.5, 0.0}}}));
      problem.addObject(
          "Dirichlet_boundary_condition",
          "edges",
          params("Dirichlet_boundary_condition",
                 {{"variable", std::string("u")},
                  {"boundary", std::vector<std::string>{"left", "right", "bottom", "top"}},
                  {"value", 0.0}}));
    };
    auto reference_mesh = std::make_shared<Mesh>(mesh);
    Problem reference(reference_mesh, Method::DualMesh);
    define(reference);
    reference.solveSteady();
    const auto expected = reference.values("u");

    DistributedOptions options;
    options.linear_tolerance = 1e-13;
    DistributedProblem distributed(mesh, Method::DualMesh, CoordinateSystem::Cartesian, options);
    define(distributed.local());
    distributed.solveSteady();
    const auto got = distributed.gatheredValues("u");
    check(got.size() == static_cast<std::size_t>(mesh.numNodes()),
          "the gathered values hold one value per node of the whole mesh");
    double worst = 0;
    for (Index n = 0; n < mesh.numNodes(); ++n)
      worst = std::max(worst, std::abs(got[n] - expected[n]));
    std::ostringstream label;
    label << "a concentrated load is applied once, not once per rank (worst difference " << worst
          << ")";
    check(worst < 1e-8, label.str());
  }

  // ---- a vector problem ---------------------------------------------------
  // Plane elasticity: two unknowns per node, so the exchanges and the global numbering carry more than one variable.
  {
    Mesh mesh = generateRectangleMesh(grid(16, 0.0, 2.0), grid(8), "Quad4");
    const auto define = [](Problem & problem)
    {
      problem.addVariable("disp_x");
      problem.addVariable("disp_y");
      problem.addObject("linear_elastic_stress",
                        "stress",
                        params("linear_elastic_stress",
                               {{"displacements", std::vector<std::string>{"disp_x", "disp_y"}},
                                {"youngs_modulus", 200.0},
                                {"poissons_ratio", 0.3}}));
      for (int c = 0; c < 2; ++c)
      {
        const std::string v = c == 0 ? "disp_x" : "disp_y";
        problem.addObject(
            "stress_divergence",
            "equilibrium_" + v,
            params("stress_divergence", {{"variable", v}, {"component", static_cast<long>(c)}}));
        problem.addObject(
            "Dirichlet_boundary_condition",
            "clamp_" + v,
            params(
                "Dirichlet_boundary_condition",
                {{"variable", v}, {"boundary", std::vector<std::string>{"left"}}, {"value", 0.0}}));
      }
      problem.addObject("traction_boundary_condition",
                        "load",
                        params("traction_boundary_condition",
                               {{"variable", std::string("disp_y")},
                                {"boundary", std::vector<std::string>{"right"}},
                                {"traction", -1.0}}));
    };
    auto reference_mesh = std::make_shared<Mesh>(mesh);
    Problem reference(reference_mesh, Method::DualMesh);
    define(reference);
    reference.solveSteady();
    for (const char * preconditioner : {"", "-ksp_type gmres -pc_type asm -sub_pc_type lu"})
    {
      DistributedOptions options;
      options.petsc_options = preconditioner;
      options.linear_tolerance = 1e-13;
      DistributedProblem distributed(mesh, Method::DualMesh, CoordinateSystem::Cartesian, options);
      define(distributed.local());
      distributed.solveSteady();
      double worst = 0, scale = 0;
      for (const char * v : {"disp_x", "disp_y"})
      {
        const auto expected = reference.values(v);
        const auto got = distributed.gatheredValues(v);
        check(got.size() == static_cast<std::size_t>(mesh.numNodes()),
              "the gathered values hold one value per node of the whole mesh");
        for (Index n = 0; n < mesh.numNodes(); ++n)
        {
          worst = std::max(worst, std::abs(got[n] - expected[n]));
          scale = std::max(scale, std::abs(expected[n]));
        }
      }
      std::ostringstream label;
      label << "plane elasticity matches the serial solution (PETSc options '" << preconditioner
            << "', worst relative difference " << worst / scale << ")";
      check(worst <= 1e-8 * scale, label.str());
    }
  }

  // ---- a part built by its rank -----------------------------------------
  // The parts hold only their own elements and sides, as when each rank builds or receives its part.
  // Triangles are used because a triangle can touch a boundary node with a vertex only, without a side on the boundary.
  // A clamped node on a partition boundary whose side belongs to another rank must still be clamped, and a total force must be spread over the area of the whole side set.
  {
    Mesh mesh = generateRectangleMesh(grid(15, 0.0, 2.0), grid(9), "Tri3");
    const auto define = [](Problem & problem)
    {
      problem.addVariable("disp_x");
      problem.addVariable("disp_y");
      problem.addObject("linear_elastic_stress",
                        "stress",
                        params("linear_elastic_stress",
                               {{"displacements", std::vector<std::string>{"disp_x", "disp_y"}},
                                {"youngs_modulus", 200.0},
                                {"poissons_ratio", 0.3}}));
      for (int c = 0; c < 2; ++c)
      {
        const std::string v = c == 0 ? "disp_x" : "disp_y";
        problem.addObject(
            "stress_divergence",
            "equilibrium_" + v,
            params("stress_divergence", {{"variable", v}, {"component", static_cast<long>(c)}}));
        problem.addObject("Dirichlet_boundary_condition",
                          "clamp_" + v,
                          params("Dirichlet_boundary_condition",
                                 {{"variable", v},
                                  {"boundary", std::vector<std::string>{"left", "bottom"}},
                                  {"value", 0.0}}));
      }
      problem.addObject("traction_boundary_condition",
                        "load",
                        params("traction_boundary_condition",
                               {{"variable", std::string("disp_y")},
                                {"boundary", std::vector<std::string>{"right", "top"}},
                                {"total_force", -3.0}}));
    };
    auto reference_mesh = std::make_shared<Mesh>(mesh);
    Problem reference(reference_mesh, Method::FiniteElement);
    define(reference);
    reference.solveSteady();
    // The cyclic partition, element e on rank e modulo the number of ranks, is the hardest case: nodes shared by up to six ranks, and many boundary nodes held without a side.
    std::vector<std::string> partitioners = availablePartitioners();
    partitioners.push_back("cyclic");
    for (const std::string & partitioner : partitioners)
    {
      std::vector<Index> elements;
      if (partitioner == "cyclic")
      {
        for (Index e = comm.rank(); e < mesh.numElements(); e += comm.size())
          elements.push_back(e);
      }
      else
      {
        // A partitioner may draw random numbers, so the root partitions and every rank takes its elements from the root's partition.
        std::vector<Index> parts;
        if (comm.isRoot())
          for (int part : partitionMesh(mesh, comm.size(), partitioner).element_part)
            parts.push_back(part);
        comm.broadcast(parts);
        for (Index e = 0; e < mesh.numElements(); ++e)
          if (parts[e] == comm.rank())
            elements.push_back(e);
      }
      DistributedOptions options;
      options.linear_tolerance = 1e-13;
      auto part = bareLocalPart(mesh, elements);
      // Count the held boundary nodes that no local side carries: the nodes the solver must add.
      Index missing = 0;
      {
        std::vector<char> carried(part.global_nodes.size(), 0);
        for (const auto & name : {"left", "bottom"})
          for (Index n : part.mesh->boundaryNodes(name))
            carried[n] = 1;
        std::vector<char> global_boundary(mesh.numNodes(), 0);
        for (const auto & name : {"left", "bottom"})
          for (Index g : mesh.boundaryNodes(name))
            global_boundary[g] = 1;
        for (std::size_t n = 0; n < part.global_nodes.size(); ++n)
          missing += global_boundary[part.global_nodes[n]] && !carried[n] ? 1 : 0;
      }
      const auto local_mesh = part.mesh;
      const auto global_nodes = part.global_nodes;
      DistributedProblem distributed(
          std::move(part), Method::FiniteElement, CoordinateSystem::Cartesian, options);
      bool node_sets_right = true;
      for (const auto & [name, sides] : mesh.sidesets())
      {
        std::vector<char> global_in(mesh.numNodes(), 0), local_in(global_nodes.size(), 0);
        for (Index g : mesh.boundaryNodes(name))
          global_in[g] = 1;
        for (Index n : local_mesh->boundaryNodes(name))
          local_in[n] = 1;
        for (std::size_t n = 0; n < global_nodes.size(); ++n)
          node_sets_right = node_sets_right && local_in[n] == global_in[global_nodes[n]];
      }
      check(!comm.any(!node_sets_right),
            "every rank holds the complete boundary node sets (" + partitioner +
                ", boundary nodes added: " + std::to_string(comm.sum(missing)) + ")");
      define(distributed.local());
      distributed.solveSteady();
      double worst = 0, scale = 0;
      for (const char * v : {"disp_x", "disp_y"})
      {
        const auto expected = reference.values(v);
        const auto got = distributed.gatheredValues(v);
        check(got.size() == static_cast<std::size_t>(mesh.numNodes()),
              "the gathered values hold one value per node of the whole mesh");
        for (Index n = 0; n < mesh.numNodes(); ++n)
        {
          worst = std::max(worst, std::abs(got[n] - expected[n]));
          scale = std::max(scale, std::abs(expected[n]));
        }
      }
      std::ostringstream label;
      label << "parts built by their ranks give the serial solution (" << partitioner
            << ", worst relative difference " << worst / scale << ")";
      check(worst <= 1e-8 * scale, label.str());
    }
  }

  // ---- periodic boundaries across ranks --------------------------------------
  // The primary and the secondary nodes of a periodic pair lie on different ranks, and with the cyclic partition a rank often holds both: the distributed solution must equal the serial one, in one direction and in two.
  {
    struct Source : Function
    {
      double value(const Point & x, double) const override
      {
        return 1.0 + std::cos(2 * M_PI * x[0] + 0.3) * std::sin(M_PI * x[1]) +
               0.5 * std::sin(2 * M_PI * x[1]);
      }
    };
    const auto define = [](Problem & problem, bool doubly)
    {
      problem.addFunction("source", std::make_shared<Source>());
      problem.addVariable("u");
      problem.addObject(
          "diffusion", "diffusion", params("diffusion", {{"variable", std::string("u")}}));
      problem.addObject(
          "body_force",
          "source",
          params("body_force", {{"variable", std::string("u")}, {"value", std::string("source")}}));
      problem.addObject(
          "periodic_boundary_condition",
          "periodic_x",
          params("periodic_boundary_condition",
                 {{"primary", std::string("left")}, {"secondary", std::string("right")}}));
      if (doubly)
      {
        problem.addObject(
            "reaction", "reaction", params("reaction", {{"variable", std::string("u")}}));
        problem.addObject(
            "periodic_boundary_condition",
            "periodic_y",
            params("periodic_boundary_condition",
                   {{"primary", std::string("bottom")}, {"secondary", std::string("top")}}));
      }
      else
        problem.addObject("Dirichlet_boundary_condition",
                          "walls",
                          params("Dirichlet_boundary_condition",
                                 {{"variable", std::string("u")},
                                  {"boundary", std::vector<std::string>{"bottom"}},
                                  {"value", 0.0}}));
    };
    std::vector<std::string> gather_scatters = {"petsc"};
    if (GatherScatter::haveGslib())
      gather_scatters.push_back("gslib");
    for (bool doubly : {false, true})
      for (const auto & [name, method] :
           std::vector<std::pair<std::string, Method>>{{"dmcdm", Method::DualMesh},
                                                       {"fem", Method::FiniteElement},
                                                       {"hfvm", Method::FiniteVolumeVertex}})
      {
        Mesh mesh = generateRectangleMesh(grid(12), grid(10), "Quad4");
        auto reference_mesh = std::make_shared<Mesh>(mesh);
        Problem reference(reference_mesh, method);
        define(reference, doubly);
        reference.solveSteady();
        const auto expected = reference.values("u");
        for (const auto & gather_scatter : gather_scatters)
          for (const std::string partition : {"automatic", "cyclic"})
          {
            DistributedOptions options;
            options.gather_scatter = gather_scatter;
            options.linear_tolerance = 1e-13;
            std::unique_ptr<DistributedProblem> distributed;
            if (partition == "cyclic")
            {
              std::vector<Index> elements;
              for (Index e = comm.rank(); e < mesh.numElements(); e += comm.size())
                elements.push_back(e);
              distributed = std::make_unique<DistributedProblem>(
                  bareLocalPart(mesh, elements), method, CoordinateSystem::Cartesian, options);
            }
            else
              distributed = std::make_unique<DistributedProblem>(
                  mesh, method, CoordinateSystem::Cartesian, options);
            define(distributed->local(), doubly);
            distributed->solveSteady();
            const auto got = distributed->gatheredValues("u");
            double worst = got.size() == expected.size() ? 0.0 : 1e300;
            for (std::size_t n = 0; n < got.size() && n < expected.size(); ++n)
              worst = std::max(worst, std::abs(got[n] - expected[n]));
            std::ostringstream label;
            label << name << (doubly ? ", doubly periodic" : ", periodic in x") << ", " << partition
                  << " partition, gather-scatter " << gather_scatter
                  << ": the distributed solution equals the serial one (worst difference " << worst
                  << ")";
            check(worst < 1e-8, label.str());
          }
      }
  }

  // ---- a gap between bodies on different ranks -------------------------------
  // Heat crosses a gap between two slabs whose meshes do not match; the slabs lie on different ranks, so the ranks that hold the primary side read the secondary elements as ghost elements.
  {
    Mesh mesh(2);
    const auto slab = [&](double x0, double x1, int nx, int ny, int block)
    {
      const Index first = mesh.numNodes();
      for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i)
          mesh.addNode({x0 + (x1 - x0) * i / nx, static_cast<double>(j) / ny, 0.0});
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
        {
          const Index a = first + j * (nx + 1) + i;
          mesh.addElement(ElementType::Quad4, {a, a + 1, a + nx + 2, a + nx + 1}, block);
        }
    };
    slab(0.0, 1.0, 8, 3, 0);
    slab(1.05, 2.0, 8, 5, 1);
    mesh.setBlockName(0, "inner");
    mesh.setBlockName(1, "outer");
    for (const auto & [name, position] : std::vector<std::pair<std::string, double>>{
             {"left", 0.0}, {"primary", 1.0}, {"secondary", 1.05}, {"right", 2.0}})
    {
      const double p = position;
      mesh.addSidesetByPredicate(name, [p](const Point & x) { return std::abs(x[0] - p) < 1e-9; });
    }
    const auto define = [](Problem & problem)
    {
      problem.addVariable("temperature");
      problem.addObject("heat_conduction",
                        "inner",
                        params("heat_conduction",
                               {{"variable", std::string("temperature")},
                                {"thermal_conductivity", 3.0},
                                {"block", std::vector<std::string>{"inner"}}}));
      problem.addObject("heat_conduction",
                        "outer",
                        params("heat_conduction",
                               {{"variable", std::string("temperature")},
                                {"thermal_conductivity", 16.0},
                                {"block", std::vector<std::string>{"outer"}}}));
      problem.addObject("heat_source",
                        "source",
                        params("heat_source",
                               {{"variable", std::string("temperature")},
                                {"heat_source", 100.0},
                                {"block", std::vector<std::string>{"inner"}}}));
      problem.addObject("Dirichlet_boundary_condition",
                        "cooled",
                        params("Dirichlet_boundary_condition",
                               {{"variable", std::string("temperature")},
                                {"boundary", std::vector<std::string>{"right"}},
                                {"value", 10.0}}));
      problem.addObject("gap_heat_transfer",
                        "gap",
                        params("gap_heat_transfer",
                               {{"variable", std::string("temperature")},
                                {"boundary", std::vector<std::string>{"primary"}},
                                {"secondary_boundary", std::vector<std::string>{"secondary"}},
                                {"gap_conductance", 50.0}}));
    };
    for (const auto & [name, method] :
         std::vector<std::pair<std::string, Method>>{{"dmcdm", Method::DualMesh},
                                                     {"fem", Method::FiniteElement},
                                                     {"hfvm", Method::FiniteVolumeVertex},
                                                     {"zfvm", Method::FiniteVolumeCell}})
    {
      auto reference_mesh = std::make_shared<Mesh>(mesh);
      Problem reference(reference_mesh, method);
      define(reference);
      reference.solveSteady();
      const auto expected = reference.values("temperature");
      DistributedOptions options;
      options.linear_tolerance = 1e-13;
      DistributedProblem distributed(mesh, method, CoordinateSystem::Cartesian, options);
      define(distributed.local());
      distributed.solveSteady();
      const auto got = distributed.gatheredValues("temperature");
      double worst = got.size() == expected.size() ? 0.0 : 1e300, scale = 0;
      for (std::size_t n = 0; n < got.size() && n < expected.size(); ++n)
      {
        worst = std::max(worst, std::abs(got[n] - expected[n]));
        scale = std::max(scale, std::abs(expected[n]));
      }
      std::ostringstream label;
      label << name << ": a gap between bodies on different ranks gives the serial solution (worst "
            << "relative difference " << worst / scale << ")";
      check(worst <= 1e-9 * scale, label.str());
    }
  }

  // ---- the cell-centered finite volume method ---------------------------------
  // Every part holds two layers of ghost cells and every face is integrated by one rank, so the value at every cell and boundary face is the serial one.
  // On distorted quadrilaterals the line between two cell centers is not normal to their face, so the flux needs the reconstructed gradients of both cells, and those of a ghost cell are complete only with the second layer (on triangles and tetrahedra the first layer already holds every neighbor of a ghost cell next to the part).
  // The nonlinear diffusivity checks, by the number of Newton iterations, that the Jacobian is the serial one.
  {
    struct Case
    {
      std::string name;
      Mesh mesh;
      bool nonlinear;
      BoundaryGradient gradient;
    };
    std::vector<Case> cases;
    cases.push_back({"triangles",
                     generateRectangleMesh(grid(14), grid(11), "Tri3"),
                     false,
                     BoundaryGradient::FirstOrder});
    cases.push_back({"distorted quadrilaterals with the second-order boundary gradient",
                     distortedQuadrilaterals(13, 9),
                     false,
                     BoundaryGradient::SecondOrder});
    cases.push_back({"distorted quadrilaterals with a nonlinear diffusivity",
                     distortedQuadrilaterals(10, 12),
                     true,
                     BoundaryGradient::FirstOrder});
    cases.push_back({"triangles with a nonlinear diffusivity",
                     generateRectangleMesh(grid(12), grid(10), "Tri3", "left"),
                     true,
                     BoundaryGradient::SecondOrder});
    cases.push_back({"tetrahedra",
                     generateBoxMesh(grid(4), grid(3), grid(3), "Tet4"),
                     false,
                     BoundaryGradient::FirstOrder});
    const auto define = [](Problem & problem, bool nonlinear)
    {
      if (!nonlinear)
      {
        defineConduction(problem);
        return;
      }
      problem.addVariable("temperature");
      problem.addObject("diffusion",
                        "diffusion",
                        params("diffusion",
                               {{"variable", std::string("temperature")},
                                {"diffusivity", 2.0},
                                {"solution_polynomial", std::vector<double>{1.0, 0.08}}}));
      problem.addObject(
          "body_force",
          "source",
          params("body_force", {{"variable", std::string("temperature")}, {"value", 300.0}}));
      problem.addObject("Dirichlet_boundary_condition",
                        "cold",
                        params("Dirichlet_boundary_condition",
                               {{"variable", std::string("temperature")},
                                {"boundary", std::vector<std::string>{"left", "bottom"}},
                                {"value", 1.0}}));
    };
    if (comm.size() > 1)
    {
      Mesh whole = generateRectangleMesh(grid(6), grid(6), "Quad4");
      std::vector<Index> mine;
      for (Index e = 0; e < whole.numElements(); ++e)
        if (e % comm.size() == comm.rank())
          mine.push_back(e);
      bool refused = false;
      try
      {
        DistributedProblem bare(bareLocalPart(whole, mine), Method::FiniteVolumeCell);
      }
      catch (const InputError &)
      {
        refused = true;
      }
      check(!comm.any(!refused), "zfvm refuses a part of a mesh without two layers of ghost cells");
    }
    for (const auto & c : cases)
    {
      Problem reference(std::make_shared<Mesh>(c.mesh), Method::FiniteVolumeCell);
      reference.setBoundaryGradient(c.gradient);
      define(reference, c.nonlinear);
      const SolveResult serial = reference.solveSteady();
      const auto expected = reference.values("temperature");
      std::vector<std::string> exchanges = {"petsc"};
      if (GatherScatter::haveGslib())
        exchanges.push_back("gslib");
      for (const auto & exchange : exchanges)
      {
        DistributedOptions options;
        options.linear_tolerance = 1e-13;
        options.gather_scatter = exchange;
        DistributedProblem distributed(
            c.mesh, Method::FiniteVolumeCell, CoordinateSystem::Cartesian, options);
        distributed.local().setBoundaryGradient(c.gradient);
        define(distributed.local(), c.nonlinear);
        const SolveResult parallel = distributed.solveSteady();
        const auto got = distributed.gatheredValues("temperature");
        double worst = got.size() == expected.size() ? 0.0 : 1e300, scale = 0;
        for (std::size_t n = 0; n < got.size() && n < expected.size(); ++n)
        {
          worst = std::max(worst, std::abs(got[n] - expected[n]));
          scale = std::max(scale, std::abs(expected[n]));
        }
        std::ostringstream label;
        label << "zfvm on " << c.name << " (" << exchange
              << ") gives the serial value at every cell and boundary face (worst relative "
                 "difference "
              << worst / scale << ", Newton iterations " << parallel.total_iterations << ", serial "
              << serial.total_iterations << ")";
        check(worst <= 1e-9 * scale && parallel.total_iterations == serial.total_iterations,
              label.str());
      }
    }
  }

  // ---- checkpoint and restart -------------------------------------------------
  // A run stopped half way and restarted from its checkpoint, with another partition of the mesh, ends where an uninterrupted run ends.
  if (checkpointsAvailable())
  {
    struct Bump : Function
    {
      double value(const Point & x, double t) const override
      {
        return 10.0 * std::sin(M_PI * x[0]) * (1.0 + x[1]) * std::cos(t);
      }
    };
    const auto define = [](Problem & problem)
    {
      problem.addFunction("bump", std::make_shared<Bump>());
      problem.addVariable("u");
      problem.addObject(
          "diffusion", "diffusion", params("diffusion", {{"variable", std::string("u")}}));
      problem.addObject(
          "time_derivative", "time", params("time_derivative", {{"variable", std::string("u")}}));
      problem.addObject(
          "body_force",
          "source",
          params("body_force", {{"variable", std::string("u")}, {"value", std::string("bump")}}));
      problem.addObject("Dirichlet_boundary_condition",
                        "walls",
                        params("Dirichlet_boundary_condition",
                               {{"variable", std::string("u")},
                                {"boundary", std::vector<std::string>{"left", "right"}},
                                {"value", 0.0}}));
    };
    Mesh mesh = distortedQuadrilaterals(10, 8);
    const std::string file = "dualmesh_parallel_tests.chk";
    for (const auto & [name, method] : std::vector<std::pair<std::string, Method>>{
             {"dmcdm", Method::DualMesh}, {"zfvm", Method::FiniteVolumeCell}})
    {
      TransientOptions whole;
      whole.end_time = 1.0;
      whole.dt = 0.05;
      whole.theta = 0.5;
      DistributedOptions first_options;
      first_options.linear_tolerance = 1e-13;
      DistributedProblem uninterrupted(mesh, method, CoordinateSystem::Cartesian, first_options);
      define(uninterrupted.local());
      uninterrupted.solveTransient(whole);
      const auto expected = uninterrupted.gatheredValues("u");

      TransientOptions first_half = whole;
      first_half.end_time = 0.5;
      first_half.checkpoint_file = file;
      first_half.checkpoint_interval = 3;
      DistributedProblem stopped(mesh, method, CoordinateSystem::Cartesian, first_options);
      define(stopped.local());
      stopped.solveTransient(first_half);

      TransientOptions second_half = whole;
      second_half.restart_file = file;
      DistributedOptions other_options = first_options;
      other_options.partitioner = "simple";
      DistributedProblem restarted(mesh, method, CoordinateSystem::Cartesian, other_options);
      define(restarted.local());
      const auto result = restarted.solveTransient(second_half);
      const auto got = restarted.gatheredValues("u");
      double worst = got.size() == expected.size() ? 0.0 : 1e300;
      for (std::size_t n = 0; n < got.size() && n < expected.size(); ++n)
        worst = std::max(worst, std::abs(got[n] - expected[n]));
      std::ostringstream label;
      label << name
            << ": a run restarted from its checkpoint on another partition ends where an "
               "uninterrupted run ends ("
            << result.time_steps << " steps after the restart, worst difference " << worst << ")";
      check(worst < 1e-12 && result.time_steps == 10, label.str());
      comm.barrier();
      if (comm.isRoot())
        std::remove(file.c_str());
    }
  }

  // ---- the projection time integration of a flow ------------------------------
  // The decaying Taylor-Green vortex, with its velocity prescribed on the walls and in a periodic box, advanced by the projection splitting: the distributed run gives the serial velocity and, up to its constant, the serial pressure.
  {
    struct TaylorGreen : Function
    {
      int field = 0;
      double nu = 0.1;
      double value(const Point & x, double t) const override
      {
        const double decay = std::exp(-2.0 * nu * t);
        if (field == 0)
          return -std::cos(x[0]) * std::sin(x[1]) * decay;
        return std::sin(x[0]) * std::cos(x[1]) * decay;
      }
    };
    const double two_pi = 2.0 * M_PI;
    Mesh mesh = generateRectangleMesh(grid(12, 0.0, two_pi), grid(10, 0.0, two_pi), "Quad4");
    ProjectionSettings settings;
    settings.velocities = {"u", "v"};
    settings.pressure = "pressure";
    settings.density = 1.0;
    settings.dynamic_viscosity = 0.1;
    settings.order = 2;
    for (const auto & [method_name, method] :
         std::vector<std::pair<std::string, Method>>{{"fem", Method::FiniteElement},
                                                     {"dmcdm", Method::DualMesh},
                                                     {"hfvm", Method::FiniteVolumeVertex}})
      for (const bool periodic : {false, true})
      {
        const auto define = [&](Problem & problem)
        {
          for (int field = 0; field < 2; ++field)
          {
            auto exact = std::make_shared<TaylorGreen>();
            exact->field = field;
            const std::string name = field == 0 ? "u" : "v";
            problem.addVariable(name, {}, exact);
            problem.addFunction(name + "_exact", exact);
            // The projection solver assembles its own operators; a kernel marks the unknowns as active.
            problem.addObject("diffusion", name, params("diffusion", {{"variable", name}}));
            if (!periodic)
              problem.addObject(
                  "Dirichlet_boundary_condition",
                  name + "_walls",
                  params("Dirichlet_boundary_condition",
                         {{"variable", name},
                          {"boundary", std::vector<std::string>{"left", "right", "bottom", "top"}},
                          {"value", name + "_exact"}}));
          }
          problem.addVariable("pressure");
          problem.addObject("diffusion",
                            "pressure",
                            params("diffusion", {{"variable", std::string("pressure")}}));
          if (periodic)
            for (const auto & [name, primary, secondary] :
                 std::vector<std::tuple<std::string, std::string, std::string>>{
                     {"periodic_x", "left", "right"}, {"periodic_y", "bottom", "top"}})
              problem.addObject("periodic_boundary_condition",
                                name,
                                params("periodic_boundary_condition",
                                       {{"primary", primary}, {"secondary", secondary}}));
        };
        TransientOptions transient;
        transient.end_time = 0.4;
        transient.dt = 0.05;
        Problem reference(std::make_shared<Mesh>(mesh), method);
        define(reference);
        reference.setTimeIntegrator(
            std::make_shared<ProjectionSolver>(reference, nullptr, settings));
        reference.solveTransient(transient);
        DistributedProblem distributed(mesh, method);
        define(distributed.local());
        distributed.local().setTimeIntegrator(
            std::make_shared<ProjectionSolver>(distributed.local(), &distributed, settings));
        const auto result = distributed.solveTransient(transient);
        double worst_velocity = 0, worst_pressure = 0;
        for (const std::string name : {"u", "v"})
        {
          const auto got = distributed.gatheredValues(name);
          const auto expected = reference.values(name);
          for (std::size_t n = 0; n < got.size(); ++n)
            worst_velocity = std::max(worst_velocity, std::abs(got[n] - expected[n]));
        }
        const auto got = distributed.gatheredValues("pressure");
        const auto expected = reference.values("pressure");
        double got_mean = 0, expected_mean = 0;
        for (std::size_t n = 0; n < got.size(); ++n)
        {
          got_mean += got[n] / static_cast<double>(got.size());
          expected_mean += expected[n] / static_cast<double>(got.size());
        }
        for (std::size_t n = 0; n < got.size(); ++n)
          worst_pressure = std::max(worst_pressure,
                                    std::abs((got[n] - got_mean) - (expected[n] - expected_mean)));
        std::ostringstream label;
        label << "projection (" << method_name << "), Taylor-Green vortex "
              << (periodic ? "in a periodic box" : "with walls")
              << ": the distributed run gives the serial solution (" << result.time_steps
              << " steps, worst difference " << worst_velocity << " in the velocity and "
              << worst_pressure << " in the pressure)";
        check(result.time_steps == 8 && worst_velocity < 1e-7 && worst_pressure < 1e-7,
              label.str());
      }
  }

  // ---- the conjugate gradient method -------------------------------------
  // The Galerkin matrix of the finite element method is symmetric positive definite, so CG applies with a symmetric preconditioner.
  {
    Mesh mesh = generateRectangleMesh(grid(16), grid(16), "Quad4");
    auto reference_mesh = std::make_shared<Mesh>(mesh);
    Problem reference(reference_mesh, Method::FiniteElement);
    defineConduction(reference);
    reference.solveSteady();
    const auto expected = reference.values("temperature");
    for (const char * preconditioner : {"jacobi", "hypre", "gamg"})
    {
      DistributedOptions options;
      options.petsc_options = std::string("-ksp_type cg -pc_type ") + preconditioner;
      options.linear_tolerance = 1e-13;
      DistributedProblem distributed(
          mesh, Method::FiniteElement, CoordinateSystem::Cartesian, options);
      defineConduction(distributed.local());
      const auto result = distributed.solveSteady();
      const auto got = distributed.gatheredValues("temperature");
      check(got.size() == static_cast<std::size_t>(mesh.numNodes()),
            "the gathered values hold one value per node of the whole mesh");
      double worst = 0;
      for (Index n = 0; n < mesh.numNodes(); ++n)
        worst = std::max(worst, std::abs(got[n] - expected[n]));
      std::ostringstream label;
      label << "fem with cg and " << preconditioner << " matches the serial solution ("
            << result.linear_iterations << " iterations, worst difference " << worst << ")";
      check(worst < 1e-8, label.str());
    }
  }

  // ---- multigrid does not degrade as ranks are added --------------------
  // The automatic choice on the Poisson problem of a 48 by 48 mesh is GMRES with BoomerAMG, whose iteration count should not grow with the number of ranks.
  {
    Mesh mesh = generateRectangleMesh(grid(48), grid(48), "Quad4");
    DistributedOptions options;
    options.linear_tolerance = 1e-10;
    DistributedProblem distributed(mesh, Method::DualMesh, CoordinateSystem::Cartesian, options);
    Problem & problem = distributed.local();
    problem.addVariable("u");
    problem.addObject(
        "diffusion", "diffusion", params("diffusion", {{"variable", std::string("u")}}));
    problem.addObject("body_force",
                      "source",
                      params("body_force", {{"variable", std::string("u")}, {"value", 1.0}}));
    problem.addObject(
        "Dirichlet_boundary_condition",
        "walls",
        params("Dirichlet_boundary_condition",
               {{"variable", std::string("u")},
                {"boundary", std::vector<std::string>{"left", "right", "bottom", "top"}},
                {"value", 0.0}}));
    const auto result = distributed.solveSteady();
    std::ostringstream label;
    label << "GMRES with BoomerAMG takes " << result.linear_iterations << " iterations on "
          << comm.size() << " ranks (bound 20)";
    check(result.linear_iterations <= 20, label.str());
  }

  // ---- transient -----------------------------------------------------------
  {
    Mesh mesh = generateRectangleMesh(grid(10), grid(10), "Quad4");
    struct Hill : Function
    {
      double value(const Point & x, double) const override
      {
        return std::sin(M_PI * x[0]) * std::sin(M_PI * x[1]);
      }
    };
    const auto define = [](Problem & problem)
    {
      problem.addVariable("temperature", {}, std::make_shared<Hill>());
      problem.addObject("diffusion",
                        "diffusion",
                        params("diffusion", {{"variable", std::string("temperature")}}));
      problem.addObject("time_derivative",
                        "time",
                        params("time_derivative", {{"variable", std::string("temperature")}}));
      problem.addObject(
          "Dirichlet_boundary_condition",
          "edges",
          params("Dirichlet_boundary_condition",
                 {{"variable", std::string("temperature")},
                  {"boundary", std::vector<std::string>{"left", "right", "bottom", "top"}},
                  {"value", 0.0}}));
    };
    TransientOptions transient;
    transient.end_time = 0.02;
    transient.dt = 0.002;
    transient.theta = 0.5;

    auto reference_mesh = std::make_shared<Mesh>(mesh);
    Problem reference(reference_mesh, Method::DualMesh);
    define(reference);
    reference.solveTransient(transient);
    const auto expected = reference.values("temperature");

    DistributedOptions options;
    options.linear_tolerance = 1e-13;
    DistributedProblem distributed(mesh, Method::DualMesh, CoordinateSystem::Cartesian, options);
    define(distributed.local());
    distributed.solveTransient(transient);
    const auto got = distributed.gatheredValues("temperature");
    check(got.size() == static_cast<std::size_t>(mesh.numNodes()),
          "the gathered values hold one value per node of the whole mesh");
    double worst = 0;
    for (Index n = 0; n < mesh.numNodes(); ++n)
      worst = std::max(worst, std::abs(got[n] - expected[n]));
    std::ostringstream label;
    label << "a transient distributed solve matches the serial one (worst difference " << worst
          << ")";
    check(worst < 1e-8, label.str());
  }

  if (comm.isRoot())
    std::cout << "\n" << (g_failures ? "FAILED" : "all parallel tests passed") << "\n";
  return g_failures ? 1 : 0;
}
