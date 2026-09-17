// SPDX-License-Identifier: LGPL-2.1-or-later
//
// C++ unit tests (self-contained, no external test framework required).
// The extensive verification suite against Reddy's book lives in tests/python.
#include "dualmesh/base/Problem.h"
#include "dualmesh/core/ADReal.h"
#include "dualmesh/fe/Assembly.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace dualmesh;

namespace
{
struct TestCase
{
  std::string name;
  std::function<void()> body;
};
std::vector<TestCase> &
registry()
{
  static std::vector<TestCase> r;
  return r;
}
struct Registrar
{
  Registrar(const std::string & n, std::function<void()> f) { registry().push_back({n, f}); }
};
int g_failures = 0;
#define TEST(name)                                                                                 \
  void name();                                                                                     \
  Registrar reg_##name(#name, name);                                                               \
  void name()
#define CHECK_NEAR(a, b, tol)                                                                      \
  do                                                                                               \
  {                                                                                                \
    const double va = (a), vb = (b);                                                               \
    if (!(std::abs(va - vb) <= (tol)))                                                             \
    {                                                                                              \
      std::ostringstream os;                                                                       \
      os << __FILE__ << ":" << __LINE__ << ": " #a " = " << va << " != " #b " = " << vb            \
         << " (tol " << (tol) << ")";                                                              \
      throw std::runtime_error(os.str());                                                          \
    }                                                                                              \
  } while (0)
#define CHECK(c)                                                                                   \
  do                                                                                               \
  {                                                                                                \
    if (!(c))                                                                                      \
      throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +            \
                               ": check failed: " #c);                                             \
  } while (0)

InputParameters
params(const std::string & type, std::initializer_list<std::pair<std::string, ParameterValue>> kv)
{
  auto p = Factory::instance().validParams(type);
  for (const auto & [k, v] : kv)
    p.set(k, v);
  return p;
}

std::vector<double>
linspace(double a, double b, int n)
{
  std::vector<double> x(n + 1);
  for (int i = 0; i <= n; ++i)
    x[i] = a + (b - a) * i / n;
  return x;
}
} // namespace

// ---------------------------------------------------------------------------
TEST(ad_arithmetic)
{
  ADReal x = ADReal::independent(2.0, 0, 2);
  ADReal y = ADReal::independent(3.0, 1, 2);
  ADReal f = x * x * y + sin(x) / y - pow(y, 2.5);
  CHECK_NEAR(f.value(), 12 + std::sin(2.0) / 3 - std::pow(3.0, 2.5), 1e-14);
  CHECK_NEAR(f.derivative(0), 2 * 2 * 3 + std::cos(2.0) / 3, 1e-14);
  CHECK_NEAR(f.derivative(1), 4 - std::sin(2.0) / 9 - 2.5 * std::pow(3.0, 1.5), 1e-13);
  ADReal g = x;
  g *= g;
  CHECK_NEAR(g.derivative(0), 4.0, 1e-15);
}

TEST(gauss_rules)
{
  for (int n = 1; n <= 8; ++n)
  {
    QuadratureSpec q;
    q.points = n;
    std::vector<double> x, w;
    rule1D(q, x, w);
    // exact for x^(2n-1) and x^(2n-2)
    double s = 0, t = 0;
    for (int i = 0; i < n; ++i)
    {
      s += w[i] * std::pow(x[i], 2 * n - 2);
      t += w[i] * std::pow(x[i], 2 * n - 1);
    }
    CHECK_NEAR(s, 2.0 / (2 * n - 1), 1e-13);
    CHECK_NEAR(t, 0.0, 1e-13);
  }
}

TEST(dual_mesh_partitions_element)
{
  // The control volumes of an element must tile it, and the interface
  // area vectors of every control volume must close with the boundary.
  for (auto [elem, dim] : std::vector<std::pair<std::string, int>>{
           {"Quad4", 2}, {"Tri3", 2}, {"Hex8", 3}, {"Tet4", 3}})
  {
    Mesh m = dim == 2 ? generateRectangleMesh({0, 1.3}, {0, 0.7}, elem)
                      : generateBoxMesh({0, 1.3}, {0, 0.7}, {0, 0.9}, elem);
    // distort a node
    m.transformNodes([](const Point & p)
                     { return Point{p[0] + 0.1 * p[1] * p[0], p[1] + 0.05 * p[0], p[2]}; });
    for (Index e = 0; e < m.numElements(); ++e)
    {
      std::vector<IntegrationPoint> pts;
      QuadratureSpec q;
      q.points = 3;
      buildElementPoints(m, e, true, PointSet::Volume, q, pts);
      double vol = 0;
      for (auto & p : pts)
        vol += p.weight;
      CHECK_NEAR(vol, elementMeasure(m, e), 1e-12);
      // closure: sum over interfaces (signed) + element boundary pieces = 0
      const auto & el = m.element(e);
      std::vector<Point> closure(el.numNodes(), Point{0, 0, 0});
      buildElementPoints(m, e, true, PointSet::Faces, q, pts);
      for (auto & p : pts)
      {
        closure[p.owner] = closure[p.owner] + p.area;
        closure[p.neighbor] = closure[p.neighbor] - p.area;
      }
      const auto & ref = ReferenceElement::get(el.type);
      for (int s = 0; s < ref.numSides(); ++s)
      {
        buildSidePoints(m, {e, s}, true, q, pts);
        for (auto & p : pts)
          closure[p.owner] = closure[p.owner] + p.area;
      }
      for (auto & c : closure)
        CHECK_NEAR(norm(c), 0.0, 1e-12);
    }
  }
}

TEST(reddy_example_5_3_1_linear_fin)
{
  // -u'' + 400 u = 0, u(0) = 300, u'(L) + 2 u(L) = 0, L = 0.05, 5 elements.
  auto mesh = std::make_shared<Mesh>(generateLineMesh(linspace(0, 0.05, 5)));
  Problem pb(mesh);
  pb.addVariable("u");
  pb.addObject("Diffusion", "diff", params("Diffusion", {{"variable", std::string("u")}}));
  pb.addObject("Reaction",
               "react",
               params("Reaction", {{"variable", std::string("u")}, {"coefficient", 400.0}}));
  pb.addObject("DirichletBC",
               "left",
               params("DirichletBC",
                      {{"variable", std::string("u")},
                       {"boundary", std::vector<std::string>{"left"}},
                       {"value", 300.0}}));
  pb.addObject("RobinBC",
               "right",
               params("RobinBC",
                      {{"variable", std::string("u")},
                       {"boundary", std::vector<std::string>{"right"}},
                       {"transfer_coefficient", 2.0}}));
  pb.solveSteady();
  const auto u = pb.values("u");
  const double expected[] = {300.0, 257.62, 225.59, 202.64, 187.83, 180.57};
  for (int i = 0; i < 6; ++i)
    CHECK_NEAR(u[i], expected[i], 0.006);
}

TEST(patch_test_linear_field)
{
  // A linear field is reproduced exactly on distorted meshes (all elements).
  for (std::string elem : {"Quad4", "Tri3", "Hex8", "Tet4"})
  {
    const bool three = elem == "Hex8" || elem == "Tet4";
    Mesh base = three
                    ? generateBoxMesh(linspace(0, 1, 3), linspace(0, 1, 3), linspace(0, 1, 3), elem)
                    : generateRectangleMesh(linspace(0, 1, 4), linspace(0, 1, 4), elem);
    base.transformNodes(
        [](const Point & p)
        {
          const double b = p[0] * (1 - p[0]) * p[1] * (1 - p[1]);
          return Point{p[0] + 0.3 * b, p[1] - 0.2 * b, p[2]};
        });
    auto mesh = std::make_shared<Mesh>(base);
    for (auto method : {Method::DualMesh, Method::FiniteElement})
    {
      Problem pb(mesh, method);
      pb.addVariable("u");
      pb.addObject("Diffusion", "", params("Diffusion", {{"variable", std::string("u")}}));
      std::vector<std::string> all = {"left", "right", "bottom", "top"};
      if (three)
      {
        all.push_back("back");
        all.push_back("front");
      }
      struct Lin : Function
      {
        double value(const Point & x, double) const override
        {
          return 1 + 2 * x[0] - 3 * x[1] + 0.5 * x[2];
        }
      };
      pb.addFunction("exact", std::make_shared<Lin>());
      pb.addObject("DirichletBC",
                   "",
                   params("DirichletBC",
                          {{"variable", std::string("u")},
                           {"boundary", all},
                           {"value", std::string("exact")}}));
      pb.solveSteady();
      Lin exact;
      for (Index n = 0; n < mesh->numNodes(); ++n)
        CHECK_NEAR(pb.solution()[n], exact.value(mesh->node(n), 0), 1e-10);
    }
  }
}

int
main()
{
  int passed = 0;
  for (const auto & t : registry())
  {
    try
    {
      t.body();
      ++passed;
      std::cout << "[  OK  ] " << t.name << "\n";
    }
    catch (const std::exception & e)
    {
      ++g_failures;
      std::cout << "[ FAIL ] " << t.name << ": " << e.what() << "\n";
    }
  }
  std::cout << passed << " passed, " << g_failures << " failed\n";
  return g_failures ? 1 : 0;
}
