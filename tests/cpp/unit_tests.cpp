// SPDX-License-Identifier: LGPL-2.1-or-later
//
// C++ unit tests (self-contained, no external test framework required).
// The extensive verification suite against Reddy's book lives in tests/python.
#include "dualmesh/base/Problem.h"
#include "dualmesh/core/ADReal.h"
#include "dualmesh/fe/Assembly.h"
#include "dualmesh/linalg/IncompleteLU.h"

#include <Eigen/IterativeLinearSolvers>

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

  // Copies carry exactly the derivatives in use, and a fused update equals
  // the expression it replaces.
  ADReal h = ADReal::independent(1.5, 2, 3);
  ADReal copy(h);
  CHECK(copy.size() == 3);
  CHECK_NEAR(copy.derivative(2), 1.0, 0.0);
  CHECK_NEAR(copy.derivative(0), 0.0, 0.0);
  copy = f;
  CHECK(copy.size() == 2);
  CHECK_NEAR(copy.derivative(1), f.derivative(1), 0.0);
  ADReal fused = f;
  fused.addScaledBy(h, -0.25);
  const ADReal plain = f + h * -0.25;
  CHECK(fused.size() == 3);
  CHECK_NEAR(fused.value(), plain.value(), 1e-15);
  for (int i = 0; i < 3; ++i)
    CHECK_NEAR(fused.derivative(i), plain.derivative(i), 1e-15);
}

TEST(simplex_quadrature_is_exact)
{
  // Every monomial x^i y^j z^k of degree up to the stated one is integrated
  // exactly over the reference simplex, where the exact value is
  // i! j! k! / (i + j + k + dim)!.
  const auto factorial = [](int n)
  {
    double f = 1;
    for (int k = 2; k <= n; ++k)
      f *= k;
    return f;
  };
  const std::pair<int, int> cases[] = {{2, 1}, {2, 4}, {2, 5}, {3, 1}, {3, 2}, {3, 3}, {3, 5}};
  for (const auto & [dim, degree] : cases)
  {
    std::vector<Point> x;
    std::vector<double> w;
    CHECK(simplexQuadrature(dim, degree, x, w));
    for (double wi : w)
      CHECK(wi > 0);
    for (const auto & p : x)
    {
      double sum = 0;
      for (int d = 0; d < dim; ++d)
      {
        CHECK(p[d] > 0);
        sum += p[d];
      }
      CHECK(sum < 1);
    }
    for (int i = 0; i <= degree; ++i)
      for (int j = 0; i + j <= degree; ++j)
        for (int k = 0; i + j + k <= degree; ++k)
        {
          if (dim == 2 && k > 0)
            continue;
          double q = 0;
          for (std::size_t n = 0; n < x.size(); ++n)
            q += w[n] * std::pow(x[n][0], i) * std::pow(x[n][1], j) * std::pow(x[n][2], k);
          const double exact =
              factorial(i) * factorial(j) * factorial(k) / factorial(i + j + k + dim);
          CHECK_NEAR(q, exact, 1e-15);
        }
  }
  // Beyond the tabulated degrees the caller falls back to the collapsed rule.
  std::vector<Point> x;
  std::vector<double> w;
  CHECK(!simplexQuadrature(3, 6, x, w));
}

TEST(incomplete_lu_without_fill)
{
  // On a tridiagonal matrix the exact LU factors have no fill, so ILU(0) is
  // the exact factorisation and one application solves the system.
  const int n = 50;
  std::vector<Eigen::Triplet<double>> t;
  for (int i = 0; i < n; ++i)
  {
    t.emplace_back(i, i, 4.0 + 0.1 * i);
    if (i > 0)
      t.emplace_back(i, i - 1, -1.0 - 0.01 * i);
    if (i + 1 < n)
      t.emplace_back(i, i + 1, -2.0);
  }
  SparseMatrix A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  Eigen::VectorXd x_exact = Eigen::VectorXd::LinSpaced(n, -1.0, 2.0);
  const Eigen::VectorXd b = A * x_exact;
  IncompleteLU0 ilu(A);
  CHECK(ilu.info() == Eigen::Success);
  CHECK((ilu.solve(b) - x_exact).norm() < 1e-12 * x_exact.norm());

  // On the five-point Laplacian the factors do fill, so ILU(0) is only a
  // preconditioner: BiCGSTAB with it converges, in far fewer iterations
  // than without it.
  const int m = 30, N = m * m;
  t.clear();
  for (int j = 0; j < m; ++j)
    for (int i = 0; i < m; ++i)
    {
      const int r = j * m + i;
      t.emplace_back(r, r, 4.0);
      if (i > 0)
        t.emplace_back(r, r - 1, -1.0);
      if (i + 1 < m)
        t.emplace_back(r, r + 1, -1.0);
      if (j > 0)
        t.emplace_back(r, r - m, -1.0);
      if (j + 1 < m)
        t.emplace_back(r, r + m, -1.0);
    }
  SparseMatrix L(N, N);
  L.setFromTriplets(t.begin(), t.end());
  const Eigen::VectorXd rhs = Eigen::VectorXd::Ones(N);
  Eigen::BiCGSTAB<SparseMatrix, IncompleteLU0> with;
  with.setTolerance(1e-10);
  with.compute(L);
  const Eigen::VectorXd y = with.solve(rhs);
  Eigen::BiCGSTAB<SparseMatrix, Eigen::IdentityPreconditioner> without;
  without.setTolerance(1e-10);
  without.compute(L);
  const Eigen::VectorXd z = without.solve(rhs);
  CHECK((L * z - rhs).norm() <= 1e-9 * rhs.norm());
  CHECK(with.info() == Eigen::Success);
  CHECK((L * y - rhs).norm() <= 1e-9 * rhs.norm());
  CHECK(2 * with.iterations() < without.iterations());
}

TEST(gauss_rules)
{
  for (int n = 1; n <= 20; ++n)
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

TEST(quadratic_dual_mesh_partitions_element)
{
  // The same two requirements as for the linear elements, on the quadratic
  // ones: the control domains of an element tile it exactly, and the area
  // vectors of every control domain close.  The quadratic simplices are the
  // interesting case, because their control domains are unions of several
  // sub-cells rather than single patches.
  for (auto [elem, dim] : std::vector<std::pair<std::string, int>>{
           {"Edge2", 1}, {"Quad4", 2}, {"Tri3", 2}, {"Hex8", 3}, {"Tet4", 3}})
  {
    Mesh linear =
        dim == 1
            ? generateLineMesh(linspace(0, 1.3, 3))
            : (dim == 2 ? generateRectangleMesh(linspace(0, 1.3, 3), linspace(0, 0.7, 3), elem)
                        : generateBoxMesh(
                              linspace(0, 1.3, 2), linspace(0, 0.7, 2), linspace(0, 0.9, 2), elem));
    Mesh m = linear.secondOrder();
    // Distort, so that the test does not rely on the element being straight.
    m.transformNodes([](const Point & p)
                     { return Point{p[0] + 0.1 * p[1] * p[0], p[1] + 0.05 * p[0], p[2]}; });
    for (Index e = 0; e < m.numElements(); ++e)
    {
      CHECK(elementIsQuadratic(m.element(e).type));
      std::vector<IntegrationPoint> pts;
      QuadratureSpec q;
      q.points = 4;
      buildElementPoints(m, e, true, PointSet::Volume, q, pts);
      double vol = 0;
      for (auto & p : pts)
        vol += p.weight;
      CHECK_NEAR(vol, elementMeasure(m, e), 1e-11);

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
        CHECK_NEAR(norm(c), 0.0, 1e-11);
    }
  }
}

TEST(prism_dual_mesh_partitions_element)
{
  // The prism's dual is the triangle's median dual times the halves of the
  // height.  It must tile the element and close, like every other dual.
  Mesh m = generateBoxMesh(linspace(0, 1.3, 2), linspace(0, 0.7, 2), linspace(0, 0.9, 2), "Wedge6");
  m.transformNodes(
      [](const Point & p)
      { return Point{p[0] + 0.1 * p[1] * p[0], p[1] + 0.05 * p[0], p[2] + 0.1 * p[0] * p[1]}; });
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

TEST(new_element_shape_functions)
{
  // The Kronecker property, the partition of unity and the vanishing sum of
  // the gradients, for the elements added for the finite element and finite
  // volume methods.
  for (auto type :
       {ElementType::Quad8, ElementType::Hex20, ElementType::Wedge6, ElementType::Pyramid5})
  {
    const auto & ref = ReferenceElement::get(type);
    double N[kMaxElementNodes];
    Point dN[kMaxElementNodes];
    for (int a = 0; a < ref.numNodes(); ++a)
    {
      // The pyramid's apex is a singular point of its rational basis, where
      // only the value (not the gradient) is defined; step just below it.
      Point xi = ref.node(a);
      if (type == ElementType::Pyramid5 && a == 4)
        xi[2] = 1 - 1e-9;
      ref.shape(xi, N, dN);
      for (int b = 0; b < ref.numNodes(); ++b)
        CHECK_NEAR(N[b], a == b ? 1.0 : 0.0, 1e-8);
    }
    for (Point xi : std::vector<Point>{{0.1, 0.2, 0.3}, {0.25, 0.15, 0.4}, {-0.2, 0.1, 0.5}})
    {
      if (!ref.contains(xi))
        continue;
      ref.shape(xi, N, dN);
      double sum = 0;
      Point gradient_sum{0, 0, 0};
      for (int a = 0; a < ref.numNodes(); ++a)
      {
        sum += N[a];
        gradient_sum = gradient_sum + dN[a];
      }
      CHECK_NEAR(sum, 1.0, 1e-12);
      CHECK_NEAR(norm(gradient_sum), 0.0, 1e-12);
      // The gradients agree with central differences of the values.
      for (int d = 0; d < ref.dimension(); ++d)
      {
        double Np[kMaxElementNodes], Nm[kMaxElementNodes];
        Point xp = xi, xm = xi;
        xp[d] += 1e-6;
        xm[d] -= 1e-6;
        ref.shape(xp, Np, nullptr);
        ref.shape(xm, Nm, nullptr);
        for (int a = 0; a < ref.numNodes(); ++a)
          CHECK_NEAR(dN[a][d], (Np[a] - Nm[a]) / 2e-6, 1e-7);
      }
    }
  }
}

TEST(quadratic_shape_functions)
{
  // A Lagrange basis is one at its own node and zero at the others, the basis
  // sums to one everywhere, and the gradients sum to zero.  Together these
  // guarantee that a constant is reproduced exactly and that the isoparametric
  // map is well posed.
  for (auto type : {ElementType::Edge3,
                    ElementType::Tri6,
                    ElementType::Quad9,
                    ElementType::Tet10,
                    ElementType::Hex27})
  {
    const auto & ref = ReferenceElement::get(type);
    double N[kMaxElementNodes];
    Point dN[kMaxElementNodes];
    for (int a = 0; a < ref.numNodes(); ++a)
    {
      ref.shape(ref.node(a), N, dN);
      for (int b = 0; b < ref.numNodes(); ++b)
        CHECK_NEAR(N[b], a == b ? 1.0 : 0.0, 1e-13);
    }
    // A few interior points.
    for (double t : {0.1, 0.25, 0.4})
    {
      Point xi{t, ref.dimension() > 1 ? 0.5 * t : 0.0, ref.dimension() > 2 ? 0.25 * t : 0.0};
      if (ref.isTensor())
        xi = {t, ref.dimension() > 1 ? -0.5 + t : 0.0, ref.dimension() > 2 ? 0.3 : 0.0};
      ref.shape(xi, N, dN);
      double sum = 0;
      Point gradient_sum{0, 0, 0};
      for (int a = 0; a < ref.numNodes(); ++a)
      {
        sum += N[a];
        gradient_sum = gradient_sum + dN[a];
      }
      CHECK_NEAR(sum, 1.0, 1e-12);
      CHECK_NEAR(norm(gradient_sum), 0.0, 1e-12);
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
