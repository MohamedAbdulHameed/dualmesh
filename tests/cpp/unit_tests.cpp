// SPDX-License-Identifier: LGPL-2.1-or-later
//
// C++ unit tests (self-contained, no external test framework required).
// The extensive verification suite against Reddy's book lives in tests/python.
#include "dualmesh/base/Problem.h"
#include "dualmesh/core/ADReal.h"
#include "dualmesh/core/Equation.h"
#include "dualmesh/core/ParsedFunction.h"
#include "dualmesh/fe/Assembly.h"
#include "dualmesh/linalg/LinearSolverChoice.h"
#include "dualmesh/mesh/Bisection.h"
#include "dualmesh/mesh/Subdivision.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
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

TEST(parsed_expression_infinite_slope_times_zero_derivative)
{
  // [1 - exp(-d/c)]^(2/3) at d = 0: the power has an infinite slope, but the
  // base does not depend on c there, so the c-derivative is zero, not NaN.
  // Where the base does depend on an unknown (d), the derivative is infinite.
  const ParsedExpression e("(1 - exp(-d/c))^(2/3)", {"d", "c"});
  const ADReal args[2] = {ADReal::independent(0.0, 0, 2), ADReal::independent(0.3, 1, 2)};
  const ADReal f = e.value(args);
  CHECK_NEAR(f.value(), 0.0, 0.0);
  CHECK(f.derivative(1) == 0.0);
  CHECK(std::isinf(f.derivative(0)));
  // A finite slope is unchanged: sqrt at 4 has slope 1/4.
  const ParsedExpression g("sqrt(x)", {"x"});
  const ADReal x[1] = {ADReal::independent(4.0, 0, 1)};
  CHECK_NEAR(g.value(x).derivative(0), 0.25, 1e-15);
}

TEST(the_size_of_terms_that_cancel_is_kept)
{
  // mu (F - 1/F) at F = 1 + 1e-7: the value is about 2e-7 mu, and its terms are of the size of mu,
  // so its round-off is about eps mu, a relative error of 1e-9 of the value.  The size is
  // mu (|F| + 2/|F|): the quotient 1/F counts its own size and that of F carried by the derivative.
  const ParsedExpression stress("mu*(F - 1/F)", {"mu", "F"});
  const double mu = 4.0e6;
  const double variables[2] = {mu, 1.0 + 1e-7};
  CHECK_NEAR(stress.value(variables), mu * 2e-7, 1e-6 * mu * 2e-7);
  const double size = stress.termSize(variables);
  CHECK_NEAR(size, 3.0 * mu, 1e-6 * mu);
  // Terms that do not cancel have the size of the value.
  const ParsedExpression sum("a + b*c", {"a", "b", "c"});
  const double positive[3] = {1.0, 2.0, 3.0};
  CHECK_NEAR(sum.termSize(positive), 7.0, 1e-15);
  // A function carries the size of its argument by its derivative: exp(x - x) at x = 10.
  const ParsedExpression exponential("exp(x - x)", {"x"});
  const double ten[1] = {10.0};
  CHECK_NEAR(exponential.termSize(ten), 1.0 + 20.0, 1e-12);
}

TEST(equations_that_cannot_be_sorted_are_refused)
{
  EquationScope scope;
  scope.dimension = 2;
  scope.fields = {{"u", 0}};
  for (const std::string bad :
       {"x*laplacian(u) = 1", "sin(laplacian(u)) = 1", "laplacian(u) +", "du/dt = laplacian(w)"})
  {
    bool refused = false;
    try
    {
      reduceEquations(scope, {bad});
    }
    catch (const InputError &)
    {
      refused = true;
    }
    CHECK(refused);
  }
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

TEST(corner_shape_functions)
{
  // A first-order variable on a quadratic element (the pressure of the
  // Taylor-Hood element) is interpolated with the shape functions of the
  // linear element with the same corners.  This rests on three facts that
  // are checked here for every quadratic type: the corners are the first
  // nodes of the quadratic element and sit where the linear element's nodes
  // sit; the corner basis is a partition of unity; and on an affinely mapped
  // element it reproduces a linear field and its gradient exactly, with the
  // gradients taken through the quadratic element's own Jacobian.
  for (auto type : {ElementType::Edge3,
                    ElementType::Tri6,
                    ElementType::Quad9,
                    ElementType::Quad8,
                    ElementType::Tet10,
                    ElementType::Hex27,
                    ElementType::Hex20})
  {
    const auto & quadratic = ReferenceElement::get(type);
    const auto & linear = ReferenceElement::get(elementCornerType(type));
    const int dim = quadratic.dimension();
    const int corners = linear.numNodes();
    CHECK(corners <= MappedPoint::kMaxCorners);
    for (int a = 0; a < corners; ++a)
      CHECK_NEAR(norm(quadratic.node(a) - linear.node(a)), 0.0, 1e-15);
    // An affine map x = M xi + c that shears and stretches the element.
    const double M[3][3] = {{1.3, 0.4, -0.2}, {-0.3, 0.9, 0.25}, {0.1, -0.2, 1.1}};
    const Point c{0.2, -0.7, 0.5};
    const auto map = [&](const Point & xi)
    {
      Point x{0, 0, 0};
      for (int i = 0; i < dim; ++i)
      {
        x[i] = c[i];
        for (int j = 0; j < dim; ++j)
          x[i] += M[i][j] * xi[j];
      }
      return x;
    };
    Mesh mesh(dim);
    std::vector<Index> nodes;
    for (int a = 0; a < quadratic.numNodes(); ++a)
      nodes.push_back(mesh.addNode(map(quadratic.node(a))));
    mesh.addElement(type, nodes);
    const Element & el = mesh.element(0);
    // A linear field f(x) = 2 + g . x.
    const Point g{0.7, -1.9, 0.45};
    const auto field = [&](const Point & x)
    {
      double f = 2.0;
      for (int i = 0; i < dim; ++i)
        f += g[i] * x[i];
      return f;
    };
    for (double t : {0.1, 0.25, 0.4})
    {
      Point xi{t, dim > 1 ? 0.5 * t : 0.0, dim > 2 ? 0.25 * t : 0.0};
      if (quadratic.isTensor() || type == ElementType::Quad8 || type == ElementType::Hex20)
        xi = {t, dim > 1 ? -0.5 + t : 0.0, dim > 2 ? 0.3 : 0.0};
      MappedPoint mp;
      mapPoint(mesh, el, xi, mp);
      CHECK(mp.num_corners == 0);
      mapCornerShape(el, mp);
      CHECK(mp.num_corners == corners);
      double sum = 0, value = 0;
      Point gradient_sum{0, 0, 0}, gradient{0, 0, 0};
      for (int a = 0; a < corners; ++a)
      {
        const double fa = field(mesh.node(el.nodes[a]));
        sum += mp.corner_shape[a];
        value += mp.corner_shape[a] * fa;
        gradient_sum = gradient_sum + mp.corner_gradient[a];
        gradient = gradient + fa * mp.corner_gradient[a];
      }
      CHECK_NEAR(sum, 1.0, 1e-13);
      CHECK_NEAR(norm(gradient_sum), 0.0, 1e-12);
      CHECK_NEAR(value, field(mp.x), 1e-12);
      for (int i = 0; i < dim; ++i)
        CHECK_NEAR(gradient[i], g[i], 1e-12);
    }
  }
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
      pb.addObject("diffusion", "", params("diffusion", {{"variable", std::string("u")}}));
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
      pb.addObject("Dirichlet",
                   "",
                   params("Dirichlet",
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

TEST(every_side_is_wound_outward_and_its_area_vectors_close)
{
  // One element of every type, the reference element under a sheared map
  // with a positive determinant.  The right-hand normal of each side, from
  // the physical positions of its nodes in their stored order, points away
  // from the element, and the side area vectors satisfy the divergence
  // theorem: they sum to zero, and the integral of x . n over the boundary
  // is the dimension times the measure.
  const double A[3][3] = {{1.2, 0.3, 0.1}, {0.1, 0.9, 0.2}, {0.0, 0.2, 1.1}};
  for (int k = 0; k < kNumElementTypes; ++k)
  {
    const auto type = static_cast<ElementType>(k);
    const auto & ref = ReferenceElement::get(type);
    const int dim = ref.dimension();
    Mesh m(dim);
    std::vector<Index> nodes;
    for (int i = 0; i < ref.numNodes(); ++i)
    {
      Point x{0.5, -0.25, 0.75};
      for (int r = 0; r < dim; ++r)
        for (int c = 0; c < dim; ++c)
          x[r] += A[r][c] * ref.node(i)[c];
      for (int r = dim; r < 3; ++r)
        x[r] = 0.0;
      nodes.push_back(m.addNode(x));
    }
    m.addElement(type, nodes);
    const Point centre = m.elementCentroid(0);
    Point closure{0, 0, 0};
    double flux_of_x = 0;
    QuadratureSpec q;
    q.points = 3;
    std::vector<IntegrationPoint> pts;
    MappedPoint mp;
    for (int s = 0; s < ref.numSides(); ++s)
    {
      const auto sn = m.sideNodes({0, s});
      const Point a = m.node(sn[0]);
      Point n{0, 0, 0};
      if (dim == 1)
        n = a - centre;
      else if (dim == 2)
        n = {m.node(sn[1])[1] - a[1], a[0] - m.node(sn[1])[0], 0};
      else
        n = cross(m.node(sn[1]) - a, m.node(sn[2]) - a);
      CHECK(dot(n, m.sideCentroid({0, s}) - centre) > 0);
      buildSidePoints(m, {0, s}, false, q, pts);
      for (const auto & p : pts)
      {
        mapPoint(m, m.element(0), p.xi, mp);
        closure = closure + p.area;
        flux_of_x += dot(mp.x, p.area);
      }
    }
    CHECK_NEAR(norm(closure), 0.0, 1e-12);
    CHECK_NEAR(flux_of_x, dim * elementMeasure(m, 0), 1e-12);
  }
}

TEST(the_options_of_the_user_are_sorted_by_solver)
{
  const user_options_parts parts = user_choice(
      "-ksp_type gmres -snes_linesearch_damping 0.5 -dualmesh_schur pcd -pc_type ilu -snes_monitor "
      "-ksp_rtol 1e-8");
  CHECK(parts.split.Schur == SchurApproximation::ConvectionDiffusion);
  CHECK(!parts.split.physics);
  CHECK(parts.linear == "-ksp_type gmres -pc_type ilu -ksp_rtol 1e-8");
  CHECK(parts.nonlinear == "-snes_linesearch_damping 0.5 -snes_monitor");
  // A negative number is a value, not an option.
  CHECK(user_choice("-snes_linesearch_damping -0.5").nonlinear == "-snes_linesearch_damping -0.5");
  CHECK(user_choice("-dualmesh_split physics").split.physics);
  for (const std::string bad : {"-dualmesh_schur lcs",
                                "-dualmesh_schur",
                                "-dualmesh_split blocks",
                                "-dualmesh_schur lsc -dualmesh_split physics"})
  {
    bool refused = false;
    try
    {
      user_choice(bad);
    }
    catch (const InputError &)
    {
      refused = true;
    }
    CHECK(refused);
  }
}

TEST(the_linear_solvers_are_tried_in_order)
{
  using kind = linear_solve_kind;
  linear_system_traits system;
  system.dimension = 3;
  system.global_unknowns = 100000;
  system.variables = 3;
  system.displacement_variables = 3;
  // A large solid: smoothed aggregation, which may fail, then the direct solver.
  std::vector<linear_solve_attempt> attempts = linear_solve_attempts(system, {});
  CHECK(attempts.size() == 2);
  CHECK(attempts[0].kind == kind::GAMG && !attempts[0].must_converge);
  CHECK(attempts[1].kind == kind::direct);
  // A small one is factorised.
  system.global_unknowns = 4000;
  attempts = linear_solve_attempts(system, {});
  CHECK(attempts.size() == 1 && attempts[0].kind == kind::direct);
  // A large flow: the least squares commutator, then the direct solver.
  system.global_unknowns = 100000;
  system.displacement_variables = 0;
  system.flow = true;
  attempts = linear_solve_attempts(system, {});
  CHECK(attempts.size() == 2 && attempts[0].kind == kind::Schur_split);
  CHECK(attempts[0].Schur_approximation == SchurApproximation::LeastSquaresCommutator);
  // A split that the user chose must converge, and so must the options of the user.
  user_split split;
  split.Schur = SchurApproximation::PressureMass;
  attempts = linear_solve_attempts(system, split);
  CHECK(attempts.size() == 1 && attempts[0].must_converge);
  CHECK(attempts[0].Schur_approximation == SchurApproximation::PressureMass);
  system.user_options = "-ksp_type cg";
  attempts = linear_solve_attempts(system, {});
  CHECK(attempts.size() == 1 && attempts[0].kind == kind::user_options);
  // Any other large system: BoomerAMG, then the direct solver; one in one dimension is factorised.
  system.user_options.clear();
  system.flow = false;
  attempts = linear_solve_attempts(system, {});
  CHECK(attempts.size() == 2 && attempts[0].kind == kind::BoomerAMG);
  system.dimension = 1;
  attempts = linear_solve_attempts(system, {});
  CHECK(attempts.size() == 1 && attempts[0].kind == kind::direct);
}

namespace
{
/// Whether every edge of a triangular mesh is a side of one triangle (on the boundary) or of two
/// (inside), which holds when no node hangs.
bool
conforming(const Mesh & mesh)
{
  std::map<std::pair<Index, Index>, int> count;
  for (Index e = 0; e < mesh.numElements(); ++e)
    for (int k = 0; k < 3; ++k)
    {
      const std::vector<Index> ends = mesh.sideNodes({e, k});
      ++count[{std::min(ends[0], ends[1]), std::max(ends[0], ends[1])}];
    }
  std::size_t boundary = 0;
  for (const auto & [edge, n] : count)
  {
    if (n > 2)
      return false;
    boundary += n == 1;
  }
  return boundary == mesh.exteriorSides().size();
}

double
area(const Mesh & mesh)
{
  double sum = 0.0;
  for (Index e = 0; e < mesh.numElements(); ++e)
    sum += elementMeasure(mesh, e);
  return sum;
}

double
sideLength(const Mesh & mesh, const std::string & name)
{
  double sum = 0.0;
  for (const Side & side : mesh.sidesets().at(name))
  {
    const std::vector<Index> ends = mesh.sideNodes(side);
    sum += norm(mesh.node(ends[0]) - mesh.node(ends[1]));
  }
  return sum;
}
} // namespace

TEST(newest_vertex_bisection_refines_and_coarsens_conformingly)
{
  const Mesh initial = generateRectangleMesh(linspace(0, 2, 4), linspace(0, 1, 2), "Tri3");
  Bisection hierarchy(initial);
  // Refine the triangles near the corner (0, 0) five times.
  for (int cycle = 0; cycle < 5; ++cycle)
  {
    const Mesh & mesh = hierarchy.mesh();
    std::vector<char> marked(static_cast<std::size_t>(mesh.numElements()), 0);
    for (Index e = 0; e < mesh.numElements(); ++e)
    {
      Point centre{0, 0, 0};
      for (int k = 0; k < mesh.element(e).numNodes(); ++k)
        centre = centre + (1.0 / 3.0) * mesh.node(mesh.element(e).nodes[k]);
      marked[static_cast<std::size_t>(e)] = norm(centre) < 0.5;
    }
    CHECK(hierarchy.refine(marked, 100));
    CHECK(conforming(hierarchy.mesh()));
    CHECK_NEAR(area(hierarchy.mesh()), 2.0, 1e-12);
    CHECK_NEAR(sideLength(hierarchy.mesh(), "left"), 1.0, 1e-12);
    CHECK_NEAR(sideLength(hierarchy.mesh(), "bottom"), 2.0, 1e-12);
  }
  CHECK(hierarchy.mesh().numElements() > 4 * initial.numElements());
  // A linear field transfers exactly to the refined mesh.
  const auto linear = [](const Point & x) { return 3.0 * x[0] - 2.0 * x[1] + 1.0; };
  // Coarsening every triangle repeatedly goes back to the initial mesh, and the field with it.
  for (int cycle = 0; cycle < 20; ++cycle)
  {
    const Mesh before = hierarchy.mesh();
    Eigen::VectorXd values(before.numNodes());
    for (Index n = 0; n < before.numNodes(); ++n)
      values[n] = linear(before.node(n));
    const bool changed =
        hierarchy.coarsen(std::vector<char>(static_cast<std::size_t>(before.numElements()), 1));
    CHECK(conforming(hierarchy.mesh()));
    CHECK_NEAR(area(hierarchy.mesh()), 2.0, 1e-12);
    const Eigen::VectorXd after = hierarchy.transfer(values, 1);
    for (Index n = 0; n < hierarchy.mesh().numNodes(); ++n)
      CHECK_NEAR(after[n], linear(hierarchy.mesh().node(n)), 1e-12);
    if (!changed)
      break;
  }
  CHECK(hierarchy.mesh().numElements() == initial.numElements());
  CHECK_NEAR(sideLength(hierarchy.mesh(), "top"), 2.0, 1e-12);
  for (int g : hierarchy.generations())
    CHECK(g == 0);
  // Refinement transfers a linear field exactly too.
  Eigen::VectorXd values(hierarchy.mesh().numNodes());
  for (Index n = 0; n < hierarchy.mesh().numNodes(); ++n)
    values[n] = linear(hierarchy.mesh().node(n));
  CHECK(hierarchy.refine(std::vector<char>(static_cast<std::size_t>(initial.numElements()), 1), 1));
  const Eigen::VectorXd refined = hierarchy.transfer(values, 1);
  for (Index n = 0; n < hierarchy.mesh().numNodes(); ++n)
    CHECK_NEAR(refined[n], linear(hierarchy.mesh().node(n)), 1e-12);
  // The generation caps the refinement of a marked triangle.
  CHECK(!hierarchy.refine(std::vector<char>(hierarchy.mesh().numElements(), 1), 1));
}

namespace
{
/// Whether every face of a tetrahedral mesh is a side of one tetrahedron (on the boundary) or of
/// two (inside), and every tetrahedron has a positive volume.
bool
conformingAndOriented(const Mesh & mesh)
{
  std::map<std::array<Index, 3>, int> count;
  for (Index e = 0; e < mesh.numElements(); ++e)
  {
    const auto & n = mesh.element(e).nodes;
    const Point a = mesh.node(n[0]);
    if (dot(mesh.node(n[1]) - a, cross(mesh.node(n[2]) - a, mesh.node(n[3]) - a)) <= 0.0)
      return false;
    for (int k = 0; k < 4; ++k)
    {
      std::vector<Index> c = mesh.sideNodes({e, k});
      std::sort(c.begin(), c.end());
      ++count[{c[0], c[1], c[2]}];
    }
  }
  std::size_t boundary = 0;
  for (const auto & [face, n] : count)
  {
    if (n > 2)
      return false;
    boundary += n == 1;
  }
  return boundary == mesh.exteriorSides().size();
}

double
volume(const Mesh & mesh)
{
  double sum = 0.0;
  for (Index e = 0; e < mesh.numElements(); ++e)
    sum += elementMeasure(mesh, e);
  return sum;
}

double
sideArea(const Mesh & mesh, const std::string & name)
{
  double sum = 0.0;
  for (const Side & side : mesh.sidesets().at(name))
  {
    // A triangle, or a planar quadrilateral by its diagonals.
    const std::vector<Index> c = mesh.sideNodes(side);
    sum += c.size() == 4 ? 0.5 * norm(cross(mesh.node(c[2]) - mesh.node(c[0]),
                                            mesh.node(c[3]) - mesh.node(c[1])))
                         : 0.5 * norm(cross(mesh.node(c[1]) - mesh.node(c[0]),
                                            mesh.node(c[2]) - mesh.node(c[0])));
  }
  return sum;
}
} // namespace

TEST(longest_edge_bisection_refines_and_coarsens_tetrahedra_conformingly)
{
  const Mesh initial =
      generateBoxMesh(linspace(0, 2, 2), linspace(0, 1, 2), linspace(0, 1, 2), "Tet4");
  CHECK(conformingAndOriented(initial));
  Bisection hierarchy(initial);
  for (int cycle = 0; cycle < 4; ++cycle)
  {
    const Mesh & mesh = hierarchy.mesh();
    std::vector<char> marked(static_cast<std::size_t>(mesh.numElements()), 0);
    for (Index e = 0; e < mesh.numElements(); ++e)
    {
      Point centre{0, 0, 0};
      for (int k = 0; k < mesh.element(e).numNodes(); ++k)
        centre = centre + 0.25 * mesh.node(mesh.element(e).nodes[k]);
      marked[static_cast<std::size_t>(e)] = norm(centre) < 0.6;
    }
    CHECK(hierarchy.refine(marked, 100));
    CHECK(conformingAndOriented(hierarchy.mesh()));
    CHECK_NEAR(volume(hierarchy.mesh()), 2.0, 1e-12);
    CHECK_NEAR(sideArea(hierarchy.mesh(), "left"), 1.0, 1e-12);
    CHECK_NEAR(sideArea(hierarchy.mesh(), "back"), 2.0, 1e-12);
  }
  CHECK(hierarchy.mesh().numElements() > 2 * initial.numElements());
  const auto linear = [](const Point & x) { return 3.0 * x[0] - 2.0 * x[1] + x[2] + 1.0; };
  for (int cycle = 0; cycle < 40; ++cycle)
  {
    const Mesh before = hierarchy.mesh();
    Eigen::VectorXd values(before.numNodes());
    for (Index n = 0; n < before.numNodes(); ++n)
      values[n] = linear(before.node(n));
    const bool changed =
        hierarchy.coarsen(std::vector<char>(static_cast<std::size_t>(before.numElements()), 1));
    CHECK(conformingAndOriented(hierarchy.mesh()));
    CHECK_NEAR(volume(hierarchy.mesh()), 2.0, 1e-12);
    const Eigen::VectorXd after = hierarchy.transfer(values, 1);
    for (Index n = 0; n < hierarchy.mesh().numNodes(); ++n)
      CHECK_NEAR(after[n], linear(hierarchy.mesh().node(n)), 1e-12);
    if (!changed)
      break;
  }
  CHECK(hierarchy.mesh().numElements() == initial.numElements());
  CHECK_NEAR(sideArea(hierarchy.mesh(), "front"), 2.0, 1e-12);
  // Refinement transfers a linear field exactly too.
  Eigen::VectorXd values(hierarchy.mesh().numNodes());
  for (Index n = 0; n < hierarchy.mesh().numNodes(); ++n)
    values[n] = linear(hierarchy.mesh().node(n));
  CHECK(hierarchy.refine(std::vector<char>(static_cast<std::size_t>(initial.numElements()), 1), 3));
  const Eigen::VectorXd refined = hierarchy.transfer(values, 1);
  for (Index n = 0; n < hierarchy.mesh().numNodes(); ++n)
    CHECK_NEAR(refined[n], linear(hierarchy.mesh().node(n)), 1e-12);
}

TEST(quadratic_simplices_refine_on_their_curved_geometry_and_carry_quadratic_fields)
{
  // A quadratic field transfers exactly, through refinement and coarsening, on Tri6 and Tet10.
  const auto quadratic = [](const Point & x)
  { return x[0] * x[0] + x[0] * x[1] - 2.0 * x[1] * x[1] + 0.5 * x[2] * x[0] + 3.0 * x[0] + 1.0; };
  for (const Mesh & initial :
       {generateRectangleMesh(linspace(0, 2, 3), linspace(0, 1, 2), "Tri3").secondOrder(),
        generateBoxMesh(linspace(0, 2, 2), linspace(0, 1, 2), linspace(0, 1, 2), "Tet4")
            .secondOrder()})
  {
    Bisection hierarchy(initial);
    const auto check = [&](const Eigen::VectorXd & values)
    {
      for (Index n = 0; n < hierarchy.mesh().numNodes(); ++n)
        CHECK_NEAR(values[n], quadratic(hierarchy.mesh().node(n)), 1e-12);
    };
    for (int cycle = 0; cycle < 3; ++cycle)
    {
      const Mesh before = hierarchy.mesh();
      Eigen::VectorXd values(before.numNodes());
      for (Index n = 0; n < before.numNodes(); ++n)
        values[n] = quadratic(before.node(n));
      std::vector<char> marked(static_cast<std::size_t>(before.numElements()), 0);
      marked[0] = 1;
      CHECK(hierarchy.refine(marked, 100));
      check(hierarchy.transfer(values, 1));
      CHECK_NEAR(
          initial.dimension() == 2 ? area(hierarchy.mesh()) : volume(hierarchy.mesh()), 2.0, 1e-12);
    }
    for (int cycle = 0; cycle < 20; ++cycle)
    {
      const Mesh before = hierarchy.mesh();
      Eigen::VectorXd values(before.numNodes());
      for (Index n = 0; n < before.numNodes(); ++n)
        values[n] = quadratic(before.node(n));
      const bool changed =
          hierarchy.coarsen(std::vector<char>(static_cast<std::size_t>(before.numElements()), 1));
      check(hierarchy.transfer(values, 1));
      if (!changed)
        break;
    }
    CHECK(hierarchy.mesh().numElements() == initial.numElements());
    CHECK(hierarchy.mesh().numNodes() == initial.numNodes());
  }

  // One Tri6 whose long side bows onto the unit circle: its bisection makes the middle node of
  // that side a corner, and the middles of the two halves come from the quadratic map, within
  // 0.01 of the circle, where the middles of straight halves would be 0.076 inside it.
  Mesh curved(2);
  const double s = std::sqrt(0.5);
  for (const Point & x : {Point{0, 0, 0},
                          Point{1, 0, 0},
                          Point{0, 1, 0},
                          Point{0.5, 0, 0},
                          Point{s, s, 0},
                          Point{0, 0.5, 0}})
    curved.addNode(x);
  curved.addElement(ElementType::Tri6, {0, 1, 2, 3, 4, 5});
  Bisection hierarchy(curved);
  CHECK(hierarchy.refine({1}, 1));
  int on_circle = 0;
  for (Index n = 0; n < hierarchy.mesh().numNodes(); ++n)
  {
    const Point & x = hierarchy.mesh().node(n);
    const double radius = std::hypot(x[0], x[1]);
    if (x[0] > 0.01 && x[1] > 0.01 && radius > 0.9)
    {
      CHECK(std::abs(radius - 1.0) < 0.01);
      ++on_circle;
    }
  }
  // The middle of the side and the middles of its halves.
  CHECK(on_circle == 3);
}

TEST(subdivision_with_hanging_nodes_refines_and_coarsens_quadrilaterals_and_hexahedra)
{
  // A multilinear field: its value at a hanging node is the mean of the ends of the hanging edge or
  // of the corners of the hanging face, so the constraints must reproduce it exactly.
  const auto multilinear = [](const Point & x)
  { return 1.0 + x[0] + 2.0 * x[1] - x[2] + 3.0 * x[0] * x[1] + x[0] * x[1] * x[2]; };
  for (const Mesh & initial :
       {generateRectangleMesh(linspace(0, 2, 3), linspace(0, 1, 2), "Quad4"),
        generateBoxMesh(linspace(0, 2, 2), linspace(0, 1, 2), linspace(0, 1, 2), "Hex8")})
  {
    const bool three = initial.dimension() == 3;
    Subdivision hierarchy(initial);
    const auto measure = [&]()
    { return three ? volume(hierarchy.mesh()) : area(hierarchy.mesh()); };
    const auto values = [&]()
    {
      Eigen::VectorXd v(hierarchy.mesh().numNodes());
      for (Index n = 0; n < hierarchy.mesh().numNodes(); ++n)
        v[n] = multilinear(hierarchy.mesh().node(n));
      return v;
    };
    const auto checkField = [&](const Eigen::VectorXd & v)
    {
      for (Index n = 0; n < hierarchy.mesh().numNodes(); ++n)
        CHECK_NEAR(v[n], multilinear(hierarchy.mesh().node(n)), 1e-12);
    };
    const auto checkHanging = [&]()
    {
      for (const auto & h : hierarchy.hanging())
      {
        double sum = 0.0;
        for (const auto & [node, weight] : h.on)
          sum += weight * multilinear(hierarchy.mesh().node(node));
        CHECK_NEAR(sum, multilinear(hierarchy.mesh().node(h.node)), 1e-12);
      }
    };
    for (int cycle = 0; cycle < 3; ++cycle)
    {
      const Eigen::VectorXd before = values();
      const Mesh & mesh = hierarchy.mesh();
      std::vector<char> marked(static_cast<std::size_t>(mesh.numElements()), 0);
      for (Index e = 0; e < mesh.numElements(); ++e)
        marked[static_cast<std::size_t>(e)] = norm(mesh.node(mesh.element(e).nodes[0])) < 1e-12;
      CHECK(hierarchy.refine(marked, 10));
      CHECK_NEAR(measure(), 2.0, 1e-12);
      checkField(hierarchy.transfer(before, 1));
      checkHanging();
      CHECK(!hierarchy.hanging().empty());
      if (three)
        CHECK_NEAR(sideArea(hierarchy.mesh(), "left"), 1.0, 1e-12);
      else
        CHECK_NEAR(sideLength(hierarchy.mesh(), "left"), 1.0, 1e-12);
    }
    // The cell at the corner is three levels down, and the balance keeps every neighbour within
    // one level: the levels range over 0 to 3.
    const std::vector<int> levels = hierarchy.levels();
    CHECK(*std::max_element(levels.begin(), levels.end()) == 3);
    for (int cycle = 0; cycle < 10; ++cycle)
    {
      const Eigen::VectorXd before = values();
      const bool changed = hierarchy.coarsen(
          std::vector<char>(static_cast<std::size_t>(hierarchy.mesh().numElements()), 1));
      CHECK_NEAR(measure(), 2.0, 1e-12);
      checkField(hierarchy.transfer(before, 1));
      checkHanging();
      if (!changed)
        break;
    }
    CHECK(hierarchy.mesh().numElements() == initial.numElements());
    CHECK(hierarchy.hanging().empty());
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
