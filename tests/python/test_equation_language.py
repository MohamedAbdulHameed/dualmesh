# SPDX-License-Identifier: LGPL-2.1-or-later
"""The equation language of general_form_PDE and the conversion of expressions to SymPy and back.

The equation front end of dualmesh.model.syntax reads an equation as it is printed and sorts its
terms into a conservation law. Each check evaluates a term of the law at chosen numbers and
compares it with the value of that term worked out by hand from the equation. The converter
writes an expression in SymPy and reads SymPy back into the engine's language, and each check
compares a value at a point with the value worked out by hand. The symbolic derivative of
dualmesh.model.calculus is checked against its closed form at chosen numbers.
"""

import math

import pytest
from dualmesh import mms
from dualmesh._core import Expression, InputError, ParsedFunction, expression_functions
from dualmesh.model import (
    differentiate,
    equation_scope,
    graph,
    parse,
    read_condition,
    read_equations,
)
from dualmesh.model.engine_form import write_expression
from dualmesh.model.sympy_form import from_sympy, to_sympy
from dualmesh.model.syntax import parse_in_scope
from dualmesh.units import base, dimensionless


def value(node, numbers: dict) -> float:
    """The number that node takes with each leaf named by its text set to numbers[text]."""
    replacements = {}
    for leaf in graph.leaves(node):
        if leaf.kind == "constant" and leaf.value is not None:
            replacements[leaf] = graph.number(leaf.value)
        elif str(leaf) in numbers:
            replacements[leaf] = graph.number(numbers[str(leaf)], leaf.dimension)
    result = graph.substitute(node, replacements)
    assert result.kind == "number", f"'{result}' reads more than {sorted(numbers)}"
    return result.value


def scalar_terms(equation: str, coordinate_system="cartesian", **definitions):
    """The terms of the one equation of the scalar field u on a plane."""
    scope = equation_scope({"u": 0}, 2, coordinate_system, definitions)
    (terms,) = read_equations([equation], [""], scope)["u"].components(2)
    return terms


# ---------------------------------------------------------------------------
# Sorting the terms
# ---------------------------------------------------------------------------
def test_the_flux_of_a_laplacian_is_minus_the_gradient():
    """-laplacian(u) = 1 is div(-grad u) = 1, so F = -grad u and s = 1."""
    terms = scalar_terms("-laplacian(u) = 1")
    gradient = {"grad(u)[0]": 0.3, "grad(u)[1]": -0.7}
    assert [value(component, gradient) for component in terms.flux] == pytest.approx([-0.3, 0.7])
    assert value(terms.source, {}) == 1.0
    assert terms.capacity is None and terms.mass is None and terms.storage is None


def test_a_transient_heat_equation_takes_its_capacity_flux_and_source():
    """rho*cp*dT/dt = div(k*grad(T)) + q with rho = 2, cp = 3, k = 5 and q = 7: the capacity of
    dT/dt is rho cp = 6, the flux F = -k grad T and the source q = 7."""
    scope = equation_scope({"T": 0}, 2, "cartesian", {"rho": 2.0, "cp": 3.0, "k": 5.0, "q": 7.0})
    (terms,) = read_equations(["rho*cp*dT/dt = div(k*grad(T)) + q"], [""], scope)["T"].components(2)
    assert value(terms.capacity, {}) == pytest.approx(6.0)
    gradient = {"grad(T)[0]": 1.0, "grad(T)[1]": 2.0}
    assert [value(component, gradient) for component in terms.flux] == pytest.approx([-5.0, -10.0])
    assert value(terms.source, {}) == pytest.approx(7.0)
    assert {"rho", "cp", "k", "q"} <= scope.reads


def test_the_fields_of_a_system_follow_the_time_derivative_and_the_field_left_over():
    """The Cahn-Hilliard pair dc/dt = div(M grad w), w = c^3 - c - ell^2 laplacian(c), with M = 2
    and ell = 0.5: the first equation governs c, by its time derivative, and the second w, the
    field left over, with the flux ell^2 grad c and the source c^3 - c - w."""
    scope = equation_scope({"c": 0, "w": 0}, 2, "cartesian", {"M": 2.0, "ell": 0.5})
    laws = read_equations(
        ["dc/dt = div(M*grad(w))", "w = c^3 - c - ell^2*laplacian(c)"], ["", ""], scope
    )
    (c_terms,) = laws["c"].components(2)
    (w_terms,) = laws["w"].components(2)
    assert value(c_terms.capacity, {}) == 1.0
    gradient_w = {"grad(w)[0]": 1.0, "grad(w)[1]": 3.0}
    assert [value(entry, gradient_w) for entry in c_terms.flux] == pytest.approx([-2.0, -6.0])
    gradient_c = {"grad(c)[0]": 1.0, "grad(c)[1]": 3.0}
    assert [value(entry, gradient_c) for entry in w_terms.flux] == pytest.approx([0.25, 0.75])
    assert value(w_terms.source, {"c": 2.0, "w": 1.0}) == pytest.approx(8.0 - 2.0 - 1.0)


def test_the_flux_of_a_vector_equation_is_the_tensor_inside_its_divergence():
    """div(2*symmetric(grad(U))) = 0: the flux is F = 2 symmetric(grad U), with
    F_xy = dU_x/dy + dU_y/dx."""
    scope = equation_scope({"U": 1}, 2)
    components = read_equations(["div(2*symmetric(grad(U))) = 0"], [""], scope)["U"].components(2)
    gradient = {
        "grad(U_x)[0]": 1.0,
        "grad(U_x)[1]": 2.0,
        "grad(U_y)[0]": 3.0,
        "grad(U_y)[1]": 4.0,
    }
    rows = [[value(entry, gradient) for entry in terms.flux] for terms in components]
    assert [value for row in rows for value in row] == pytest.approx([2.0, 5.0, 5.0, 8.0])
    assert [terms.source for terms in components] == [None, None]


def test_an_axisymmetric_divergence_in_place_carries_the_factor_of_the_radius():
    """In axisymmetric coordinates div([u, 0]) = du/dr + u/r. The argument holds no derivative,
    so the divergence is a source term: at r = 2, u = 3 and du/dr = 5 the source is -(5 + 3/2)."""
    terms = scalar_terms("-laplacian(u) + div([u, 0]) = 0", "axisymmetric")
    numbers = {"u": 3.0, "grad(u)[0]": 5.0, "position()[0]": 2.0}
    assert value(terms.source, numbers) == pytest.approx(-6.5)


def test_x_in_axisymmetric_coordinates_is_refused():
    """The coordinates of an axisymmetric problem are r and z. The engine would read x as r, so
    a manufactured source that took x for a constant would differ from the solved equation."""
    with pytest.raises(InputError, match="'x' is not a coordinate of this problem"):
        scalar_terms("-div((1 + x)*grad(u)) = 1", "axisymmetric")


@pytest.mark.parametrize(
    "equation, message",
    [
        ("x*laplacian(u) = 1", "constant in space"),
        ("sin(laplacian(u)) = 1", "stands only as a term"),
        ("laplacian(u) +", "expected a number"),
        ("du/dt = laplacian(w)", "'w' would be differentiated"),
        ("laplacian(laplacian(u)) = 1", "fourth-order equation"),
        ("d2u/dx2 = 1", "second derivatives in space"),
        ("-div([du/dx]) = 1", "has 2 components"),
    ],
)
def test_an_equation_that_cannot_be_sorted_is_refused(equation, message):
    scope = equation_scope({"u": 0}, 2)
    with pytest.raises(InputError, match=message):
        read_equations([equation], [""], scope)


def test_two_equations_that_could_govern_either_field_take_keys():
    """u = 1 and v = 2 hold no time derivative and no flux, so a key names the field of each.
    Under the keys v and u, the source of the equation of v is 1 - u."""
    scope = equation_scope({"u": 0, "v": 0}, 2)
    with pytest.raises(InputError, match="could govern any of the fields u, v"):
        read_equations(["u = 1", "v = 2"], ["", ""], scope)
    (terms,) = read_equations(["u = 1", "v = 2"], ["v", "u"], scope)["v"].components(2)
    assert value(terms.source, {"u": 4.0}) == pytest.approx(-3.0)


def test_a_definition_reads_only_the_definitions_before_it():
    with pytest.raises(InputError, match="'k0' is defined after this expression"):
        equation_scope({"u": 0}, 2, "cartesian", {"k": "2*k0", "k0": 1.0})
    scope = equation_scope({"u": 0}, 2, "cartesian", {"k0": 1.5, "k": "2*k0"})
    assert value(scope["k"], {}) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------
def heat():
    """-div(k grad u) = 0 with k = 2, whose flux is F = -k grad u, and h = 4."""
    scope = equation_scope({"u": 0}, 2, "cartesian", {"k": 2.0, "h": 4.0})
    return scope, read_equations(["-div(k*grad(u)) = 0"], [""], scope)


def test_a_flux_written_out_with_du_dn_is_the_normal_flux_of_the_equation():
    """-k du/dn is n . F, so -k*du/dn = h*(u - 1) prescribes n . F = 8 at u = 3. The text
    k*du/dn is -n . F, which the reader refuses."""
    scope, laws = heat()
    ((name, component, flux, node),) = read_condition("-k*du/dn = h*(u - 1)", scope, laws)
    assert (name, component, flux) == ("u", (), True)
    assert value(node, {"u": 3.0}) == pytest.approx(8.0)
    with pytest.raises(InputError, match="is not the normal flux of the equation of u"):
        read_condition("k*du/dn = 1", scope, laws)


def test_a_field_on_the_left_prescribes_its_value_and_normal_flux_names_the_flux():
    scope, laws = heat()
    ((_, _, flux, node),) = read_condition("u = 300 + 10*y", scope, laws)
    assert not flux and value(node, {"position()[1]": 2.0}) == pytest.approx(320.0)
    ((_, _, flux, node),) = read_condition("normal_flux = 5", scope, laws)
    assert flux and value(node, {}) == 5.0
    two = equation_scope({"u": 0, "v": 0}, 2)
    laws = read_equations(["-laplacian(u) = v", "-laplacian(v) = 1"], ["", ""], two)
    with pytest.raises(InputError, match="could be the flux of u, v"):
        read_condition("normal_flux = 1", two, laws)


# ---------------------------------------------------------------------------
# One formula language
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, name",
    [
        ("sym(grad(U))", "symmetric"),
        ("dev(grad(U))", "deviatoric"),
        ("det(grad(U))", "determinant"),
        ("ddot(grad(U), grad(U))", "inner"),
        ("if(U_x < 1, 2, 3)", "conditional"),
    ],
)
def test_each_function_has_one_name(text, name):
    """A second spelling, such as sym for symmetric, is an unknown function, and the one name
    reads the same text."""
    scope = equation_scope({"U": 1}, 2)
    written = text.split("(")[0]
    with pytest.raises(InputError, match=rf"unknown function '{written}'"):
        parse_in_scope(text, scope, equation=False)
    parse_in_scope(name + text[len(written) :], scope, equation=False)


@pytest.mark.parametrize("name, argument_count", expression_functions())
def test_every_function_of_the_engine_parses_and_lowers_in_the_model_language(name, argument_count):
    """Each function of the table of the engine is a call of the model language of the same name.
    At u = 0.3 and v = 0.7 the call of numbers folds to the value of the engine, the call of
    fields lowers back onto the engine with that value, and its symbolic derivative by u matches
    the central difference of the engine."""
    names = ["u", "v"][:argument_count]
    at = dict(zip(names, [0.3, 0.7]))
    engine = Expression(f"{name}({', '.join(names)})", names)
    expected = engine(**at)
    folded = parse(f"{name}({', '.join(map(str, at.values()))})")
    assert folded.kind == "number" and folded.value == pytest.approx(expected, rel=1e-14)
    fields = {key: graph.field_component(key, physical_dimension=dimensionless) for key in names}
    node = parse(f"{name}({', '.join(names)})", fields)
    assert Expression(write_expression(node, "test"), names)(**at) == pytest.approx(
        expected, rel=1e-15
    )
    slope = Expression(write_expression(differentiate(node, fields["u"]), "test"), names)
    step = 1e-6
    above, below = (engine(**{**at, "u": at["u"] + shift}) for shift in (step, -step))
    assert slope(**at) == pytest.approx((above - below) / (2.0 * step), rel=1e-6, abs=1e-8)


def test_square_brackets_hold_a_quantity_or_the_items_of_a_vector():
    assert parse("[2 m]") is graph.number(2.0, base("length"))
    assert parse("[1e-5 1/K]").value == pytest.approx(1.0e-5)
    vector = parse("[1, [3 m]/[1 m]]")
    assert vector.kind == "list" and vector.rank == 1
    assert [item.value for item in vector.operands] == [1.0, 3.0]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("(1 < 2) and (3 > 4)", 0.0),
        ("(1 < 2) or (3 > 4)", 1.0),
        ("2 <= 2", 1.0),
        ("2 != 2", 0.0),
        ("conditional(1 >= 2, 5, 7)", 7.0),
        ("2**3**2", 512.0),
    ],
)
def test_comparisons_and_logic_reach_the_engine_as_one_or_zero(text, expected):
    assert ParsedFunction(write_expression(parse(text), "test"))(0.0, 0.0, 0.0, 0.0) == expected


# ---------------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "exponent, base_value, expected",
    [(1.5, 0.0, 0.0), (1.0, 0.0, 1.0), (1.5, 0.4, 1.5 * math.sqrt(0.4))],
)
def test_a_power_with_a_named_exponent_takes_the_power_rule(exponent, base_value, expected):
    """d(u^n)/du = n u^(n - 1) for a named constant n, finite at the zero base u = 0."""
    u = graph.field_component("u", physical_dimension=dimensionless)
    slope = differentiate(u ** graph.constant("n", exponent), u)
    assert value(slope, {str(u): base_value}) == pytest.approx(expected, abs=1e-15)


def test_a_power_whose_exponent_reads_the_variable_takes_the_logarithm():
    """d(u^u)/du = u^u (log(u) + 1), which is 0.5^0.5 (1 - log(2)) at u = 0.5."""
    u = graph.field_component("u", physical_dimension=dimensionless)
    slope = differentiate(u**u, u)
    expected = math.sqrt(0.5) * (1.0 - math.log(2.0))
    assert value(slope, {str(u): 0.5}) == pytest.approx(expected, rel=1e-14)


# ---------------------------------------------------------------------------
# SymPy
# ---------------------------------------------------------------------------
def test_an_expression_converts_to_sympy():
    sympy = pytest.importorskip("sympy")
    scope = equation_scope({"u": 0}, 2)
    text = "x^2*sin(y) + atan2(u, 2) + abs(x - 1)"
    node = graph.expand(parse_in_scope(text, scope, equation=False), 2)
    x, y, u = sympy.symbols("x y u", real=True)
    leaves = {"position()[0]": x, "position()[1]": y, "u": u}
    converted = to_sympy(node, lambda leaf: leaves[str(leaf)])
    expected = x**2 * sympy.sin(y) + sympy.atan2(u, 2) + sympy.Abs(x - 1)
    assert sympy.simplify(converted - expected) == 0


def test_sympy_output_reaches_the_engine_through_the_one_printer():
    """Abs, E, pi and a Piecewise of SymPy become abs, numbers and if of the engine."""
    sympy = pytest.importorskip("sympy")
    x, t = sympy.symbols("x t", real=True)
    expression = (
        sympy.Abs(x - 1) + sympy.E * t + sympy.pi + sympy.Piecewise((1, x < 0.5), (2, True))
    )
    function = ParsedFunction(mms._text(expression))
    assert function(0.25, 0.0, 0.0, 2.0) == pytest.approx(0.75 + 2.0 * math.e + math.pi + 1.0)
    assert function(0.75, 0.0, 0.0, 2.0) == pytest.approx(0.25 + 2.0 * math.e + math.pi + 2.0)


def test_a_graph_survives_the_round_trip_through_sympy():
    sympy = pytest.importorskip("sympy")
    scope = equation_scope({}, 1)
    text = "exp(-x)*cosh(x) + log10(1 + x^2) + erf(x)"
    node = graph.expand(parse_in_scope(text, scope, equation=False), 1)
    x = sympy.Symbol("x", real=True)
    back = from_sympy(to_sympy(node, lambda leaf: x))
    function = ParsedFunction(write_expression(back, "test"))
    expected = math.exp(-0.6) * math.cosh(0.6) + math.log10(1.36) + math.erf(0.6)
    assert function(0.6, 0.0, 0.0, 0.0) == pytest.approx(expected, rel=1e-14)
