"""The solid at finite strain: one model for every elastic energy.

The input gives the strain energy per unit undeformed volume $\\psi$ as a formula of the right
Cauchy-Green tensor $\\mathbf{C} = \\mathbf{F}^T\\mathbf{F}$, through its invariants. Here
$\\mathbf{F} = \\mathbf{I} + \\nabla\\mathbf{u}$ is the deformation gradient, with $\\mathbf{u}$
the displacement in m and $\\nabla$ the gradient on the undeformed body. The formula reads:

- `I_C`, the trace of $\\mathbf{C}$;
- `trace_C_squared`, the trace of $\\mathbf{C}^2$;
- `J`, the determinant of $\\mathbf{F}$, the ratio of the deformed volume to the undeformed one;
- `shear_modulus` $\\mu$ and `first_Lame_parameter` $\\lambda$, both in Pa, from the Young modulus
  $E$ and the Poisson ratio $\\nu$ of the properties, as $\\mu = E/(2(1+\\nu))$ and
  $\\lambda = E\\nu/((1+\\nu)(1-2\\nu))$.

The first Piola-Kirchhoff stress $\\mathbf{P} = \\partial\\psi/\\partial\\mathbf{F}$ is the
symbolic derivative of the formula. The momentum balance $-\\nabla\\cdot\\mathbf{P} = 0$ on the
undeformed body becomes one term per displacement component, whose flux is a row of
$-\\mathbf{P}$. The engine differentiates that flux for Newton's method. A traction on a
boundary is $\\mathbf{P}\\mathbf{N}$ per unit undeformed area, with $\\mathbf{N}$ the outward unit
normal of the undeformed boundary.

In plane strain the out-of-plane stretch is 1. Every energy has the same small-strain limit:
linear elasticity with the same $E$ and $\\nu$.
"""

from dualmesh.input import units
from dualmesh.input.lower import engine_component_names, write_expression
from dualmesh.input.units import input_error
from dualmesh.model import parse
from dualmesh.model.differentiate import differentiate
from dualmesh.model.expression import gradient_component, number

# The named energies. Each vanishes, with its stress, in the undeformed state.
named_energies = {
    # The compressible neo-Hookean energy of Simo and Hughes, Computational Inelasticity (1998).
    "neo-Hookean": "shear_modulus/2*(I_C - 3) - shear_modulus*log(J) + first_Lame_parameter/2*log(J)^2",
    # The Saint Venant-Kirchhoff energy lambda/2 (tr E)^2 + mu tr(E^2), with E = (C - I)/2.
    "Saint_Venant_Kirchhoff": (
        "first_Lame_parameter/8*(I_C - 3)^2 + shear_modulus/4*(trace_C_squared - 2*I_C + 3)"
    ),
}


def Lame_parameters(Young_modulus: float, Poisson_ratio: float) -> tuple[float, float]:
    """The shear modulus and the first Lame parameter of an isotropic material, in Pa."""
    shear_modulus = Young_modulus / (2.0 * (1.0 + Poisson_ratio))
    first_Lame_parameter = (
        Young_modulus * Poisson_ratio / ((1.0 + Poisson_ratio) * (1.0 - 2.0 * Poisson_ratio))
    )
    return shear_modulus, first_Lame_parameter


def deformation_gradient(dimension: int) -> list[list]:
    """The 3 by 3 deformation gradient, whose in-plane part reads the displacement gradient."""
    one, zero = number(1.0), number(0.0)
    tensor = [[one if row == column else zero for column in range(3)] for row in range(3)]
    for row in range(dimension):
        for column in range(dimension):
            gradient = gradient_component("displacement", (row,), column)
            tensor[row][column] = (one if row == column else zero) + gradient
    return tensor


def first_Piola_Kirchhoff_stress(
    energy: str, dimension: int, Young_modulus: float, Poisson_ratio: float, path: str
):
    """The in-plane rows of P, as expressions of the model language, for the energy formula or name."""
    text = named_energies.get(energy, energy)
    F = deformation_gradient(dimension)
    C = [
        [sum((F[k][a] * F[k][b] for k in range(3)), number(0.0)) for b in range(3)]
        for a in range(3)
    ]
    I_C = C[0][0] + C[1][1] + C[2][2]
    trace_C_squared = sum((C[a][b] * C[a][b] for a in range(3) for b in range(3)), number(0.0))
    J = (
        F[0][0] * (F[1][1] * F[2][2] - F[1][2] * F[2][1])
        - F[0][1] * (F[1][0] * F[2][2] - F[1][2] * F[2][0])
        + F[0][2] * (F[1][0] * F[2][1] - F[1][1] * F[2][0])
    )
    shear_modulus, first_Lame_parameter = Lame_parameters(Young_modulus, Poisson_ratio)
    pressure = units.parse_unit("Pa")[1]
    symbols = {
        "I_C": I_C,
        "trace_C_squared": trace_C_squared,
        "J": J,
        "shear_modulus": number(shear_modulus, pressure),
        "first_Lame_parameter": number(first_Lame_parameter, pressure),
    }
    try:
        energy_density = parse(text, symbols)
    except input_error as error:
        raise input_error(f"{path}: {error}") from None
    if energy_density.rank != 0 or energy_density.dimension != pressure:
        raise input_error(
            f"{path}: the strain energy '{text}' must be a scalar in J/m^3, "
            f"and it is in {units.SI_unit_text(energy_density.dimension)}."
        )
    return [
        [
            differentiate(energy_density, gradient_component("displacement", (row,), column))
            for column in range(dimension)
        ]
        for row in range(dimension)
    ]


def add_finite_strain_solid(
    problem,
    energy: str,
    dimension: int,
    Young_modulus: float,
    Poisson_ratio: float,
    path: str,
    weight: list[float] | None = None,
):
    """Adds the displacement and its momentum balance at finite strain, and returns the solid.

    `weight` is the force of gravity per volume of the undeformed body, rho_0 g in N/m^3, which
    stays the same as the body deforms (a dead load)."""
    names = list(engine_component_names["displacement"][:dimension])
    for name in names:
        problem.add_variable(name)
    stress = first_Piola_Kirchhoff_stress(energy, dimension, Young_modulus, Poisson_ratio, path)
    for row, name in enumerate(names):
        flux = [f"-({write_expression(stress[row][column], path)})" for column in range(dimension)]
        source = {"source": repr(weight[row])} if weight and weight[row] != 0.0 else {}
        problem.add_kernel("parsed_kernel", variable=name, flux=flux, **source)
    return finite_strain_solid(problem, names)


class finite_strain_solid:
    """The displacement of the solid at finite strain, which takes the conditions of the input."""

    def __init__(self, problem, names: list[str]):
        self.problem = problem
        self.names = names

    def variable_names(self) -> list[str]:
        return list(self.names)

    def set_value(self, variable: str, expression: str, boundary: str) -> None:
        self.problem.add_boundary_condition(
            "Dirichlet", boundary, variable=variable, value=expression
        )

    def set_traction(self, components: list[str], boundary: str) -> None:
        """The traction P N per unit undeformed area, the natural quantity of each displacement equation."""
        if len(components) != len(self.names):
            raise input_error(
                f"a traction takes {len(self.names)} components, got {len(components)}."
            )
        for name, component in zip(self.names, components):
            self.problem.add_boundary_condition("Neumann", boundary, variable=name, flux=component)
