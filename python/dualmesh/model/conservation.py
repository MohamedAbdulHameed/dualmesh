"""The balance law that every model in dualmesh is made of.

A conservation law states, at each point of the domain $\\Omega$,

$$
\\frac{\\partial \\mathbf{m}}{\\partial t} + \\frac{\\partial^2 \\mathbf{a}}{\\partial t^2}
+ \\nabla\\cdot\\mathbf{F} = \\mathbf{s},
$$

with $t$ the time in $\\mathrm{s}$, $\\mathbf{m}$ the storage, $\\mathbf{a}$ the
inertia, $\\mathbf{F}$ the flux and $\\mathbf{s}$ the source. Each term is an
expression of the unknown $\\mathbf{u}$, its gradient, other fields, internal
state $\\mathbf{q}$ and constants. Each internal state follows a rate law
$\\dot{\\mathbf{q}} = \\mathbf{g}$. A boundary flux is a prescribed outward
normal flux $\\bar{F}_n = \\mathbf{F}\\cdot\\mathbf{n}$ on a named part of the
boundary, with $\\mathbf{n}$ the outward unit normal. A positive boundary flux
therefore removes the conserved quantity. `flux_name` is the standard name
of $\\mathbf{F}$, such as `heat_flux`, which a boundary flux postprocessor names.

A storage written as the antiderivative of a coefficient $c$ by the unknown,
$\\mathbf{m} = \\int^{u} c\\,\\mathrm{d}u'$, has the rate $c\\,\\partial u/\\partial t$.
The equations of general_form_PDE write their time derivatives so, and an
inertia always so.

The law holds expressions only. It never names a test function or a method, so the
models of the input and the equations of general_form_PDE lower the same law onto the
general form of the engine, for FEM and DMCDM.
"""

import dataclasses

from dualmesh._core import InputError
from dualmesh.model import calculus, dimensions, graph
from dualmesh.units import base

dimension_error = dimensions.dimension_error
# The number zero stands for a term the law does not have.
is_absent = dimensions.is_zero

length = base("length")
time_dimension = base("time")
cell_term = "cell"
cell_term_parts = ("storage", "inertia", "flux", "source")


class conservation_law:
    """One balance law with its unknown, terms, internal state and boundary fluxes."""

    def __init__(
        self,
        name: str,
        unknown,
        storage=None,
        inertia=None,
        flux=None,
        source=None,
        state_rates: dict | None = None,
        boundary_fluxes: dict | None = None,
        flux_name: str | None = None,
    ):
        if not isinstance(name, str) or not name.isidentifier():
            raise InputError(
                f"the name of a conservation law must be a word of letters, digits "
                f"and underscores, got {name!r}"
            )
        if not isinstance(unknown, graph.expression) or unknown.kind != "field":
            raise InputError(
                f"the unknown of conservation law '{name}' must be a field, got {unknown!r}"
            )
        self.name = name
        self.unknown = unknown
        self.storage = self.term_expression(storage)
        self.inertia = self.term_expression(inertia)
        self.flux = self.term_expression(flux)
        self.source = self.term_expression(source)
        self.state_rates = {}
        for state_node, rate in (state_rates or {}).items():
            if not isinstance(state_node, graph.expression) or state_node.kind != "state":
                raise InputError(
                    f"conservation law '{name}': each key of state_rates must be a state, "
                    f"got {state_node!r}"
                )
            self.state_rates[state_node] = graph.as_expression(rate)
        self.boundary_fluxes = {}
        for boundary_flux_name, boundary_flux in (boundary_fluxes or {}).items():
            if (
                not isinstance(boundary_flux_name, str)
                or not boundary_flux_name.isidentifier()
                or boundary_flux_name == cell_term
            ):
                raise InputError(
                    f"conservation law '{name}': a boundary flux name must be a word "
                    f"other than '{cell_term}', got {boundary_flux_name!r}"
                )
            self.boundary_fluxes[boundary_flux_name] = graph.as_expression(boundary_flux)
        if flux_name is not None and (
            not isinstance(flux_name, str) or not flux_name.isidentifier()
        ):
            raise InputError(
                f"conservation law '{name}': the flux name must be a word, got {flux_name!r}"
            )
        self.flux_name = flux_name

    @staticmethod
    def term_expression(value):
        """Return value as an expression. None becomes the number zero."""
        return graph.as_expression(0.0 if value is None else value)

    def cell_expressions(self) -> list:
        """Return every expression of the cell term: storage, inertia, flux, source and state rates."""
        return [self.storage, self.inertia, self.flux, self.source, *self.state_rates.values()]

    def check(self) -> None:
        """Raise dimension_error when the ranks or dimensions of the terms disagree.

        Raise InputError when a term reads a state that has no rate in state_rates.
        """
        self._check_states_read()
        rate_dimensions = self._term_rate_dimensions()
        self._check_state_rates()
        if rate_dimensions:
            self._check_boundary_fluxes(next(iter(rate_dimensions.values())) * length)

    def _check_states_read(self) -> None:
        """Raise InputError when a term or a boundary flux reads a state with no rate."""
        with_rate = {state_node.name for state_node in self.state_rates}
        read = {
            node.name
            for node in self.leaves_of(
                self.cell_expressions() + list(self.boundary_fluxes.values())
            )
            if node.kind == "state"
        }
        without_rate = sorted(read - with_rate)
        if without_rate:
            raise InputError(
                f"conservation law '{self.name}' reads the state {', '.join(without_rate)} "
                f"with no rate. Give each state its rate in state_rates"
            )

    def _term_rate_dimensions(self) -> dict:
        """Return the dimension of the source that each present cell term gives, the rate of the
        conserved amount, after checking the rank of each term and that the dimensions agree."""
        unknown_rank = self.unknown.rank
        expected_rank = {
            "storage": unknown_rank,
            "inertia": unknown_rank,
            "flux": unknown_rank + 1,
            "source": unknown_rank,
        }
        rate_dimensions = {}
        for part in cell_term_parts:
            node = getattr(self, part)
            if is_absent(node):
                continue
            if node.rank != expected_rank[part]:
                raise dimension_error(
                    f"conservation law '{self.name}': the {part} '{node}' has rank {node.rank}, but the "
                    f"unknown '{self.unknown.name}' of rank {unknown_rank} needs rank {expected_rank[part]}"
                )
            rate_dimensions[part] = {
                "storage": node.dimension / time_dimension,
                "inertia": node.dimension / time_dimension**2,
                "flux": node.dimension / length,
                "source": node.dimension,
            }[part]
        if not rate_dimensions and not self.boundary_fluxes:
            raise dimension_error(
                f"conservation law '{self.name}' has no storage, inertia, flux, "
                f"source or boundary flux"
            )
        if len(set(rate_dimensions.values())) > 1:
            listed = ", ".join(f"{part} gives {rate}" for part, rate in rate_dimensions.items())
            raise dimension_error(
                f"conservation law '{self.name}': the terms disagree in dimension. The storage over time, "
                f"the inertia over time squared, the flux over length and the source must agree, but {listed}"
            )
        return rate_dimensions

    def _check_state_rates(self) -> None:
        """Raise dimension_error when the rate of a state has not its rank, or not its
        dimension over time."""
        for state_node, rate in self.state_rates.items():
            if rate.rank != state_node.rank:
                raise dimension_error(
                    f"conservation law '{self.name}': the rate of state '{state_node.name}' "
                    f"has rank {rate.rank}, but the state has rank {state_node.rank}"
                )
            if rate.dimension != state_node.dimension / time_dimension:
                raise dimension_error(
                    f"conservation law '{self.name}': the rate of state '{state_node.name}' has the dimension "
                    f"{rate.dimension}, but the state over time has {state_node.dimension / time_dimension}"
                )

    def _check_boundary_fluxes(self, flux_dimension) -> None:
        """Raise dimension_error when a boundary flux has not the rank of the unknown, or not
        flux_dimension, the dimension of the flux of the law."""
        unknown_rank = self.unknown.rank
        for flux_name, boundary_flux in self.boundary_fluxes.items():
            if is_absent(boundary_flux):
                continue
            if boundary_flux.rank != unknown_rank:
                raise dimension_error(
                    f"conservation law '{self.name}': the boundary flux '{flux_name}' has rank "
                    f"{boundary_flux.rank}, but the unknown has rank {unknown_rank}"
                )
            if boundary_flux.dimension != flux_dimension:
                raise dimension_error(
                    f"conservation law '{self.name}': the boundary flux '{flux_name}' has the dimension "
                    f"{boundary_flux.dimension}, but the flux of the law has {flux_dimension}"
                )

    def leaves_of(self, expressions) -> set:
        found = set()
        for node in expressions:
            found |= graph.leaves(node)
        return found

    def states(self) -> list:
        """Return the state nodes, sorted by name."""
        return sorted(self.state_rates, key=lambda node: node.name)

    def components(self, space_dimension: int) -> list:
        """Return the scalar terms of each scalar component of the unknown, in a space of
        space_dimension, as `scalar_terms`.

        A storage or an inertia written as the antiderivative of a coefficient c by the
        component u takes the rate form c du/dt, or c d2u/dt2, with c read by `differentiate`.
        Any other storage keeps its expression m, whose rate is dm/dt.
        """
        count = 1 if self.unknown.rank == 0 else space_dimension
        unknown = graph.expand(self.unknown, space_dimension)
        unknown = [unknown] if self.unknown.rank == 0 else unknown

        def per_component(node):
            if is_absent(node):
                return [None] * count
            expanded = graph.expand(node, space_dimension)
            return [expanded] if self.unknown.rank == 0 else expanded

        def rate_coefficient(node, leaf, what: str):
            if node is None:
                return None
            if node.kind != "antiderivative":
                raise InputError(
                    f"conservation law '{self.name}': the {what} '{node}' is lowered as the "
                    f"antiderivative of its coefficient only"
                )
            return calculus.differentiate(node, leaf)

        result = []
        for leaf, storage, inertia, flux, source in zip(
            unknown,
            per_component(self.storage),
            per_component(self.inertia),
            per_component(self.flux),
            per_component(self.source),
            strict=True,
        ):
            if flux is not None and all(is_absent(entry) for entry in flux):
                flux = None
            if source is not None and is_absent(source):
                source = None
            capacity = None
            if storage is not None and storage.kind == "antiderivative":
                capacity, storage = calculus.differentiate(storage, leaf), None
            result.append(
                scalar_terms(
                    unknown=leaf,
                    storage=storage,
                    capacity=capacity,
                    mass=rate_coefficient(inertia, leaf, "inertia"),
                    flux=flux,
                    source=source,
                )
            )
        return result

    def __repr__(self) -> str:
        return f"conservation_law({self.name}, unknown {self.unknown.name})"


@dataclasses.dataclass(frozen=True)
class scalar_terms:
    """The terms of the equation of one scalar component of the unknown, as scalar expressions.

    `unknown` is the leaf of the component. Each term is None where the law does not have it:
    `storage` m of the rate dm/dt, `capacity` c of the rate c du/dt, `mass` c of the inertia
    c d2u/dt2, `flux` with one expression per direction of space, and `source`.
    """

    unknown: object
    storage: object
    capacity: object
    mass: object
    flux: list | None
    source: object
