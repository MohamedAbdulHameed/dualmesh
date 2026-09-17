# SPDX-License-Identifier: LGPL-2.1-or-later
"""Base classes for kernels, boundary conditions, and materials written in Python.

Subclass these to add physics without touching C++.  Inside the callbacks the
solution is delivered as :class:`dualmesh.ADReal` numbers, so the Jacobian used
by Newton's method remains exact::

    import dualmesh as dm

    class NonlinearBar(dm.PythonKernel):
        "Axial bar with a = EA (1 + 1.5 u' + 0.5 u'^2)."

        def setup(self, problem):
            self.axial_stiffness = self.parameters["axial_stiffness"]

        def has_flux(self):
            return True

        def compute_flux(self, ctx):
            strain = ctx.gradient(self.variable)[0]
            a = self.axial_stiffness * (1.0 + 1.5 * strain + 0.5 * strain * strain)
            return [a * strain]
"""

from __future__ import annotations

import itertools

from . import _core

_counter = itertools.count()


def _base_parameters(
    valid_params,
    variable=None,
    name=None,
    block=(),
    quadrature="gauss2",
    reduced_integration=False,
    scale_with_load=False,
    boundary=None,
    type_name="PythonObject",
):
    params = valid_params()
    if variable is not None:
        params.set("variable", variable)
    if boundary is not None:
        params.set("boundary", [boundary] if isinstance(boundary, str) else list(boundary))
    params.set("block", [block] if isinstance(block, str) else list(block))
    params.set("quadrature", quadrature)
    params.set("reduced_integration", bool(reduced_integration))
    params.set("scale_with_load", bool(scale_with_load))
    params.set("_name", name or f"{type_name}_{next(_counter)}")
    params.set("_type", type_name)
    return params


class _PythonObjectMixin:
    """Shared behaviour: keep user parameters, resolve names at setup time."""

    def setup(self, problem):  # pragma: no cover - default is a no-op
        """Resolve names (variables, functions, properties) before solving."""

    def initial_setup(self, problem):
        type(self).__mro__[1].initial_setup(self, problem)  # C++ base implementation
        self.problem = problem
        self.setup(problem)


class PythonKernel(_core.Kernel, _PythonObjectMixin):
    """A kernel implemented in Python (flux and/or source)."""

    def __init__(
        self,
        variable: str,
        name: str | None = None,
        block=(),
        quadrature: str = "gauss2",
        reduced_integration: bool = False,
        scale_with_load: bool = False,
        **parameters,
    ):
        params = _base_parameters(
            _core.Kernel.valid_params,
            variable=variable,
            name=name,
            block=block,
            quadrature=quadrature,
            reduced_integration=reduced_integration,
            scale_with_load=scale_with_load,
            type_name=type(self).__name__,
        )
        _core.Kernel.__init__(self, params)
        self.parameters = dict(parameters)
        for key, value in parameters.items():
            setattr(self, key, value)

    def initial_setup(self, problem):
        _core.Kernel.initial_setup(self, problem)
        self.problem = problem
        self.setup(problem)

    # The defaults below make a subclass that only defines one of the two
    # callbacks work without further boilerplate.
    def has_flux(self):
        return type(self).compute_flux is not PythonKernel.compute_flux

    def has_source(self):
        return type(self).compute_source is not PythonKernel.compute_source

    def compute_flux(self, ctx):  # pragma: no cover - overridden by subclasses
        return [0.0, 0.0, 0.0]

    def compute_source(self, ctx):  # pragma: no cover - overridden by subclasses
        return 0.0


class PythonBoundaryCondition(_core.IntegratedBC, _PythonObjectMixin):
    """An integrated boundary condition implemented in Python.

    ``compute_boundary_flux`` returns the outward normal flux ``q = n . F``.
    """

    def __init__(
        self,
        variable: str,
        boundary,
        name: str | None = None,
        block=(),
        quadrature: str = "gauss2",
        reduced_integration: bool = False,
        scale_with_load: bool = False,
        **parameters,
    ):
        params = _base_parameters(
            _core.IntegratedBC.valid_params,
            variable=variable,
            boundary=boundary,
            name=name,
            block=block,
            quadrature=quadrature,
            reduced_integration=reduced_integration,
            scale_with_load=scale_with_load,
            type_name=type(self).__name__,
        )
        _core.IntegratedBC.__init__(self, params)
        self.parameters = dict(parameters)
        for key, value in parameters.items():
            setattr(self, key, value)

    def initial_setup(self, problem):
        _core.IntegratedBC.initial_setup(self, problem)
        self.problem = problem
        self.setup(problem)

    def compute_boundary_flux(self, ctx):  # pragma: no cover
        return 0.0


class PythonNodalBoundaryCondition(_core.NodalBC, _PythonObjectMixin):
    """An essential (Dirichlet) boundary condition implemented in Python."""

    def __init__(
        self,
        variable: str,
        boundary,
        name: str | None = None,
        scale_with_load: bool = False,
        **parameters,
    ):
        params = _base_parameters(
            _core.NodalBC.valid_params,
            variable=variable,
            boundary=boundary,
            name=name,
            scale_with_load=scale_with_load,
            type_name=type(self).__name__,
        )
        _core.NodalBC.__init__(self, params)
        self.parameters = dict(parameters)
        for key, value in parameters.items():
            setattr(self, key, value)

    def initial_setup(self, problem):
        _core.NodalBC.initial_setup(self, problem)
        self.problem = problem
        self.setup(problem)

    def compute_value(self, x, t):  # pragma: no cover
        return 0.0


class PythonMaterial(_core.Material, _PythonObjectMixin):
    """A material implemented in Python.

    Declare properties in ``declare_properties`` and fill them in
    ``compute_properties``::

        class TemperatureDependentConductivity(dm.PythonMaterial):
            def declare_properties(self, registry):
                self.conductivity_id = registry.declare("thermal_conductivity")

            def setup(self, problem):
                self.temperature = problem.variable_index("temperature")

            def compute_properties(self, ctx):
                T = ctx.coefficient_value(self.temperature)
                ctx.set_property(self.conductivity_id, 20.0 + 0.2 * T)
    """

    def __init__(self, name: str | None = None, block=(), **parameters):
        params = _core.Material.valid_params()
        params.set("block", [block] if isinstance(block, str) else list(block))
        params.set("_name", name or f"{type(self).__name__}_{next(_counter)}")
        params.set("_type", type(self).__name__)
        _core.Material.__init__(self, params)
        self.parameters = dict(parameters)
        for key, value in parameters.items():
            setattr(self, key, value)

    def initial_setup(self, problem):
        _core.Material.initial_setup(self, problem)
        self.problem = problem
        self.setup(problem)

    def setup(self, problem):  # pragma: no cover
        pass

    def declare_properties(self, registry):  # pragma: no cover
        raise NotImplementedError("declare_properties must be implemented")

    def compute_properties(self, ctx):  # pragma: no cover
        pass
