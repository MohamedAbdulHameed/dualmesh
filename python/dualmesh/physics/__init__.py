# SPDX-License-Identifier: LGPL-2.1-or-later
r"""The physics level: sets of equations named in physical terms.

A physics is added to a problem by its type name, creates its variables, and
generates the kernels and the property objects of its equations when the problem is
initialised::

    import dualmesh

    problem = dualmesh.Problem(dualmesh.read_mesh("wrench.msh"), method="fem")
    solid = problem.add_physics("solid_mechanics", youngs_modulus=200.0e9, poissons_ratio=0.3)
    solid.add_boundary_condition("fixed", boundary="jaws")
    solid.add_boundary_condition("total_force = [0, -150, 0]", boundary="grip")
    problem.solve()

Each physics defines its variables.  heat_conduction has ``temperature``,
solid_mechanics has ``disp_x``, ``disp_y`` and ``disp_z``, and fluid_dynamics
has ``velocity_x``, ``velocity_y``, ``velocity_z`` and ``pressure``.  The
fields of general_form_PDE are named in its ``variables``.  The boundary
conditions of a physics are written as text.  A variable on the left
prescribes its value, and the name of the flux of the physics on the left
prescribes the flux.  A vector (``traction``, ``total_force``) is split into
one object per component.  The extra terms of a physics take the type
names and the parameters of the objects (:doc:`/objects`), with the
variables of the physics filled in.  A coupling between two physics
(``thermal_expansion`` or ``nonisothermal_flow``) is added with
:meth:`dualmesh.Problem.add_coupling`.

The package holds one module per physics: ``base`` (the :class:`Physics` and
:class:`Coupling` base classes and the registry), ``heat`` (heat_conduction),
``general_form`` (general_form_PDE), ``solid`` (solid_mechanics), ``fluid``
(fluid_dynamics) and ``couplings``.
"""

# The registry is re-exported for dualmesh.list_objects and the generated syntax pages.
from .base import _REGISTRY as _REGISTRY
from .base import (
    Coupling,
    Physics,
    create,
    describe,
    is_registered,
    label,
    problem_method,
    register,
    registered,
    unique_name,
)
from .couplings import NonisothermalFlow, ThermalExpansion
from .fluid import FLOW_FORMULATIONS, FLOW_TIME_INTEGRATIONS, FluidDynamics
from .general_form import GeneralFormPDE
from .heat import HeatConduction
from .solid import FORMULATIONS, HYPERELASTIC, MEMBERS, STRUCTURAL, THEORIES, SolidMechanics

__all__ = [
    "FLOW_FORMULATIONS",
    "FLOW_TIME_INTEGRATIONS",
    "FORMULATIONS",
    "HYPERELASTIC",
    "MEMBERS",
    "STRUCTURAL",
    "THEORIES",
    "Coupling",
    "FluidDynamics",
    "GeneralFormPDE",
    "HeatConduction",
    "NonisothermalFlow",
    "Physics",
    "SolidMechanics",
    "ThermalExpansion",
    "create",
    "describe",
    "is_registered",
    "label",
    "problem_method",
    "register",
    "registered",
    "unique_name",
]
