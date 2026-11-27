# SPDX-License-Identifier: LGPL-2.1-or-later
"""The solid_mechanics physics class: its parameters and their checks, its variables, its
boundary conditions and its results.  The class builds its objects by the functions of the other
modules of the solid package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from ...parameters import InputError, parameter
from ..base import Physics, condition_sides, is_condition_text, label, register, vector_components
from . import continuum, inertia, interfaces, rotations, structures
from .structures import MEMBERS, STRUCTURAL

FORMULATIONS = (
    "plane_stress",
    "plane_strain",
    "generalized_plane_strain",
    "axisymmetric",
    "three_dimensional",
    "plate",
    "beam",
    "circular_plate",
    "truss",
    "shell",
)


@register
@dataclass
class SolidMechanics(Physics):
    r"""The deformation of a solid, as a continuum or as a plate, a beam, a
    truss or a shell.

    The displacement :math:`\mathbf{u}` in :math:`\mathrm{m}` satisfies the
    balance of momentum

    .. math::

        \rho \frac{\partial^2 \mathbf{u}}{\partial t^2}
        - \nabla \cdot \boldsymbol{\sigma} = \mathbf{f},

    with the density :math:`\rho` in :math:`\mathrm{kg/m^3}`, the Cauchy
    stress :math:`\boldsymbol{\sigma}` in :math:`\mathrm{Pa}` and the body
    force per unit volume :math:`\mathbf{f}` in :math:`\mathrm{N/m^3}`.  The
    flux of the equation of the component :math:`u_i` is minus row
    :math:`i` of :math:`\boldsymbol{\sigma}`, so the boundary condition
    ``traction`` prescribes :math:`\boldsymbol{\sigma} \mathbf{n}` on a
    boundary of outward unit normal :math:`\mathbf{n}`.  The inertia term is
    present when ``density`` is given.  The material is linear elastic and
    isotropic at small strain unless the parameters say otherwise.  The
    variables are ``disp_x``, ``disp_y`` and ``disp_z``, or ``disp_r`` and
    ``disp_z`` in axisymmetric coordinates.

    The continuum formulations are three_dimensional, the default on a
    three-dimensional mesh, axisymmetric, the default in axisymmetric
    coordinates, and plane_stress, plane_strain and generalized_plane_strain
    on a two-dimensional Cartesian mesh, where the formulation must be given.

    The structural formulations reduce a thin body to its mid-surface or its
    axis.  A plate follows the first-order (Mindlin) theory on a
    two-dimensional mesh of the mid-plane, with the variables ``disp_x``,
    ``disp_y``, ``disp_z``, ``phi_x`` and ``phi_y``.  A beam follows the
    Euler-Bernoulli or the Timoshenko theory on a one-dimensional mesh along
    :math:`x`, with the variables ``disp_x``, ``disp_z`` and
    ``bending_moment``, or ``phi`` in the displacement form.  A circular plate
    follows the first-order or the classical theory on a radial mesh in
    axisymmetric coordinates, with the variables ``disp_r``, ``disp_z`` and
    ``phi`` or ``bending_moment``.  A section is given by its material
    (``youngs_modulus``, ``poissons_ratio`` and the ``thickness`` of a plate
    or the ``cross_section`` of a beam) or by its section stiffnesses, as for
    a laminated section.

    The members in space take a mesh of fewer dimensions than its space.  A
    truss is a set of bars on a line mesh in one, two or three dimensions,
    which carry the axial force only, with ``cross_section={"area": A}`` and
    :math:`A` the area of a bar in :math:`\mathrm{m^2}`.  A shell is a set of
    flat facets on a quadrilateral surface mesh in three dimensions, with
    membrane, bending and shear, its ``thickness``, and the variables
    ``disp_x``, ``disp_y``, ``disp_z``, ``rotation_x``, ``rotation_y`` and
    ``rotation_z``, the rotations about the global axes."""

    type_name: ClassVar[str] = "solid_mechanics"
    flux_name: ClassVar[str] = "traction"
    module: ClassVar[str] = "solid_mechanics"
    vector: ClassVar[bool] = True

    youngs_modulus: float | str | Sequence[float] | None = parameter(
        None,
        unit="Pa",
        description="Young's modulus E: a constant, a function, or an expression of the fields and the coordinates, or three moduli E1, E2, E3 along the material axes of an orthotropic continuum. In a structural formulation or a shell z is the height above the mid-surface, so E(z) gives a graded section (a shell evaluates E at the point x + z n through the thickness, with n the unit normal of its facet). A truss takes a number. Default: for a continuum, the property youngs_modulus of the property objects of the problem, such as constant or parsed, so that the modulus can differ between blocks.",
    )
    poissons_ratio: float | Sequence[float] | None = parameter(
        None,
        description="Poisson's ratio nu, with -1 < nu < 0.5, or nu12, nu13, nu23 of an orthotropic continuum. Default for a continuum: the property poissons_ratio of the property objects of the problem, since no one value suits every material. A plate, a circular plate and a Timoshenko beam given by its material need it given, and an Euler-Bernoulli beam or a beam given by its section stiffnesses does not use it.",
    )
    shear_modulus: Sequence[float] | None = parameter(
        None,
        unit="Pa",
        description="The shear moduli G12, G13, G23 of an orthotropic continuum. Default none: an isotropic material takes its shear modulus from E and nu.",
    )
    orientation: object = parameter(
        None,
        description="The material axes 1, 2, 3 of an orthotropic or anisotropic continuum. In three dimensions give axis 1 and axis 2 as two vectors of three components in the global axes, [a1, a2], and axis 3 is their cross product. In two dimensions give the angle of axis 1 from x about z, in degrees, and axis 3 is z, or the hoop direction of an axisymmetric problem. Default: the global axes, the axes of the mesh.",
    )
    stiffness: Sequence[Sequence[float]] | None = parameter(
        None,
        unit="Pa",
        description="The 6 x 6 stiffness matrix of a fully anisotropic continuum in the material axes, in the Voigt order 11, 22, 33, 23, 13, 12 with engineering shear strains, in place of the moduli. Default none: the stiffness follows from the moduli.",
    )
    #: The displacement field: the variables disp_x, disp_y, disp_z (disp_r, disp_z in axisymmetric coordinates).
    displacement: ClassVar[str] = "disp"
    formulation: str | None = parameter(
        None,
        description="The kinematics of the body: plane_stress, plane_strain, generalized_plane_strain, axisymmetric or three_dimensional for a continuum, plate, beam or circular_plate for a structural member, or truss or shell for a member in space. Default: three_dimensional on a three-dimensional mesh and axisymmetric in axisymmetric coordinates. A two-dimensional Cartesian problem must give plane_stress or plane_strain, because neither is right for every body.",
    )
    thickness: float | None = parameter(
        None,
        unit="m",
        description="The thickness of a plate, a circular plate or a shell given by its material, where it is required, or the out-of-plane thickness of a plane problem, which multiplies its equations and its traction and pressure conditions. A three-dimensional, axisymmetric or beam problem refuses it. Default: none for a plate, and 1 m for a plane problem, so that its equations are per unit thickness.",
    )
    body_force: Sequence[float | str] | None = parameter(
        None,
        unit="N/m^3",
        description="Body force per unit volume, one value per component, each a constant or an expression of x, y, z and t, for example the weight [0, 0, -rho*g]. A truss or a shell takes numbers, one per direction of space. Default none: no body force.",
    )
    theory: str | None = parameter(
        None,
        description="The theory of a structural formulation: Euler-Bernoulli or Timoshenko for a beam, first_order or classical for a circular plate, and first_order for a plate. Default: Euler-Bernoulli for a beam, the theory of a slender beam, and first_order for a plate or a circular plate, which holds for thick and thin plates.",
    )
    beam_formulation: str | None = parameter(
        None,
        description="The formulation of the Timoshenko beam: mixed, with the bending moment as a variable, or displacement, with the rotation phi as a variable and the shear term integrated at the element centre. Default mixed: it is free of shear locking with full integration. The Euler-Bernoulli beam and the classical circular plate are always mixed, and the first-order plates always take the displacement form.",
    )
    cross_section: Mapping[str, float] | None = parameter(
        None,
        description="The cross-section of a beam given by its material: {'width': b, 'height': h}, a rectangle through whose height the material law is integrated, or {'area': A, 'second_moment': I}, a homogeneous section. 'shear_correction' gives the shear correction factor, 5/6 for a rectangle, and a Timoshenko beam of a section given by its area needs it. A truss takes {'area': A}. Default none: a continuum and a plate have no cross-section.",
    )
    extensional_stiffness: float | None = parameter(
        None,
        unit="N",
        description="Section stiffness A of a structural formulation, per unit width for a plate and with the factor 1/(1 - nu^2), for a section that the material and the geometry do not describe, such as a laminated section. Default: computed from the material and the geometry.",
    )
    coupling_stiffness: float | None = parameter(
        None,
        unit="N m",
        description="Section stiffness B, the coupling of extension and bending, zero for a section symmetric about its mid-surface. Default: 0 when the section is given by its stiffnesses, and computed from the material otherwise.",
    )
    bending_stiffness: float | None = parameter(
        None,
        unit="N m^2",
        description="Section stiffness D, the bending stiffness. Default: computed from the material and the geometry.",
    )
    shear_stiffness: float | None = parameter(
        None,
        unit="N",
        description="Section stiffness S of the transverse shear, K G A with K the shear correction factor, G the shear modulus and A the area. Default: computed from the material and the geometry.",
    )
    transverse_load: float | str = parameter(
        0.0,
        unit="N/m^2 (plate) or N/m (beam)",
        description="Distributed load along the deflection disp_z of a structural formulation, per unit area of a plate (N/m^2) or per unit length of a beam (N/m), a constant or an expression. Default 0: no load.",
    )
    axial_load: float | str = parameter(
        0.0,
        unit="N/m",
        description="Distributed axial load of a beam, or radial load of a circular plate, per unit length. A plate takes its in-plane load as in_plane_load and refuses axial_load. Default 0: no load.",
    )
    in_plane_load: Sequence[float | str] | None = parameter(
        None,
        unit="N/m^2",
        description="Distributed in-plane load [f_x, f_y] of a plate. Default none: no load.",
    )
    foundation_modulus: float = parameter(
        0.0,
        unit="N/m^3 (plate) or N/m^2 (beam)",
        description="Modulus of an elastic (Winkler) foundation under a structural formulation: the pressure per unit deflection under a plate (N/m^3), or the force per unit length and unit deflection under a beam (N/m^2). Default 0: no foundation.",
    )
    von_karman: bool = parameter(
        False,
        description="The von Karman nonlinearity of moderately large deflections, for a structural formulation. Default false: the linear theory, which holds while the deflection stays small against the thickness.",
    )
    large_deformation: bool = parameter(
        False,
        description="Finite strain and rotation, with a pressure that follows the deformed surface. Linear elasticity, creep and plasticity then take the rotated small strain of Hughes and Winget. Available for plane strain, axisymmetric and three-dimensional continua, and in plane stress with a hyperelastic law. Default false: small strain, which holds while the strains and the rotations stay small. A hyperelastic law (elasticity other than linear, or strain_energy) is always at finite strain.",
    )
    elasticity: str = parameter(
        "linear",
        description="The elastic law: linear (Hooke's law), neo-Hookean (the compressible neo-Hookean hyperelastic law) or Saint_Venant_Kirchhoff (the hyperelastic law S = lambda tr(E) I + 2 G E of the second Piola-Kirchhoff stress S and the Green-Lagrange strain E). The hyperelastic laws are at finite strain. Default linear, the law of an elastic solid at small strain.",
    )
    strain_energy: str | None = parameter(
        None,
        description="A hyperelastic law given by its strain energy psi (J/m^3) per unit volume of the stress-free state, as an expression of I_C and trace_C_squared (the traces of C and C^2, with C = F^T F the right Cauchy-Green tensor of the elastic part F of the deformation gradient), J (det F), shear_modulus and first_Lame_parameter (Pa, from youngs_modulus and poissons_ratio). The code differentiates it twice, so the Jacobian is exact. It is at finite strain and takes no elasticity. Default none: the law of elasticity.",
    )
    creep: Mapping[str, object] | None = parameter(
        None,
        description="Creep: {'rate': the equivalent creep strain rate (1/s) as an expression of von_mises_stress, temperature, t, equivalent_creep_strain and grain_size, 'constants': {name: value}, 'grain_size': the grain size (m)}. Default none: no creep.",
    )
    plasticity: Mapping[str, object] | None = parameter(
        None,
        description="J2 plasticity: {'yield_stress': an expression of equivalent_plastic_strain and temperature (the isotropic hardening), 'kinematic_hardening': {'C': [C_k], 'gamma': [gamma_k]} (the back stresses of Armstrong-Frederick or Chaboche, with gamma 0 for Prager), 'constants': {name: value}}. Default none: an elastic material.",
    )
    phase_field_fracture: Mapping[str, float] | None = parameter(
        None,
        description="Phase-field fracture (AT2): {'toughness': the fracture toughness Gc (J/m^2), 'length': the length scale l (m)}. It adds the field damage d, 0 intact and 1 broken, which degrades the tensile part of the elastic energy by (1 - d)^2 and does not heal. Available for plane strain, generalized plane strain and axisymmetric continua. Default none: no fracture.",
    )
    viscoelasticity: Mapping[str, object] | None = parameter(
        None,
        description="Viscoelasticity (generalized Maxwell): {'shear': [[G_i, tau_i], ...], 'bulk': [[K_i, tau_i], ...]}, branches of modulus G_i or K_i (Pa) that relax as exp(-t/tau_i), with tau_i in s, beside the long-term youngs_modulus and poissons_ratio, which a steady solve takes alone. Default none: an elastic material.",
    )
    temperature: float | str | None = parameter(
        None,
        unit="K",
        description="The temperature that creep and a temperature-dependent yield stress read: the name of a field, such as that of heat_conduction, or a number. Default: the temperature of a thermal_expansion coupling, so that one temperature field drives the expansion and the material law.",
    )
    axial_force: float | str = parameter(
        0.0,
        unit="N",
        description="The axial force F along z that the section of a generalized_plane_strain problem carries, which sets its uniform axial strain strain_zz, one unknown of the whole section. Default 0: a long body free along its axis.",
    )
    density: float | str | None = parameter(
        None,
        unit="kg/m^3",
        description="Density rho of the inertia rho d2u/dt2, a constant or an expression of x, y and z. A transient run integrates the inertia by the generalized-alpha method, and an eigenvalue study takes it for the natural frequencies. A beam takes rho A and, in the displacement form of Timoshenko, the rotary inertia rho I, and a plate takes rho h and rho h^3/12, with A the area, I the second moment of area and h the thickness. Default none: no inertia, for statics and for quasi-static creep in time.",
    )
    damping: Mapping[str, float] | None = parameter(
        None,
        description="Rayleigh damping C = a M + b K of a solid with a density, with M the mass matrix and K the elastic stiffness at small strain: {'mass': a (1/s), 'stiffness': b (s)}, either or both. Default none: no damping.",
    )
    initial_condition: Mapping[str, object] = parameter(
        default_factory=dict,
        description="The initial state of a transient run with inertia: {'displacement': [...], 'velocity': [...]}, one value per displacement, each a constant or an expression of x, y and z. Default: undeformed and at rest, the state before any load.",
    )
    block: Sequence[str] = parameter(
        (),
        description="Blocks where the equations hold. Default: every block, so that a body of one material needs no block names.",
    )

    @property
    def _structural(self) -> bool:
        return self.formulation in STRUCTURAL

    @property
    def _anisotropic(self) -> bool:
        """An orthotropic or fully anisotropic continuum."""
        return self.stiffness is not None or isinstance(self.youngs_modulus, (list, tuple))

    @property
    def _hyperelastic(self) -> bool:
        """A hyperelastic law: a named one or a strain energy."""
        return self.elasticity != "linear" or self.strain_energy is not None

    @property
    def _finite(self) -> bool:
        """Finite strain: asked for, or by a hyperelastic material."""
        return self.large_deformation or self._hyperelastic

    def __post_init__(self):
        """The checks of the parameters that need no problem."""
        if self.thickness is not None and not self.thickness > 0.0:
            raise InputError(
                f"solid_mechanics: the thickness must be positive, not {self.thickness}."
            )
        for key, value in (self.cross_section or {}).items():
            if not (isinstance(value, (int, float)) and value > 0.0):
                raise InputError(
                    f"solid_mechanics: cross_section['{key}'] must be a positive number, not {value!r}."
                )
        if not self.foundation_modulus >= 0.0:
            raise InputError("solid_mechanics: foundation_modulus must not be negative.")
        inertia.check_inertia(self)
        continuum.check_material(self)
        if self.formulation is not None and self.formulation not in FORMULATIONS:
            raise InputError(
                f"solid_mechanics: unknown formulation '{self.formulation}'. Use {', '.join(FORMULATIONS)}."
            )
        structures.check_structural_parameters(self)

    def _attach(self, problem) -> None:
        if self.formulation in MEMBERS:
            structures.check_member(self, problem)
            super()._attach(problem)
            return
        if problem.mesh.space_dimension > problem.mesh.dimension:
            raise InputError(
                f"{label(self.type_name, self.name)}: the mesh is a {'curve' if problem.mesh.dimension == 1 else 'surface'} in a space of {problem.mesh.space_dimension} dimensions; give formulation='{'truss' if problem.mesh.dimension == 1 else 'shell'}'."
            )
        if self.formulation is None:
            if problem.coordinates == "axisymmetric":
                self.formulation = "axisymmetric"
            elif problem.mesh.dimension == 3:
                self.formulation = "three_dimensional"
            else:
                raise InputError(
                    f"{label(self.type_name, self.name)}: give formulation='plane_stress' (a thin plate loaded in its plane) or formulation='plane_strain' (a long body restrained along its axis) for a two-dimensional problem."
                )
        if (
            self.formulation in ("axisymmetric", "circular_plate")
            and problem.coordinates != "axisymmetric"
        ):
            raise InputError(
                f"{label(self.type_name, self.name)}: the {self.formulation} formulation needs dualmesh.Problem(mesh, coordinates='axisymmetric')."
            )
        where = label(self.type_name, self.name)
        needed = {"plate": 2, "beam": 1, "circular_plate": 1}.get(self.formulation)
        if needed is not None and problem.mesh.dimension != needed:
            raise InputError(
                f"{where}: the {self.formulation} formulation needs a {needed}-dimensional mesh ({'of the mid-plane' if needed == 2 else 'along the axis or the radius'}), and this mesh has {problem.mesh.dimension} dimensions."
            )
        if self.thickness is not None and self.formulation in (
            "three_dimensional",
            "axisymmetric",
            "beam",
        ):
            what = "its cross_section" if self.formulation == "beam" else "no thickness"
            raise InputError(
                f"{where}: the {self.formulation} formulation takes {what}, so the thickness is not given."
            )
        if self.body_force is not None:
            components = 3 if self.formulation == "three_dimensional" else 2
            if len(self.body_force) != components:
                raise InputError(
                    f"{where}: body_force has {len(self.body_force)} values, and the {self.formulation} formulation needs {components}, one per component of the displacement."
                )
        if self.formulation in STRUCTURAL:
            # The one-point reduced integration of the shear unlocks the linear line and
            # quadrilateral elements only; a linear triangle still locks (0.04 of the
            # deflection of a thin plate on an 8 x 8 mesh), and triangles come with the shells.
            allowed = {"plate": {"Quad4"}}.get(self.formulation, {"Edge2"})
            mesh = problem.mesh
            other = sorted(mesh.element_types() - allowed)
            if other:
                raise InputError(
                    f"{where}: the {self.formulation} formulation takes {', '.join(sorted(allowed))} elements, and this mesh has {', '.join(other)}. Its reduced integration of the shear is one point, which unlocks these elements only."
                )
        if self._structural:
            structures.section(self)
        super()._attach(problem)

    def _plane_thickness(self) -> float:
        """The out-of-plane thickness of a plane problem: 1 m (per unit thickness) when not given."""
        return 1.0 if self.thickness is None else float(self.thickness)

    def _third(self) -> str:
        """The third variable of a beam or a circular plate."""
        mixed = self.theory in ("Euler-Bernoulli", "classical") or (
            self.theory == "Timoshenko" and self.beam_formulation != "displacement"
        )
        return "bending_moment" if mixed else "phi"

    def variable_names(self) -> list[str]:
        if self.formulation == "truss":
            return vector_components(self.displacement, self._problem.mesh.space_dimension)
        if self.formulation == "shell":
            return vector_components(self.displacement, 3) + [
                "rotation_x",
                "rotation_y",
                "rotation_z",
            ]
        if self.formulation == "plate":
            return ["disp_x", "disp_y", "disp_z", "phi_x", "phi_y"]
        if self.formulation == "beam":
            return ["disp_x", "disp_z", self._third()]
        if self.formulation == "circular_plate":
            return ["disp_r", "disp_z", self._third()]
        if self.formulation == "axisymmetric":
            return vector_components(self.displacement, 2, axisymmetric=True) + (
                ["damage"] if self.phase_field_fracture is not None else []
            )
        extra = ["damage"] if self.phase_field_fracture is not None else []
        if self.formulation == "generalized_plane_strain":
            # The uniform axial strain, one unknown of the whole section.
            return vector_components(self.displacement, 2) + ["strain_zz"] + extra
        dimension = 3 if self.formulation == "three_dimensional" else 2
        return vector_components(self.displacement, dimension) + extra

    def _dimension(self) -> int:
        return len([v for v in self.variable_names() if v.startswith("disp_")])

    def add_boundary_condition(self, condition, boundary=None, **parameters):
        # The free ends of a mixed beam or classical plate are derived from the end conditions when
        # the problem is first built; a condition added later would leave that derivation stale.
        if getattr(self, "_holds_free_ends", False):
            raise InputError(
                f"{label(self.type_name, self.name)}: the end conditions of a {self.formulation} are read when the problem is first solved, and this one comes after. Give every condition before the first solve."
            )
        if condition == "contact":
            return interfaces.add_contact(self, boundary, parameters)
        if condition == "cohesive":
            return interfaces.add_cohesive_zone(self, boundary, parameters)
        if condition == "rotation":
            return rotations.add_rotation(self, boundary, parameters)
        if is_condition_text(condition) and condition_sides(condition)[0] == "torque":
            return rotations.add_torque(self, condition_sides(condition)[1], boundary, parameters)
        if is_condition_text(condition) and condition_sides(condition)[0] == "normal_displacement":
            return interfaces.add_roller(self, condition_sides(condition)[1], boundary, parameters)
        return super().add_boundary_condition(condition, boundary=boundary, **parameters)

    def j_integral(self, crack_tip, radii) -> float:
        """The J-integral (J/m^2) of a crack along x whose tip is at ``crack_tip``, by the domain
        integral over the ring between ``radii`` (inner, outer), with traction-free faces.  It
        does not depend on the ring, a check of the mesh."""
        return float(continuum.crack_tip_integrals(self, crack_tip, radii)[0])

    def stress_intensity(self, crack_tip, radii) -> dict:
        """The stress intensity factors {"I": K_I, "II": K_II} (Pa m^0.5) of a crack along x
        whose tip is at ``crack_tip``, by the interaction integral with the Williams fields over
        the ring between ``radii``."""
        _, first, second = continuum.crack_tip_integrals(self, crack_tip, radii)
        return {"I": float(first), "II": float(second)}

    def contact_pressure(self, secondary: str) -> dict:
        """The contact pressure (Pa) at the nodes of the secondary surface of a contact: {node:
        pressure}."""
        return interfaces.contact_pressure(self, secondary)

    def _symmetry(self, boundary, where: str):
        """A plane of symmetry: the normal displacement of a continuum, and the axial
        displacement and the slope of a beam end."""
        if not self._structural:
            return super()._symmetry(boundary, where)
        return structures.beam_symmetry(self, boundary, where)

    def _fixed_variables(self) -> list[str]:
        # Clamped: every displacement and rotation; in the mixed forms the slope vanishes by
        # itself at a clamped end.
        return [v for v in self.variable_names() if v != "bending_moment"]

    def _initial_condition(self, variable: str):
        return inertia.initial_state(self, "displacement", variable)

    def _initial_velocity(self, variable: str):
        return inertia.initial_state(self, "velocity", variable)

    @property
    def displacements(self) -> list[str]:
        """The displacement variables, one per component."""
        return self.variable_names()

    def reaction_force(self, boundary: str) -> dict:
        """The force on the body through ``boundary``, in N, or in N/m in a
        plane problem, by direction: ``{"x": Fx, "y": Fy}``.  At a support it
        is the reaction, the force of the support on the body, conserved
        exactly.  On a loaded boundary it is the load.  A plate or a beam gives the
        force along each of its displacements."""
        return {
            v[len("disp_") :]: -self._problem.boundary_flux(v, boundary)
            for v in self.variable_names()
            if v.startswith("disp_")
        }

    def _variable_units(self) -> dict:
        units = {"phi": "rad", "phi_x": "rad", "phi_y": "rad", "bending_moment": "N m"}
        return {v: units.get(v, "m") for v in self.variable_names()}

    def stress(self, z=None):
        """The stress at the element centres, one row per element.

        Parameters
        ----------
        z : float or {"top", "mid", "bottom"}, optional
            For a structural formulation, the height above the mid-surface
            (m) at which the stress is evaluated.  The words are the faces and
            the mid-surface of a plate or a rectangular beam section.  A
            continuum takes no height.

        Returns
        -------
        numpy.ndarray
            A continuum: the six components of the stress property of its
            stress model (xx, yy, zz, yz, xz, xy).  A plate: sigma_xx,
            sigma_yy, sigma_xy.  A beam: sigma_xx.  A circular plate:
            sigma_rr, sigma_tt.  The structural stresses come from the
            material law of the section, with its thermal strain.
        """
        if self._structural:
            return structures.section_stress(self, z)
        if z is not None:
            raise InputError(
                f"{label(self.type_name, self.name)}: a continuum has no height z; give stress() without it."
            )
        return self._problem.property_at_centroids("stress")

    def _build(self, problem) -> None:
        if self.formulation in MEMBERS:
            structures.build_member(self, problem)
            return
        inertia.build_inertia(self, problem)
        interfaces.build_interfaces(self, problem)
        interfaces.build_rollers(self, problem)
        if self._structural:
            structures.build_structural(self, problem)
            return
        continuum.build_continuum(self, problem)

    def _resolve(self, coordinates, dimension):
        if self.formulation is None:
            if coordinates == "axisymmetric":
                self.formulation = "axisymmetric"
            elif dimension == 3:
                self.formulation = "three_dimensional"
            else:
                raise InputError(
                    f"{label(self.type_name, self.name)}: give formulation='plane_stress' or formulation='plane_strain' for a two-dimensional problem."
                )

    def _manufactured_terms(self, fields, coordinates, dimension):
        return continuum.manufactured_terms(self, fields, coordinates, dimension)
