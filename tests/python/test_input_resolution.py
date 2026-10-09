# SPDX-License-Identifier: LGPL-2.1-or-later
"""The kinds of parameters (required, default and substitute) and their
resolution for the active models: a value of the case that is not given is
replaced by a typical value with one warning, a parameter that no active
model reads is neither required nor substituted, and a value given for it
is accepted and reported."""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass

import pytest
from dualmesh._core import InputError
from dualmesh.parameters import (
    SubstitutedValueWarning,
    When,
    describe,
    describe_fields,
    keyword_checked,
    parameter,
    resolve_inputs,
    substitute,
)

WITH_CREEP = When("with creep", lambda group, models: models["creep"])


@keyword_checked
@dataclass
class Specimen:
    """An input group with one parameter of each kind."""

    radius: float = parameter(unit="m", description="The specimen radius.")
    emissivity: float = parameter(
        0.8, description="The surface emissivity. Default 0.8.", reference="a handbook"
    )
    grain_radius: float | None = substitute(
        5.0e-6,
        unit="m",
        description="The grain radius.",
        source="typical of the class of material",
        effect="the creep rate",
        read_when=WITH_CREEP,
    )
    density_fraction: float | None = substitute(
        lambda group: 0.95,
        description="The density fraction.",
        source="5 % porosity",
        effect="the density",
    )
    creep_exponent: float | None = parameter(
        None, description="The creep exponent.", read_when=WITH_CREEP, required_when_read=True
    )


def test_a_substitute_is_used_with_one_warning_when_a_model_reads_it():
    given = Specimen(radius=4.0e-3, creep_exponent=1.0)
    groups, resolution = resolve_inputs({"part": given}, {"creep": True})
    specimen = groups["part"]
    assert specimen.grain_radius == 5.0e-6 and specimen.density_fraction == 0.95
    assert given.grain_radius is None and given.density_fraction is None
    assert [s.path for s in resolution.substituted] == [
        "part.grain_radius",
        "part.density_fraction",
    ]
    assert resolution.substituted_names("part") == {"grain_radius", "density_fraction"}
    rows = {
        row[0]: row
        for row in describe(specimen, {"creep": True}, resolution.substituted_names("part"))
    }
    assert (
        rows["grain_radius"][3] == "(substitute)"
        and rows["grain_radius"][4] == "typical of the class of material"
    )
    assert rows["emissivity"][3] == "(default)" and rows["emissivity"][4] == "a handbook"
    assert rows["radius"][3] == "given" and rows["radius"][4] == ""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolution.warn("Test")
    assert len(caught) == 1 and issubclass(caught[0].category, SubstitutedValueWarning)
    text = str(caught[0].message)
    assert (
        "part.grain_radius = 5e-06 m (typical of the class of material; the creep rate)" in text
        and "part.density_fraction" in text
    )


def test_a_parameter_that_no_active_model_reads_is_not_required_substituted_or_printed():
    groups, resolution = resolve_inputs({"part": Specimen(radius=4.0e-3)}, {"creep": False})
    specimen = groups["part"]
    assert specimen.grain_radius is None
    assert [s.path for s in resolution.substituted] == ["part.density_fraction"]
    assert resolution.unused == []
    names = [row[0] for row in describe(specimen, {"creep": False})]
    assert "grain_radius" not in names and "creep_exponent" not in names and "radius" in names


def test_a_value_that_no_active_model_reads_is_accepted_and_reported():
    groups, resolution = resolve_inputs(
        {"part": Specimen(radius=4.0e-3, grain_radius=8.0e-6, density_fraction=0.96)},
        {"creep": False},
    )
    assert groups["part"].grain_radius == 8.0e-6
    assert resolution.substituted == []
    assert resolution.unused_rows() == [["part.grain_radius", "8e-06", "m", "read with creep"]]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolution.warn("Test")


def test_a_parameter_required_when_read_is_named_when_missing():
    with pytest.raises(
        InputError,
        match=r"Specimen: missing required parameter 'creep_exponent' \(The creep exponent.\), read with creep",
    ):
        resolve_inputs({"part": Specimen(radius=4.0e-3)}, {"creep": True})


def test_every_parameter_is_read_outside_an_application():
    groups, resolution = resolve_inputs({"part": Specimen(radius=4.0e-3, creep_exponent=1.0)})
    assert groups["part"].grain_radius == 5.0e-6


def test_a_group_inside_a_group_is_resolved():
    @dataclass
    class Assembly:
        specimen: Specimen = parameter(
            default_factory=lambda: Specimen(radius=4.0e-3, creep_exponent=1.0),
            description="The specimen.",
        )

    groups, resolution = resolve_inputs({"assembly": Assembly()}, {"creep": True})
    assert groups["assembly"].specimen.grain_radius == 5.0e-6
    assert resolution.substituted_names("assembly.specimen") == {"grain_radius", "density_fraction"}
    assert resolution.substituted_names("assembly") == set()


def test_a_group_that_no_active_model_reads_is_not_resolved():
    @dataclass
    class Part:
        specimen: Specimen = parameter(
            default_factory=lambda: Specimen(radius=4.0e-3),
            description="The specimen.",
            read_when=WITH_CREEP,
        )

    groups, resolution = resolve_inputs({"part": Part()}, {"creep": False})
    assert groups["part"].specimen.density_fraction is None
    assert resolution.substituted == [] and resolution.unused == []
    groups, resolution = resolve_inputs(
        {"part": Part(specimen=Specimen(radius=5.0e-3))}, {"creep": False}
    )
    assert [u.path for u in resolution.unused] == ["part.specimen"]


def test_the_reference_states_each_kind():
    text = describe_fields(Specimen)
    assert (
        "grain_radius (real, m, substitute 5e-06 (typical of the class of material) when not given, read with creep)"
        in text
    )
    assert "density_fraction (real, substitute calculated (5 % porosity) when not given)" in text
    assert "creep_exponent (real, required with creep)" in text
    assert "radius (real, m, required)" in text


def test_a_substitute_needs_its_source_and_effect():
    with pytest.raises(ValueError, match="source of the typical value and its effect"):
        substitute(1.0, source="", effect="the creep rate")
    with pytest.raises(ValueError, match="required_when_read needs the default None"):
        parameter(1.0, required_when_read=True)
    assert dataclasses.fields(Specimen)[2].default is None


def test_a_parameter_is_set_as_an_attribute_and_a_misspelled_one_is_refused():
    specimen = Specimen(radius=4.0e-3, creep_exponent=1.0)
    specimen.grain_radius = 8.0e-6
    groups, resolution = resolve_inputs({"part": specimen}, {"creep": True})
    assert groups["part"].grain_radius == 8.0e-6 and groups["part"] is not specimen
    with pytest.raises(
        InputError, match="unknown parameter 'grain_raduis'. Did you mean 'grain_radius'"
    ):
        specimen.grain_raduis = 8.0e-6


def test_a_value_set_as_an_attribute_is_validated_when_the_inputs_are_resolved():
    @keyword_checked
    @dataclass
    class Coating:
        thickness: float = parameter(unit="m", description="The coating thickness.")

        def __post_init__(self):
            if not self.thickness > 0.0:
                raise ValueError("Coating: thickness must be positive.")

    coating = Coating(thickness=1.0e-4)
    coating.thickness = -1.0e-4
    with pytest.raises(ValueError, match="thickness must be positive"):
        resolve_inputs({"coating": coating})
