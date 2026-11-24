# SPDX-License-Identifier: LGPL-2.1-or-later
"""The kinds of parameters (required, default and substitute) and their
resolution for the active models: a value of the case that is not given is
replaced by a typical value with one warning, a parameter that no active
model reads is neither required nor substituted, and a value given for it
is accepted and reported."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from dualmesh._core import InputError
from dualmesh.parameters import (
    When,
    keyword_checked,
    parameter,
    resolve_inputs,
    substitute,
)

with_creep = When("with creep", lambda group, models: models["creep"])


@keyword_checked
@dataclass
class specimen_parameters:
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
        read_when=with_creep,
    )
    density_fraction: float | None = substitute(
        lambda group: 0.95,
        description="The density fraction.",
        source="5 % porosity",
        effect="the density",
    )
    creep_exponent: float | None = parameter(
        None, description="The creep exponent.", read_when=with_creep, required_when_read=True
    )


def test_a_parameter_is_set_as_an_attribute_and_a_misspelled_one_is_refused():
    specimen = specimen_parameters(radius=4.0e-3, creep_exponent=1.0)
    specimen.grain_radius = 8.0e-6
    groups, resolution = resolve_inputs({"part": specimen}, {"creep": True})
    assert groups["part"].grain_radius == 8.0e-6 and groups["part"] is not specimen
    with pytest.raises(
        InputError, match="unknown parameter 'grain_raduis'. Did you mean 'grain_radius'"
    ):
        specimen.grain_raduis = 8.0e-6
