# SPDX-License-Identifier: LGPL-2.1-or-later
"""The reading of input files: YAML and TOML give the same tree, YAML reads numbers and
Booleans as TOML does, a CSV table gives its columns with their units, and a written tree
reads back unchanged."""

import re

import pytest
import tomllib
from dualmesh._core import InputError
from dualmesh.input.read import TOML_text, read, read_table, write_TOML, write_YAML

tree = {
    "mesh": {"rectangle": {"size": ["0.2 m", "0.1 m"], "cells": [40, 20]}},
    "physics": {"heat_conduction": {}},
    "properties": {
        "thermal_conductivity": {"fuel": "1/([0.0375 m*K/W] + [2.165e-4 m/W]*temperature)"},
        "Poisson_ratio": 0.3,
    },
    "time": {"end_time": "1000 s", "step_control": {"by": "error", "relative_tolerance": 1e-3}},
    "output": {"fields": ["temperature", "displacement"], "interval": 10},
}


@pytest.mark.parametrize("write, name", [(write_YAML, "case.yaml"), (write_TOML, "case.toml")])
def test_a_written_tree_reads_back_unchanged(tmp_path, write, name):
    write(tree, tmp_path / name)
    assert read(tmp_path / name) == tree


def test_YAML_reads_numbers_and_truth_values_as_TOML_does(tmp_path):
    """YAML 1.1 reads 1e-3 as a text, yes as true and 010 as the octal 8; TOML and YAML 1.2
    read 1e-3 as a number and true and false as the only Booleans."""
    path = tmp_path / "words.yaml"
    path.write_text("a: 1e-3\nb: yes\nc: true\nd: 010\ne: 0o10\nf: 1:30\ng: 10e-6 1/K\n")
    assert read(path) == {
        "a": 1.0e-3,
        "b": "yes",
        "c": True,
        "d": "010",
        "e": 8,
        "f": "1:30",
        "g": "10e-6 1/K",
    }


@pytest.mark.parametrize(
    "name, content, message",
    [
        ("case.json", "{}", "must end in .yaml, .yml or .toml"),
        ("case.yaml", "- mesh\n- physics\n", "must hold a mapping of sections"),
        ("case.toml", "mesh = \n", "case.toml"),
    ],
)
def test_a_bad_file_is_refused_with_its_name(tmp_path, name, content, message):
    path = tmp_path / name
    path.write_text(content)
    with pytest.raises(InputError, match=re.escape(message)):
        read(path)


def test_a_CSV_table_gives_its_columns_with_their_units(tmp_path):
    """A spreadsheet writes a byte order mark, and a dimensionless column may leave out its unit."""
    path = tmp_path / "table.csv"
    path.write_text(
        "temperature [K],  burnup_fraction , thermal_conductivity [ W/(m*K) ]\n300, 0, 7.6\n\n600, 0.5, 5.0\n",
        encoding="utf-8-sig",
    )
    assert read_table(path) == {
        "temperature": {"unit": "K", "values": [300.0, 600.0]},
        "burnup_fraction": {"unit": "1", "values": [0.0, 0.5]},
        "thermal_conductivity": {"unit": "W/(m*K)", "values": [7.6, 5.0]},
    }


@pytest.mark.parametrize(
    "content, message",
    [
        (
            "temperature (K), thermal_conductivity [W/(m*K)]\n300, 7.6\n",
            "line 1: the header cell 'temperature (K)'",
        ),
        (
            "temperature [K], temperature [K]\n300, 300\n",
            "line 1: the column 'temperature' appears twice",
        ),
        (
            "temperature [K], thermal_conductivity [W/(m*K)]\n300\n",
            "line 2: expects 2 numbers, got 1",
        ),
        (
            "temperature [K], thermal_conductivity [W/(m*K)]\n300, nan\n",
            "line 2: 'nan' is not a finite number",
        ),
    ],
)
def test_a_bad_CSV_table_is_refused_with_its_line(tmp_path, content, message):
    path = tmp_path / "table.csv"
    path.write_text(content)
    with pytest.raises(InputError, match=re.escape(message)):
        read_table(path)


def test_TOML_text_escapes_quotes_backslashes_and_control_characters():
    text_tree = {
        "output": {"file": 'a "quoted"\\ name\twith\ncontrol \x7f characters'},
        "list": [{"x": 1.0e-17}],
    }
    assert tomllib.loads(TOML_text(text_tree)) == text_tree
