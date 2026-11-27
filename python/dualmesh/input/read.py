"""Reads YAML, TOML and CSV files into the input tree, and writes the tree back.

The tree holds only mappings with text keys, lists, text, integers, floats and
Booleans, so YAML and TOML hold the same trees.
"""

import csv
import math
import re
from pathlib import Path

import tomllib
import yaml

from dualmesh._core import InputError

YAML_suffixes = (".yaml", ".yml")
TOML_suffixes = (".toml",)


class YAML_loader(yaml.SafeLoader):
    """The safe YAML loader, with the Booleans, integers and floats of YAML 1.2.

    YAML 1.1, which PyYAML follows, reads yes, no, on and off as Booleans, 010 as
    the octal 8, 1:30 as the base-60 number 90 and 1e-3 as a text. YAML 1.2 and TOML
    read true and false as the only Booleans, 0o10 as the octal 8 and 1e-3 as a
    float, so this loader does too. A text such as yes or 010 stays a text.
    """

    # A copy of the resolvers of SafeLoader without those of Booleans and integers,
    # so the plain safe loader stays as it is.
    yaml_implicit_resolvers = {
        first: [
            (tag, pattern)
            for tag, pattern in resolvers
            if tag not in ("tag:yaml.org,2002:bool", "tag:yaml.org,2002:int")
        ]
        for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }


YAML_loader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)
# The integer constructor of PyYAML reads a leading 0 as octal, and int(text, 8) accepts the prefix 0o.
YAML_loader.add_implicit_resolver(
    "tag:yaml.org,2002:int",
    re.compile(r"^(?:[-+]?(?:0|[1-9][0-9]*)|0o[0-7]+|0x[0-9a-fA-F]+)$"),
    list("-+0123456789"),
)
YAML_loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)[eE][-+]?[0-9]+$"),
    list("-+0123456789."),
)


def read(path: str | Path) -> dict:
    """Reads a .yaml, .yml or .toml file into a tree."""
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix in YAML_suffixes:
            with open(path, encoding="utf-8") as file:
                tree = yaml.load(file, Loader=YAML_loader)
        elif suffix in TOML_suffixes:
            with open(path, "rb") as file:
                tree = tomllib.load(file)
        else:
            raise InputError(f"{path}: the file name must end in .yaml, .yml or .toml.")
    except (yaml.YAMLError, tomllib.TOMLDecodeError) as error:
        raise InputError(f"{path}: {error}") from error
    except OSError as error:
        raise InputError(f"{path}: {error.strerror}") from error
    if not isinstance(tree, dict):
        raise InputError(
            f"{path}: the file must hold a mapping of sections, such as mesh and physics."
        )
    return tree


header_cell = re.compile(r"\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:\[(?P<unit>[^\[\]]*)\])?\s*")


def read_table(path: str | Path) -> dict:
    """Reads a CSV table into the columns of the inline table form.

    The first line names each column with its unit in square brackets, as in
    `temperature [K], thermal_conductivity [W/(m*K)]`. A dimensionless column may
    leave out the brackets. Each further line holds one number per column. The
    result maps each column name to {"unit": unit, "values": [...]}, in the order
    of the columns, with the numbers in the unit of the column.
    """
    path = Path(path)
    try:
        # The codec utf-8-sig drops the byte order mark that spreadsheet programs write.
        with open(path, encoding="utf-8-sig", newline="") as file:
            lines = [
                (number, row)
                for number, row in enumerate(csv.reader(file), start=1)
                if any(cell.strip() for cell in row)
            ]
    except OSError as error:
        raise InputError(f"{path}: {error.strerror}") from error
    except csv.Error as error:
        raise InputError(f"{path}: {error}") from error
    if not lines:
        raise InputError(
            f"{path}: the file is empty. Its first line names each column with its unit, "
            "as in 'temperature [K], thermal_conductivity [W/(m*K)]'."
        )
    header_number, header = lines[0]
    columns = {}
    for cell in header:
        match = header_cell.fullmatch(cell)
        if match is None:
            raise InputError(
                f"{path}, line {header_number}: the header cell '{cell.strip()}' is not a name "
                "with its unit in square brackets, as in 'temperature [K]'."
            )
        name = match["name"]
        if name in columns:
            raise InputError(f"{path}, line {header_number}: the column '{name}' appears twice.")
        unit = (match["unit"] or "").strip() or "1"
        columns[name] = {"unit": unit, "values": []}
    for number, row in lines[1:]:
        if len(row) != len(columns):
            raise InputError(
                f"{path}, line {number}: expects {len(columns)} numbers, got {len(row)}."
            )
        for column, cell in zip(columns.values(), row, strict=True):
            try:
                value = float(cell)
            except ValueError:
                raise InputError(
                    f"{path}, line {number}: '{cell.strip()}' is not a number."
                ) from None
            if not math.isfinite(value):
                raise InputError(f"{path}, line {number}: '{cell.strip()}' is not a finite number.")
            column["values"].append(value)
    return columns


def write_YAML(tree: dict, path: str | Path) -> None:
    """Writes a tree to a YAML file, in the order of the tree."""
    # safe_dump passes sort_keys and allow_unicode on to dump_all.
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(tree, file, sort_keys=False, allow_unicode=True)


def write_TOML(tree: dict, path: str | Path) -> None:
    """Writes a tree to a TOML file."""
    with open(path, "w", encoding="utf-8") as file:
        file.write(TOML_text(tree))


# TOML writing. The standard library reads TOML but does not write it.

bare_key = re.compile(r"[A-Za-z0-9_-]+")
TOML_escapes = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def TOML_text(tree: dict) -> str:
    """The TOML document of a tree.

    The values of a table come before its sub-tables, as TOML requires. A table that
    holds only sub-tables gets no header of its own, so {"mesh": {"rectangle": {...}}}
    becomes [mesh.rectangle]. A mapping inside a list becomes an inline table.
    """
    lines: list[str] = []
    _write_table(tree, [], lines)
    return "\n".join(lines).lstrip("\n") + "\n"


def _write_table(table: dict, keys: list[str], lines: list[str]) -> None:
    values = {key: value for key, value in table.items() if not isinstance(value, dict)}
    tables = {key: value for key, value in table.items() if isinstance(value, dict)}
    if keys and (values or not tables):
        lines.append("")
        lines.append("[" + ".".join(_key(key) for key in keys) + "]")
    for key, value in values.items():
        lines.append(f"{_key(key)} = {_value(value, [*keys, key])}")
    for key, value in tables.items():
        _write_table(value, [*keys, key], lines)


def _key(key) -> str:
    if not isinstance(key, str):
        raise InputError(f"TOML keys must be text, got the key {key!r}.")
    return key if bare_key.fullmatch(key) else _string(key)


def _string(text: str) -> str:
    escaped = "".join(
        TOML_escapes.get(character)
        or (
            f"\\u{ord(character):04X}"
            if ord(character) < 0x20 or ord(character) == 0x7F
            else character
        )
        for character in text
    )
    return f'"{escaped}"'


def _value(value, keys: list[str]) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_value(entry, keys) for entry in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(
                f"{_key(key)} = {_value(entry, [*keys, key])}" for key, entry in value.items()
            )
            + "}"
        )
    raise InputError(f"{'.'.join(keys)}: TOML cannot hold the value {value!r}.")
