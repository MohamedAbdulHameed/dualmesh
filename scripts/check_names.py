"""Checks the naming rule and the comment rule in the C++ and Python sources.

The naming rule:

- A name is full words in lower case joined by underscores.
- A person's name keeps its capitals (Young_modulus, Newton_solver).
- An abbreviation is all capitals (DMCDM_test_rule, to_SI), and the name of a
  program keeps the spelling of its authors (PETSc, Gmsh).
- A name has no shortened words: write coordinates for coords.
- No variable has the name of its type (mesh mesh in C++, mesh: mesh in
  Python). The type is the one the source writes: the C++ type before the
  name, without const, * and &, and the Python annotation. The two names
  match in their exact spelling, so path: pathlib.Path passes. The script
  does not infer a type from a value, so problem = dualmesh.problem() passes.
  In C++ it checks T name, and const T& name and const T* name. It skips
  T& name and T* name without const, since a * a and a && a read the same.

The comment rule: every comment starts with a capital letter, unless it
starts with an identifier written as code: a name in backticks, a name that
holds an underscore or a digit, or a name followed by (, ., ->, :: or [.
An abbreviation such as e.g. or i.e. is prose and fails. A comment that
starts with no letter, such as a formula or a number, has no letter to
capitalize and passes, except one that starts with ( and a lower-case word.
A comment after code on its line that is a unit, such as m/s or J/(kg*K),
passes.
Two kinds of comments that tools read also pass, in their full syntax: the
pragmas "# noqa", "# type: ignore", "# pragma: no cover", "# fmt: off" and
"// NOLINT", and the closing comment "// namespace dualmesh" after the brace
that closes a C++ namespace.

The script checks the names that the sources declare, including the aliases
of imports, and not the names they use from other libraries. It accepts the
names that a tool fixes only where the tool fixes them: the pytest fixtures
as arguments of the tests and fixtures under tests/, the arguments of the
pytest hooks as arguments of the pytest_ hooks under tests/, the PETSc
handle dm and argc and argv in C++, and the PyYAML class attribute
yaml_implicit_resolvers in Python. It also checks the names that
cpp/bindings exposes to Python, the file names, and the keys of
standard_names.toml.

Run it from any directory:

    python scripts/check_names.py            # the source directories of the repository
    python scripts/check_names.py cpp/mesh   # only these files or directories

It prints one line per problem and exits with status 1 when it finds one.
"""

import argparse
import ast
import io
import pathlib
import re
import sys
import tokenize
import tomllib

repository = pathlib.Path(__file__).resolve().parent.parent
# The directories that follow the rules so far; the others join as they are rewritten.
source_directories = ("tests/python", "examples", "verification", "scripts")
standard_names_file = repository / "python" / "dualmesh" / "input" / "standard_names.toml"
cpp_suffixes = {".h", ".cpp"}
python_suffixes = {".py"}
checked_file_suffixes = cpp_suffixes | python_suffixes | {".toml", ".yaml", ".yml"}

# Names of persons, written as the person writes them. Jacobian keeps the
# capital of Jacobi.
person_names = frozenset(
    {
        "Young",
        "Poisson",
        "Newton",
        "Lagrange",
        "Gauss",
        "Lobatto",
        "Hermite",
        "McNabb",
        "Foster",
        "Fourier",
        "Fick",
        "Boltzmann",
        "Stefan",
        "Avogadro",
        "Mises",
        "Coulomb",
        "Taylor",
        "Hood",
        "Jacobian",
        "Nanson",
        "Reddy",
        "Arrhenius",
        "Cauchy",
        "Kirchhoff",
        "Euler",
        "Bernoulli",
        "Timoshenko",
        "Mindlin",
        "Reissner",
        "Krylov",
        "Schur",
        "Dirichlet",
        "Neumann",
        "Robin",
        "Laplace",
        "Navier",
        "Stokes",
        "Reynolds",
        "Peclet",
        "Courant",
        "Kovasznay",
        "Cook",
        # The algebraic multigrid of hypre.
        "BoomerAMG",
        "Hughes",
        "Galerkin",
        "Petrov",
        "Hessian",
        "Hencky",
        "Piola",
        "Green",
        "Lame",
        "Voigt",
        "Mandel",
        "Kelvin",
        "Maxwell",
        "Norton",
        "Prandtl",
        "Nusselt",
        "Rayleigh",
        "Boussinesq",
        "Darcy",
        "Biot",
        "Fischer",
        "Burmeister",
        "Nitsche",
        "Simo",
        "Hill",
        "Drucker",
        "Prager",
        "Tresca",
        "Paris",
        "Griffith",
        "Irwin",
        "Hertz",
        "Mohr",
        "Raviart",
        "Thomas",
        "Nedelec",
        "Crank",
        "Nicolson",
        "Runge",
        "Kutta",
        "Adams",
        "Rosenbrock",
        "Gear",
        "Hilbert",
        "Sobolev",
        "Planck",
        "Celsius",
        "Kuhn",
        "Voce",
        "Miehe",
        "Prony",
        "Armstrong",
        "Frederick",
        "Chaboche",
    }
)

# Abbreviations, all in capitals, and the names of programs, as their authors
# write them (PETSc, Gmsh).
abbreviations = frozenset(
    {
        # The NAFEMS benchmarks LE1 and LE10.
        "LE1",
        "LE10",
        # The Roman numerals of the fracture modes, as in K_I and K_II.
        "I",
        "II",
        "III",
        "Gmsh",
        "DMCDM",
        "FEM",
        "MPI",
        "PETSc",
        "SLEPc",
        "SNES",
        "KSP",
        "PC",
        "TS",
        "SF",
        "DM",
        "IS",
        "SUPG",
        "PSPG",
        "BDF",
        "SI",
        "JIT",
        "ABI",
        "AD",
        "DWR",
        "AMG",
        "ISO",
        "CODATA",
        "SHA",
        "YAML",
        "TOML",
        "CSV",
        "VTK",
        "HDF5",
        "XDMF",
        "XML",
        "DMPlex",
        "EPS",
        "ST",
        "TSAdapt",
        "CI",
        "API",
        "CPU",
        "GPU",
        "MMS",
        "RHS",
        "LHS",
        "ID",
        "IO",
        "OS",
        "URL",
        "UTF8",
        "ASCII",
        "BDDC",
        "GAMG",
        "LU",
        "ILU",
        "CG",
        "GMRES",
        "SPD",
        "LBB",
        "CFL",
        "ODE",
        "PDE",
        "DAE",
        "NaN",
        "SPDX",
        "LGPL",
        "CLI",
        "NAFEMS",
        "L2",
        "H1",
        "J2",
        "AT2",
    }
)

# Lower-case words that are also person names: the unit names of the SI, which
# the SI writes in lower case, and English words.
person_names_in_lower_case_words = frozenset(
    {
        "newton",
        "pascal",
        "joule",
        "watt",
        "coulomb",
        "volt",
        "weber",
        "tesla",
        "henry",
        "siemens",
        "hertz",
        "kelvin",
        "ampere",
        "ohm",
        "farad",
        "green",
        "hill",
        "paris",
        "thomas",
        "crank",
        "gear",
        "robin",
        "hood",
        "foster",
        "lame",
    }
)

# English words that are also abbreviations in lower case.
abbreviations_that_are_words = frozenset({"is", "ad", "id", "io", "os", "st", "ts", "pc", "sf"})

# Shortened words and the full words that replace them.
shortened_words = {
    "coord": "coordinate",
    "coords": "coordinates",
    "idx": "index",
    "ind": "index",
    "num": "number",
    "cnt": "count",
    "tmp": "temporary",
    "temp": "temporary or temperature",
    "ptr": "pointer",
    "len": "length",
    "val": "value",
    "vals": "values",
    "arr": "array",
    "buf": "buffer",
    "msg": "message",
    "err": "error",
    "cfg": "configuration",
    "config": "configuration",
    "dim": "dimension",
    "dims": "dimensions",
    "elem": "element",
    "elems": "elements",
    "ctx": "context",
    "param": "parameter",
    "params": "parameters",
    "info": "information",
    "src": "source",
    "dst": "destination",
    "prev": "previous",
    "cur": "current",
    "curr": "current",
    "max": "maximum",
    "min": "minimum",
    "avg": "average",
    "init": "initial or initialize",
    "args": "arguments",
    "kwargs": "keyword arguments",
    "argc": "argument count",
    "argv": "arguments",
    "res": "residual or result",
    "jac": "Jacobian",
    "mat": "matrix",
    "vec": "vector",
    "dof": "degree of freedom",
    "dofs": "degrees of freedom",
    "nb": "nanobind",
    "np": "numpy",
    "pts": "points",
    "qp": "quadrature point",
    "eqn": "equation",
    "func": "function",
    "fn": "function",
    "obj": "object",
    "lib": "library",
    "dir": "directory",
    "str": "string",
    "doc": "document",
    "bc": "boundary condition",
    "bcs": "boundary conditions",
    "ic": "initial condition",
    "tol": "tolerance",
    "rel": "relative",
    "abs": "absolute",
    "iter": "iteration",
    "iters": "iterations",
    "loc": "local",
    "glob": "global",
    "nnz": "number of nonzeros",
}
# A one-letter word is a symbol (T, x), except n inside a longer name (n_nodes).
shortened_letters_in_longer_names = {"n": "number"}

# C++ names that a convention fixes: the PETSc DM handle dm, which
# mesh::dm() and function_space::dm() return, and the arguments of main.
fixed_cpp_names = frozenset({"dm", "argc", "argv"})

# Python names that a library fixes: the table of implicit resolvers that a
# PyYAML loader class overrides.
fixed_python_names = frozenset({"yaml_implicit_resolvers"})

# The built-in fixtures of pytest. They pass only as arguments of tests and fixtures.
pytest_fixture_names = frozenset(
    {
        "cache",
        "capfd",
        "capfdbinary",
        "caplog",
        "capsys",
        "capsysbinary",
        "doctest_namespace",
        "monkeypatch",
        "pytestconfig",
        "pytester",
        "record_property",
        "record_testsuite_property",
        "record_xml_attribute",
        "recwarn",
        "request",
        "subtests",
        "testdir",
        "tmp_path",
        "tmp_path_factory",
        "tmpdir",
        "tmpdir_factory",
    }
)
# The arguments of the pytest hooks. They pass only as arguments of pytest_ hooks.
pytest_hook_argument_names = frozenset(
    {
        "args",
        "config",
        "early_config",
        "item",
        "items",
        "metafunc",
        "parser",
        "pluginmanager",
        "session",
    }
)

# File names that a tool or the package layout fixes.
fixed_file_names = frozenset(
    {
        "kernel_abi.h",
        "cli.py",
        "__init__.py",
        "__main__.py",
        "conftest.py",
        "CMakeLists.txt",
    }
)

# Comments that tools read, in their full syntax. Text after a pragma is a
# comment of its own and follows the comment rule.
python_pragma = re.compile(
    r"(noqa(:\s*[A-Z]+[0-9]+(\s*,\s*[A-Z]+[0-9]+)*)?"
    r"|ruff: noqa(:\s*[A-Z]+[0-9]+(\s*,\s*[A-Z]+[0-9]+)*)?"
    r"|type: ignore(\[[a-z-]+(,\s*[a-z-]+)*\])?"
    r"|pragma: no (cover|branch)"
    r"|fmt: (off|on|skip))(?=$|[\s#.,;])"
)
cpp_pragma = re.compile(
    r"(NOLINT(NEXTLINE|BEGIN|END)?(\([A-Za-z0-9.,*-]+\))?"
    r"|clang-format (off|on))(?=$|[\s.,;])"
)
# The comment that clang-format puts after the brace that closes a namespace.
namespace_closing_comment = re.compile(r"namespace(\s+[A-Za-z_]\w*(::[A-Za-z_]\w*)*)?")
# A unit written as symbols joined by / or *, with powers: m/s, J/(kg*K), 1/s^2.
unit_expression = re.compile(r"[A-Za-z0-9()]+(\^-?[0-9]+)?([/*][A-Za-z0-9()]+(\^-?[0-9]+)?)+")
# A unit written as one symbol: m, mm, Pa, K.
unit_symbols = frozenset(
    {
        "m",
        "cm",
        "mm",
        "um",
        "km",
        "s",
        "ms",
        "h",
        "kg",
        "g",
        "K",
        "N",
        "kN",
        "Pa",
        "kPa",
        "MPa",
        "GPa",
        "J",
        "W",
        "rad",
        "deg",
        "Hz",
        "mol",
        "A",
        "V",
    }
)

cpp_keywords = frozenset(
    """
alignas alignof and and_eq asm auto bitand bitor bool break case catch char char8_t char16_t
char32_t class compl concept const consteval constexpr constinit const_cast continue co_await
co_return co_yield decltype default delete do double dynamic_cast else enum explicit export
extern false float for friend goto if inline int long mutable namespace new noexcept not not_eq
nullptr operator or or_eq private protected public register reinterpret_cast requires return
short signed sizeof static static_assert static_cast struct switch template this thread_local
throw true try typedef typeid typename union unsigned using virtual void volatile wchar_t while
xor xor_eq override final
""".split()
)
# Keywords that can stand right before a declared name, as a type or a specifier does.
cpp_declaration_keywords = frozenset(
    """
auto bool char char8_t char16_t char32_t const consteval constexpr constinit double explicit
extern float friend inline int long mutable register short signed static thread_local unsigned
virtual void volatile wchar_t
""".split()
)
# Keywords after which the next name is declared.
cpp_introducing_keywords = frozenset(
    {"class", "struct", "enum", "union", "namespace", "concept", "typename"}
)
# Tokens after which a name ends its declaration.
cpp_declaration_followers = frozenset({"(", ")", ";", ",", "=", "{", "[", ":"})
# Tokens that can stand right before the type of a declaration. A type after
# another token, such as = or return, starts an expression: a in x = a * b.
cpp_declaration_starts = frozenset({"", ";", "{", "}", "(", ",", ":", ">", "]"})
# PETSc, SLEPc and MPI names in their CamelCase convention, such as PetscInt or
# DMPlexCreate. A name with an underscore after the prefix, such as
# PETSc_arguments or MPI_communicator, is a dualmesh name and gets checked.
external_cpp_name = re.compile(
    r"^(Petsc|PETSc|SLEPc|Vec|Mat|DM|SNES|KSP|TS|PC|IS|EPS|ST|SF|MPI)[A-Z]"
)

lower_case_abbreviations = {name.lower(): name for name in abbreviations}
lower_case_person_names = {name.lower(): name for name in person_names}


def word_problem(word: str, word_count: int) -> str | None:
    """Returns what is wrong with one word of a name, or None."""
    if word in person_names or word in abbreviations:
        return None
    if len(word) == 1:
        if word_count > 1 and word in shortened_letters_in_longer_names:
            return f"write {shortened_letters_in_longer_names[word]!r} for {word!r}"
        return None
    if word == word.lower():
        if word in shortened_words:
            return f"write {shortened_words[word]!r} for the shortened word {word!r}"
        if word in lower_case_abbreviations and word not in abbreviations_that_are_words:
            return f"write the abbreviation {lower_case_abbreviations[word]!r} in capitals"
        if word in lower_case_person_names and word not in person_names_in_lower_case_words:
            return f"write the person name {lower_case_person_names[word]!r} with its capitals"
        return None
    if word.upper() == word:
        return f"{word!r} is not in the abbreviations of scripts/check_names.py"
    return f"{word!r} mixes capitals and lower case, and it is no person name or abbreviation"


def name_problems(name: str, fixed_names: frozenset[str] = frozenset()) -> list[str]:
    """Returns what is wrong with one name, or an empty list.

    fixed_names holds the names that a tool fixes where the name stands.
    """
    if name.startswith("__") and name.endswith("__"):
        return []
    stripped = name.strip("_")
    if not stripped or stripped in fixed_names:
        return []
    words = stripped.split("_")
    problems = []
    for word in words:
        if not word:
            problems.append("two underscores in a row")
            continue
        problem = word_problem(word, len(words))
        if problem:
            problems.append(problem)
    return problems


class report:
    """Collects the problems of every file."""

    def __init__(self):
        self.problems: set[tuple[str, int, str]] = set()

    def add(self, path: pathlib.Path, line: int, message: str) -> None:
        self.problems.add((display_path(path), line, message))

    def check_name(
        self,
        path: pathlib.Path,
        line: int,
        name: str,
        what: str,
        fixed_names: frozenset[str] = frozenset(),
    ) -> None:
        for problem in name_problems(name, fixed_names):
            self.add(path, line, f"{what} {name!r}: {problem}")

    def print_all(self) -> int:
        for path, line, message in sorted(self.problems):
            print(f"{path}:{line}: {message}")
        return len(self.problems)


def display_path(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(repository))
    except ValueError:
        return str(path)


# Comments.


def comment_problem(text: str, pragma: re.Pattern) -> str | None:
    """Returns what is wrong with the first line of a comment, or None.

    A comment that starts with a capital, with a backtick or with no letter
    passes, except one that starts with ( and a lower-case word. A lower-case
    first word passes when it is written as code: it
    holds an underscore or a digit, which no English word does, or a call, a
    member, a qualified name or a subscript follows it. An abbreviation such
    as e.g. or i.e. is no member. A pragma passes, and the text after it is
    checked as a comment of its own.
    """
    body = text.strip()
    while match := pragma.match(body):
        body = body[match.end() :].lstrip(" \t#.,;")
    if body.startswith("("):
        body = body[1:]
    if not body or not body[0].isalpha() or body[0].isupper():
        return None
    word = re.match(r"[A-Za-z_][A-Za-z0-9_]*", body).group(0)
    rest = body[len(word) :]
    if "_" in word or any(character.isdigit() for character in word):
        return None
    is_member = re.match(r"\.[A-Za-z_]", rest) and not re.match(r"\.[A-Za-z]\.", rest)
    if rest.startswith(("(", "::", "->", "[")) or is_member:
        return None
    return f"a comment starts with the lower-case word {word!r}"


def check_comment_blocks(path: pathlib.Path, comments, problems: report) -> None:
    """Checks the first line of each block of comments.

    comments holds (line, text, trailing) in line order. Full-line comments on
    consecutive lines form one block, and an empty comment line ends a block.
    A trailing comment, after code on its line, is a block of its own.
    """
    is_cpp = path.suffix in cpp_suffixes
    pragma = cpp_pragma if is_cpp else python_pragma
    previous_line = None
    for line, text, trailing in comments:
        starts_block = trailing or previous_line is None or line != previous_line + 1
        previous_line = None if trailing or not text.strip() else line
        if is_cpp and trailing and namespace_closing_comment.fullmatch(text.strip()):
            continue
        if trailing and (unit_expression.fullmatch(text.strip()) or text.strip() in unit_symbols):
            continue
        if starts_block:
            problem = comment_problem(text, pragma)
            if problem:
                problems.add(path, line, problem)


# Python sources.


def pytest_argument_kind(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Returns what the arguments of a function are when pytest calls it, or "argument".

    pytest passes arguments by name to a hook, a test and a fixture.
    """
    if node.name.startswith("pytest_"):
        return "argument of a pytest hook"
    if node.name.startswith("test_"):
        return "argument of a pytest test or fixture"
    for decorator in node.decorator_list:
        called = decorator.func if isinstance(decorator, ast.Call) else decorator
        if ast.unparse(called) in ("pytest.fixture", "fixture"):
            return "argument of a pytest test or fixture"
    return "argument"


def annotation_names(annotation: ast.expr | None) -> set[str]:
    """Returns the type names of an annotation: mesh for mesh, "mesh", dualmesh.mesh or mesh | None."""
    if isinstance(annotation, ast.Name):
        return {annotation.id}
    if isinstance(annotation, ast.Attribute):
        return {annotation.attr}
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return annotation_names(annotation.left) | annotation_names(annotation.right)
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            return annotation_names(ast.parse(annotation.value, mode="eval").body)
        except SyntaxError:
            return set()
    return set()


def python_declarations(tree: ast.AST, under_tests: bool, path: pathlib.Path, problems: report):
    """Yields (line, name, what) for every name a Python module declares.

    It reports a variable that has the name of its type. under_tests tells
    whether the module lies under tests/, where pytest runs it.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.lineno, node.name, "function"
            what = pytest_argument_kind(node) if under_tests else "argument"
            arguments = node.args
            for argument in [
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                arguments.vararg,
                arguments.kwarg,
            ]:
                if argument is not None and argument.arg not in ("self", "cls"):
                    yield argument.lineno, argument.arg, what
                    if argument.arg in annotation_names(argument.annotation):
                        problems.add(
                            path,
                            argument.lineno,
                            f"the variable {argument.arg!r} has the name of its type",
                        )
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            name = target.id if isinstance(target, ast.Name) else getattr(target, "attr", None)
            if name in annotation_names(node.annotation):
                problems.add(path, node.lineno, f"the variable {name!r} has the name of its type")
        elif isinstance(node, ast.Lambda):
            for argument in [
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
                node.args.vararg,
                node.args.kwarg,
            ]:
                if argument is not None:
                    yield argument.lineno, argument.arg, "argument"
        elif isinstance(node, ast.ClassDef):
            yield node.lineno, node.name, "class"
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            yield node.lineno, node.id, "variable"
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name)
            and node.value.id in ("self", "cls")
        ):
            yield node.lineno, node.attr, "attribute"
        elif isinstance(node, ast.ExceptHandler) and node.name:
            yield node.lineno, node.name, "variable"
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            yield node.lineno, node.name, "variable"
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.asname:
                    yield node.lineno, alias.asname, "import alias"


def python_comments(source: str):
    """Yields (line, text, trailing) for every comment of a Python source."""
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        line, column = token.start
        if line == 1 and token.string.startswith("#!"):
            continue
        trailing = bool(token.line[:column].strip())
        yield line, token.string.lstrip("#"), trailing


def read_python(path: pathlib.Path, problems: report):
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
        comments = list(python_comments(source))
    except (SyntaxError, tokenize.TokenError) as failure:
        problems.add(path, getattr(failure, "lineno", 0) or 0, f"cannot parse the file: {failure}")
        return [], []
    under_tests = display_path(path).startswith("tests/")
    return list(python_declarations(tree, under_tests, path, problems)), comments


# C++ sources.


def split_cpp(source: str):
    """Splits C++ source into code and comments.

    Returns the code with comments, string literals, character literals and
    preprocessor lines replaced by spaces, the code with only the comments
    replaced, and the comments as (line, text, trailing). Newlines stay, so
    line numbers stay valid.
    """
    code = []
    code_with_strings = []
    comments = []
    line = 1
    line_has_code = False
    index = 0
    length = len(source)

    def blank(text):
        return "".join("\n" if character == "\n" else " " for character in text)

    while index < length:
        character = source[index]
        following = source[index + 1] if index + 1 < length else ""
        if character == "/" and following == "/":
            end = source.find("\n", index)
            end = length if end == -1 else end
            comments.append((line, source[index + 2 : end].lstrip("/!"), line_has_code))
            code.append(blank(source[index:end]))
            code_with_strings.append(blank(source[index:end]))
            index = end
            continue
        if character == "/" and following == "*":
            end = source.find("*/", index + 2)
            end = length if end == -1 else end + 2
            text = source[index + 2 : end - 2]
            first_line = next((part for part in text.split("\n") if part.strip(" *!")), "")
            comments.append((line, first_line.strip().lstrip("*!"), line_has_code))
            code.append(blank(source[index:end]))
            code_with_strings.append(blank(source[index:end]))
            line += source.count("\n", index, end)
            index = end
            continue
        if character == "#" and not line_has_code:
            end = index
            while True:
                end = source.find("\n", end)
                if end == -1:
                    end = length
                    break
                if source[end - 1] != "\\":
                    break
                end += 1
            code.append(blank(source[index:end]))
            code_with_strings.append(blank(source[index:end]))
            line += source.count("\n", index, end)
            index = end
            continue
        raw = re.match(r'R"([^(\s]*)\(', source[index : index + 20]) if character == "R" else None
        if raw and (index == 0 or not (source[index - 1].isalnum() or source[index - 1] == "_")):
            closing = ")" + raw.group(1) + '"'
            end = source.find(closing, index + raw.end())
            end = length if end == -1 else end + len(closing)
            code.append(blank(source[index:end]))
            code_with_strings.append(source[index:end])
            line += source.count("\n", index, end)
            line_has_code = True
            index = end
            continue
        separator = (
            character == "'" and index > 0 and source[index - 1].isalnum() and following.isalnum()
        )
        if character in "\"'" and not separator:
            end = index + 1
            while end < length and source[end] != character and source[end] != "\n":
                end += 2 if source[end] == "\\" else 1
            end = min(end + 1, length)
            code.append(blank(source[index:end]))
            code_with_strings.append(source[index:end])
            line_has_code = True
            index = end
            continue
        code.append(character)
        code_with_strings.append(character)
        if character == "\n":
            line += 1
            line_has_code = False
        elif not character.isspace():
            line_has_code = True
        index += 1
    return "".join(code), "".join(code_with_strings), comments


cpp_token = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d[\w.]*|::|->|\.\.\.|&&|\S")


def cpp_tokens(code: str):
    """Returns the tokens of C++ code as (line, text)."""
    tokens = []
    line = 1
    position = 0
    for match in cpp_token.finditer(code):
        line += code.count("\n", position, match.start())
        position = match.start()
        tokens.append((line, match.group(0)))
    return tokens


def is_identifier(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text)) and text not in cpp_keywords


def is_type_like(text: str) -> bool:
    """Tells whether a token can stand right before the name in a declaration."""
    return (
        is_identifier(text)
        or text in cpp_declaration_keywords
        or text in (">", "*", "&", "&&", "...")
    )


def cpp_type_start(tokens, position: int) -> int:
    """Returns the position of the first token of the qualified name that ends at position."""
    start = position
    while start >= 2 and tokens[start - 1][1] == "::" and is_identifier(tokens[start - 2][1]):
        start -= 2
    if start >= 1 and tokens[start - 1][1] == "::":
        start -= 1
    return start


def cpp_declared_type(tokens, position: int) -> tuple[bool, str | None]:
    """Tells whether the name at position is declared, and returns its type name or None.

    The name follows a type, a specifier, *, &, && or a closing >. The type
    name is the name before it, after skipping const, volatile, * and &. A
    name after * or & without const is declared only when its type starts a
    statement or an argument, since a * b and a && b are expressions. Its type
    name is then None, since a * a reads like T* name.
    """
    back = position - 1
    skipped = set()
    while back > 0 and tokens[back][1] in ("*", "&", "&&", "const", "volatile"):
        skipped.add(tokens[back][1])
        back -= 1
    type_token = tokens[back][1]
    has_operator = bool(skipped & {"*", "&", "&&"})
    if not is_identifier(type_token):
        is_declared = (
            not has_operator or type_token in cpp_declaration_keywords or type_token == ">"
        )
        return is_declared, None
    start = cpp_type_start(tokens, back)
    leading = tokens[start - 1][1] if start > 0 else ""
    is_qualified = bool(skipped & {"const", "volatile"}) or leading in ("const", "volatile")
    if not has_operator or is_qualified:
        return True, type_token
    return leading in cpp_declaration_starts, None


def cpp_declarations(tokens):
    """Yields (line, name, what, type_name) for the names C++ code declares.

    type_name is the type of a declared variable as cpp_declared_type finds
    it, or None. The rules are heuristic: a name is declared when a type, a
    specifier, *, &, or a closing > comes before it and one of
    ( ) ; , = { [ : comes after it, or when class, struct, enum, union,
    namespace, concept or typename comes before it. A name after ::, . or ->
    and a name before :: are uses, never declarations.
    """
    count = len(tokens)
    for position, (line, text) in enumerate(tokens):
        if not is_identifier(text):
            continue
        before = tokens[position - 1][1] if position > 0 else ""
        after = tokens[position + 1][1] if position + 1 < count else ""
        if before in ("::", ".", "->") or after == "::":
            continue
        if before in cpp_introducing_keywords:
            ahead = tokens[position - 2][1] if position > 1 else ""
            if before == "namespace" and ahead == "using":
                continue
            yield line, text, "type" if before != "namespace" else "namespace", None
            continue
        if before == "using" and after == "=":
            yield line, text, "type", None
            continue
        if is_type_like(before) and after in cpp_declaration_followers:
            is_declared, type_name = cpp_declared_type(tokens, position)
            if is_declared:
                yield line, text, "name", type_name
    yield from cpp_enumerators(tokens)


def cpp_enumerators(tokens):
    """Yields the enumerators of every enum body as (line, name, what, None)."""
    count = len(tokens)
    for position, (_, text) in enumerate(tokens):
        if text != "enum":
            continue
        start = position + 1
        while start < count and tokens[start][1] not in ("{", ";"):
            start += 1
        if start >= count or tokens[start][1] == ";":
            continue
        depth = 0
        previous = "{"
        for line, item in tokens[start + 1 :]:
            if item in ("(", "[", "{"):
                depth += 1
            elif item in (")", "]"):
                depth -= 1
            elif item == "}":
                if depth == 0:
                    break
                depth -= 1
            elif depth == 0 and previous in ("{", ",") and is_identifier(item):
                yield line, item, "enumerator", None
            previous = item if depth == 0 else previous


# Names that bindings expose to Python: functions, classes, attributes, exceptions and arguments.
python_visible_name = re.compile(
    r'\.def(?:_\w+)?\(\s*"([^"]+)"'
    r'|class_<[^;{]*?>\(\s*\w+\s*,\s*"([^"]+)"'
    r'|exception<[^;{]*?>\(\s*\w+\s*,\s*"([^"]+)"'
    r'|\barg\(\s*"([^"]+)"'
)


def read_cpp(path: pathlib.Path, problems: report):
    source = path.read_text(encoding="utf-8")
    code, code_with_strings, comments = split_cpp(source)
    declarations = []
    for line, name, what, type_name in cpp_declarations(cpp_tokens(code)):
        if external_cpp_name.match(name):
            continue
        # Upper-case names are macros and include guards, which keep the macro convention.
        if name.upper() == name and len(name) > 1 and name not in abbreviations:
            continue
        declarations.append((line, name, what))
        if what == "name" and type_name == name:
            problems.add(path, line, f"the variable {name!r} has the name of its type")
    for match in python_visible_name.finditer(code_with_strings):
        name = next(group for group in match.groups() if group)
        line = code_with_strings.count("\n", 0, match.start()) + 1
        declarations.append((line, name, "name exposed to Python"))
    return declarations, comments


# Files.


def source_files(arguments: list[str]) -> list[pathlib.Path]:
    roots = [pathlib.Path(argument) for argument in arguments] or [
        repository / directory for directory in source_directories
    ]
    files = []
    for root in roots:
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for candidate in candidates:
            hidden = any(
                part.startswith(".") or part == "__pycache__" for part in candidate.parts[-3:]
            )
            if candidate.is_file() and candidate.suffix in checked_file_suffixes and not hidden:
                files.append(candidate)
    return files


def check_file_name(path: pathlib.Path, problems: report) -> None:
    if path.name in fixed_file_names:
        return
    for problem in name_problems(path.stem):
        problems.add(path, 1, f"file name {path.name!r}: {problem}")


def standard_name_keys() -> list[str]:
    if not standard_names_file.exists():
        return []
    with standard_names_file.open("rb") as file:
        return list(tomllib.load(file))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="files or directories, by default the sources")
    arguments = parser.parse_args()

    problems = report()
    files = source_files(arguments.paths)
    declared = []
    comments_by_file = []
    for path in files:
        check_file_name(path, problems)
        if path.suffix in python_suffixes:
            declarations, comments = read_python(path, problems)
        elif path.suffix in cpp_suffixes:
            declarations, comments = read_cpp(path, problems)
        else:
            continue
        declared.extend((path, line, name, what) for line, name, what in declarations)
        comments_by_file.append((path, comments))

    for name in standard_name_keys():
        problems.check_name(standard_names_file, 1, name, "standard name")

    for path, line, name, what in declared:
        if what == "argument of a pytest hook":
            fixed_names = pytest_hook_argument_names
        elif what == "argument of a pytest test or fixture":
            fixed_names = pytest_fixture_names
        elif path.suffix in cpp_suffixes:
            fixed_names = fixed_cpp_names
        else:
            fixed_names = fixed_python_names
        problems.check_name(path, line, name, what, fixed_names)

    for path, comments in comments_by_file:
        check_comment_blocks(path, comments, problems)

    problem_count = problems.print_all()
    if problem_count:
        print(f"{problem_count} problems in {len(files)} files", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
