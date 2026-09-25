# ==============================================================================
# tests/test_import_discipline.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# Static source-discipline tests; project and numerical modules are not imported.
# ==============================================================================

"""Lock native-package imports and the live OpenMP guard in source form.

Dynamic import arguments are intentionally checked only when they are string
literals. Loader aliases created through assignment are also intentionally not
followed because doing so would require data-flow analysis and would flag the
legitimate monkeypatch pattern in test_hmm_contract.py.
"""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()
EXCLUDED_DIRS = {
    "__pycache__",
    "data",
    "log",
    ".git",
    ".pytest_cache",
    "build",
    "dist",
    "__MACOSX",
}
NATIVE_IMPORT_ALLOWLIST = {
    "xgboost": {Path("XGBoost.py")},
    "catboost": {Path("meta_model.py")},
}


def _python_sources() -> list[tuple[Path, Path]]:
    sources = []
    for path in PROJECT_ROOT.rglob("*.py"):
        resolved = path.resolve()
        relative = resolved.relative_to(PROJECT_ROOT)
        if resolved == THIS_FILE or path.name.startswith("._"):
            continue
        if any(
            part in EXCLUDED_DIRS or part.startswith(".venv")
            for part in relative.parts[:-1]
        ):
            continue
        sources.append((resolved, relative))
    return sorted(sources, key=lambda item: item[1].as_posix())


def _parse_source(path: Path, relative: Path) -> ast.Module:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise AssertionError(f"cannot read {relative}: {exc}") from exc
    try:
        return ast.parse(source, filename=relative.as_posix())
    except SyntaxError as exc:
        raise AssertionError(f"cannot parse {relative}: {exc}") from exc


def _bound_loader_names(tree: ast.Module) -> set[str]:
    names = {"__import__"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module_root = (node.module or "").split(".", 1)[0]
        for alias in node.names:
            if module_root == "importlib" and alias.name == "import_module":
                names.add(alias.asname or alias.name)
            elif module_root == "builtins" and alias.name == "__import__":
                names.add(alias.asname or alias.name)
    return names


def _literal_dynamic_load(node: ast.Call, bound_names: set[str]) -> str | None:
    if not node.args:
        return None
    argument = node.args[0]
    if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
        return None

    function = node.func
    if isinstance(function, ast.Attribute):
        is_loader = function.attr in {"import_module", "__import__"}
    else:
        is_loader = isinstance(function, ast.Name) and function.id in bound_names
    return argument.value if is_loader else None


def _scan_native_loads() -> list[tuple[str, Path, int, str]]:
    matches = []
    for path, relative in _python_sources():
        tree = _parse_source(path, relative)
        bound_names = _bound_loader_names(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                module_names = [alias.name for alias in node.names]
                kind = "import"
            elif isinstance(node, ast.ImportFrom):
                module_names = [node.module or ""]
                kind = "from-import"
            elif isinstance(node, ast.Call):
                dynamic_name = _literal_dynamic_load(node, bound_names)
                module_names = [dynamic_name] if dynamic_name is not None else []
                kind = "dynamic-import"
            else:
                continue

            for module_name in module_names:
                package = module_name.split(".", 1)[0]
                if package in NATIVE_IMPORT_ALLOWLIST:
                    matches.append((package, relative, node.lineno, kind))
    return matches


def _is_os_environ(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _is_omp_subscript(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and _is_os_environ(node.value)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "OMP_NUM_THREADS"
    )


def _is_omp_setdefault(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "setdefault"
        and _is_os_environ(node.func.value)
        and bool(node.args)
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "OMP_NUM_THREADS"
    )


def test_native_package_loads_are_confined_to_allowlisted_modules():
    matches = _scan_native_loads()
    violations = [
        match
        for match in matches
        if match[1] not in NATIVE_IMPORT_ALLOWLIST[match[0]]
    ]
    details = "\n".join(
        f"{relative}:{line}: {package} via {kind}"
        for package, relative, line, kind in violations
    )
    assert not violations, f"native package loads outside allowlist:\n{details}"

    for package, allowed_paths in NATIVE_IMPORT_ALLOWLIST.items():
        assert any(
            found_package == package and relative in allowed_paths
            for found_package, relative, _line, _kind in matches
        ), f"scan broken for {package}"


def test_main_omp_guard_is_unique_setdefault_before_native_imports():
    main_path = PROJECT_ROOT / "main.py"
    tree = _parse_source(main_path, Path("main.py"))

    setters: list[tuple[str, ast.AST]] = []
    for node in ast.walk(tree):
        if _is_omp_setdefault(node):
            setters.append(("setdefault", node))
        elif isinstance(node, ast.Assign) and any(
            _is_omp_subscript(target) for target in node.targets
        ):
            setters.append(("assignment", node))
        elif isinstance(node, ast.AugAssign) and _is_omp_subscript(node.target):
            setters.append(("augmented-assignment", node))

    assert len(setters) == 1, f"expected one OMP setter in main.py, found {setters}"
    kind, setter = setters[0]
    assert kind == "setdefault"
    assert isinstance(setter, ast.Call)
    assert len(setter.args) == 2
    assert not setter.keywords
    assert all(isinstance(argument, ast.Constant) for argument in setter.args)
    assert [argument.value for argument in setter.args] == ["OMP_NUM_THREADS", "1"]

    later_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if (
            isinstance(node, ast.Import)
            and len(node.names) == 1
            and node.names[0].name == "os"
        ):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            later_imports.append(node)

    assert later_imports
    assert all(setter.lineno < node.lineno for node in later_imports)
