"""Comprehensive tests for pyweight.parser."""

from __future__ import annotations

import ast
import keyword as _keyword
from pathlib import Path
from textwrap import dedent

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pyweight.models import Confidence, ModuleInfo
from pyweight.parser import (
    _ImportVisitor,
    _PurityClassifier,
    _UsageVisitor,
    get_symbol_origin,
    parse_module,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, src: str) -> Path:
    p = tmp_path / "mod.py"
    p.write_text(dedent(src), encoding="utf-8")
    return p


def _parse(tmp_path: Path, src: str, name: str = "mod") -> ModuleInfo:
    p = _write(tmp_path, src)
    return parse_module(p, name)


# ---------------------------------------------------------------------------
# Basic import extraction
# ---------------------------------------------------------------------------


def test_module_level_import(tmp_path: Path) -> None:
    info = _parse(tmp_path, "import os\n")
    assert len(info.imports) == 1
    ref = info.imports[0]
    assert ref.module == "os"
    assert ref.scope == "module"
    assert ref.level == 0
    assert not ref.is_star
    assert not ref.is_type_checking
    assert not ref.is_conditional


def test_from_import(tmp_path: Path) -> None:
    info = _parse(tmp_path, "from pathlib import Path\n")
    assert len(info.imports) == 1
    ref = info.imports[0]
    assert ref.module == "pathlib"
    assert ref.names == ("Path",)
    assert ref.scope == "module"
    assert ref.level == 0


def test_relative_import_level(tmp_path: Path) -> None:
    info = _parse(tmp_path, "from .. import core\n")
    assert info.imports[0].level == 2
    assert info.imports[0].module == ""


def test_function_scoped_import(tmp_path: Path) -> None:
    src = """\
        def foo():
            import json
        """
    info = _parse(tmp_path, src)
    ref = info.imports[0]
    assert ref.scope == "foo"
    assert ref.module == "json"


def test_nested_function_scoped_import(tmp_path: Path) -> None:
    src = """\
        def outer():
            def inner():
                import re
        """
    info = _parse(tmp_path, src)
    ref = info.imports[0]
    assert ref.scope == "outer.inner"


def test_class_scoped_import(tmp_path: Path) -> None:
    src = """\
        class MyClass:
            import os
        """
    info = _parse(tmp_path, src)
    ref = info.imports[0]
    assert ref.scope == "MyClass"


def test_async_function_import(tmp_path: Path) -> None:
    src = """\
        async def handler():
            import asyncio
        """
    info = _parse(tmp_path, src)
    ref = info.imports[0]
    assert ref.scope == "handler"
    assert ref.module == "asyncio"


# ---------------------------------------------------------------------------
# TYPE_CHECKING detection
# ---------------------------------------------------------------------------


def test_type_checking_bare_name(tmp_path: Path) -> None:
    src = """\
        from __future__ import annotations
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from pathlib import Path
        """
    info = _parse(tmp_path, src)
    tc_refs = [r for r in info.imports if r.is_type_checking]
    assert len(tc_refs) == 1
    assert tc_refs[0].module == "pathlib"
    assert tc_refs[0].names == ("Path",)


def test_type_checking_typing_attribute(tmp_path: Path) -> None:
    src = """\
        import typing
        if typing.TYPE_CHECKING:
            from collections import OrderedDict
        """
    info = _parse(tmp_path, src)
    tc_refs = [r for r in info.imports if r.is_type_checking]
    assert len(tc_refs) == 1
    assert tc_refs[0].module == "collections"


def test_type_checking_does_not_bleed_to_other_imports(tmp_path: Path) -> None:
    src = """\
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            import abc
        import os
        """
    info = _parse(tmp_path, src)
    abc_refs = [r for r in info.imports if r.module == "abc"]
    os_refs = [r for r in info.imports if r.module == "os"]
    assert abc_refs[0].is_type_checking
    assert not os_refs[0].is_type_checking


def test_type_checking_orelse_is_conditional(tmp_path: Path) -> None:
    src = """\
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from pathlib import Path
        else:
            from os.path import abspath
        """
    info = _parse(tmp_path, src)
    tc_refs = [r for r in info.imports if r.is_type_checking]
    cond_refs = [r for r in info.imports if r.is_conditional and not r.is_type_checking]
    assert any(r.module == "pathlib" for r in tc_refs)
    assert any(r.module == "os.path" for r in cond_refs)


# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------


def test_conditional_try_except(tmp_path: Path) -> None:
    src = """\
        try:
            import tomllib
        except ImportError:
            tomllib = None  # type: ignore[assignment]
        """
    info = _parse(tmp_path, src)
    refs = [r for r in info.imports if r.module == "tomllib"]
    assert len(refs) == 1
    assert refs[0].is_conditional
    assert not refs[0].is_type_checking


def test_conditional_if_block(tmp_path: Path) -> None:
    src = """\
        import sys
        if sys.version_info >= (3, 11):
            import tomllib
        """
    info = _parse(tmp_path, src)
    tomllib_refs = [r for r in info.imports if r.module == "tomllib"]
    assert tomllib_refs[0].is_conditional


def test_non_conditional_import_not_marked(tmp_path: Path) -> None:
    info = _parse(tmp_path, "import os\n")
    assert not info.imports[0].is_conditional


# ---------------------------------------------------------------------------
# Star imports
# ---------------------------------------------------------------------------


def test_star_import(tmp_path: Path) -> None:
    info = _parse(tmp_path, "from os.path import *\n")
    assert len(info.imports) == 1
    ref = info.imports[0]
    assert ref.is_star
    assert ref.names == ("*",)
    assert ref.module == "os.path"
    assert info.star_imports == ("os.path",)


def test_relative_star_import(tmp_path: Path) -> None:
    info = _parse(tmp_path, "from . import *\n")
    star = [r for r in info.imports if r.is_star]
    assert len(star) == 1
    assert star[0].level == 1


# ---------------------------------------------------------------------------
# __getattr__ detection
# ---------------------------------------------------------------------------


def test_getattr_detection(tmp_path: Path) -> None:
    src = """\
        def __getattr__(name: str) -> object:
            return None
        """
    info = _parse(tmp_path, src)
    assert info.has_getattr


def test_getattr_in_class_not_detected(tmp_path: Path) -> None:
    src = """\
        class Foo:
            def __getattr__(self, name: str) -> object:
                return None
        """
    info = _parse(tmp_path, src)
    assert not info.has_getattr


def test_no_getattr(tmp_path: Path) -> None:
    info = _parse(tmp_path, "x = 1\n")
    assert not info.has_getattr


# ---------------------------------------------------------------------------
# __all__ extraction
# ---------------------------------------------------------------------------


def test_all_from_list(tmp_path: Path) -> None:
    src = '__all__ = ["foo", "bar"]\n'
    info = _parse(tmp_path, src)
    assert info.has_all
    assert info.all_names == frozenset({"foo", "bar"})


def test_all_from_tuple(tmp_path: Path) -> None:
    src = '__all__ = ("foo", "bar")\n'
    info = _parse(tmp_path, src)
    assert info.has_all
    assert info.all_names == frozenset({"foo", "bar"})


def test_no_all(tmp_path: Path) -> None:
    info = _parse(tmp_path, "x = 1\n")
    assert not info.has_all
    assert info.all_names == frozenset()


# ---------------------------------------------------------------------------
# defined_names
# ---------------------------------------------------------------------------


def test_defined_names_functions_and_classes(tmp_path: Path) -> None:
    src = """\
        class Foo: pass
        def bar(): pass
        async def baz(): pass
        """
    info = _parse(tmp_path, src)
    assert "Foo" in info.defined_names
    assert "bar" in info.defined_names
    assert "baz" in info.defined_names


def test_defined_names_simple_assignment(tmp_path: Path) -> None:
    src = "CONSTANT = 42\n"
    info = _parse(tmp_path, src)
    assert "CONSTANT" in info.defined_names


# ---------------------------------------------------------------------------
# is_init
# ---------------------------------------------------------------------------


def test_is_init_true(tmp_path: Path) -> None:
    p = tmp_path / "__init__.py"
    p.write_text("", encoding="utf-8")
    info = parse_module(p, "mypkg")
    assert info.is_init


def test_is_init_false(tmp_path: Path) -> None:
    p = tmp_path / "other.py"
    p.write_text("", encoding="utf-8")
    info = parse_module(p, "mypkg.other")
    assert not info.is_init


# ---------------------------------------------------------------------------
# Usage visitor
# ---------------------------------------------------------------------------


def test_usage_module_level(tmp_path: Path) -> None:
    src = """\
        import os
        x = os.path.join("a", "b")
        """
    info = _parse(tmp_path, src)
    assert "module" in info.usage_scopes.get("os", frozenset())


def test_usage_inside_function(tmp_path: Path) -> None:
    src = """\
        import json
        def process():
            return json.loads("{}")
        """
    info = _parse(tmp_path, src)
    assert "process" in info.usage_scopes.get("json", frozenset())
    assert "module" not in info.usage_scopes.get("json", frozenset())


def test_usage_multiple_scopes(tmp_path: Path) -> None:
    src = """\
        import os
        x = os.getcwd()
        def foo():
            return os.listdir(".")
        """
    info = _parse(tmp_path, src)
    scopes = info.usage_scopes.get("os", frozenset())
    assert "module" in scopes
    assert "foo" in scopes


def test_usage_scopes_are_frozensets(tmp_path: Path) -> None:
    src = """\
        import os
        x = os.getcwd()
        """
    info = _parse(tmp_path, src)
    for scopes in info.usage_scopes.values():
        assert isinstance(scopes, frozenset)


# ---------------------------------------------------------------------------
# Purity classifier
# ---------------------------------------------------------------------------


def test_purity_pure_module(tmp_path: Path) -> None:
    src = """\
        import os
        class Foo: pass
        def bar(): pass
        """
    info = _parse(tmp_path, src)
    assert info.purity == "pure"


def test_purity_side_effects_bare_call(tmp_path: Path) -> None:
    src = """\
        import sys
        print("loading", file=sys.stderr)
        """
    info = _parse(tmp_path, src)
    assert info.purity == "side_effects"


def test_purity_side_effects_custom_call(tmp_path: Path) -> None:
    src = """\
        def register(name: str) -> None:
            pass
        register("my_module")
        """
    info = _parse(tmp_path, src)
    assert info.purity == "side_effects"


def test_purity_only_top_level(tmp_path: Path) -> None:
    """Calls inside functions must not affect module purity."""
    src = """\
        def setup() -> None:
            print("initializing")
        """
    info = _parse(tmp_path, src)
    assert info.purity == "pure"


def test_purity_unknown_assignment_call(tmp_path: Path) -> None:
    """Module-level assignment whose RHS is a call is ambiguous — purity 'unknown'."""
    src = "result = some_factory()\n"
    info = _parse(tmp_path, src)
    assert info.purity == "unknown"


def test_purity_side_effects_beats_unknown(tmp_path: Path) -> None:
    """side_effects takes precedence over unknown when both are present."""
    src = """\
        result = some_factory()
        print("loaded")
        """
    info = _parse(tmp_path, src)
    assert info.purity == "side_effects"


# ---------------------------------------------------------------------------
# get_symbol_origin
# ---------------------------------------------------------------------------


def test_symbol_origin_explicit_high(tmp_path: Path) -> None:
    src = "from pathlib import Path\n"
    info = _parse(tmp_path, src)
    origin, confidence = get_symbol_origin(info, "Path")
    assert origin == "pathlib"
    assert confidence == Confidence.HIGH


def test_symbol_origin_missing_low(tmp_path: Path) -> None:
    src = "import os\n"
    info = _parse(tmp_path, src)
    origin, confidence = get_symbol_origin(info, "NonExistent")
    assert origin is None
    assert confidence == Confidence.LOW


def test_symbol_origin_star_low(tmp_path: Path) -> None:
    src = "from os.path import *\n"
    info = _parse(tmp_path, src)
    _origin, confidence = get_symbol_origin(info, "join")
    assert confidence == Confidence.LOW


def test_symbol_origin_star_medium_with_all(tmp_path: Path) -> None:
    src = "from utils import *\n"
    info = _parse(tmp_path, src)
    all_names_by_module: dict[str, tuple[str, ...]] = {"utils": ("helper", "join")}
    origin, confidence = get_symbol_origin(info, "helper", all_names_by_module)
    assert origin == "utils"
    assert confidence == Confidence.MEDIUM


def test_symbol_origin_star_low_when_not_in_all(tmp_path: Path) -> None:
    src = "from utils import *\n"
    info = _parse(tmp_path, src)
    all_names_by_module: dict[str, tuple[str, ...]] = {"utils": ("helper",)}
    _origin, confidence = get_symbol_origin(info, "missing", all_names_by_module)
    assert confidence == Confidence.LOW


def test_symbol_origin_two_star_imports_symbol_in_second(tmp_path: Path) -> None:
    """Symbol only in the second star-import source must return MEDIUM, not LOW."""
    src = "from alpha import *\nfrom beta import *\n"
    info = _parse(tmp_path, src)
    # symbol lives only in beta's __all__
    all_names_by_module: dict[str, tuple[str, ...]] = {
        "alpha": ("other_name",),
        "beta": ("target_symbol",),
    }
    origin, confidence = get_symbol_origin(info, "target_symbol", all_names_by_module)
    assert origin == "beta"
    assert confidence == Confidence.MEDIUM


# ---------------------------------------------------------------------------
# Cache parameter (stub — Cache type not yet implemented)
# ---------------------------------------------------------------------------


def test_parse_module_no_cache(tmp_path: Path) -> None:
    """parse_module(path, name, cache=None) should always parse from disk."""
    src = "import os\n"
    p = _write(tmp_path, src)
    info = parse_module(p, "mod", cache=None)
    assert info.qualified_name == "mod"
    assert len(info.imports) == 1


def test_parse_module_unicode_error(tmp_path: Path) -> None:
    p = tmp_path / "bad.py"
    p.write_bytes(b"\xff\xfe invalid utf-8 \x80\x81")
    info = parse_module(p, "bad")
    assert info.qualified_name == "bad"
    assert info.imports == ()
    assert info.purity == "unknown"


def test_parse_module_syntax_error(tmp_path: Path) -> None:
    p = tmp_path / "broken.py"
    p.write_text("def foo(\n    bar\n  baz\n", encoding="utf-8")
    info = parse_module(p, "broken")
    assert info.qualified_name == "broken"
    assert info.imports == ()
    assert info.purity == "unknown"


# ---------------------------------------------------------------------------
# Multiple imports in one from-import statement
# ---------------------------------------------------------------------------


def test_multiple_names_from_import(tmp_path: Path) -> None:
    src = "from os.path import join, exists, dirname\n"
    info = _parse(tmp_path, src)
    assert len(info.imports) == 3
    names = {r.names[0] for r in info.imports}
    assert names == {"join", "exists", "dirname"}
    assert all(r.module == "os.path" for r in info.imports)


# ---------------------------------------------------------------------------
# Alias handling
# ---------------------------------------------------------------------------


def test_import_alias(tmp_path: Path) -> None:
    src = "import numpy as np\n"
    info = _parse(tmp_path, src)
    assert info.imports[0].alias == "np"


def test_from_import_alias(tmp_path: Path) -> None:
    src = "from collections import OrderedDict as OD\n"
    info = _parse(tmp_path, src)
    assert info.imports[0].alias == "OD"
    assert info.imports[0].names == ("OrderedDict",)


# ---------------------------------------------------------------------------
# lineno
# ---------------------------------------------------------------------------


def test_lineno_positive(tmp_path: Path) -> None:
    src = """\
        import os
        import sys
        """
    info = _parse(tmp_path, src)
    for ref in info.imports:
        assert ref.lineno > 0


# ---------------------------------------------------------------------------
# Tests against real sample_package fixtures
# ---------------------------------------------------------------------------

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "sample_package"


def test_sample_core_purity() -> None:
    path = FIXTURE_ROOT / "core.py"
    info = parse_module(path, "sample_package.core")
    assert info.purity == "pure"


def test_sample_core_defined_names() -> None:
    path = FIXTURE_ROOT / "core.py"
    info = parse_module(path, "sample_package.core")
    assert "CoreClass" in info.defined_names
    assert "helper_fn" in info.defined_names


def test_sample_core_all() -> None:
    path = FIXTURE_ROOT / "core.py"
    info = parse_module(path, "sample_package.core")
    assert info.has_all
    assert "CoreClass" in info.all_names
    assert "helper_fn" in info.all_names


def test_sample_guarded_type_checking() -> None:
    path = FIXTURE_ROOT / "guarded.py"
    info = parse_module(path, "sample_package.guarded")
    tc = [r for r in info.imports if r.is_type_checking]
    assert any(r.module == "heavy" or r.names == ("HeavyProcessor",) for r in tc)


def test_sample_guarded_function_import() -> None:
    path = FIXTURE_ROOT / "guarded.py"
    info = parse_module(path, "sample_package.guarded")
    fn_imports = [r for r in info.imports if r.scope == "create_processor"]
    assert any(r.names == ("HeavyProcessor",) for r in fn_imports)


def test_sample_conditional_try() -> None:
    path = FIXTURE_ROOT / "conditional.py"
    info = parse_module(path, "sample_package.conditional")
    cond = [r for r in info.imports if r.is_conditional]
    assert any(r.module == "tomllib" for r in cond)


def test_sample_side_effects_purity() -> None:
    path = FIXTURE_ROOT / "side_effects.py"
    info = parse_module(path, "sample_package.side_effects")
    assert info.purity == "side_effects"


def test_sample_init_barrel() -> None:
    path = FIXTURE_ROOT / "__init__.py"
    info = parse_module(path, "sample_package")
    assert info.is_init
    assert info.has_all
    assert "CoreClass" in info.all_names
    # star import from .core
    assert len(info.star_imports) >= 1


def test_sample_init_star_import() -> None:
    path = FIXTURE_ROOT / "__init__.py"
    info = parse_module(path, "sample_package")
    star = [r for r in info.imports if r.is_star]
    assert len(star) >= 1
    assert all(r.level > 0 for r in star)  # relative


def test_sample_heavy_not_pure() -> None:
    path = FIXTURE_ROOT / "heavy.py"
    info = parse_module(path, "sample_package.heavy")
    # No bare calls at module level — should be pure
    assert info.purity == "pure"


def test_sample_deferred_candidate_defined_names() -> None:
    path = FIXTURE_ROOT / "deferred_candidate.py"
    info = parse_module(path, "sample_package.deferred_candidate")
    assert "CONSTANT" in info.defined_names
    assert "load_data" in info.defined_names
    assert "get_constant" in info.defined_names


def test_sample_deferred_candidate_json_used_in_function() -> None:
    path = FIXTURE_ROOT / "deferred_candidate.py"
    info = parse_module(path, "sample_package.deferred_candidate")
    scopes = info.usage_scopes.get("json", frozenset())
    assert "load_data" in scopes
    assert "module" not in scopes


def test_sample_models_all() -> None:
    path = FIXTURE_ROOT / "models.py"
    info = parse_module(path, "sample_package.models")
    assert info.has_all
    assert "SomeModel" in info.all_names


def test_sample_hub_imports() -> None:
    path = FIXTURE_ROOT / "hub.py"
    info = parse_module(path, "sample_package.hub")
    modules = {r.module for r in info.imports}
    assert "conditional" in modules or any("conditional" in m for m in modules)


def test_sample_utils_helpers_relative() -> None:
    path = FIXTURE_ROOT / "utils" / "helpers.py"
    info = parse_module(path, "sample_package.utils.helpers")
    relative = [r for r in info.imports if r.level > 0]
    assert len(relative) >= 1


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------


def _parse_source(source: str, tmp_path: Path) -> ModuleInfo:
    p = tmp_path / "hyp_mod.py"
    p.write_text(source, encoding="utf-8")
    return parse_module(p, "hyp_mod")


def test_hypothesis_empty_source(tmp_path: Path) -> None:
    """Empty file produces a valid ModuleInfo with no imports."""
    info = _parse_source("", tmp_path)
    assert isinstance(info.imports, tuple)
    assert len(info.imports) == 0


@pytest.fixture
def simple_module_info(tmp_path: Path) -> ModuleInfo:
    src = """\
        import os
        from pathlib import Path
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            import abc
        try:
            import tomllib
        except ImportError:
            pass
        from . import *
        """
    p = tmp_path / "hyp_mod.py"
    p.write_text(dedent(src), encoding="utf-8")
    return parse_module(p, "hyp_mod")


def test_every_import_ref_valid_lineno(simple_module_info: ModuleInfo) -> None:
    for ref in simple_module_info.imports:
        assert ref.lineno > 0, f"lineno must be > 0: {ref}"


def test_every_import_ref_valid_level(simple_module_info: ModuleInfo) -> None:
    for ref in simple_module_info.imports:
        assert ref.level >= 0, f"level must be >= 0: {ref}"


def test_every_import_ref_has_scope(simple_module_info: ModuleInfo) -> None:
    for ref in simple_module_info.imports:
        assert isinstance(ref.scope, str), f"scope must be str: {ref}"
        assert len(ref.scope) > 0, f"scope must be non-empty: {ref}"


def test_every_import_ref_module_is_str(simple_module_info: ModuleInfo) -> None:
    for ref in simple_module_info.imports:
        assert isinstance(ref.module, str), f"module must be str: {ref}"


_IDENTIFIER_STRATEGY = st.text(
    alphabet=st.characters(categories=("Ll", "Lu", "Nd"), include_characters="_"),
    min_size=1,
    max_size=16,
).filter(
    lambda s: (
        s.isidentifier()
        and not s[0].isdigit()
        and not _keyword.iskeyword(s)
        and not _keyword.issoftkeyword(s)
    )
)


@given(_IDENTIFIER_STRATEGY)
@settings(max_examples=20)
def test_hypothesis_import_names_tuple(name: str) -> None:
    """ImportRef.names is always a tuple."""
    import tempfile

    src = f"from os import {name}\n"
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(src)
        tmp = Path(f.name)
    try:
        info = parse_module(tmp, "hyp_mod")
        for ref in info.imports:
            assert isinstance(ref.names, tuple)
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# _ImportVisitor unit tests (internal)
# ---------------------------------------------------------------------------


def test_import_visitor_directly() -> None:
    src = "import os\nfrom sys import argv\n"
    tree = ast.parse(src)
    v = _ImportVisitor()
    v.visit(tree)
    assert len(v.imports) == 2
    assert v.imports[0].module == "os"
    assert v.imports[1].module == "sys"


def test_usage_visitor_attribute_chain() -> None:
    src = "x = foo.bar.baz\n"
    tree = ast.parse(src)
    v = _UsageVisitor()
    v.visit(tree)
    # Only "foo" should be recorded, not "bar" or "baz"
    assert "foo" in v.usage
    assert "bar" not in v.usage
    assert "baz" not in v.usage


def test_usage_attribute_chain_with_subscript_args(tmp_path: Path) -> None:
    """Names in subscript/call args inside attribute chains are tracked."""
    src = "import foo, bar, baz\nresult = foo.data[bar].method(baz)\n"
    info = _parse(tmp_path, src)
    scopes = info.usage_scopes
    assert "module" in scopes.get("foo", frozenset())
    assert "module" in scopes.get("bar", frozenset())
    assert "module" in scopes.get("baz", frozenset())


def test_purity_classifier_pure() -> None:
    src = "import os\nclass Foo: pass\n"
    tree = ast.parse(src)
    c = _PurityClassifier()
    assert c.classify(tree) == "pure"


def test_purity_classifier_side_effects() -> None:
    src = "print('hello')\n"
    tree = ast.parse(src)
    c = _PurityClassifier()
    assert c.classify(tree) == "side_effects"
