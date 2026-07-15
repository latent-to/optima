"""Unit tests for the DSL tracing-JIT compile allowlist (optima/dsl_jit_policy.py).

Stdlib-only module (ast) — these tests exercise the fail-closed alias analysis
and the call-shape predicate directly, independent of engine_tree wiring.
"""

from __future__ import annotations

import ast

import pytest

from optima import dsl_jit_policy as pol


def _receivers(src: str, *, local=None):
    tree = ast.parse(src)
    resolver = None
    if local is not None:
        local_set = set(local)
        resolver = lambda parts: ".".join(parts) in local_set  # noqa: E731
    return tree, pol.admitted_receivers(tree, module_resolves_locally=resolver)


def _compile_calls(tree):
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in pol.ADMITTED_ATTRS
    ]


def _admits(src, *, local=None) -> bool:
    tree, recv = _receivers(src, local=local)
    calls = _compile_calls(tree)
    assert calls, "test source has no <x>.compile(...) call"
    return all(pol.is_admitted_call(c, recv) for c in calls)


# ---- admitted spellings -------------------------------------------------

def test_import_as_alias_is_admitted():
    assert _admits("import cutlass.cute as cute\nx = cute.compile(fn, a, b)\n")


def test_from_import_is_admitted():
    assert _admits("from cutlass import cute\nx = cute.compile(fn)\n")


def test_from_import_renamed_is_admitted():
    assert _admits("from cutlass import cute as cc\nx = cc.compile(fn)\n")


def test_multiple_positional_args_admitted():
    assert _admits("import cutlass.cute as cute\nx = cute.compile(fn, a, b, c)\n")


# ---- fail-closed: receiver rebinding / shadowing ------------------------

@pytest.mark.parametrize(
    "src",
    [
        "import cutlass.cute as cute\ncute = cute\nx = cute.compile(fn)\n",
        "import cutlass.cute as cute\ndef f(cute):\n    return cute.compile(fn)\n",
        "import cutlass.cute as cute\nimport types as cute\nx = cute.compile(fn)\n",
        "import cutlass.cute as cute\nfor cute in xs:\n    cute.compile(fn)\n",
        "import cutlass.cute as cute\nwith ctx() as cute:\n    cute.compile(fn)\n",
        "import cutlass.cute as cute\ntry:\n    pass\nexcept E as cute:\n    cute.compile(fn)\n",
        "import cutlass.cute as cute\nclass cute:\n    pass\nx = cute.compile(fn)\n",
        "import cutlass.cute as cute\n(cute := other)\nx = cute.compile(fn)\n",
    ],
)
def test_rebinding_anywhere_withdraws(src):
    assert not _admits(src)


# ---- fail-closed: wrong module / wrong receiver -------------------------

@pytest.mark.parametrize(
    "src",
    [
        "import types as cute\nx = cute.compile(fn)\n",           # not a table module
        "def f():\n    gemm = object()\n    return gemm.compile(q)\n",  # arbitrary obj
        "import cutlass\nx = cutlass.cute.compile(fn)\n",         # dotted chain, not a Name
    ],
)
def test_non_table_receiver_not_admitted(src):
    tree, recv = _receivers(src)
    calls = _compile_calls(tree)
    assert calls
    assert not any(pol.is_admitted_call(c, recv) for c in calls)


# ---- fail-closed: vendored-local module withdraws -----------------------

def test_local_module_shadow_withdraws():
    src = "import cutlass.cute as cute\nx = cute.compile(fn)\n"
    # cutlass resolves inside the bundle tree -> withdraw
    assert not _admits(src, local=["cutlass"])
    assert not _admits(src, local=["cutlass.cute"])
    # unrelated local module does not withdraw
    assert _admits(src, local=["kernels.helper"])


# ---- fail-closed: argument-shape guard ----------------------------------

@pytest.mark.parametrize(
    "arg",
    ['"import os"', "b'code'", '"""x = 1"""'],
)
def test_string_first_arg_not_admitted(arg):
    src = f"import cutlass.cute as cute\nx = cute.compile({arg})\n"
    assert not _admits(src)


def test_string_keyword_arg_not_admitted():
    assert not _admits('import cutlass.cute as cute\nx = cute.compile(src="import os")\n')


def test_starred_first_arg_not_admitted():
    # *args hides the real first positional -> cannot verify it is not source
    # text -> fail closed.
    assert not _admits("import cutlass.cute as cute\nx = cute.compile(*args)\n")


def test_no_args_not_admitted():
    # no inspectable first positional -> fail closed
    assert not _admits("import cutlass.cute as cute\nx = cute.compile()\n")


# ---- fail-closed: namespace reflection poisons the whole file -----------

@pytest.mark.parametrize(
    "src",
    [
        # wildcard import can inject any name, incl. the receiver, at runtime
        "import cutlass.cute as cute\nfrom helper import *\nx = cute.compile(fn)\n",
        # sys.modules reflection can rebind the module global
        "import cutlass.cute as cute\nimport sys\n"
        "sys.modules[__name__].cute = evil\nx = cute.compile(fn)\n",
        "import cutlass.cute as cute\nimport sys as s\n"
        "s.modules[__name__].cute = evil\nx = cute.compile(fn)\n",
        # setattr / delattr reflective rebind
        "import cutlass.cute as cute\nsetattr(m, 'cute', evil)\nx = cute.compile(fn)\n",
        "import cutlass.cute as cute\nobj.__setattr__('cute', evil)\nx = cute.compile(fn)\n",
        "import cutlass.cute as cute\nd = m.__dict__\nx = cute.compile(fn)\n",
        "import cutlass.cute as cute\ng = globals()\nx = cute.compile(fn)\n",
    ],
)
def test_namespace_reflection_withdraws(src):
    assert not _admits(src)


def test_unrelated_modules_method_is_not_sys_modules():
    # model.modules() (torch) must NOT be mistaken for sys.modules reflection
    assert _admits("import cutlass.cute as cute\nfor m in model.modules():\n    pass\nx = cute.compile(fn)\n")


# ---- fail-closed: PEP 695 type-parameter shadowing ----------------------

def test_pep695_type_param_shadow_withdraws():
    # 'def f[cute]()' introduces a type-param binder named cute -> >1 binding
    src = "import cutlass.cute as cute\ndef f[cute](x):\n    return x\ny = cute.compile(fn)\n"
    import sys as _sys
    if _sys.version_info < (3, 12):
        pytest.skip("PEP 695 syntax requires Python 3.12+")
    assert not _admits(src)


# ---- table integrity ----------------------------------------------------

def test_table_is_absolute_and_nonempty():
    assert pol.DSL_JIT_ENTRYPOINTS
    for e in pol.DSL_JIT_ENTRYPOINTS:
        assert e.module and "." in e.module  # fully-qualified
        assert not e.module.startswith(".")   # absolute
        assert e.attr
