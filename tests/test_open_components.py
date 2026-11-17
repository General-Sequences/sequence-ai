# SPDX-FileCopyrightText: Copyright (c) 2026 General Sequences Company
# SPDX-License-Identifier: Apache-2.0
"""
Every public name of the package documents what it is and what it takes: a docstring on each module, class and
function, and a type on each parameter and return of a function. A component someone adds to their own code is read
before it is run.
"""
import importlib
import inspect
import pkgutil

import pytest

import sequence_ai.engine
import sequence_ai.network

MODULES = sorted(f"{package.__name__}.{m.name}" for package in (sequence_ai.engine, sequence_ai.network)
                 for m in pkgutil.iter_modules(package.__path__) if not m.name.startswith("_"))


def _public(module):
    for name, thing in vars(module).items():
        if name.startswith("_") or getattr(thing, "__module__", None) != module.__name__:
            continue
        if inspect.isfunction(thing):
            yield name, thing
        elif inspect.isclass(thing):
            yield name, thing
            for method_name, method in vars(thing).items():
                if not method_name.startswith("_") and inspect.isfunction(method):
                    yield f"{name}.{method_name}", method


@pytest.mark.parametrize("name", MODULES)
def test_every_public_name_has_a_docstring_and_every_function_its_types(name):
    try:
        module = importlib.import_module(name)
    except ImportError as missing:                       # a component whose own dependency is not installed here
        pytest.skip(str(missing))
    assert (module.__doc__ or "").strip(), f"{name}: the module has no docstring"
    undocumented, untyped = [], []
    for path, thing in _public(module):
        if not (thing.__doc__ or "").strip():
            # a method that only overrides one a base class documents adds nothing new
            if not any(path.split(".")[-1] in vars(base) for base in getattr(thing, "__mro__", ())[1:]):
                undocumented.append(path)
        if inspect.isfunction(thing):
            signature = inspect.signature(thing)
            for parameter in signature.parameters.values():
                star = parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
                if parameter.name in ("self", "cls") or star:
                    continue
                if parameter.annotation is parameter.empty:
                    untyped.append(f"{path}({parameter.name})")
            if signature.return_annotation is signature.empty:
                untyped.append(f"{path} -> ?")
    assert not undocumented, f"{name}: no docstring on {undocumented}"
    assert not untyped, f"{name}: no type on {untyped}"
