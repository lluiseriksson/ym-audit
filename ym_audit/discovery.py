from __future__ import annotations

import inspect
from types import ModuleType
from typing import Callable, List, Tuple

def test_callables_in_module(m: ModuleType) -> List[Tuple[str, Callable[[], None]]]:
    out = []
    for name in dir(m):
        if not name.startswith("test_"):
            continue
        obj = getattr(m, name)
        if callable(obj) and inspect.isfunction(obj):
            out.append((name, obj))
    return sorted(out, key=lambda x: x[0])

def run_all_test_callables_in_module(m: ModuleType) -> None:
    tests = test_callables_in_module(m)
    if not tests:
        raise RuntimeError(f"No test_ functions found in module {m.__name__}")
    for name, fn in tests:
        fn()
