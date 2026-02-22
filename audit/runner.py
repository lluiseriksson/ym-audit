from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, List

@dataclass
class EqTest:
    id: str
    kind: str
    deps: List[str]
    statement: str
    fn: Callable[[], Any]

REGISTRY: Dict[str, EqTest] = {}
