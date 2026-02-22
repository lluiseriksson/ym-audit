from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

@dataclass(frozen=True)
class AuditTest:
    id: str
    kind: str
    statement: str
    acceptance: str = ""
    deps: List[str] = field(default_factory=list)
    timeout_s: float = 60.0
    seed: Optional[int] = 0
    fn: Callable[[], Any] = lambda: None

REGISTRY: Dict[str, AuditTest] = {}

def register(t: AuditTest) -> None:
    if not t.id or not isinstance(t.id, str):
        raise ValueError("Test id must be a non-empty string")
    if t.id in REGISTRY:
        raise KeyError(f"Duplicate test id: {t.id}")
    REGISTRY[t.id] = t

# Backwards-compat alias
EqTest = AuditTest
