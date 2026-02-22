"""
Compatibility shim: old imports `audit.runner` keep working.
Prefer importing from `ym_audit.*`.
"""

from ym_audit.registry import REGISTRY, AuditTest, register

# Backwards-compat alias used by existing tests
EqTest = AuditTest
