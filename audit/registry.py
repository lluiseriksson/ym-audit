"""
Equation/Test registry for the YM audit suite.
"""

EQUATION_REGISTRY = {}


def register(test_id, test_func, kind="exact", description="", overwrite=False):
    """Register a test function in the global registry."""
    if test_id in EQUATION_REGISTRY and not overwrite:
        raise ValueError(f"Test {test_id} already registered. Use overwrite=True.")
    EQUATION_REGISTRY[test_id] = {
        "func": test_func,
        "kind": kind,
        "description": description,
    }


def clear():
    """Clear all registered tests."""
    EQUATION_REGISTRY.clear()
