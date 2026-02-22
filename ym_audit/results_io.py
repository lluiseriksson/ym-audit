from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8")

def write_results_json(path: str, payload: Dict[str, Any]) -> str:
    b = canonical_json_bytes(payload)
    digest = sha256_bytes(b)
    with open(path, "wb") as f:
        f.write(b)
    return digest
