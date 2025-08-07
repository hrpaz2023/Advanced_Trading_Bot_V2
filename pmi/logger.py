# pmi/logger.py
"""Pequeño helper para registrar decisiones PMI en un JSONL."""

from __future__ import annotations
import os, json, datetime as dt
from pathlib import Path
from typing import Dict, Any

def log_pmi_decision(data: Dict[str, Any], path: str = "logs/pmi_decisions.jsonl") -> bool:
    """
    Appendea una línea JSON (UTF-8) con la decisión del PMI.
    Retorna True si logró guardar, False si hubo error.
    """
    try:
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        safe = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                for k, v in data.items()}
        safe.setdefault("timestamp_utc", dt.datetime.utcnow().isoformat())
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False
