# pmi/logger.py
from __future__ import annotations
import sys
import json
from pathlib import Path
from dataclasses import asdict, is_dataclass
import datetime as dt
from datetime import timezone
import enum

# numpy / pandas son opcionales: solo si están instalados
try:
    import numpy as np
except Exception:
    np = None  # type: ignore
try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore


def _to_serializable(obj):
    """Convierte cualquier objeto común (dataclass, Enum, numpy/pandas, datetime) a JSON serializable."""
    # None / primitivos
    if obj is None or isinstance(obj, (str, bool, int, float)):
        # sanea floats no válidos
        if isinstance(obj, float):
            if obj != obj or obj in (float('inf'), float('-inf')):
                return None
        return obj

    # dataclass -> dict
    if is_dataclass(obj):
        obj = asdict(obj)

    # Enums
    if isinstance(obj, enum.Enum):
        # preferimos el nombre; si no, el valor
        try:
            return obj.name
        except Exception:
            return obj.value

    # dict
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}

    # list/tuple/set
    if isinstance(obj, (list, tuple, set)):
        return [_to_serializable(x) for x in obj]

    # datetime / date
    if isinstance(obj, dt.datetime):
        if obj.tzinfo is None:
            obj = obj.replace(tzinfo=timezone.utc)
        return obj.isoformat()
    if isinstance(obj, dt.date):
        return obj.isoformat()

    # numpy
    if np is not None:
        if isinstance(obj, np.generic):
            try:
                return obj.item()
            except Exception:
                return float(obj) if hasattr(obj, "__float__") else str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

    # pandas
    if pd is not None:
        if isinstance(obj, pd.Timestamp):
            try:
                # asegurar UTC
                if obj.tz is None:
                    obj = obj.tz_localize("UTC")
                else:
                    obj = obj.tz_convert("UTC")
                return obj.to_pydatetime().isoformat()
            except Exception:
                return obj.isoformat()
        if isinstance(obj, (pd.Series, pd.Index)):
            return obj.tolist()

    # fallback
    try:
        return str(obj)
    except Exception:
        return None


def log_pmi_decision(decision, path: str = "logs/pmi_decisions.jsonl") -> bool:
    """
    Registra 1 línea JSON por decisión del PMI.
    Acepta dataclass, dict u objetos con __dict__.
    Devuelve True si escribe, False si falla.
    """
    try:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Normalizamos el payload
        if is_dataclass(decision):
            payload = asdict(decision)
        elif isinstance(decision, dict):
            payload = dict(decision)
        else:
            # mejor esfuerzo: __dict__ o cast a str
            try:
                payload = dict(decision)
            except Exception:
                payload = vars(decision).copy() if hasattr(decision, "__dict__") else {"value": str(decision)}

        # Timestamp ISO-UTC
        payload["_ts_utc"] = dt.datetime.now(timezone.utc).isoformat()

        # Serialización robusta
        serializable = _to_serializable(payload)

        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(serializable, ensure_ascii=False) + "\n")
        return True

    except Exception as e:
        # Log al stderr con detalle, y un intento mínimo de escritura
        print(f"[PMI logger] ERROR al guardar '{path}': {e}", file=sys.stderr)
        try:
            minimal = {"_ts_utc": dt.datetime.now(timezone.utc).isoformat(), "error": str(e)}
            with Path(path).open("a", encoding="utf-8") as f:
                f.write(json.dumps(minimal, ensure_ascii=False) + "\n")
            return True
        except Exception:
            return False
