# pmi/logger.py
import os
import json
import logging
from dataclasses import asdict, is_dataclass

# Importaciones robustas para los enums y dataclasses
try:
    # Asumiendo que pmi es un paquete
    from .decision import PMIDecision
    from .enums import DecisionAction
except ImportError:
    # Fallback si la estructura es diferente o se ejecuta directamente
    try:
        from pmi.decision import PMIDecision
        from pmi.enums import DecisionAction
    except ImportError:
        # Si no se encuentran, usamos placeholders para evitar que el programa crashee
        # al importar, aunque el logging no funcionará completamente.
        PMIDecision = type("PMIDecision", (), {})
        DecisionAction = type("DecisionAction", (), {})

logger = logging.getLogger("bot")

def _json_serializer(obj):
    """Serializador JSON seguro que maneja enums y dataclasses."""
    if isinstance(obj, DecisionAction):
        return obj.name  # Convertir el enum a su nombre en string (ej: "CLOSE")
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def log_pmi_decision(decision: PMIDecision, path: str = "logs/pmi_decisions.jsonl") -> bool:
    logger.info(f"[DEBUG] Ejecutando log_pmi_decision para ticket={getattr(decision, 'ticket', None)} y acción={getattr(decision, 'action', None)}")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Convertir el dataclass a un diccionario serializable
        log_data = asdict(decision)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, default=_json_serializer, ensure_ascii=False) + "\n")
        logger.info(f"[DEBUG] Registro exitoso en {path} para ticket={getattr(decision, 'ticket', None)}")
        return True
    except Exception as e:
        logger.error(f"\u274c Error al guardar log de decisi\u00f3n PMI en '{path}': {e}")
        return False
        return True
    except Exception as e:
        logger.error(f"❌ Error al guardar log de decisión PMI en '{path}': {e}")
        return False