# ------------------------------------------------------------
# File: pmi/enums.py
# ------------------------------------------------------------
"""Enumeraciones y tipos estáticos utilizados por PMI."""

from enum import Enum, auto


class DecisionAction(Enum):
    """Acciones posibles que el PMI puede retornar por posición."""

    CLOSE = auto()
    PARTIAL_CLOSE = auto()
    HOLD = auto()
    TIGHTEN_SL = auto()
    REENTER_AFTER_COOLDOWN = auto()
