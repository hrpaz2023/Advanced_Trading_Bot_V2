# ------------------------------------------------------------
# File: pmi/decision.py
# ------------------------------------------------------------
"""Objeto ligero para transportar la decisión del PMI."""

from dataclasses import dataclass, field
from typing import Dict, Any
from .enums import DecisionAction


@dataclass
class PMIDecision:
    ticket: int
    action: DecisionAction
    confidence: float
    close_score: float = 0.0
    reversal_probability: float = 0.0
    cooldown_minutes: int | None = None
    reason: str = ""
    telemetry: Dict[str, Any] = field(default_factory=dict)