# ------------------------------------------------------------
# File: pmi/smart_position_manager.py
# ------------------------------------------------------------
"""Punto de orquestación: decide acciones sobre posiciones abiertas."""

from __future__ import annotations

from typing import List, Dict, Any
from .decision import PMIDecision
from .enums import DecisionAction
from .probability_calculator import ProbabilityCalculator
from .timing_manager import TimingManager
from .risk_guards import RiskGuards
from .trend_change_detector import TrendChangeDetector
from .correlation_engine import CorrelationEngine


class SmartPositionManager:
    def __init__(
        self,
        calc: ProbabilityCalculator | None = None,
        timer: TimingManager | None = None,
        guards: RiskGuards | None = None,
        trend: TrendChangeDetector | None = None,
        corr: CorrelationEngine | None = None,
        close_threshold: float = 0.65,
    ):
        self.calc = calc or ProbabilityCalculator()
        self.timer = timer or TimingManager()
        self.guards = guards or RiskGuards()
        self.trend = trend or TrendChangeDetector()
        self.corr = corr or CorrelationEngine()
        self.close_threshold = close_threshold

    # --------------------------------------------------
    def evaluate(self, positions: List[Dict[str, Any]], market_snapshot: Dict[str, Any]) -> List[PMIDecision]:
        """Evalúa cada posición y devuelve decisiones PMI."""
        decisions: List[PMIDecision] = []
        # TODO: iterar posiciones, calcular factores y tomar decisiones
        return decisions