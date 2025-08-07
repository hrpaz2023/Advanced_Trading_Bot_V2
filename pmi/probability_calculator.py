# ------------------------------------------------------------
# File: pmi/probability_calculator.py
# ------------------------------------------------------------
"""Combina factores individuales en un único score de cierre."""

from __future__ import annotations

from typing import Dict


class ProbabilityCalculator:
    def __init__(self, weights: Dict[str, float] | None = None):
        self.weights = weights or {
            "corr_signal": 0.30,
            "trend_change": 0.40,
            "time_in_profit": 0.20,
            "volatility": 0.10,
        }

    # --------------------------------------------------
    def combine(self, factors: Dict[str, float]) -> float:
        """Devuelve score de cierre ponderado (0‑1)."""
        score = 0.0
        for k, w in self.weights.items():
            score += w * factors.get(k, 0.0)
        return score