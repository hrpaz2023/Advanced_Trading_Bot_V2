# File: pmi/probability_calculator.py
"""Combina factores individuales en un único score de cierre."""

from __future__ import annotations

from typing import Dict


class ProbabilityCalculator:
    def __init__(self, weights: Dict[str, float] | None = None):
        # Pesos iniciales (se calibran después con logs)
        self.weights = weights or {
            "corr_signal": 0.30,     # fuerza de divergencia con pares
            "trend_change": 0.45,    # prob. de giro (TCD)
            "time_in_profit": 0.15,  # mayor tiempo en profit → más propensión a cerrar
            "volatility": 0.10,      # mayor vol → priorizar protección
        }

    def combine(self, factors: Dict[str, float]) -> float:
        """
        Devuelve score de cierre ponderado (0-1).
        Faltantes se consideran 0.
        """
        score = 0.0
        total_w = 0.0
        for k, w in self.weights.items():
            score += w * float(factors.get(k, 0.0))
            total_w += w
        if total_w <= 0:
            return 0.0
        out = score / total_w
        return float(max(0.0, min(1.0, out)))
