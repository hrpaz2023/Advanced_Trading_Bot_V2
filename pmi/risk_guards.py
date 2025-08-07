# ------------------------------------------------------------
# File: pmi/risk_guards.py
# ------------------------------------------------------------
"""Salvaguardas de riesgo para limitar exposición y cierres masivos."""

from __future__ import annotations

from typing import Dict


class RiskGuards:
    def __init__(self, max_exposure_per_group: float = 0.2):
        self.max_exposure_per_group = max_exposure_per_group
        self._group_exposure: Dict[str, float] = {}

    def update_exposure(self, group: str, risk: float):
        self._group_exposure[group] = risk

    def check(self, group: str, new_risk: float) -> bool:
        """True si excede el límite y debería bloquear entrada."""
        return (self._group_exposure.get(group, 0.0) + new_risk) > self.max_exposure_per_group
