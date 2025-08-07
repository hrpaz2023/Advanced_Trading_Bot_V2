# ------------------------------------------------------------
# File: pmi/correlation_engine.py
# ------------------------------------------------------------
"""Cálculo de correlaciones y detección de divergencias."""

from __future__ import annotations

import pandas as pd
import numpy as np
from collections import deque
from typing import Dict, Deque


class CorrelationEngine:
    """Mantiene buffers de precios y calcula correlaciones rolling."""

    def __init__(self, window: int = 400):
        self.window = window
        self._buffers: Dict[str, Deque[float]] = {}

    # --------------------------------------------------
    # Data update
    # --------------------------------------------------
    def update(self, symbol: str, close_price: float) -> None:
        """Añade un nuevo precio de cierre al buffer del símbolo."""
        buf = self._buffers.setdefault(symbol, deque(maxlen=self.window))
        buf.append(close_price)

    # --------------------------------------------------
    # Analysis helpers
    # --------------------------------------------------
    def _compute_corr(self, s1: str, s2: str) -> float | None:
        b1, b2 = self._buffers.get(s1), self._buffers.get(s2)
        if b1 is None or b2 is None or len(b1) < 30 or len(b2) < 30:
            return None
        return float(np.corrcoef(b1, b2)[0, 1])

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def detect_divergence(self, main_symbol: str, pairs: list[str],
                          threshold: float = -0.3) -> Dict[str, float]:
        """Detecta rupturas de correlación.

        Args:
            main_symbol: símbolo que se está evaluando.
            pairs: lista de símbolos correlacionados a chequear.
            threshold: valor máximo permitido antes de marcar ruptura.
        Returns:
            Dict con pares que rompen correlación y su nuevo coef.
        """
        signals: Dict[str, float] = {}
        for peer in pairs:
            corr = self._compute_corr(main_symbol, peer)
            if corr is not None and corr < threshold:
                signals[peer] = corr
        return signals
