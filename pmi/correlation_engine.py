# File: pmi/correlation_engine.py
"""Cálculo de correlaciones y detección de divergencias."""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, Deque, List


class CorrelationEngine:
    """
    Mantiene buffers de precios y calcula correlaciones rolling.
    También estima un "divergence_score" normalizado (0-1) por símbolo,
    útil como factor para el cierre de posiciones.
    """

    def __init__(self, window: int = 400, min_samples: int = 60):
        self.window = window
        self.min_samples = min_samples
        self._buffers: Dict[str, Deque[float]] = {}

    # --------------------------------------------------
    # Data update
    # --------------------------------------------------
    def update(self, symbol: str, close_price: float) -> None:
        """Añade un nuevo precio de cierre al buffer del símbolo."""
        buf = self._buffers.setdefault(symbol, deque(maxlen=self.window))
        buf.append(float(close_price))

    # --------------------------------------------------
    # Analysis helpers
    # --------------------------------------------------
    def _get_series(self, symbol: str) -> np.ndarray | None:
        buf = self._buffers.get(symbol)
        if buf is None or len(buf) < self.min_samples:
            return None
        return np.asarray(buf, dtype=float)

    def _rolling_corr(self, a: np.ndarray, b: np.ndarray, win: int = 60) -> float | None:
        n = min(len(a), len(b))
        if n < win:
            return None
        a_win = a[-win:]
        b_win = b[-win:]
        c = np.corrcoef(a_win, b_win)[0, 1]
        return float(c)

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def detect_divergence(
        self,
        main_symbol: str,
        peers: List[str],
        corr_break_threshold: float = -0.25,
        window_corr: int = 60,
    ) -> Dict[str, float]:
        """
        Devuelve pares con ruptura de correlación (corr < threshold).
        """
        base = self._get_series(main_symbol)
        signals: Dict[str, float] = {}
        if base is None:
            return signals

        for p in peers:
            s = self._get_series(p)
            if s is None:
                continue
            c = self._rolling_corr(base, s, win=window_corr)
            if c is not None and c < corr_break_threshold:
                signals[p] = c
        return signals

    def divergence_score(
        self,
        main_symbol: str,
        peers: List[str],
        window_corr: int = 60,
    ) -> float:
        """
        Combina correlaciones con pares en un score [0,1],
        donde 0 = sin divergencia, 1 = divergencia fuerte generalizada.
        """
        base = self._get_series(main_symbol)
        if base is None:
            return 0.0

        vals = []
        for p in peers:
            s = self._get_series(p)
            if s is None:
                continue
            c = self._rolling_corr(base, s, win=window_corr)
            if c is not None:
                # mapear corr [-1,1] a "divergencia" [0,1]
                # 1 - c → 0 si c=1 (muy correl.), 2 si c=-1; lo normalizamos a [0,1]
                div = (1 - c) / 2.0
                vals.append(div)

        if not vals:
            return 0.0
        # promedio suavizado
        return float(np.clip(np.mean(vals), 0.0, 1.0))
