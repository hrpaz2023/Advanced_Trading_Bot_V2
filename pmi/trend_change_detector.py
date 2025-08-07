# File: pmi/trend_change_detector.py
"""Estimación de cambio de tendencia mediante indicadores técnicos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib as ta
from typing import Dict


class TrendChangeDetector:
    """
    Calcula la probabilidad (0-1) de que se produzca un giro de tendencia
    en los próximos N candles combinando señales de RSI, MACD y expansión
    de volatilidad (ATR).  Las heurísticas son simples y se calibrarán con
    datos reales más adelante.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        atr_period: int = 14,
    ):
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.atr_period = atr_period

    # ------------------------------------------------------------------
    def estimate_probability(self, candles: pd.DataFrame) -> Dict[str, float]:
        """
        Args
        ----
        candles : DataFrame
            Columnas requeridas: ['open', 'high', 'low', 'close', 'volume'].

        Returns
        -------
        dict  Ejemplo →
            {
                "probability": 0.73,
                "details": {
                    "rsi": 29.8,
                    "rsi_signal": 1,
                    "macd_hist": -0.0007,
                    "macd_cross": -1,
                    "atr": 0.0019,
                    "atr_change": 0.12
                }
            }
        """
        # --- recorta a la ventana necesaria
        lookback = max(self.macd_slow, self.rsi_period, self.atr_period) + 3
        df = candles.tail(lookback).copy()

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)

        # ----------------------- RSI
        rsi = ta.RSI(close, timeperiod=self.rsi_period)
        rsi_current = rsi[-1]
        # señal: 1 = sobreventa (posible giro alcista), -1 = sobrecompra
        rsi_signal = 1 if rsi_current < 30 else -1 if rsi_current > 70 else 0

        # ----------------------- MACD
        macd, macd_sig, macd_hist = ta.MACD(
            close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal,
        )
        # cruce del histograma por cero → aviso de giro
        macd_hist_curr = macd_hist[-1]
        macd_hist_prev = macd_hist[-2] if len(macd_hist) >= 2 else 0
        macd_cross = 0
        if macd_hist_prev < 0 < macd_hist_curr:
            macd_cross = 1       # giro alcista
        elif macd_hist_prev > 0 > macd_hist_curr:
            macd_cross = -1      # giro bajista

        # ----------------------- ATR
        atr = ta.ATR(high, low, close, timeperiod=self.atr_period)
        atr_change = (
            (atr[-1] - atr[-2]) / atr[-2] if len(atr) >= 2 and atr[-2] != 0 else 0
        )

        # ----------------------- Heurística de combinación
        prob = (
            0.5
            + 0.25 * rsi_signal
            + 0.25 * macd_cross
            + 0.1 * np.tanh(atr_change * 5)
        )
        prob_clipped = float(np.clip(prob, 0.0, 1.0))

        details = dict(
            rsi=float(rsi_current),
            rsi_signal=rsi_signal,
            macd_hist=float(macd_hist_curr),
            macd_cross=macd_cross,
            atr=float(atr[-1]),
            atr_change=float(atr_change),
        )

        return {"probability": prob_clipped, "details": details}
