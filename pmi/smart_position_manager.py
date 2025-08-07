# File: pmi/smart_position_manager.py
"""Punto de orquestación: decide acciones sobre posiciones abiertas."""

from __future__ import annotations

from typing import List, Dict, Any
import datetime as dt

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
        close_thresholds: Dict[str, float] | None = None,
        peers_map: Dict[str, list[str]] | None = None,
    ):
        self.calc = calc or ProbabilityCalculator()
        self.timer = timer or TimingManager()
        self.guards = guards or RiskGuards()
        self.trend = trend or TrendChangeDetector()
        self.corr = corr or CorrelationEngine()

        # Umbrales de acción
        self.close_thresholds = close_thresholds or {
            "tighten_sl": 0.70,
            "partial_close": 0.82,
            "close": 0.90,
        }

        # Pares de referencia por símbolo (ajústalo a tus instrumentos)
        self.peers_map = peers_map or {
            "EURUSD": ["GBPUSD"],
            "GBPUSD": ["EURUSD"],
            "AUDUSD": ["NZDUSD"],
            "USDJPY": ["EURUSD", "GBPUSD"],
        }

    # --------------------------------------------------
    def _factors_for_position(
        self,
        pos: Dict[str, Any],
        market_snapshot: Dict[str, Any],
        candles_by_symbol: Dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        """
        Calcula factores normalizados [0,1] por posición.
        - corr_signal   : divergencia media con peers
        - trend_change  : prob. giro desde TCD
        - time_in_profit: normaliza minutos en profit (cap a 120m → 1.0)
        - volatility    : ATR relativo (si está en snapshot, else 0)
        """
        symbol = pos["symbol"]
        # --- Divergencia con peers
        peers = self.peers_map.get(symbol, [])
        corr_signal = self.corr.divergence_score(symbol, peers)

        # --- Probabilidad de cambio de tendencia (si tenemos velas)
        trend_prob = 0.0
        if candles_by_symbol and symbol in candles_by_symbol:
            df = candles_by_symbol[symbol]
            try:
                out = self.trend.estimate_probability(df)
                trend_prob = float(out.get("probability", 0.0))
            except Exception:
                trend_prob = 0.0

        # --- Tiempo en profit (placeholder si no hay P/L)
        # si viene en pos['minutes_in_profit'] úsalo, sino 0
        mip = float(pos.get("minutes_in_profit", 0.0))
        time_in_profit = max(0.0, min(1.0, mip / 120.0))  # hasta 2h → 1.0

        # --- Volatilidad (si snapshot trae atr_rel, usa eso)
        vol = 0.0
        ms = market_snapshot.get(symbol, {})
        if isinstance(ms, dict):
            vol = float(ms.get("atr_rel", 0.0))

        return {
            "corr_signal": corr_signal,
            "trend_change": trend_prob,
            "time_in_profit": time_in_profit,
            "volatility": vol,
        }

    # --------------------------------------------------
    def evaluate(
        self,
        positions: List[Dict[str, Any]],
        market_snapshot: Dict[str, Any],
        candles_by_symbol: Dict[str, Any] | None = None,
        now: dt.datetime | None = None,
    ) -> List[PMIDecision]:
        """
        Evalúa cada posición y devuelve decisiones PMI.
        No envía órdenes — el bot principal actuará si corresponde.
        """
        now = now or dt.datetime.utcnow()
        decisions: List[PMIDecision] = []

        for pos in positions:
            symbol = pos["symbol"]
            ticket = int(pos["ticket"])
            # cooldown por símbolo (anti-churn)
            if self.timer.is_cooldown(symbol, now):
                decisions.append(PMIDecision(
                    ticket=ticket,
                    action=DecisionAction.HOLD,
                    confidence=0.0,
                    close_score=0.0,
                    reason="cooldown",
                    telemetry={"cooldown": True},
                ))
                continue

            # Actualiza buffers de correlación si tenemos último precio
            last_price = market_snapshot.get(symbol, {}).get("close")
            if last_price is not None:
                try:
                    self.corr.update(symbol, float(last_price))
                except Exception:
                    pass

            # Calcula factores y score
            factors = self._factors_for_position(pos, market_snapshot, candles_by_symbol)
            close_score = self.calc.combine(factors)

            # Determina acción
            if close_score >= self.close_thresholds["close"]:
                action = DecisionAction.CLOSE
                confidence = close_score
                reason = "close_threshold"
                self.timer.set_cooldown(symbol, minutes=5, now=now)  # anti-reentrada inmediata
            elif close_score >= self.close_thresholds["partial_close"]:
                action = DecisionAction.PARTIAL_CLOSE
                confidence = close_score
                reason = "partial_threshold"
            elif close_score >= self.close_thresholds["tighten_sl"]:
                action = DecisionAction.TIGHTEN_SL
                confidence = close_score
                reason = "tighten_threshold"
            else:
                action = DecisionAction.HOLD
                confidence = 1.0 - close_score
                reason = "hold"

            decisions.append(PMIDecision(
                ticket=ticket,
                action=action,
                confidence=float(confidence),
                close_score=float(close_score),
                reason=reason,
                telemetry={"factors": factors},
            ))

        return decisions
