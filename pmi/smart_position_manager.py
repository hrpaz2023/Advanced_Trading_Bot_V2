# pmi/smart_position_manager.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import datetime as dt

try:
    import pandas as pd  # solo se usa si llega candles_by_symbol
except Exception:  # pragma: no cover
    pd = None  # degradación suave

# ---------------------------------------------------------------------
# Enums (usamos los tuyos si existen; si no, fallback local)
# ---------------------------------------------------------------------
try:
    from .enums import DecisionAction
except Exception:
    class DecisionAction:  # fallback mínimo
        HOLD = "HOLD"
        TIGHTEN_SL = "TIGHTEN_SL"
        PARTIAL_CLOSE = "PARTIAL_CLOSE"
        CLOSE = "CLOSE"


# ---------------------------------------------------------------------
# Dataclass de salida (si ya tenés algo similar, este es compatible)
# ---------------------------------------------------------------------
@dataclass
class PMIDecision:
    ticket: int
    symbol: str
    action: Any  # DecisionAction o str
    reason: str
    close_score: float = 0.0
    fraction: float = 0.0
    telemetry: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------
# SmartPositionManager
# ---------------------------------------------------------------------
class SmartPositionManager:
    """
    PMI: decide TIGHTEN/PARTIAL/CLOSE según:
    - Escalera basada en R (PnL/ATR aprox)
    - Score de debilidad (TCD, slope EMA20, contracción ATR, RSI, proximidad S/R)
    - Umbrales externos en configs/pmi_config.json
    - S/R diarios desde configs/daily_sentiment_YYYYMMDD.json (opcional)
    """

    def __init__(
        self,
        mode: str = "active",
        close_thresholds: Optional[Dict[str, Any]] = None,
        pmi_config_path: str = "configs/pmi_config.json",
        daily_sentiment_path: Optional[str] = None,
        logger: Any = None,
    ) -> None:
        self.mode = (mode or "observer").lower()
        self.logger = logger

        # 1) Cargar thresholds desde pmi_config.json (si no vinieron por parámetro)
        cfg = self._load_pmi_config(pmi_config_path)
        if close_thresholds is None:
            close_thresholds = cfg.get("thresholds", {}) or {}
        self.close_thresholds = self._merge_thresholds(close_thresholds)

        # 2) Cargar S/R desde daily sentiment (opcional)
        self.sr_levels = self._load_daily_sentiment(daily_sentiment_path)

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------
    def evaluate(
        self,
        positions: List[Dict[str, Any]],
        market_snapshot: Dict[str, Dict[str, float]],
        candles_by_symbol: Optional[Dict[str, "pd.DataFrame"]] = None,
        now: Optional[dt.datetime] = None,
        signal_context_by_symbol: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[PMIDecision]:
        """
        Devuelve lista de decisiones PMI por posición abierta.
        - positions: [{"ticket", "symbol", "type" (BUY/SELL), "volume", "price_open", "sl" (opc), "tp" (opc)}]
        - market_snapshot: {symbol: {"close":..., "atr":..., "atr_rel":...}}
        - candles_by_symbol: {symbol: df con columnas típicas ["time","open","high","low","close"]}
        - signal_context_by_symbol: {symbol: {"signal_side","ml_confidence","historical_prob_lb90","tcd_prob", ...}}
        """
        now = now or dt.datetime.now(dt.timezone.utc)
        out: List[PMIDecision] = []

        for pos in positions:
            symbol = str(pos.get("symbol"))
            ticket = int(pos.get("ticket", 0))
            side = str(pos.get("type", "")).upper()  # BUY/SELL
            if not symbol or side not in ("BUY", "SELL"):
                continue

            snap = market_snapshot.get(symbol, {}) if market_snapshot else {}
            close = _num(snap.get("close"))
            atr_abs = _num(snap.get("atr"))
            atr_rel = _num(snap.get("atr_rel"))
            if atr_abs <= 0 and atr_rel > 0 and close > 0:
                atr_abs = close * atr_rel

            price_open = _num(pos.get("price_open"))
            if close <= 0 or price_open <= 0 or atr_abs <= 0:
                # sin datos suficientes -> HOLD
                out.append(PMIDecision(
                    ticket=ticket, symbol=symbol,
                    action=DecisionAction.HOLD, reason="insufficient_snapshot",
                    telemetry={"have_close": close > 0, "have_open": price_open > 0, "have_atr": atr_abs > 0}
                ))
                continue

            # PnL en ATR ("R" aproximado)
            pnl_r = _pnl_r(side, price_open, close, atr_abs)

            # Señal reciente / TCD / prob LB90
            ctx = (signal_context_by_symbol or {}).get(symbol, {}) if signal_context_by_symbol else {}
            ml = _num(ctx.get("ml_confidence"))
            lb90 = _num(ctx.get("historical_prob_lb90"))
            tcd_prob = _num(ctx.get("tcd_prob"))
            sig_side = (ctx.get("signal_side") or "").upper()

            # S/R diarios si existen
            sr = (self.sr_levels.get(symbol) if self.sr_levels else {}) or {}
            support = _num(sr.get("support"))
            resistance = _num(sr.get("resistance"))

            # Debilidad del movimiento actual
            deb_score, deb_factors = self._weakness_score(
                symbol=symbol,
                side=side,
                close=close,
                atr=atr_abs,
                tcd_prob=tcd_prob,
                lb90=lb90,
                ml=ml,
                sig_side=sig_side,
                support=support if support > 0 else None,
                resistance=resistance if resistance > 0 else None,
                df=(candles_by_symbol or {}).get(symbol) if candles_by_symbol else None,
            )

            # Decisiones por escalera R-based
            ladder_decision = self._ladder_decision(pnl_r)

            # Decisiones por debilidad
            weakness_decision = self._weakness_decision(deb_score)

            # Selección de acción final (la “más fuerte”)
            final_decision, reason, fraction = self._combine_actions(ladder_decision, weakness_decision)

            # Si hay señal opuesta fuerte, puede superar
            opp_action, opp_reason = self._opposite_signal_decision(side, sig_side, ml, lb90)
            if opp_action is not None:
                final_decision, reason = opp_action, opp_reason

            # En modo observer → no aplica, pero reporta
            action_to_report = final_decision if self.mode == "active" else DecisionAction.HOLD

            out.append(PMIDecision(
                ticket=ticket,
                symbol=symbol,
                action=action_to_report,
                reason=reason,
                close_score=deb_score,
                fraction=fraction,
                telemetry={
                    "factors": deb_factors,
                    "ml": ml, "lb90": lb90, "tcd": tcd_prob,
                    "pos_side": side, "sig_side": sig_side,
                    "r_pnl": pnl_r,
                    "sr_support": support if support > 0 else None,
                    "sr_resistance": resistance if resistance > 0 else None,
                }
            ))

        return out

    # ---------------------------------------------------------
    # Internals
    # ---------------------------------------------------------
    def _load_pmi_config(self, path: str) -> Dict[str, Any]:
        defaults = {
            "mode": "active",
            "thresholds": {}
        }
        try:
            p = Path(path)
            if not p.exists():
                self._log(f"PMI config no encontrado: {path}. Uso defaults.")
                return defaults
            with p.open("r", encoding="utf-8") as f:
                cfg = json.load(f) or {}
            return cfg
        except Exception as e:
            self._log(f"PMI config inválido ({path}): {e}. Uso defaults.")
            return defaults

    def _merge_thresholds(self, user_thr: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge de thresholds del JSON con defaults internos:
        """
        d = {
            # Escalera R-based
            "ladder": {
                "be_at": 0.30,
                "tighten_at": 0.60,
                "partial1_at": 1.00,
                "partial2_at": 1.50,
                "partial_fraction": 0.50
            },
            # Debilidad
            "weak_score": {
                "tighten": 0.55,
                "partial": 0.70,
                "close": 0.85
            },
            # TCD puro
            "tcd": {
                "tighten": 0.55,
                "close": 0.70
            },
            # Señal opuesta
            "opp_partial_ml": 0.55,
            "opp_partial_lb90": 0.50,
            "opp_close_ml": 0.58,
            "opp_close_lb90": 0.53
        }
        try:
            u = dict(user_thr or {})
            # sub-bloques
            for k in ("ladder", "weak_score", "tcd"):
                if k in u and isinstance(u[k], dict):
                    d[k].update(u[k])
            # llaves planas
            for k in ("opp_partial_ml", "opp_partial_lb90", "opp_close_ml", "opp_close_lb90"):
                if k in u:
                    d[k] = float(u[k])
        except Exception:
            pass
        return d

    def _load_daily_sentiment(self, sentiment_path: Optional[str]) -> Dict[str, Dict[str, float]]:
        """
        Lee soporte/resistencia desde:
          - path explícito, o
          - configs/daily_sentiment_YYYYMMDD.json (UTC hoy)
        Estructura esperada:
        { "pairs": { "EURUSD": {"support":1.1445,"resistance":1.167, ...}, ... } }
        """
        try:
            p = None
            if sentiment_path:
                p = Path(sentiment_path)
            else:
                today = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
                p = Path(f"configs/daily_sentiment_{today}.json")
            if not p.exists():
                self._log(f"Daily sentiment no encontrado ({p}). S/R no disponibles.")
                return {}
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f) or {}
            pairs = data.get("pairs", {}) or {}
            out = {}
            for sym, vals in pairs.items():
                out[str(sym).upper()] = {
                    "support": _num((vals or {}).get("support")),
                    "resistance": _num((vals or {}).get("resistance")),
                }
            self._log(f"✅ Soportes/Resistencias cargados de {p}")
            return out
        except Exception as e:
            self._log(f"Daily sentiment inválido: {e}. S/R no disponibles.")
            return {}

    def _weakness_score(
        self,
        symbol: str,
        side: str,
        close: float,
        atr: float,
        tcd_prob: float,
        lb90: float,
        ml: float,
        sig_side: str,
        support: Optional[float],
        resistance: Optional[float],
        df: Optional["pd.DataFrame"],
    ) -> (float, Dict[str, Any]):
        """
        Combina factores de debilidad en [0..1].
        Pesos:
          - TCD: 0.40
          - EMA20 slope <= 0: 0.20
          - Contracción ATR: 0.15
          - RSI “salida de momentum”: 0.15
          - Proximidad a S/R: 0.10
        """
        weights = {"tcd": 0.40, "slope": 0.20, "atr_contract": 0.15, "rsi_exit": 0.15, "sr_near": 0.10}
        f = {"tcd": 0.0, "slope": 0.0, "atr_contract": 0.0, "rsi_exit": 0.0, "sr_near": 0.0}

        # TCD
        f["tcd"] = float(max(0.0, min(1.0, tcd_prob))) if tcd_prob == tcd_prob else 0.0  # NaN-safe

        # EMA20 slope / ATR change / RSI a partir de DF si existe
        if pd is not None and isinstance(df, pd.DataFrame) and len(df) >= 25 and "close" in df.columns:
            # slope: diferencia de medias últimas 20 vs 20-previas
            try:
                ema = df["close"].ewm(span=20, adjust=False).mean()
                slope = float(ema.iloc[-1] - ema.iloc[-5])  # simple pendiente
                f["slope"] = 1.0 if (side == "BUY" and slope <= 0) or (side == "SELL" and slope >= 0) else 0.0
            except Exception:
                pass
            # ATR contracción: prox con std/true range if present; degradación suave
            try:
                if "high" in df.columns and "low" in df.columns:
                    tr = (df["high"] - df["low"]).abs()
                    atr20 = tr.rolling(20).mean().iloc[-1]
                    atr20_prev = tr.rolling(20).mean().iloc[-6]
                    change = (atr20 - atr20_prev) / max(1e-9, abs(atr20_prev))
                    f["atr_contract"] = 1.0 if change < -0.05 else (0.5 if change < 0 else 0.0)
            except Exception:
                pass
            # RSI básico
            try:
                delta = df["close"].diff()
                up = delta.clip(lower=0.0).rolling(14).mean()
                down = (-delta.clip(upper=0.0)).rolling(14).mean()
                rs = up / (down + 1e-9)
                rsi = 100 - (100 / (1 + rs))
                rsi_last = float(rsi.iloc[-1])
                if side == "BUY":
                    f["rsi_exit"] = 1.0 if rsi_last < 60 else (0.5 if rsi_last < 65 else 0.0)
                else:
                    f["rsi_exit"] = 1.0 if rsi_last > 40 else (0.5 if rsi_last > 35 else 0.0)
            except Exception:
                pass

        # Proximidad a S/R (en ATR)
        try:
            if side == "BUY" and resistance:
                dist = max(0.0, resistance - close) / max(1e-9, atr)
                f["sr_near"] = 1.0 if dist < 0.25 else (0.5 if dist < 0.50 else 0.0)
            elif side == "SELL" and support:
                dist = max(0.0, close - support) / max(1e-9, atr)
                f["sr_near"] = 1.0 if dist < 0.25 else (0.5 if dist < 0.50 else 0.0)
        except Exception:
            pass

        score = sum(f[k] * weights[k] for k in weights)
        return float(max(0.0, min(1.0, score))), f

    def _ladder_decision(self, pnl_r: float) -> PMIDecision | None:
        thr = self.close_thresholds.get("ladder", {})
        be_at = float(thr.get("be_at", 0.30))
        tighten_at = float(thr.get("tighten_at", 0.60))
        p1_at = float(thr.get("partial1_at", 1.00))
        p2_at = float(thr.get("partial2_at", 1.50))
        pf = float(thr.get("partial_fraction", 0.50))

        if pnl_r >= p2_at:
            return PMIDecision(ticket=0, symbol="", action=DecisionAction.PARTIAL_CLOSE,
                               reason="ladder_partial2", fraction=pf)
        if pnl_r >= p1_at:
            return PMIDecision(ticket=0, symbol="", action=DecisionAction.PARTIAL_CLOSE,
                               reason="ladder_partial1", fraction=pf)
        if pnl_r >= tighten_at:
            return PMIDecision(ticket=0, symbol="", action=DecisionAction.TIGHTEN_SL,
                               reason="ladder_tighten")
        if pnl_r >= be_at:
            return PMIDecision(ticket=0, symbol="", action=DecisionAction.TIGHTEN_SL,
                               reason="ladder_break_even")
        return None

    def _weakness_decision(self, score: float) -> PMIDecision | None:
        thr = self.close_thresholds.get("weak_score", {})
        t_tight = float(thr.get("tighten", 0.55))
        t_part = float(thr.get("partial", 0.70))
        t_close = float(thr.get("close", 0.85))
        if score >= t_close:
            return PMIDecision(ticket=0, symbol="", action=DecisionAction.CLOSE,
                               reason="weak_score_close", close_score=score)
        if score >= t_part:
            return PMIDecision(ticket=0, symbol="", action=DecisionAction.PARTIAL_CLOSE,
                               reason="weak_score_partial", close_score=score, fraction=float(
                                   self.close_thresholds.get("ladder", {}).get("partial_fraction", 0.50)))
        if score >= t_tight:
            return PMIDecision(ticket=0, symbol="", action=DecisionAction.TIGHTEN_SL,
                               reason="weak_score_tighten", close_score=score)
        # adicional: reglas puras por TCD si se desean
        tcd = self.close_thresholds.get("tcd", {})
        tcd_tight = float(tcd.get("tighten", 0.55))
        tcd_close = float(tcd.get("close", 0.70))
        if score == score:  # evita NaN
            pass
        return None

    def _opposite_signal_decision(self, pos_side: str, sig_side: str, ml: float, lb90: float):
        """
        Señal opuesta con high-confidence puede forzar PARTIAL/CLOSE.
        """
        if sig_side not in ("BUY", "SELL") or pos_side not in ("BUY", "SELL"):
            return None, ""
        if sig_side == pos_side:
            return None, ""
        thr = self.close_thresholds
        if ml >= float(thr.get("opp_close_ml", 0.58)) and lb90 >= float(thr.get("opp_close_lb90", 0.53)):
            return DecisionAction.CLOSE, "opposite_signal_strong"
        if ml >= float(thr.get("opp_partial_ml", 0.55)) and lb90 >= float(thr.get("opp_partial_lb90", 0.50)):
            return DecisionAction.PARTIAL_CLOSE, "opposite_signal_medium"
        return None, ""

    def _combine_actions(self, a: Optional[PMIDecision], b: Optional[PMIDecision]):
        """
        Selecciona la acción “más fuerte”.
        CLOSE > PARTIAL > TIGHTEN > HOLD
        """
        order = {DecisionAction.CLOSE: 3, DecisionAction.PARTIAL_CLOSE: 2,
                 DecisionAction.TIGHTEN_SL: 1, DecisionAction.HOLD: 0}

        cand = [x for x in (a, b) if x is not None]
        if not cand:
            return DecisionAction.HOLD, "hold", 0.0
        best = max(cand, key=lambda x: order.get(x.action, 0))
        return best.action, best.reason, best.fraction or 0.0

    def _log(self, msg: str) -> None:
        try:
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
        except Exception:
            print(msg)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _num(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v == v:
            return v
        return default
    except Exception:
        return default


def _pnl_r(side: str, price_open: float, close: float, atr: float) -> float:
    if atr <= 0:
        return 0.0
    if side == "BUY":
        return (close - price_open) / atr
    else:
        return (price_open - close) / atr
