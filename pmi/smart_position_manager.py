# pmi/smart_position_manager.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import datetime as dt

try:
    import pandas as pd  # solo si llegan velas
except Exception:
    pd = None

# ---------------------------------------------------------------------
# Enums (usamos los tuyos si existen; si no, fallback local)
# ---------------------------------------------------------------------
try:
    from .enums import DecisionAction
except Exception:
    class DecisionAction:
        HOLD = "HOLD"
        TIGHTEN_SL = "TIGHTEN_SL"
        PARTIAL_CLOSE = "PARTIAL_CLOSE"
        CLOSE = "CLOSE"


# ---------------------------------------------------------------------
# Estructura de salida
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
    PMI: decide TIGHTEN/PARTIAL/CLOSE combinando:
    - Escalera basada en R (PnL/ATR aprox)
    - Score de debilidad (TCD, slope EMA20, contracción ATR, RSI, proximidad S/R)
    - Umbrales externos en configs/pmi_config.json
    - S/R diarios desde configs/daily_sentiment_YYYYMMDD.json (opcional)
    - OBJETIVOS EN USD POR TRADE (usd_partial, usd_close)
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

        # 1) thresholds base
        cfg = self._load_pmi_config(pmi_config_path)
        if close_thresholds is None:
            close_thresholds = cfg.get("thresholds", {}) or {}
        self.close_thresholds = self._merge_thresholds(close_thresholds)

        # 2) objetivos en USD por trade
        self.usd_targets = self._merge_usd_targets(cfg.get("profit_targets", {}))

        # 3) S/R diarios
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
        - positions: [{"ticket","symbol","type"(BUY/SELL),"volume","price_open","sl","tp"}]
        - market_snapshot: {symbol: {"close":..., "atr":..., "atr_rel":..., "contract_size":..., "point":..., "pip":..., "usd_per_pip_per_lot":...}}
        - candles_by_symbol: {symbol: df con ["time","open","high","low","close"]}
        - signal_context_by_symbol: {symbol: {"signal_side","ml_confidence","historical_prob_lb90","tcd_prob", ...}}
        """
        now = now or dt.datetime.now(dt.timezone.utc)
        out: List[PMIDecision] = []

        for pos in positions:
            symbol = str(pos.get("symbol", "")).upper()
            ticket = int(pos.get("ticket", 0))
            side = str(pos.get("type", "")).upper()  # BUY/SELL
            vol = _num(pos.get("volume"))
            price_open = _num(pos.get("price_open"))

            if not symbol or side not in ("BUY", "SELL") or vol <= 0 or price_open <= 0:
                continue

            snap = market_snapshot.get(symbol, {}) if market_snapshot else {}
            close = _num(snap.get("close"))
            atr_abs = _num(snap.get("atr"))
            atr_rel = _num(snap.get("atr_rel"))
            if atr_abs <= 0 and atr_rel > 0 and close > 0:
                atr_abs = close * atr_rel

            if close <= 0 or atr_abs <= 0:
                out.append(PMIDecision(
                    ticket=ticket, symbol=symbol,
                    action=DecisionAction.HOLD, reason="insufficient_snapshot",
                    telemetry={"have_close": close > 0, "have_open": price_open > 0, "have_atr": atr_abs > 0}
                ))
                continue

            # ========= PnL aproximado en R (ATR) y en USD =========
            pnl_r = _pnl_r(side, price_open, close, atr_abs)
            usd_pppl = _num(snap.get("usd_per_pip_per_lot"))  # si el main lo provee, perfecto
            contract_size = _num(snap.get("contract_size"), 100_000.0)  # default FX lot
            point = _num(snap.get("point"))  # tamaño de punto (tick)
            usd_pnl = self._estimate_unrealized_usd(symbol, side, price_open, close, vol, contract_size, point, usd_pppl)

            # ========= Contexto de señal reciente / TCD =========
            ctx = (signal_context_by_symbol or {}).get(symbol, {}) if signal_context_by_symbol else {}
            ml = _num(ctx.get("ml_confidence"))
            lb90 = _num(ctx.get("historical_prob_lb90"))
            tcd_prob = _num(ctx.get("tcd_prob"))
            sig_side = (ctx.get("signal_side") or "").upper()

            # ========= S/R diarios si existen =========
            sr = (self.sr_levels.get(symbol) if self.sr_levels else {}) or {}
            support = _num(sr.get("support"))
            resistance = _num(sr.get("resistance"))

            # ========= Debilidad del movimiento =========
            deb_score, deb_factors = self._weakness_score(
                symbol=symbol, side=side, close=close, atr=atr_abs,
                tcd_prob=tcd_prob, lb90=lb90, ml=ml, sig_side=sig_side,
                support=support if support > 0 else None,
                resistance=resistance if resistance > 0 else None,
                df=(candles_by_symbol or {}).get(symbol) if candles_by_symbol else None,
            )

            # ========= 1) Regla por objetivos USD (tiene prioridad) =========
            usd_decision = self._usd_target_decision(usd_pnl)

            # ========= 2) Escalera R-based =========
            ladder_decision = self._ladder_decision(pnl_r)

            # ========= 3) Debilidad técnica =========
            weakness_decision = self._weakness_decision(deb_score)

            # ========= 4) Señal opuesta fuerte =========
            opp_action, opp_reason = self._opposite_signal_decision(side, sig_side, ml, lb90)

            # Combinar (prioridad: opp > usd > weakness > ladder > hold)
            final_action, reason, fraction = DecisionAction.HOLD, "hold", 0.0
            if opp_action is not None:
                final_action, reason = opp_action, opp_reason
            elif usd_decision is not None:
                final_action, reason, fraction = usd_decision.action, usd_decision.reason, usd_decision.fraction or 0.0
            else:
                final_action, reason, fraction = self._combine_actions(weakness_decision, ladder_decision)

            # Modo observador -> reporta HOLD
            reported_action = final_action if self.mode == "active" else DecisionAction.HOLD

            out.append(PMIDecision(
                ticket=ticket,
                symbol=symbol,
                action=reported_action,
                reason=reason,
                close_score=deb_score,
                fraction=fraction,
                telemetry={
                    "usd_pnl": usd_pnl,
                    "r_pnl": pnl_r,
                    "factors": deb_factors,
                    "ml": ml, "lb90": lb90, "tcd": tcd_prob,
                    "pos_side": side, "sig_side": sig_side,
                    "sr_support": support if support > 0 else None,
                    "sr_resistance": resistance if resistance > 0 else None,
                }
            ))

        return out

    # ---------------------------------------------------------
    # Internals
    # ---------------------------------------------------------
    def _load_pmi_config(self, path: str) -> Dict[str, Any]:
        defaults = {"mode": "active", "thresholds": {}, "profit_targets": {}}
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
        d = {
            "ladder": {"be_at": 0.30, "tighten_at": 0.60, "partial1_at": 1.00, "partial2_at": 1.50, "partial_fraction": 0.50},
            "weak_score": {"tighten": 0.55, "partial": 0.70, "close": 0.85},
            "tcd": {"tighten": 0.55, "close": 0.70},
            "opp_partial_ml": 0.55, "opp_partial_lb90": 0.50,
            "opp_close_ml": 0.58, "opp_close_lb90": 0.53
        }
        try:
            u = dict(user_thr or {})
            for k in ("ladder", "weak_score", "tcd"):
                if k in u and isinstance(u[k], dict):
                    d[k].update(u[k])
            for k in ("opp_partial_ml", "opp_partial_lb90", "opp_close_ml", "opp_close_lb90"):
                if k in u: d[k] = float(u[k])
        except Exception:
            pass
        return d

    def _merge_usd_targets(self, user: Dict[str, Any]) -> Dict[str, Any]:
        d = {"usd_partial": 300.0, "usd_close": 500.0, "partial_fraction": 0.50}
        try:
            if isinstance(user, dict):
                for k in d.keys():
                    if k in user:
                        d[k] = float(user[k])
        except Exception:
            pass
        return d

    def _load_daily_sentiment(self, sentiment_path: Optional[str]) -> Dict[str, Dict[str, float]]:
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

    # --------- Cálculos de score / escalera / señal opuesta ----------
    def _weakness_score(self, symbol, side, close, atr, tcd_prob, lb90, ml, sig_side,
                        support, resistance, df):
        weights = {"tcd": 0.40, "slope": 0.20, "atr_contract": 0.15, "rsi_exit": 0.15, "sr_near": 0.10}
        f = {"tcd": 0.0, "slope": 0.0, "atr_contract": 0.0, "rsi_exit": 0.0, "sr_near": 0.0}

        f["tcd"] = float(max(0.0, min(1.0, tcd_prob))) if tcd_prob == tcd_prob else 0.0

        if pd is not None and isinstance(df, pd.DataFrame) and len(df) >= 25 and "close" in df.columns:
            try:
                ema = df["close"].ewm(span=20, adjust=False).mean()
                slope = float(ema.iloc[-1] - ema.iloc[-5])
                f["slope"] = 1.0 if (side == "BUY" and slope <= 0) or (side == "SELL" and slope >= 0) else 0.0
            except Exception:
                pass
            try:
                if "high" in df.columns and "low" in df.columns:
                    tr = (df["high"] - df["low"]).abs()
                    atr20 = tr.rolling(20).mean().iloc[-1]
                    atr20_prev = tr.rolling(20).mean().iloc[-6]
                    change = (atr20 - atr20_prev) / max(1e-9, abs(atr20_prev))
                    f["atr_contract"] = 1.0 if change < -0.05 else (0.5 if change < 0 else 0.0)
            except Exception:
                pass
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
            return PMIDecision(0, "", DecisionAction.PARTIAL_CLOSE, "ladder_partial2", fraction=pf)
        if pnl_r >= p1_at:
            return PMIDecision(0, "", DecisionAction.PARTIAL_CLOSE, "ladder_partial1", fraction=pf)
        if pnl_r >= tighten_at:
            return PMIDecision(0, "", DecisionAction.TIGHTEN_SL, "ladder_tighten")
        if pnl_r >= be_at:
            return PMIDecision(0, "", DecisionAction.TIGHTEN_SL, "ladder_break_even")
        return None

    def _weakness_decision(self, score: float) -> PMIDecision | None:
        thr = self.close_thresholds.get("weak_score", {})
        t_tight = float(thr.get("tighten", 0.55))
        t_part = float(thr.get("partial", 0.70))
        t_close = float(thr.get("close", 0.85))
        if score >= t_close:
            return PMIDecision(0, "", DecisionAction.CLOSE, "weak_score_close", close_score=score)
        if score >= t_part:
            pf = float(self.close_thresholds.get("ladder", {}).get("partial_fraction", 0.50))
            return PMIDecision(0, "", DecisionAction.PARTIAL_CLOSE, "weak_score_partial", close_score=score, fraction=pf)
        if score >= t_tight:
            return PMIDecision(0, "", DecisionAction.TIGHTEN_SL, "weak_score_tighten", close_score=score)
        return None

    def _opposite_signal_decision(self, pos_side: str, sig_side: str, ml: float, lb90: float):
        if sig_side not in ("BUY", "SELL") or pos_side not in ("BUY", "SELL") or sig_side == pos_side:
            return None, ""
        thr = self.close_thresholds
        if ml >= float(thr.get("opp_close_ml", 0.58)) and lb90 >= float(thr.get("opp_close_lb90", 0.53)):
            return DecisionAction.CLOSE, "opposite_signal_strong"
        if ml >= float(thr.get("opp_partial_ml", 0.55)) and lb90 >= float(thr.get("opp_partial_lb90", 0.50)):
            return DecisionAction.PARTIAL_CLOSE, "opposite_signal_medium"
        return None, ""

    def _combine_actions(self, a: Optional[PMIDecision], b: Optional[PMIDecision]):
        order = {DecisionAction.CLOSE: 3, DecisionAction.PARTIAL_CLOSE: 2,
                 DecisionAction.TIGHTEN_SL: 1, DecisionAction.HOLD: 0}
        cand = [x for x in (a, b) if x is not None]
        if not cand:
            return DecisionAction.HOLD, "hold", 0.0
        best = max(cand, key=lambda x: order.get(x.action, 0))
        return best.action, best.reason, best.fraction or 0.0

    # ---------------- USD Targets ----------------
    def _usd_target_decision(self, usd_pnl: float) -> Optional[PMIDecision]:
        try:
            up = float(self.usd_targets.get("usd_partial", 0))
            uc = float(self.usd_targets.get("usd_close", 0))
            pf = float(self.usd_targets.get("partial_fraction", 0.50))
        except Exception:
            up, uc, pf = 0.0, 0.0, 0.50

        if uc > 0 and usd_pnl >= uc:
            return PMIDecision(0, "", DecisionAction.CLOSE, f"usd_close_{uc:.0f}")
        if up > 0 and usd_pnl >= up:
            return PMIDecision(0, "", DecisionAction.PARTIAL_CLOSE, f"usd_partial_{up:.0f}", fraction=pf)
        return None

    def _estimate_unrealized_usd(
        self, symbol: str, side: str, price_open: float, close: float,
        volume: float, contract_size: float, point: float, usd_pppl: float
    ) -> float:
        """
        Estimación simple de PnL en USD:
        - Si viene usd_per_pip_per_lot en snapshot → usarlo (lo ideal).
        - Si no, usar fórmulas estándar por par mayor:
            * EURUSD/GBPUSD/AUDUSD: USD_PnL = (close - open) * 100000 * volume  (BUY; invertir signo para SELL)
            * USDJPY: USD_PnL ≈ ((close - open) * 100000 * volume) / close      (BUY; invertir signo para SELL)
        """
        try:
            sign = 1.0 if side == "BUY" else -1.0
            delta = (close - price_open) * sign  # ya con el lado aplicado

            if usd_pppl and point:
                # si tenemos USD por pip por lote y el tamaño de punto
                pips = delta / max(point, 1e-9)
                return float(pips * usd_pppl * volume)

            sym = symbol.upper()
            if sym.endswith("USD") and len(sym) == 6:  # XXXUSD (EURUSD, GBPUSD, AUDUSD)
                return float(delta * 100000.0 * volume)
            if sym == "USDJPY":
                # PnL en JPY = delta * 100000 * volume ; a USD ≈ / close
                jpy_pnl = delta * 100000.0 * volume
                return float(jpy_pnl / max(close, 1e-9))
            # fallback genérico con contract_size (si vino)
            return float(delta * max(contract_size, 100000.0) * volume)
        except Exception:
            return 0.0

    def _log(self, msg: str) -> None:
        try:
            if self.logger: self.logger.info(msg)
            else: print(msg)
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
