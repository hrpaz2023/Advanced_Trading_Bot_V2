# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd  # noqa: F401
except Exception:  # pandas opcional en runtime
    pd = None  # type: ignore

from pmi.enums import DecisionAction
from pmi.decision import PMIDecision


# =========================
# Helpers numéricos
# =========================
def _num(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return default
        return float(s)
    except Exception:
        return default


def _pnl_r(side: str, price_open: float, price_close: float, atr_abs: float) -> float:
    """PnL expresado en múltiplos de ATR (R)."""
    if atr_abs <= 0.0:
        return 0.0
    diff = (price_close - price_open) if side == "BUY" else (price_open - price_close)
    return diff / atr_abs


# =========================
# SmartPositionManager
# =========================
class SmartPositionManager:
    """
    Evalúa posiciones abiertas y propone acciones (HOLD, TIGHTEN_SL, PARTIAL_CLOSE, CLOSE)
    combinando:
      - Señal opuesta (y su confianza)
      - Objetivos por USD
      - Break-even por debilidad (NUEVO)
      - Límites por tiempo / rendimiento
      - Debilidad y 'ladder' por R
      - Soportes / resistencias diarios
    """

    def __init__(
        self,
        mode: str = "observer",  # "observer" | "active"
        thresholds: Optional[Dict[str, Any]] = None,
        usd_targets: Optional[Dict[str, Any]] = None,
        hold_policy: Optional[Dict[str, Any]] = None,
        sr_levels: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.mode = (mode or "observer").lower()
        self.thresholds = self._merge_thresholds(thresholds or {})
        self.usd_targets = self._merge_usd_targets(usd_targets or {})
        self.hold_policy = self._merge_hold_policy(hold_policy or {})
        self.sr_levels = sr_levels or {}

    # ---------- carga desde config ----------
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "SmartPositionManager":
        return cls(
            mode=cfg.get("mode", "observer"),
            thresholds=cfg.get("thresholds", {}),
            usd_targets=cfg.get("usd_targets", {}),
            hold_policy=cfg.get("hold_policy", {}),
            sr_levels=cfg.get("sr_levels", {}),
        )

    # ---------- merges/config por defecto ----------
    def _merge_thresholds(self, user: Dict[str, Any]) -> Dict[str, Any]:
        d = {
            # Opposite signal (fuerte / medio)
            "opp_strong_ml": 0.65,
            "opp_strong_lb90": 0.55,
            "opp_medium_ml": 0.55,
            "opp_medium_lb90": 0.45,
            # Debilidad
            "weak_close": 0.70,
            "weak_partial": 0.60,
            # Ladder por R
            "ladder_partial_r": 1.00,
            "ladder_close_r": 1.50,
        }
        try:
            for k, v in user.items():
                d[k] = float(v)
        except Exception:
            pass
        return d

    def _merge_usd_targets(self, user: Dict[str, Any]) -> Dict[str, Any]:
        d = {
            # Objetivos USD crecientes; último suele cerrar todo
            "targets": [150.0, 350.0, 600.0],
            "fractions": [0.50, 0.50, 1.00],  # 50% + 50% + 100% (si queda)
        }
        try:
            if "targets" in user and isinstance(user["targets"], list):
                d["targets"] = [float(x) for x in user["targets"]]
            if "fractions" in user and isinstance(user["fractions"], list):
                d["fractions"] = [float(x) for x in user["fractions"]]
        except Exception:
            pass
        return d

    def _merge_hold_policy(self, user: Dict[str, Any]) -> Dict[str, Any]:
        # Defaults conservadores para M5
        d = {
            "grace_minutes": 90.0,
            "min_r_after_grace": 0.20,
            "max_hours": 12.0,
            "min_r_after_max": 0.30,
            "partial_fraction": 0.50,

            # --- NUEVO: Break-even por debilidad ---
            "breakeven_enabled": True,
            "breakeven_after_minutes": 60.0,
            "breakeven_weak_threshold": 0.55,
            "commission_per_lot": 3.0,  # tu esquema: comisión = volumen * 3 USD
            "breakeven_extra_buffer": 0.0,
        }
        try:
            for k in list(d.keys()):
                if k in user:
                    if isinstance(d[k], bool):
                        d[k] = bool(user[k])
                    else:
                        d[k] = float(user[k])
        except Exception:
            pass
        return d

    # =========================
    # Núcleo de evaluación
    # =========================
    def evaluate(
        self,
        positions: List[Dict[str, Any]],
        market_snapshot: Dict[str, Dict[str, float]],
        candles_by_symbol: Optional[Dict[str, "pd.DataFrame"]] = None,
        now: Optional[dt.datetime] = None,
        signal_context_by_symbol: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[PMIDecision]:
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

            # ---------- snapshot ----------
            snap = (market_snapshot or {}).get(symbol, {}) or {}
            close = _num(snap.get("close"))
            atr_abs = _num(snap.get("atr"))
            atr_rel = _num(snap.get("atr_rel"))
            if atr_abs <= 0 and atr_rel > 0 and close > 0:
                atr_abs = close * atr_rel

            if close <= 0 or atr_abs <= 0:
                out.append(PMIDecision(
                    ticket=ticket,
                    action=DecisionAction.HOLD,
                    reason="insufficient_snapshot",
                    telemetry={"have_close": close > 0, "have_atr": atr_abs > 0, "have_open": price_open > 0}
                ))
                continue

            # ---------- tiempo en la posición ----------
            opened_at = self._parse_open_time(pos)
            hours_open = (now - opened_at).total_seconds() / 3600.0 if opened_at else 0.0

            # ---------- PnL en R y USD ----------
            pnl_r = _pnl_r(side, price_open, close, atr_abs)
            usd_pnl = self._estimate_unrealized_usd(
                symbol=symbol, side=side, price_open=price_open, close=close,
                volume=vol,
                contract_size=_num(snap.get("contract_size"), 100_000.0),
                point=_num(snap.get("point")),
                usd_pppl=_num(snap.get("usd_per_pip_per_lot")),
            )

            # ---------- contexto de señal ----------
            ctx = (signal_context_by_symbol or {}).get(symbol, {}) if signal_context_by_symbol else {}
            ml = _num(ctx.get("ml_confidence"))
            lb90 = _num(ctx.get("historical_prob_lb90"))
            tcd_prob = _num(ctx.get("tcd_prob"))
            sig_side = (ctx.get("signal_side") or "").upper()

            # ---------- S/R diarios ----------
            sr = (self.sr_levels.get(symbol) if self.sr_levels else {}) or {}
            support = _num(sr.get("support"))
            resistance = _num(sr.get("resistance"))

            # ---------- debilidad ----------
            df_sym = (candles_by_symbol or {}).get(symbol) if candles_by_symbol else None
            weak_score, weak_factors = self._weakness_score(
                symbol=symbol, side=side, close=close, atr=atr_abs,
                tcd_prob=tcd_prob, lb90=lb90, ml=ml, sig_side=sig_side,
                support=support if support > 0 else None,
                resistance=resistance if resistance > 0 else None,
                df=df_sym,
            )

            # ---------- reglas individuales ----------
            usd_dec = self._usd_target_decision(usd_pnl)
            ladder_dec = self._ladder_decision(pnl_r)
            weak_dec = self._weakness_decision(weak_score)
            time_dec = self._time_based_decision(hours_open, pnl_r, weak_score)
            be_dec = self._breakeven_on_weakness_decision(hours_open, weak_score, usd_pnl, vol)
            opp_action, opp_reason = self._opposite_signal_decision(side, sig_side, ml, lb90)

            # ---------- consolidación (prioridades) ----------
            # 1) Señal opuesta fuerte
            # 2) Objetivos en USD
            # 3) Break-even por debilidad (NUEVO)
            # 4) Límite por tiempo
            # 5) Debilidad / Ladder por R
            final_action, reason, fraction = DecisionAction.HOLD, "hold", 0.0
            if opp_action is not None:
                final_action, reason = opp_action, opp_reason
            elif usd_dec is not None:
                final_action, reason, fraction = usd_dec.action, usd_dec.reason, usd_dec.fraction or 0.0
            elif be_dec is not None:
                final_action, reason, fraction = be_dec.action, be_dec.reason, be_dec.fraction or 0.0
            elif time_dec is not None:
                final_action, reason, fraction = time_dec.action, time_dec.reason, time_dec.fraction or 0.0
            else:
                final_action, reason, fraction = self._combine_actions(weak_dec, ladder_dec)

            out.append(PMIDecision(
                ticket=ticket,
                action=final_action,  # Registrar siempre la acción calculada
                reason=reason,
                confidence=weak_score,  # Using weak_score as confidence
                close_score=weak_score,
                telemetry={
                    "symbol": symbol,
                    "hours_open": round(hours_open, 3),
                    "usd_pnl": usd_pnl,
                    "r_pnl": pnl_r,
                    "factors": weak_factors,
                    "ml": ml, "lb90": lb90, "tcd": tcd_prob,
                    "pos_side": side, "sig_side": sig_side,
                    "sr_support": support if support > 0 else None,
                    "sr_resistance": resistance if resistance > 0 else None,
                    "fraction": fraction,
                }
            ))

        return out

    # =========================
    # Helper class for internal decision tracking
    # =========================
    class _InternalDecision:
        def __init__(self, action: DecisionAction, reason: str, fraction: float = 0.0):
            self.action = action
            self.reason = reason
            self.fraction = fraction

    # =========================
    # Reglas
    # =========================
    def _usd_target_decision(self, usd_pnl: float) -> Optional['SmartPositionManager._InternalDecision']:
        tgts = [float(x) for x in self.usd_targets.get("targets", [])]
        fracs = [float(x) for x in self.usd_targets.get("fractions", [])]
        if not tgts or not fracs:
            return None
        # el primer target que supere → aplica su fracción; si es el último y fracción >= 1 => CLOSE
        for i, t in enumerate(tgts):
            if usd_pnl >= t:
                frac = fracs[i] if i < len(fracs) else 1.0
                action = DecisionAction.CLOSE if frac >= 1.0 or i == len(tgts) - 1 else DecisionAction.PARTIAL_CLOSE
                return self._InternalDecision(action=action, reason=f"usd_target_{t:g}", fraction=frac)
        return None

    def _ladder_decision(self, r: float) -> Optional['SmartPositionManager._InternalDecision']:
        close_r = float(self.thresholds.get("ladder_close_r", 1.5))
        part_r = float(self.thresholds.get("ladder_partial_r", 1.0))
        if r >= close_r:
            return self._InternalDecision(action=DecisionAction.CLOSE, reason="ladder_close")
        if r >= part_r:
            return self._InternalDecision(action=DecisionAction.PARTIAL_CLOSE, reason="ladder_partial", fraction=0.5)
        return None

    def _weakness_decision(self, w: float) -> Optional['SmartPositionManager._InternalDecision']:
        if w >= float(self.thresholds.get("weak_close", 0.70)):
            return self._InternalDecision(action=DecisionAction.CLOSE, reason="weakness_strong")
        if w >= float(self.thresholds.get("weak_partial", 0.60)):
            return self._InternalDecision(action=DecisionAction.PARTIAL_CLOSE, reason="weakness_medium", fraction=0.5)
        return None

    def _time_based_decision(self, hours_open: float, r: float, w: float) -> Optional['SmartPositionManager._InternalDecision']:
        hp = self.hold_policy or {}
        grace_h = float(hp.get("grace_minutes", 90.0)) / 60.0
        max_h = float(hp.get("max_hours", 12.0))
        min_r_grace = float(hp.get("min_r_after_grace", 0.20))
        min_r_max = float(hp.get("min_r_after_max", 0.30))
        part_frac = float(hp.get("partial_fraction", 0.50))

        if hours_open >= max_h and r < min_r_max:
            return self._InternalDecision(action=DecisionAction.CLOSE, reason="time_limit_max_hours")

        if hours_open >= grace_h and r < min_r_grace and w >= float(self.thresholds.get("weak_partial", 0.60)):
            return self._InternalDecision(action=DecisionAction.PARTIAL_CLOSE, reason="time_grace_underperf", fraction=part_frac)

        return None

    def _breakeven_on_weakness_decision(
        self,
        hours_open: float,
        weak_score: float,
        usd_pnl: float,
        volume: float
    ) -> Optional['SmartPositionManager._InternalDecision']:
        hp = self.hold_policy or {}
        if not bool(hp.get("breakeven_enabled", True)):
            return None

        after_h = float(hp.get("breakeven_after_minutes", 60.0)) / 60.0
        weak_thr = float(hp.get("breakeven_weak_threshold", 0.55))
        c_per_lot = float(hp.get("commission_per_lot", 3.0))
        extra = float(hp.get("breakeven_extra_buffer", 0.0))

        needed_usd = max(0.0, (volume * c_per_lot) + extra)

        if hours_open >= after_h and weak_score >= weak_thr and usd_pnl >= needed_usd:
            return self._InternalDecision(
                action=DecisionAction.CLOSE,
                reason="breakeven_on_weakness"
            )
        return None

    def _opposite_signal_decision(
        self, pos_side: str, sig_side: str, ml: float, lb90: float
    ) -> Tuple[Optional[DecisionAction], str]:
        if sig_side not in ("BUY", "SELL") or pos_side not in ("BUY", "SELL"):
            return None, ""
        if sig_side == pos_side:
            return None, ""

        ml_s, lb_s = float(self.thresholds.get("opp_strong_ml", 0.65)), float(self.thresholds.get("opp_strong_lb90", 0.55))
        ml_m, lb_m = float(self.thresholds.get("opp_medium_ml", 0.55)), float(self.thresholds.get("opp_medium_lb90", 0.45))

        if ml >= ml_s and lb90 >= lb_s:
            return DecisionAction.CLOSE, "opposite_signal_strong"
        if ml >= ml_m and lb90 >= lb_m:
            return DecisionAction.PARTIAL_CLOSE, "opposite_signal_medium"
        return None, ""

    def _combine_actions(
        self,
        a: Optional['SmartPositionManager._InternalDecision'],
        b: Optional['SmartPositionManager._InternalDecision'],
    ) -> Tuple[DecisionAction, str, float]:
        # ranking de fuerza: CLOSE > PARTIAL > TIGHTEN_SL > HOLD
        rank = {
            DecisionAction.CLOSE: 3,
            DecisionAction.PARTIAL_CLOSE: 2,
            DecisionAction.TIGHTEN_SL: 1,
            DecisionAction.HOLD: 0,
        }

        cand = [x for x in (a, b) if x is not None]
        if not cand:
            return DecisionAction.HOLD, "hold", 0.0
        cand.sort(key=lambda x: rank.get(x.action, 0), reverse=True)
        top = cand[0]
        return top.action, top.reason, (top.fraction or 0.0)

    # =========================
    # Señal de debilidad (score)
    # =========================
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
        support: Optional[float] = None,
        resistance: Optional[float] = None,
        df: Optional["pd.DataFrame"] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Devuelve (score 0..1, factores).
        Heurística simple:
          - tcd_prob grande => +deb.
          - Señal opuesta => +deb (peso por ml*lb90).
          - Proximidad a S/R en contra => +deb.
          - Pendiente EMA corta (si hay df) en contra => +deb.
        """
        # pesos
        w_tcd = 0.40
        w_opp = 0.35
        w_sr = 0.15
        w_ema = 0.10

        # tcd
        s_tcd = max(0.0, min(1.0, tcd_prob))

        # opuesta ponderada por confianza
        opp = 0.0
        if sig_side in ("BUY", "SELL") and sig_side != side:
            conf = max(0.0, min(1.0, (ml * 0.6 + lb90 * 0.4)))  # mezcla
            opp = conf

        # S/R
        sr = 0.0
        if support and resistance and close > 0.0:
            dist_res = abs(resistance - close) / max(1e-9, atr)
            dist_sup = abs(close - support) / max(1e-9, atr)
            if side == "BUY":   # cerca de resistencia = debilidad
                sr = max(0.0, 1.0 - math.tanh(dist_res / 3.0))
            else:               # SELL cerca de soporte
                sr = max(0.0, 1.0 - math.tanh(dist_sup / 3.0))

        # EMA slope simple (si hay datos)
        emaw = 0.0
        if df is not None and hasattr(df, "tail"):
            try:
                tail = df["close"].tail(20).to_list()
                if len(tail) >= 5:
                    m = (tail[-1] - tail[0]) / max(1e-9, 20.0)
                    # normalizamos por ATR
                    norm = m / max(1e-9, atr)
                    if side == "BUY":
                        emaw = max(0.0, min(1.0, -norm))  # pendiente negativa = debilidad
                    else:
                        emaw = max(0.0, min(1.0, norm))   # pendiente positiva = debilidad para SELL
            except Exception:
                pass

        score = (w_tcd * s_tcd) + (w_opp * opp) + (w_sr * sr) + (w_ema * emaw)
        score = max(0.0, min(1.0, score))

        return score, {
            "tcd": s_tcd,
            "corr_signal": opp,
            "volatility": 0.01,      # placeholder estable (conserva compatibilidad con tus logs)
            "time_in_profit": 0.0,   # idem (opcional si luego lo calculas)
            "sr": sr,
            "ema_slope": emaw,
        }

    # =========================
    # Utilidades
    # =========================
    def _parse_open_time(self, pos: Dict[str, Any]) -> Optional[dt.datetime]:
        # MT5: 'time' viene en segundos epoch (UTC)
        t = pos.get("time") or pos.get("open_time") or pos.get("time_msc")
        try:
            if isinstance(t, (int, float)):
                return dt.datetime.fromtimestamp(float(t), tz=dt.timezone.utc)
            # algunas integraciones guardan str ISO
            if isinstance(t, str):
                return dt.datetime.fromisoformat(t.replace("Z", "+00:00"))
        except Exception:
            pass
        return None

    def _estimate_unrealized_usd(
        self,
        symbol: str,
        side: str,
        price_open: float,
        close: float,
        volume: float,
        contract_size: float,
        point: float,
        usd_pppl: float,
    ) -> float:
        """
        Estima PnL no realizado en USD.
        Preferencia:
          1) usd_pppl (USD per pip per lot) si viene
          2) contract_size/point (aprox) como fallback
        """
        if close <= 0 or price_open <= 0 or volume <= 0:
            return 0.0

        diff = (close - price_open) if side == "BUY" else (price_open - close)
        if usd_pppl > 0 and point > 0:
            pips = diff / point
            return pips * usd_pppl * volume

        # fallback grosero
        # valor del tick aproximado:
        tick_val = (contract_size * point) if (contract_size > 0 and point > 0) else 1.0
        pips = diff / max(point, 1e-9)
        return pips * tick_val * volume