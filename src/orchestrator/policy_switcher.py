# src/orchestrator/policy_switcher.py
# -----------------------------------------------------------------------------
# Orquestador de señales: thresholds, sizing, límites de exposición/correlación,
# sesiones (robusto si faltan start_utc/end_utc) y registro básico de posiciones.
# Maneja global_insights.json como dict, list o None sin romper.
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import datetime
import pytz
from math import isfinite


@dataclass
class OpenPosition:
    ticket: Any
    symbol: str
    side: str
    lots: float
    opened_at_utc: datetime.datetime


class PolicySwitcher:
    def __init__(self,
                 config_path: str = "orchestrator_config.json",
                 global_insights_path: str = "reports/global_insights.json"):
        self.cfg = self._load_json(config_path, "orchestrator_config") or {}

        # Cargar insights en bruto (puede ser dict, list o None)
        self.insights_raw = self._load_json(global_insights_path, "global_insights", required=False)

        # --- Thresholds con defaults seguros ---
        t = (self.cfg.get("thresholds") or {})
        self.t = {
            "ml_confidence_min": float(t.get("ml_confidence_min", 0.40)),
            "historical_prob_min": float(t.get("historical_prob_min", 0.40)),
            "orchestrator_score_min": float(t.get("orchestrator_score_min", 0.00)),
            "max_spread_pips": float(t.get("max_spread_pips", 3.0)),
            "max_atr_multiple": float(t.get("max_atr_multiple", 4.0)),
            "dd_guard": float(t.get("dd_guard", -2000.0)),
            "pf_guard": float(t.get("pf_guard", 1.10)),
        }

        # --- Sizing ---
        s = (self.cfg.get("sizing") or {})
        self.risk_per_trade_pct = float(s.get("risk_per_trade_pct", 1.0))
        self.base_lots = float(s.get("base_lots", 0.50))
        self.atr_scale = (s.get("atr_scale") or {"M5": 1.0})

        # --- Exposición y correlación ---
        e = (self.cfg.get("exposure_limits") or {})
        self.max_total_positions = int(e.get("max_total_positions", 6))
        self.max_symbol_positions = int(e.get("max_symbol_positions", 2))
        self.max_direction_net_lots = float(e.get("max_direction_net_lots", 2.0))

        c = (self.cfg.get("correlation_caps") or {})
        self.corr_window_bars = int(c.get("window_bars", 288))
        self.cap_if_rho_gt = float(c.get("cap_if_rho_gt", 0.85))
        self.max_simultaneous_same_direction = int(c.get("max_simultaneous_same_direction", 2))

        # --- Sesiones ---
        self.tz_utc = pytz.UTC
        self.session_windows = self._build_session_windows(self.cfg.get("sessions", {}))

        # Registro simple de posiciones
        self.open_positions: Dict[Any, OpenPosition] = {}

    # -------------------------------------------------------------------------
    # Carga JSON
    # -------------------------------------------------------------------------
    def _load_json(self, path: str, name: str, required: bool = True) -> Optional[dict]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            if required:
                raise
            return None

    # -------------------------------------------------------------------------
    # Sesiones robustas (no exige start_utc/end_utc en JSON)
    # -------------------------------------------------------------------------
    def _build_session_windows(self, sessions_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Construye ventanas de sesión robustas. Si faltan start_utc/end_utc en el JSON,
        aplica valores por defecto (UTC) y respeta 'priority' si viene.
        """
        windows: List[Dict[str, Any]] = []

        def _add(name, start, end, priority):
            windows.append({
                "name": name,
                "start_utc": int(start),
                "end_utc": int(end),
                "priority": int(priority)
            })

        # ¿Vienen horas explícitas?
        has_explicit = False
        for name, cfg in (sessions_cfg or {}).items():
            if isinstance(cfg, dict) and ("start_utc" in cfg and "end_utc" in cfg):
                has_explicit = True
                _add(name, cfg["start_utc"], cfg["end_utc"], cfg.get("priority", 1))

        if not has_explicit:
            # Defecto (UTC): Asia:0–7, Londres:7–16, NY:12–21, Off:21–24
            _add("asia",    0, 7,  (sessions_cfg.get("asia", {}) or {}).get("priority", 2))
            _add("london",  7, 16, (sessions_cfg.get("london", {}) or {}).get("priority", 3))
            _add("newyork", 12, 21,(sessions_cfg.get("newyork", {}) or {}).get("priority", 3))
            _add("off",     21, 24,(sessions_cfg.get("off", {}) or {}).get("priority", 1))
        return windows

    def _get_session(self, now_utc: datetime.datetime) -> str:
        """Devuelve la sesión actual con prioridad en overlaps. Maneja ventanas que cruzan medianoche."""
        if now_utc.tzinfo is None:
            now_utc = self.tz_utc.localize(now_utc)
        h = now_utc.hour

        current = None
        best_prio = -1
        for w in self.session_windows:
            start_h = w["start_utc"]
            end_h = w["end_utc"]
            if start_h <= end_h:
                in_window = (start_h <= h < end_h)
            else:
                # cruza medianoche (ej. 22-03)
                in_window = (h >= start_h or h < end_h)
            if in_window and w.get("priority", 1) > best_prio:
                best_prio = w["priority"]
                current = w["name"]

        if current is None:
            off = next((w["name"] for w in self.session_windows if w["name"].lower() == "off"), None)
            return off or (self.session_windows[0]["name"] if self.session_windows else "off")
        return current

    # -------------------------------------------------------------------------
    # Helpers de insights
    # -------------------------------------------------------------------------
    def _get_account_metrics(self) -> Dict[str, Any]:
        """
        Devuelve un dict con métricas de cuenta desde global_insights.*.
        Tolera dict, list o None.
        """
        data = self.insights_raw
        if isinstance(data, dict):
            return data.get("account_metrics", {}) or {}
        if isinstance(data, list):
            # buscar el primer item que contenga "account_metrics"
            for item in data:
                if isinstance(item, dict) and "account_metrics" in item:
                    return item.get("account_metrics", {}) or {}
        return {}

    # -------------------------------------------------------------------------
    # API pública: aprobación y registro de posiciones
    # -------------------------------------------------------------------------
    def approve_signal(self,
                       symbol: str,
                       strategy: str,
                       action: str,
                       payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide si aprobar la señal y calcula tamaño. 'payload' esperado:
            atr: float
            confidence: float (0..1)
            historical_prob: float (0..1)
            base_lots: float
            now_utc: datetime
            spread_pips: opcional
            orchestrator_score: opcional
        """
        now_utc = payload.get("now_utc") or datetime.datetime.now(tz=self.tz_utc)
        session = self._get_session(now_utc)

        # --- Gates básicos ---
        conf = float(payload.get("confidence", 1.0) or 0.0)
        hist = float(payload.get("historical_prob", 1.0) or 0.0)
        orch = float(payload.get("orchestrator_score", 0.0) or 0.0)
        spread = payload.get("spread_pips", None)
        atr = float(payload.get("atr", 0.0) or 0.0)

        reasons: List[str] = []
        approved = True

        if conf < self.t["ml_confidence_min"]:
            approved = False
            reasons.append(f"Confianza ML {conf:.2%} < {self.t['ml_confidence_min']:.2%}")

        if hist < self.t["historical_prob_min"]:
            approved = False
            reasons.append(f"Prob. histórica {hist:.2%} < {self.t['historical_prob_min']:.2%}")

        if orch < self.t["orchestrator_score_min"]:
            approved = False
            reasons.append(f"Puntaje orquestador {orch:.2f} < {self.t['orchestrator_score_min']:.2f}")

        if spread is not None:
            try:
                if float(spread) > self.t["max_spread_pips"]:
                    approved = False
                    reasons.append(f"Spread {float(spread):.2f}p > {self.t['max_spread_pips']:.2f}p")
            except Exception:
                pass

        # --- Guardas globales (si existen en insights) ---
        acct = self._get_account_metrics()
        dd = acct.get("drawdown", None)
        pf = acct.get("profit_factor", None)
        if dd is not None and isfinite(dd) and dd < self.t["dd_guard"]:
            approved = False
            reasons.append(f"DD guard: {dd:.0f} < {self.t['dd_guard']:.0f}")
        if pf is not None and isfinite(pf) and pf < self.t["pf_guard"]:
            approved = False
            reasons.append(f"PF guard: {pf:.2f} < {self.t['pf_guard']:.2f}")

        # --- Exposición simple por símbolo/dirección ---
        dir_key = f"{symbol}:{action.upper()}"
        same_dir = sum(1 for op in self.open_positions.values() if f"{op.symbol}:{op.side.upper()}" == dir_key)
        if same_dir >= self.max_simultaneous_same_direction:
            approved = False
            reasons.append(f"Exposición por dirección alcanzada ({same_dir} >= {self.max_simultaneous_same_direction})")

        # --- Sizing (base_lots escalado por ATR y sesión) ---
        lots = float(payload.get("base_lots", self.base_lots))
        tf_scale = self.atr_scale.get("M5", 1.0)
        if atr > 0:
            lots = max(0.01, round(lots * tf_scale / max(1.0, atr), 2))  # escala por volatilidad
        else:
            lots = max(0.01, round(lots * tf_scale, 2))

        return {
            "approved": approved,
            "position_size": lots if approved else 0.0,
            "reason": " & ".join(reasons) if reasons else "Criterios cumplidos",
            "session": session
        }

    def register_open(self, ticket: Any, symbol: str, side: str, lots: float):
        """Registro de posición abierta simple (para límites de exposición)."""
        try:
            self.open_positions[ticket] = OpenPosition(
                ticket=ticket,
                symbol=symbol,
                side=side,
                lots=float(lots),
                opened_at_utc=datetime.datetime.now(tz=self.tz_utc),
            )
        except Exception:
            pass

    def register_close(self, ticket: Any):
        """Liberar posición al cierre."""
        self.open_positions.pop(ticket, None)
