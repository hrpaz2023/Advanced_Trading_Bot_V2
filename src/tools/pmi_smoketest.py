# -*- coding: utf-8 -*-
"""
Smoke-test rápido de SmartPositionManager:
- Señal opuesta fuerte -> CLOSE
- Señal opuesta media -> PARTIAL_CLOSE
- Debilidad fuerte por TCD -> CLOSE
- NUEVO: Break-even por debilidad -> CLOSE si cubre comisión
"""
from __future__ import annotations
import datetime as dt

from pmi.smart_position_manager import SmartPositionManager
from pmi.enums import DecisionAction

# --- Config mínima de prueba (puedes cargar desde configs/pmi_config.json en tu entorno) ---
cfg = {
    "mode": "active",
    "thresholds": {
        "opp_strong_ml": 0.65,
        "opp_strong_lb90": 0.55,
        "opp_medium_ml": 0.55,
        "opp_medium_lb90": 0.45,
        "weak_close": 0.70,
        "weak_partial": 0.60,
        "ladder_partial_r": 1.00,
        "ladder_close_r": 1.50,
    },
    "usd_targets": {
        "targets": [150.0, 350.0, 600.0],
        "fractions": [0.50, 0.50, 1.00]
    },
    "hold_policy": {
        "grace_minutes": 90,
        "min_r_after_grace": 0.20,
        "max_hours": 12,
        "min_r_after_max": 0.30,
        "partial_fraction": 0.50,

        # NUEVO
        "breakeven_enabled": True,
        "breakeven_after_minutes": 60,
        "breakeven_weak_threshold": 0.55,
        "commission_per_lot": 3.0,
        "breakeven_extra_buffer": 0.0
    },
    "sr_levels": {
        "EURUSD": {"support": 1.1600, "resistance": 1.1710}
    }
}

spm = SmartPositionManager.from_config(cfg)

# ----- Datos sintéticos -----
now = dt.datetime.now(dt.timezone.utc)

positions = [
    {"ticket": 111, "symbol": "EURUSD", "type": "BUY", "volume": 1.0, "price_open": 1.1600, "time": (now - dt.timedelta(hours=2)).timestamp()},
    {"ticket": 222, "symbol": "EURUSD", "type": "SELL", "volume": 1.0, "price_open": 1.1700, "time": (now - dt.timedelta(hours=1.5)).timestamp()},
    {"ticket": 333, "symbol": "EURUSD", "type": "BUY", "volume": 1.0, "price_open": 1.1650, "time": (now - dt.timedelta(hours=3)).timestamp()},
    {"ticket": 444, "symbol": "EURUSD", "type": "BUY", "volume": 2.0, "price_open": 1.1650, "time": (now - dt.timedelta(hours=1.2)).timestamp()},  # BE case
]

snapshot = {
    "EURUSD": {
        "close": 1.1656,
        "atr": 0.0006,
        "contract_size": 100000,
        "point": 0.0001,
        "usd_per_pip_per_lot": 10.0,
    }
}

# contexto de señales
ctx = {
    "EURUSD": {
        "signal_side": "SELL",         # opuesta a BUY
        "ml_confidence": 0.66,
        "historical_prob_lb90": 0.58,
        "tcd_prob": 0.40
    }
}

decisions = spm.evaluate(positions, snapshot, signal_context_by_symbol=ctx, now=now)

for d in decisions:
    print(f"ticket={d.ticket} action={d.action} reason={d.reason} close_score={round(d.close_score or 0.0, 3)} meta={d.telemetry}")

# Esperado:
# - 111 -> CLOSE (opposite strong)
# - 222 -> PARTIAL (opposite medium) [en este set podría salir HOLD si no cruza umbrales, es sintético]
# - 333 -> CLOSE (weakness alta simulada por S/R + TCD + opposite)
# - 444 -> CLOSE (breakeven_on_weakness) si el usd_pnl >= comisión (2 lotes * 3 USD = 6 USD)
