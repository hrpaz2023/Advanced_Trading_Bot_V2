# src/tools/pmi_smoketest.py
import datetime as dt
from pmi.smart_position_manager import SmartPositionManager

def main():
    now = dt.datetime.now(dt.timezone.utc)

    # Instanciamos PMI en modo activo (usa tus umbrales por defecto)
    pmi = SmartPositionManager(mode="active")

    # Posiciones ficticias (no usa MT5)
    positions = [
        {"ticket": 111, "symbol": "EURUSD", "type": "BUY",  "volume": 2.0, "price_open": 1.12345},
        {"ticket": 222, "symbol": "USDJPY", "type": "SELL", "volume": 2.0, "price_open": 147.200},
        {"ticket": 333, "symbol": "AUDUSD", "type": "BUY",  "volume": 2.0, "price_open": 0.65200},
    ]

    # Snapshot simple (close/atr_rel) – suficiente para el combine() y factores
    market_snapshot = {
        "EURUSD": {"close": 1.12400, "atr_rel": 0.010},
        "USDJPY": {"close": 147.000, "atr_rel": 0.020},
        "AUDUSD": {"close": 0.65180, "atr_rel": 0.015},
    }

    # Contexto de señales por símbolo:
    # - EURUSD: señal opuesta fuerte → debería disparar CLOSE
    # - USDJPY: señal opuesta media → debería disparar PARTIAL
    # - AUDUSD: TCD alto (>= tcd_close 0.70) → debería disparar CLOSE por TCD
    signal_context_by_symbol = {
        "EURUSD": {"signal_side": "SELL", "ml_confidence": 0.66, "historical_prob_lb90": 0.58, "tcd_prob": 0.40},
        "USDJPY": {"signal_side": "BUY",  "ml_confidence": 0.59, "historical_prob_lb90": 0.53, "tcd_prob": 0.40},
        "AUDUSD": {"signal_side": "SELL", "ml_confidence": 0.40, "historical_prob_lb90": 0.40, "tcd_prob": 0.72},
    }

    # Ejecutamos evaluación: no envía órdenes, solo devuelve decisiones
    decisions = pmi.evaluate(
        positions=positions,
        market_snapshot=market_snapshot,
        candles_by_symbol=None,
        now=now,
        signal_context_by_symbol=signal_context_by_symbol,
    )

    print("\n=== PMI SMOKE-TEST ===")
    for d in decisions:
        # d es un PMIDecision; tratamos de imprimir bonito
        ticket = getattr(d, "ticket", None) or d.get("ticket")
        action = str(getattr(d, "action", None) or d.get("action"))
        reason = getattr(d, "reason", "") or d.get("reason", "")
        close_score = getattr(d, "close_score", None) or d.get("close_score", 0.0)
        telemetry = getattr(d, "telemetry", {}) or d.get("telemetry", {})
        print(f"ticket={ticket} action={action} reason={reason} close_score={close_score:.3f} meta={telemetry}")

if __name__ == "__main__":
    main()
