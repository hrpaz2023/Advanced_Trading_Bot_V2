
"""
backtest_ab_orchestrator.py
---------------------------
Evalúa el impacto del PolicySwitcher vs un baseline.
Requiere:
  - reports/global_insights.json (desde consolidate_global_insights.py)
  - logs/signals_history.csv o logs/trades_history.csv con columnas mínimas:
        timestamp_utc, symbol, strategy, side, entry_price, atr, ml_confidence, historical_prob, pnl (opcional)
Salida:
  - reports/backtests/ab_summary.json / .csv + un resumen Markdown
"""

import os, json, math, datetime, pytz
import pandas as pd
import numpy as np

from policy_switcher import PolicySwitcher

def _ensure_dirs():
    os.makedirs("reports/backtests", exist_ok=True)

def _load_signals(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No existe {csv_path}. Exporta tus señales/trades históricos.")
    df = pd.read_csv(csv_path)
    # Normalizar columnas
    rename = {
        "confidence": "ml_confidence",
        "historical_prob": "historical_prob",
        "time": "timestamp_utc"
    }
    for k,v in rename.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]
    # Parse time
    if not np.issubdtype(df["timestamp_utc"].dtype, np.datetime64):
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    return df.dropna(subset=["timestamp_utc","symbol","strategy","side","entry_price"])

def simulate(df: pd.DataFrame, switcher: PolicySwitcher, base_lots: float = 0.1):
    equity_base, equity_policy = 10000.0, 10000.0
    curve_base, curve_policy = [], []
    wins_b, wins_p = 0, 0
    trades_b, trades_p = 0, 0
    
    for _, row in df.sort_values("timestamp_utc").iterrows():
        # baseline: siempre opera con base_lots
        trades_b += 1
        pnl = float(row.get("pnl", 0.0) or 0.0)
        equity_base += pnl
        curve_base.append(equity_base)
        
        # policy: decidir
        payload = {
            "atr": float(row.get("atr", 0.0) or 0.0),
            "confidence": float(row.get("ml_confidence", 1.0) or 1.0),
            "historical_prob": float(row.get("historical_prob", 1.0) or 1.0),
            "base_lots": base_lots,
            "now_utc": row["timestamp_utc"].to_pydatetime()
        }
        verdict = switcher.approve_signal(row["symbol"], row["strategy"], row["side"], payload)
        if verdict["approved"] and verdict["position_size"] > 0:
            trades_p += 1
            # Escalar PnL por tamaño relativo (asume linealidad)
            scale = verdict["position_size"] / base_lots if base_lots > 0 else 1.0
            equity_policy += pnl * scale
        curve_policy.append(equity_policy)
        
        if pnl > 0:
            wins_b += 1
            if verdict["approved"] and verdict["position_size"] > 0:
                wins_p += 1
    
    def _metrics(curve, total, wins):
        if not curve:
            return {"trades":0,"wr":0,"ret":0,"dd":0,"sharpe":0}
        ret = (curve[-1] / 10000.0) - 1.0
        wr = wins / total if total>0 else 0.0
        dd = 0.0
        peak = -1e9
        for x in curve:
            peak = max(peak, x)
            dd = max(dd, (peak - x)/peak if peak>0 else 0.0)
        # proxy sharpe: media diaria / std diaria (asumiendo PnL por fila ~día/slot)
        rets = np.diff([10000.0]+curve)/10000.0
        sharpe = (rets.mean() / (rets.std()+1e-9)) * np.sqrt(252) if len(rets)>2 else 0.0
        return {"trades": total, "wr": wr, "ret": ret, "dd": dd, "sharpe": sharpe}
    
    m_base = _metrics(curve_base, trades_b, wins_b)
    m_pol  = _metrics(curve_policy, trades_p, wins_p)
    return m_base, m_pol

def main():
    _ensure_dirs()
    switcher = PolicySwitcher()
    # intenta con signals_history, si no con trades_history
    for candidate in ["logs/signals_history.csv","logs/trades_history.csv"]:
        if os.path.exists(candidate):
            csv_path = candidate
            break
    else:
        raise FileNotFoundError("No se encontró logs/signals_history.csv ni logs/trades_history.csv")
    
    df = _load_signals(csv_path)
    m_b, m_p = simulate(df, switcher, base_lots=0.1)
    
    out = {
        "baseline": m_b,
        "policy": m_p,
        "delta": {
            "ret": m_p["ret"] - m_b["ret"],
            "wr":  m_p["wr"] - m_b["wr"],
            "dd":  m_p["dd"] - m_b["dd"],
            "sharpe": m_p["sharpe"] - m_b["sharpe"]
        }
    }
    with open("reports/backtests/ab_summary.json","w") as f:
        json.dump(out, f, indent=2)
    pd.DataFrame([
        {"metric":"ret","baseline":m_b["ret"],"policy":m_p["ret"]},
        {"metric":"wr","baseline":m_b["wr"],"policy":m_p["wr"]},
        {"metric":"dd","baseline":m_b["dd"],"policy":m_p["dd"]},
        {"metric":"sharpe","baseline":m_b["sharpe"],"policy":m_p["sharpe"]},
        {"metric":"trades","baseline":m_b["trades"],"policy":m_p["trades"]},
    ]).to_csv("reports/backtests/ab_summary.csv", index=False)
    
    md = f"""# A/B Orchestrator Summary
- Baseline: ret={m_b['ret']:.2%}, wr={m_b['wr']:.1%}, dd={m_b['dd']:.1%}, sharpe={m_b['sharpe']:.2f}, trades={m_b['trades']}
- Policy:   ret={m_p['ret']:.2%}, wr={m_p['wr']:.1%}, dd={m_p['dd']:.1%}, sharpe={m_p['sharpe']:.2f}, trades={m_p['trades']}
- Δ: ret={out['delta']['ret']:.2%}, wr={out['delta']['wr']:.1%}, dd={out['delta']['dd']:.1%}, sharpe={out['delta']['sharpe']:.2f}
"""
    with open("reports/backtests/ab_summary.md","w") as f:
        f.write(md)
    print("✅ Backtest A/B completado -> reports/backtests/ab_summary.{json,csv,md}")

if __name__ == "__main__":
    main()
