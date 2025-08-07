
"""
consolidate_global_insights.py
------------------------------
Consolida métricas clave desde múltiples reportes en un único
`reports/global_insights.json` (y CSV) que el PolicySwitcher puede consumir.

Uso:
    python consolidate_global_insights.py --config orchestrator_config.json

Requisitos:
- orchestrator_config.json con las rutas de reports.
    {
      "paths": {
        "cross_correlation_dir": "reports/cross_correlation_reports",
        "loss_reports_dir": "reports/multi_asset_loss_reports",
        "win_reports_dir": "reports/exhaustive_win_reports",
        "advanced_reports_dir": "reports/ultimate_analysis_reports_v2",
        "causal_reports_dir": "reports/causal_reports",
        "output_dir": "reports"
      }
    }
"""

import os
import re
import json
import csv
import glob
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

def _safe_load_json(fp: str):
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _index_key(asset: str, strategy: str) -> str:
    return f"{(asset or '').upper()}::{(strategy or '').lower()}"

def _extract_asset_strategy_from_content(data: Dict[str, Any], fallback_name: str) -> Tuple[str, str]:
    asset = data.get("asset") or data.get("symbol") or data.get("Symbol")
    strategy = data.get("strategy") or data.get("Strategy")
    if (not asset or not strategy) and fallback_name:
        # intentar inferir desde nombre de archivo: EURUSD_multi_filter_scalper_xxx.json
        base = Path(fallback_name).stem
        # separadores comunes
        tokens = re.split(r"[._\- ]+", base)
        # heurística simple: asset = token con letras y tal vez JPY/USD/EUR..., strategy = resto con guiones bajos
        # si encuentra asset típico (mayúscula + 3-6 chars)
        cand_assets = [t for t in tokens if t.isupper() and 3 <= len(t) <= 10]
        if cand_assets:
            asset = asset or cand_assets[0]
        # strategy: primer token minúscula con guiones o tokens unidos desde la 2da posición
        cand_strats = [t for t in tokens if t.islower() or "_" in t]
        if cand_strats:
            strategy = strategy or cand_strats[0]
    return asset or "UNKNOWN", strategy or "UNKNOWN"

def _extract_metrics_generic(data: Dict[str, Any]) -> Dict[str, Any]:
    """Busca métricas relevantes con nombres variados de claves."""
    out = {}
    flat = {}

    def flatten(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                flatten(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                flatten(f"{prefix}[{i}]", v)
        else:
            flat[prefix] = obj

    flatten("", data)

    # mapa de posibles nombres -> métrica canónica
    candidates = {
        "profit_factor": ["profit_factor", "pf", "profitfactor", "metrics.pf", "profit_factor_overall"],
        "win_rate": ["win_rate", "wr", "winrate", "win_rate_overall", "metrics.win_rate"],
        "max_drawdown_pct": ["max_drawdown_pct", "max_dd", "drawdown_pct", "max_drawdown", "metrics.max_drawdown_pct"],
        "ATE": ["ATE", "ate", "ate_proxy", "ATE_proxy"],
        "policy_value": ["policy_value", "policy_value_proxy", "policyvalue"],
        "best_session": ["best_session", "best_session_causal", "timing.best_session", "session.best"],
        "best_volatility_regime": ["best_volatility_regime", "best_vol_regime_causal", "volatility.best_regime"],
        "consistency_score": ["consistency_score", "metrics.consistency", "stability.consistency_score"],
    }

    # tomar la primera coincidencia disponible
    for canon, keys in candidates.items():
        for k in keys:
            # buscar exacto o por sufijo (case-insensitive)
            for fk, fv in flat.items():
                if fk.lower() == k.lower() or fk.lower().endswith("."+k.lower()):
                    out[canon] = fv
                    break
            if canon in out:
                break

    # saneo numéricos
    for k in ("profit_factor","win_rate","max_drawdown_pct","ATE","policy_value","consistency_score"):
        if k in out:
            try:
                out[k] = float(out[k])
            except Exception:
                pass
    return out

def _merge_metrics(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Completa en dst sólo lo que falte; no pisa valores existentes."""
    for k, v in src.items():
        if k not in dst or dst[k] in (None, "", "N/A"):
            dst[k] = v
    return dst

def _collect_from_dir(d: str) -> List[Dict[str, Any]]:
    if not d or not os.path.isdir(d):
        return []
    files = glob.glob(os.path.join(d, "**", "*.json"), recursive=True)
    rows = []
    for fp in files:
        data = _safe_load_json(fp)
        if not isinstance(data, (dict, list)):
            continue
        if isinstance(data, list):
            # algunos reportes guardan lista de objetos; consolidar cada uno
            for item in data:
                if isinstance(item, dict):
                    rows.append((fp, item))
        else:
            rows.append((fp, data))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = _safe_load_json(args.config)
    if not cfg or "paths" not in cfg:
        raise SystemExit(f"Config inválida o sin 'paths': {args.config}")

    paths = cfg["paths"]
    cross_corr_dir   = paths.get("cross_correlation_dir", "")
    loss_reports_dir = paths.get("loss_reports_dir", "")
    win_reports_dir  = paths.get("win_reports_dir", "")
    advanced_dir     = paths.get("advanced_reports_dir", "")
    causal_dir       = paths.get("causal_reports_dir", "")
    output_dir       = paths.get("output_dir", "reports")
    os.makedirs(output_dir, exist_ok=True)

    # Recolectar
    buckets = {
        "cross_corr": _collect_from_dir(cross_corr_dir),
        "loss":       _collect_from_dir(loss_reports_dir),
        "win":        _collect_from_dir(win_reports_dir),
        "advanced":   _collect_from_dir(advanced_dir),
        "causal":     _collect_from_dir(causal_dir),
    }

    # Agregar por (asset,strategy)
    agg: Dict[str, Dict[str, Any]] = {}
    provenance: Dict[str, Dict[str, str]] = {}

    for bucket_name, items in buckets.items():
        for fp, data in items:
            asset, strat = _extract_asset_strategy_from_content(data, fp)
            key = _index_key(asset, strat)
            if key not in agg:
                agg[key] = {"asset": asset.upper(), "strategy": strat}
                provenance[key] = {}
            metrics = _extract_metrics_generic(data)
            _merge_metrics(agg[key], metrics)
            # guardar fuente si aporta algo
            if metrics:
                provenance[key][bucket_name] = fp

    # Filtrar combos vacíos
    rows = [v for v in agg.values() if len(v.keys()) > 2]

    # Guardar JSON
    out_json = os.path.join(output_dir, "global_insights.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"insights": rows, "provenance": provenance}, f, indent=2)
    print(f"✅ Escrito {out_json} (combos={len(rows)})")

    # Guardar CSV plano
    out_csv = os.path.join(output_dir, "global_insights.csv")
    # columnas canónicas
    cols = ["asset","strategy","profit_factor","win_rate","max_drawdown_pct","ATE","policy_value","best_session","best_volatility_regime","consistency_score"]
    # completar vacíos
    for r in rows:
        for c in cols:
            if c not in r:
                r[c] = ""
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"✅ Escrito {out_csv}")

if __name__ == "__main__":
    main()
