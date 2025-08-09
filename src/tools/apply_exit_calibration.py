# -*- coding: utf-8 -*-
"""
Aplica la calibración de salidas (Monte Carlo) a configs/pmi_config.json.

Uso típico:
  python src/tools/apply_exit_calibration.py ^
    --summary reports/exit_20250809/exit_mc_summary.json ^
    --out configs/pmi_config.json --set-mode active

Opcionales:
  --reports-dir reports        # si no pasás --summary, buscará el más reciente en reports/exit_*/
  --symbols EURUSD,GBPUSD,...  # limitar símbolos a actualizar
  --lb90-min 0.25              # si querés forzar LB90 global
  --ensure-candles --candles-root data/candles   # intenta consolidar parquet si faltan
  --dry-run                    # sólo muestra cambios, no escribe
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import sys

# -------------------------------
# Utils JSON
# -------------------------------
def _load_json(path: Path, default=None):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {} if default is None else default
    except Exception as e:
        print(f"⚠️  No pude leer {path}: {e}")
        return {} if default is None else default

def _dump_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"✅ Escrito: {path}")

# -------------------------------
# Descubrimiento de summary
# -------------------------------
def _find_latest_summary(reports_dir: Path) -> Path | None:
    # Busca directorios tipo reports/exit_YYYYMMDD
    if not reports_dir.exists():
        return None
    candidates = []
    for d in reports_dir.iterdir():
        if d.is_dir() and d.name.startswith("exit_"):
            summ = d / "exit_mc_summary.json"
            if summ.exists():
                # ordenar por timestamp del archivo
                candidates.append((summ.stat().st_mtime, summ))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

# -------------------------------
# Parse de summary robusto
# -------------------------------
_PARAM_KEYS = {
    "targets_usd", "fractions", "breakeven_after_min", "breakeven_weak_threshold",
    "commission_per_lot", "commission_buffer_usd", "grace_minutes", "max_hours"
}

def _extract_params(obj: dict) -> dict:
    """
    Acepta varias formas:
      - {"best_params": {...}}
      - {"best": {...}}
      - {"params": {...}}
      - o directo con las keys
    """
    if not isinstance(obj, dict):
        return {}
    for k in ("best_params", "best", "params", "recommended", "thresholds"):
        if isinstance(obj.get(k), dict):
            return {kk: obj[k].get(kk) for kk in _PARAM_KEYS if kk in obj[k]}
    # plano
    return {kk: obj.get(kk) for kk in _PARAM_KEYS if kk in obj}

def _parse_summary(summary: dict, symbols_filter: set[str] | None = None) -> dict:
    """
    Devuelve: { "EURUSD": {params...}, ... }
    """
    out = {}
    # casos: {"symbols": {...}} o {"EURUSD": {...}, ...}
    sym_map = None
    if isinstance(summary.get("symbols"), dict):
        sym_map = summary["symbols"]
    else:
        # tomar claves que parezcan símbolos (letras y longitud 6 o similar)
        sym_map = {k: v for k, v in summary.items() if isinstance(v, dict) and k.isalpha() and 5 <= len(k) <= 7}

    if not sym_map:
        print("⚠️  Summary no tiene mapa de símbolos reconocible; nada para aplicar.")
        return out

    for sym, node in sym_map.items():
        sym_u = sym.upper()
        if symbols_filter and sym_u not in symbols_filter:
            continue
        params = _extract_params(node)
        if params:
            out[sym_u] = params

    # defaults globales si el summary los trae
    defaults = {}
    if isinstance(summary.get("defaults"), dict):
        defaults = _extract_params(summary["defaults"])
    return {"symbols": out, "defaults": defaults}

# -------------------------------
# Merge con pmi_config.json
# -------------------------------
def _merge_pmi_config(current: dict, summary_params: dict,
                      set_mode: str | None, lb90_min: float | None,
                      symbols_filter: set[str] | None) -> dict:
    cfg = dict(current) if current else {}
    cfg.setdefault("mode", "active")
    cfg.setdefault("lb90_min", 0.25)
    cfg.setdefault("defaults", {})
    cfg.setdefault("symbols", {})

    if set_mode:
        cfg["mode"] = set_mode
    if lb90_min is not None:
        cfg["lb90_min"] = float(lb90_min)

    # defaults desde summary (no pisamos si no vienen)
    for k, v in (summary_params.get("defaults") or {}).items():
        cfg["defaults"][k] = v

    # por símbolo
    to_apply = summary_params.get("symbols") or {}
    for sym, params in to_apply.items():
        if symbols_filter and sym not in symbols_filter:
            continue
        node = cfg["symbols"].get(sym, {})
        node.update(params or {})
        cfg["symbols"][sym] = node

    return cfg

# -------------------------------
# Candles check + consolidación
# -------------------------------
def _try_import_consolidate():
    # intentos de import según tu repo
    mods = [
        "consolidate_nested",
        "src.tools.consolidate_nested"
    ]
    for m in mods:
        try:
            mod = __import__(m, fromlist=["consolidate_candles"])
            if hasattr(mod, "consolidate_candles"):
                return mod.consolidate_candles
        except Exception:
            continue
    return None

def _ensure_candles_ready(candles_root: Path, symbols: set[str]):
    missing = []
    for s in symbols:
        f = candles_root / f"{s}_M5.parquet"
        if not f.exists():
            missing.append(s)
    if not missing:
        print("✓ Candles consolidadas presentes.")
        return

    print(f"⚠️  Faltan parquet consolidados para: {', '.join(missing)}")
    consolidate_candles = _try_import_consolidate()
    if consolidate_candles is None:
        print("→ No pude importar consolidate_nested.consolidate_candles()")
        print("  Sugerencia: ejecutá manualmente:")
        print(f"  python consolidate_nested.py --candles-dir {candles_root} --timeframe M5")
        return
    # Ejecutar consolidación
    try:
        consolidate_candles(str(candles_root), timeframe="M5")
    except TypeError:
        # por si la firma difiere (compatibilidad)
        consolidate_candles(str(candles_root))
    print("✓ Consolidación intentada. Verificá nuevamente si persisten faltantes.")

# -------------------------------
# Diff bonito
# -------------------------------
def _pretty_diff(before: dict, after: dict, symbols: set[str] | None):
    import copy
    b = copy.deepcopy(before or {})
    a = copy.deepcopy(after or {})
    # limitar a símbolos de interés
    if symbols:
        b.setdefault("symbols", {})
        a.setdefault("symbols", {})
        b["symbols"] = {k: v for k, v in b["symbols"].items() if k in symbols}
        a["symbols"] = {k: v for k, v in a["symbols"].items() if k in symbols}
    print("\n=== PREVIEW DE CAMBIOS EN pmi_config.json ===")
    print("ANTES:")
    print(json.dumps(b, indent=2, ensure_ascii=False))
    print("\nDESPUÉS:")
    print(json.dumps(a, indent=2, ensure_ascii=False))
    print("=============================================\n")

# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser(description="Aplica calibración de salidas a pmi_config.json")
    ap.add_argument("--summary", type=str, default=None, help="Ruta a exit_mc_summary.json")
    ap.add_argument("--reports-dir", type=str, default="reports", help="Directorio base con exit_*/")
    ap.add_argument("--out", type=str, default="configs/pmi_config.json", help="Archivo pmi_config.json a escribir")
    ap.add_argument("--symbols", type=str, default=None, help="Símbolos a aplicar (coma-separados)")
    ap.add_argument("--set-mode", type=str, choices=["active", "observer"], default=None, help="Forzar modo PMI")
    ap.add_argument("--lb90-min", type=float, default=None, help="Forzar lb90_min global")
    ap.add_argument("--ensure-candles", action="store_true", help="Verifica/Consolida parquet en data/candles")
    ap.add_argument("--candles-root", type=str, default="data/candles", help="Raíz de velas consolidadas")
    ap.add_argument("--dry-run", action="store_true", help="Sólo mostrar cambios (no escribe)")
    args = ap.parse_args()

    symbols_filter = None
    if args.symbols:
        symbols_filter = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}

    # 1) summary
    summary_path = Path(args.summary) if args.summary else _find_latest_summary(Path(args.reports_dir))
    if not summary_path or not summary_path.exists():
        print("❌ No encontré exit_mc_summary.json (usa --summary o --reports-dir).")
        sys.exit(1)

    summary = _load_json(summary_path, default={})
    parsed = _parse_summary(summary, symbols_filter)
    if not parsed.get("symbols"):
        print("❌ El summary no contiene parámetros aplicables.")
        sys.exit(1)

    # 2) opcional: verificar/consolidar velas
    if args.ensure_candles:
        _ensure_candles_ready(Path(args.candles_root), set(parsed["symbols"].keys()))

    # 3) merge con pmi_config.json
    out_path = Path(args.out)
    current_cfg = _load_json(out_path, default={})
    merged = _merge_pmi_config(current_cfg, parsed, args.set_mode, args.lb90_min, symbols_filter)

    # preview
    _pretty_diff(current_cfg, merged, symbols_filter)

    # 4) escribir
    if args.dry_run:
        print("🟡 DRY-RUN: no se escribió el archivo. Usa sin --dry-run para aplicar.")
    else:
        _dump_json(out_path, merged)

if __name__ == "__main__":
    main()
