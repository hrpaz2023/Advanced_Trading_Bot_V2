# -*- coding: utf-8 -*-
"""
Validador de estructura: modelos ML + ChromaDB (y chequeos básicos de rutas)
----------------------------------------------------------------------------

Qué valida por cada (símbolo, estrategia) declarado en configs/global_config.json:
- Artefactos ML en models/<estrategia>/
- Presencia de registros en ChromaDB (colección 'historical_market_states')

Además:
- Lee la ruta de Chroma de orchestrator_config.json (o usa --chroma_path)
- Verifica existencia de reports/global_insights.json (opcional)
- Imprime resumen legible y opcionalmente escribe un reporte JSON
- Devuelve exit code 1 si hay faltantes y se pasa --fail_on_missing

Uso (desde la raíz del repo):
    python src/tools/validate_model_and_chroma_coverage.py \
        --config configs/global_config.json \
        --orchestrator_config orchestrator_config.json \
        --report validation_report.json \
        --fail_on_missing
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ----------------------------- Utilidades ----------------------------------

@dataclass
class Combo:
    symbol: str
    strategy: str

def load_controllers(config_path: Path) -> List[Combo]:
    if not config_path.exists():
        raise FileNotFoundError(f"No existe {config_path}")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    combos: List[Combo] = []
    if "controllers" in cfg and cfg["controllers"]:
        for c in cfg["controllers"]:
            combos.append(Combo(symbol=c["symbol"], strategy=c["strategy"]))
    else:
        # fallback: si no hay controllers, usar trading_symbols con estrategia por defecto
        for s in cfg.get("trading_symbols", []):
            combos.append(Combo(symbol=s, strategy="ema_crossover"))
    return combos

def expected_model_files(symbol: str, strategy: str) -> List[str]:
    return [
        f"{symbol}_confirmation_model.pkl",
        f"{symbol}_feature_scaler.pkl",
        f"{symbol}_model_features.joblib",
    ]

def check_models_for_combo(models_root: Path, combo: Combo) -> Tuple[bool, Dict[str, str], List[str]]:
    """
    Busca en models/<estrategia>/ los 3 archivos esperados.
    Devuelve: (ok, paths_encontrados, faltantes)
    """
    strat_dir = models_root / combo.strategy
    found: Dict[str, str] = {}
    missing: List[str] = []

    if not strat_dir.exists():
        # si no existe la carpeta de estrategia, todos faltan
        return (False, found, expected_model_files(combo.symbol, combo.strategy))

    for fname in expected_model_files(combo.symbol, combo.strategy):
        fp = strat_dir / fname
        if fp.exists():
            found[fname] = str(fp.resolve())
        else:
            missing.append(fname)
    return (len(missing) == 0, found, missing)

def read_orchestrator_chroma_path(orch_cfg_path: Optional[Path]) -> Optional[str]:
    if orch_cfg_path and orch_cfg_path.exists():
        try:
            cfg = json.loads(orch_cfg_path.read_text(encoding="utf-8"))
            return (cfg.get("chroma", {}) or {}).get("path")
        except Exception:
            return None
    return None
def check_chroma_combo(chroma_path: str, combo: Combo) -> tuple:
    """
    Verifica presencia en ChromaDB intentando:
    1) query_texts (por si la colección usa encoder textual)
    2) Si hay error de dimensión, auto-detecta el 'expected_dim' del mensaje
       y reintenta con query_embeddings = [0.0]*expected_dim (filtro por metadatos).
    """
    try:
        import chromadb  # type: ignore
    except Exception as e:
        return (False, f"chromadb_not_installed:{e}")

    try:
        client = chromadb.PersistentClient(path=chroma_path)
        try:
            col = client.get_collection("historical_market_states")
        except Exception:
            return (False, "collection_missing")

        # 1) Intento textual (funciona si la colección fue creada con encoder de texto)
        try:
            res = col.query(
                query_texts=["probe"],
                n_results=1,
                where={"$and": [{"symbol": combo.symbol}, {"strategy": combo.strategy}]},
            )
            ok = bool(res and res.get("ids") and res["ids"][0])
            return (ok, "ok" if ok else "no_match_for_combo")
        except Exception as e:
            msg = str(e)

        # 2) Si falló, intentar inferir 'expected_dim' del error y reintentar con embeddings dummy
        import re
        m = re.search(r"dimension of (\d+)", msg) or re.search(r"dimension (\d+)", msg)
        if not m:
            return (False, f"query_error:{msg}")

        expected_dim = int(m.group(1))
        try:
            dummy = [[0.0] * expected_dim]  # 1 vector del tamaño esperado
            res = col.query(
                query_embeddings=dummy,
                n_results=1,
                where={"$and": [{"symbol": combo.symbol}, {"strategy": combo.strategy}]},
            )
            ok = bool(res and res.get("ids") and res["ids"][0])
            return (ok, "ok_num" if ok else "no_match_for_combo_num")
        except Exception as e2:
            return (False, f"query_error_num:{e2}")

    except Exception as e:
        return (False, f"client_error:{e}")


def exists_global_insights(output_dir: Path) -> bool:
    gi = output_dir / "global_insights.json"
    return gi.exists()

def print_table(rows: List[Dict[str, str]]):
    if not rows:
        print("No hay filas para mostrar.")
        return
    # ancho de columnas
    cols = ["symbol", "strategy", "ml_ok", "ml_missing", "chroma_ok", "chroma_status"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    # encabezado
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))

# ------------------------------- Main --------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Validador de estructura: modelos ML + ChromaDB.")
    ap.add_argument("--config", default="configs/global_config.json", help="Ruta a global_config.json")
    ap.add_argument("--models_dir", default="models", help="Directorio base de modelos")
    ap.add_argument("--orchestrator_config", default="orchestrator_config.json", help="Ruta a orchestrator_config.json (para leer chroma.path)")
    ap.add_argument("--chroma_path", default=None, help="Ruta de ChromaDB (anula la del orchestrator)")
    ap.add_argument("--output_dir", default="reports", help="Para chequear global_insights.json")
    ap.add_argument("--report", default=None, help="Escribe un JSON con el resultado (p.ej. validation_report.json)")
    ap.add_argument("--fail_on_missing", action="store_true", help="Exit code 1 si hay faltantes ML o Chroma")
    args = ap.parse_args()

    root = Path(".").resolve()
    config_path = Path(args.config)
    models_root = Path(args.models_dir)
    orch_cfg_path = Path(args.orchestrator_config) if args.orchestrator_config else None
    out_dir = Path(args.output_dir)

    # 1) Combos
    try:
        combos = load_controllers(config_path)
    except Exception as e:
        print(f"❌ Error leyendo {config_path}: {e}")
        sys.exit(2)

    # 2) Ruta Chroma
    chroma_path = args.chroma_path or read_orchestrator_chroma_path(orch_cfg_path) or str((root / "db" / "chroma_db").resolve())
    print(f"CHROMA_PATH = {chroma_path}")

    # 3) Validación por combo
    rows = []
    any_ml_missing = False
    any_chroma_missing = False

    for cb in combos:
        ml_ok, found, missing = check_models_for_combo(models_root, cb)
        if not ml_ok:
            any_ml_missing = True
        chroma_ok, chroma_status = check_chroma_combo(chroma_path, cb)
        if not chroma_ok and chroma_status not in ("chromadb_not_installed",):
            any_chroma_missing = True

        rows.append({
            "symbol": cb.symbol,
            "strategy": cb.strategy,
            "ml_ok": "YES" if ml_ok else "NO",
            "ml_missing": ",".join(missing) if missing else "",
            "chroma_ok": "YES" if chroma_ok else "NO",
            "chroma_status": chroma_status,
        })

    # 4) Consolidado disponible
    gi_ok = exists_global_insights(out_dir)

    # 5) Salida
    print("\n=== Validación por controlador ===")
    print_table(rows)
    print("\nGlobal Insights:", "OK" if gi_ok else "NO (falta reports/global_insights.json)")

    summary = {
        "root": str(root),
        "config": str(config_path.resolve()),
        "models_dir": str(models_root.resolve()),
        "chroma_path": chroma_path,
        "global_insights_present": gi_ok,
        "results": rows,
        "counts": {
            "controllers": len(rows),
            "ml_ok": sum(1 for r in rows if r["ml_ok"] == "YES"),
            "ml_missing": sum(1 for r in rows if r["ml_ok"] != "YES"),
            "chroma_ok": sum(1 for r in rows if r["chroma_ok"] == "YES"),
            "chroma_missing": sum(1 for r in rows if r["chroma_ok"] != "YES"),
        }
    }

    if args.report:
        Path(args.report).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n💾 Reporte escrito en {args.report}")

    if args.fail_on_missing and (any_ml_missing or any_chroma_missing or not gi_ok):
        sys.exit(1)

if __name__ == "__main__":
    main()
