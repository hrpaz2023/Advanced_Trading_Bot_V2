
# scan_project.py
# ----------------
# Escanea el proyecto (excepto venv/.git/__pycache__) y arma un reporte de:
# - Ubicación real de: configs, reports/*, models/*, db/chroma_db, scripts clave
# - Cobertura ML/Chroma por (símbolo, estrategia) en configs/global_config.json
# - Sugerencia de orchestrator_config.json con rutas corregidas
#
# Uso (desde la raíz del repo):
#   python scan_project.py --root . --out scan_report.json --write-proposals
#
import os, json, re, argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

SKIP_DIRS = {"venv", ".git", "__pycache__", ".idea", ".vscode", ".pytest_cache", "node_modules"}

KEY_DIR_NAMES = {
    "reports_root": ["reports"],
    "models_root": ["models"],
    "db_root": ["db"],
    "ultimate_analysis": ["ultimate_analysis_reports_v2"],
    "causal_reports": ["causal_reports"],
    "cross_corr": ["cross_correlation_reports"],
    "loss_reports": ["multi_asset_loss_reports"],
    "win_reports": ["exhaustive_win_reports"],
    "chroma_db": ["chroma_db"],
}

KEY_FILES = [
    "main_bot.py",
    "execution_controller.py",
    "policy_switcher.py",
    "orchestrator_config.json",
    "consolidate_global_insights.py",
    "backtest_ab_orchestrator.py",
    "causal_trading_analyzer.py",
    "advanced_analyzer_v2.py",
    "run_pipeline.py"
]

@dataclass
class Combo:
    symbol: str
    strategy: str

def walk_project(root: Path):
    found = {"dirs": {}, "files": {}}
    for k in KEY_DIR_NAMES:
        found["dirs"][k] = []

    for k in KEY_FILES:
        found["files"][k] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # prune
        dn = Path(dirpath).name
        if dn in SKIP_DIRS:
            dirnames[:] = []
            continue
        # prune children
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        # dir matches
        for key, names in KEY_DIR_NAMES.items():
            for nm in names:
                if Path(dirpath).name.lower() == nm.lower():
                    found["dirs"][key].append(str(Path(dirpath).resolve()))

        # file matches
        for f in filenames:
            if f in KEY_FILES:
                found["files"][f].append(str(Path(dirpath, f).resolve()))

    return found

def find_first(paths: List[str]) -> str:
    return paths[0] if paths else ""

def load_global_config(root: Path) -> Dict:
    # Try typical locations
    candidates = [root/"configs"/"global_config.json", root/"global_config.json"]
    for c in candidates:
        if c.exists():
            with open(c, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}

def load_orchestrator_config(root: Path) -> Dict:
    candidates = [root/"orchestrator_config.json", root/"configs"/"orchestrator_config.json"]
    for c in candidates:
        if c.exists():
            with open(c, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}

def combos_from_config(cfg: Dict) -> List[Combo]:
    combos: List[Combo] = []
    if not cfg:
        return combos
    if "controllers" in cfg and cfg["controllers"]:
        for c in cfg["controllers"]:
            combos.append(Combo(symbol=c["symbol"], strategy=c["strategy"]))
    else:
        for s in cfg.get("trading_symbols", []):
            combos.append(Combo(symbol=s, strategy="ema_crossover"))
    return combos

def find_models_for_combo(root: Path, combo: Combo) -> Dict:
    # Buscar en cualquier carpeta "models/<estrategia>"
    result = {"strategy_dir": "", "files": {}, "missing": []}
    expected = [
        f"{combo.symbol}_confirmation_model.pkl",
        f"{combo.symbol}_feature_scaler.pkl",
        f"{combo.symbol}_model_features.joblib"
    ]
    # Buscar todas las carpetas models del proyecto
    models_dirs = list(root.rglob("models"))
    # Priorizar 'models' + subcarpeta estrategia
    for mdir in models_dirs:
        sdir = mdir / combo.strategy
        if sdir.exists() and sdir.is_dir():
            result["strategy_dir"] = str(sdir.resolve())
            for fn in expected:
                fp = sdir / fn
                result["files"][fn] = str(fp.resolve()) if fp.exists() else ""
            result["missing"] = [fn for fn in expected if not (sdir / fn).exists()]
            return result
    # Si no hay subcarpeta por estrategia, buscar por patrón a nivel models
    if models_dirs:
        sdir = models_dirs[0]
        result["strategy_dir"] = str(sdir.resolve())
        for fn in expected:
            fp = sdir / fn
            result["files"][fn] = str(fp.resolve()) if fp.exists() else ""
        result["missing"] = [fn for fn in expected if not (sdir / fn).exists()]
    return result

def guess_reports_paths(found) -> Dict:
    # Elige la mejor coincidencia para cada carpeta de reports
    pick = {}
    pick["reports_root"] = find_first(found["dirs"]["reports_root"])
    for k in ("ultimate_analysis","causal_reports","cross_corr","loss_reports","win_reports"):
        pick[k] = find_first(found["dirs"][k])
        # Si no se encontró directamente, intenta bajo reports_root
        if not pick[k] and pick["reports_root"]:
            candidate = Path(pick["reports_root"]) / KEY_DIR_NAMES[k][0]
            if candidate.exists():
                pick[k] = str(candidate.resolve())
    return pick

def guess_db_paths(found) -> Dict:
    pick = {}
    pick["db_root"] = find_first(found["dirs"]["db_root"])
    pick["chroma_db"] = find_first(found["dirs"]["chroma_db"])
    # fallback si no existe chroma_db directo
    if not pick["chroma_db"] and pick["db_root"]:
        candidate = Path(pick["db_root"]) / "chroma_db"
        if candidate.exists():
            pick["chroma_db"] = str(candidate.resolve())
    return pick

def propose_orchestrator_config(root: Path, found) -> Dict:
    rp = guess_reports_paths(found)
    dp = guess_db_paths(found)

    proposal = {
        "paths": {
            "cross_correlation_dir": rp.get("cross_corr",""),
            "loss_reports_dir": rp.get("loss_reports",""),
            "win_reports_dir": rp.get("win_reports",""),
            "advanced_reports_dir": rp.get("ultimate_analysis",""),
            "causal_reports_dir": rp.get("causal_reports",""),
            "output_dir": rp.get("reports_root","") or str((root/"reports").resolve())
        },
        "chroma": {
            "path": dp.get("chroma_db","")
        }
    }
    return proposal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--out", type=str, default="scan_report.json")
    ap.add_argument("--write-proposals", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    found = walk_project(root)

    # Load configs
    gcfg = load_global_config(root)
    ocfg = load_orchestrator_config(root)
    combos = combos_from_config(gcfg)

    # Models coverage
    models = {}
    for cb in combos:
        key = f"{cb.symbol}:{cb.strategy}"
        models[key] = find_models_for_combo(root, cb)

    # propose orch config from discovered paths
    proposal = propose_orchestrator_config(root, found)

    report = {
        "root": str(root),
        "found_dirs": found["dirs"],
        "found_files": found["files"],
        "global_config": {
            "path": str((root/'configs'/'global_config.json').resolve()) if (root/'configs'/'global_config.json').exists() else "",
            "controllers_count": len(combos),
            "controllers": [{"symbol": c.symbol, "strategy": c.strategy} for c in combos]
        },
        "orchestrator_config_present": bool(ocfg),
        "models_coverage": models,
        "proposed_orchestrator_config": proposal
    }

    # Write report JSON
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Reporte escrito en {args.out}")
    if args.write_proposals:
        out_cfg = root / "proposed_orchestrator_config.json"
        with open(out_cfg, "w", encoding="utf-8") as f:
            json.dump(proposal, f, indent=2)
        print(f"✅ Propuesta de orchestrator_config.json escrita en {out_cfg}")

if __name__ == "__main__":
    main()
