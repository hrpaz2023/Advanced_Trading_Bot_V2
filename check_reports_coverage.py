# check_reports_coverage.py (v2)
import argparse, json, os, re
from pathlib import Path

# patrones a limpiar al tomar el nombre de archivo
SUFFIX_PATTERNS = [
    r"_report$",
    r"_win_analysis(_\d{8}_\d{6})?$",
    r"_loss_analysis(_\d{8}_\d{6})?$",
    r"_cross_corr(_\d{8}_\d{6})?$",
    r"_causal(_\d{8}_\d{6})?$",
]

def sanitize_strategy(raw: str) -> str:
    s = raw
    for pat in SUFFIX_PATTERNS:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    return s

def combo_from_filename(fp: Path):
    m = re.match(r"([A-Z]{3,10})[_\-]([A-Za-z0-9_]+)", fp.stem)
    if not m:
        return None, None
    sym, strat = m.group(1), sanitize_strategy(m.group(2))
    return sym.upper(), strat

def combo_from_json_or_name(data, fp: Path):
    asset = None
    strat = None
    if isinstance(data, dict):
        asset = data.get("asset") or data.get("symbol") or data.get("Symbol")
        strat = data.get("strategy") or data.get("Strategy")
    if not asset or not strat:
        return combo_from_filename(fp)
    return asset.upper(), sanitize_strategy(strat)

def collect(dir_path: str):
    combos = set(); count = 0
    if not dir_path or not os.path.isdir(dir_path):
        return combos, count
    for f in Path(dir_path).rglob("*.json"):
        count += 1
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            data = None
        if isinstance(data, list):
            # si es lista, intenta por filename normalizado
            s, st = combo_from_filename(f)
        else:
            s, st = combo_from_json_or_name(data, f)
        if s and st: combos.add((s, st))
    return combos, count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="orchestrator_config.json")
    ap.add_argument("--expect", type=int, default=20)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    paths = cfg.get("paths", {})
    buckets = {
        "advanced": paths.get("advanced_reports_dir", ""),
        "win": paths.get("win_reports_dir", ""),
        "loss": paths.get("loss_reports_dir", ""),
        "cross": paths.get("cross_correlation_dir", ""),
        "causal": paths.get("causal_reports_dir", ""),
    }

    coverage = {}; union = set()
    for name, p in buckets.items():
        combos, files = collect(p)
        coverage[name] = {"count_files": files, "combos": sorted(list(combos))}
        union |= combos

    print("=== Cobertura por bucket ===")
    for name, info in coverage.items():
        print(f"{name:8s}: files={info['count_files']:4d} | combos={len(info['combos']):2d}")

    print("\n=== Unión de combos detectados (normalizada) ===")
    for s, st in sorted(union):
        print(f"- {s}/{st}")
    print(f"\nTotal combos: {len(union)} (esperado aprox: {args.expect})")

    Path("reports_coverage.json").write_text(
        json.dumps({"coverage": coverage, "union_combos": sorted(list(union)), "union_count": len(union), "expected": args.expect}, indent=2),
        encoding="utf-8"
    )
    print("\n💾 Reporte escrito en reports_coverage.json")

if __name__ == "__main__":
    main()
