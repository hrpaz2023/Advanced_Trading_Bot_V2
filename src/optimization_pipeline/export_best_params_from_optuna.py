import argparse, json, os, re, sqlite3, sys
from pathlib import Path

def parse_combo_from_study_name(study_name: str):
    # Ejemplos válidos:
    #  EURUSD_ema_crossover
    #  GBPUSD_multi_filter_scalper_v2   -> symbol=GBPUSD, strategy=multi_filter_scalper_v2
    m = re.match(r"^([A-Z]{3,10})[_\-](.+)$", study_name)
    if not m:
        return None, None
    symbol = m.group(1).upper()
    strategy = m.group(2)
    return symbol, strategy

def try_optuna_best_params(db_path: Path, study_name: str):
    """Intenta cargar con Optuna y devolver params del best_trial."""
    try:
        import optuna
    except Exception:
        return None
    storage = f"sqlite:///{db_path}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        # single-obj: best_trial; multi-obj: best_trials[0]
        if getattr(study, "best_trials", None):
            bt = study.best_trials[0]
        else:
            bt = study.best_trial
        return dict(bt.params) if bt and bt.params is not None else None
    except Exception:
        return None

def sql_fetch_best_params(db_path: Path, study_id: int, direction: str):
    """Fallback SQL: obtiene best_trial por study_id y devuelve dict de params."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        # estado COMPLETE puede ser 1 (v2) o 'COMPLETE' (v3)
        # tomamos ambos
        order = "DESC" if direction.upper() == "MAXIMIZE" else "ASC"
        cur.execute(f"""
            SELECT t.trial_id, v.value
            FROM trials t
            JOIN trial_values v ON t.trial_id = v.trial_id
            WHERE t.study_id = ?
              AND (t.state = 1 OR t.state = 'COMPLETE')
            ORDER BY v.value {order}
            LIMIT 1
        """, (study_id,))
        row = cur.fetchone()
        if not row:
            return None
        trial_id = row["trial_id"]

        cur.execute("""
            SELECT param_name, param_value, value_json
            FROM trial_params
            WHERE trial_id = ?
        """, (trial_id,))
        params = {}
        for r in cur.fetchall():
            name = r["param_name"]
            raw = r["param_value"]
            valj = r["value_json"]
            val = None
            if valj:
                try:
                    val = json.loads(valj)
                except Exception:
                    val = None
            if val is None:
                try:
                    if "." in str(raw):
                        val = float(raw)
                    else:
                        val = int(raw)
                except Exception:
                    val = raw
            params[name] = val
        return params if params else None
    finally:
        conn.close()

def list_studies(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute("SELECT study_id, study_name FROM studies")
        studies = cur.fetchall()
        out = []
        for s in studies:
            study_id = s["study_id"]
            study_name = s["study_name"]
            # dirección (MAXIMIZE/MINIMIZE) – v3: table 'study_directions', v2 igual
            direction = "MAXIMIZE"
            try:
                cur.execute("SELECT direction FROM study_directions WHERE study_id = ? ORDER BY objective ORDER BY 1", (study_id,))
                row = cur.fetchone()
                if row is not None:
                    # 1 = MINIMIZE / 0 = MINIMIZE en algunas versiones → normalizamos:
                    # Optuna internamente usa 0(minimize) / 1(maximize) o strings; probamos lo típico
                    d = row["direction"]
                    if str(d).upper() in ("MINIMIZE", "0"):
                        direction = "MINIMIZE"
                    else:
                        direction = "MAXIMIZE"
            except Exception:
                # fallback: asume maximize
                direction = "MAXIMIZE"
            out.append((study_id, study_name, direction))
        return out
    finally:
        conn.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies_dir", default="optimization_studies_advanced")
    ap.add_argument("--out", default="configs/optimized_parameters.json")
    args = ap.parse_args()

    studies_dir = Path(args.studies_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not studies_dir.exists():
        print(f"❌ No existe {studies_dir}")
        sys.exit(1)

    combined = {}
    dbs = list(studies_dir.glob("*.db"))
    if not dbs:
        print("⚠️ No se encontraron .db en", studies_dir)

    for db in dbs:
        studies = list_studies(db)
        if not studies:
            print(f"⚠️ {db.name}: sin 'studies' definidos.")
            continue

        for study_id, study_name, direction in studies:
            symbol, strategy = parse_combo_from_study_name(study_name)
            if not symbol or not strategy:
                # sólo exportamos estudios que cumplan el patrón SYMBOL_strategy
                continue

            # 1) Intento Optuna API
            params = try_optuna_best_params(db, study_name)
            # 2) Fallback SQL
            if not params:
                params = sql_fetch_best_params(db, study_id, direction)

            key = f"{symbol}:{strategy}"
            if params:
                combined[key] = params
                print(f"✅ {db.name} :: {study_name} -> {key} ({len(params)} params)")
            else:
                print(f"⚠️ {db.name} :: {study_name} -> sin trial COMPLETE o sin params")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print(f"\n💾 Parámetros exportados a: {out_path} (combos={len(combined)})")

if __name__ == "__main__":
    main()
