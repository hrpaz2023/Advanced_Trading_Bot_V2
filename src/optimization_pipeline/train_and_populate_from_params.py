
# train_and_populate_from_params.py
# ---------------------------------
# Toma configs/optimized_parameters.json y, para cada (símbolo, estrategia),
# ejecuta entrenamiento de modelos ML y poblado de ChromaDB.
#
# Intenta importar funciones del proyecto:
#   - src.model_training.train_model import train_model
#   - src.data_preparation.populate_chroma import populate_database
#
# Notas:
# - Establece CHROMA_PATH desde orchestrator_config.json antes de poblar.
# - No modifica tus módulos; es un wrapper de conveniencia.
#
# Uso:
#   python train_and_populate_from_params.py --params configs/optimized_parameters.json --limit 0
#
import argparse, json, os, importlib, sys, time
from pathlib import Path

def dynamic_import(path):
    try:
        return __import__(path, fromlist=["*"])
    except Exception as e:
        print(f"❌ No pude importar {path}: {e}")
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="configs/optimized_parameters.json")
    ap.add_argument("--config", default="orchestrator_config.json")
    ap.add_argument("--limit", type=int, default=0, help="Procesar solo los primeros N combos (0 = todos)")
    args = ap.parse_args()

    params_path = Path(args.params)
    if not params_path.exists():
        print(f"❌ No existe {params_path}")
        return

    # Leer CHROMA_PATH
    chroma_path = None
    try:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        chroma_path = (cfg.get("chroma", {}) or {}).get("path")
    except Exception:
        pass
    if chroma_path:
        os.environ["CHROMA_PATH"] = chroma_path
        print(f"ENV CHROMA_PATH={chroma_path}")

    data = json.loads(params_path.read_text(encoding="utf-8"))
    items = list(data.items())
    if args.limit and args.limit > 0:
        items = items[:args.limit]

    # Imports dinámicos
    tm = dynamic_import("src.model_training.train_model")
    pc = dynamic_import("src.data_preparation.populate_chroma")

    train_model = getattr(tm, "train_model", None) if tm else None
    populate_database = getattr(pc, "populate_database", None) if pc else None

    if not train_model and not populate_database:
        print("⚠️ No pude importar train_model ni populate_database. Considera usar run_pipeline.py directamente.")
        return

    # Procesar combos
    for key, params in items:
        try:
            symbol, strategy = key.split(":", 1)
        except ValueError:
            print(f"⚠️ Clave inválida en params: {key}")
            continue
        print(f"\n🚀 {symbol}/{strategy}")

        # Entrenar
        if train_model:
            try:
                # Intento 1: train_model(symbol, strategy, params=params)
                try:
                    train_model(symbol, strategy, params=params)
                except TypeError:
                    # Intento 2: train_model(symbol, strategy) y que lea params internamente
                    train_model(symbol, strategy)
                print("   ✅ Modelo entrenado")
            except Exception as e:
                print(f"   ❌ Error entrenando: {e}")

        # Poblar Chroma
        if populate_database:
            try:
                # populate_database(symbol, strategy, params=params) (si acepta)
                try:
                    populate_database(symbol, strategy, params=params)
                except TypeError:
                    populate_database(symbol, strategy)
                print("   ✅ Chroma poblado")
            except Exception as e:
                print(f"   ❌ Error poblando Chroma: {e}")

    print("\n🏁 Proceso finalizado.")

if __name__ == "__main__":
    main()
