
# chroma_reset.py
# ---------------
# Resetea la colección ChromaDB 'historical_market_states' usando la ruta
# definida en orchestrator_config.json (o db/chroma_db por defecto).
#
# Uso:
#   python chroma_reset.py --config orchestrator_config.json
#
import argparse, json, os
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="orchestrator_config.json")
    args = ap.parse_args()

    cfg = {}
    try:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    except Exception:
        pass

    chroma_path = (cfg.get("chroma", {}) or {}).get("path") or str((Path("db")/"chroma_db").resolve())
    print(f"➡ Usando Chroma en: {chroma_path}")
    os.makedirs(chroma_path, exist_ok=True)

    try:
        import chromadb
    except Exception as e:
        print(f"❌ chromadb no instalado: {e}")
        return

    client = chromadb.PersistentClient(path=chroma_path)
    try:
        client.delete_collection("historical_market_states")
        print("🗑️  Colección eliminada.")
    except Exception:
        print("ℹ️  Colección no existía o no se pudo borrar (continuo).")
    client.create_collection("historical_market_states")
    print("✅ Colección creada nuevamente.")

if __name__ == "__main__":
    main()
