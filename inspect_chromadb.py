#!/usr/bin/env python3
# inspect_chromadb.py

import chromadb
import json
from collections import Counter

def inspect_chromadb():
    try:
        client = chromadb.PersistentClient(path="db/chroma_db")
        collection = client.get_collection("historical_market_states")
        
        total_docs = collection.count()
        print(f"📊 Total documentos en ChromaDB: {total_docs}")
        
        if total_docs == 0:
            print("❌ ChromaDB está vacío!")
            return
        
        # Muestra representativa
        n_sample = min(100, total_docs)
        sample = collection.query(
            query_embeddings=[[0.0] * 253],  # Vector neutro
            n_results=n_sample,
            include=["metadatas", "embeddings"]
        )
        
        metadatas = sample['metadatas'][0]
        embeddings = sample['embeddings'][0]
        
        print(f"\n🔍 ANÁLISIS DE {len(metadatas)} DOCUMENTOS:")
        print("="*50)
        
        # Estructura de metadatos
        if metadatas:
            first_meta = metadatas[0]
            print(f"📋 Estructura metadatos:")
            for key, value in first_meta.items():
                print(f"   {key}: {value} ({type(value).__name__})")
        
        # Distribución de outcomes
        outcomes = [m.get('outcome') for m in metadatas]
        print(f"\n🎯 Distribución outcomes:")
        print(f"   Valores: {dict(Counter(outcomes))}")
        print(f"   Tipos: {dict(Counter(type(o).__name__ for o in outcomes))}")
        
        # Símbolos y estrategias
        symbols = [m.get('symbol') for m in metadatas]
        strategies = [m.get('strategy') for m in metadatas]
        
        print(f"\n📈 Símbolos disponibles:")
        for symbol, count in Counter(symbols).items():
            print(f"   {symbol}: {count} docs")
        
        print(f"\n🔧 Estrategias disponibles:")
        for strategy, count in Counter(strategies).items():
            print(f"   {strategy}: {count} docs")
        
        # Calidad de embeddings
        if embeddings is not None and len(embeddings) > 0:
            first_embedding = embeddings[0]
            print(f"\n📐 Embeddings:")
            print(f"   Dimensión: {len(first_embedding)}")
            # Ensure first_embedding is a list or array for min/max
            if len(first_embedding) > 0:
                print(f"   Rango: [{min(first_embedding):.3f}, {max(first_embedding):.3f}]")
                print(f"   Media: {sum(first_embedding)/len(first_embedding):.3f}")
        
        print("\n✅ Inspección completada")
        
    except Exception as e:
        print(f"❌ Error inspeccionando ChromaDB: {e}")

if __name__ == "__main__":
    inspect_chromadb()