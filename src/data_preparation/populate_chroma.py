# src/data_preparation/populate_chroma.py

import pandas as pd
import chromadb
import joblib
import os
from tqdm import tqdm
import numpy as np

def find_model_files(symbol, strategy_name):
    """
    ✅ Busca archivos de modelo en múltiples ubicaciones posibles
    """
    possible_locations = [
        f"models/{strategy_name}/{symbol}_feature_scaler.pkl",
        f"models/{symbol}_{strategy_name}_feature_scaler.pkl",
        f"models/{strategy_name}_feature_scaler.pkl",
        f"models/scaler_{symbol}_{strategy_name}.pkl"
    ]
    
    model_features_locations = [
        f"models/{strategy_name}/{symbol}_model_features.joblib",
        f"models/{symbol}_{strategy_name}_model_features.joblib", 
        f"models/{strategy_name}_model_features.joblib",
        f"models/features_{symbol}_{strategy_name}.joblib"
    ]
    
    scaler_path = None
    features_path = None
    
    for path in possible_locations:
        if os.path.exists(path):
            scaler_path = path
            break
    
    for path in model_features_locations:
        if os.path.exists(path):
            features_path = path
            break
    
    return scaler_path, features_path

def populate_database(symbol, strategy_name):
    """
    ✅ Carga los datos de features, los normaliza y los inserta en ChromaDB.
    Versión adaptada con mejor manejo de errores y rutas flexibles.
    """
    print(f"🔄 Poblando ChromaDB para {symbol} / {strategy_name}")
    print("-" * 50)

    # ✅ RUTAS DE ARCHIVOS CON BÚSQUEDA FLEXIBLE
    features_path = f"data/features/{symbol}_features.parquet"
    
    if not os.path.exists(features_path):
        print(f"❌ No se encontró archivo de features: {features_path}")
        print("💡 Ejecuta 'python generate_features.py' primero")
        return False

    # Buscar archivos de modelo en múltiples ubicaciones
    scaler_path, model_features_path = find_model_files(symbol, strategy_name)
    
    if not scaler_path:
        print(f"❌ No se encontró scaler para {symbol}_{strategy_name}")
        print("💡 Ejecuta el entrenamiento de modelos primero")
        return False
        
    if not model_features_path:
        print(f"❌ No se encontraron model_features para {symbol}_{strategy_name}")
        print("💡 Ejecuta el entrenamiento de modelos primero")
        return False

    print(f"📁 Archivos encontrados:")
    print(f"   Features: {features_path}")
    print(f"   Scaler: {scaler_path}")
    print(f"   Model Features: {model_features_path}")

    try:
        # ✅ CARGAR DATOS
        print("📊 Cargando datos de features...")
        df = pd.read_parquet(features_path)
        print(f"   Registros cargados: {len(df)}")
        
        print("⚙️ Cargando scaler y features del modelo...")
        scaler = joblib.load(scaler_path)
        model_features = joblib.load(model_features_path)
        print(f"   Features del modelo: {len(model_features)}")

        # ✅ VERIFICAR COMPATIBILIDAD DE FEATURES
        missing_features = [f for f in model_features if f not in df.columns]
        if missing_features:
            print(f"❌ Features faltantes en el dataset: {missing_features}")
            print("💡 Regenera los features o re-entrena el modelo")
            return False

        # ✅ PREPARAR DATOS PARA CHROMADB
        print("🔧 Preparando datos para ChromaDB...")
        df_model_data = df[model_features].copy()
        
        # Verificar y limpiar datos
        initial_rows = len(df_model_data)
        df_model_data = df_model_data.dropna()
        final_rows = len(df_model_data)
        
        if final_rows < initial_rows:
            print(f"⚠️ Removidos {initial_rows - final_rows} registros con NaN")
        
        if final_rows == 0:
            print("❌ No hay datos válidos para procesar")
            return False

        print("📐 Normalizando datos con el scaler del modelo...")
        data_scaled = scaler.transform(df_model_data)
        print(f"   Datos normalizados: {data_scaled.shape}")

        # ✅ CONFIGURAR CHROMADB
        print("💾 Configurando ChromaDB...")
        db_path = "db/chroma_db"
        os.makedirs(db_path, exist_ok=True)
        
        client = chromadb.PersistentClient(path=db_path)
        collection_name = "historical_market_states"
        collection = client.get_or_create_collection(name=collection_name)
        
        initial_count = collection.count()
        print(f"   Vectores existentes en colección: {initial_count}")

        # ✅ PREPARAR METADATOS
        print("📋 Preparando metadatos...")
        
        # Filtrar df para que coincida con df_model_data después del dropna
        df_filtered = df.loc[df_model_data.index]
        
        metadatas = []
        ids = []
        
        for idx, row in df_filtered.iterrows():
            # Manejar target de forma segura
            target_value = row.get('target', 0)
            if pd.isna(target_value):
                target_value = 0
            
            # Crear metadata
            metadata = {
                "symbol": symbol,
                "strategy": strategy_name,
                "timestamp": str(idx),
                "outcome": int(target_value)
            }
            metadatas.append(metadata)
            
            # Crear ID único
            if hasattr(idx, 'strftime'):
                timestamp_str = idx.strftime('%Y%m%d%H%M%S')
            else:
                timestamp_str = str(idx).replace(' ', '_').replace(':', '').replace('-', '')
            
            id_str = f"{symbol}_{strategy_name}_{timestamp_str}"
            ids.append(id_str)

        # ✅ INSERTAR EN LOTES
        print(f"🚀 Insertando {len(data_scaled)} vectores en ChromaDB...")
        
        batch_size = 3000  # Reducido para mejor estabilidad
        successful_inserts = 0
        
        for i in tqdm(range(0, len(data_scaled), batch_size), desc="Insertando lotes"):
            batch_end = min(i + batch_size, len(data_scaled))
            
            try:
                # Preparar lote
                batch_embeddings = data_scaled[i:batch_end].tolist()
                batch_metadatas = metadatas[i:batch_end]
                batch_ids = ids[i:batch_end]
                
                # Verificar que no hay IDs duplicados en este lote
                unique_ids = list(set(batch_ids))
                if len(unique_ids) != len(batch_ids):
                    print(f"⚠️ IDs duplicados en lote {i//batch_size + 1}, limpiando...")
                    # Usar solo IDs únicos
                    seen = set()
                    clean_embeddings = []
                    clean_metadatas = []
                    clean_ids = []
                    
                    for j, id_val in enumerate(batch_ids):
                        if id_val not in seen:
                            seen.add(id_val)
                            clean_embeddings.append(batch_embeddings[j])
                            clean_metadatas.append(batch_metadatas[j])
                            clean_ids.append(id_val)
                    
                    batch_embeddings = clean_embeddings
                    batch_metadatas = clean_metadatas
                    batch_ids = clean_ids

                # Insertar lote
                collection.add(
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                
                successful_inserts += len(batch_embeddings)
                
            except Exception as e:
                print(f"❌ Error en lote {i//batch_size + 1}: {e}")
                # Continuar con el siguiente lote
                continue

        # ✅ VERIFICAR RESULTADOS
        final_count = collection.count()
        new_vectors = final_count - initial_count
        
        print(f"\n✅ Proceso completado!")
        print(f"   Vectores insertados exitosamente: {successful_inserts}")
        print(f"   Total vectores en colección: {final_count}")
        print(f"   Nuevos vectores añadidos: {new_vectors}")
        
        if successful_inserts > 0:
            # ✅ FIX: Verificar calidad de los datos insertados con sintaxis correcta de ChromaDB
            try:
                sample_results = collection.query(
                    query_embeddings=[data_scaled[0].tolist()],
                    n_results=3,
                    where={
                        "$and": [
                            {"symbol": {"$eq": symbol}},
                            {"strategy": {"$eq": strategy_name}}
                        ]
                    }
                )
                
                if sample_results and len(sample_results['ids'][0]) > 0:
                    print(f"✅ Verificación exitosa: ChromaDB respondió con {len(sample_results['ids'][0])} resultados")
                else:
                    print(f"⚠️ Advertencia: ChromaDB no devolvió resultados en la verificación")
                    
            except Exception as e:
                print(f"⚠️ Error en verificación (no crítico): {e}")
                # Intentar verificación simple sin filtros
                try:
                    simple_results = collection.query(
                        query_embeddings=[data_scaled[0].tolist()],
                        n_results=1
                    )
                    if simple_results and len(simple_results['ids'][0]) > 0:
                        print(f"✅ Verificación simple exitosa: ChromaDB funciona correctamente")
                    else:
                        print(f"⚠️ ChromaDB no respondió en verificación simple")
                except Exception as e2:
                    print(f"⚠️ Error en verificación simple: {e2}")
            
            return True
        else:
            print(f"❌ No se insertaron vectores exitosamente")
            return False

    except Exception as e:
        print(f"❌ Error durante la población de ChromaDB: {e}")
        import traceback
        traceback.print_exc()
        return False

def populate_all_strategies():
    """
    ✅ Popula ChromaDB para todas las estrategias entrenadas
    """
    print("🚀 POBLANDO CHROMADB PARA TODAS LAS ESTRATEGIAS")
    print("=" * 60)
    
    # Buscar archivos de modelos entrenados
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("❌ Directorio models/ no encontrado")
        print("💡 Entrena modelos primero con 'python run_pipeline.py'")
        return
    
    # Encontrar todas las combinaciones entrenadas
    trained_combinations = []
    
    for strategy_dir in os.listdir(models_dir):
        strategy_path = os.path.join(models_dir, strategy_dir)
        if os.path.isdir(strategy_path):
            # Buscar archivos de scaler en este directorio de estrategia
            scaler_files = [f for f in os.listdir(strategy_path) if f.endswith('_feature_scaler.pkl')]
            
            for scaler_file in scaler_files:
                # Extraer símbolo del nombre del archivo
                symbol = scaler_file.replace('_feature_scaler.pkl', '')
                trained_combinations.append((symbol, strategy_dir))
    
    if not trained_combinations:
        print("❌ No se encontraron modelos entrenados")
        print("💡 Ejecuta 'python run_pipeline.py' primero")
        return
    
    print(f"📊 Combinaciones encontradas: {len(trained_combinations)}")
    for symbol, strategy in trained_combinations:
        print(f"   • {symbol}_{strategy}")
    
    # Poplar ChromaDB para cada combinación
    successful = 0
    failed = 0
    
    for symbol, strategy in trained_combinations:
        print(f"\n[{successful + failed + 1}/{len(trained_combinations)}] Procesando {symbol}_{strategy}")
        
        success = populate_database(symbol, strategy)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\n" + "=" * 60)
    print(f"📊 RESUMEN DE POBLACIÓN CHROMADB")
    print(f"=" * 60)
    print(f"✅ Exitosos: {successful}")
    print(f"❌ Fallidos: {failed}")
    print(f"📈 Tasa de éxito: {successful/(successful+failed)*100:.1f}%")
    
    if successful > 0:
        print(f"\n🎉 ChromaDB poblado exitosamente!")
        print(f"🚀 Sistema listo para trading en vivo")
    else:
        print(f"\n❌ Error poblando ChromaDB")

if __name__ == "__main__":
    populate_all_strategies()