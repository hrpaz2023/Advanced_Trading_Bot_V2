#!/usr/bin/env python3
"""
Script para consolidar archivos de velas desde directorios por símbolo.
Estructura: data/candles/SYMBOL/SYMBOL_timestamp.parquet -> data/candles/SYMBOL_M5.parquet
"""

import pandas as pd
from pathlib import Path
import argparse

def consolidate_candles(candles_dir: str, timeframe: str = "M5"):
    """
    Consolida archivos de velas por símbolo desde subdirectorios
    """
    candles_path = Path(candles_dir)
    
    print(f"=== CONSOLIDACIÓN DE VELAS ===")
    print(f"Directorio base: {candles_path.absolute()}")
    
    if not candles_path.exists():
        print(f"ERROR: El directorio {candles_path} no existe")
        return
    
    # Buscar subdirectorios de símbolos
    symbol_dirs = [d for d in candles_path.iterdir() if d.is_dir()]
    print(f"Directorios de símbolos encontrados: {len(symbol_dirs)}")
    
    for symbol_dir in sorted(symbol_dirs):
        symbol = symbol_dir.name.upper()
        print(f"\n--- Procesando {symbol} ---")
        print(f"Directorio: {symbol_dir}")
        
        # Buscar archivos .parquet en el directorio del símbolo
        parquet_files = list(symbol_dir.glob("*.parquet"))
        print(f"Archivos .parquet encontrados: {len(parquet_files)}")
        
        if not parquet_files:
            print(f"  No hay archivos .parquet en {symbol_dir}")
            # Mostrar qué hay en el directorio
            all_files = list(symbol_dir.iterdir())
            print(f"  Contenido del directorio: {[f.name for f in all_files[:5]]}")
            continue
        
        # Mostrar algunos ejemplos
        print(f"  Ejemplos: {[f.name for f in parquet_files[:3]]}")
        if len(parquet_files) > 3:
            print(f"  ... y {len(parquet_files) - 3} más")
        
        # Leer y concatenar todos los archivos
        dfs = []
        total_rows = 0
        
        for file in sorted(parquet_files):
            try:
                df = pd.read_parquet(file)
                rows = len(df)
                total_rows += rows
                dfs.append(df)
                print(f"    ✓ {file.name}: {rows} filas")
            except Exception as e:
                print(f"    ✗ Error en {file.name}: {e}")
        
        if not dfs:
            print(f"  No se pudieron leer archivos para {symbol}")
            continue
        
        print(f"  Total filas leídas: {total_rows}")
        
        # Concatenar todos los DataFrames
        df_combined = pd.concat(dfs, ignore_index=True)
        
        # Normalizar columna de timestamp
        timestamp_cols = [c for c in df_combined.columns if 'time' in c.lower()]
        print(f"  Columnas de tiempo: {timestamp_cols}")
        
        if "timestamp" in df_combined.columns:
            time_col = "timestamp"
        elif "time" in df_combined.columns:
            time_col = "time"
        elif timestamp_cols:
            time_col = timestamp_cols[0]
        else:
            print(f"  ERROR: No se encontró columna de tiempo en {df_combined.columns}")
            continue
        
        # Convertir a datetime UTC
        df_combined[time_col] = pd.to_datetime(df_combined[time_col], utc=True)
        df_combined = df_combined.rename(columns={time_col: "timestamp"})
        
        # Eliminar duplicados y ordenar
        initial_rows = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        final_rows = len(df_combined)
        
        print(f"  Filas después de limpiar: {final_rows} (eliminados {initial_rows - final_rows} duplicados)")
        
        # Validar que tenemos las columnas OHLC
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df_combined.columns]
        
        if missing_cols:
            print(f"  ERROR: Faltan columnas: {missing_cols}")
            print(f"  Columnas disponibles: {list(df_combined.columns)}")
            continue
        
        # Convertir precios a numérico
        for col in required_cols:
            df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
        
        # Calcular ATR si no existe
        if "atr" not in df_combined.columns:
            try:
                tr1 = (df_combined["high"] - df_combined["low"]).abs()
                tr2 = (df_combined["high"] - df_combined["close"].shift(1)).abs()
                tr3 = (df_combined["low"] - df_combined["close"].shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                df_combined["atr"] = tr.rolling(14, min_periods=1).mean()
                print(f"    ✓ ATR calculado")
            except Exception as e:
                print(f"    ✗ Error calculando ATR: {e}")
                df_combined["atr"] = 0.0
        
        # Guardar archivo consolidado en el directorio base
        output_file = candles_path / f"{symbol}_{timeframe.upper()}.parquet"
        
        try:
            df_combined.to_parquet(output_file, index=False)
            print(f"  ✓ Guardado: {output_file.name}")
            print(f"    Filas finales: {len(df_combined)}")
            
            # Mostrar rango de fechas
            if len(df_combined) > 0:
                start_date = df_combined['timestamp'].min()
                end_date = df_combined['timestamp'].max()
                print(f"    Período: {start_date.strftime('%Y-%m-%d %H:%M')} -> {end_date.strftime('%Y-%m-%d %H:%M')}")
                
                # Mostrar estadísticas básicas
                print(f"    Precio promedio: {df_combined['close'].mean():.5f}")
                print(f"    ATR promedio: {df_combined['atr'].mean():.5f}")
                
        except Exception as e:
            print(f"  ✗ Error guardando {output_file}: {e}")
    
    print(f"\n=== CONSOLIDACIÓN COMPLETADA ===")
    
    # Mostrar archivos finales creados
    consolidated_files = list(candles_path.glob("*_*.parquet"))
    print(f"Archivos consolidados creados: {len(consolidated_files)}")
    for f in consolidated_files:
        print(f"  {f.name}")

def main():
    parser = argparse.ArgumentParser(description="Consolidar archivos de velas desde subdirectorios por símbolo")
    parser.add_argument("--candles-dir", default="data/candles", help="Directorio base de velas")
    parser.add_argument("--timeframe", default="M5", help="Timeframe para el nombre del archivo")
    
    args = parser.parse_args()
    
    consolidate_candles(args.candles_dir, args.timeframe)

if __name__ == "__main__":
    main()