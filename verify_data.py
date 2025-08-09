#!/usr/bin/env python3
"""
Script para verificar la calidad de los datos consolidados
"""

import pandas as pd
from pathlib import Path

def verify_consolidated_data(candles_dir: str = "data/candles"):
    """
    Verifica la calidad de los datos consolidados
    """
    candles_path = Path(candles_dir)
    
    # Buscar archivos consolidados
    consolidated_files = list(candles_path.glob("*_M5.parquet"))
    
    print("=== VERIFICACIÓN DE DATOS CONSOLIDADOS ===\n")
    
    for file in sorted(consolidated_files):
        symbol = file.stem.replace("_M5", "")
        print(f"--- {symbol} ---")
        
        try:
            df = pd.read_parquet(file)
            
            # Información básica
            print(f"Filas: {len(df)}")
            print(f"Columnas: {list(df.columns)}")
            
            # Verificar timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            print(f"Período: {df['timestamp'].min()} -> {df['timestamp'].max()}")
            
            # Verificar continuidad (gaps)
            time_diffs = df['timestamp'].diff().dropna()
            expected_diff = pd.Timedelta(minutes=5)  # M5
            
            gaps = time_diffs[time_diffs > expected_diff * 1.5]  # Permitir 50% tolerancia
            print(f"Gaps detectados: {len(gaps)}")
            
            if len(gaps) > 0:
                print(f"  Gaps más grandes:")
                for i, gap in gaps.head(3).items():
                    gap_time = df.loc[i, 'timestamp']
                    print(f"    {gap_time}: {gap}")
            
            # Verificar precios
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    prices = df[col].dropna()
                    print(f"{col.upper()}: min={prices.min():.5f}, max={prices.max():.5f}, avg={prices.mean():.5f}")
            
            # Verificar ATR
            if 'atr' in df.columns:
                atr_valid = df['atr'].dropna()
                if len(atr_valid) > 0:
                    print(f"ATR: min={atr_valid.min():.6f}, max={atr_valid.max():.6f}, avg={atr_valid.mean():.6f}")
                else:
                    print("ATR: Todos los valores son NaN")
            
            # Verificar consistencia OHLC
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                # High debe ser >= que Open, Close
                high_issues = ((df['high'] < df['open']) | (df['high'] < df['close'])).sum()
                # Low debe ser <= que Open, Close  
                low_issues = ((df['low'] > df['open']) | (df['low'] > df['close'])).sum()
                
                print(f"Inconsistencias OHLC: High={high_issues}, Low={low_issues}")
            
            # Estadísticas de variación
            if 'close' in df.columns:
                returns = df['close'].pct_change().dropna()
                if len(returns) > 0:
                    print(f"Retornos: std={returns.std():.6f}, min={returns.min():.6f}, max={returns.max():.6f}")
            
            print(f"Archivo: {file.name} ({file.stat().st_size / 1024:.1f} KB)")
            
        except Exception as e:
            print(f"ERROR leyendo {file}: {e}")
        
        print()

if __name__ == "__main__":
    verify_consolidated_data()