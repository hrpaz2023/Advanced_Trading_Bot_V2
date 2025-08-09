#!/usr/bin/env python3
"""
Script para recalcular y arreglar ATR en archivos consolidados
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_atr(df, period=14):
    """
    Calcula ATR (Average True Range) usando el método de Wilder
    """
    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    # True Range es el máximo de los tres
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR usando Exponential Moving Average (método de Wilder)
    # α = 1/period, pero Wilder usa α = 1/period
    alpha = 1.0 / period
    atr = true_range.ewm(alpha=alpha, adjust=False).mean()
    
    return atr

def fix_atr_in_files(candles_dir: str = "data/candles"):
    """
    Recalcula ATR en todos los archivos consolidados
    """
    candles_path = Path(candles_dir)
    consolidated_files = list(candles_path.glob("*_M5.parquet"))
    
    print("=== ARREGLANDO ATR EN ARCHIVOS CONSOLIDADOS ===\n")
    
    for file in sorted(consolidated_files):
        symbol = file.stem.replace("_M5", "")
        print(f"--- {symbol} ---")
        
        try:
            # Leer archivo
            df = pd.read_parquet(file)
            print(f"Filas: {len(df)}")
            
            # Verificar columnas necesarias
            required = ['high', 'low', 'close']
            if not all(col in df.columns for col in required):
                print(f"ERROR: Faltan columnas {required}")
                continue
            
            # Verificar ATR actual
            if 'atr' in df.columns:
                atr_valid = df['atr'].dropna()
                print(f"ATR actual válido: {len(atr_valid)}/{len(df)}")
            
            # Recalcular ATR
            atr_new = calculate_atr(df, period=14)
            df['atr'] = atr_new
            
            # Verificar nuevo ATR
            atr_valid_new = df['atr'].dropna()
            print(f"ATR nuevo válido: {len(atr_valid_new)}/{len(df)}")
            
            if len(atr_valid_new) > 0:
                print(f"ATR stats: min={atr_valid_new.min():.6f}, max={atr_valid_new.max():.6f}, avg={atr_valid_new.mean():.6f}")
                
                # Calcular ATR relativo (ATR / precio)
                if 'close' in df.columns:
                    df['atr_rel'] = df['atr'] / df['close']
                    atr_rel_valid = df['atr_rel'].dropna()
                    if len(atr_rel_valid) > 0:
                        print(f"ATR relativo: avg={atr_rel_valid.mean():.4%}")
            
            # Guardar archivo actualizado
            df.to_parquet(file, index=False)
            print(f"✓ Archivo actualizado: {file.name}")
            
        except Exception as e:
            print(f"ERROR procesando {file}: {e}")
        
        print()

if __name__ == "__main__":
    fix_atr_in_files()