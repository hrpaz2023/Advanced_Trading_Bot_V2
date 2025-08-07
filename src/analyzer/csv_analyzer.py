#!/usr/bin/env python3
"""
Analizador de Resultados CSV
Examina los datos de optimización para entender por qué no hay estrategias viables
"""

import pandas as pd
import numpy as np

def analyze_optimization_results():
    """Analiza los resultados de optimización"""
    print("📊 ANÁLISIS DE RESULTADOS DE OPTIMIZACIÓN FTMO")
    print("=" * 60)
    
    # Cargar datos
    try:
        summary_df = pd.read_csv("ftmo_optimization_summary.csv")
        detailed_df = pd.read_csv("ftmo_optimization_detailed.csv")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    print(f"📋 DATOS CARGADOS:")
    print(f"  - Summary: {len(summary_df)} combinaciones")
    print(f"  - Detailed: {len(detailed_df)} trials")
    
    # Análisis de valores objetivo
    print(f"\n🎯 ANÁLISIS DE VALORES OBJETIVO:")
    print("-" * 40)
    
    obj_values = summary_df['Objective_Value'].values
    print(f"Rango de valores: {obj_values.min():.6f} a {obj_values.max():.6f}")
    print(f"Promedio: {obj_values.mean():.6f}")
    print(f"Desviación estándar: {obj_values.std():.6f}")
    print(f"Mediana: {np.median(obj_values):.6f}")
    
    # Contar valores únicos
    unique_values = np.unique(obj_values)
    print(f"Valores únicos: {len(unique_values)}")
    
    if len(unique_values) <= 10:
        print("Distribución de valores:")
        for val in unique_values:
            count = np.sum(obj_values == val)
            print(f"  {val:.6f}: {count} combinaciones")
    
    # Mostrar mejores y peores
    print(f"\n🏆 TOP 5 MEJORES COMBINACIONES:")
    top_5 = summary_df.nlargest(5, 'Objective_Value')
    for idx, row in top_5.iterrows():
        print(f"  {row['Pair']}_{row['Strategy']}: {row['Objective_Value']:.6f}")
    
    print(f"\n📉 5 PEORES COMBINACIONES:")
    bottom_5 = summary_df.nsmallest(5, 'Objective_Value')
    for idx, row in bottom_5.iterrows():
        print(f"  {row['Pair']}_{row['Strategy']}: {row['Objective_Value']:.6f}")
    
    # Análisis por estrategia y par
    print(f"\n📈 PERFORMANCE POR ESTRATEGIA:")
    strategy_stats = summary_df.groupby('Strategy')['Objective_Value'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(strategy_stats.round(6))
    
    print(f"\n🌍 PERFORMANCE POR PAR:")
    pair_stats = summary_df.groupby('Pair')['Objective_Value'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(pair_stats.round(6))
    
    # Detectar problemas comunes
    print(f"\n🔍 DIAGNÓSTICO DE PROBLEMAS:")
    print("-" * 40)
    
    # Problema 1: Todos los valores iguales
    if len(unique_values) == 1:
        print("❌ PROBLEMA CRÍTICO: Todos los valores objetivo son idénticos!")
        print("   Posibles causas:")
        print("   - Error en la función objetivo")
        print("   - Datos de backtest incorrectos")
        print("   - Problema en el pipeline de optimización")
    
    # Problema 2: Valores muy pequeños o cero
    if obj_values.max() <= 0.001:
        print("⚠️  ADVERTENCIA: Valores objetivo muy pequeños")
        print("   Posibles causas:")
        print("   - Función objetivo mal configurada")
        print("   - Todas las estrategias son no rentables")
        print("   - Escala incorrecta en la función objetivo")
    
    # Problema 3: Poca variación
    if obj_values.std() < 0.001:
        print("⚠️  ADVERTENCIA: Muy poca variación en los resultados")
        print("   Posible causa: Optimización no está funcionando correctamente")
    
    # Análisis de parámetros
    print(f"\n⚙️ ANÁLISIS DE PARÁMETROS:")
    print("-" * 40)
    
    param_cols = [col for col in summary_df.columns if col.startswith('param_') or col in ['period', 'std_dev', 'fast_period', 'slow_period']]
    
    if param_cols:
        print("Rangos de parámetros optimizados:")
        for col in param_cols[:8]:  # Mostrar solo los primeros 8
            if col in summary_df.columns:
                values = summary_df[col].dropna()
                if len(values) > 0:
                    print(f"  {col}: [{values.min():.2f} - {values.max():.2f}] (promedio: {values.mean():.2f})")
    
    # Recomendaciones
    print(f"\n💡 RECOMENDACIONES:")
    print("-" * 40)
    
    if len(unique_values) == 1:
        print("1. Revisar la función objetivo en el script de optimización")
        print("2. Verificar que los backtests se ejecuten correctamente")
        print("3. Comprobar que los datos históricos estén disponibles")
    elif obj_values.max() <= 0:
        print("1. Todas las estrategias son no rentables - revisar criterios")
        print("2. Considerar ajustar parámetros de las estrategias")
        print("3. Verificar período de backtesting")
    elif obj_values.max() > 0:
        viable_count = np.sum(obj_values > 0)
        print(f"✅ {viable_count} estrategias tienen valores positivos")
        print("1. Ajustar umbrales en analyze_results.py")
        print("2. Considerar criterios menos estrictos inicialmente")
    
    return summary_df, detailed_df

def show_sample_data(df, title, n=3):
    """Muestra datos de ejemplo"""
    print(f"\n📋 {title} (primeras {n} filas):")
    print("-" * 60)
    
    # Seleccionar columnas clave
    key_cols = ['Pair', 'Strategy', 'Objective_Value']
    param_cols = [col for col in df.columns if 'period' in col.lower() or 'std' in col.lower()][:3]
    
    display_cols = key_cols + param_cols
    display_cols = [col for col in display_cols if col in df.columns]
    
    print(df[display_cols].head(n).to_string(index=False))

if __name__ == "__main__":
    summary_df, detailed_df = analyze_optimization_results()
    
    if summary_df is not None:
        show_sample_data(summary_df, "MEJORES COMBINACIONES", 5)
        show_sample_data(detailed_df, "TODOS LOS TRIALS", 5)