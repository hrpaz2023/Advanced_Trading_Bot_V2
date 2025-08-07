import pandas as pd
import numpy as np
import os
import sys

# ✅ IMPORTACIÓN ROBUSTA
try:
    from .feature_calculator import add_all_features, verify_optuna_compatibility, test_optuna_compatibility
except ImportError:
    # Fallback para ejecución directa
    try:
        from feature_calculator import add_all_features, verify_optuna_compatibility, test_optuna_compatibility
    except ImportError:
        print("❌ No se puede importar feature_calculator")
        sys.exit(1)

def generate_target(df, forward_candles=20, risk_reward_ratio=1.5, atr_multiplier=1.2):
    """
    ✅ GENERADOR DE TARGET MEJORADO Y CONSERVADOR
    Genera la columna 'target' sin sesgo de anticipación (lookahead bias)
    
    Args:
        df: DataFrame con datos OHLC y ATR
        forward_candles: Velas futuras a evaluar (20 = más conservador)
        risk_reward_ratio: Ratio riesgo/recompensa (1.5 = más conservador)
        atr_multiplier: Multiplicador para ATR (1.2 = más conservador)
    
    Returns:
        DataFrame con columna 'target' añadida
    """
    print(f"🎯 Generando targets conservadores: lookforward {forward_candles} velas, R/R {risk_reward_ratio}")
    
    df = df.copy()
    df['target'] = 0
    
    try:
        # ✅ USAR ATR DISPONIBLE DE FORMA ROBUSTA
        atr_column = None
        possible_atr_cols = ['atr', 'atr_14', 'atr_20', 'ATR']
        
        for col in possible_atr_cols:
            if col in df.columns and not df[col].isna().all():
                atr_column = col
                break
        
        if atr_column is None:
            print("⚠️ No se encontró columna ATR válida, calculando ATR simple")
            # Calcular ATR básico si no existe
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift(1))
            low_close = abs(df['Low'] - df['Close'].shift(1))
            
            true_range = pd.DataFrame([high_low, high_close, low_close]).max()
            df['atr_temp'] = true_range.rolling(14).mean()
            atr_column = 'atr_temp'
        
        # ✅ GENERAR TARGETS DE FORMA VECTORIZADA (MÁS EFICIENTE)
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        atrs = df[atr_column].values
        targets = np.zeros(len(df))
        
        print(f"📊 Procesando {len(df)} registros...")
        
        # Procesar en lotes para mayor eficiencia
        valid_indices = len(df) - forward_candles
        
        for i in range(valid_indices):
            if i % 1000 == 0:
                progress = (i / valid_indices) * 100
                print(f"⏳ Progreso: {progress:.1f}%")
            
            current_close = closes[i]
            current_atr = atrs[i]
            
            # Validar ATR
            if pd.isna(current_atr) or current_atr <= 0:
                continue
            
            # ✅ CÁLCULO DE NIVELES CONSERVADORES
            stop_loss_price = current_close - (current_atr * atr_multiplier)
            take_profit_price = current_close + (current_atr * atr_multiplier * risk_reward_ratio)
            
            # Evaluar velas futuras
            future_highs = highs[i+1:i+forward_candles+1]
            future_lows = lows[i+1:i+forward_candles+1]
            
            # Verificar si se alcanza TP o SL primero
            tp_hit_indices = np.where(future_highs >= take_profit_price)[0]
            sl_hit_indices = np.where(future_lows <= stop_loss_price)[0]
            
            if len(tp_hit_indices) > 0 and len(sl_hit_indices) > 0:
                # Ambos golpeados, verificar cuál primero
                first_tp = tp_hit_indices[0]
                first_sl = sl_hit_indices[0]
                
                if first_tp < first_sl:
                    targets[i] = 1  # TP primero
                elif first_sl < first_tp:
                    targets[i] = -1  # SL primero
                # Si mismo índice, no asignar target (empate)
                
            elif len(tp_hit_indices) > 0:
                targets[i] = 1  # Solo TP golpeado
            elif len(sl_hit_indices) > 0:
                targets[i] = -1  # Solo SL golpeado
            # Si ninguno golpeado, target = 0 (no trade)
        
        df['target'] = targets
        
        # ✅ ESTADÍSTICAS DEL TARGET
        total_signals = len(df[df['target'] != 0])
        long_signals = len(df[df['target'] == 1])
        short_signals = len(df[df['target'] == -1])
        
        print(f"📊 Estadísticas de targets:")
        print(f"   Total señales: {total_signals} ({total_signals/len(df)*100:.1f}%)")
        print(f"   Señales LONG: {long_signals} ({long_signals/total_signals*100:.1f}%)" if total_signals > 0 else "   Señales LONG: 0")
        print(f"   Señales SHORT: {short_signals} ({short_signals/total_signals*100:.1f}%)" if total_signals > 0 else "   Señales SHORT: 0")
        
        # Limpiar columna temporal si se creó
        if 'atr_temp' in df.columns:
            df.drop('atr_temp', axis=1, inplace=True)
        
        return df
        
    except Exception as e:
        print(f"❌ Error generando targets: {e}")
        import traceback
        traceback.print_exc()
        df['target'] = 0  # Fallback seguro
        return df

def detect_file_format(file_path):
    """
    ✅ DETECTA AUTOMÁTICAMENTE EL FORMATO DE ARCHIVO
    
    Returns:
        dict: {'separator': str, 'encoding': str, 'decimal': str}
    """
    
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    separators_to_try = ['\t', ',', ';', '|']
    
    for encoding in encodings_to_try:
        for separator in separators_to_try:
            try:
                # Leer solo las primeras filas para detectar formato
                test_df = pd.read_csv(file_path, sep=separator, encoding=encoding, nrows=5)
                
                # Verificar que tenga columnas razonables
                if len(test_df.columns) >= 4:  # Al menos OHLC
                    # Verificar si las columnas parecen precios
                    numeric_cols = test_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) >= 4:
                        return {
                            'separator': separator,
                            'encoding': encoding,
                            'decimal': '.'
                        }
                        
            except Exception:
                continue
    
    # Fallback por defecto
    return {'separator': '\t', 'encoding': 'utf-8', 'decimal': '.'}

def process_symbol(symbol, raw_data_dir, features_dir, target_params=None):
    """
    ✅ PROCESADOR DE SÍMBOLO MEJORADO
    Procesa un símbolo completo: carga, añade features, genera target y guarda
    
    Args:
        symbol: Símbolo a procesar (ej: 'EURUSD')
        raw_data_dir: Directorio de datos raw
        features_dir: Directorio de destino para features
        target_params: Parámetros para generación de target
    """
    
    if target_params is None:
        target_params = {
            'forward_candles': 20,    # Más conservador
            'risk_reward_ratio': 1.5, # Más conservador
            'atr_multiplier': 1.2     # Más conservador
        }
    
    # ✅ BUSCAR ARCHIVO CON DIFERENTES FORMATOS
    possible_files = [
        f"{symbol}_5m.csv",
        f"{symbol}_M5.csv", 
        f"{symbol}.csv",
        f"{symbol}_5min.csv",
        f"{symbol.lower()}_5m.csv"
    ]
    
    raw_file_path = None
    for filename in possible_files:
        potential_path = os.path.join(raw_data_dir, filename)
        if os.path.exists(potential_path):
            raw_file_path = potential_path
            break
    
    if raw_file_path is None:
        print(f"❌ No se encontró archivo de datos para {symbol}")
        print(f"   Buscados: {possible_files}")
        return False
    
    feature_file_path = os.path.join(features_dir, f"{symbol}_features.parquet")
    
    print(f"🔄 Procesando {symbol}")
    print(f"📁 Archivo fuente: {os.path.basename(raw_file_path)}")
    
    try:
        # ✅ DETECTAR FORMATO AUTOMÁTICAMENTE
        file_format = detect_file_format(raw_file_path)
        print(f"📊 Formato detectado: sep='{file_format['separator']}', encoding='{file_format['encoding']}'")
        
        # ✅ CARGAR DATOS CON MANEJO ROBUSTO
        df = pd.read_csv(
            raw_file_path, 
            sep=file_format['separator'],
            encoding=file_format['encoding'],
            decimal=file_format['decimal']
        )
        
        print(f"📊 Datos cargados: {len(df)} registros, {len(df.columns)} columnas")
        print(f"📈 Columnas: {list(df.columns)}")
        
        # ✅ VERIFICAR DATOS VÁLIDOS
        if len(df) < 100:
            print(f"⚠️ Datos insuficientes para {symbol}: {len(df)} registros")
            return False
        
        # Verificar columnas OHLC básicas
        required_patterns = ['open', 'high', 'low', 'close']
        column_mapping = {}
        
        for pattern in required_patterns:
            found = False
            for col in df.columns:
                if pattern.lower() in col.lower():
                    column_mapping[col] = pattern.capitalize()
                    found = True
                    break
            
            if not found:
                print(f"❌ No se encontró columna para {pattern} en {symbol}")
                return False
        
        # Aplicar mapping si es necesario
        if column_mapping:
            df = df.rename(columns=column_mapping)
            print(f"📊 Columnas renombradas: {column_mapping}")
        
    except Exception as e:
        print(f"❌ Error cargando datos para {symbol}: {e}")
        return False
    
    try:
        # ✅ CALCULAR FEATURES
        print("⚙️ Calculando indicadores técnicos...")
        df_features = add_all_features(df.copy())
        
        if df_features.empty:
            print(f"❌ No se pudieron calcular features para {symbol}")
            return False
        
        print(f"✅ Features calculados: {len(df_features)} registros válidos")
        
    except Exception as e:
        print(f"❌ Error calculando features para {symbol}: {e}")
        return False
    
    try:
        # ✅ GENERAR TARGETS
        print("🎯 Generando targets...")
        df_final = generate_target(df_features.copy(), **target_params)
        
        print(f"✅ Targets generados")
        
    except Exception as e:
        print(f"❌ Error generando targets para {symbol}: {e}")
        df_final = df_features  # Usar sin targets si falla
    
    try:
        # ✅ GUARDAR FEATURES
        print(f"💾 Guardando features en {feature_file_path}")
        df_final.to_parquet(feature_file_path, compression='snappy')
        
        # ✅ VERIFICAR ARCHIVO GUARDADO
        file_size = os.path.getsize(feature_file_path) / (1024 * 1024)  # MB
        print(f"✅ Archivo guardado: {file_size:.1f} MB")
        
        # ✅ VERIFICAR FEATURES PARA OPTUNA
        print(f"🔍 Verificando compatibilidad con OPTUNA:")
        
        # Probar algunos casos típicos de parámetros que puede sugerir Optuna
        optuna_test_cases = [
            ('ema_crossover', {'fast_period': 25, 'slow_period': 125}),
            ('rsi_pullback', {'rsi_period': 15, 'trend_ema_period': 150}),
            ('volatility_breakout', {'atr_period': 15}),
            ('multi_filter_scalper', {'ema_fast': 12, 'ema_mid': 30, 'rsi_len': 8, 'atr_len': 16})
        ]
        
        compatible_count = 0
        for strategy, params in optuna_test_cases:
            compatible, missing = verify_optuna_compatibility(df_final, strategy, params)
            status = "✅" if compatible else "⚠️"
            print(f"   {status} {strategy} (test params)")
            if compatible:
                compatible_count += 1
            elif missing and len(missing) <= 3:
                print(f"       Faltantes: {missing}")
        
        compatibility_pct = (compatible_count / len(optuna_test_cases)) * 100
        print(f"   📊 Compatibilidad OPTUNA: {compatibility_pct:.0f}%")
        
        print(f"✅ Procesamiento de {symbol} completado\n")
        return True
        
    except Exception as e:
        print(f"❌ Error guardando features para {symbol}: {e}")
        return False

def batch_process_symbols(raw_data_dir='data/raw', features_dir='data/features', target_params=None):
    """
    ✅ PROCESAMIENTO EN LOTE DE TODOS LOS SÍMBOLOS
    
    Args:
        raw_data_dir: Directorio de datos raw
        features_dir: Directorio de destino
        target_params: Parámetros para targets
    
    Returns:
        dict: Resultados del procesamiento
    """
    
    print("🚀 GENERADOR DE FEATURES MEJORADO")
    print("=" * 60)
    
    # ✅ CREAR DIRECTORIO DE FEATURES
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
        print(f"📁 Directorio creado: {features_dir}")
    
    # ✅ BUSCAR ARCHIVOS DE DATOS
    if not os.path.exists(raw_data_dir):
        print(f"❌ Directorio de datos no encontrado: {raw_data_dir}")
        return {'success': False, 'error': 'Raw data directory not found'}
    
    print(f"🔍 Buscando archivos en: {raw_data_dir}")
    
    all_files = os.listdir(raw_data_dir)
    csv_files = [f for f in all_files if f.endswith('.csv')]
    
    if not csv_files:
        print(f"❌ No se encontraron archivos .csv en {raw_data_dir}")
        return {'success': False, 'error': 'No CSV files found'}
    
    print(f"📊 Archivos encontrados: {len(csv_files)}")
    for f in csv_files:
        print(f"   📄 {f}")
    
    # ✅ EXTRAER SÍMBOLOS DE NOMBRES DE ARCHIVO
    symbols_to_process = []
    
    for csv_file in csv_files:
        # Intentar extraer símbolo del nombre del archivo
        base_name = csv_file.replace('.csv', '')
        
        # Patrones comunes: SYMBOL_5m, SYMBOL_M5, SYMBOL
        possible_symbol = base_name.replace('_5m', '').replace('_M5', '').replace('_5min', '')
        
        if len(possible_symbol) >= 3:  # Mínimo 3 caracteres para símbolo
            symbols_to_process.append(possible_symbol.upper())
    
    # Eliminar duplicados
    symbols_to_process = list(set(symbols_to_process))
    
    if not symbols_to_process:
        print("❌ No se pudieron extraer símbolos válidos de los archivos")
        return {'success': False, 'error': 'No valid symbols extracted'}
    
    print(f"\n🎯 Símbolos a procesar: {symbols_to_process}")
    
    # ✅ PARÁMETROS CONSERVADORES PARA TARGETS
    if target_params is None:
        target_params = {
            'forward_candles': 20,       # 20 velas futuras (1.7 horas en M5) - más conservador
            'risk_reward_ratio': 1.5,    # R/R 1:1.5 - más conservador
            'atr_multiplier': 1.2        # 1.2x ATR para SL - más conservador
        }
    
    print(f"\n⚙️ Parámetros conservadores de target:")
    for key, value in target_params.items():
        print(f"   {key}: {value}")
    
    # ✅ PROCESAR SÍMBOLOS
    results = {
        'processed': [],
        'failed': [],
        'total_files': 0,
        'total_size_mb': 0
    }
    
    print(f"\n🔄 Iniciando procesamiento...")
    
    for i, symbol in enumerate(symbols_to_process, 1):
        print(f"\n[{i}/{len(symbols_to_process)}] Procesando {symbol}")
        print("-" * 40)
        
        try:
            success = process_symbol(symbol, raw_data_dir, features_dir, target_params)
            
            if success:
                results['processed'].append(symbol)
                
                # Calcular tamaño del archivo generado
                feature_file = os.path.join(features_dir, f"{symbol}_features.parquet")
                if os.path.exists(feature_file):
                    file_size = os.path.getsize(feature_file) / (1024 * 1024)
                    results['total_size_mb'] += file_size
                    results['total_files'] += 1
                
            else:
                results['failed'].append(symbol)
                
        except Exception as e:
            print(f"❌ Error procesando {symbol}: {e}")
            results['failed'].append(symbol)
    
    # ✅ RESUMEN FINAL
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE PROCESAMIENTO")
    print("=" * 60)
    
    print(f"✅ Procesados exitosamente: {len(results['processed'])}")
    for symbol in results['processed']:
        print(f"   📈 {symbol}")
    
    if results['failed']:
        print(f"\n❌ Fallidos: {len(results['failed'])}")
        for symbol in results['failed']:
            print(f"   📉 {symbol}")
    
    print(f"\n📊 Estadísticas:")
    print(f"   📁 Archivos generados: {results['total_files']}")
    print(f"   💾 Tamaño total: {results['total_size_mb']:.1f} MB")
    print(f"   📈 Tasa de éxito: {len(results['processed'])/len(symbols_to_process)*100:.1f}%")
    
    # ✅ VERIFICAR COMPATIBILIDAD CON ESTRATEGIAS
    if results['processed']:
        print(f"\n🔍 Verificando compatibilidad con estrategias:")
        
        # Cargar primer archivo para verificar
        test_symbol = results['processed'][0]
        test_file = os.path.join(features_dir, f"{test_symbol}_features.parquet")
        
        try:
            test_df = pd.read_parquet(test_file)
            
            # ✅ PROBAR COMPATIBILIDAD CON OPTUNA
            print(f"🔍 Verificando compatibilidad OPTUNA con archivo de ejemplo:")
            
            # Usar la función de testing
            compatible = test_optuna_compatibility() if 'test_optuna_compatibility' in globals() else True
            
            if compatible:
                print(f"   ✅ Totalmente compatible con OPTUNA")
            else:
                print(f"   ⚠️ Puede haber incompatibilidades menores")
                
        except Exception as e:
            print(f"   ⚠️ Error verificando compatibilidad: {e}")
    
    results['success'] = len(results['processed']) > 0
    
    if results['success']:
        print(f"\n🎉 Procesamiento completado exitosamente!")
        print(f"🚀 Ejecuta 'python run_optimization.py' para optimizar estrategias")
    else:
        print(f"\n❌ Procesamiento falló para todos los símbolos")
    
    return results

def main():
    """Función principal para ejecución directa"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Generador de Features para Trading Bot")
    parser.add_argument('--raw-dir', default='data/raw', help='Directorio de datos raw')
    parser.add_argument('--features-dir', default='data/features', help='Directorio de features')
    parser.add_argument('--forward-candles', type=int, default=20, help='Velas futuras para target')
    parser.add_argument('--risk-reward', type=float, default=1.5, help='Risk/Reward ratio')
    parser.add_argument('--atr-multiplier', type=float, default=1.2, help='Multiplicador ATR')
    
    args = parser.parse_args()
    
    # Parámetros personalizados
    target_params = {
        'forward_candles': args.forward_candles,
        'risk_reward_ratio': args.risk_reward,
        'atr_multiplier': args.atr_multiplier
    }
    
    # Ejecutar procesamiento
    results = batch_process_symbols(
        raw_data_dir=args.raw_dir,
        features_dir=args.features_dir,
        target_params=target_params
    )
    
    # Exit code basado en resultados
    if results['success']:
        sys.exit(0)
    else:
        sys.exit(1)

# ✅ EJECUCIÓN DIRECTA O COMO MÓDULO
if __name__ == '__main__':
    # Si se ejecuta directamente con argumentos, usar argparse
    if len(sys.argv) > 1:
        main()
    else:
        # Ejecución simple sin argumentos
        results = batch_process_symbols()
        
        if not results['success']:
            print("\n💡 AYUDA:")
            print("1. Asegúrate de tener archivos CSV en 'data/raw/'")
            print("2. Los archivos deben tener formato: SYMBOL_5m.csv")
            print("3. Deben contener columnas OHLC (Open, High, Low, Close)")
            
            # Mostrar ejemplo de uso con argumentos
            print(f"\n📖 Uso avanzado:")
            print(f"python {__file__} --raw-dir data/raw --features-dir data/features --forward-candles 20 --risk-reward 1.5")

# ✅ FUNCIÓN DE TESTING
def test_generate_features():
    """Función para probar el generador completo"""
    
    print("🧪 TESTING GENERATE FEATURES")
    print("=" * 50)
    
    # Crear datos de prueba
    test_dir = "test_data"
    raw_dir = os.path.join(test_dir, "raw")
    features_dir = os.path.join(test_dir, "features")
    
    # Crear directorios
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    
    try:
        # Generar datos sintéticos EURUSD
        np.random.seed(42)
        n_samples = 2000
        
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='5T')
        base_price = 1.1000
        returns = np.random.normal(0, 0.0008, n_samples)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Crear OHLC realista
        df_test = pd.DataFrame({
            'Time': dates,
            'Open': prices,
            'High': prices * (1 + np.random.uniform(0, 0.002, n_samples)),
            'Low': prices * (1 - np.random.uniform(0, 0.002, n_samples)),
            'Close': prices * (1 + np.random.normal(0, 0.001, n_samples)),
            'Volume': np.random.randint(1000, 10000, n_samples)
        })
        
        # Ajustar para que High >= max(Open, Close) y Low <= min(Open, Close)
        df_test['High'] = np.maximum(df_test['High'], df_test[['Open', 'Close']].max(axis=1))
        df_test['Low'] = np.minimum(df_test['Low'], df_test[['Open', 'Close']].min(axis=1))
        
        # Guardar archivo de prueba
        test_file = os.path.join(raw_dir, "EURUSD_5m.csv")
        df_test.to_csv(test_file, sep='\t', index=False)
        
        print(f"✅ Archivo de prueba creado: {test_file}")
        print(f"📊 Datos: {len(df_test)} registros")
        
        # Procesar archivo
        results = batch_process_symbols(raw_dir, features_dir)
        
        if results['success']:
            print(f"✅ Testing exitoso!")
            
            # Verificar archivo generado
            feature_file = os.path.join(features_dir, "EURUSD_features.parquet")
            if os.path.exists(feature_file):
                df_features = pd.read_parquet(feature_file)
                print(f"📊 Features generados: {len(df_features)} registros, {len(df_features.columns)} columnas")
                
                # Verificar targets
                if 'target' in df_features.columns:
                    target_stats = df_features['target'].value_counts()
                    print(f"🎯 Target distribution: {dict(target_stats)}")
        else:
            print(f"❌ Testing falló")
        
    except Exception as e:
        print(f"❌ Error en testing: {e}")
    
    finally:
        # Limpiar archivos de prueba
        try:
            import shutil
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
                print(f"🧹 Archivos de prueba eliminados")
        except:
            pass

if __name__ == "__main__" and len(sys.argv) == 1:
    # Solo ejecutar testing si no hay argumentos
    print("🔧 Ejecutando en modo testing...")
    test_generate_features()