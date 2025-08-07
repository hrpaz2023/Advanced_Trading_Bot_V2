import pandas as pd
import talib as ta
import numpy as np

# =============================================================================
# OPTIMIZED FEATURE CALCULATOR
# =============================================================================

def get_optuna_parameter_ranges():
    """
    Retorna los rangos EXACTOS que usa Optuna para optimización.
    """
    return {
        'ema_crossover': {'fast_period': range(10, 51), 'slow_period': range(30, 201)},
        'channel_reversal': {'period': range(15, 51), 'std_dev': None},
        'rsi_pullback': {'rsi_level': range(20, 41), 'trend_ema_period': [50, 100, 150, 200], 'rsi_period': range(10, 22)},
        'volatility_breakout': {'period': range(15, 61), 'atr_period': range(10, 22)},
        'multi_filter_scalper': {'ema_fast': range(5, 21), 'ema_mid': range(15, 51), 'ema_slow': range(50, 151), 'rsi_len': range(5, 15), 'atr_len': range(10, 22), 'pivot_lb': range(2, 6)}
    }

def calculate_all_required_indicators(df):
    """
    ✅ FUNCIÓN OPTIMIZADA: Calcula todos los indicadores necesarios para OPTUNA.
    Usa el patrón "recolectar y concatenar" para evitar la fragmentación y mejorar el rendimiento.
    """
    print("📊 Calculando indicadores dinámicos para OPTUNA (modo optimizado)...")
    param_ranges = get_optuna_parameter_ranges()
    
    # ✅ OPTIMIZACIÓN: Crear un diccionario para almacenar todas las nuevas columnas.
    new_columns = {}

    try:
        # --- Recolectar todos los períodos necesarios ---
        ema_periods = set(param_ranges['ema_crossover']['fast_period']) | set(param_ranges['ema_crossover']['slow_period']) | set(param_ranges['rsi_pullback']['trend_ema_period']) | set(param_ranges['multi_filter_scalper']['ema_fast']) | set(param_ranges['multi_filter_scalper']['ema_mid']) | set(param_ranges['multi_filter_scalper']['ema_slow'])
        rsi_periods = set(param_ranges['rsi_pullback']['rsi_period']) | set(param_ranges['multi_filter_scalper']['rsi_len'])
        atr_periods = set(param_ranges['volatility_breakout']['atr_period']) | set(param_ranges['multi_filter_scalper']['atr_len']) | {14}
        
        # --- Calcular y almacenar indicadores en el diccionario ---
        print(f"📈 Calculando {len(ema_periods)} EMAs...")
        for period in sorted(list(ema_periods)):
            if len(df) >= period * 2:
                new_columns[f'ema_{period}'] = ta.EMA(df['close'], timeperiod=period)

        print(f"📊 Calculando {len(rsi_periods)} RSIs...")
        for period in sorted(list(rsi_periods)):
            if len(df) >= period * 2:
                new_columns[f'rsi_{period}'] = ta.RSI(df['close'], timeperiod=period)

        print(f"⚡ Calculando {len(atr_periods)} ATRs y sus MAs...")
        for period in sorted(list(atr_periods)):
            if len(df) >= period * 2:
                atr_col = ta.ATR(df['high'], df['low'], df['close'], timeperiod=period)
                new_columns[f'atr_{period}'] = atr_col
                new_columns[f'atr_ma_{period}'] = ta.SMA(atr_col, timeperiod=period)

        print("🌐 Calculando Bandas de Bollinger...")
        for period in [20]:
            if len(df) >= period * 2:
                upper, middle, lower = ta.BBANDS(df['close'], timeperiod=period, nbdevup=2, nbdevdn=2, matype=0)
                new_columns[f'bb_upper_{period}'] = upper
                new_columns[f'bb_middle_{period}'] = middle
                new_columns[f'bb_lower_{period}'] = lower

        print("➕ Calculando indicadores adicionales...")
        if len(df) >= 50:
            macd, macd_signal, _ = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            stoch_k, _ = ta.STOCH(df['high'], df['low'], df['close'], fastk_period=5, slowk_period=3, slowd_period=3)
            new_columns['sma_20'] = ta.SMA(df['close'], timeperiod=20)
            new_columns['macd'] = macd
            new_columns['macd_signal'] = macd_signal
            new_columns['stoch_k'] = stoch_k
            new_columns['true_range'] = ta.TRANGE(df['high'], df['low'], df['close'])

        print("📈 Calculando features de Price Action y Sesión...")
        new_columns['price_change'] = df['close'].pct_change()
        new_columns['hl_spread'] = (df['high'] - df['low']) / df['close']
        new_columns['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # ✅ FIX: Manejo correcto de timestamps y sesiones
        if hasattr(df.index, 'hour'):
            hour = df.index.hour
            new_columns['hour'] = hour
            new_columns['session_london'] = ((hour >= 8) & (hour < 12)).astype(int)
            new_columns['session_ny'] = ((hour >= 13) & (hour < 17)).astype(int)
            new_columns['session_overlap'] = ((hour >= 13) & (hour < 16)).astype(int)
        else:
            # Fallback para cuando no hay timestamp válido
            default_hour = 12
            new_columns['hour'] = default_hour
            new_columns['session_london'] = 0  # False como int
            new_columns['session_ny'] = 1      # True como int (asumimos hora NY)
            new_columns['session_overlap'] = 0 # False como int
        
        # Asegurar que tenemos ATR genérico
        if 'atr_14' in new_columns:
            new_columns['atr'] = new_columns['atr_14']
        elif 'atr_20' in new_columns:
            new_columns['atr'] = new_columns['atr_20']
        else:
            # Fallback si no tenemos ningún ATR
            new_columns['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)

        # ✅ OPTIMIZACIÓN: Concatenar todas las nuevas columnas al DataFrame en una sola operación.
        df_final = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
        print(f"✅ Indicadores dinámicos calculados y unidos exitosamente. Total columnas: {len(df_final.columns)}")
        return df_final

    except Exception as e:
        print(f"❌ Error calculando indicadores dinámicos: {e}")
        import traceback
        traceback.print_exc()
        return df # Devolver el DataFrame original en caso de error

def add_all_features(df):
    """
    ✅ CALCULADOR DE FEATURES OPTIMIZADO PARA OPTUNA
    """
    df_copy = df.copy() # Trabajar sobre una copia para evitar SettingWithCopyWarning
    
    # ✅ Mapeo robusto de columnas
    column_mapping = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
    df_copy.rename(columns={col: col.lower() for col in df_copy.columns if col in column_mapping}, inplace=True)

    required_cols = ['open', 'high', 'low', 'close']
    if any(col not in df_copy.columns for col in required_cols):
        print(f"❌ Columnas faltantes: {[col for col in required_cols if col not in df_copy.columns]}")
        return df

    print(f"📊 Procesando {len(df_copy)} registros para compatibilidad con OPTUNA...")
    
    # ✅ Llamar a la función optimizada
    df_with_features = calculate_all_required_indicators(df_copy)
    
    # ✅ Verificar si los indicadores se calcularon correctamente
    if len(df_with_features.columns) == len(df_copy.columns):
        print("❌ Indicadores principales fallaron, aplicando fallback básico...")
        try:
            df_with_features['rsi_14'] = ta.RSI(df_with_features['close'], timeperiod=14)
            df_with_features['atr_14'] = ta.ATR(df_with_features['high'], df_with_features['low'], df_with_features['close'], timeperiod=14)
            df_with_features['atr'] = df_with_features['atr_14']
            df_with_features['ema_50'] = ta.EMA(df_with_features['close'], timeperiod=50)
            df_with_features['ema_200'] = ta.EMA(df_with_features['close'], timeperiod=200)
            print("✅ Indicadores básicos de fallback aplicados")
        except Exception as e:
            print(f"❌ Error en fallback: {e}")

    # ✅ Manejo robusto de timestamp
    if not pd.api.types.is_datetime64_any_dtype(df_with_features.index):
        time_col_found = None
        for col_name in ['Time', 'time', 'Timestamp', 'timestamp', 'Date', 'DateTime']:
            if col_name.lower() in df_with_features.columns:
                time_col_found = col_name.lower()
                break
        
        if time_col_found:
            try:
                df_with_features.set_index(pd.to_datetime(df_with_features[time_col_found]), inplace=True)
            except Exception as e:
                print(f"⚠️ Error convirtiendo timestamp: {e}")
                df_with_features.index = pd.date_range(start='2020-01-01', periods=len(df_with_features), freq='5T')
        else:
            # Crear timestamp sintético
            df_with_features.index = pd.date_range(start='2020-01-01', periods=len(df_with_features), freq='5T')

    # ✅ Limpiar datos
    initial_len = len(df_with_features)
    df_with_features.dropna(inplace=True)
    final_len = len(df_with_features)
    
    # ✅ Restaurar nombres de columnas originales
    df_with_features.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    
    print(f"✅ Features OPTUNA-ready: {initial_len} -> {final_len} registros")
    print(f"📊 Total columnas calculadas: {len(df_with_features.columns)}")
    
    return df_with_features

def verify_optuna_compatibility(df, strategy_name, strategy_params):
    """
    ✅ VERIFICA QUE EL DF TENGA TODOS LOS FEATURES PARA PARÁMETROS ESPECÍFICOS
    """
    missing_features = []
    try:
        if strategy_name == 'ema_crossover':
            required = [f"ema_{strategy_params['fast_period']}", f"ema_{strategy_params['slow_period']}"]
        elif strategy_name == 'rsi_pullback':
            required = [f"rsi_{strategy_params['rsi_period']}", f"ema_{strategy_params['trend_ema_period']}"]
        elif strategy_name == 'volatility_breakout':
            required = [f"atr_{strategy_params['atr_period']}"]
        elif strategy_name == 'multi_filter_scalper':
            required = [f"ema_{strategy_params['ema_fast']}", f"ema_{strategy_params['ema_mid']}", f"rsi_{strategy_params['rsi_len']}", f"atr_{strategy_params['atr_len']}", f"atr_ma_{strategy_params['atr_len']}"]
        elif strategy_name == 'channel_reversal':
            required = ['Open', 'High', 'Low', 'Close']
        else:
            required = []
        
        missing_features = [f for f in required if f not in df.columns]
        return len(missing_features) == 0, missing_features
        
    except Exception as e:
        print(f"❌ Error verificando compatibilidad: {e}")
        return False, [f"Error: {e}"]

def test_optuna_compatibility():
    """🧪 Función para probar compatibilidad con OPTUNA"""
    print("🧪 TESTING COMPATIBILIDAD CON OPTUNA" + "\n" + "=" * 50)
    
    np.random.seed(42)
    prices = 1.1000 * np.exp(np.cumsum(np.random.normal(0, 0.001, 2000)))
    df_test = pd.DataFrame({
        'Open': prices,
        'High': prices * (1 + np.random.uniform(0, 0.002, 2000)),
        'Low': prices * (1 - np.random.uniform(0, 0.002, 2000)),
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, 2000)
    }, index=pd.date_range("2023-01-01", periods=2000, freq="5min"))
    df_test['High'] = df_test[['Open', 'Close']].max(axis=1) * (1 + np.random.uniform(0, 0.001))
    df_test['Low'] = df_test[['Open', 'Close']].min(axis=1) * (1 - np.random.uniform(0, 0.001))
    
    print(f"📊 Datos de prueba: {len(df_test)} registros")
    df_with_features = add_all_features(df_test.copy())
    
    test_cases = [
        ('ema_crossover', {'fast_period': 15, 'slow_period': 75}),
        ('rsi_pullback', {'rsi_period': 10, 'trend_ema_period': 50}),
        ('volatility_breakout', {'atr_period': 10}),
        ('multi_filter_scalper', {'ema_fast': 5, 'ema_mid': 25, 'rsi_len': 5, 'atr_len': 10}),
        ('channel_reversal', {'period': 20, 'std_dev': 2.0})
    ]
    
    print(f"\n🔍 Probando {len(test_cases)} casos de parámetros OPTUNA:")
    all_compatible = True
    for strategy, params in test_cases:
        compatible, missing = verify_optuna_compatibility(df_with_features, strategy, params)
        status = "✅" if compatible else "❌"
        print(f"   {status} {strategy} {params}" + (f" - Faltantes: {missing}" if not compatible else ""))
        if not compatible: all_compatible = False
    
    print("\n" + ("🎉 ¡TODAS LAS COMBINACIONES SON COMPATIBLES!" if all_compatible else "⚠️ Hay incompatibilidades que deben corregirse"))
    
    feature_cols = [col for col in df_with_features.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    print(f"\n📊 Features calculados ({len(feature_cols)} total)")
    print(f"✅ Testing completado")
    return all_compatible

if __name__ == "__main__":
    test_optuna_compatibility()