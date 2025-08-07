# src/backtesting/backtester.py (Versión Corregida)
import pandas as pd
import os
import json
import talib as ta
import numpy as np
from scipy.signal import find_peaks
from numba import jit, njit

class RiskManager:
    """RiskManager simple integrado - optimizado"""
    def __init__(self):
        try:
            with open('configs/risk_config.json', 'r') as f:
                config = json.load(f)
                self.risk_per_trade = config.get('risk_per_trade_percent', 1.0) / 100.0
                self.rr_ratio = config.get('default_rr_ratio', 2.0)
        except:
            self.risk_per_trade = 0.01
            self.rr_ratio = 2.0
    
    def calculate_position_size(self, equity, sl_pips, symbol):
        if sl_pips <= 0: return 0.01
        pip_value = 10.0 if "JPY" not in symbol else 0.1
        risk_amount = equity * self.risk_per_trade
        position_size = risk_amount / (sl_pips * pip_value)
        return max(0.01, min(50.0, round(position_size, 2)))
    
    def get_trade_params(self, entry_price, atr, signal):
        atr_multiplier = 2.0
        if signal == 1:
            sl_price = entry_price - (atr * atr_multiplier)
            tp_price = entry_price + (atr * atr_multiplier * self.rr_ratio)
        else:
            sl_price = entry_price + (atr * atr_multiplier)
            tp_price = entry_price - (atr * atr_multiplier * self.rr_ratio)
        return sl_price, tp_price

# ✅ OPTIMIZACIÓN 1: Funciones numba para cálculos críticos
@njit
def calculate_pnl_fast(exit_price, entry_price, side, lot_size, contract_size, is_jpy_pair):
    """Cálculo ultrarrápido de PnL usando numba"""
    pnl_points = (exit_price - entry_price) * (1 if side == 1 else -1)
    
    if is_jpy_pair:
        pnl = (pnl_points / exit_price) * contract_size * lot_size
    else:
        pnl = pnl_points * contract_size * lot_size
    
    return pnl

@njit
def check_exit_conditions_fast(side, current_low, current_high, sl, tp):
    """Verificación ultrarrápida de condiciones de salida"""
    if side == 1:  # Long position
        if current_low <= sl:
            return True, sl
        elif current_high >= tp:
            return True, tp
    else:  # Short position
        if current_high >= sl:
            return True, sl
        elif current_low <= tp:
            return True, tp
    
    return False, 0.0

@njit
def calculate_session_flags_fast(hours):
    """Cálculo vectorizado de sesiones de trading"""
    london = ((hours >= 7) & (hours < 16)).astype(np.int8)
    ny = ((hours >= 12) & (hours < 21)).astype(np.int8)
    return london, ny

class Backtester:
    def __init__(self, symbol, strategy, initial_equity=10000.0):
        self.symbol = symbol
        self.strategy = strategy
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.risk_manager = RiskManager()
        self.trade_log = []
        self.equity_curve = []
        self._indicator_cache = {}
        
        # ✅ OPTIMIZACIÓN 2: Pre-calcular valores constantes
        self.contract_size = 100000
        self.is_jpy_pair = "JPY" in symbol
        
        try:
            with open('configs/risk_config.json', 'r') as f:
                self.sim_config = json.load(f).get('simulation', {})
        except:
            self.sim_config = {'slippage_points': 2, 'commission_per_lot': 7.0}

    def _get_required_indicators(self):
        """✅ OPTIMIZACIÓN 3: Cache de metadatos de estrategia"""
        if hasattr(self, '_required_indicators_cache'):
            return self._required_indicators_cache
            
        required = {}
        s_name = self.strategy.__class__.__name__.lower()
        if 'emacrossover' in s_name: 
            required['ema'] = [self.strategy.fast_period, self.strategy.slow_period, 200]  # ✅ Agregar EMA 200
        elif 'rsipullback' in s_name:
            required['rsi'] = [self.strategy.rsi_period]
            required['ema'] = [self.strategy.trend_ema_period]
            required['atr'] = [14]  # ✅ ATR requerido
        elif 'volatilitybreakout' in s_name: 
            required['atr'] = [self.strategy.atr_period]
            required['ema'] = [200]  # ✅ EMA 200 requerida
        elif 'multifilterscalper' in s_name:
            required['ema'] = [self.strategy.ema_fast, self.strategy.ema_mid, self.strategy.ema_slow]
            required['rsi'] = [self.strategy.rsi_len]
            required['atr'] = [self.strategy.atr_len]
        elif 'lokzreversal' in s_name:
            required['ema'] = [200]  # ✅ EMA 200 requerida
            required['atr'] = [14]   # ✅ ATR requerido
        elif 'channelreversal' in s_name:
            required['ema'] = [200]  # ✅ EMA 200 requerida
            required['atr'] = [14]   # ✅ ATR requerido
        
        self._required_indicators_cache = required
        return required

    def _calculate_indicator_smart(self, df, indicator_type, period):
        """✅ OPTIMIZACIÓN 4: Cálculo más eficiente de indicadores"""
        cache_key = f"{indicator_type}_{period}"
        if cache_key in self._indicator_cache: 
            return
        
        expected_column = f"{indicator_type}_{period}"
        if expected_column in df.columns and not df[expected_column].isna().all():
            self._indicator_cache[cache_key] = True
            return

        try:
            # ✅ Usar arrays numpy directamente para mejor rendimiento
            close_array = df['Close'].values
            
            if indicator_type == 'ema': 
                result = ta.EMA(close_array, timeperiod=period)
            elif indicator_type == 'rsi': 
                result = ta.RSI(close_array, timeperiod=period)
            elif indicator_type == 'atr': 
                result = ta.ATR(df['High'].values, df['Low'].values, close_array, timeperiod=period)
            else: 
                return
                
            df[expected_column] = result
            self._indicator_cache[cache_key] = True
        except Exception as e:
            print(f"⚠️ Error calculando {indicator_type}_{period}: {e}")

    def _prepare_data_smart(self, df):
        """✅ CORRECCIÓN: Preparación de datos más robusta"""
        df = df.copy()
        print(f"⚡ Preparación de datos para {self.strategy.__class__.__name__}")
        
        # 1. Calcular todos los indicadores técnicos primero
        required_indicators = self._get_required_indicators()
        for ind_type, periods in required_indicators.items():
            for p in periods: 
                self._calculate_indicator_smart(df, ind_type, p)

        # 2. ✅ CORRECCIÓN: Asegurar ATR estándar y específicos de estrategia
        if 'atr_14' not in df.columns:
            self._calculate_indicator_smart(df, 'atr', 14)
        
        # ✅ CORRECCIÓN: Crear alias ATR para estrategias específicas
        strategy_name = self.strategy.__class__.__name__.lower()
        if 'multifilterscalper' in strategy_name:
            atr_col = f'atr_{self.strategy.atr_len}'
            if atr_col not in df.columns and 'atr_14' in df.columns:
                df[atr_col] = df['atr_14']
        elif 'rsipullback' in strategy_name:
            if 'atr_14' in df.columns:
                df['atr'] = df['atr_14']  # RSI Pullback espera 'atr'

        # 3. Eliminar NaNs de los indicadores técnicos
        initial_len = len(df)
        df.dropna(inplace=True)
        print(f"   - Limpieza de indicadores: {initial_len} -> {len(df)} registros")

        # 4. ✅ CORRECCIÓN: Calcular Pivots para MultiFilterScalper sobre datos limpios
        if 'multifilterscalper' in strategy_name:
            pivot_lb = getattr(self.strategy, 'pivot_lb', 3)
            
            print(f"   - Calculando pivots con lookback={pivot_lb}...")
            
            # ✅ OPTIMIZACIÓN 6: Usar arrays numpy para find_peaks
            high_values = df['High'].values
            low_values = df['Low'].values
            
            # Encontrar picos y valles
            high_peaks, _ = find_peaks(high_values, distance=pivot_lb)
            low_peaks, _ = find_peaks(-low_values, distance=pivot_lb)
            
            print(f"   - Encontrados {len(high_peaks)} pivot highs, {len(low_peaks)} pivot lows")
            
            # Inicializar con NaN y asignar valores
            df['pivot_high'] = np.nan
            df['pivot_low'] = np.nan
            
            if len(high_peaks) > 0:
                df.iloc[high_peaks, df.columns.get_loc('pivot_high')] = high_values[high_peaks]
            if len(low_peaks) > 0:
                df.iloc[low_peaks, df.columns.get_loc('pivot_low')] = low_values[low_peaks]
            
            # Forward fill más eficiente
            df[['pivot_high', 'pivot_low']] = df[['pivot_high', 'pivot_low']].ffill()
            
            # ✅ CORRECCIÓN: Solo eliminar NaNs si existen pivots
            pivot_data_available = not (df['pivot_high'].isna().all() or df['pivot_low'].isna().all())
            if pivot_data_available:
                df.dropna(subset=['pivot_high', 'pivot_low'], inplace=True)
                print(f"   - Datos con pivots válidos: {len(df)} registros")
            else:
                print("   ⚠️ No se encontraron pivots válidos - la estrategia no generará señales")

        # 5. ✅ OPTIMIZACIÓN 7: Cálculo vectorizado de sesiones
        print("   - Calculando sesiones de trading...")
        df.index = pd.to_datetime(df.index, utc=True)
        hours = df.index.hour.values
        london, ny = calculate_session_flags_fast(hours)
        df['session_london'] = london
        df['session_ny'] = ny
        
        print(f"   ✅ Datos finales listos: {len(df)} registros")
        return df

    def run(self, return_data=False):
        """
        ✅ CORRECCIÓN: Backtest con manejo correcto de estrategias específicas
        """
        features_path = f"data/features/{self.symbol}_features.parquet"
        if not os.path.exists(features_path):
            print(f"❌ Archivo de features no encontrado: {features_path}")
            return None
            
        data = pd.read_parquet(features_path)
        print(f"📊 Cargados {len(data)} registros para {self.symbol}")
        
        data = self._prepare_data_smart(data)
        if data.empty:
            print("❌ Datos vacíos después de la preparación.")
            return None

        # ✅ OPTIMIZACIÓN 9: Pre-convertir a arrays numpy para acceso rápido
        close_prices = data['Close'].values
        high_prices = data['High'].values
        low_prices = data['Low'].values
        
        # ✅ CORRECCIÓN: Manejo robusto de ATR
        atr_column = None
        strategy_name = self.strategy.__class__.__name__.lower()
        
        if 'multifilterscalper' in strategy_name:
            atr_column = f'atr_{self.strategy.atr_len}'
        elif 'rsipullback' in strategy_name:
            atr_column = 'atr_14'
        else:
            atr_column = 'atr_14'
            
        if atr_column not in data.columns:
            atr_column = 'atr_14'  # Fallback
            
        atr_values = data[atr_column].values if atr_column in data.columns else np.full(len(data), 0.001)
        timestamps = data.index.values
        
        position_open = False
        current_trade = {}
        data_len = len(data)
        
        print(f"🚀 Iniciando backtest con {data_len} barras...")

        # ✅ CORRECCIÓN: Loop principal con manejo específico de estrategias
        for i in range(1, data_len):
            try:
                # Acceso directo a arrays numpy (mucho más rápido)
                current_close = close_prices[i]
                current_high = high_prices[i]
                current_low = low_prices[i]
                current_atr = atr_values[i]
                current_timestamp = timestamps[i]

                # Slice histórico solo cuando sea necesario
                if not position_open:  # Solo calcular señal si no hay posición abierta
                    historical_slice = data.iloc[max(0, i-200):i+1]
                    
                    # ✅ CORRECCIÓN PRINCIPAL: Manejo específico para MultiFilterScalper
                    if 'multifilterscalper' in strategy_name:
                        # Verificar que existen los pivots
                        if 'pivot_high' in historical_slice.columns and 'pivot_low' in historical_slice.columns:
                            last_pivot_high = historical_slice['pivot_high'].iloc[-1]
                            last_pivot_low = historical_slice['pivot_low'].iloc[-1]
                            
                            # Solo llamar si los pivots no son NaN
                            if not (pd.isna(last_pivot_high) or pd.isna(last_pivot_low)):
                                signal = self.strategy.get_signal(historical_slice, last_pivot_high, last_pivot_low)
                            else:
                                signal = 0
                        else:
                            signal = 0
                    else:
                        # Para todas las demás estrategias
                        signal = self.strategy.get_signal(historical_slice)
                else:
                    signal = 0

                if position_open:
                    # ✅ Usar función numba optimizada para verificar salidas
                    close_position, exit_price = check_exit_conditions_fast(
                        current_trade['side'], current_low, current_high,
                        current_trade['sl'], current_trade['tp']
                    )

                    if close_position:
                        # ✅ Usar función numba optimizada para PnL
                        pnl = calculate_pnl_fast(
                            exit_price, current_trade['entry_price'],
                            current_trade['side'], current_trade['size'],
                            self.contract_size, self.is_jpy_pair
                        )
                        
                        self.equity += pnl
                        self.trade_log.append({
                            "entry_time": current_trade['entry_time'], 
                            "exit_time": current_timestamp,
                            "side": "LONG" if current_trade['side'] == 1 else "SHORT", 
                            "entry_price": current_trade['entry_price'],
                            "exit_price": exit_price, 
                            "pnl": pnl
                        })
                        position_open, current_trade = False, {}

                if not position_open and signal != 0:
                    entry_price = current_close
                    sl_price, tp_price = self.risk_manager.get_trade_params(entry_price, current_atr, signal)
                    
                    sl_pips = abs(entry_price - sl_price) * (100 if self.is_jpy_pair else 10000)
                    size_in_lots = self.risk_manager.calculate_position_size(self.equity, sl_pips, self.symbol)

                    if size_in_lots > 0:
                        current_trade = {
                            'side': signal, 'entry_price': entry_price, 'sl': sl_price, 'tp': tp_price,
                            'size': size_in_lots, 'entry_time': current_timestamp
                        }
                        position_open = True

                # ✅ OPTIMIZACIÓN 11: Reducir frecuencia de equity curve
                if i % 100 == 0 or not position_open:  # Solo cada 100 barras o al cerrar trades
                    self.equity_curve.append({'timestamp': current_timestamp, 'equity': self.equity})

            except Exception as e:
                print(f"⚠️ Error en iteración {i}: {e}")
                continue

        print(f"✅ Backtest completado. Total trades: {len(self.trade_log)}")
        
        report = self.generate_report()
        if return_data:
            return report, data
        return report

    def generate_report(self):
        """✅ OPTIMIZACIÓN 12: Generación de reporte más eficiente"""
        if not self.trade_log: 
            return {'Total Trades': 0, 'Win Rate': 0, 'Profit Factor': 0, 'Net Profit': 0, 'Max Drawdown': 0, 'Sharpe Ratio': 0}
        
        # Usar numpy para cálculos más rápidos
        pnl_array = np.array([trade['pnl'] for trade in self.trade_log])
        
        total_pnl = np.sum(pnl_array)
        winning_trades = pnl_array > 0
        win_count = np.sum(winning_trades)
        
        # Cálculo eficiente de profit factor
        positive_pnl = np.sum(pnl_array[winning_trades])
        negative_pnl = np.sum(pnl_array[~winning_trades])
        profit_factor = positive_pnl / abs(negative_pnl) if negative_pnl < 0 else 999
        
        # Cálculo eficiente de drawdown
        cumulative_pnl = np.cumsum(pnl_array)
        equity_series = self.initial_equity + cumulative_pnl
        peak_equity = np.maximum.accumulate(np.concatenate(([self.initial_equity], equity_series)))
        drawdown = (equity_series - peak_equity[1:]) / peak_equity[1:]
        max_drawdown = np.min(drawdown) * 100
        
        report = {
            'Total Trades': len(self.trade_log),
            'Win Rate': (win_count / len(self.trade_log)) * 100,
            'Profit Factor': profit_factor,
            'Net Profit': total_pnl,
            'Max Drawdown': max_drawdown,
            'Sharpe Ratio': 0.0
        }
        
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in report.items()}

    def get_trade_log(self):
        if hasattr(self, 'trade_log') and self.trade_log:
            return pd.DataFrame(self.trade_log)
        return pd.DataFrame()