# EMA CROSSOVER PRÁCTICA - ASEGURADA
import pandas as pd
import numpy as np

class EmaCrossover:
    def __init__(self, fast_period=12, slow_period=26, use_trend_filter=False, 
                 use_volatility_filter=False, **kwargs):
        """Totalmente compatible"""
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.use_trend_filter = use_trend_filter
        self.use_volatility_filter = use_volatility_filter
        self.name = f"EMA Crossover Practical ({fast_period}/{slow_period})"
        
        # ATR realistas
        self.min_atr_thresholds = {
            'EURUSD': 0.0008, 'GBPUSD': 0.0010, 'AUDUSD': 0.0007,
            'USDJPY': 0.08, 'USDCAD': 0.0008, 'NZDUSD': 0.0009,
        }
        
        self.trade_count = 0
        self.bar_count = 0
        
        # Compatibilidad adicional
        self.adaptive_atr = kwargs.get('adaptive_atr', True)
        self.trend_weight = kwargs.get('trend_weight', 0.6)
        self.crossover_strength_filter = kwargs.get('crossover_strength_filter', True)

    def get_signal(self, historical_data):
        """Simple crossover"""
        if len(historical_data) < self.slow_period + 5:
            return 0
        
        self.bar_count += 1
        
        previous = historical_data.iloc[-2]
        current = historical_data.iloc[-1]
        
        fast_col = f'ema_{self.fast_period}'
        slow_col = f'ema_{self.slow_period}'

        if not all(col in current.index for col in [fast_col, slow_col]):
            return 0

        # Crossover
        curr_fast = current[fast_col]
        curr_slow = current[slow_col]
        prev_fast = previous[fast_col]
        prev_slow = previous[slow_col]
        
        crossed_up = prev_fast <= prev_slow and curr_fast > curr_slow
        crossed_down = prev_fast >= prev_slow and curr_fast < curr_slow
        
        if not crossed_up and not crossed_down:
            return 0
        
        signal = 1 if crossed_up else -1
        
        # Filtros básicos (opcionales)
        if self.use_volatility_filter:
            if 'atr_14' in current.index:
                atr = current['atr_14']
                symbol = self.detect_symbol(historical_data)
                min_atr = self.min_atr_thresholds.get(symbol, 0.0008)
                if atr < min_atr * 0.1:
                    return 0
        
        if self.use_trend_filter:
            if 'ema_200' in current.index:
                price = current['Close']
                ema_200 = current['ema_200']
                if signal == 1 and price < ema_200 * 0.98:
                    return 0
                elif signal == -1 and price > ema_200 * 1.02:
                    return 0
        
        self.trade_count += 1
        return signal
    
    def detect_symbol(self, historical_data):
        try:
            avg_price = historical_data['Close'].tail(10).mean()
            if avg_price > 50: return 'USDJPY'
            elif 1.2 <= avg_price <= 1.4: return 'EURUSD'
            elif 1.0 <= avg_price < 1.2: return 'GBPUSD'
            elif 0.6 <= avg_price < 1.0: return 'AUDUSD'
            else: return 'EURUSD'
        except: return 'EURUSD'
    
    # Métodos de compatibilidad
    def calculate_crossover_strength(self, *args): return 1.0
    def get_adaptive_atr_threshold(self, *args): return 0.0008
    def calculate_trend_strength(self, *args): return 1.0
