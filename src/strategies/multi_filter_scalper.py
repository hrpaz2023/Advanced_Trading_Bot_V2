# MULTI FILTER SCALPER PRÁCTICA - CORREGIDA
import pandas as pd
import numpy as np

class MultiFilterScalper:
    def __init__(self, ema_fast=12, ema_mid=26, ema_slow=50, rsi_len=14, 
                 rsi_buy=40, rsi_sell=60, atr_len=14, atr_mult=2.0, 
                 pivot_lb=3, prox_pct=3.0, scoring_mode=False, debug_mode=False, **kwargs):
        """CORREGIDO - Todos los parámetros del optimizador"""
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        self.rsi_len = rsi_len
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell
        self.atr_len = atr_len  # ✅ AGREGADO
        self.atr_mult = atr_mult
        self.pivot_lb = pivot_lb
        self.prox_pct = prox_pct
        self.name = f"Multi Filter Scalper Practical ({ema_fast}/{ema_mid})"
        self.last_signal = None

    def get_signal(self, historical_data, last_pivot_high=None, last_pivot_low=None):
        """EMA crossover + RSI simple"""
        if len(historical_data) < max(self.ema_slow + 5, 20):
            return 0
        
        current = historical_data.iloc[-1]
        previous = historical_data.iloc[-2]
        
        ema_fast_col = f'ema_{self.ema_fast}'
        ema_slow_col = f'ema_{self.ema_slow}'
        rsi_col = f'rsi_{self.rsi_len}'
        
        if not all(col in current.index for col in [ema_fast_col, ema_slow_col, rsi_col]):
            return 0
        
        # EMA crossover
        curr_fast = current[ema_fast_col]
        curr_slow = current[ema_slow_col]
        prev_fast = previous[ema_fast_col]
        prev_slow = previous[ema_slow_col]
        
        crossed_up = prev_fast <= prev_slow and curr_fast > curr_slow
        crossed_down = prev_fast >= prev_slow and curr_fast < curr_slow
        
        # RSI filter
        current_rsi = current[rsi_col]
        
        signal = 0
        
        # Buy
        if crossed_up and current_rsi < 70:
            if self.last_signal != 'BUY':
                self.last_signal = 'BUY'
                signal = 1
        # Sell        
        elif crossed_down and current_rsi > 30:
            if self.last_signal != 'SELL':
                self.last_signal = 'SELL'
                signal = -1
        # Reset
        elif not crossed_up and not crossed_down:
            if (curr_fast > curr_slow and current_rsi > 60) or (curr_fast < curr_slow and current_rsi < 40):
                self.last_signal = None
        
        return signal

    # Métodos de compatibilidad
    def enable_debug(self): pass
    def get_debug_info(self): return []
    def print_debug_summary(self): pass
    def get_optimized_parameters_for_symbol(self, symbol): return {}

# Factory de compatibilidad
class MultiFilterScalperFactory:
    @staticmethod
    def create_for_symbol(symbol, debug_mode=False):
        return MultiFilterScalper()
    
    @staticmethod
    def create_aggressive(debug_mode=False):
        return MultiFilterScalper(ema_fast=8, ema_mid=20, rsi_buy=35, rsi_sell=65)
    
    @staticmethod  
    def create_conservative(debug_mode=False):
        return MultiFilterScalper(ema_fast=20, ema_mid=50, rsi_buy=45, rsi_sell=55)
