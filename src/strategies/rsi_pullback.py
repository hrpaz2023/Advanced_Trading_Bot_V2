# RSI PULLBACK PRÁCTICA - CORREGIDA
import pandas as pd
import numpy as np

class RsiPullback:
    def __init__(self, rsi_oversold=35, rsi_overbought=65, rsi_period=14, 
                 rsi_level=35, trend_ema_period=200, **kwargs):
        """CORREGIDO - Compatibilidad con optimizador"""
        # Usar rsi_level si se proporciona, sino usar rsi_oversold
        self.rsi_oversold = rsi_level if rsi_level else rsi_oversold
        self.rsi_overbought = 100 - self.rsi_oversold  # Simétrico
        self.rsi_period = rsi_period
        self.trend_ema_period = trend_ema_period
        self.name = f"RSI Pullback Practical ({self.rsi_oversold}/{self.rsi_overbought})"
        
        # Compatibilidad adicional
        self.adaptive_atr = kwargs.get('adaptive_atr', True)
        self.rsi_smoothing = kwargs.get('rsi_smoothing', True)
        self.trend_weight = kwargs.get('trend_weight', 0.5)

    def get_signal(self, historical_data):
        """RSI simple"""
        if len(historical_data) < 20:
            return 0
        
        current = historical_data.iloc[-1]
        previous = historical_data.iloc[-2]
        
        rsi_col = f'rsi_{self.rsi_period}'
        if rsi_col not in current.index:
            return 0
        
        current_rsi = current[rsi_col]
        previous_rsi = previous[rsi_col]
        
        # Cruce de niveles
        if previous_rsi <= self.rsi_oversold and current_rsi > self.rsi_oversold:
            return 1
        elif previous_rsi >= self.rsi_overbought and current_rsi < self.rsi_overbought:
            return -1
        
        return 0
    
    # Métodos de compatibilidad
    def get_adaptive_atr_threshold(self, historical_data, lookback=50):
        return 0.0008
    
    def calculate_trend_strength(self, current_candle, ema_col):
        return 1.0
    
    def detect_symbol(self, historical_data):
        return 'EURUSD'
