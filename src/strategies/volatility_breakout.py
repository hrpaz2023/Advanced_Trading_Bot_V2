# ====================================================================================
# VOLATILITY BREAKOUT PRÁCTICA - CORREGIDA PARA OPTIMIZADOR
# ====================================================================================

import pandas as pd
import numpy as np

class VolatilityBreakout:
    def __init__(self, period=15, atr_period=14, atr_multiplier=0.5, use_atr_filter=True, **kwargs):
        """CORREGIDA - Compatible con optimizador"""
        self.period = period
        self.atr_period = atr_period  # ✅ AGREGADO
        self.atr_multiplier = atr_multiplier
        self.use_atr_filter = use_atr_filter
        self.name = f"Volatility Breakout Practical ({period})"
        
        # Compatibilidad adicional
        self.adaptive_atr = kwargs.get('adaptive_atr', True)
        self.volatility_threshold_percentile = kwargs.get('volatility_threshold_percentile', 20)
        self.trend_weight = kwargs.get('trend_weight', 0.6)

    def get_signal(self, historical_data):
        """Lógica simple de breakout"""
        if len(historical_data) < self.period + 5:
            return 0
        
        current = historical_data.iloc[-1]
        
        # Canal simple
        lookback_data = historical_data.iloc[-(self.period+1):-1]
        upper_channel = lookback_data['High'].max()
        lower_channel = lookback_data['Low'].min()
        
        # ATR filter (permisivo)
        strong_candle = True
        if self.use_atr_filter:
            atr_col = f'atr_{self.atr_period}'
            if atr_col in current.index:
                current_atr = current[atr_col]
                min_range = current_atr * self.atr_multiplier
                candle_range = current['High'] - current['Low']
                strong_candle = candle_range > min_range
        
        # Breakout
        if current['Close'] > upper_channel and strong_candle:
            return 1
        elif current['Close'] < lower_channel and strong_candle:
            return -1
        
        return 0
