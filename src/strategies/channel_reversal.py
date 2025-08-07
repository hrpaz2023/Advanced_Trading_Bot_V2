# ====================================================================================
# CHANNEL REVERSAL PRÁCTICA - Bollinger Bands simples y efectivas
# OBJETIVO: 100-150 trades/año con lógica simple
# ====================================================================================

import pandas as pd
import numpy as np

class ChannelReversal:
    def __init__(self, period=18, std_dev=1.8, use_session_filter=True):
        """
        Versión PRÁCTICA de Channel Reversal
        - Bollinger Bands simples
        - Eliminado ATR over-restrictivo
        - Target: 100-150 trades por año
        """
        self.period = period
        self.std_dev = std_dev  # Más permisivo que 2.0
        self.use_session_filter = use_session_filter
        self.name = f"Channel Reversal Practical ({period}/{std_dev})"

    def get_signal(self, historical_data):
        """Lógica simple de reversión en Bollinger Bands"""
        if len(historical_data) < self.period + 5:
            return 0
        
        current_candle = historical_data.iloc[-1]
        previous_candle = historical_data.iloc[-2]

        # FILTRO DE SESIÓN (muy permisivo)
        if self.use_session_filter:
            valid_session = self._check_trading_session(current_candle)
            if not valid_session:
                return 0

        # CORE: Bollinger Bands simples
        recent_data = historical_data.tail(self.period)
        sma = recent_data['Close'].mean()
        std = recent_data['Close'].std()
        
        bb_upper = sma + (self.std_dev * std)
        bb_lower = sma - (self.std_dev * std)
        
        # Lógica de reversión simple
        current_close = current_candle['Close']
        previous_close = previous_candle['Close']
        
        # Buy en toque de banda inferior
        if previous_close > bb_lower and current_close <= bb_lower:
            return 1
        
        # Sell en toque de banda superior  
        elif previous_close < bb_upper and current_close >= bb_upper:
            return -1
        
        return 0
    
    def _check_trading_session(self, current_candle):
        """Filtro de sesión MUY permisivo"""
        # Intentar usar columnas de sesión
        session_cols = ['session_london', 'session_ny', 'session_tokyo', 'session_sydney']
        if any(current_candle.get(col, 0) == 1 for col in session_cols):
            return True
        
        # Fallback: aceptar 22 de 24 horas
        try:
            if hasattr(current_candle.name, 'hour'):
                hour = current_candle.name.hour
                return not (2 <= hour <= 4)  # Solo evitar 2-4 AM
        except:
            pass
        
        return True  # Default: aceptar
