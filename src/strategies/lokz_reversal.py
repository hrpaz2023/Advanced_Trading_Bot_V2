# LOKZ REVERSAL PRÁCTICA - Compatible con optimizador
import pandas as pd
import numpy as np

class LokzReversal:
    def __init__(self, asia_session="00:00-08:00", lokz_session="08:00-10:00", 
                 timezone="UTC", sl_atr_mult=0.5, tp1_atr_mult=0.5, **kwargs):
        """LOKZ optimizado para trading diario"""
        self.name = "LOKZ Reversal Practical"
        
        # Sesiones (UTC)
        self.asia_start_hour = 0
        self.asia_end_hour = 8  
        self.lokz_start_hour = 8
        self.lokz_end_hour = 10
        
        self.sl_atr_mult = sl_atr_mult
        self.tp1_atr_mult = tp1_atr_mult
        
        # Estado
        self.last_trade_day = None
        self.asia_high = None
        self.asia_low = None
        self.lokz_open = None
        self.lokz_high = None
        self.lokz_low = None
        self.trade_taken_today = False

    def get_signal(self, historical_data):
        """1 trade por día máximo"""
        if len(historical_data) < 50:
            return 0
        
        current = historical_data.iloc[-1]
        
        # Obtener hora
        try:
            if hasattr(current.name, 'hour'):
                current_hour = current.name.hour
            else:
                current_hour = pd.to_datetime(current.name).hour
        except:
            return 0
        
        # Nuevo día
        try:
            if hasattr(current.name, 'date'):
                current_day = current.name.date()
            else:
                current_day = pd.to_datetime(current.name).date()
        except:
            current_day = str(current.name)[:10]
        
        if current_day != self.last_trade_day:
            self._reset_daily_state()
            self.last_trade_day = current_day
        
        if self.trade_taken_today:
            return 0
        
        # Asia session (0-8 UTC)
        if 0 <= current_hour < 8:
            if self.asia_high is None:
                self.asia_high = current['High']
                self.asia_low = current['Low']
            else:
                self.asia_high = max(self.asia_high, current['High'])
                self.asia_low = min(self.asia_low, current['Low'])
        
        # LOKZ (8-10 UTC)
        elif 8 <= current_hour < 10:
            if self.lokz_open is None:
                self.lokz_open = current['Open']
                self.lokz_high = current['High']
                self.lokz_low = current['Low']
            else:
                self.lokz_high = max(self.lokz_high, current['High'])
                self.lokz_low = min(self.lokz_low, current['Low'])
        
        # Post-LOKZ (10+ UTC)
        elif current_hour >= 10 and not self.trade_taken_today:
            return self._evaluate_reversal(current)
        
        return 0
    
    def _evaluate_reversal(self, current):
        """Evalúa reversión"""
        if not all([self.asia_high, self.asia_low, self.lokz_open, 
                   self.lokz_high, self.lokz_low]):
            return 0
        
        self.trade_taken_today = True
        
        # Filtro: LOKZ con movimiento
        lokz_range = self.lokz_high - self.lokz_low
        asia_range = self.asia_high - self.asia_low
        
        if lokz_range < asia_range * 0.2:
            return 0
        
        # Reversión
        if current['Close'] < self.lokz_open:  # LOKZ bajista -> buy
            if abs(current['Close'] - self.lokz_low) < lokz_range * 0.5:
                return 1
        elif current['Close'] > self.lokz_open:  # LOKZ alcista -> sell
            if abs(current['Close'] - self.lokz_high) < lokz_range * 0.5:
                return -1
        
        return 0
    
    def _reset_daily_state(self):
        """Reset diario"""
        self.asia_high = None
        self.asia_low = None
        self.lokz_open = None
        self.lokz_high = None
        self.lokz_low = None
        self.trade_taken_today = False
