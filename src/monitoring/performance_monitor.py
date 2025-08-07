"""
Módulo de monitoreo de rendimiento del trading bot
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
from pathlib import Path

class PerformanceMonitor:
    """Monitor de rendimiento en tiempo real"""
    
    def __init__(self, risk_config_path: str = "configs/risk_config.json"):
        """Inicializa el monitor con configuración de riesgo"""
        self.risk_config = self._load_risk_config(risk_config_path)
        self.alerts = self.risk_config.get("performance_alerts", {})
        
    def _load_risk_config(self, config_path: str) -> Dict[str, Any]:
        """Carga configuración de riesgo"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def check_daily_performance(self, trades_df: pd.DataFrame) -> List[str]:
        """Verifica rendimiento diario y genera alertas"""
        alerts = []
        today = datetime.now().date()
        
        # Filtrar trades de hoy
        today_trades = trades_df[trades_df['date'].dt.date == today]
        
        if len(today_trades) == 0:
            return alerts
        
        # Calcular métricas
        daily_pnl = today_trades['pnl'].sum()
        win_rate = (today_trades['pnl'] > 0).mean()
        consecutive_losses = self._calculate_consecutive_losses(today_trades)
        
        # Verificar umbrales
        max_daily_loss = self.alerts.get("max_daily_loss", 0.05)
        min_win_rate = self.alerts.get("min_win_rate_threshold", 0.4)
        max_consecutive = self.alerts.get("max_consecutive_losses", 5)
        
        if daily_pnl < -max_daily_loss:
            alerts.append(f"Daily loss exceeded: {daily_pnl:.4f}")
        
        if win_rate < min_win_rate:
            alerts.append(f"Win rate below threshold: {win_rate:.2f}")
        
        if consecutive_losses >= max_consecutive:
            alerts.append(f"Consecutive losses: {consecutive_losses}")
        
        return alerts
    
    def _calculate_consecutive_losses(self, trades_df: pd.DataFrame) -> int:
        """Calcula pérdidas consecutivas"""
        if len(trades_df) == 0:
            return 0
        
        consecutive = 0
        for _, trade in trades_df.tail(10).iterrows():
            if trade['pnl'] < 0:
                consecutive += 1
            else:
                break
        
        return consecutive
    
    def generate_performance_report(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Genera reporte de rendimiento"""
        if len(trades_df) == 0:
            return {}
        
        return {
            "total_trades": len(trades_df),
            "win_rate": (trades_df['pnl'] > 0).mean(),
            "total_pnl": trades_df['pnl'].sum(),
            "avg_trade": trades_df['pnl'].mean(),
            "max_drawdown": self._calculate_max_drawdown(trades_df),
            "sharpe_ratio": self._calculate_sharpe_ratio(trades_df)
        }
    
    def _calculate_max_drawdown(self, trades_df: pd.DataFrame) -> float:
        """Calcula drawdown máximo"""
        cumulative = trades_df['pnl'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_sharpe_ratio(self, trades_df: pd.DataFrame) -> float:
        """Calcula ratio de Sharpe"""
        if len(trades_df) < 2:
            return 0.0
        
        returns = trades_df['pnl']
        return returns.mean() / returns.std() if returns.std() != 0 else 0.0
