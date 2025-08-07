import os
import json
from datetime import date
import logging

class SentimentFilter:
    """
    Tercer filtro del sistema: Valida señales de trading contra un
    sesgo direccional y de confianza definido por un analista.
    """
    def __init__(self, config_dir="configs"):
        self.logger = logging.getLogger(__name__)
        self.sentiment_config = self._load_sentiment_config(config_dir)

    def _load_sentiment_config(self, config_dir):
        today_str = date.today().strftime('%Y%m%d')
        config_file = os.path.join(config_dir, f"daily_sentiment_{today_str}.json")

        if not os.path.exists(config_file):
            self.logger.warning(f"No se encontró archivo de sentimiento para hoy: {config_file}. El filtro de sentimiento estará inactivo.")
            return None
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.logger.info(f"Filtro de sentimiento cargado exitosamente desde {config_file}.")
                return config.get('pairs', {})
        except Exception as e:
            self.logger.error(f"Error cargando el filtro de sentimiento: {e}")
            return None

    def validate_signal(self, symbol: str, signal: int, ml_confidence: float) -> tuple[bool, str]:
        """
        Valida una señal contra la configuración de sentimiento del día.

        Args:
            symbol (str): Símbolo del par (ej: 'EURUSD').
            signal (int): 1 para compra, -1 para venta.
            ml_confidence (float): Confianza del modelo de ML (0.0 a 1.0).

        Returns:
            tuple[bool, str]: (True si la señal es válida, Razón del rechazo).
        """
        if not self.sentiment_config or symbol not in self.sentiment_config:
            return True, "Filtro de sentimiento inactivo o par no encontrado."

        pair_sentiment = self.sentiment_config[symbol]
        sentiment_bias = pair_sentiment.get('bias', 'off')
        sentiment_confidence_pct = pair_sentiment.get('confidence', 50)

        if sentiment_bias == 'off' or not pair_sentiment.get('active', False):
            return False, f"Rechazado por Sentimiento: Trading para {symbol} está DESACTIVADO."

        # 1. Validación de Bias
        if (signal == 1 and sentiment_bias == 'short') or \
           (signal == -1 and sentiment_bias == 'long'):
            return False, f"Rechazado por Sentimiento: Señal ({'BUY' if signal==1 else 'SELL'}) contradice Bias ({sentiment_bias})."

        # 2. Modulación de Confianza (Lógica Superadora)
        # Se requiere más confianza del modelo si el sentimiento del analista es bajo.
        required_ml_confidence = 0.60 # Umbral base del modelo ML [cite: 103]
        
        # Ajustamos el umbral requerido basado en la confianza del dashboard
        # Si la confianza del dashboard es alta (85%), bajamos el requisito del ML.
        # Si es baja (65%), lo subimos para ser más exigentes.
        adjustment_factor = 1 - ((sentiment_confidence_pct - 75) / 100) # Centrado en 75%
        adjusted_threshold = required_ml_confidence * adjustment_factor
        
        if ml_confidence < adjusted_threshold:
            return False, f"Rechazado por Sentimiento: Confianza ML ({ml_confidence:.2f}) no supera umbral ajustado ({adjusted_threshold:.2f}) por confianza de dashboard ({sentiment_confidence_pct}%)."

        return True, "Aprobado por Filtro de Sentimiento."