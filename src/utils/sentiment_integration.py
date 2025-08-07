import json
import os
from datetime import datetime, date
import logging

class DailySentimentManager:
    """
    Gestor de sentimiento diario para integrar con el trading bot
    """
    
    def __init__(self, config_dir="configs"):
        self.config_dir = config_dir
        self.logger = logging.getLogger(__name__)
        
        # Crear directorio si no existe
        os.makedirs(config_dir, exist_ok=True)
    
    def get_today_config_file(self):
        """Retorna el path del archivo de configuración de hoy"""
        today = date.today().strftime('%Y%m%d')
        return os.path.join(self.config_dir, f"daily_sentiment_{today}.json")
    
    def save_sentiment_config(self, config_data):
        """
        Guarda la configuración de sentimiento del día
        
        Args:
            config_data (dict): Configuración desde el dashboard web
        """
        config_file = self.get_today_config_file()
        
        try:
            # Añadir metadatos
            full_config = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "source": "web_dashboard",
                    "version": "1.0"
                },
                "sentiment_config": config_data
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(full_config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Configuración de sentimiento guardada: {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error guardando configuración: {e}")
            return False
    
    def load_sentiment_config(self, specific_date=None):
        """
        Carga la configuración de sentimiento del día (o fecha específica)
        
        Args:
            specific_date (str, optional): Fecha en formato YYYYMMDD
            
        Returns:
            dict: Configuración de sentimiento o None si no existe
        """
        if specific_date:
            config_file = os.path.join(self.config_dir, f"daily_sentiment_{specific_date}.json")
        else:
            config_file = self.get_today_config_file()
        
        if not os.path.exists(config_file):
            self.logger.info(f"No existe configuración de sentimiento para hoy: {config_file}")
            return None
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                full_config = json.load(f)
            print("Configuración de sentimiento cargada.")


        except FileNotFoundError:
            # ✅ AÑADIDO: Maneja el caso en que el archivo no existe
            print("⚠️ Archivo de configuración de sentimiento no encontrado. Usando valores por defecto.")
            full_config = {} # Asigna un valor por defecto para que el programa no falle


        except json.JSONDecodeError:
            # ✅ AÑADIDO: Maneja el caso en que el archivo JSON está corrupto
            print("❌ Error al leer el archivo de configuración de sentimiento. Está mal formado.")
            full_config = {} # Asigna un valor por defecto

        except Exception as e:
            # ✅ AÑADIDO: Una captura general para cualquier otro error
            print(f"❌ Ocurrió un error inesperado al cargar la configuración: {e}")
            full_config = {} # Asigna un valor por defecto














