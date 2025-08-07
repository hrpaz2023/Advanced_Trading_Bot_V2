# src/utils/logger_config.py
import logging
import os

# --- Construcción de Ruta Absoluta y Robusta ---
# 1. Obtener la ruta del directorio raíz del proyecto (dos niveles por encima de 'src/utils')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# 2. Definir la carpeta de logs usando la ruta absoluta del proyecto
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# 3. Crear el directorio de logs si no existe
os.makedirs(LOG_DIR, exist_ok=True)
# --- Fin de la construcción de ruta ---

LOG_FORMAT = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def setup_logger(name, log_file, level=logging.INFO):
    """Función para configurar un logger."""
    # Usar la ruta absoluta para el archivo de log
    handler = logging.FileHandler(os.path.join(LOG_DIR, log_file))
    handler.setFormatter(LOG_FORMAT)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger

# --- Creación de los Loggers Específicos (sin cambios en la lógica) ---
trades_logger = setup_logger('trades_logger', 'trades.log')
rejected_logger = setup_logger('rejected_logger', 'rejected_signals.log')
activity_logger = setup_logger('activity_logger', 'bot_activity.log')

print("✅ Sistema de logging configurado desde logger_config.py (con rutas absolutas)")