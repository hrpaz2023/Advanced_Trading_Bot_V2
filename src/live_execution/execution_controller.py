from __future__ import annotations

NAIVE_AWARE_WARN_ONCE = False
# src/live_execution/execution_controller.py
# -----------------------------------------------------------------------------
# OptimizedExecutionController
# Controlador de ejecución por símbolo/estrategia:
# - Genera señal base (TA) según estrategia
# - Confirma con Modelo ML + ChromaDB numérico (253 dims)
# - Verifica que NO haya posiciones abiertas antes de generar señales
# - Lee configuración real desde orchestrator_config.json
# - Usa análisis causal real desde causal_insights.json
# - Implementa ponderación por distancia y normalización de outcomes
# - Sistema de tracking y logging de señales rechazadas
# -----------------------------------------------------------------------------


import os
import sys
import time
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ML
import joblib
import pickle
import gzip

# Chroma (opcional, robusto si no está disponible)
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False

# MT5 opcional (usado solo como fallback si TradingClient no provee OHLC)
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except Exception:
    MT5_AVAILABLE = False


# -----------------------------------------------------------------------------
# Logger básico
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(_sh)
logger.setLevel(logging.INFO)

from datetime import datetime, timezone
import pandas as pd


def is_future_or_equal(mt, ref):
    """Devuelve True si mt >= ref tras normalizar ambas fechas a UTC-aware.
    Si alguna no puede parsearse, devuelve False (no bloquea por look-ahead)."""
    try:
        mt2 = to_aware_utc(mt)
        ref2 = to_aware_utc(ref)
        if mt2 is None or ref2 is None:
            return False
        # Unificar tz-aware por si a alguien se le escapó algo
        if getattr(mt2, "tzinfo", None) is None:
            from datetime import timezone
            mt2 = mt2.replace(tzinfo=timezone.utc)
        if getattr(ref2, "tzinfo", None) is None:
            from datetime import timezone
            ref2 = ref2.replace(tzinfo=timezone.utc)
        return mt2 >= ref2
    except Exception:
        return False
def to_aware_utc(ts):
    """Convierte ts (str/int/float/datetime/pd.Timestamp) a datetime aware en UTC. Devuelve None si no puede."""
    if ts is None:
        return None

    # pandas.Timestamp
    if isinstance(ts, pd.Timestamp):
        try:
            if ts.tz is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return ts.to_pydatetime()
        except Exception:
            pass

    # datetime.datetime
    if isinstance(ts, datetime):
        try:
            return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
        except Exception:
            pass

    # numérico epoch (ms o s)
    if isinstance(ts, (int, float)):
        try:
            # Heurística ms vs s
            sec = ts/1000.0 if ts > 1e12 else ts
            return datetime.fromtimestamp(sec, tz=timezone.utc)
        except Exception:
            pass

    # string ISO / otros
    if isinstance(ts, str):
        s = ts.strip()
        try:
            # ISO con Z
            if s.endswith("Z"):
                s = s.replace("Z", "+00:00")
            # Pandas parser robusto
            t = pd.to_datetime(s, utc=True, errors="coerce")
            if pd.isna(t):
                return None
            return t.to_pydatetime()
        except Exception:
            pass

    return None



# -----------------------------------------------------------------------------
# Utilidades de TA (sin dependencias externas)
# -----------------------------------------------------------------------------
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=1).mean()
    rs = np.where(loss == 0, np.nan, gain / loss)
    out = 100 - (100 / (1 + rs))
    return pd.Series(out, index=series.index).bfill()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']; low = df['low']; close = df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()

def donchian_channels(df: pd.DataFrame, period: int = 20) -> Tuple[pd.Series, pd.Series]:
    upper = df['high'].rolling(window=period, min_periods=period).max()
    lower = df['low'].rolling(window=period, min_periods=period).min()
    return upper, lower


# -----------------------------------------------------------------------------
# Controlador
# -----------------------------------------------------------------------------
class OptimizedExecutionController:

    def _make_where(self):
        """Filtro Chroma compatible con la API actual.
        Si hay timeframe, usa $and; si no, solo symbol.
        """
        tf = getattr(self, 'timeframe', None)
        if tf:
            return {"$and": [{"symbol": {"$eq": self.symbol}}, {"timeframe": {"$eq": tf}}]}
        return {"symbol": {"$eq": self.symbol}}
    LOOKBACK_BARS = 400

    def __init__(
        self,
        trading_client: Any,
        symbol: str,
        strategy_name: str,
        strategy_params: Optional[Dict[str, Any]] = None,
        notifier: Any = None
    ):
        self.trading_client = trading_client
        self.symbol = symbol
        self.strategy_name = strategy_name
        self.strategy_params = strategy_params or {}
        self.notifier = notifier

        self.timeframe: str = getattr(self, "timeframe", "M5")

        # Configuración dinámica desde orchestrator_config.json
        self.orchestrator_config = self._load_orchestrator_config()
        self.chroma_config = self.orchestrator_config.get("chroma", {})
        self.thresholds = self.orchestrator_config.get("thresholds", {})
        self.filters_config = self.orchestrator_config.get("filters", {})
        
        # Umbrales desde configuración (NO hardcodeados)
        self.MODEL_CONFIDENCE_THRESHOLD = float(self.thresholds.get("ml_confidence_min", 0.45))
        self.HISTORICAL_PROB_THRESHOLD = float(self.thresholds.get("historical_prob_min", 0.45))
        
        # Cargar análisis causal real
        self.causal_insights = self._load_causal_insights()
        self.correlation_matrix = self._load_correlation_matrix()

        self.models_dir = os.path.join("models", self.strategy_name)
        self.model = None
        self.scaler = None
        self.model_features: List[str] = []

        self.chroma_client = None
        self.collection = None
        self.chroma_path = self.chroma_config.get("path", os.path.join("db", "chroma_db"))
        self.chroma_collection_name = self.chroma_config.get("collection", "historical_market_states")
        self.chroma_n_results = self.chroma_config.get("n_results", 10)

        self.perf = { "last_signal_ms": None, "avg_signal_ms": None, "signals": 0 }

        # Control de posiciones - referencia al PolicySwitcher para fallback
        self.policy_switcher = None

        # 🔧 NUEVO: Sistema de tracking de señales rechazadas
        self.last_rejected_signal = None
        self.last_signal_attempt = None
        self.signal_stats = {
            'total_attempts': 0,
            'signals_generated': 0,
            'signals_rejected': 0,
            'rejection_reasons': {}
        }
        
        # 🔧 NUEVO: Función de logging externa (será asignada por main_bot)
        self._external_log_function = None

        self._load_ml_artifacts()
        self._init_chroma()

        logger.info(f"✅ {self.symbol}_{self.strategy_name} listo. "
                   f"ML: {'OK' if self.model else 'NO'} | "
                   f"Chroma: {'OK' if self.collection else 'NO'} | "
                   f"Causal: {'OK' if self.causal_insights else 'NO'} | "
                   f"Umbrales: ML={self.MODEL_CONFIDENCE_THRESHOLD} Hist={self.HISTORICAL_PROB_THRESHOLD}")

    def set_policy_switcher(self, policy_switcher):
        """Establece referencia al PolicySwitcher para verificación de posiciones"""
        self.policy_switcher = policy_switcher

    # 🔧 NUEVO: Método para recibir función de logging desde main_bot
    def set_external_log_function(self, log_function):
        """Establece la función de logging externa desde main_bot"""
        self._external_log_function = log_function

    # -------------------------------------------------------------------------
    # Carga de configuraciones y datos reales
    # -------------------------------------------------------------------------
    def _load_orchestrator_config(self) -> Dict[str, Any]:
        """Carga configuración real desde orchestrator_config.json"""
        config_path = "orchestrator_config.json"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"✅ Configuración orquestador cargada desde {config_path}")
            return config
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar {config_path}: {e}. Usando valores por defecto.")
            return {
                "thresholds": {"ml_confidence_min": 0.45, "historical_prob_min": 0.45},
                "chroma": {"path": "db/chroma_db", "collection": "historical_market_states", "n_results": 10},
                "filters": {"use_causal_filter": False, "use_live_corr": False}
            }

    def _load_causal_insights(self) -> Optional[Dict[str, Any]]:
        """Carga análisis causal real desde causal_insights.json"""
        if not self.filters_config.get("use_causal_filter", False):
            logger.info(f"🔧 Filtro causal deshabilitado para {self.symbol}_{self.strategy_name}")
            return None
            
        causal_path = os.path.join(
            self.orchestrator_config.get("paths", {}).get("causal_reports_dir", "reports/causal_reports"),
            "causal_insights.json"
        )
        
        try:
            with open(causal_path, 'r', encoding='utf-8') as f:
                causal_data = json.load(f)
            
            # Buscar datos específicos para este combo
            combo_key = f"{self.symbol}_{self.strategy_name}"
            for insight in causal_data:
                if (insight.get("asset") == self.symbol and 
                    insight.get("strategy") == self.strategy_name):
                    logger.info(f"✅ Datos causales cargados para {combo_key}: "
                               f"PF={insight.get('profit_factor', 0):.3f} "
                               f"ATE={insight.get('ATE', 0):.1f} "
                               f"Sig={insight.get('statistical_significance', {}).get('significance', 'N/A')}")
                    return insight
            
            logger.warning(f"⚠️ No se encontraron datos causales para {combo_key}")
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Error cargando datos causales: {e}")
            return None

    def _load_correlation_matrix(self) -> Optional[Dict[str, Any]]:
        """Carga matriz de correlaciones si el filtro está habilitado"""
        if not self.filters_config.get("use_live_corr", False):
            return None
            
        try:
            correlation_dir = self.orchestrator_config.get("paths", {}).get("cross_correlation_dir", "reports/cross_correlation_reports")
            # Buscar el archivo más reciente de correlaciones
            import glob
            pattern = os.path.join(correlation_dir, "*.csv")
            correlation_files = glob.glob(pattern)
            
            if correlation_files:
                latest_file = max(correlation_files, key=os.path.getctime)
                corr_df = pd.read_csv(latest_file, index_col=0)
                logger.info(f"✅ Matriz de correlaciones cargada: {latest_file}")
                return corr_df.to_dict()
            else:
                logger.warning(f"⚠️ No se encontraron archivos de correlación en {correlation_dir}")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Error cargando correlaciones: {e}")
            return None

    # -------------------------------------------------------------------------
    # Verificación de posiciones abiertas
    # -------------------------------------------------------------------------
    def _get_mt5_positions(self, symbol: str) -> List[Dict[str, Any]]:
        """Obtiene posiciones abiertas desde MT5 para el símbolo específico"""
        if not MT5_AVAILABLE:
            return []
        
        try:
            positions = mt5.positions_get(symbol=symbol)
            if positions is None:
                return []
            
            result = []
            for pos in positions:
                result.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': pos.type,  # 0=BUY, 1=SELL
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'profit': pos.profit,
                    'comment': pos.comment
                })
            return result
        except Exception as e:
            logger.warning(f"⚠️ Error obteniendo posiciones MT5 para {symbol}: {e}")
            return []

    def _get_trading_client_positions(self, symbol: str) -> List[Dict[str, Any]]:
        """Intenta obtener posiciones desde TradingClient con varios métodos"""
        if self.trading_client is None:
            return []

        methods = [
            'get_positions',
            'positions_get', 
            'get_open_positions',
            'open_positions',
            'positions'
        ]

        for method_name in methods:
            if hasattr(self.trading_client, method_name):
                try:
                    method = getattr(self.trading_client, method_name)
                    # Intenta con símbolo específico primero
                    try:
                        positions = method(symbol=symbol)
                    except TypeError:
                        # Si no acepta parámetros, obtiene todas y filtra
                        all_positions = method()
                        if isinstance(all_positions, (list, tuple)):
                            positions = [p for p in all_positions 
                                       if isinstance(p, dict) and p.get('symbol') == symbol]
                        else:
                            positions = all_positions

                    if positions:
                        logger.debug(f"✅ Posiciones obtenidas vía {method_name}: {len(positions) if isinstance(positions, list) else 1}")
                        return positions if isinstance(positions, list) else [positions]
                        
                except Exception as e:
                    logger.debug(f"Método {method_name} falló: {e}")
                    continue
        
        return []

    def _has_open_position(self, symbol: str) -> bool:
        """
        Verifica si hay posiciones abiertas para el símbolo.
        Prioriza MT5, luego TradingClient, luego PolicySwitcher como fallback.
        """
        # 1. Verificar en MT5 (fuente más confiable)
        mt5_positions = self._get_mt5_positions(symbol)
        if mt5_positions:
            logger.info(f"🔒 [{symbol}] Posición detectada en MT5: {len(mt5_positions)} posición(es)")
            return True

        # 2. Verificar en TradingClient
        client_positions = self._get_trading_client_positions(symbol)
        if client_positions:
            logger.info(f"🔒 [{symbol}] Posición detectada en TradingClient: {len(client_positions)} posición(es)")
            return True

        # 3. Fallback: Verificar en PolicySwitcher
        if self.policy_switcher and hasattr(self.policy_switcher, 'open_positions'):
            try:
                open_positions = getattr(self.policy_switcher, 'open_positions', {})
                symbol_positions = [pos for pos in open_positions.values() 
                                  if isinstance(pos, dict) and pos.get('symbol') == symbol]
                if symbol_positions:
                    logger.info(f"🔒 [{symbol}] Posición detectada en PolicySwitcher: {len(symbol_positions)} posición(es)")
                    return True
            except Exception as e:
                logger.debug(f"Error verificando PolicySwitcher: {e}")

        # 4. No hay posiciones detectadas
        logger.debug(f"✅ [{symbol}] Sin posiciones abiertas detectadas")
        return False

    # -------------------------------------------------------------------------
    # Carga de modelos (robusta)
    # -------------------------------------------------------------------------
    def _try_joblib(self, path: str):
        try:
            return joblib.load(path)
        except Exception:
            return None

    def _try_pickle(self, path: str):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _try_gzip_pickle(self, path: str):
        try:
            with gzip.open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _candidate_paths(self, base_dir: str, symbol: str):
        return [
            os.path.join(base_dir, f"{symbol}_confirmation_model.pkl"),
            os.path.join(base_dir, f"{symbol}_confirmation_model.joblib"),
            os.path.join(base_dir, f"{symbol}_model.pkl"),
            os.path.join(base_dir, f"{symbol}_model.joblib"),
            os.path.join(base_dir, "confirmation_model.pkl"),
            os.path.join(base_dir, "confirmation_model.joblib"),
        ], [
            os.path.join(base_dir, f"{symbol}_feature_scaler.pkl"),
            os.path.join(base_dir, f"{symbol}_feature_scaler.joblib"),
            os.path.join(base_dir, "feature_scaler.pkl"),
            os.path.join(base_dir, "feature_scaler.joblib"),
        ], [
            os.path.join(base_dir, f"{symbol}_model_features.joblib"),
            os.path.join(base_dir, "model_features.joblib"),
        ]

    def _smart_load(self, path: str):
        obj = self._try_joblib(path)
        if obj is not None:
            return obj
        obj = self._try_pickle(path)
        if obj is not None:
            return obj
        obj = self._try_gzip_pickle(path)
        return obj

    def _load_ml_artifacts(self):
        """
        Intenta cargar modelo, scaler y features buscando nombres alternativos y
        probando diferentes loaders. Loguea causas claras si no se puede.
        """
        try:
            model_cands, scaler_cands, feats_cands = self._candidate_paths(self.models_dir, self.symbol)

            # FEATURES (obligatorio para alinear columnas)
            feats = None
            for p in feats_cands:
                if os.path.exists(p):
                    try:
                        feats = joblib.load(p)
                        if isinstance(feats, list) and all(isinstance(c, str) for c in feats):
                            self.model_features = feats
                            break
                    except Exception:
                        pass
            if not self.model_features:
                logger.warning(f"⚠️ Features no encontradas para {self.symbol} en {self.models_dir} "
                               f"(buscado: {', '.join(os.path.basename(x) for x in feats_cands)})")

            # SCALER
            for p in scaler_cands:
                if os.path.exists(p):
                    obj = self._smart_load(p)
                    if obj is not None:
                        self.scaler = obj
                        break
            if self.scaler is None:
                logger.warning(f"⚠️ Scaler no encontrado o ilegible para {self.symbol} en {self.models_dir}")

            # MODEL
            for p in model_cands:
                if os.path.exists(p):
                    obj = self._smart_load(p)
                    if obj is not None:
                        self.model = obj
                        break
            if self.model is None:
                logger.warning(f"⚠️ Modelo no encontrado o ilegible para {self.symbol} en {self.models_dir}")

            # Validaciones cruzadas
            if self.model and (self.scaler is None or not self.model_features):
                logger.warning(f"⚠️ Modelo cargado pero faltan scaler/features para {self.symbol} ({self.strategy_name})")
            if self.scaler and not self.model_features:
                logger.warning(f"⚠️ Scaler cargado sin features (orden de columnas) para {self.symbol} ({self.strategy_name})")

            if not self.model or not self.scaler or not self.model_features:
                logger.warning(f"⚠️ Modelos ML no disponibles para {self.symbol} en {self.models_dir}")

        except Exception as e:
            logger.warning(f"⚠️ Error cargando artefactos ML ({self.symbol}/{self.strategy_name}): {e}")

    # -------------------------------------------------------------------------
    # Chroma
    # -------------------------------------------------------------------------
    def _init_chroma(self):
        if not CHROMA_AVAILABLE:
            logger.warning("⚠️ ChromaDB no disponible en el entorno.")
            return
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=ChromaSettings(allow_reset=False, anonymized_telemetry=False)
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.chroma_collection_name
            )
            logger.info(f"✅ ChromaDB inicializado: {self.chroma_path}/{self.chroma_collection_name}")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo inicializar ChromaDB: {e}")
            self.collection = None

    def _normalize_outcome(self, outcome) -> int:
        """
        Normaliza outcome a formato binario {0, 1} según especificación del manual:
        BUY/SELL/±1/0/bool -> {1,0}
        """
        if outcome is None:
            return 0
        
        # String outcomes
        if isinstance(outcome, str):
            if outcome.upper() in ['BUY', 'LONG', 'UP', '1', 'TRUE', 'WIN', 'PROFIT']:
                return 1
            elif outcome.upper() in ['SELL', 'SHORT', 'DOWN', '0', 'FALSE', 'LOSS']:
                return 0
        
        # Numeric outcomes
        elif isinstance(outcome, (int, float)):
            if outcome > 0:
                return 1
            else:
                return 0
        
        # Boolean outcomes
        elif isinstance(outcome, bool):
            return 1 if outcome else 0
        
        # Default case
        return 0

    # -------------------------------------------------------------------------
    # OHLC helpers
    # -------------------------------------------------------------------------
    def _map_timeframe_to_mt5(self, tf: str) -> Optional[int]:
        tf = (tf or "M5").upper()
        return getattr(mt5, f"TIMEFRAME_{tf}", None)

    def _get_ohlc(self, bars: int) -> Optional[pd.DataFrame]:
        """Intenta obtener OHLC desde trading_client; si no, fallback a MT5."""
        try_methods = [
            ("get_ohlc", lambda: self.trading_client.get_ohlc(self.symbol, self.timeframe, bars)),
            ("get_rates", lambda: self.trading_client.get_rates(self.symbol, self.timeframe, bars)),
            ("get_candles", lambda: self.trading_client.get_candles(self.symbol, self.timeframe, bars)),
            ("copy_rates", lambda: self.trading_client.copy_rates(self.symbol, self.timeframe, bars)),
        ]
        if self.trading_client is not None:
            for name, fn in try_methods:
                if hasattr(self.trading_client, name):
                    try:
                        df = fn()
                        if df is not None and len(df) > 0:
                            return self._normalize_ohlc_df(df)
                    except Exception as e:
                        logger.debug(f"Fallo al llamar {name} en trading_client: {e}")
                        pass

        if MT5_AVAILABLE:
            try:
                mt5_tf = self._map_timeframe_to_mt5(self.timeframe)
                if mt5_tf is None:
                    logger.warning(f"Timeframe '{self.timeframe}' no es válido para MT5.")
                    return None
                
                rates = mt5.copy_rates_from_pos(self.symbol, mt5_tf, 0, bars)
                if rates is not None:
                    df = pd.DataFrame(rates)
                    return self._normalize_ohlc_df(df)
            except Exception as e:
                logger.warning(f"Error en fallback de MT5: {e}")
                pass

        return None

    def _normalize_ohlc_df(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_map = {"time":"timestamp", "Time":"timestamp", "open":"open", "Open":"open",
                    "high":"high", "High":"high", "low":"low", "Low":"low", "close":"close", "Close":"close",
                    "volume":"volume", "Volume":"volume", "tick_volume":"volume", "real_volume": "volume"}
        _df = df.rename(columns={k:v for k,v in cols_map.items() if k in df.columns}).copy()
        
        if "timestamp" not in _df.columns and isinstance(_df.index, pd.DatetimeIndex):
            _df["timestamp"] = _df.index
        
        if "timestamp" in _df.columns:
            _df["timestamp"] = pd.to_datetime(_df["timestamp"], unit='s', errors='coerce', utc=True)
            _df = _df.dropna(subset=["timestamp"])
        else:
            return pd.DataFrame()

        required_cols = ["timestamp", "open", "high", "low", "close"]
        for col in required_cols:
            if col not in _df.columns: return pd.DataFrame()
        
        if "volume" not in _df.columns: _df["volume"] = 0

        _df = _df[required_cols + ["volume"]].sort_values("timestamp").reset_index(drop=True)
        return _df.tail(self.LOOKBACK_BARS)

    # -------------------------------------------------------------------------
    # Señal base por estrategia (TA simple + params opcionales)
    # -------------------------------------------------------------------------
    def _compute_strategy_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if df is None or len(df) < 50:
            return None

        close = df["close"]
        _atr = atr(df, int(self.strategy_params.get("atr_period", 14))).iloc[-1]
        entry_price = float(close.iloc[-1])
        strat = self.strategy_name.lower()

        if strat == "ema_crossover":
            fast = int(self.strategy_params.get("ema_fast", 12))
            slow = int(self.strategy_params.get("ema_slow", 26))
            efast = ema(close, fast); eslow = ema(close, slow)
            if efast.iloc[-2] <= eslow.iloc[-2] and efast.iloc[-1] > eslow.iloc[-1]:
                return {"action":"BUY","entry_price":entry_price,"atr":float(_atr)}
            if efast.iloc[-2] >= eslow.iloc[-2] and efast.iloc[-1] < eslow.iloc[-1]:
                return {"action":"SELL","entry_price":entry_price,"atr":float(_atr)}
            return None

        if strat == "rsi_pullback":
            per = int(self.strategy_params.get("rsi_period", 14))
            up = float(self.strategy_params.get("rsi_upper", 70.0))
            lo = float(self.strategy_params.get("rsi_lower", 30.0))
            r = rsi(close, per)
            if r.iloc[-2] < lo and r.iloc[-1] > lo:
                return {"action":"BUY","entry_price":entry_price,"atr":float(_atr)}
            if r.iloc[-2] > up and r.iloc[-1] < up:
                return {"action":"SELL","entry_price":entry_price,"atr":float(_atr)}
            return None

        if strat == "multi_filter_scalper":
            ema_fast = ema(close, 8)
            ema_slow = ema(close, 21)
            rsi_val = rsi(close, 14).iloc[-1]
            
            if (ema_fast.iloc[-1] > ema_slow.iloc[-1] and 
                ema_fast.iloc[-2] <= ema_slow.iloc[-2] and 
                rsi_val < 70):
                return {"action":"BUY","entry_price":entry_price,"atr":float(_atr)}
            
            if (ema_fast.iloc[-1] < ema_slow.iloc[-1] and 
                ema_fast.iloc[-2] >= ema_slow.iloc[-2] and 
                rsi_val > 30):
                return {"action":"SELL","entry_price":entry_price,"atr":float(_atr)}
            return None

        if strat == "volatility_breakout":
            atr_period = int(self.strategy_params.get("atr_period", 20))
            atr_multiplier = float(self.strategy_params.get("atr_multiplier", 2.0))
            lookback = int(self.strategy_params.get("lookback", 20))
            
            current_atr = atr(df, atr_period).iloc[-1]
            recent_high = df["high"].rolling(lookback).max().iloc[-2]
            recent_low = df["low"].rolling(lookback).min().iloc[-2]
            
            current_close = close.iloc[-1]
            
            if current_close > recent_high + (current_atr * atr_multiplier):
                return {"action":"BUY","entry_price":entry_price,"atr":float(_atr)}
            
            if current_close < recent_low - (current_atr * atr_multiplier):
                return {"action":"SELL","entry_price":entry_price,"atr":float(_atr)}
            return None

        if strat == "channel_reversal":
            period = int(self.strategy_params.get("channel_period", 20))
            upper_channel, lower_channel = donchian_channels(df, period)
            rsi_val = rsi(close, 14).iloc[-1]
            
            current_close = close.iloc[-1]
            
            if (current_close <= lower_channel.iloc[-1] * 1.001 and rsi_val < 30):
                return {"action":"BUY","entry_price":entry_price,"atr":float(_atr)}
            
            if (current_close >= upper_channel.iloc[-1] * 0.999 and rsi_val > 70):
                return {"action":"SELL","entry_price":entry_price,"atr":float(_atr)}
            return None

        # Estrategia por defecto
        logger.warning(f"⚠️ Estrategia '{strat}' no implementada, usando EMA crossover por defecto")
        fast = 12; slow = 26
        efast = ema(close, fast); eslow = ema(close, slow)
        if efast.iloc[-2] <= eslow.iloc[-2] and efast.iloc[-1] > eslow.iloc[-1]:
            return {"action":"BUY","entry_price":entry_price,"atr":float(_atr)}
        if efast.iloc[-2] >= eslow.iloc[-2] and efast.iloc[-1] < eslow.iloc[-1]:
            return {"action":"SELL","entry_price":entry_price,"atr":float(_atr)}
        return None

    # -------------------------------------------------------------------------
    # Preparación de features para ML (alineadas a self.model_features)
    # -------------------------------------------------------------------------
    def _prepare_live_feature_row(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if not self.model_features:
            return None

        feats: Dict[str, float] = {}
        close = df["close"]
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last

        # --- Base Features ---
        feats["close"] = float(last["close"])
        feats["open"] = float(last["open"])
        feats["high"] = float(last["high"])
        feats["low"] = float(last["low"])
        feats["volume"] = float(last.get("volume", 0.0))
        feats["candle_ret"] = (last["close"] / prev["close"] - 1.0) if prev["close"] else 0.0

        # --- Dynamic Feature Calculation based on self.model_features ---
        for feature_name in self.model_features:
            if feature_name in feats:
                continue # Ya calculado
            
            try:
                # Extraer tipo de indicador y período (ej: "ema_12" -> "ema", "12")
                match = re.match(r"([a-zA-Z]+)(\d+)", feature_name)
                if not match: continue

                indicator, period_str = match.groups()
                period = int(period_str)
                
                if indicator == "ema":
                    feats[feature_name] = float(ema(close, period).iloc[-1])
                elif indicator == "sma":
                    feats[feature_name] = float(sma(close, period).iloc[-1])
                elif indicator == "rsi":
                    feats[feature_name] = float(rsi(close, period).iloc[-1])
                elif indicator == "atr":
                    feats[feature_name] = float(atr(df, period).iloc[-1])
                # Añadir más indicadores aquí si es necesario
            
            except Exception as e:
                logger.warning(f"No se pudo calcular la feature dinámica '{feature_name}': {e}")
                feats[feature_name] = 0.0 # Default value on error

        # Asegura que todas las columnas estén presentes y en el orden correcto
        final_row = {col: float(feats.get(col, 0.0)) for col in self.model_features}
        return pd.DataFrame([final_row], columns=self.model_features)

    # -------------------------------------------------------------------------
    # Análisis Causal Real (no simulado)
    # -------------------------------------------------------------------------
    
    def _apply_causal_analysis(self, signal: int, signal_data: Dict[str, Any]) -> Tuple[float, str]:
        """
        Aplica análisis causal real desde causal_insights.json.
        Retorna (factor_ajuste, razon_rechazo_o_aprobacion)
        """
        if not self.causal_insights:
            return 1.0, "Sin datos causales disponibles"

        try:
            profit_factor = self.causal_insights.get("profit_factor", 1.0)
            ate = self.causal_insights.get("ATE", 0.0)
            significance = self.causal_insights.get("statistical_significance", {}).get("significance", "not_significant")
            best_regime = self.causal_insights.get("best_volatility_regime_causal", "unknown")

            # Filtros según configuración
            causal_config = self.filters_config.get("causal", {})
            min_pf_hard = float(causal_config.get("min_pf_hard", 0.95))
            min_ate_threshold = float(causal_config.get("min_ate_threshold", -1.0))
            pf_penalty_factor = float(causal_config.get("pf_penalty_factor", 0.9))
            enforce_regime = bool(causal_config.get("enforce_regime", False))

            # HARD REJECT sólo si confluyen señales realmente negativas
            # (PF < min_pf_hard) Y (ATE < min_ate_threshold) Y (significancia alta o significativa)
            if (profit_factor < min_pf_hard) and (ate < min_ate_threshold) and (significance in ("significant","highly_significant")):
                return 0.0, f"PF causal duro ({profit_factor:.3f} < {min_pf_hard}) y ATE {ate:.1f} con {significance}"

            # FACTOR DE AJUSTE por significancia
            significance_multiplier = {
                "highly_significant": 1.1,
                "significant": 1.05,
                "not_significant": 0.97
            }.get(significance, 1.0)

            # FACTOR DE AJUSTE por Profit Factor
            if profit_factor >= 1.2:
                pf_multiplier = 1.12
            elif profit_factor >= 1.1:
                pf_multiplier = 1.06
            elif profit_factor < 1.0:
                pf_multiplier = pf_penalty_factor
            else:
                pf_multiplier = 1.0

            # Penalización suave por ATE negativo (sin bloquear)
            ate_penalty = 1.0 if ate >= 0 else max(0.85, 1.0 + (ate / 100.0))

            # Verificación de régimen (opcional)
            regime_multiplier = 1.0
            if enforce_regime:
                current_atr = signal_data.get("atr", 0.0)
                current_regime = "high_vol" if current_atr > 0.001 else "low_vol"
                if best_regime != "unknown" and current_regime != best_regime:
                    regime_multiplier = 0.92
                    logger.info(f"⚠️ [{self.symbol}] Régimen subóptimo: actual={current_regime}, óptimo={best_regime}")

            final_factor = significance_multiplier * pf_multiplier * regime_multiplier * ate_penalty
            reason = (f"Causal OK: PF={profit_factor:.3f} ATE={ate:.1f} Sig={significance} "
                      f"Régimen={best_regime} → Factor={final_factor:.3f}")
            logger.info(f"🧠 [{self.symbol}] {reason}")
            return final_factor, reason

        except Exception as e:
            logger.warning(f"⚠️ Error en análisis causal: {e}")
            return 1.0, f"Error causal: {e}"

    # -------------------------------------------------------------------------
    # Análisis de Correlaciones Real
    # -------------------------------------------------------------------------
    def _apply_correlation_analysis(self, signal: int) -> Tuple[float, str]:
        """
        Aplica análisis de correlaciones desde cross_correlation_reports.
        Retorna (factor_ajuste, razon)
        """
        if not self.correlation_matrix:
            return 1.0, "Sin datos de correlación"
        
        try:
            # Obtener correlaciones para este símbolo
            symbol_correlations = self.correlation_matrix.get(self.symbol, {})
            
            # Buscar correlaciones altas con otros símbolos
            high_correlations = []
            correlation_threshold = self.orchestrator_config.get("correlation_caps", {}).get("cap_if_rho_gt", 0.85)
            
            for other_symbol, correlation in symbol_correlations.items():
                if other_symbol != self.symbol and abs(correlation) > correlation_threshold:
                    high_correlations.append((other_symbol, correlation))
            
            if high_correlations:
                # Verificar si hay posiciones en símbolos altamente correlacionados
                correlated_positions = 0
                for other_symbol, corr in high_correlations:
                    if self._has_open_position(other_symbol):
                        correlated_positions += 1
                
                if correlated_positions > 0:
                    # Penalizar por alta correlación con posiciones existentes
                    penalty_factor = 0.7  # Reducir confianza significativamente
                    reason = f"Alta correlación con {correlated_positions} posiciones activas"
                    logger.info(f"⚠️ [{self.symbol}] {reason}")
                    return penalty_factor, reason
            
            return 1.0, "Sin conflictos de correlación"
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis de correlación: {e}")
            return 1.0, f"Error correlación: {e}"

    # 🔧 NUEVO: Método para logging de señales rechazadas
    def _log_rejected_signal(self, signal_data: Dict[str, Any], reason: str):
        """Log de señales rechazadas usando la función externa o logging directo"""
        try:
            log_data = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "symbol": self.symbol,
                "strategy": self.strategy_name,
                "side": signal_data.get("action", ""),
                "entry_price": signal_data.get("entry_price", ""),
                "atr": signal_data.get("atr", 0.0),
                "ml_confidence": signal_data.get("confidence", 0.0),
                "historical_prob": signal_data.get("historical_prob", 0.0),
                "pnl": "",
                "status": "rejected_by_controller",
                "rejection_reason": reason,
                "position_size": 0,
                "ticket": ""
            }
            
            # Usar función externa si está disponible (preferido)
            if self._external_log_function:
                success = self._external_log_function(log_data)
                if success:
                    logger.debug(f"✅ Señal rechazada loggeada externamente: {self.symbol}")
                    return
            
            # Fallback: logging directo al CSV
            self._log_signal_direct_to_csv(log_data)
            
        except Exception as e:
            logger.error(f"❌ Error logging señal rechazada para {self.symbol}: {e}")

    def _log_signal_direct_to_csv(self, log_data: Dict[str, Any]):
        """Logging directo al CSV desde el controller (fallback)"""
        try:
            import csv
            
            log_path = "logs/signals_history.csv"
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            write_header = not os.path.exists(log_path)
            
            fieldnames = [
                "timestamp_utc", "symbol", "strategy", "side", "entry_price",
                "atr", "ml_confidence", "historical_prob", "pnl", "status",
                "rejection_reason", "position_size", "ticket"
            ]
            
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    w.writeheader()
                complete_row = {field: log_data.get(field, "") for field in fieldnames}
                w.writerow(complete_row)
            
            logger.debug(f"✅ Señal loggeada directamente al CSV: {self.symbol}")
            
        except Exception as e:
            logger.error(f"❌ Error en logging directo CSV: {e}")

    # -------------------------------------------------------------------------
    # Señales principales CON TRACKING Y LOGGING DE RECHAZOS
    # -------------------------------------------------------------------------
    def get_trading_signal(self) -> Optional[Dict[str, Any]]:
        """Método original, ahora con tracking y logging de rechazos"""
        result = self.get_trading_signal_with_details()
        
        if result and result['status'] == 'generated':
            return result['signal']
        else:
            return None

    # ------------------------------------------------------------------
    # MÉTODO PRINCIPAL: generación de señal + detalles
    # ------------------------------------------------------------------
    def get_trading_signal_with_details(
        self,
        symbol: str = None,
        candles: pd.DataFrame | None = None,
        *,
        forced_side: str | None = None,
        extra_context: dict[str, Any] | None = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Devuelve:
          {
              'signal'          : dict  → datos completos de la señal,
              'rejection_reason': str   → None si se aprobó,
              'status'          : 'generated' | 'rejected'
          }

        * `extra_context` permite adjuntar info externa (Trend-Change, Corr, etc.).
        """
        # Contador global
        self.signal_stats['total_attempts'] += 1

        # Limpia estado previo
        self.last_rejected_signal = None
        self.last_signal_attempt = None

        # --- Señal forzada ---------------------------------------------------
        if forced_side in ('BUY', 'SELL'):
            base_signal = {
                "action": forced_side,
                "entry_price": float(candles['close'].iloc[-1]) if candles is not None else 0.0,
                "confidence": 1.0,
                "historical_prob": 0.5,
                **(extra_context or {}),
            }
            self.signal_stats['signals_generated'] += 1
            return {'signal': base_signal, 'rejection_reason': None, 'status': 'generated'}

        # --- Delegamos al método interno con tracking ------------------------
        signal_data = self._get_trading_signal_with_tracking()

        if signal_data and not signal_data.get('_rejected'):
            # Señal válida
            if extra_context:
                signal_data.update(extra_context)
            self.signal_stats['signals_generated'] += 1
            return {'signal': signal_data, 'rejection_reason': None, 'status': 'generated'}

        # Señal rechazada
        if self.last_rejected_signal:
            if extra_context:
                self.last_rejected_signal['signal_data'].update(extra_context)
            self.signal_stats['signals_rejected'] += 1
            reason = self.last_rejected_signal['reason']
            return {
                'signal': self.last_rejected_signal['signal_data'],
                'rejection_reason': reason,
                'status': 'rejected'
            }

        # No hubo señal base
        return None

        
        # Obtener la señal usando el método interno con tracking
        signal_data = self._get_trading_signal_with_tracking()
        
        if signal_data and not signal_data.get('_rejected'):
            # Señal válida generada
            self.signal_stats['signals_generated'] += 1
            return {
                'signal': signal_data,
                'rejection_reason': None,
                'status': 'generated'
            }
        elif self.last_rejected_signal:
            # Señal rechazada con información completa
            self.signal_stats['signals_rejected'] += 1
            reason = self.last_rejected_signal['reason']
            
            # Contar razones de rechazo
            reason_key = reason.split(' ')[0].split('(')[0]  # Primera palabra sin paréntesis
            self.signal_stats['rejection_reasons'][reason_key] = (
                self.signal_stats['rejection_reasons'].get(reason_key, 0) + 1
            )
            
            return {
                'signal': self.last_rejected_signal['signal_data'],
                'rejection_reason': reason,
                'status': 'rejected'
            }
        else:
            # No hay señal base generada
            return None

    def _get_trading_signal_with_tracking(self) -> Optional[Dict[str, Any]]:
        """Método interno que implementa toda la lógica con tracking completo"""
        t0 = time.perf_counter()

        df = self._get_ohlc(self.LOOKBACK_BARS)
        if df is None or len(df) < 50:
            logger.warning(f"⚠️ No hay suficientes barras para {self.symbol}")
            return None
        logger.debug(f"✅ Datos obtenidos para {self.symbol}: {len(df)} velas")

        # Generar señal base
        base = self._compute_strategy_signal(df)
        if base is None:
            return None

        signal = 1 if base["action"].upper() == "BUY" else 0
        side_str = base["action"].upper()
        logger.info(f"🎯 [{self.symbol}] SEÑAL BASE: {side_str} (Estrategia: {self.strategy_name})")

        # Almacenar intento de señal para tracking
        self.last_signal_attempt = {
            'signal_data': base.copy(),
            'timestamp': datetime.now(timezone.utc)
        }

        # Preparar features para ML
        live_df_for_model = None
        scaled_df = None
        if self.model is not None and self.scaler is not None and self.model_features:
            live_df_for_model = self._prepare_live_feature_row(df)
            if live_df_for_model is not None:
                try:
                    scaled_df = pd.DataFrame(self.scaler.transform(live_df_for_model),
                                             columns=self.model_features)
                except Exception as e:
                    logger.error(f"Error escalando features para {self.symbol}: {e}")
                    return None

        
        # --- CHROMA NUMÉRICO: estimación robusta y sin fugas temporales ---
        historical_prob = 0.5  # Valor neutro por defecto
        chroma_samples_found = 0

        if self.collection is not None and scaled_df is not None:
            try:
                # Asegurar 1 sola fila de consulta
                if getattr(scaled_df, "shape", (0,))[0] != 1:
                    logger.warning(f"[{self.symbol}] scaled_df no tiene 1 fila; usando la primera si existe.")

                # Determinar tiempo de la señal para evitar look-ahead
                signal_time = base.get("timestamp") or base.get("time") or base.get("ts")
                if isinstance(signal_time, str):
                    try:
                        signal_time = datetime.fromisoformat(signal_time.replace("Z","+00:00"))
                    except Exception:
                        signal_time = datetime.now(timezone.utc)
                elif not isinstance(signal_time, datetime):
                    signal_time = datetime.now(timezone.utc)

                # Realizar query con metadatos y (opcional) distancias para diagnóstico
                results = self.collection.query(
                    query_embeddings=scaled_df.values.tolist(),  # 1 sola fila
                    n_results=int(self.chroma_n_results * 3),    # buscar más para diversificar
                    where=self._make_where(),  # ⟵ ahora sí
                    include=["metadatas", "distances"]           # ⟵ sin "ids"
                )


                metas = (results or {}).get("metadatas", [[]])
                metas = metas[0] if metas else []
                ids = (results or {}).get("ids", [[]])
                ids = ids[0] if ids else []
                distances = (results or {}).get("distances", [[]])
                distances = distances[0] if distances else []

                if metas:
                    def _parse_ts(m):
                        t = m.get("timestamp") or m.get("time") or m.get("ts")
                        if isinstance(t, str):
                            try:
                                return datetime.fromisoformat(t.replace("Z","+00:00"))
                            except Exception:
                                return None
                        return t if isinstance(t, datetime) else None

                    # Filtro: outcome válido y no mirar futuro (anti-leakage)
                    valid = []
                    seen = set()
                    for m, dist in zip(metas, distances if distances else [None]*len(metas)):
                        if not isinstance(m, dict):
                            continue
                        out = m.get("outcome")
                        if out is None:
                            continue
                        try:
                            out_i = int(out)
                        except Exception:
                            continue
                        if out_i not in (-1, 0, 1):
                            continue

                        mt = _parse_ts(m)
                        if mt is None or is_future_or_equal(mt, signal_time):
                            continue

                        key = (mt, m.get("entry_price"), m.get("trade_id"))
                        if key in seen:
                            continue
                        seen.add(key)

                        m["_outcome"] = out_i
                        m["_distance"] = float(dist) if dist is not None else None
                        valid.append(m)

                    # Decide si contamos 0 (break-even). Aquí los EXCLUIMOS del denominador.
                    valid = [m for m in valid if m["_outcome"] in (-1, 1)]

                    chroma_samples_found = len(valid)

                    if chroma_samples_found > 0:
                        is_buy = (signal == 1)
                        wins = sum(
                            1 for m in valid
                            if (m["_outcome"] > 0 and is_buy) or (m["_outcome"] < 0 and not is_buy)
                        )

                        # Estimación robusta: Wilson lower bound (90%)
                        import math
                        def wilson_lower_bound(w, n, z=1.64):
                            if n == 0:
                                return 0.5
                            p = w / n
                            denom = 1 + z*z/n
                            center = p + z*z/(2*n)
                            margin = z*math.sqrt((p*(1-p) + z*z/(4*n))/n)
                            return (center - margin) / denom

                        p_hat = wins / chroma_samples_found
                        p_lo  = wilson_lower_bound(wins, chroma_samples_found, z=1.64)  # ~90% CI conservador
                        historical_prob = max(0.001, min(0.999, p_hat))  # para logging / downstream
                        historical_prob_lb90 = p_lo          # alias para logging/persistencia
                        n_eff = chroma_samples_found         # si el reason usa n_eff

                        # Logging de diagnóstico
                        dists = [m.get("_distance") for m in valid if m.get("_distance") is not None]
                        if dists:
                            try:
                                avg_d = float(np.mean(dists)); min_d = float(np.min(dists)); max_d = float(np.max(dists))
                                logger.info(f"📊 [{self.symbol}] ChromaDB: {wins}/{chroma_samples_found} éxitos | "
                                            f"p̂={p_hat:.2%}, LB90={p_lo:.2%} ({side_str}) | "
                                            f"dist avg={avg_d:.1f} min={min_d:.1f} max={max_d:.1f}")
                            except Exception:
                                logger.info(f"📊 [{self.symbol}] ChromaDB: {wins}/{chroma_samples_found} éxitos | "
                                            f"p̂={p_hat:.2%}, LB90={p_lo:.2%} ({side_str})")
                        else:
                            logger.info(f"📊 [{self.symbol}] ChromaDB: {wins}/{chroma_samples_found} éxitos | "
                                        f"p̂={p_hat:.2%}, LB90={p_lo:.2%} ({side_str})")

                        # Decisión conservadora: usar LB90 vs umbral
                        min_samples = max(7, self.chroma_n_results // 3)
                        if chroma_samples_found >= min_samples and p_lo < self.HISTORICAL_PROB_THRESHOLD:
                            reason = (f"Prob. histórica conservadora baja (LB90={p_lo:.2%} "
                                      f"< {self.HISTORICAL_PROB_THRESHOLD:.2%}) con "
                                      f"{chroma_samples_found} casos.")
                            logger.info(f"❌ [{self.symbol}] RECHAZADA por {reason}")
                            signal_data_for_log = base.copy()
                            signal_data_for_log.update({
                                "confidence": 0.0,
                                "historical_prob": historical_prob,
                                "historical_lb90": p_lo,
                                "chroma_samples": chroma_samples_found,
                                "historical_prob_lb90": historical_prob_lb90,
                                "chroma_samples": chroma_samples_found,
                                "n_eff": float(n_eff),
                                "min_samples": int(min_samples)
                            })
                            self.last_rejected_signal = {
                                'signal_data': signal_data_for_log,
                                'reason': reason,
                                'timestamp': datetime.now(timezone.utc)
                            }
                            self._log_rejected_signal(signal_data_for_log, reason)
                            self._send_rejection_notification(side_str, reason)
                            self._update_perf(t0, approved=False, block_reason="chroma_filter")
                            return None
                    else:
                        logger.info(f"ℹ️ [{self.symbol}] ChromaDB: Sin casos válidos (±1) antes de la señal.")
                        historical_prob = 0.5
                else:
                    logger.info(f"ℹ️ [{self.symbol}] ChromaDB: Sin resultados para la consulta.")
                    historical_prob = 0.5

            except Exception as e:
                logger.warning(f"⚠️ Error validando con ChromaDB: {e}")
                historical_prob = 0.5

                logger.warning(f"⚠️ Error validando con ChromaDB: {e}")
                historical_prob = 0.5

        # --- MODELO ML CON LOGGING ---
        ml_confidence = 0.5  # Valor neutro por defecto
        if self.model is not None and scaled_df is not None:
            try:
                proba = self.model.predict_proba(scaled_df)[0]
                ml_confidence = float(proba[1] if signal == 1 else proba[0])
                
                # 🔧 FILTRO CON LOGGING: Confianza ML
                if ml_confidence < self.MODEL_CONFIDENCE_THRESHOLD:
                    reason = f"Modelo IA (conf: {ml_confidence:.2%} < {self.MODEL_CONFIDENCE_THRESHOLD:.2%})"
                    logger.info(f"❌ [{self.symbol}] RECHAZADA por {reason}")
                    
                    # 🔧 LOGGING DE RECHAZO
                    signal_data_for_log = base.copy()
                    signal_data_for_log.update({
                        "confidence": ml_confidence,
                        "historical_prob": historical_prob
                    })
                    
                    self.last_rejected_signal = {
                        'signal_data': signal_data_for_log,
                        'reason': reason,
                        'timestamp': datetime.now(timezone.utc)
                    }
                    
                    self._log_rejected_signal(signal_data_for_log, reason)
                    self._send_rejection_notification(side_str, reason)
                    self._update_perf(t0, approved=False, block_reason="ml_filter")
                    return None
                    
            except Exception as e:
                logger.warning(f"⚠️ Error validando con modelo ML: {e}")
                ml_confidence = 0.5

        # --- ANÁLISIS CAUSAL REAL CON LOGGING ---
        causal_factor = 1.0
        causal_reason = "Filtro causal deshabilitado"
        
        if self.filters_config.get("use_causal_filter", False):
            causal_factor, causal_reason = self._apply_causal_analysis(signal, base)
            if causal_factor == 0.0:  # Rechazo duro
                logger.info(f"❌ [{self.symbol}] RECHAZADA por {reason}")
                
                # 🔧 LOGGING DE RECHAZO CAUSAL
                signal_data_for_log = base.copy()
                signal_data_for_log.update({
                    "confidence": ml_confidence,
                    "historical_prob": historical_prob
                })
                
                self.last_rejected_signal = {
                    'signal_data': signal_data_for_log,
                    'reason': causal_reason,
                    'timestamp': datetime.now(timezone.utc)
                }
                
                self._log_rejected_signal(signal_data_for_log, causal_reason)
                self._send_rejection_notification(side_str, causal_reason)
                self._update_perf(t0, approved=False, block_reason="causal_filter")
                return None

        # --- ANÁLISIS DE CORRELACIONES ---
        correlation_factor = 1.0
        correlation_reason = "Filtro correlación deshabilitado"
        
        if self.filters_config.get("use_live_corr", False):
            correlation_factor, correlation_reason = self._apply_correlation_analysis(signal)

        # --- APLICAR FACTORES DE AJUSTE ---
        final_ml_confidence = ml_confidence * causal_factor * correlation_factor
        final_ml_confidence = max(0.0, min(1.0, final_ml_confidence))

        logger.info(f"✅ [{self.symbol}] SEÑAL CONFIRMADA | "
                   f"ML:{ml_confidence:.3f}→{final_ml_confidence:.3f} | "
                   f"Hist:{historical_prob:.3f} | "
                   f"Samples:{chroma_samples_found}")
        
        self._update_perf(t0, approved=True)

        return {
            "action": side_str,
            "entry_price": float(df["close"].iloc[-1]),
            "atr": float(base.get("atr", 0.0)),
            "confidence": float(final_ml_confidence),
            "historical_prob": float(historical_prob),
            "timestamp": str(df["timestamp"].iloc[-1]),
            "causal_factor": float(causal_factor),
            "correlation_factor": float(correlation_factor),
            "chroma_samples": chroma_samples_found
        }

    # 🔧 NUEVO: Método para obtener estadísticas de señales
    def get_signal_stats(self) -> Dict[str, Any]:
        """Devuelve estadísticas de señales del controller"""
        return {
            'symbol': self.symbol,
            'strategy': self.strategy_name,
            'stats': self.signal_stats.copy()
        }

    # -------------------------------------------------------------------------
    # Ejecución
    # -------------------------------------------------------------------------
    def execute_trade_with_size(self, signal_data: Dict[str, Any], lots: float) -> Optional[Dict[str, Any]]:
        # 🔒 VERIFICACIÓN FINAL antes de ejecutar
        #if self._has_open_position(self.symbol):
        #    logger.error(f"🔒 [{self.symbol}] EJECUCIÓN ABORTADA: Posición detectada antes de ejecutar orden")
        #    return {"error": f"Posición ya existe para {self.symbol}"}

        side = signal_data.get("action", "BUY").upper()
        price = float(signal_data.get("entry_price", 0.0))
        
        methods = [
            ("market_order", lambda: self.trading_client.market_order(self.symbol, side, lots, price=price, metadata=signal_data)),
            ("send_order", lambda: self.trading_client.send_order(self.symbol, side, lots, price=price, metadata=signal_data)),
            ("place_market_order", lambda: self.trading_client.place_market_order(self.symbol, side, lots)),
            ("order_send", lambda: self.trading_client.order_send(self.symbol, side, lots, price)),
        ]
        last_err = None
        if self.trading_client is None:
            return {"error": "TradingClient no disponible"}

        for name, fn in methods:
            if hasattr(self.trading_client, name):
                try:
                    res = fn()
                    if isinstance(res, dict) and res.get("ticket"):
                        logger.info(f"✅ [{self.symbol}] Orden ejecutada: Ticket {res['ticket']}")
                        return res
                    if res and isinstance(res, (int, str)):
                        logger.info(f"✅ [{self.symbol}] Orden ejecutada: Ticket {res}")
                        return {"ticket": res, "price": price}
                except Exception as e:
                    last_err = e
                    logger.debug(f"Método {name} falló: {e}")
                    continue

        logger.error(f"❌ [{self.symbol}] No se pudo enviar orden: {last_err}")
        return {"error": f"No se pudo enviar orden ({last_err})"}

    # -------------------------------------------------------------------------
    # Notificaciones y métricas
    # -------------------------------------------------------------------------
    def _send_rejection_notification(self, side: str, reason: str):
        if not self.notifier:
            return
        title = f"🚫 {self.symbol} {side} rechazado"
        body = f"Razón: {reason}\nEstrategia: {self.strategy_name}"
        try:
            if hasattr(self.notifier, "send"):
                self.notifier.send(title, body)
            elif hasattr(self.notifier, "send_notification"):
                self.notifier.send_notification(title, body)
        except Exception:
            pass

    def _update_perf(self, t0: float, approved: bool, block_reason: str = None):
        dt = (time.perf_counter() - t0) * 1000.0
        self.perf["last_signal_ms"] = dt
        n = self.perf["signals"] = self.perf["signals"] + 1
        prev = self.perf["avg_signal_ms"] or dt
        self.perf["avg_signal_ms"] = prev + (dt - prev) / max(1, n)
        
        if approved:
            status = "✅ Señal procesada"
        else:
            if block_reason == "existing_position":
                status = "🔒 Señal bloqueada (posición existe)"
            elif block_reason == "final_position_check":
                status = "🔒 Señal bloqueada (verificación final)"
            elif block_reason in ["chroma_filter", "ml_filter"]:
                status = "⛔ Señal rechazada (filtros IA)"
            elif block_reason == "causal_filter":
                status = "🧠 Señal rechazada (filtro causal)"
            else:
                status = "⛔ Señal rechazada"
        
        logger.debug(f"{status} en {dt/1000.0:.2f}s")