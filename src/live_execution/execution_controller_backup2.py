# src/live_execution/execution_controller.py
# -----------------------------------------------------------------------------
# OptimizedExecutionController
# Controlador de ejecución por símbolo/estrategia:
# - Genera señal base (TA) según estrategia
# - Confirma con Modelo ML + ChromaDB numérico (253 dims)
# - Verifica que NO haya posiciones abiertas antes de generar señales
# - Devuelve payload para PolicySwitcher y ejecuta orden con TradingClient
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import sys
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
import re

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
    LOOKBACK_BARS = 400
    MODEL_CONFIDENCE_THRESHOLD = 0.45
    HISTORICAL_PROB_THRESHOLD = 0.45

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

        self.models_dir = os.path.join("models", self.strategy_name)
        self.model = None
        self.scaler = None
        self.model_features: List[str] = []

        self.chroma_client = None
        self.collection = None
        self.chroma_path = os.environ.get("CHROMA_PATH", os.path.join("db", "chroma_db"))
        self.chroma_collection_name = "historical_market_states"
        self.chroma_n_results = 10

        self.perf = { "last_signal_ms": None, "avg_signal_ms": None, "signals": 0 }

        # Configuración de umbrales
        self.MODEL_CONFIDENCE_THRESHOLD = float(self.strategy_params.get(
            "model_conf_threshold", self.MODEL_CONFIDENCE_THRESHOLD
        ))
        self.HISTORICAL_PROB_THRESHOLD = float(self.strategy_params.get(
            "hist_prob_threshold", self.HISTORICAL_PROB_THRESHOLD
        ))

        # Control de posiciones - referencia al PolicySwitcher para fallback
        self.policy_switcher = None

        self._load_ml_artifacts()
        self._init_chroma()

        logger.info(f"✅ {self.symbol}_{self.strategy_name} listo. ML: {'OK' if self.model else 'NO'} | Chroma: {'OK' if self.collection else 'NO'}")

    def set_policy_switcher(self, policy_switcher):
        """Establece referencia al PolicySwitcher para verificación de posiciones"""
        self.policy_switcher = policy_switcher

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
        except Exception as e:
            logger.warning(f"⚠️ No se pudo inicializar ChromaDB: {e}")
            self.collection = None

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
            # Ejemplo básico de scalper con múltiples filtros
            ema_fast = ema(close, 8)
            ema_slow = ema(close, 21)
            rsi_val = rsi(close, 14).iloc[-1]
            
            # BUY: EMA rápida > EMA lenta y RSI < 70
            if (ema_fast.iloc[-1] > ema_slow.iloc[-1] and 
                ema_fast.iloc[-2] <= ema_slow.iloc[-2] and 
                rsi_val < 70):
                return {"action":"BUY","entry_price":entry_price,"atr":float(_atr)}
            
            # SELL: EMA rápida < EMA lenta y RSI > 30
            if (ema_fast.iloc[-1] < ema_slow.iloc[-1] and 
                ema_fast.iloc[-2] >= ema_slow.iloc[-2] and 
                rsi_val > 30):
                return {"action":"SELL","entry_price":entry_price,"atr":float(_atr)}
            return None

        if strat == "volatility_breakout":
            # Breakout basado en ATR
            atr_period = int(self.strategy_params.get("atr_period", 20))
            atr_multiplier = float(self.strategy_params.get("atr_multiplier", 2.0))
            lookback = int(self.strategy_params.get("lookback", 20))
            
            current_atr = atr(df, atr_period).iloc[-1]
            recent_high = df["high"].rolling(lookback).max().iloc[-2]  # Excluye vela actual
            recent_low = df["low"].rolling(lookback).min().iloc[-2]
            
            current_close = close.iloc[-1]
            prev_close = close.iloc[-2]
            
            # BUY: Precio rompe máximo reciente + ATR
            if current_close > recent_high + (current_atr * atr_multiplier):
                return {"action":"BUY","entry_price":entry_price,"atr":float(_atr)}
            
            # SELL: Precio rompe mínimo reciente - ATR  
            if current_close < recent_low - (current_atr * atr_multiplier):
                return {"action":"SELL","entry_price":entry_price,"atr":float(_atr)}
            return None

        if strat == "channel_reversal":
            # Reversal en canales de Donchian
            period = int(self.strategy_params.get("channel_period", 20))
            upper_channel, lower_channel = donchian_channels(df, period)
            rsi_val = rsi(close, 14).iloc[-1]
            
            current_close = close.iloc[-1]
            prev_close = close.iloc[-2]
            
            # BUY: Precio toca canal inferior y RSI < 30 (sobreventa)
            if (current_close <= lower_channel.iloc[-1] * 1.001 and  # Pequeña tolerancia
                rsi_val < 30):
                return {"action":"BUY","entry_price":entry_price,"atr":float(_atr)}
            
            # SELL: Precio toca canal superior y RSI > 70 (sobrecompra)
            if (current_close >= upper_channel.iloc[-1] * 0.999 and  # Pequeña tolerancia
                rsi_val > 70):
                return {"action":"SELL","entry_price":entry_price,"atr":float(_atr)}
            return None

        # Estrategia por defecto si no se reconoce
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
    # Señal principal CON VERIFICACIÓN DE POSICIONES
    # -------------------------------------------------------------------------
    def get_trading_signal(self) -> Optional[Dict[str, Any]]:
        t0 = time.perf_counter()

        # 🔒 VERIFICACIÓN DE POSICIONES ABIERTAS - PRIMERA LÍNEA DE DEFENSA
        if self._has_open_position(self.symbol):
            logger.info(f"🔒 [{self.symbol}] SEÑAL BLOQUEADA: Ya existe una posición abierta")
            self._update_perf(t0, approved=False, block_reason="existing_position")
            return None

        df = self._get_ohlc(self.LOOKBACK_BARS)
        if df is None or len(df) < 50:
            logger.warning(f"⚠️ No hay suficientes barras para {self.symbol}")
            return None
        logger.debug(f"Datos obtenidos para {self.symbol}: {len(df)} velas")

        base = self._compute_strategy_signal(df)
        if base is None:
            return None

        signal = 1 if base["action"].upper() == "BUY" else 0
        side_str = base["action"].upper()
        logger.info(f"🎯 [{self.symbol}] SEÑAL BASE: {side_str} (Estrategia: {self.strategy_name})")

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

        # --- CHROMA NUMÉRICO ---
        historical_prob = self.HISTORICAL_PROB_THRESHOLD
        if self.collection is not None and scaled_df is not None:
            try:
                results = self.collection.query(
                    query_embeddings=scaled_df.values.tolist(),
                    n_results=self.chroma_n_results,
                    where={"$and": [{"symbol": self.symbol}, {"strategy": self.strategy_name}]},
                    include=["metadatas"]
                )
                if results and results.get("ids") and results["ids"][0]:
                    metas = results.get("metadatas", [[]])[0]
                    outcomes = [m.get("outcome") for m in metas if isinstance(m, dict) and "outcome" in m]
                    if outcomes:
                        historical_prob = sum(1 for o in outcomes if o == signal) / len(outcomes)
                        if historical_prob < self.HISTORICAL_PROB_THRESHOLD:
                            reason = f"ChromaDB (prob: {historical_prob:.2%} < {self.HISTORICAL_PROB_THRESHOLD:.2%})"
                            logger.info(f"❌ [{self.symbol}] RECHAZADA por {reason}")
                            self._send_rejection_notification(side_str, reason)
                            self._update_perf(t0, approved=False, block_reason="chroma_filter")
                            return None
            except Exception as e:
                logger.warning(f"⚠️ Error validando con ChromaDB: {e}")

        # --- MODELO ML ---
        ml_confidence = 1.0
        if self.model is not None and scaled_df is not None:
            try:
                proba = self.model.predict_proba(scaled_df)[0]
                ml_confidence = float(proba[1] if signal == 1 else proba[0])
                if ml_confidence < self.MODEL_CONFIDENCE_THRESHOLD:
                    reason = f"Modelo IA (conf: {ml_confidence:.2%} < {self.MODEL_CONFIDENCE_THRESHOLD:.2%})"
                    logger.info(f"❌ [{self.symbol}] RECHAZADA por {reason}")
                    self._send_rejection_notification(side_str, reason)
                    self._update_perf(t0, approved=False, block_reason="ml_filter")
                    return None
            except Exception as e:
                logger.warning(f"⚠️ Error validando con modelo ML: {e}")

        # 🔒 VERIFICACIÓN FINAL DE POSICIONES (doble verificación antes de confirmar)
        if self._has_open_position(self.symbol):
            logger.warning(f"🔒 [{self.symbol}] SEÑAL RECHAZADA: Posición detectada en verificación final")
            self._update_perf(t0, approved=False, block_reason="final_position_check")
            return None

        logger.info(f"✅ [{self.symbol}] SEÑAL CONFIRMADA POR FILTROS IA")
        self._update_perf(t0, approved=True)

        return {
            "action": side_str,
            "entry_price": float(df["close"].iloc[-1]),
            "atr": float(base.get("atr", 0.0)),
            "confidence": float(ml_confidence),
            "historical_prob": float(historical_prob),
            "timestamp": str(df["timestamp"].iloc[-1]),
        }

    # -------------------------------------------------------------------------
    # Ejecución
    # -------------------------------------------------------------------------
    def execute_trade_with_size(self, signal_data: Dict[str, Any], lots: float) -> Optional[Dict[str, Any]]:
        # 🔒 VERIFICACIÓN FINAL antes de ejecutar
        if self._has_open_position(self.symbol):
            logger.error(f"🔒 [{self.symbol}] EJECUCIÓN ABORTADA: Posición detectada antes de ejecutar orden")
            return {"error": f"Posición ya existe para {self.symbol}"}

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
            else:
                status = "⛔ Señal rechazada"
        
        logger.debug(f"{status} en {dt/1000.0:.2f}s")