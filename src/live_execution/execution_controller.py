# src/live_execution/execution_controller.py
# =============================================================================
# OptimizedExecutionController
# - Señal base (TA) según estrategia
# - Confirmación con Modelo ML + ChromaDB (vector o solo metadatos)
# - Wilson LB (90%) con filtros anti-lookahead
# - Fallback escalonado en Chroma (Tier1: symbol+strategy → Tier2: symbol → Tier3: sin filtro)
# - Prevención de ejecución si hay posición abierta
# - Compatibilidad de firma con main_bot (df como primer posicional)
# =============================================================================

from __future__ import annotations

import os
import sys
import re
import math
import time
import json
import gzip
import pickle
import logging
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
from datetime import datetime, timezone

# ML
import joblib

# Chroma (opcional)
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False

# MT5 (opcional)
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except Exception:
    MT5_AVAILABLE = False


# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(_sh)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Fechas seguras (aware UTC)
# -----------------------------------------------------------------------------
def to_aware_utc(ts):
    """Convierte ts (str/int/float/datetime/pd.Timestamp) a datetime aware en UTC. Devuelve None si no puede."""
    if ts is None:
        return None

    if isinstance(ts, pd.Timestamp):
        try:
            if ts.tz is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return ts.to_pydatetime()
        except Exception:
            return None

    if isinstance(ts, datetime):
        try:
            return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    if isinstance(ts, (int, float)):
        try:
            sec = ts / 1000.0 if ts > 1e12 else ts
            return datetime.fromtimestamp(sec, tz=timezone.utc)
        except Exception:
            return None

    if isinstance(ts, str):
        s = ts.strip()
        try:
            if s.endswith("Z"):
                s = s.replace("Z", "+00:00")
            t = pd.to_datetime(s, utc=True, errors="coerce")
            if pd.isna(t):
                return None
            return t.to_pydatetime()
        except Exception:
            return None

    return None


def is_future_or_equal(mt, ref):
    """True si mt >= ref tras normalizar (aware UTC). Si falla, False."""
    try:
        mt2 = to_aware_utc(mt)
        ref2 = to_aware_utc(ref)
        if mt2 is None or ref2 is None:
            return False
        if getattr(mt2, "tzinfo", None) is None:
            mt2 = mt2.replace(tzinfo=timezone.utc)
        if getattr(ref2, "tzinfo", None) is None:
            ref2 = ref2.replace(tzinfo=timezone.utc)
        return mt2 >= ref2
    except Exception:
        return False


# -----------------------------------------------------------------------------
# TA helpers
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
# Wilson LB (90%)
# -----------------------------------------------------------------------------
def wilson_lb90(wins: int, n: int) -> float:
    if n <= 0:
        return 0.0
    z = 1.6448536269514722  # 90%
    p = wins / n
    denom = 1 + (z*z)/n
    center = p + (z*z)/(2*n)
    margin = z * math.sqrt((p*(1-p) + (z*z)/(4*n)) / n)
    return max(0.0, min(1.0, (center - margin) / denom))


# -----------------------------------------------------------------------------
# Controlador
# -----------------------------------------------------------------------------
class OptimizedExecutionController:
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

        # Config general
        self.orchestrator_config = self._load_orchestrator_config()
        self.chroma_config = self.orchestrator_config.get("chroma", {})
        self.thresholds = self.orchestrator_config.get("thresholds", {})
        self.filters_config = self.orchestrator_config.get("filters", {})

        # Umbrales
        self.MODEL_CONFIDENCE_THRESHOLD = float(self.thresholds.get("ml_confidence_min", 0.45))
        self.HISTORICAL_PROB_THRESHOLD = float(self.thresholds.get("historical_prob_min", 0.45))

        # Causal/Correlación
        self.causal_insights = self._load_causal_insights()
        self.correlation_matrix = self._load_correlation_matrix()

        # ML
        self.models_dir = os.path.join("models", self.strategy_name)
        self.model = None
        self.scaler = None
        self.model_features: List[str] = []
        self._load_ml_artifacts()

        # Chroma (con override por ENV)
        self.chroma_client = None
        self.collection = None
        # override por variables de entorno si existen
        env_dir = os.getenv("CHROMA_DIR")
        env_col = os.getenv("CHROMA_COLLECTION")
        self.chroma_path = env_dir or self.chroma_config.get("path", "data/chroma")
        self.chroma_collection_name = env_col or self.chroma_config.get("collection", "forex_v1")
        self.chroma_n_results = int(self.chroma_config.get("n_results", 9))
        self._init_chroma()

        # PolicySwitcher (inyectado desde fuera)
        self.policy_switcher = None

        # Logging y stats
        self._external_log_function = None
        self.perf = {"last_signal_ms": None, "avg_signal_ms": None, "signals": 0}
        self.last_rejected_signal = None
        self.last_signal_attempt = None
        self.signal_stats = {
            'total_attempts': 0,
            'signals_generated': 0,
            'signals_rejected': 0,
            'rejection_reasons': {}
        }

        logger.info(
            f"✅ {self.symbol}_{self.strategy_name} listo. "
            f"ML: {'OK' if self.model else 'NO'} | "
            f"Chroma: {'OK' if self.collection else 'NO'} [{self.chroma_path}/{self.chroma_collection_name}] | "
            f"Causal: {'OK' if self.causal_insights else 'NO'} | "
            f"Umbrales: ML={self.MODEL_CONFIDENCE_THRESHOLD} Hist={self.HISTORICAL_PROB_THRESHOLD}"
        )

    # -------------------------------------------------------------------------
    # Config loaders
    # -------------------------------------------------------------------------
    def _load_orchestrator_config(self) -> Dict[str, Any]:
        path = "orchestrator_config.json"
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar {path}: {e}. Usando defaults.")
            return {
                "thresholds": {"ml_confidence_min": 0.45, "historical_prob_min": 0.45},
                "chroma": {"path": "data/chroma", "collection": "forex_v1", "n_results": 9},
                "filters": {"use_causal_filter": False, "use_live_corr": False}
            }

    def _load_causal_insights(self) -> Optional[Dict[str, Any]]:
        if not self.filters_config.get("use_causal_filter", False):
            return None
        try:
            causal_path = os.path.join(
                self.orchestrator_config.get("paths", {}).get("causal_reports_dir", "reports/causal_reports"),
                "causal_insights.json"
            )
            with open(causal_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for insight in data:
                if insight.get("asset") == self.symbol and insight.get("strategy") == self.strategy_name:
                    pf = insight.get("profit_factor", 0.0)
                    ate = insight.get("ATE", 0.0)
                    sig = insight.get("statistical_significance", {}).get("significance", "N/A")
                    logger.info(f"✅ Causal para {self.symbol}_{self.strategy_name}: PF={pf:.3f} ATE={ate:.1f} Sig={sig}")
                    return insight
            logger.warning(f"⚠️ Sin causal para {self.symbol}_{self.strategy_name}")
            return None
        except Exception as e:
            logger.warning(f"⚠️ Error cargando causal_insights: {e}")
            return None

    def _load_correlation_matrix(self) -> Optional[Dict[str, Any]]:
        if not self.filters_config.get("use_live_corr", False):
            return None
        try:
            import glob
            corr_dir = self.orchestrator_config.get("paths", {}).get("cross_correlation_dir", "reports/cross_correlation_reports")
            files = glob.glob(os.path.join(corr_dir, "*.csv"))
            if not files:
                logger.warning(f"⚠️ No hay archivos de correlación en {corr_dir}")
                return None
            latest = max(files, key=os.path.getctime)
            df = pd.read_csv(latest, index_col=0)
            logger.info(f"✅ Matriz de correlaciones cargada: {latest}")
            return df.to_dict()
        except Exception as e:
            logger.warning(f"⚠️ Error cargando correlaciones: {e}")
            return None

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
            self.collection = self.chroma_client.get_or_create_collection(name=self.chroma_collection_name)
            logger.info(f"✅ ChromaDB inicializado: {self.chroma_path}/{self.chroma_collection_name}")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo inicializar ChromaDB: {e}")
            self.collection = None

    def _make_where(self) -> Dict[str, Any]:
        """Filtro WHERE por symbol y timeframe si está disponible."""
        tf = getattr(self, "timeframe", None)
        if tf:
            return {"$and": [{"symbol": {"$eq": self.symbol}}, {"timeframe": {"$eq": tf}}]}
        return {"symbol": {"$eq": self.symbol}}

    @staticmethod
    def _norm_symbol(s: str) -> str:
        return (s or "").strip().upper()

    @staticmethod
    def _norm_strategy(s: str) -> str:
        return (s or "").strip().lower()

    def _query_chroma_with_fallback(self, query_embedding: Optional[np.ndarray], n: int = 9):
        """
        Fallback escalonado:
        Tier1: where={"symbol":SYM, "strategy":STRAT}
        Tier2: where={"symbol":SYM}
        Tier3: where={}
        Retorna dict con (metas, distancias) o None.
        """
        if self.collection is None:
            return None

        sym = self._norm_symbol(self.symbol)
        strat = self._norm_strategy(self.strategy_name)

        tiers = [
            {"symbol": sym, "strategy": strat},
            {"symbol": sym},
            {},
        ]

        for i, where in enumerate(tiers, 1):
            try:
                if query_embedding is not None:
                    res = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n,
                        where=where,
                        include=["metadatas", "distances"],
                    )
                else:
                    # Fallback sin embedding: obtener por metadatos y truncar a n
                    got = self.collection.get(where=where)
                    ids = (got.get("ids") or [])
                    if not ids:
                        continue
                    metas = (got.get("metadatas") or [])[:n]
                    # 👇 FIJATE AQUÍ: se corrige el corchete extra
                    res = {
                        "metadatas": [metas],
                        "distances": [[0.0] * len(metas)]
                    }

                metas = (res.get("metadatas") or [[]])[0]
                if not metas:
                    continue
                dists = (res.get("distances") or [[]])[0]
                logger.debug(f"Chroma tier{i} ok: {len(metas)} candidatas")
                return {"tier": i, "metas": metas, "distances": dists}
            except Exception as e:
                logger.warning(f"Chroma tier{i} falló: {e}")
                continue

        logger.info(f"ℹ️ [{sym}] ChromaDB: Sin resultados para la consulta. (colección='{self.chroma_collection_name}')")
        return None


    # -------------------------------------------------------------------------
    # Posiciones abiertas
    # -------------------------------------------------------------------------
    def set_policy_switcher(self, policy_switcher):
        self.policy_switcher = policy_switcher

    def set_external_log_function(self, log_function):
        self._external_log_function = log_function

    def _get_mt5_positions(self, symbol: str) -> List[Dict[str, Any]]:
        if not MT5_AVAILABLE:
            return []
        try:
            positions = mt5.positions_get(symbol=symbol)
            if not positions:
                return []
            out = []
            for p in positions:
                out.append({
                    "ticket": int(p.ticket),
                    "symbol": p.symbol,
                    "type": p.type,  # 0=BUY, 1=SELL
                    "volume": float(p.volume),
                    "price_open": float(p.price_open),
                    "profit": float(p.profit),
                    "comment": p.comment,
                })
            return out
        except Exception as e:
            logger.debug(f"MT5 positions_get falló: {e}")
            return []

    def _get_trading_client_positions(self, symbol: str) -> List[Dict[str, Any]]:
        if self.trading_client is None:
            return []
        methods = ["get_positions", "positions_get", "get_open_positions", "open_positions", "positions"]
        for m in methods:
            if hasattr(self.trading_client, m):
                try:
                    fn = getattr(self.trading_client, m)
                    try:
                        pos = fn(symbol=symbol)
                    except TypeError:
                        all_pos = fn()
                        pos = [p for p in (all_pos or []) if isinstance(p, dict) and p.get("symbol") == symbol]
                    if pos:
                        return pos if isinstance(pos, list) else [pos]
                except Exception:
                    continue
        return []

    def _has_open_position(self, symbol: str) -> bool:
        if self._get_mt5_positions(symbol):
            logger.info(f"🔒 [{symbol}] Posición detectada en MT5")
            return True
        if self._get_trading_client_positions(symbol):
            logger.info(f"🔒 [{symbol}] Posición detectada en TradingClient")
            return True
        if self.policy_switcher and hasattr(self.policy_switcher, "open_positions"):
            try:
                vals = self.policy_switcher.open_positions.values()
                if any(isinstance(p, dict) and p.get("symbol") == symbol for p in vals):
                    logger.info(f"🔒 [{symbol}] Posición detectada en PolicySwitcher")
                    return True
            except Exception:
                pass
        logger.debug(f"✅ [{symbol}] Sin posiciones abiertas")
        return False

    # -------------------------------------------------------------------------
    # ML artifacts
    # -------------------------------------------------------------------------
    def _candidate_paths(self, base_dir: str, symbol: str):
        return (
            [
                os.path.join(base_dir, f"{symbol}_confirmation_model.pkl"),
                os.path.join(base_dir, f"{symbol}_confirmation_model.joblib"),
                os.path.join(base_dir, f"{symbol}_model.pkl"),
                os.path.join(base_dir, f"{symbol}_model.joblib"),
                os.path.join(base_dir, "confirmation_model.pkl"),
                os.path.join(base_dir, "confirmation_model.joblib"),
            ],
            [
                os.path.join(base_dir, f"{symbol}_feature_scaler.pkl"),
                os.path.join(base_dir, f"{symbol}_feature_scaler.joblib"),
                os.path.join(base_dir, "feature_scaler.pkl"),
                os.path.join(base_dir, "feature_scaler.joblib"),
            ],
            [
                os.path.join(base_dir, f"{symbol}_model_features.joblib"),
                os.path.join(base_dir, "model_features.joblib"),
            ],
        )

    def _smart_load(self, path: str):
        try:
            return joblib.load(path)
        except Exception:
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                try:
                    with gzip.open(path, "rb") as f:
                        return pickle.load(f)
                except Exception:
                    return None

    def _load_ml_artifacts(self):
        try:
            model_cands, scaler_cands, feats_cands = self._candidate_paths(self.models_dir, self.symbol)

            # features
            for p in feats_cands:
                if os.path.exists(p):
                    obj = joblib.load(p)
                    if isinstance(obj, list) and all(isinstance(c, str) for c in obj):
                        self.model_features = obj
                        break
            if not self.model_features:
                logger.warning(f"⚠️ Features no encontradas para {self.symbol} en {self.models_dir}")

            # scaler
            for p in scaler_cands:
                if os.path.exists(p):
                    self.scaler = self._smart_load(p)
                    if self.scaler is not None:
                        break
            if self.scaler is None:
                logger.warning(f"⚠️ Scaler no encontrado para {self.symbol} en {self.models_dir}")

            # model
            for p in model_cands:
                if os.path.exists(p):
                    self.model = self._smart_load(p)
                    if self.model is not None:
                        break
            if self.model is None:
                logger.warning(f"⚠️ Modelo no encontrado para {self.symbol} en {self.models_dir}")

            if self.model and (self.scaler is None or not self.model_features):
                logger.warning(f"⚠️ Modelo cargado pero faltan scaler/features ({self.symbol})")
        except Exception as e:
            logger.warning(f"⚠️ Error cargando artefactos ML ({self.symbol}/{self.strategy_name}): {e}")

    # -------------------------------------------------------------------------
    # OHLC helpers
    # -------------------------------------------------------------------------
    def _map_timeframe_to_mt5(self, tf: str) -> Optional[int]:
        tf = (tf or "M5").upper()
        return getattr(mt5, f"TIMEFRAME_{tf}", None)

    def _normalize_ohlc_df(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_map = {
            "time": "timestamp", "Time": "timestamp",
            "open": "open", "Open": "open",
            "high": "high", "High": "high",
            "low": "low", "Low": "low",
            "close": "close", "Close": "close",
            "volume": "volume", "Volume": "volume",
            "tick_volume": "volume", "real_volume": "volume"
        }
        _df = df.rename(columns={k: v for k, v in cols_map.items() if k in df.columns}).copy()

        if "timestamp" not in _df.columns and isinstance(_df.index, pd.DatetimeIndex):
            _df["timestamp"] = _df.index

        if "timestamp" in _df.columns:
            _df["timestamp"] = pd.to_datetime(_df["timestamp"], unit='s', errors='coerce', utc=True)
            _df = _df.dropna(subset=["timestamp"])
        else:
            return pd.DataFrame()

        req = ["timestamp", "open", "high", "low", "close"]
        if any(col not in _df.columns for col in req):
            return pd.DataFrame()

        if "volume" not in _df.columns:
            _df["volume"] = 0

        _df = _df[req + ["volume"]].sort_values("timestamp").reset_index(drop=True)
        return _df.tail(self.LOOKBACK_BARS)

    def _get_ohlc(self, bars: int) -> Optional[pd.DataFrame]:
        # TradingClient methods
        if self.trading_client is not None:
            try_methods = [
                ("get_ohlc", lambda: self.trading_client.get_ohlc(self.symbol, self.timeframe, bars)),
                ("get_rates", lambda: self.trading_client.get_rates(self.symbol, self.timeframe, bars)),
                ("get_candles", lambda: self.trading_client.get_candles(self.symbol, self.timeframe, bars)),
                ("copy_rates", lambda: self.trading_client.copy_rates(self.symbol, self.timeframe, bars)),
            ]
            for name, fn in try_methods:
                if hasattr(self.trading_client, name):
                    try:
                        df = fn()
                        if df is not None and len(df) > 0:
                            return self._normalize_ohlc_df(df)
                    except Exception:
                        continue

        # MT5 fallback
        if MT5_AVAILABLE:
            try:
                mt5_tf = self._map_timeframe_to_mt5(self.timeframe)
                rates = mt5.copy_rates_from_pos(self.symbol, mt5_tf, 0, bars)
                if rates is not None:
                    df = pd.DataFrame(rates)
                    return self._normalize_ohlc_df(df)
            except Exception:
                pass

        return None

    # -------------------------------------------------------------------------
    # Señal por estrategia (TA)
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
            upper, lower = donchian_channels(df, period)
            rsi_val = rsi(close, 14).iloc[-1]
            current_close = close.iloc[-1]
            if (current_close <= lower.iloc[-1] * 1.001 and rsi_val < 30):
                return {"action":"BUY","entry_price":entry_price,"atr":float(_atr)}
            if (current_close >= upper.iloc[-1] * 0.999 and rsi_val > 70):
                return {"action":"SELL","entry_price":entry_price,"atr":float(_atr)}
            return None

        # default = ema crossover
        fast, slow = 12, 26
        efast = ema(close, fast); eslow = ema(close, slow)
        if efast.iloc[-2] <= eslow.iloc[-2] and efast.iloc[-1] > eslow.iloc[-1]:
            return {"action":"BUY","entry_price":entry_price,"atr":float(_atr)}
        if efast.iloc[-2] >= eslow.iloc[-2] and efast.iloc[-1] < eslow.iloc[-1]:
            return {"action":"SELL","entry_price":entry_price,"atr":float(_atr)}
        return None

    # -------------------------------------------------------------------------
    # Features para ML
    # -------------------------------------------------------------------------
    def _prepare_live_feature_row(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if not self.model_features:
            return None

        feats: Dict[str, float] = {}
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        close = df["close"]

        feats["close"] = float(last["close"])
        feats["open"] = float(last["open"])
        feats["high"] = float(last["high"])
        feats["low"] = float(last["low"])
        feats["volume"] = float(last.get("volume", 0.0))
        feats["candle_ret"] = (last["close"] / prev["close"] - 1.0) if prev["close"] else 0.0

        for feature_name in self.model_features:
            if feature_name in feats:
                continue
            try:
                m = re.match(r"([a-zA-Z_]+)_?(\d+)?", feature_name)
                if not m:
                    continue
                indicator, per = m.groups()
                period = int(per) if per else None

                if indicator == "ema" and period:
                    feats[feature_name] = float(ema(close, period).iloc[-1])
                elif indicator == "sma" and period:
                    feats[feature_name] = float(sma(close, period).iloc[-1])
                elif indicator == "rsi" and period:
                    feats[feature_name] = float(rsi(close, period).iloc[-1])
                elif indicator == "atr" and period:
                    feats[feature_name] = float(atr(df, period).iloc[-1])
            except Exception:
                feats[feature_name] = 0.0

        final_row = {col: float(feats.get(col, 0.0)) for col in self.model_features}
        return pd.DataFrame([final_row], columns=self.model_features)

    # -------------------------------------------------------------------------
    # Causal y Correlación
    # -------------------------------------------------------------------------
    def _apply_causal_analysis(self, signal: int, base: Dict[str, Any]) -> Tuple[float, str]:
        if not self.causal_insights:
            return 1.0, "Sin datos causales"
        try:
            pf = float(self.causal_insights.get("profit_factor", 1.0))
            ate = float(self.causal_insights.get("ATE", 0.0))
            significance = self.causal_insights.get("statistical_significance", {}).get("significance", "not_significant")
            best_regime = self.causal_insights.get("best_volatility_regime_causal", "unknown")

            causal_cfg = self.filters_config.get("causal", {})
            min_pf_hard = float(causal_cfg.get("min_pf_hard", 0.95))
            min_ate_threshold = float(causal_cfg.get("min_ate_threshold", -1.0))
            pf_penalty_factor = float(causal_cfg.get("pf_penalty_factor", 0.9))
            enforce_regime = bool(causal_cfg.get("enforce_regime", False))

            if (pf < min_pf_hard) and (ate < min_ate_threshold) and (significance in ("significant", "highly_significant")):
                return 0.0, f"PF duro ({pf:.3f} < {min_pf_hard}) y ATE {ate:.1f} con {significance}"

            sig_mult = {"highly_significant": 1.1, "significant": 1.05, "not_significant": 0.97}.get(significance, 1.0)
            if pf >= 1.2:
                pf_mult = 1.12
            elif pf >= 1.1:
                pf_mult = 1.06
            elif pf < 1.0:
                pf_mult = pf_penalty_factor
            else:
                pf_mult = 1.0

            ate_penalty = 1.0 if ate >= 0 else max(0.85, 1.0 + (ate / 100.0))

            regime_mult = 1.0
            if enforce_regime:
                current_atr = base.get("atr", 0.0)
                current_regime = "high_vol" if current_atr > 0.001 else "low_vol"
                if best_regime != "unknown" and current_regime != best_regime:
                    regime_mult = 0.92
                    logger.info(f"⚠️ [{self.symbol}] Régimen subóptimo: actual={current_regime}, óptimo={best_regime}")

            final = sig_mult * pf_mult * regime_mult * ate_penalty
            reason = (f"PF={pf:.3f} ATE={ate:.1f} Sig={significance} Régimen={best_regime} → Factor={final:.3f}")
            logger.info(f"🧠 [{self.symbol}] Causal OK: {reason}")
            return final, reason
        except Exception as e:
            logger.warning(f"⚠️ Error causal: {e}")
            return 1.0, f"Error causal: {e}"

    def _apply_correlation_analysis(self, signal: int) -> Tuple[float, str]:
        if not self.correlation_matrix:
            return 1.0, "Sin correlación"
        try:
            symbol_corrs = self.correlation_matrix.get(self.symbol, {})
            high = []
            cap = self.orchestrator_config.get("correlation_caps", {}).get("cap_if_rho_gt", 0.85)
            for other, rho in symbol_corrs.items():
                if other != self.symbol and abs(rho) > cap:
                    high.append((other, rho))
            if high:
                active = sum(1 for (other, _) in high if self._has_open_position(other))
                if active > 0:
                    logger.info(f"⚠️ [{self.symbol}] Alta correlación con {active} posiciones activas")
                    return 0.7, "Correlación alta con posiciones abiertas"
            return 1.0, "Sin conflictos"
        except Exception as e:
            logger.warning(f"⚠️ Error correlación: {e}")
            return 1.0, f"Error correlación: {e}"

    # -------------------------------------------------------------------------
    # Logging de rechazos (fallback CSV si no hay función externa)
    # -------------------------------------------------------------------------
    def _log_rejected_signal(self, signal_data: Dict[str, Any], reason: str):
        try:
            payload = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "symbol": self.symbol,
                "strategy": self.strategy_name,
                "side": signal_data.get("action", ""),
                "entry_price": signal_data.get("entry_price", ""),
                "atr": signal_data.get("atr", 0.0),
                "ml_confidence": signal_data.get("confidence", 0.0),
                "historical_prob": signal_data.get("historical_prob", 0.0),
                "status": "rejected_by_controller",
                "rejection_reason": reason,
                "position_size": 0,
                "ticket": "",
            }
            if self._external_log_function:
                ok = self._external_log_function(payload)
                if ok:
                    return
            # Fallback CSV
            import csv
            log_path = "logs/signals_history.csv"
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            write_header = not os.path.exists(log_path)
            fields = ["timestamp_utc","symbol","strategy","side","entry_price","atr",
                      "ml_confidence","historical_prob","status","rejection_reason",
                      "position_size","ticket"]
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                if write_header:
                    w.writeheader()
                w.writerow({k: payload.get(k, "") for k in fields})
        except Exception as e:
            logger.error(f"❌ Error logging rechazo: {e}")

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
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.perf["last_signal_ms"] = dt_ms
        n = self.perf["signals"] = self.perf["signals"] + 1
        prev = self.perf["avg_signal_ms"] or dt_ms
        self.perf["avg_signal_ms"] = prev + (dt_ms - prev) / max(1, n)
        # log de debug opcional

    # -------------------------------------------------------------------------
    # API pública de señales
    # -------------------------------------------------------------------------
    def get_trading_signal(self) -> Optional[Dict[str, Any]]:
        res = self.get_trading_signal_with_details()
        if res and res.get("status") == "generated":
            return res["signal"]
        return None

    def get_trading_signal_with_details(
        self,
        *args,
        forced_side: Optional[str] = None,
        extra_context: Optional[Dict[str, Any]] = None,
        candles: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Compatible con:
          - get_trading_signal_with_details(df, extra_context=...)
          - get_trading_signal_with_details(symbol=None, candles=df, extra_context=...)
        Devuelve dict con:
          { 'signal': {...}, 'rejection_reason': str|None, 'status': 'generated'|'rejected' }
        """
        self.signal_stats['total_attempts'] += 1
        self.last_rejected_signal = None
        self.last_signal_attempt = None

        # Compat: primer posicional puede ser DataFrame (tu main_bot actual)
        if args:
            if isinstance(args[0], pd.DataFrame):
                candles = args[0]
            elif isinstance(args[0], str):
                # símbolo posicional (no lo usamos, trabajamos con self.symbol)
                pass

        # Señal forzada (testing/manual)
        if forced_side in ("BUY", "SELL"):
            base_signal = {
                "action": forced_side,
                "entry_price": float(candles['close'].iloc[-1]) if candles is not None else 0.0,
                "atr": float(atr(candles, 14).iloc[-1]) if candles is not None else 0.0,
                "confidence": 1.0,
                "historical_prob": 0.5,
            }
            if extra_context: base_signal.update(extra_context)
            self.signal_stats['signals_generated'] += 1
            return {"signal": base_signal, "rejection_reason": None, "status": "generated"}

        # Señal normal
        signal_data = self._get_trading_signal_with_tracking(df_override=candles)

        if signal_data and not signal_data.get("_rejected"):
            if extra_context:
                signal_data.update(extra_context)
            self.signal_stats['signals_generated'] += 1
            return {"signal": signal_data, "rejection_reason": None, "status": "generated"}

        if self.last_rejected_signal:
            if extra_context:
                self.last_rejected_signal["signal_data"].update(extra_context)
            self.signal_stats["signals_rejected"] += 1
            reason = self.last_rejected_signal["reason"]
            # contar razón (clave corta)
            key = reason.split(" ")[0].split("(")[0]
            self.signal_stats["rejection_reasons"][key] = self.signal_stats["rejection_reasons"].get(key, 0) + 1
            return {
                "signal": self.last_rejected_signal["signal_data"],
                "rejection_reason": reason,
                "status": "rejected",
            }

        return None

    # -------------------------------------------------------------------------
    # Núcleo con tracking
    # -------------------------------------------------------------------------
    def _get_trading_signal_with_tracking(self, df_override: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        t0 = time.perf_counter()

        # 1) Datos
        df = df_override if (df_override is not None and not df_override.empty) else self._get_ohlc(self.LOOKBACK_BARS)
        if df is None or len(df) < 50:
            logger.warning(f"⚠️ No hay suficientes barras para {self.symbol}")
            return None
        # 2) Señal base
        base = self._compute_strategy_signal(df)
        if base is None:
            return None

        signal = 1 if base["action"].upper() == "BUY" else 0
        side_str = base["action"].upper()
        logger.info(f"🎯 [{self.symbol}] SEÑAL BASE: {side_str} (Estrategia: {self.strategy_name})")

        self.last_signal_attempt = {"signal_data": base.copy(), "timestamp": datetime.now(timezone.utc)}

        # 3) Features → scaler/model
        scaled_df = None
        if self.model is not None and self.scaler is not None and self.model_features:
            live_row = self._prepare_live_feature_row(df)
            if live_row is not None:
                try:
                    scaled_df = pd.DataFrame(self.scaler.transform(live_row), columns=self.model_features)
                except Exception as e:
                    logger.error(f"Error escalando features para {self.symbol}: {e}")
                    return None

        # 4) CHROMA + Wilson LB90
        historical_prob = 0.5
        historical_prob_lb90 = 0.5
        chroma_samples_found = 0
        if self.collection is not None:
            try:
                # Usamos embedding solo si tenemos scaled_df de 1 fila y dimensión consistente
                query_emb = None
                if scaled_df is not None and getattr(scaled_df, "shape", (0,))[0] >= 1:
                    query_emb = scaled_df.values[0].tolist()

                fetched = self._query_chroma_with_fallback(query_embedding=query_emb, n=max(9, self.chroma_n_results))
                if fetched:
                    metas = fetched["metas"]
                    dists = fetched["distances"] or [None] * len(metas)

                    # Tiempo de la señal para evitar lookahead
                    signal_time = base.get("timestamp") or base.get("time") or base.get("ts") or df["timestamp"].iloc[-1]
                    signal_time = to_aware_utc(signal_time) or datetime.now(timezone.utc)

                    valid = []
                    seen = set()
                    for m, dist in zip(metas, dists):
                        if not isinstance(m, dict):
                            continue
                        out = m.get("outcome")
                        try:
                            out_i = int(out)
                        except Exception:
                            continue
                        if out_i not in (-1, 0, 1):
                            continue
                        mt = m.get("timestamp") or m.get("time") or m.get("ts")
                        if mt is None or is_future_or_equal(mt, signal_time):
                            continue
                        key = (m.get("timestamp"), m.get("entry_price"), m.get("trade_id"))
                        if key in seen:
                            continue
                        seen.add(key)
                        m["_outcome"] = out_i
                        m["_distance"] = float(dist) if dist is not None else None
                        valid.append(m)

                    # Excluir 0 (break-even) del denominador para una lectura más clara
                    valid = [m for m in valid if m["_outcome"] in (-1, 1)]
                    chroma_samples_found = len(valid)

                    if chroma_samples_found > 0:
                        is_buy = (signal == 1)
                        wins = sum(1 for m in valid if (m["_outcome"] > 0 and is_buy) or (m["_outcome"] < 0 and not is_buy))
                        p_hat = wins / chroma_samples_found
                        p_lo = wilson_lb90(wins, chroma_samples_found)
                        historical_prob = max(0.001, min(0.999, p_hat))
                        historical_prob_lb90 = p_lo

                        dvals = [m.get("_distance") for m in valid if m.get("_distance") is not None]
                        if dvals:
                            avg_d, min_d, max_d = float(np.mean(dvals)), float(np.min(dvals)), float(np.max(dvals))
                            logger.info(
                                f"📊 [{self.symbol}] Chroma: {wins}/{chroma_samples_found} éxitos | "
                                f"p̂={p_hat:.2%}, LB90={p_lo:.2%} ({side_str}) | "
                                f"dist avg={avg_d:.1f} min={min_d:.1f} max={max_d:.1f}"
                            )
                        else:
                            logger.info(
                                f"📊 [{self.symbol}] Chroma: {wins}/{chroma_samples_found} éxitos | "
                                f"p̂={p_hat:.2%}, LB90={p_lo:.2%} ({side_str})"
                            )

                        # Rechazo conservador por LB90
                        min_samples = max(7, self.chroma_n_results // 3)
                        if chroma_samples_found >= min_samples and p_lo < self.HISTORICAL_PROB_THRESHOLD:
                            reason = (f"Prob. histórica conservadora baja (LB90={p_lo:.2%} "
                                      f"< {self.HISTORICAL_PROB_THRESHOLD:.2%}) con {chroma_samples_found} casos.")
                            logger.info(f"❌ [{self.symbol}] RECHAZADA por {reason}")
                            signal_log = base.copy()
                            signal_log.update({
                                "confidence": 0.0,
                                "historical_prob": historical_prob,
                                "historical_prob_lb90": historical_prob_lb90,
                                "chroma_samples": chroma_samples_found,
                                "n_eff": chroma_samples_found,
                                "min_samples": min_samples,
                            })
                            self.last_rejected_signal = {
                                "signal_data": signal_log,
                                "reason": reason,
                                "timestamp": datetime.now(timezone.utc),
                            }
                            self._log_rejected_signal(signal_log, reason)
                            self._send_rejection_notification(side_str, reason)
                            self._update_perf(t0, approved=False, block_reason="chroma_filter")
                            return None
                    else:
                        logger.info(f"ℹ️ [{self.symbol}] Chroma: sin casos válidos (±1) antes de la señal.")
                else:
                    logger.info(f"ℹ️ [{self.symbol}] ChromaDB: Sin resultados para la consulta.")
            except Exception as e:
                logger.warning(f"⚠️ Error validando con ChromaDB: {e}")

        # 5) Modelo ML
        ml_confidence = 0.5
        if self.model is not None and scaled_df is not None:
            try:
                proba = self.model.predict_proba(scaled_df)[0]
                ml_confidence = float(proba[1] if signal == 1 else proba[0])
                if ml_confidence < self.MODEL_CONFIDENCE_THRESHOLD:
                    reason = f"Modelo IA (conf: {ml_confidence:.2%} < {self.MODEL_CONFIDENCE_THRESHOLD:.2%})"
                    logger.info(f"❌ [{self.symbol}] RECHAZADA por {reason}")
                    signal_log = base.copy()
                    signal_log.update({"confidence": ml_confidence, "historical_prob": historical_prob})
                    self.last_rejected_signal = {
                        "signal_data": signal_log,
                        "reason": reason,
                        "timestamp": datetime.now(timezone.utc),
                    }
                    self._log_rejected_signal(signal_log, reason)
                    self._send_rejection_notification(side_str, reason)
                    self._update_perf(t0, approved=False, block_reason="ml_filter")
                    return None
            except Exception as e:
                logger.warning(f"⚠️ Error validando con modelo ML: {e}")
                ml_confidence = 0.5

        # 6) Causal
        causal_factor, _ = (1.0, "Sin causal")
        if self.filters_config.get("use_causal_filter", False):
            causal_factor, causal_reason = self._apply_causal_analysis(signal, base)
            if causal_factor == 0.0:
                logger.info(f"❌ [{self.symbol}] RECHAZADA por {causal_reason}")
                signal_log = base.copy()
                signal_log.update({"confidence": ml_confidence, "historical_prob": historical_prob})
                self.last_rejected_signal = {
                    "signal_data": signal_log,
                    "reason": causal_reason,
                    "timestamp": datetime.now(timezone.utc),
                }
                self._log_rejected_signal(signal_log, causal_reason)
                self._send_rejection_notification(side_str, causal_reason)
                self._update_perf(t0, approved=False, block_reason="causal_filter")
                return None

        # 7) Correlación
        corr_factor, _ = (1.0, "Sin correlación")
        if self.filters_config.get("use_live_corr", False):
            corr_factor, corr_reason = self._apply_correlation_analysis(signal)

        # 8) Final
        final_ml_confidence = max(0.0, min(1.0, ml_confidence * causal_factor * corr_factor))

        logger.info(
            f"✅ [{self.symbol}] SEÑAL CONFIRMADA | ML:{ml_confidence:.3f}→{final_ml_confidence:.3f} | "
            f"Hist:{historical_prob:.3f} (LB90:{historical_prob_lb90:.3f}) | Samples:{chroma_samples_found}"
        )
        self._update_perf(t0, approved=True)

        return {
            "action": side_str,
            "entry_price": float(df["close"].iloc[-1]),
            "atr": float(base.get("atr", 0.0)),
            "confidence": float(final_ml_confidence),
            "historical_prob": float(historical_prob),
            "historical_prob_lb90": float(historical_prob_lb90),
            "chroma_samples": int(chroma_samples_found),
            "n_eff": int(chroma_samples_found),
            "timestamp": str(df["timestamp"].iloc[-1]),
        }

    # -------------------------------------------------------------------------
    # Ejecución
    # -------------------------------------------------------------------------
    def execute_trade_with_size(self, signal_data: Dict[str, Any], lots: float) -> Optional[Dict[str, Any]]:
        side = signal_data.get("action", "BUY").upper()
        price = float(signal_data.get("entry_price", 0.0))
        if self.trading_client is None:
            return {"error": "TradingClient no disponible"}

        methods = [
            ("market_order", lambda: self.trading_client.market_order(self.symbol, side, lots, price=price, metadata=signal_data)),
            ("send_order", lambda: self.trading_client.send_order(self.symbol, side, lots, price=price, metadata=signal_data)),
            ("place_market_order", lambda: self.trading_client.place_market_order(self.symbol, side, lots)),
            ("order_send", lambda: self.trading_client.order_send(self.symbol, side, lots, price)),
        ]
        last_err = None
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
                    continue

        logger.error(f"❌ [{self.symbol}] No se pudo enviar orden: {last_err}")
        return {"error": f"No se pudo enviar orden ({last_err})"}

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------
    def get_signal_stats(self) -> Dict[str, Any]:
        return {"symbol": self.symbol, "strategy": self.strategy_name, "stats": self.signal_stats.copy()}
