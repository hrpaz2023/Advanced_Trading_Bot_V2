# main_bot.py — encabezado (imports principales + utilidades)
# ------------------------------------------------------------------
import os, sys, csv, time, json, signal, logging, threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# Tiempo / snapshots
import datetime as dt            # (dt es el MÓDULO → usar dt.datetime.now/utcnow)
from datetime import timezone
from pathlib import Path

# Terceros
import pytz
import pandas as pd

# --- PMI ---------------------------------------------------------------------
from pmi.smart_position_manager import SmartPositionManager
from pmi.logger                 import log_pmi_decision
from pmi.decision               import PMIDecision
from pmi.trend_change_detector  import TrendChangeDetector

# -----------------------------------------------------------------------------

# Zona horaria local (para logs de cuenta regresiva)
local_tz = pytz.timezone("America/Argentina/Tucuman")


# ---------- PATHS DEL PROYECTO ----------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
for p in (PROJECT_ROOT, SRC_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------- IMPORTS DINÁMICOS ----------
# Execution Controller
try:
    from src.live_execution.execution_controller import OptimizedExecutionController as ExecutionController
    EC_IMPORTED = "src"
except Exception:
    try:
        from execution_controller import OptimizedExecutionController as ExecutionController
        EC_IMPORTED = "local"
    except Exception as e:
        print("❌ No se pudo importar OptimizedExecutionController:", e)
        raise

# Policy Switcher
try:
    from policy_switcher import PolicySwitcher
except Exception:
    from src.orchestrator.policy_switcher import PolicySwitcher

# Cycle Manager
try:
    from optimized_cycle_manager import OptimizedM5CycleManager
except Exception:
    from src.live_execution.optimized_cycle_manager import OptimizedM5CycleManager

# TradingClient y Notifier (si existen en tu src)
TradingClient = None
Notifier = None
try:
    from src.trading.trading_client import TradingClient as _TC
    TradingClient = _TC
except Exception as e:
    print("⚠️ TradingClient no disponible:", e)

try:
    from src.utils.notifier import Notifier as _NT
    Notifier = _NT
except Exception as e:
    print("⚠️ Notifier no disponible:", e)

# News filter (opcional)
try:
    from src.filters.news_filter import NewsFilter
    NEWS_FILTER_AVAILABLE = True
except Exception:
    NEWS_FILTER_AVAILABLE = False

# MetaTrader5 (opcional)
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except Exception as e:
    print("⚠️ MetaTrader5 no disponible:", e)
    MT5_AVAILABLE = False


# ---------- LOGGING MEJORADO ----------
class ColoredFormatter(logging.Formatter):
    """Formatter con colores para diferentes tipos de log"""
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Verde
        'WARNING': '\033[33m',   # Amarillo
        'ERROR': '\033[31m',     # Rojo
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

# En main_bot.py, reemplaza tu setup_logging por este:
def setup_logging():
    logger = logging.getLogger("bot")
    try:
        os.makedirs("logs", exist_ok=True)
        logger.setLevel(logging.INFO)
        
        # --- Terminal Handler ---
        term_fmt = ColoredFormatter("%(asctime)s | %(levelname)s | %(message)s")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(term_fmt)

        # --- File Handler (con comprobación de errores) ---
        log_file = "logs/runtime.log"
        file_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(file_fmt)

        if not logger.handlers:
            logger.addHandler(fh)
            logger.addHandler(sh)
        
        logger.info(f"✅ Logging configurado. El log se guardará en: {os.path.abspath(log_file)}")

    except Exception as e:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!! ERROR CRÍTICO AL CONFIGURAR EL ARCHIVO DE LOG runtime.log !!!")
        print(f"!!! Error: {e}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        if not logger.handlers:
            sh = logging.StreamHandler(sys.stdout) # Asegura que al menos la terminal funcione
            logger.addHandler(sh)
        logger.error("El logger de archivo no pudo ser inicializado. Los logs solo aparecerán en la terminal.")

    return logger


# ---------- HELPERS ----------
def safe_now_utc():
    return dt.datetime.now(timezone.utc)


def log_signal_for_backtest(row: Dict[str, Any], path: str = "logs/signals_history.csv"):
    """Log mejorado que garantiza el guardado de todas las señales"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    try:
        with open(path, "a", newline="", encoding="utf-8") as f:
            fieldnames = [
                "timestamp_utc",
                "symbol",
                "strategy",
                "side",
                "entry_price",
                "atr",
                "ml_confidence",
                "historical_prob",
                "historical_prob_lb90",
                "chroma_samples",
                "causal_factor",
                "causal_ate",
                "causal_pf",
                "causal_significance",
                "pnl",
                "status",
                "rejection_reason",
                "position_size",
                "ticket"
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            complete_row = {field: row.get(field, "") for field in fieldnames}
            w.writerow(complete_row)
        
        # Also write JSON audit line
        try:
            log_signal_audit_json(complete_row)
        except Exception as _:
            pass
        return True
    except Exception as e:
        # Usamos el logger para que el error quede en runtime.log
        logger = logging.getLogger("bot")
        logger.error(f"❌ Error guardando log de señal en signals_history.csv: {e}")
        return False




def log_signal_audit_json(row: Dict[str, Any], path: str = "logs/signals_history.jsonl") -> bool:
    """Append a JSON audit record per signal for machine analysis."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        safe_row = {}
        for k, v in row.items():
            try:
                import json as _json
                _json.dumps(v)
                safe_row[k] = v
            except Exception:
                safe_row[k] = str(v)
        with open(path, "a", encoding="utf-8") as f:
            import json as _json
            f.write(_json.dumps(safe_row, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        logger = logging.getLogger("bot")
        logger.error(f"❌ Error guardando log JSON de señal: {e}")
        return False


def _augment_log_with_extras(row: Dict[str, Any], signal: Dict[str, Any]) -> None:
    """Copy optional analysis fields from signal into log row if present."""
    if not isinstance(signal, dict):
        return
    keys = ("historical_prob_lb90","chroma_samples","causal_factor","causal_ate","causal_pf","causal_significance")
    for k in keys:
        if k not in row:
            row[k] = signal.get(k, "")

def format_signal_output(symbol: str, strategy: str, signal_data: Dict, verdict: Dict = None,
                        execution_result: Dict = None, news_blocked: bool = False, 
                        position_blocked: bool = False) -> str:
    separator = "=" * 80
    header = f"\n{separator}\n🎯 SEÑAL DETECTADA: {symbol} | Estrategia: {strategy.upper()}\n{separator}"
    signal_info = (f"\n📊 DATOS DE LA SEÑAL:"
                   f"\n   • Acción: {signal_data.get('action', 'N/A').upper()}"
                   f"\n   • Precio entrada: {signal_data.get('entry_price', 'N/A')}"
                   f"\n   • ATR: {signal_data.get('atr', 'N/A')}"
                   f"\n   • Timestamp: {safe_now_utc().strftime('%H:%M:%S UTC')}")
    
    prob_info = ""
    if not position_blocked:  # Solo mostrar si la señal llegó a procesarse
        prob_info = (f"\n📈 ANÁLISIS DE PROBABILIDADES:"
                     f"\n   • Confianza ML: {signal_data.get('confidence', 1.0):.3f} ({signal_data.get('confidence', 1.0)*100:.1f}%)"
                     f"\n   • Probabilidad Histórica: {signal_data.get('historical_prob', 1.0):.3f} ({signal_data.get('historical_prob', 1.0)*100:.1f}%)")
    
    decision_info = ""
    if position_blocked:
        decision_info = (f"\n🔒 DECISIÓN: BLOQUEADA POR POSICIÓN EXISTENTE"
                         f"\n   • Razón: Ya existe una posición abierta para {symbol}")
    elif news_blocked:
        decision_info = (f"\n🚫 DECISIÓN: BLOQUEADA POR NOTICIAS"
                         f"\n   • Razón: Evento de noticias detectado para {symbol}")
    elif verdict:
        if verdict.get("approved", False):
            decision_info = (f"\n✅ DECISIÓN: APROBADA"
                             f"\n   • Tamaño posición: {verdict.get('position_size', 'N/A')} lotes"
                             f"\n   • Razón: {verdict.get('reason', 'Criterios cumplidos')}")
        else:
            decision_info = (f"\n🚫 DECISIÓN: RECHAZADA"
                             f"\n   • Razón: {verdict.get('reason', 'Criterios no cumplidos')}"
                             f"\n   • Tamaño calculado: {verdict.get('position_size', 0)} lotes")
    
    execution_info = ""
    if execution_result:
        if execution_result.get("ticket"):
            execution_info = (f"\n🎯 EJECUCIÓN: EXITOSA"
                              f"\n   • Ticket: {execution_result.get('ticket')}"
                              f"\n   • Precio ejecutado: {execution_result.get('price', 'N/A')}")
        else:
            execution_info = (f"\n❌ EJECUCIÓN: FALLIDA"
                              f"\n   • Error: {execution_result.get('error', 'Error desconocido')}")
    
    footer = f"\n{separator}\n"
    return header + signal_info + prob_info + decision_info + execution_info + footer


def init_mt5_from_config(global_cfg: Dict[str, Any], logger: logging.Logger) -> bool:
    """Inicializa MT5 aceptando 'mt5' o 'mt5_credentials' en el config."""
    if not MT5_AVAILABLE:
        logger.info("ℹ️ MT5 no disponible en el entorno.")
        return False
    mt5_obj = (global_cfg or {}).get("mt5", None)
    mt5_creds = (global_cfg or {}).get("mt5_credentials", None)
    cfg = {}
    if isinstance(mt5_obj, dict):
        cfg.update(mt5_obj)
    if isinstance(mt5_creds, dict):
        cfg.setdefault("enabled", True)
        cfg["login"] = cfg.get("login", mt5_creds.get("login"))
        cfg["password"] = cfg.get("password", mt5_creds.get("password"))
        cfg["server"] = cfg.get("server", mt5_creds.get("server"))
        cfg["path"] = cfg.get("path", mt5_creds.get("path"))
    if not cfg:
        logger.info("ℹ️ MT5 deshabilitado por configuración.")
        return False
    enabled = cfg.get("enabled", True)
    if not enabled:
        logger.info("ℹ️ MT5 deshabilitado por configuración.")
        return False
    path = cfg.get("path")
    login = cfg.get("login")
    password = cfg.get("password")
    server = cfg.get("server")
    ok = mt5.initialize(path) if path else mt5.initialize()
    if not ok:
        logger.error(f"❌ MT5 initialize falló: {mt5.last_error()}")
        return False
    if login and password and server:
        try:
            l_ok = mt5.login(int(login), password=password, server=server)
        except Exception:
            l_ok = mt5.login(login, password=password, server=server)
        if not l_ok:
            logger.error(f"❌ MT5 login falló: {mt5.last_error()}")
            return False
        logger.info(f"✅ MT5 conectado: login={login}, server={server}")
    else:
        logger.info("✅ MT5 inicializado (sin login explícito).")
    return True


def shutdown_mt5(logger: logging.Logger):
    if MT5_AVAILABLE:
        try:
            mt5.shutdown()
            logger.info("🛑 MT5 shutdown OK.")
        except Exception:
            pass


STRATEGY_MAPPING = {
    "ema_crossover": "ema_crossover",
    "EMA Crossover": "ema_crossover",
    "multi_filter_scalper": "multi_filter_scalper",
    "MultiFilterScalper": "multi_filter_scalper",
    "rsi_pullback": "rsi_pullback",
    "RSI Pullback": "rsi_pullback",
    "volatility_breakout": "volatility_breakout",
    "Volatility Breakout": "volatility_breakout",
    "channel_reversal": "channel_reversal",
    "Channel Reversal": "channel_reversal",
    "lokz_reversal": "lokz_reversal",
    "LokzReversal": "lokz_reversal",
}


# Estructura de especificación para armar controllers
@dataclass
class ControllerSpec:
    symbol: str
    strategy: str
    params: Dict[str, Any] = field(default_factory=dict)



# ---------- BOT ----------
# ---------- BOT ----------
class OrchestratedMT5Bot:
    def __init__(
        self,
        config_path: str = "configs/global_config.json",
        risk_path: str = "configs/risk_config.json",
        orch_cfg_path: str = "orchestrator_config.json",
        insights_path: str = "reports/global_insights.json",
        base_lots: float = 0.10,
        cycle_seconds: int = 30,
        symbols: list[str] | None = None,
        time_frame: str = "M5",
        logger: logging.Logger | None = None,
        # --- PMI opcional ---
        pmi: SmartPositionManager | None = None,
        trend_detector: TrendChangeDetector | None = None,
        # --- Flags PMI ---
        pmi_active: bool = False,
        pmi_partial_close_ratio: float = 0.5,
        pmi_active_symbols: Optional[List[str]] | None = None,
        **kwargs,
    ):
        """Constructor con soporte PMI opcional."""
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        # Logger
        self.logger = logger or setup_logging()

        # Config básicos
        self.symbols = symbols or ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]
        self.time_frame = time_frame
        self.base_lots = base_lots
        self.cycle_seconds = cycle_seconds
        self.stop_event = threading.Event()

        # ✅ FIX: Inicializar TODOS los stats necesarios TEMPRANO
        self.stats = {
            "signals_generated": 0,
            "signals_approved": 0,
            "signals_executed": 0,
            "signals_rejected": 0,
            "signals_rejected_by_controller": 0,
            "positions_blocked": 0,
            "signals_blocked_by_position": 0,    # ← Key missing fix
            "signals_blocked_by_news": 0,
            "signals_blocked_by_risk": 0,
            "execution_errors": 0,               # ← Key missing fix
            "news_blocks": 0,                    # ← Key missing fix
        }

        # ✅ FIX 2: Inicializar trend_change_detector TEMPRANO
        try:
            self.trend_change_detector = TrendChangeDetector() if TrendChangeDetector else None
        except Exception:
            self.trend_change_detector = None

        # Cargar config PMI desde JSON
        pmi_cfg = self._load_pmi_config()

        # PMI (Position Management Intelligence)
        try:
            self.pmi = SmartPositionManager(
                mode=pmi_cfg.get("mode", "active"),
                close_thresholds=pmi_cfg.get("thresholds", None),
            )
        except Exception:
            # fallback por si cambió la firma
            self.pmi = SmartPositionManager()

        # Flag: ¿aplicamos acciones del PMI?
        self.pmi_active = (getattr(self.pmi, "mode", "observer") == "active")

        # Contexto de la última señal por símbolo (para "señal opuesta" y TCD)
        self._last_signal_ctx = {}

        # Telemetría PMI
        self.pmi_stats = {
            "evaluations": 0,
            "close_signals": 0,
            "partial_close": 0,
            "tighten_sl": 0,
        }

        # Contadores de señales del loop (para banners/logs)
        self.stats = {
            "signals_generated": 0,
            "signals_approved": 0,
            "signals_executed": 0,
            "positions_blocked": 0,
            "signals_rejected": 0,               # <-- faltaba
            "signals_rejected_by_controller": 0  # útil si diferenciás rechazos del controller
        }

        # Si querés sincronizar lb90_min con los controllers, hacelo donde instancias controllers,
        # leyendo pmi_cfg["lb90_min"] y pasándolo a cada controlador como parámetro.
        # ---------------------------------------------------------------

        # Carga de configs y setup de subsistemas (ajusta según tus helpers)
        self.global_cfg = self._load_json(config_path, "global_config")
        self.risk_cfg = self._load_json(risk_path, "risk_config")
        self.timeframe = self.global_cfg.get("timeframe", "M5")

        self.notifier = self._setup_notifier()
        self.news_filter = self._setup_news_filter()

        self.mt5_connected = init_mt5_from_config(self.global_cfg, self.logger)
        self.client = self._setup_trading_client()

        self.policy = PolicySwitcher(config_path=orch_cfg_path, global_insights_path=insights_path)
        self.controllers = self._build_controllers()
        self.cycle_mgr = OptimizedM5CycleManager()
        self._setup_signals()

        self.logger.info("✅ Bot MT5 orquestado inicializado con control de posiciones.")
        self._print_startup_summary()

    def _load_pmi_config(self, path: str = "configs/pmi_config.json") -> dict:
        """
        Lee configs/pmi_config.json si existe. Devuelve dict con defaults seguros si falta o está mal.
        Estructura esperada:
        {
        "mode": "active" | "observer",
        "thresholds": { ... },
        "lb90_min": 0.25
        }
        """
        defaults = {
            "mode": "active",
            "thresholds": {
                "tighten_sl": 0.70,
                "partial_close": 0.82,
                "close": 0.90,
                "tcd_tighten": 0.55,
                "tcd_close": 0.70,
                "opp_partial_ml": 0.55,
                "opp_partial_lb90": 0.50,
                "opp_close_ml": 0.58,
                "opp_close_lb90": 0.53,
                "partial_fraction": 0.50
            },
            "lb90_min": 0.25
        }
        try:
            p = Path(path)
            if not p.exists():
                self.logger.warning(f"PMI config no encontrado: {path}. Uso defaults.")
                return defaults
            with p.open("r", encoding="utf-8") as f:
                cfg = json.load(f) or {}
            # merge superficial (mantiene defaults si faltan llaves)
            out = dict(defaults)
            out["mode"] = str(cfg.get("mode", defaults["mode"])).lower()
            thr = dict(defaults["thresholds"])
            thr.update(cfg.get("thresholds", {}) or {})
            out["thresholds"] = thr
            out["lb90_min"] = float(cfg.get("lb90_min", defaults["lb90_min"]))
            self.logger.info(f"✅ pmi_config cargado: {path}")
            return out
        except Exception as e:
            try:
                self.logger.warning(f"PMI config inválido ({path}): {e}. Uso defaults.")
            except Exception:
                print(f"PMI config inválido ({path}): {e}. Uso defaults.")
            return defaults

    def _inc_stat(self, key: str, n: int = 1) -> None:
        try:
            self.stats[key] = int(self.stats.get(key, 0)) + int(n)
        except Exception:
            self.stats[key] = self.stats.get(key, 0)

    # Add these helper methods to the OrchestratedMT5Bot class

    def _inc_stat(self, key: str, n: int = 1) -> None:
        """Safely increment a statistic, initializing if not exists."""
        try:
            if key not in self.stats:
                self.stats[key] = 0
            self.stats[key] = int(self.stats.get(key, 0)) + int(n)
        except Exception as e:
            self.logger.warning(f"Error incrementing stat {key}: {e}")
            self.stats[key] = self.stats.get(key, 0)

    def _get_stat(self, key: str, default: int = 0) -> int:
        """Safely get a statistic value."""
        return self.stats.get(key, default)

    def _ensure_all_stats_exist(self):
        """Ensure all required stats keys exist with default values."""
        required_stats = {
            "signals_generated": 0,
            "signals_approved": 0,
            "signals_executed": 0,
            "signals_rejected": 0,
            "signals_rejected_by_controller": 0,
            "positions_blocked": 0,
            "signals_blocked_by_position": 0,
            "signals_blocked_by_news": 0,
            "signals_blocked_by_risk": 0,
            "execution_errors": 0,
            "news_blocks": 0,
        }
        
        for key, default_value in required_stats.items():
            if key not in self.stats:
                self.stats[key] = default_value

    def _print_startup_summary(self):
        import os, json

        def _load_risk_cfg_for_banner():
            paths = ["configs/risk_config.json", "risk_config.json"]
            for p in paths:
                if os.path.exists(p):
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            raw = json.load(f)
                        return raw.get("position_sizing", raw)
                    except Exception:
                        pass
            # fallback si no hay config
            return {"mode": "fixed", "fixed_lots": 0.10, "symbol_overrides": {}}

        def _unique_symbols_from_controllers():
            seen = set()
            ordered = []
            for c in getattr(self, "controllers", []):
                s = getattr(c, "symbol", None)
                if s and s not in seen:
                    seen.add(s)
                    ordered.append(s)
            # si no hay controllers aún, usa majors por defecto
            return ordered or ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]

        def _resolve_display_lots(symbols, risk_cfg):
            mode = str(risk_cfg.get("mode", "fixed")).lower()
            sym_over = risk_cfg.get("symbol_overrides", {}) or {}
            fixed = risk_cfg.get("fixed_lots", risk_cfg.get("base_lot", 0.10))

            if mode == "fixed":
                if sym_over:
                    parts = [f"{s}:{float(sym_over.get(s, fixed)):.2f}" for s in symbols]
                    return f"{', '.join(parts)}", "fixed"
                else:
                    return f"{float(fixed):.2f}", "fixed"

            pct = float(risk_cfg.get("risk_per_trade", 1.0))
            return f"{pct:.2f}% (percent_risk)", "percent_risk"

        try:
            risk_cfg = _load_risk_cfg_for_banner()
            unique_symbols = _unique_symbols_from_controllers()
            lots_txt, sizing_mode = _resolve_display_lots(unique_symbols, risk_cfg)

            print("\n" + "="*80)
            print("🚀 BOT MT5 ORQUESTADO - RESUMEN DE CONFIGURACIÓN")
            print("="*80)
            print(f"📊 Timeframe: {self.timeframe}")
            print(f"💰 Lotes ({sizing_mode}): {lots_txt}")
            print(f"⏱️  Ciclo: {self.cycle_seconds}s")
            print(f"🎯 Controllers activos: {len(self.controllers)}")
            if self.controllers:
                print("\n📈 INSTRUMENTOS Y ESTRATEGIAS:")
                for i, c in enumerate(self.controllers, 1):
                    print(f"   {i}. {c.symbol} → {c.strategy_name}")
            print(f"\n🔌 Conexiones:")
            print(f"   • MT5: {'✅ Conectado' if self.mt5_connected else '❌ Desconectado'}")
            print(f"   • TradingClient: {'✅ Activo' if self.client else '❌ No disponible'}")
            print(f"   • Notifier: {'✅ Activo' if self.notifier else '❌ No disponible'}")
            print(f"   • NewsFilter: {'✅ Activo' if self.news_filter else '❌ No disponible'}")
            print(f"\n🔒 CONTROL DE POSICIONES: ACTIVADO")
            print("   • Prevención de múltiples posiciones por símbolo")
            print("   • Verificación en tiempo real con MT5")
            pmi_mode = getattr(self, "pmi", None)
            pmi_mode_txt = getattr(pmi_mode, "mode", "unknown") if pmi_mode else "disabled"
            lb90_min = getattr(getattr(self, "controllers", [None])[0], "lb90_min", 0.25) if self.controllers else 0.25
            print(f"\n🧠 PMI: modo={pmi_mode_txt} | LB90_min={lb90_min:.2f}")
            print("="*80 + "\n")
        except Exception as e:
            print(f"⚠️ No se pudo imprimir el resumen de configuración: {e}")

        try:
            thr = getattr(self.pmi, "close_thresholds", {})
            print(f"\n🧠 PMI: modo={getattr(self.pmi,'mode','unknown')} | LB90_min={pmi_cfg.get('lb90_min',0.25):.2f}")
            print(f"   • Umbrales: close={thr.get('close'):.2f} partial={thr.get('partial_close'):.2f} tighten={thr.get('tighten_sl'):.2f} | tcd_close={thr.get('tcd_close'):.2f}")
        except Exception:
            pass
    
    # --------- setup ---------
    def _load_json(self, path, name):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.logger.info(f"✅ {name} cargado: {path}")
            return data
        except Exception as e:
            self.logger.error(f"❌ No se pudo cargar {name} en {path}: {e}")
            raise

    def _setup_notifier(self):
        if Notifier is None:
            self.logger.info("ℹ️ Notifier no disponible.")
            return None
        try:
            notifier = Notifier()
            try:
                notifier.send("🚀 Bot iniciado", "Bot MT5 orquestado en ejecución con control de posiciones.")
            except Exception:
                if hasattr(notifier, "send_notification"):
                    notifier.send_notification("🚀 Bot iniciado", "Bot MT5 orquestado en ejecución con control de posiciones.")
            self.logger.info("✅ Notifier activo.")
            return notifier
        except Exception as e:
            self.logger.warning(f"⚠️ No se pudo inicializar Notifier: {e}")
            return None

    def _setup_news_filter(self):
        if not NEWS_FILTER_AVAILABLE:
            return None
        try:
            symbols = []
            if self.global_cfg:
                if "controllers" in self.global_cfg:
                    symbols = sorted(set([c["symbol"] for c in self.global_cfg["controllers"]]))
                else:
                    symbols = self.global_cfg.get("trading_symbols", [])
            nf = NewsFilter(symbols_to_monitor=symbols)
            self.logger.info(f"✅ NewsFilter activo para {len(symbols)} símbolos.")
            return nf
        except Exception as e:
            self.logger.warning(f"⚠️ NewsFilter no inicializado: {e}")
            return None

    def _news_blocks(self, symbol, when=None):
        """Devuelve True si hay evento bloqueante para el símbolo. Soporta múltiples firmas."""
        if not self.news_filter:
            return False
        cand_methods = [
            "should_block", "should_block_symbol", "should_block_trade",
            "blocks", "is_blocked", "has_blocking_event", "has_event", "block_trade"
        ]
        for m in cand_methods:
            if hasattr(self.news_filter, m):
                fn = getattr(self.news_filter, m)
                try:
                    return bool(fn(symbol, when)) if when is not None else bool(fn(symbol))
                except TypeError:
                    try:
                        return bool(fn(symbol))
                    except Exception:
                        continue
                except Exception:
                    continue
        try:
            blocked = getattr(self.news_filter, "blocked_symbols", None)
            if isinstance(blocked, (set, list, tuple)) and symbol in blocked:
                return True
        except Exception:
            pass
        return False

    def _setup_trading_client(self):
        """Inicializa TradingClient probando distintas firmas comunes."""
        if TradingClient is None:
            self.logger.warning("⚠️ TradingClient no disponible. Se ejecutará en modo no-operativo.")
            return None
        mt5_obj = (self.global_cfg or {}).get("mt5", {}) or {}
        mt5_creds = (self.global_cfg or {}).get("mt5_credentials", {}) or {}
        creds = {}
        creds.update(mt5_obj)
        creds.update(mt5_creds)
        login = creds.get("login")
        password = creds.get("password")
        server = creds.get("server")
        magic_number = (self.global_cfg or {}).get("magic_number") or creds.get("magic_number")
        try:
            login = int(login) if login is not None and str(login).isdigit() else login
        except Exception:
            pass
        missing = [k for k, v in [("login", login), ("password", password), ("server", server), ("magic_number", magic_number)]
                   if v in (None, "", "xxx", "xxxxx")]
        if missing:
            self.logger.warning(f"⚠️ TradingClient no inicializado: faltan campos {missing}. "
                                f"Revisa 'mt5'/'mt5_credentials' y 'magic_number' en configs/global_config.json.")
            return None
        attempts = [
            ("positional", lambda: TradingClient(login, password, server, magic_number)),
            ("keywords",  lambda: TradingClient(login=login, password=password, server=server, magic_number=magic_number)),
            ("kwargs+magic", lambda: TradingClient(**{
                "login": login, "password": password, "server": server, "magic_number": magic_number
            })),
            ("global_cfg", lambda: TradingClient(self.global_cfg)),
        ]
        last_err = None
        for name, ctor in attempts:
            try:
                client = ctor()
                for m in ("connect", "initialize", "init"):
                    if hasattr(client, m):
                        try:
                            getattr(client, m)()
                            break
                        except Exception as e:
                            self.logger.warning(f"⚠️ TradingClient.{m} falló en intento {name}: {e}")
                self.logger.info(f"✅ TradingClient inicializado (método: {name}).")
                return client
            except TypeError as e:
                last_err = e
                self.logger.debug(f"Firma {name} no compatible: {e}")
            except Exception as e:
                last_err = e
                self.logger.warning(f"⚠️ TradingClient intento {name} falló: {e}")
        self.logger.error(f"❌ No se pudo inicializar TradingClient con ninguna firma. Último error: {last_err}")
        return None

    def _resolve_position_size(self, symbol: str, entry_price: float, atr: float, fallback: float = 0.10) -> float:
        """
        Devuelve lotes finales usando el RiskManager (que ya lee overrides/fixed_lots de configs/risk_config.json).
        Si algo falla, usa 'fallback'.
        """
        try:
            # SL aproximado si la estrategia no lo provee: 2*ATR en pips (simple y robusto)
            is_jpy = symbol.endswith("JPY")
            pip_factor = 100.0 if is_jpy else 10000.0
            sl_pips = max(10.0, 2.0 * float(atr) * pip_factor)

            # Equity de la cuenta si está disponible
            equity = None
            try:
                import MetaTrader5 as mt5
                info = mt5.account_info()
                equity = float(getattr(info, "equity", 0.0)) if info else None
            except Exception:
                pass

            # Calcular tamaño con RiskManager (este ya respeta overrides/fixed)
            if hasattr(self, "risk_manager") and self.risk_manager:
                size = self.risk_manager.calculate_position_size(
                    account_equity=(equity or 10000.0),  # fallback de equity
                    stop_loss_pips=sl_pips,
                    symbol=symbol
                )
            else:
                size = fallback

            # Seguridad mínima
            return max(0.01, float(size))

        except Exception as e:
            try:
                self.logger.warning(f"_resolve_position_size fallback por error: {e}")
            except Exception:
                pass
            return max(0.01, float(fallback))


# 🔧 MODIFICAR el método _build_controllers (alrededor de línea 470)
    def _build_controllers(self) -> List[ExecutionController]:
        controllers: List[ExecutionController] = []
        specs: List[ControllerSpec] = []
        
        if self.global_cfg and "controllers" in self.global_cfg:
            for item in self.global_cfg["controllers"]:
                specs.append(
                    ControllerSpec(
                        symbol=item["symbol"],
                        strategy=STRATEGY_MAPPING.get(item["strategy"], item["strategy"]),
                        params=item.get("params", {}),
                    )
                )
        else:
            symbols = self.global_cfg.get("trading_symbols", []) if self.global_cfg else []
            default_strategy = "ema_crossover"
            for s in symbols:
                specs.append(ControllerSpec(symbol=s, strategy=default_strategy, params={}))
        
        for sp in specs:
            try:
                c = ExecutionController(
                    trading_client=self.client,
                    symbol=sp.symbol,
                    strategy_name=sp.strategy,
                    strategy_params=sp.params,
                    notifier=self.notifier,
                )
                setattr(c, "timeframe", self.timeframe)
                # 🔒 VINCULAR POLICY SWITCHER AL CONTROLLER PARA VERIFICACIÓN DE POSICIONES
                c.set_policy_switcher(self.policy)
                
                # 🔧 NUEVO: Vincular función de logging al controller
                c.set_external_log_function(log_signal_for_backtest)
                
                controllers.append(c)
            except Exception as e:
                self.logger.error(f"❌ No se pudo crear controller {sp.symbol}/{sp.strategy}: {e}")
        
        self.logger.info(f"✅ Controllers construidos: {len(controllers)} (con control de posiciones)")
        return controllers

    def _setup_signals(self):
        def _handle_stop(signum, frame):
            self.logger.info("⏹️  Señal de parada recibida, cerrando bot...")
            self._print_final_stats()
            self.stop_event.set()
        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(s, _handle_stop)
            except Exception:
                pass

    def _print_final_stats(self):
        print("\n" + "="*80)
        print("📊 ESTADÍSTICAS FINALES DEL BOT")
        print("="*80)
        print(f"🎯 Señales generadas: {self.stats['signals_generated']}")
        print(f"✅ Señales aprobadas: {self.stats['signals_approved']}")
        print(f"🚫 Señales rechazadas: {self.stats['signals_rejected']}")
        print(f"🔒 Bloqueadas por posición: {self.stats['signals_blocked_by_position']}")
        print(f"💰 Operaciones ejecutadas: {self.stats['signals_executed']}")
        print(f"📰 Bloqueadas por noticias: {self.stats['news_blocks']}")
        print(f"❌ Errores de ejecución: {self.stats['execution_errors']}")
        
        if self.stats['signals_generated'] > 0:
            approval_rate = (self.stats['signals_approved'] / self.stats['signals_generated']) * 100
            position_block_rate = (self.stats['signals_blocked_by_position'] / self.stats['signals_generated']) * 100
            execution_rate = (self.stats['signals_executed'] / self.stats['signals_approved']) * 100 if self.stats['signals_approved'] > 0 else 0
            print(f"\n📈 Tasa de aprobación: {approval_rate:.1f}%")
            print(f"🔒 Tasa de bloqueo por posición: {position_block_rate:.1f}%")
            print(f"🎯 Tasa de ejecución: {execution_rate:.1f}%")
        
        # 🔧 NUEVO: Mostrar estadísticas detalladas por controller
        self._print_controller_stats()
        print("="*80 + "\n")

    def _print_controller_stats(self):
        lines = []
        controllers = getattr(self, "controllers", [])
        if not controllers:
            self.logger.info("📊 No hay controllers cargados.")
            return

        for c in controllers:
            sym   = getattr(c, "symbol", "?")
            strat = getattr(c, "strategy_name", "?")
            cs    = getattr(c, "stats", {}) or {}

            g   = cs.get("signals_generated", 0)
            a   = cs.get("signals_approved", 0)
            e   = cs.get("signals_executed", 0)
            rej = cs.get("signals_rejected", 0)
            pb  = cs.get("signals_blocked_by_position", 0)
            err = cs.get("execution_errors", 0)
            ne  = cs.get("not_executed", 0)

            lines.append(
                f"   • {sym}/{strat}: gen={g} apr={a} exec={e} rej={rej} pos❌={pb} not_exec={ne} err_exec={err}"
            )

        self.logger.info("📊 Estadísticas por controller:\n" + "\n".join(lines))

    # --- Helpers de sizing y redondeo por bróker ---
    def _get_broker_volume_specs(self, symbol: str):
        try:
            import MetaTrader5 as mt5
            si = mt5.symbol_info(symbol)
            if si:
                return float(si.volume_min), float(si.volume_max), float(si.volume_step)
        except Exception:
            pass
        # Defaults típicos para FX
        return 0.01, 100.0, 0.01

    def _round_volume(self, vol: float, step: float) -> float:
        # round hacia abajo al múltiplo del step
        return max(step, (int(vol / step)) * step)

    def _compute_lot_size(self, symbol: str, atr_value: float | None = None) -> float:
        """Devuelve el tamaño final a enviar al bróker."""
        # 1) base_lots desde orchestrator_config o desde el parámetro del bot
        sizing_cfg = getattr(self, 'orchestrator_config', {}).get("sizing", {})
        base_lots_cfg = float(sizing_cfg.get("base_lots", self.base_lots))
        method = (self.risk_cfg.get("position_sizing_method") or "fixed").lower()

        lots = base_lots_cfg
        if method == "fixed":
            lots = base_lots_cfg
        else:
            # atr_based (y otros) — si no tenés stop/ATR claro, no subestimes: usa al menos base_lots
            lots = max(base_lots_cfg, lots)

        # 2) recortes por límites locales
        max_local = float(self.global_cfg.get("trading_settings", {}).get("max_position_size", 100.0))
        lots = min(lots, max_local)

        # 3) recortes por exposición direccional si lo tienes activo
        exp_cfg = getattr(self, 'orchestrator_config', {}).get("exposure_limits", {})
        max_dir_net = float(exp_cfg.get("max_direction_net_lots", 999.0))
        # Nota: si ya controlas net exposure fuera, podés omitir este recorte aquí

        # 4) Ajuste al step del bróker
        vmin, vmax, vstep = self._get_broker_volume_specs(symbol)
        lots = min(vmax, max(vmin, lots))
        lots = self._round_volume(lots, vstep)

        # Log visible
        self.logger.info(f"📏 Sizing {symbol}: method={method} base={base_lots_cfg} → lots_final={lots} (step={vstep}, min={vmin}, max={vmax})")
        return lots



    # ------------------------------------------------------------------
    # Helper: descarga velas y guarda snapshot .parquet  (REEMPLAZAR COMPLETO)
    # ------------------------------------------------------------------
    def _fetch_candles(self, symbol: str, timeframe: str = "M5", n: int = 400):
        """
        Devuelve un DataFrame con las últimas *n* velas del símbolo
        y guarda un snapshot de las últimas 200 en data/candles/<símbolo>/.
        """
        try:
            if not MT5_AVAILABLE or not self.mt5_connected:
                self.logger.warning(f"_fetch_candles: MT5 no disponible para {symbol}.")
                return None

            tf_map = {
                "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1,
            }
            tf_const = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_M5)

            rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, n)

            # ✅ Validación segura para numpy arrays
            if rates is None or len(rates) == 0:
                self.logger.warning(f"_fetch_candles: sin datos para {symbol}")
                return None

            df = pd.DataFrame(rates)
            # normalización de columnas
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            if "tick_volume" in df.columns:
                df.rename(columns={"tick_volume": "volume"}, inplace=True)

            # ---------- Snapshot ----------
            try:
                ts = dt.datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")  # ← sin utcnow()
                out_dir = Path("data") / "candles" / symbol
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"{symbol}_{ts}.parquet"
                df.tail(200).to_parquet(out_file, index=False)
                self.logger.debug(f"Snapshot velas {symbol} → {out_file}")
            except Exception as e:
                # no interrumpir si falla sólo el snapshot
                self.logger.warning(f"No pude guardar snapshot velas {symbol}: {e}")
            # ------------------------------

            return df

        except Exception as e:
            self.logger.error(f"_fetch_candles: error copiando velas {symbol}: {e}")
            return None

    # ------------------------------------------------------------------
    # PMI: recoger posiciones vivas desde TradingClient o MT5
    # ------------------------------------------------------------------
    def _collect_open_positions(self) -> list[dict]:
        """
        Devuelve una lista de posiciones con campos mínimos:
        ticket, symbol, volume, price_open, sl, tp, time
        """
        # 1) Intenta usar tu TradingClient si existe
        try:
            if hasattr(self, "trading_client") and hasattr(self.trading_client, "get_open_positions"):
                pos = self.trading_client.get_open_positions()
                if isinstance(pos, list):
                    return pos
        except Exception as e:
            self.logger.warning(f"PMI: trading_client.get_open_positions() falló: {e}")

        # 2) Fallback a MT5 nativo
        try:
            import MetaTrader5 as mt5  # por si no está en el scope
            raw = mt5.positions_get()
            out = []
            if raw:
                for p in raw:
                    out.append({
                        "ticket": int(p.ticket),
                        "symbol": str(p.symbol),
                        "volume": float(p.volume),
                        "price_open": float(p.price_open),
                        "sl": float(p.sl) if hasattr(p, "sl") else None,
                        "tp": float(p.tp) if hasattr(p, "tp") else None,
                        "time": int(p.time),  # epoch
                    })
            return out
        except Exception as e:
            self.logger.warning(f"PMI: mt5.positions_get() falló: {e}")
            return []

    # ------------------------------------------------------------------
    # PMI: aplicar decisión (modo activo opcional)
    # ------------------------------------------------------------------
    def _apply_pmi_decision(self, decision: "PMIDecision", position: dict) -> None:

        if not getattr(self, "pmi_active", False):
            return  # modo observador

        action = str(decision.action).upper()  # 'CLOSE' / 'PARTIAL_CLOSE' / 'TIGHTEN_SL' / 'HOLD'
        ticket = int(decision.ticket)
        symbol = str(position.get("symbol"))
        volume = float(position.get("volume", 0.0))
        partial_ratio = getattr(self, "pmi_partial_close_ratio", 0.5)

        # Whitelist (piloto)
        wl = getattr(self, "pmi_active_symbols", set())
        if wl and symbol not in wl:
            self.logger.info(f"PMI: {symbol} fuera de whitelist, acción={action} omitida")
            return

        # Umbral de seguridad
        close_score = float(getattr(decision, "close_score", 0.0))
        if action in ("CLOSE", "PARTIAL_CLOSE", "TIGHTEN_SL") and close_score < 0.92:
            self.logger.info(f"PMI: score {close_score:.3f} < 0.92, omito acción {action} ({symbol})")
            return

        try:
            tc = getattr(self, "trading_client", None) or getattr(self, "client", None)

            if action == "CLOSE":
                if tc and hasattr(tc, "close_position"):
                    tc.close_position(ticket=ticket)
                    self.logger.info(f"PMI: CLOSE ejecutado ticket={ticket} ({symbol})")

            elif action == "PARTIAL_CLOSE":
                vol_to_close = max(0.0, round(volume * partial_ratio, 2))
                if vol_to_close > 0 and tc and hasattr(tc, "partial_close"):
                    tc.partial_close(ticket=ticket, volume=vol_to_close)
                    self.logger.info(f"PMI: PARTIAL_CLOSE {vol_to_close} lotes ticket={ticket} ({symbol})")

            elif action == "TIGHTEN_SL":
                if tc and hasattr(tc, "move_stop_to_be"):
                    tc.move_stop_to_be(ticket=ticket)
                    self.logger.info(f"PMI: TIGHTEN_SL → BE ticket={ticket} ({symbol})")

            # HOLD → no-op
        except Exception as e:
            self.logger.error(f"PMI: error aplicando acción {action} ticket={ticket}: {e}")
  

    # ------------------------------------------------------------------
    # PMI: paso de integración por ciclo
    # ------------------------------------------------------------------
    def _pmi_integration_step(self, candles_by_symbol: dict[str, "pd.DataFrame"]) -> None:
        """
        1) Lee posiciones abiertas
        2) Arma market_snapshot (close, atr_rel) usando velas si hay; si no, cae a tick MT5
        3) Llama a PMI.evaluate(...)
        4) Loguea decisiones y (si pmi_active) ejecuta acciones
        """
        # 1) posiciones vivas
        positions = self._collect_open_positions()
        if not positions:
            return

        # 2) market_snapshot con fallback a tick
        snapshot: dict[str, dict] = {}

        # a) desde velas ya descargadas
        for sym, df in (candles_by_symbol or {}).items():
            try:
                last_close = float(df["close"].iloc[-1])
                atr_rel = float(df.get("atr_rel", pd.Series([0.0])).iloc[-1]) if isinstance(df, pd.DataFrame) else 0.0
                snapshot[sym] = {"close": last_close, "atr_rel": atr_rel}
            except Exception:
                pass

        # b) fallback para símbolos en posiciones abiertas que no quedaron en snapshot
        try:
            import MetaTrader5 as mt5
            for p in positions:
                sym = str(p.get("symbol"))
                if sym and sym not in snapshot:
                    tick = mt5.symbol_info_tick(sym)
                    if tick:
                        # usa el precio relevante al lado de la posición si querés; aquí close ~ mid simple
                        mid = float((tick.bid + tick.ask) / 2.0) if tick.bid and tick.ask else float(tick.last or 0.0)
                        snapshot[sym] = {"close": mid, "atr_rel": 0.0}
        except Exception:
            pass

        # 3) evaluar con PMI (pasamos velas y contexto de señal para manejar señales opuestas)
        now_utc = dt.datetime.now(timezone.utc)
        try:
            decisions = self.pmi.evaluate(
                positions=positions,
                market_snapshot=snapshot,
                candles_by_symbol=candles_by_symbol,
                now=now_utc,
                signal_context_by_symbol=getattr(self, "_last_signal_ctx", {}),
            ) or []
        except TypeError:
            # si tu versión de PMI no acepta signal_context_by_symbol aún
            decisions = self.pmi.evaluate(
                positions=positions,
                market_snapshot=snapshot,
                candles_by_symbol=candles_by_symbol,
                now=now_utc,
            ) or []

        # 4) registrar y (opcional) ejecutar
        try:
            from pmi.logger import log_pmi_decision
        except Exception:
            def log_pmi_decision(*args, **kwargs):
                return False  # fallback silencioso

        for dec in decisions:
            try:
                log_pmi_decision(dec)
            except Exception as e:
                self.logger.warning(f"PMI: no pude loguear decisión {dec}: {e}")

            if getattr(self, "pmi_active", False):
                try:
                    pos = next((p for p in positions if int(p.get("ticket")) == int(dec.ticket)), None)
                    if pos:
                        self._apply_pmi_decisions([dec], positions)
                except Exception as e:
                    self.logger.error(f"PMI: error aplicando decisión {dec}: {e}")


    # === BEGIN: PMI SIGNAL CONTEXT UPDATER ===
    def _update_signal_context(self, symbol: str, signal_result: dict, extra_context: dict | None = None):
        """
        Extrae lado/ML/LB90/TCD de la señal detectada y lo guarda en self._last_signal_ctx[symbol].
        Llamar SIEMPRE que haya señal (aprobada, rechazada o bloqueada).
        """
        try:
            ctx = {}
            
            # ✅ FIX 3: Verificar que signal_result no sea None
            if signal_result is None:
                return
                
            # lado de la señal
            side = signal_result.get("side") or signal_result.get("action") or signal_result.get("signal_side")
            if side:
                ctx["signal_side"] = str(side).upper()

            # métricas de controller/ML
            for k_src, k_dst in [
                ("ml_confidence", "ml_confidence"),
                ("historical_prob_lb90", "historical_prob_lb90"),
                ("tcd_prob", "tcd_prob"),
            ]:
                v = signal_result.get(k_src)
                if v is not None and v != "":
                    try:
                        ctx[k_dst] = float(v)
                    except Exception:
                        pass

            # si el TCD vino en extra_context
            if extra_context and "tcd" in extra_context:
                try:
                    tcd_prob = extra_context["tcd"].get("prob") or extra_context["tcd"].get("prob_tc")
                    if tcd_prob is not None:
                        ctx["tcd_prob"] = float(tcd_prob)
                except Exception:
                    pass

            if ctx:
                self._last_signal_ctx[symbol] = ctx
        except Exception as e:
            try:
                self.logger.warning(f"_update_signal_context error: {e}")
            except Exception:
                print(f"_update_signal_context error: {e}")
    # === END: PMI SIGNAL CONTEXT UPDATER ===

    # === BEGIN: PMI APPLY DECISIONS ===
    def _apply_pmi_decisions(self, decisions: list, open_positions: list):
        """
        Aplica decisiones del PMI si el PMI está en modo 'active'.
        - CLOSE: cierra por ticket.
        - PARTIAL_CLOSE: cierra fracción (por defecto 50% si viene en telemetry['fraction']).
        - TIGHTEN_SL: ajusta SL (aquí ejemplo simple: mover SL a breakeven si está en profit; ajusta según tu RiskManager si querés algo más sofisticado).
        """
        if not getattr(self.pmi, "is_active", lambda: False)():
            return

        # Mapeo rápido ticket -> (symbol, volume, side, price_open)
        by_ticket = {}
        for p in open_positions:
            by_ticket[int(p["ticket"])] = {
                "symbol": p["symbol"],
                "volume": float(p.get("volume", 0.0) or p.get("lots", 0.0) or 0.0),
                "side": str(p.get("type", p.get("side", ""))).upper(),
                "price_open": float(p.get("price_open", 0.0) or p.get("entry_price", 0.0) or 0.0),
            }

        import MetaTrader5 as mt5

        for d in decisions:
            action = getattr(d, "action", None) or d.get("action")
            ticket = int(getattr(d, "ticket", 0) or d.get("ticket", 0))
            reason = getattr(d, "reason", "") or d.get("reason", "")
            telemetry = getattr(d, "telemetry", {}) or d.get("telemetry", {}) or {}

            if not ticket or ticket not in by_ticket:
                continue

            sym = by_ticket[ticket]["symbol"]
            vol = by_ticket[ticket]["volume"]
            pos_side = by_ticket[ticket]["side"]
            if vol <= 0:
                continue

            # --- CLOSE ---
            if str(action).upper().endswith("CLOSE") and "PARTIAL" not in str(action).upper():
                # cerrar todo: enviar orden contraria
                side = "SELL" if pos_side == "BUY" else "BUY"
                self.logger.info(f"PMI CLOSE ticket={ticket} symbol={sym} reason={reason}")
                self.client.market_order(sym, side, vol, metadata={"strategy": "PMI_CLOSE"})
                # cooldown corto para evitar reentrada inmediata
                try:
                    self.pmi.timer.set_cooldown(sym, minutes=5)
                except Exception:
                    pass
                continue

            # --- PARTIAL_CLOSE ---
            if "PARTIAL" in str(action).upper():
                frac = float(telemetry.get("fraction", self.pmi.close_thresholds.get("partial_fraction", 0.5)))
                close_vol = max(0.01, round(vol * frac, 2))
                side = "SELL" if pos_side == "BUY" else "BUY"
                self.logger.info(f"PMI PARTIAL_CLOSE ticket={ticket} symbol={sym} fraction={frac:.2f} reason={reason}")
                self.client.market_order(sym, side, close_vol, metadata={"strategy": "PMI_PARTIAL"})
                continue

            # --- TIGHTEN_SL ---
            if "TIGHTEN" in str(action).upper():
                # Ejemplo simple: mover SL a breakeven (si está en profit). Puedes reemplazar por tu RiskManager.
                try:
                    # Obtener posición en MT5
                    pos = None
                    for p in mt5.positions_get():
                        if p.ticket == ticket:
                            pos = p; break
                    if not pos:
                        continue

                    # Precio de BE: precio de apertura
                    be_price = float(pos.price_open)
                    # Actualizar SL vía order_modify no está soportado en posiciones (es para órdenes pendientes).
                    # En MT5 se hace via TRADE_ACTION_SLTP, pero el wrapper puro no lo expone; atajo:
                    # re-crear request con sl actualizado usando order_send con type=ORDER_TYPE_BUY/SELL y 'sl' en request
                    tick = mt5.symbol_info_tick(sym)
                    if not tick:
                        continue
                    order_type = mt5.ORDER_TYPE_BUY if pos.type == 0 else mt5.ORDER_TYPE_SELL
                    px = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": sym,
                        "sl": be_price,
                        "tp": pos.tp,
                        "position": ticket,
                        "magic": int(self.client.magic_number),
                        "type": order_type,
                        "price": px,
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    self.logger.info(f"PMI TIGHTEN_SL ticket={ticket} symbol={sym} move SL->BE: {be_price}")
                    mt5.order_send(request)
                except Exception as e:
                    self.logger.warning(f"PMI TIGHTEN_SL fallo: {e}")
                continue
    # === END: PMI APPLY DECISIONS ===



    # ------------------------------------------------------------------
    # PMI – evaluación en modo observador
    # ------------------------------------------------------------------
    def _evaluate_open_positions(self):
        """
        Recorre las posiciones abiertas (PolicySwitcher + MT5),
        genera decisiones PMI y las registra en logs/pmi_decisions.jsonl.
        No ejecuta órdenes (modo observador).
        """
        try:
            # 1) Recopilar posiciones (broker y PolicySwitcher)
            open_pos: List[Dict[str, Any]] = []

            # a) MT5
            if MT5_AVAILABLE and self.mt5_connected:
                try:
                    mt5_pos = mt5.positions_get()
                    for p in mt5_pos or []:
                        open_pos.append({
                            "ticket":   int(p.ticket),
                            "symbol":   p.symbol,
                            "type":     "BUY" if p.type == 0 else "SELL",
                            "volume":   float(p.volume),
                            "price":    float(p.price_open),
                            "open_time": int(p.time),
                        })
                except Exception as e:
                    self.logger.debug(f"PMI: error leyendo posiciones MT5: {e}")

            # b) PolicySwitcher (si expone open_positions)
            try:
                if hasattr(self.policy, "open_positions"):
                    for t, pos in self.policy.open_positions.items():
                        if isinstance(pos, dict):
                            d = pos.copy()
                            d.setdefault("ticket", t)
                            open_pos.append(d)
            except Exception as e:
                self.logger.debug(f"PMI: error leyendo open_positions: {e}")

            if not open_pos:
                return

            # 2) Snapshot mínimo (a falta de último close real)
            market_snap = {p["symbol"]: {"close": p.get("price", 0.0)} for p in open_pos}

            # 3) Evaluar con PMI
            decisions = self.pmi.evaluate(open_pos, market_snap) or []
            self.pmi_stats["evaluations"] += 1

            # 4) Registrar decisiones (usa el dataclass, no __dict__)
            for dec in decisions:
                if dec.action.name in ("CLOSE", "PARTIAL_CLOSE", "TIGHTEN_SL"):
                    self.pmi_stats[dec.action.name.lower()] += 1
                ok = log_pmi_decision(dec)
                if not ok:
                    self.logger.warning("PMI: no pude guardar decisión en jsonl")

        except Exception as e:
            self.logger.debug(f"PMI: error durante evaluación: {e}")



    def _verify_no_existing_position(self, symbol: str) -> bool:
        """
        Verificación adicional a nivel de bot para evitar posiciones duplicadas.
        Devuelve True si NO hay posición (es seguro operar), False si hay posición.
        """
        # 1. Verificar en MT5 directamente (más confiable)
        if MT5_AVAILABLE:
            try:
                positions = mt5.positions_get(symbol=symbol)
                if positions and len(positions) > 0:
                    self.logger.warning(f"🔒 [{symbol}] Posición detectada en MT5 (verificación bot): {len(positions)} posición(es)")
                    return False
            except Exception as e:
                self.logger.debug(f"Error verificando MT5 en bot: {e}")

        # 2. Verificar en PolicySwitcher
        try:
            if hasattr(self.policy, 'open_positions'):
                open_positions = getattr(self.policy, 'open_positions', {})
                symbol_positions = [pos for pos in open_positions.values() 
                                  if isinstance(pos, dict) and pos.get('symbol') == symbol]
                if symbol_positions:
                    self.logger.warning(f"🔒 [{symbol}] Posición detectada en PolicySwitcher (verificación bot): {len(symbol_positions)} posición(es)")
                    return False
        except Exception as e:
            self.logger.debug(f"Error verificando PolicySwitcher en bot: {e}")

        # 3. No hay posiciones detectadas - es seguro operar
        return True

    # --------- loop ---------
    def run(self):
        """Complete run method with all fixes applied"""
        
        # Ensure all stats exist at the start
        self._ensure_all_stats_exist()
        
        self.logger.info("🚀 Iniciando loop principal (CycleManager M5 + Control de Posiciones)...")
        
        while not self.stop_event.is_set():
            try:
                # Evaluación rápida (observador) al inicio del ciclo
                self._evaluate_open_positions()

                plan = self.cycle_mgr.get_cycle_plan(self.controllers)
                if plan["action"] == "analyze_new_candle":
                    controllers_to_process = plan["controllers_to_process"]
                    t_cycle_start = time.perf_counter()

                    # Donde juntamos las velas para el PMI
                    candles_by_symbol: Dict[str, pd.DataFrame] = {}

                    for c in controllers_to_process:
                        t0 = time.perf_counter()
                        had_signal = False
                        confirmed = False
                        position_blocked = False
                        news_blocked = False

                        # ① Descargar velas
                        df = self._fetch_candles(c.symbol, timeframe="M5", n=400)
                        if df is None or df.empty:
                            continue
                        candles_by_symbol[c.symbol] = df

                        # ② TCD (trend change)
                        try:
                            if self.trend_change_detector:
                                tcd_out = self.trend_change_detector.estimate_probability(df)
                                prob_tc = float(tcd_out.get("probability", 0.0))
                                tcd_details = {k: v for k, v in tcd_out.items() if k != "probability"}
                                self.logger.info(f"[{c.symbol}] TCD prob={prob_tc:.3f} details={tcd_details}")
                            else:
                                prob_tc = 0.0
                                tcd_details = {}
                        except Exception as e:
                            self.logger.warning(f"[{c.symbol}] TCD error: {e}")
                            prob_tc = 0.0
                            tcd_details = {}

                        # ③ Verificar filtro de noticias temprano
                        try:
                            if self._news_blocks(c.symbol):
                                news_blocked = True
                                self._inc_stat("signals_blocked_by_news")
                                self.logger.info(f"🚫 [{c.symbol}] Bloqueado por noticias")
                                continue
                        except Exception as e:
                            self.logger.warning(f"Error verificando filtro de noticias para {c.symbol}: {e}")

                        # ④ Señal + contexto extra
                        extra_context = {
                            "tcd_prob": prob_tc,
                            "tcd": tcd_details,
                        }
                        signal_result = c.get_trading_signal_with_details(df, extra_context=extra_context)
                        self._update_signal_context(c.symbol, signal_result, extra_context=extra_context)

                        if not signal_result:
                            continue

                        try:
                            signal_data = signal_result["signal"]

                            # ───────── Filtro posición abierta (bot-level) ─────────
                            if not self._verify_no_existing_position(c.symbol):
                                position_blocked = True
                                signal_data["status"] = "blocked_position"
                                signal_data["rejection_reason"] = "Open position"

                                _augment_log_with_extras(signal_data, signal_data)
                                log_signal_for_backtest({
                                    "timestamp_utc": safe_now_utc().isoformat(),
                                    "symbol": c.symbol,
                                    "strategy": c.strategy_name,
                                    "side": signal_data["action"],
                                    "entry_price": signal_data["entry_price"],
                                    "atr": signal_data.get("atr", 0.0),
                                    "ml_confidence": signal_data.get("confidence", 0.0),
                                    "historical_prob": signal_data.get("historical_prob", 0.0),
                                    "historical_prob_lb90": signal_data.get("historical_prob_lb90", ""),
                                    "chroma_samples": signal_data.get("chroma_samples", ""),
                                    "status": "blocked_position",
                                    "rejection_reason": "Open position",
                                    "position_size": 0,
                                    "ticket": "",
                                })
                                
                                # ✅ FIX: Use safe increment method
                                self._inc_stat("signals_blocked_by_position")
                                
                                print(format_signal_output(
                                    c.symbol, c.strategy_name, signal_data,
                                    verdict=None, execution_result=None,
                                    position_blocked=True
                                ))
                                continue
                            # ──────────────────────────────────────────────────────

                            rejection_reason = signal_result.get("rejection_reason")
                            status = signal_result.get("status", "unknown")

                            if signal_data and signal_data.get("action"):
                                had_signal = True
                                self._inc_stat("signals_generated")

                                if status == "rejected" and rejection_reason:
                                    self._inc_stat("signals_rejected")
                                    output = format_signal_output(
                                        c.symbol, c.strategy_name, signal_data,
                                        verdict={"approved": False, "reason": rejection_reason}
                                    )
                                    print(output)
                                    continue

                            # Log inicial (señales que avanzan)
                            log_data = {
                                "timestamp_utc": safe_now_utc().isoformat(),
                                "symbol": c.symbol,
                                "strategy": c.strategy_name,
                                "side": signal_data.get("action"),
                                "entry_price": signal_data.get("entry_price"),
                                "atr": signal_data.get("atr", 0.0),
                                "ml_confidence": signal_data.get("confidence", 1.0),
                                "historical_prob": signal_data.get("historical_prob", 1.0),
                                "historical_prob_lb90": signal_data.get("historical_prob_lb90", ""),
                                "chroma_samples": signal_data.get("chroma_samples", ""),
                                "causal_factor": signal_data.get("causal_factor", ""),
                                "causal_ate": signal_data.get("causal_ate", ""),
                                "causal_pf": signal_data.get("causal_pf", ""),
                                "causal_significance": signal_data.get("causal_significance", ""),
                                "pnl": "",
                                "status": "generated",
                                "rejection_reason": "",
                                "position_size": 0,
                                "ticket": "",
                            }
                            _augment_log_with_extras(log_data, signal_data)
                            log_saved = log_signal_for_backtest(log_data)
                            if not log_saved:
                                self.logger.warning(f"⚠️ No se pudo guardar log inicial para {c.symbol}")

                            # Policy approval
                            payload = {
                                "atr": float(signal_data.get("atr", 0.0) or 0.0),
                                "confidence": float(signal_data.get("confidence", 1.0) or 1.0),
                                "historical_prob": float(signal_data.get("historical_prob", 1.0) or 1.0),
                                "base_lots": self.base_lots,
                                "now_utc": safe_now_utc(),
                            }
                            verdict = self.policy.approve_signal(
                                c.symbol, c.strategy_name, signal_data["action"], payload
                            )

                            execution_result = None
                            if verdict["approved"] and verdict["position_size"] > 0:
                                self._inc_stat("signals_approved")
                                log_data.update({
                                    "status": "approved",
                                    "rejection_reason": "",
                                    "position_size": verdict["position_size"],
                                })
                                log_signal_for_backtest(log_data)

                                # Verificación final anti-doble posición
                                if not self._verify_no_existing_position(c.symbol):
                                    self.logger.error(f"🔒 [{c.symbol}] EJECUCIÓN CANCELADA: Posición detectada en verificación final")
                                    self._inc_stat("signals_blocked_by_position")
                                    log_data.update({
                                        "status": "position_blocked",
                                        "rejection_reason": "Posición existente (verificación final)",
                                    })
                                    log_signal_for_backtest(log_data)
                                    continue

                                res = c.execute_trade_with_size(signal_data, verdict["position_size"])
                                execution_result = res
                                if res and res.get("ticket"):
                                    confirmed = True
                                    self._inc_stat("signals_executed")
                                    ticket = res["ticket"]
                                    log_data.update({
                                        "status": "executed",
                                        "ticket": ticket,
                                    })
                                    log_signal_for_backtest(log_data)
                                    self.policy.register_open(ticket, c.symbol, signal_data["action"], verdict["position_size"])
                                    if self.notifier:
                                        try:
                                            self.notifier.send(
                                                f"✅ Trade {c.symbol} {signal_data['action']}",
                                                f"Lots: {verdict['position_size']}\n"
                                                f"Price: {signal_data['entry_price']}\nATR: {signal_data.get('atr')}",
                                            )
                                        except Exception:
                                            if hasattr(self.notifier, "send_notification"):
                                                self.notifier.send_notification(
                                                    f"✅ Trade {c.symbol} {signal_data['action']}",
                                                    f"Lots: {verdict['position_size']}\n"
                                                    f"Price: {signal_data['entry_price']}\nATR: {signal_data.get('atr')}",
                                                )
                                else:
                                    self._inc_stat("execution_errors")
                                    log_data.update({
                                        "status": "execution_failed",
                                        "rejection_reason": res.get("error", "Error de ejecución desconocido") if res else "Sin respuesta del broker",
                                    })
                                    log_signal_for_backtest(log_data)
                            else:
                                self._inc_stat("signals_rejected")
                                log_data.update({
                                    "status": "rejected",
                                    "rejection_reason": verdict.get("reason", "Criterios no cumplidos"),
                                    "position_size": verdict.get("position_size", 0),
                                })
                                log_signal_for_backtest(log_data)

                            output = format_signal_output(
                                c.symbol, c.strategy_name, signal_data,
                                verdict, execution_result, news_blocked, position_blocked
                            )
                            print(output)

                        except Exception as inner_e:
                            self.logger.error(f"Error procesando señal para {c.symbol}: {inner_e}")
                            continue

                        finally:
                            proc_time = time.perf_counter() - t0
                            self.cycle_mgr.update_controller_metrics(c, proc_time, had_signal, confirmed)

                    # --- PMI integration step (usa las velas ya descargadas) ---
                    try:
                        self._pmi_integration_step(candles_by_symbol)
                    except Exception as e:
                        self.logger.error(f"PMI step error: {e}")

                    cycle_time = time.perf_counter() - t_cycle_start
                    summary_msg = (f"🔄 Ciclo completado: {len(controllers_to_process)} controllers en {cycle_time:.2f}s "
                                f"| Gen: {self._get_stat('signals_generated')} "
                                f"| Apr: {self._get_stat('signals_approved')} "
                                f"| Exec: {self._get_stat('signals_executed')} "
                                f"| Pos❌: {self._get_stat('signals_blocked_by_position')} "
                                f"(plan={plan.get('reason')})")
                    self.logger.info(summary_msg)

                    # Cuenta regresiva (usar dt.datetime)
                    next_at = plan.get("next_check_at")
                    if next_at:
                        try:
                            target = dt.datetime.fromisoformat(next_at)
                            nowu = dt.datetime.now(timezone.utc)
                            delta = target - nowu
                            total = max(0, int(delta.total_seconds()))
                            mm, ss = divmod(total, 60)
                            hh, mm = divmod(mm, 60)
                            local_str = target.astimezone(local_tz).strftime("%H:%M:%S")
                            self.logger.info(
                                f"⏱ Próxima verificación en {hh:02d}:{mm:02d}:{ss:02d} "
                                f"(UTC {target.strftime('%H:%M:%S')} | Local {local_str})"
                            )
                        except Exception:
                            pass

                    time.sleep(max(1, int(plan.get("wait_seconds", self.cycle_seconds))))

                elif plan["action"] in ("wait_for_stability", "wait_for_next_candle"):
                    wait_msg = (f"⏳ {plan['reason']} (espera {plan['wait_seconds']}s) | "
                                f"Stats: G:{self._get_stat('signals_generated')} "
                                f"A:{self._get_stat('signals_approved')} E:{self._get_stat('signals_executed')} "
                                f"Pos❌:{self._get_stat('signals_blocked_by_position')}")
                    self.logger.info(wait_msg)

                    # Cuenta regresiva (usar dt.datetime)
                    next_at = plan.get("next_check_at")
                    if next_at:
                        try:
                            target = dt.datetime.fromisoformat(next_at)
                            nowu = dt.datetime.now(timezone.utc)
                            delta = target - nowu
                            total = max(0, int(delta.total_seconds()))
                            mm, ss = divmod(total, 60)
                            hh, mm = divmod(mm, 60)
                            local_str = target.astimezone(local_tz).strftime("%H:%M:%S")
                            self.logger.info(
                                f"⏱ Próxima verificación en {hh:02d}:{mm:02d}:{ss:02d} "
                                f"(UTC {target.strftime('%H:%M:%S')} | Local {local_str})"
                            )
                        except Exception:
                            pass

                    time.sleep(max(1, int(plan["wait_seconds"])))
                else:
                    time.sleep(self.cycle_seconds)

            except Exception as e:
                self.logger.exception(f"❌ Error en loop principal: {e}")
                time.sleep(max(5, self.cycle_seconds))

        shutdown_mt5(self.logger)
        self._print_final_stats()
        self.logger.info("🛑 Bot detenido correctamente.")


# ---------- MAIN ----------
if __name__ == "__main__":
    bot = OrchestratedMT5Bot(
        config_path="configs/global_config.json",
        risk_path="configs/risk_config.json",
        orch_cfg_path="orchestrator_config.json",
        insights_path="reports/global_insights.json",
        base_lots=0.10,
        cycle_seconds=30,
    )
    bot.run()
            