# main_bot.py — encabezado (imports principales + instancias globales)
# ------------------------------------------------------------------
import os, sys, csv, time, json, signal, logging, threading
from typing import Dict, Any, List, Optional
from datetime import datetime as dt
import pytz
import pandas as pd  # ⇦ NUEVO
from pathlib import Path      # ⇦ NUEVO
import datetime as dt            # dt es el MÓDULO (dt.datetime.now/utcnow)
from datetime import timezone    # para timezone.utc

# --- PMI ---------------------------------------------------------------------
from pmi.smart_position_manager import SmartPositionManager
from pmi.logger                 import log_pmi_decision
from pmi.decision               import PMIDecision
# (Trend-Change Detector ya estaba importado)
from pmi.trend_change_detector  import TrendChangeDetector

tcd = TrendChangeDetector()              # instancia única
spm = SmartPositionManager()             # «modo observador»
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


@dataclass
class ControllerSpec:
    symbol: str
    strategy: str
    params: Dict[str, Any] = field(default_factory=dict)


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
    ):
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        self.logger = setup_logging()
        self.base_lots = base_lots
        self.cycle_seconds = cycle_seconds
        self.stop_event = threading.Event()

        # Estadísticas del bot (ampliadas)
        self.stats = {
            "signals_generated": 0,
            "signals_approved": 0,
            "signals_rejected": 0,
            "signals_executed": 0,
            "signals_blocked_by_position": 0,  # NUEVA ESTADÍSTICA
            "news_blocks": 0,
            "execution_errors": 0
        }

        # 1) Configs
        self.global_cfg = self._load_json(config_path, "global_config")
        self.risk_cfg = self._load_json(risk_path, "risk_config")
        self.timeframe = self.global_cfg.get("timeframe", "M5")

        # 2) Notifier
        self.notifier = self._setup_notifier()

        # 3) News filter (opcional)
        self.news_filter = self._setup_news_filter()

        # 4) MT5 (si aplica)
        self.mt5_connected = init_mt5_from_config(self.global_cfg, self.logger)

        # 5) TradingClient
        self.client = self._setup_trading_client()

        # 6) Policy switcher
        self.policy = PolicySwitcher(config_path=orch_cfg_path, global_insights_path=insights_path)

        # 7) Controllers
        self.controllers = self._build_controllers()

        # 8) Cycle manager
        self.cycle_mgr = OptimizedM5CycleManager()

        # 9) Señales del sistema
        self._setup_signals()

        self.logger.info("✅ Bot MT5 orquestado inicializado con control de posiciones.")
        self._print_startup_summary()

        # ---------- NUEVO: instancia PMI -----------------
        self.pmi = spm
        # Contador de decisiones PMI (útil para diagnosticar)
        self.pmi_stats = {
            "evaluations": 0,
            "close_signals": 0,
            "partial_close": 0,
            "tighten_sl": 0,
        }
        # --------------------------------------------------


    def _print_startup_summary(self):
        print("\n" + "="*80)
        print("🚀 BOT MT5 ORQUESTADO - RESUMEN DE CONFIGURACIÓN")
        print("="*80)
        print(f"📊 Timeframe: {self.timeframe}")
        print(f"💰 Lotes base: {self.base_lots}")
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
        print("="*80 + "\n")

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


    # ------------------------------------------------------------------
    # Helper: descarga velas y guarda snapshot .parquet
    # ------------------------------------------------------------------
    def _fetch_candles(self, symbol: str, timeframe: str = "M5", n: int = 400):
        """
        Devuelve un DataFrame con las últimas *n* velas del símbolo
        y guarda un snapshot de las últimas 200 en data/candles/<símbolo>/.
        """
        if not MT5_AVAILABLE or not self.mt5_connected:
            self.logger.warning(f"_fetch_candles: MT5 no disponible para {symbol}.")
            return None

        tf_map = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1,
        }
        tf_const = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_M5)

        try:
            rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, n)
            if not rates:
                self.logger.warning(f"_fetch_candles: sin datos para {symbol}")
                return None

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df.rename(columns={"tick_volume": "volume"}, inplace=True)

            # ---------- Snapshot ----------
            try:
                ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                out_dir = Path("data") / "candles" / symbol
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"{symbol}_{ts}.parquet"
                df.tail(200).to_parquet(out_file, index=False)
                self.logger.debug(f"Snapshot velas {symbol} → {out_file}")
            except Exception as e:
                self.logger.warning(f"No pude guardar snapshot velas {symbol}: {e}")
            # ------------------------------

            return df

        except Exception as e:
            self.logger.error(f"_fetch_candles: error copiando velas {symbol}: {e}")
            return None

    # ------------------------------------------------------------------
    # PMI – evaluación en modo observador
    # ------------------------------------------------------------------
    def _evaluate_open_positions(self):
        """
        Recorre las posiciones abiertas (PolicySwitcher + MT5),
        genera decisiones PMI y las registra en logs/pmi_decisions.jsonl.
        No ejecuta órdenes todavía.
        """
        try:
            # 1) Recopilar posiciones (broker y PolicySwitcher)
            open_pos: List[Dict[str, Any]] = []

            # a) MT5 (si hay conexión)
            if MT5_AVAILABLE and self.mt5_connected:
                try:
                    mt5_pos = mt5.positions_get()
                    for p in mt5_pos or []:
                        open_pos.append({
                            "ticket":   int(p.ticket),
                            "symbol":   p.symbol,
                            "type":     "BUY" if p.type == 0 else "SELL",
                            "volume":   p.volume,
                            "price":    p.price_open,
                            "open_time": p.time,
                        })
                except Exception as e:
                    self.logger.debug(f"PMI: error leyendo posiciones MT5: {e}")

            # b) PolicySwitcher (si dispone de open_positions)
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

            # 2) Snapshot de mercado mínimo (solo ASK/BID o último close)
            #    → ejemplo: {'EURUSD': {'close': 1.0967, 'atr': 0.0009}, …}
            #    aquí simplificamos con close dummy = price
            market_snap = {p["symbol"]: {"close": p["price"]} for p in open_pos}

            # 3) Evaluar con PMI
            decisions = self.pmi.evaluate(open_pos, market_snap) or []
            self.pmi_stats["evaluations"] += 1

            # 4) Registrar
            for dec in decisions:
                assert isinstance(dec, PMIDecision)
                if dec.action.name in ("CLOSE", "PARTIAL_CLOSE", "TIGHTEN_SL"):
                    self.pmi_stats[dec.action.name.lower()] += 1
                ok = log_pmi_decision(dec.__dict__)
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
        self.logger.info("🚀 Iniciando loop principal (CycleManager M5 + Control de Posiciones)...")
        while not self.stop_event.is_set():
            try:
                self._evaluate_open_positions()
                plan = self.cycle_mgr.get_cycle_plan(self.controllers)
                if plan["action"] == "analyze_new_candle":
                    controllers_to_process = plan["controllers_to_process"]
                    t_cycle_start = time.perf_counter()
                    for c in controllers_to_process:
                        t0 = time.perf_counter()
                        had_signal = False
                        confirmed = False
                        position_blocked = False
                        news_blocked = False

                            # ─── ① Descarga las 400 velas que ya usas ───
                        df = self._fetch_candles(c.symbol, timeframe="M5", n=400)
                        if df is None or df.empty:
                            continue

                        # ─── ② Trend-Change Detector (PASO B) ───
                        tcd_out = tcd.estimate_probability(df)
                        prob_tc  = float(tcd_out.get("probability", 0.0))
                        details  = tcd_out.get("details", {})

                        # Guarda por si quieres pasarlo al controller o al logger
                        extra_context = {
                            "trend_change_prob": prob_tc,
                            "trend_details": details,
                        }

                        self.logger.debug(
                            f"[PMI-TCD] {c.symbol} prob_giro={prob_tc:.2f}  detalles={details}"
                        )

                        # ─── ③ Resto de tu flujo habitual ───
                        signal_result = c.get_trading_signal_with_details(
                            df,
                            extra_context=extra_context  # solo si tu controller acepta extras
                        )
                        if not signal_result:
                            continue
                        
                        try:
                            # 🔧 NUEVO: Usar método mejorado con detalles completos
                            signal_result = None
                            if hasattr(c, 'get_trading_signal_with_details'):
                                signal_result = c.get_trading_signal_with_details()
                            else:
                                # Fallback al método original
                                signal_data = c.get_trading_signal()
                                if signal_data:
                                    signal_result = {'signal': signal_data, 'rejection_reason': None, 'status': 'generated'}

                            if not signal_result:
                                continue
                           
                            signal_data = signal_result['signal']

                            # ───────────────────────── FILTRO POSICIÓN ABIERTA  ─────────────────────────
                            if not self._verify_no_existing_position(c.symbol):
                                # ❶ Marca la señal como bloqueada, pero conserva TODOS los campos
                                signal_data["status"] = "blocked_position"
                                signal_data["rejection_reason"] = "Open position"

                                # ❷ Registra el evento con todos los datos útiles
                                _augment_log_with_extras(signal_data, signal_data)   # copia extras si faltan
                                log_signal_for_backtest({
                                    "timestamp_utc": safe_now_utc().isoformat(),
                                    "symbol": c.symbol,
                                    "strategy": c.strategy_name,
                                    "side": signal_data["action"],          # BUY / SELL real
                                    "entry_price": signal_data["entry_price"],
                                    "atr": signal_data.get("atr", 0.0),
                                    "ml_confidence": signal_data.get("confidence", 0.0),
                                    "historical_prob": signal_data.get("historical_prob", 0.0),
                                    "historical_prob_lb90": signal_data.get("historical_prob_lb90", ""),
                                    "chroma_samples": signal_data.get("chroma_samples", ""),
                                    "status": "blocked_position",
                                    "rejection_reason": "Open position",
                                    "position_size": 0,
                                    "ticket": ""
                                })

                                # ❸ Estadística y salida por consola
                                self.stats['signals_blocked_by_position'] += 1
                                print(format_signal_output(
                                    c.symbol, c.strategy_name, signal_data,
                                    verdict=None, execution_result=None,
                                    position_blocked=True
                                ))
                                continue           # <── NO pasa a policy / ejecución
                            # ─────────────────────────────────────────────────────────────────────────────

                            rejection_reason = signal_result.get('rejection_reason')
                            status = signal_result.get('status', 'unknown')
                            
                            # 🔧 CONTADOR: SIEMPRE contar como señal si hay datos básicos
                            if signal_data and signal_data.get('action'):
                                had_signal = True
                                self.stats['signals_generated'] += 1
                                
                                # Si fue rechazada por el controller, procesarla y continuar
                                if status == 'rejected' and rejection_reason:
                                    self.stats['signals_rejected'] += 1
                                    
                                    # El logging ya se hizo en el controller, solo mostrar output
                                    output = format_signal_output(
                                        c.symbol, 
                                        c.strategy_name, 
                                        signal_data,
                                        verdict={'approved': False, 'reason': rejection_reason}
                                    )
                                    print(output)
                                    continue

                            # Verificar si la señal fue bloqueada por posición (detección alternativa)
                            if (signal_data and signal_data.get('action') and 
                                not signal_data.get('confidence') and 
                                status != 'rejected'):
                                position_blocked = True
                                self.stats['signals_blocked_by_position'] += 1
                                
                                log_data = {
                                    "timestamp_utc": safe_now_utc().isoformat(),
                                    "symbol": c.symbol,
                                    "strategy": c.strategy_name,
                                    "side": signal_data.get("action"),
                                    "entry_price": signal_data.get("entry_price"),
                                    "atr": signal_data.get("atr", 0.0),
                                    "ml_confidence": 0.0,
                                    "historical_prob": 0.0,
                                    "pnl": "",
                                    "status": "position_blocked",
                                    "rejection_reason": "Posición existente detectada en controller (fallback)",
                                    "position_size": 0,
                                    "ticket": ""
                                }
                                _augment_log_with_extras(log_data, signal_data)
                                log_saved = log_signal_for_backtest(log_data)
                                if not log_saved:
                                    self.logger.warning(f"⚠️ No se pudo guardar log inicial para {c.symbol}")
                                
                                output = format_signal_output(c.symbol, c.strategy_name, signal_data,
                                                            position_blocked=True)
                                print(output)
                                
                            # Log inicial de señal generada (para señales que llegan hasta aquí)
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
                                "ticket": ""
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
                                self.stats['signals_approved'] += 1
                                log_data.update({
                                    "status": "approved",
                                    "rejection_reason": "",
                                    "position_size": verdict["position_size"]
                                })
                                log_signal_for_backtest(log_data)
                                
                                # 🔒 VERIFICACIÓN FINAL ANTES DE EJECUCIÓN
                                if not self._verify_no_existing_position(c.symbol):
                                    self.logger.error(f"🔒 [{c.symbol}] EJECUCIÓN CANCELADA: Posición detectada en verificación final")
                                    self.stats['signals_blocked_by_position'] += 1
                                    log_data.update({
                                        "status": "position_blocked",
                                        "rejection_reason": "Posición existente (verificación final)"
                                    })
                                    log_signal_for_backtest(log_data)
                                    continue

                                res = c.execute_trade_with_size(signal_data, verdict["position_size"])
                                execution_result = res
                                if res and res.get("ticket"):
                                    confirmed = True
                                    self.stats['signals_executed'] += 1
                                    ticket = res["ticket"]
                                    log_data.update({
                                        "status": "executed",
                                        "ticket": ticket
                                    })
                                    log_signal_for_backtest(log_data)
                                    self.policy.register_open(
                                        ticket, c.symbol, signal_data["action"], verdict["position_size"]
                                    )
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
                                    self.stats['execution_errors'] += 1
                                    log_data.update({
                                        "status": "execution_failed",
                                        "rejection_reason": res.get('error', 'Error de ejecución desconocido') if res else 'Sin respuesta del broker'
                                    })
                                    log_signal_for_backtest(log_data)
                            else:
                                self.stats['signals_rejected'] += 1
                                log_data.update({
                                    "status": "rejected",
                                    "rejection_reason": verdict.get('reason', 'Criterios no cumplidos'),
                                    "position_size": verdict.get('position_size', 0)
                                })
                                log_signal_for_backtest(log_data)
                            
                            output = format_signal_output(c.symbol, c.strategy_name, signal_data,
                                                        verdict, execution_result, news_blocked, position_blocked)
                            print(output)

                        finally:
                            proc_time = time.perf_counter() - t0
                            self.cycle_mgr.update_controller_metrics(c, proc_time, had_signal, confirmed)

                    cycle_time = time.perf_counter() - t_cycle_start
                    summary_msg = (f"🔄 Ciclo completado: {len(controllers_to_process)} controllers en {cycle_time:.2f}s "
                                f"| Gen: {self.stats['signals_generated']} "
                                f"| Apr: {self.stats['signals_approved']} "
                                f"| Exec: {self.stats['signals_executed']} "
                                f"| Pos❌: {self.stats['signals_blocked_by_position']} "
                                f"(plan={plan.get('reason')})")
                    self.logger.info(summary_msg)

                    # ⏱ Cuenta regresiva hasta la próxima verificación (UTC y Local)
                    next_at = plan.get("next_check_at")
                    if next_at:
                        try:
                            target = datetime.fromisoformat(next_at)
                            nowu = datetime.now(timezone.utc)
                            delta = target - nowu
                            total = max(0, int(delta.total_seconds()))
                            mm, ss = divmod(total, 60)
                            hh, mm = divmod(mm, 60)
                            local_str = target.astimezone(local_tz).strftime('%H:%M:%S')
                            self.logger.info(
                                f"⏱ Próxima verificación en {hh:02d}:{mm:02d}:{ss:02d} "
                                f"(UTC {target.strftime('%H:%M:%S')} | Local {local_str})"
                            )
                        except Exception:
                            pass

                    time.sleep(max(1, int(plan.get("wait_seconds", self.cycle_seconds))))

                elif plan["action"] in ("wait_for_stability", "wait_for_next_candle"):
                    wait_msg = (f"⏳ {plan['reason']} (espera {plan['wait_seconds']}s) | "
                                f"Stats: G:{self.stats['signals_generated']} "
                                f"A:{self.stats['signals_approved']} E:{self.stats['signals_executed']} "
                                f"Pos❌:{self.stats['signals_blocked_by_position']}")
                    self.logger.info(wait_msg)

                    # ⏱ Cuenta regresiva también durante la espera
                    next_at = plan.get("next_check_at")
                    if next_at:
                        try:
                            target = datetime.fromisoformat(next_at)
                            nowu = datetime.now(timezone.utc)
                            delta = target - nowu
                            total = max(0, int(delta.total_seconds()))
                            mm, ss = divmod(total, 60)
                            hh, mm = divmod(mm, 60)
                            local_str = target.astimezone(local_tz).strftime('%H:%M:%S')
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