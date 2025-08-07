# main_bot.py — Orquestado con PolicySwitcher + MT5 + Notificaciones + CycleManager M5 + Control de Posiciones
# -----------------------------------------------------------------------------------
import os
import sys
import csv
import time
import json
import signal
import logging
import threading
import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import timezone
import pytz



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


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("bot")
    logger.setLevel(logging.INFO)
    fmt = ColoredFormatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler("logs/runtime.log", encoding="utf-8")
    file_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(file_fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


# ---------- HELPERS ----------
def safe_now_utc():
    return datetime.now(timezone.utc)


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
        return True
    except Exception as e:
        print(f"❌ Error guardando log de señal: {e}")
        return False


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
        print(f"🔒 Bloqueadas por posición: {self.stats['signals_blocked_by_position']}")  # NUEVA ESTADÍSTICA
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
        print("="*80 + "\n")

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
                plan = self.cycle_mgr.get_cycle_plan(self.controllers)
                if plan["action"] == "analyze_new_candle":
                    controllers_to_process = plan["controllers_to_process"]
                    t_cycle_start = time.perf_counter()
                    for c in controllers_to_process:
                        t0 = time.perf_counter()
                        had_signal = False
                        confirmed = False
                        position_blocked = False
                        try:
                            signal_data = c.get_trading_signal()
                            if not signal_data:
                                # Si get_trading_signal() retorna None, podría ser por posición existente
                                # pero no lo contamos como señal generada si no hay datos de señal
                                continue
                            
                            # Señal detectada - incrementar contador
                            had_signal = True
                            self.stats['signals_generated'] += 1

                            # Verificar si la señal fue bloqueada por posición existente en el controller
                            # (esto se detecta por la ausencia de datos de confianza/probabilidad)
                            if (signal_data.get('confidence') is None and 
                                signal_data.get('historical_prob') is None):
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
                                    "rejection_reason": "Posición existente detectada",
                                    "position_size": 0,
                                    "ticket": ""
                                }
                                log_signal_for_backtest(log_data)
                                
                                output = format_signal_output(c.symbol, c.strategy_name, signal_data,
                                                            position_blocked=True)
                                print(output)
                                continue

                            # 🔒 VERIFICACIÓN ADICIONAL A NIVEL DE BOT
                            if not self._verify_no_existing_position(c.symbol):
                                position_blocked = True
                                self.stats['signals_blocked_by_position'] += 1
                                
                                log_data = {
                                    "timestamp_utc": safe_now_utc().isoformat(),
                                    "symbol": c.symbol,
                                    "strategy": c.strategy_name,
                                    "side": signal_data.get("action"),
                                    "entry_price": signal_data.get("entry_price"),
                                    "atr": signal_data.get("atr", 0.0),
                                    "ml_confidence": signal_data.get("confidence", 1.0),
                                    "historical_prob": signal_data.get("historical_prob", 1.0),
                                    "pnl": "",
                                    "status": "position_blocked",
                                    "rejection_reason": "Posición existente (verificación bot)",
                                    "position_size": 0,
                                    "ticket": ""
                                }
                                log_signal_for_backtest(log_data)
                                
                                output = format_signal_output(c.symbol, c.strategy_name, signal_data,
                                                            position_blocked=True)
                                print(output)
                                continue

                            # Log inicial de señal generada
                            log_data = {
                                "timestamp_utc": safe_now_utc().isoformat(),
                                "symbol": c.symbol,
                                "strategy": c.strategy_name,
                                "side": signal_data.get("action"),
                                "entry_price": signal_data.get("entry_price"),
                                "atr": signal_data.get("atr", 0.0),
                                "ml_confidence": signal_data.get("confidence", 1.0),
                                "historical_prob": signal_data.get("historical_prob", 1.0),
                                "pnl": "",
                                "status": "generated",
                                "rejection_reason": "",
                                "position_size": 0,
                                "ticket": ""
                            }
                            log_saved = log_signal_for_backtest(log_data)
                            if not log_saved:
                                self.logger.warning(f"⚠️ No se pudo guardar log inicial para {c.symbol}")

                            # Verificar noticias
                            news_blocked = False
                            if self._news_blocks(c.symbol, safe_now_utc()):
                                news_blocked = True
                                self.stats['news_blocks'] += 1
                                log_data.update({
                                    "status": "news_blocked",
                                    "rejection_reason": "Evento de noticias detectado"
                                })
                                log_signal_for_backtest(log_data)
                                output = format_signal_output(c.symbol, c.strategy_name, signal_data,
                                                            news_blocked=True)
                                print(output)
                                continue

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
                            
                        except Exception as e:
                            self.logger.exception(f"❌ Error procesando controller {c.symbol}/{c.strategy_name}: {e}")
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