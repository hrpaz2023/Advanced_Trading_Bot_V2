# main_bot.py — FIXED VERSION (Bug fixes for KeyError and other issues)

import os, sys, csv, time, json, signal, logging, threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# Tiempo / snapshots
import datetime as dt
from datetime import timezone
from pathlib import Path

# Terceros
import pytz
import pandas as pd

# --- PMI ---------------------------------------------------------------------
from pmi.smart_position_manager import SmartPositionManager
from pmi.logger import log_pmi_decision
from pmi.decision import PMIDecision
from pmi.trend_change_detector import TrendChangeDetector

# -----------------------------------------------------------------------------

# Zona horaria local (para logs de cuenta regresiva)
local_tz = pytz.timezone("America/Argentina/Tucuman")

# ... [Previous imports and setup code remain the same until class definition] ...

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

        # ✅ FIX 1: Inicializar stats COMPLETO al principio del constructor
        self.stats = {
            "signals_generated": 0,
            "signals_approved": 0,
            "signals_rejected": 0,
            "signals_blocked_by_position": 0,
            "signals_executed": 0,
            "news_blocks": 0,
            "execution_errors": 0,
            "signals_rejected_by_controller": 0,
            "positions_blocked": 0,  # Alternative name used in some places
        }

        # ✅ FIX 2: Inicializar trend_change_detector con manejo de errores
        try:
            self.trend_change_detector = TrendChangeDetector() if TrendChangeDetector else None
        except Exception as e:
            self.logger.warning(f"Could not initialize TrendChangeDetector: {e}")
            self.trend_change_detector = None

        # Cargar config PMI desde JSON
        pmi_cfg = self._load_pmi_config()

        # PMI (Position Management Intelligence)
        try:
            self.pmi = SmartPositionManager(
                mode=pmi_cfg.get("mode", "active"),
                close_thresholds=pmi_cfg.get("thresholds", None),
            )
        except Exception as e:
            self.logger.warning(f"Could not initialize PMI: {e}")
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

        # Resto de la inicialización...
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
            self.logger.warning(f"PMI config inválido ({path}): {e}. Uso defaults.")
            return defaults

    def _inc_stat(self, key: str, n: int = 1) -> None:
        """✅ FIX 3: Safe stats increment with key initialization"""
        try:
            if key not in self.stats:
                self.stats[key] = 0
            self.stats[key] = int(self.stats.get(key, 0)) + int(n)
        except Exception as e:
            self.logger.warning(f"Error incrementing stat {key}: {e}")
            self.stats[key] = self.stats.get(key, 0)

    def _update_signal_context(self, symbol: str, signal_result: dict, extra_context: dict | None = None):
        """
        ✅ FIX 4: Safe signal context update with None checks
        """
        try:
            if signal_result is None:
                return
                
            ctx = {}
            
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
            self.logger.warning(f"_update_signal_context error: {e}")

    def _fetch_candles(self, symbol: str, timeframe: str = "M5", n: int = 400):
        """
        ✅ FIX 5: Safe candle fetching with proper error handling
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
                ts = dt.datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                out_dir = Path("data") / "candles" / symbol
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"{symbol}_{ts}.parquet"
                df.tail(200).to_parquet(out_file, index=False)
                self.logger.debug(f"Snapshot velas {symbol} → {out_file}")
            except Exception as e:
                # no interrumpir si falla sólo el snapshot
                self.logger.warning(f"No pude guardar snapshot velas {symbol}: {e}")

            return df

        except Exception as e:
            self.logger.error(f"_fetch_candles: error copiando velas {symbol}: {e}")
            return None

    def run(self):
        """✅ FIX 6: Main loop with comprehensive error handling and safe stats access"""
        self.logger.info("🚀 Iniciando loop principal (CycleManager M5 + Control de Posiciones)...")
        
        while not self.stop_event.is_set():
            try:
                # Evaluación rápida (observador) al inicio del ciclo
                try:
                    self._evaluate_open_positions()
                except Exception as e:
                    self.logger.debug(f"Error in position evaluation: {e}")

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

                        try:
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

                            # ③ Señal + contexto extra
                            extra_context = {
                                "tcd_prob": prob_tc,
                                "tcd": tcd_details,
                            }
                            signal_result = c.get_trading_signal_with_details(df, extra_context=extra_context)
                            self._update_signal_context(c.symbol, signal_result, extra_context=extra_context)

                            if not signal_result:
                                continue

                            signal_data = signal_result.get("signal")
                            if not signal_data:
                                continue

                            # ───────── Filtro posición abierta (bot-level) ─────────
                            if not self._verify_no_existing_position(c.symbol):
                                signal_data["status"] = "blocked_position"
                                signal_data["rejection_reason"] = "Open position"

                                _augment_log_with_extras(signal_data, signal_data)
                                log_signal_for_backtest({
                                    "timestamp_utc": safe_now_utc().isoformat(),
                                    "symbol": c.symbol,
                                    "strategy": c.strategy_name,
                                    "side": signal_data.get("action", ""),
                                    "entry_price": signal_data.get("entry_price", ""),
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
                                self._inc_stat("signals_blocked_by_position")
                                print(format_signal_output(
                                    c.symbol, c.strategy_name, signal_data,
                                    verdict=None, execution_result=None,
                                    position_blocked=True
                                ))
                                continue

                            # Continue with signal processing...
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

                            # ... rest of signal processing logic remains the same ...

                        except Exception as e:
                            self.logger.error(f"Error processing controller {c.symbol}: {e}")
                            continue
                        finally:
                            proc_time = time.perf_counter() - t0
                            try:
                                self.cycle_mgr.update_controller_metrics(c, proc_time, had_signal, confirmed)
                            except Exception as e:
                                self.logger.debug(f"Error updating controller metrics: {e}")

                    # --- PMI integration step ---
                    try:
                        self._pmi_integration_step(candles_by_symbol)
                    except Exception as e:
                        self.logger.error(f"PMI step error: {e}")

                    cycle_time = time.perf_counter() - t_cycle_start
                    
                    # ✅ FIX 7: Safe stats access in summary message
                    summary_msg = (
                        f"🔄 Ciclo completado: {len(controllers_to_process)} controllers en {cycle_time:.2f}s "
                        f"| Gen: {self.stats.get('signals_generated', 0)} "
                        f"| Apr: {self.stats.get('signals_approved', 0)} "
                        f"| Exec: {self.stats.get('signals_executed', 0)} "
                        f"| Pos❌: {self.stats.get('signals_blocked_by_position', 0)} "
                        f"(plan={plan.get('reason', 'unknown')})"
                    )
                    self.logger.info(summary_msg)

                    # Countdown logic...
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
                        except Exception as e:
                            self.logger.debug(f"Error in countdown display: {e}")

                    time.sleep(max(1, int(plan.get("wait_seconds", self.cycle_seconds))))

                elif plan["action"] in ("wait_for_stability", "wait_for_next_candle"):
                    # ✅ FIX 8: Safe stats access in wait message
                    wait_msg = (
                        f"⏳ {plan.get('reason', 'Waiting')} (espera {plan.get('wait_seconds', self.cycle_seconds)}s) | "
                        f"Stats: G:{self.stats.get('signals_generated', 0)} "
                        f"A:{self.stats.get('signals_approved', 0)} E:{self.stats.get('signals_executed', 0)} "
                        f"Pos❌:{self.stats.get('signals_blocked_by_position', 0)}"
                    )
                    self.logger.info(wait_msg)

                    # Countdown for wait states...
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
                        except Exception as e:
                            self.logger.debug(f"Error in countdown display: {e}")

                    time.sleep(max(1, int(plan.get("wait_seconds", self.cycle_seconds))))
                else:
                    time.sleep(self.cycle_seconds)

            except KeyboardInterrupt:
                self.logger.info("⏹️ Keyboard interrupt received, stopping bot...")
                break
            except Exception as e:
                self.logger.exception(f"❌ Error en loop principal: {e}")
                time.sleep(max(5, self.cycle_seconds))

        # Cleanup
        try:
            shutdown_mt5(self.logger)
        except Exception as e:
            self.logger.error(f"Error shutting down MT5: {e}")
            
        self._print_final_stats()
        self.logger.info("🛑 Bot detenido correctamente.")

    def _print_final_stats(self):
        """✅ FIX 9: Safe final stats printing"""
        try:
            print("\n" + "="*80)
            print("📊 ESTADÍSTICAS FINALES DEL BOT")
            print("="*80)
            print(f"🎯 Señales generadas: {self.stats.get('signals_generated', 0)}")
            print(f"✅ Señales aprobadas: {self.stats.get('signals_approved', 0)}")
            print(f"🚫 Señales rechazadas: {self.stats.get('signals_rejected', 0)}")
            print(f"🔒 Bloqueadas por posición: {self.stats.get('signals_blocked_by_position', 0)}")
            print(f"💰 Operaciones ejecutadas: {self.stats.get('signals_executed', 0)}")
            print(f"📰 Bloqueadas por noticias: {self.stats.get('news_blocks', 0)}")
            print(f"❌ Errores de ejecución: {self.stats.get('execution_errors', 0)}")
            
            signals_generated = self.stats.get('signals_generated', 0)
            if signals_generated > 0:
                signals_approved = self.stats.get('signals_approved', 0)
                signals_blocked = self.stats.get('signals_blocked_by_position', 0)
                
                approval_rate = (signals_approved / signals_generated) * 100
                position_block_rate = (signals_blocked / signals_generated) * 100
                execution_rate = (self.stats.get('signals_executed', 0) / signals_approved) * 100 if signals_approved > 0 else 0
                
                print(f"\n📈 Tasa de aprobación: {approval_rate:.1f}%")
                print(f"🔒 Tasa de bloqueo por posición: {position_block_rate:.1f}%")
                print(f"🎯 Tasa de ejecución: {execution_rate:.1f}%")
            
            self._print_controller_stats()
            print("="*80 + "\n")
        except Exception as e:
            self.logger.error(f"Error printing final stats: {e}")

    # ... [Rest of the methods remain the same] ...

# ---------- MAIN ----------
if __name__ == "__main__":
    try:
        bot = OrchestratedMT5Bot(
            config_path="configs/global_config.json",
            risk_path="configs/risk_config.json",
            orch_cfg_path="orchestrator_config.json",
            insights_path="reports/global_insights.json",
            base_lots=0.10,
            cycle_seconds=30,
        )
        bot.run()
    except KeyboardInterrupt:
        print("🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
