# advanced_analyzer_v2.py - Análisis completo con ML, auto-optimización
# + Export a nivel trade para pipeline causal
# VERSIÓN CORREGIDA E INTEGRADA - FIX CAUSAL EXPORT
import json
import os
import pandas as pd
import numpy as np
import optuna
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
import warnings
import traceback
from datetime import datetime, timedelta
from pathlib import Path

def _canonicalize_time_column(df, prefer_col=None, fallback_name="timestamp"):
    """
    Devuelve (df_normalizado, nombre_col_tiempo) o (None, None) si no se pudo.
    - Acepta índices datetime y los convierte a columna.
    - Fuerza timezone UTC y elimina NaT.
    """
    import pandas as pd

    if df is None or len(df) == 0:
        return None, None

    d = df.copy()

    # 1) Si ya tenemos preferida
    candidates = [prefer_col] if prefer_col else []
    # 2) Candidatas típicas
    candidates += ['timestamp', 'time', 'Date', 'date', 'datetime', 'Time', 'Timestamp']

    for c in candidates:
        if c and c in d.columns:
            d[c] = pd.to_datetime(d[c], utc=True, errors='coerce')
            d = d.dropna(subset=[c])
            return d, c

    # 3) Si el índice es datetime, convertirlo a columna
    if isinstance(d.index, pd.DatetimeIndex):
        d = d.reset_index()
        idx_col = d.columns[0]
        d.rename(columns={idx_col: fallback_name}, inplace=True)
        d[fallback_name] = pd.to_datetime(d[fallback_name], utc=True, errors='coerce')
        d = d.dropna(subset=[fallback_name])
        return d, fallback_name

    return None, None
import sqlite3

warnings.filterwarnings('ignore')

# ML Libraries con manejo de errores
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    ML_AVAILABLE = True
    print("✅ sklearn disponible")
except ImportError as e:
    print(f"⚠️ sklearn no disponible: {e}")
    ML_AVAILABLE = False

# Importaciones de estrategias con manejo de errores
STRATEGY_IMPORTS = {}
AVAILABLE_STRATEGIES = []

try:
    from src.backtesting.backtester import Backtester
    BACKTESTER_AVAILABLE = True
except ImportError:
    print("⚠️ Backtester no disponible")
    BACKTESTER_AVAILABLE = False

# Intentar importar estrategias
strategy_modules = [
    ('ema_crossover', 'EmaCrossover'),
    ('channel_reversal', 'ChannelReversal'),
    ('rsi_pullback', 'RsiPullback'),
    ('volatility_breakout', 'VolatilityBreakout'),
    ('multi_filter_scalper', 'MultiFilterScalper'),
    ('lokz_reversal', 'LokzReversal')
]

for module_name, class_name in strategy_modules:
    try:
        module = __import__(f'src.strategies.{module_name}', fromlist=[class_name])
        strategy_class = getattr(module, class_name)
        STRATEGY_IMPORTS[module_name] = strategy_class
        AVAILABLE_STRATEGIES.append(module_name)
        print(f"✅ Estrategia cargada: {module_name}")
    except ImportError as e:
        print(f"⚠️ No se pudo cargar {module_name}: {e}")
    except Exception as e:
        print(f"❌ Error cargando {module_name}: {e}")

# Configuración
SYMBOLS = ['AUDUSD', 'EURUSD', 'GBPUSD', 'USDJPY']

class UltimateTradeAnalyzerV2:
    def __init__(self, optimization_dir='optimization_studies'):
        self.optimization_dir = Path(optimization_dir)
        self.export_dir = Path('ultimate_analysis_reports_v2')
        self.auto_optimization_dir = Path('auto_optimized_strategies_v2')

        # Crear directorios
        self.export_dir.mkdir(exist_ok=True)
        self.auto_optimization_dir.mkdir(exist_ok=True)

        self.analysis_results = {}
        self.generate_code = True
        self.current_params = {}

        print('📁 Directorios configurados:')
        print(f'   Optimización: {self.optimization_dir}')
        print(f'   Exportación : {self.export_dir}')
        print(f'   Código auto : {self.auto_optimization_dir}')

    # -------------------- UTILIDADES DB/OPTUNA --------------------
    def verify_db_file(self, db_path: Path):
        try:
            if not db_path.exists():
                return False, 'Archivo no existe'
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            required_tables = ['studies', 'trials']
            missing = [t for t in required_tables if t not in tables]
            if missing:
                return False, f'Tablas faltantes: {missing}'
            return True, 'DB válida'
        except Exception as e:
            return False, f'Error verificando DB: {e}'

    def load_optimized_params_from_db(self, symbol, strategy_name):
        study_name = f'{symbol}_{strategy_name}'
        db_file = self.optimization_dir / f'{study_name}.db'

        is_valid, msg = self.verify_db_file(db_file)
        if not is_valid:
            print(f"⚠️ {study_name}: {msg} - usando parámetros por defecto")
            return self.get_default_params(symbol, strategy_name)

        try:
            storage_url = f"sqlite:///{db_file}"
            summaries = optuna.study.get_all_study_summaries(storage=storage_url)
            names = [s.study_name for s in summaries]
            if study_name not in names:
                print(f"⚠️ Estudio {study_name} no encontrado")
                return self.get_default_params(symbol, strategy_name)
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
            if not completed:
                print(f"⚠️ {study_name}: sin trials completados válidos")
                return self.get_default_params(symbol, strategy_name)
            best = max(completed, key=lambda x: x.value)
            if best and best.params:
                print(f"✅ {study_name}: Cargado - Score={best.value:.4f}, Trials={len(completed)}")
                return best.params
            print(f"⚠️ {study_name}: best trial sin parámetros")
            return self.get_default_params(symbol, strategy_name)
        except Exception as e:
            print(f"❌ Error cargando {study_name}: {e}")
            return self.get_default_params(symbol, strategy_name)

    def get_default_params(self, symbol, strategy_name):
        defaults = {
            'multi_filter_scalper': {
                'ema_fast': 9, 'ema_mid': 21, 'ema_slow': 55,
                'rsi_buy': 45, 'rsi_sell': 55, 'prox_pct': 0.1,
                'pivot_lb': 5, 'rsi_len': 14, 'atr_len': 14, 'atr_mult': 1.5
            },
            'rsi_pullback': {'rsi_level': 30, 'trend_ema_period': 200, 'rsi_period': 14},
            'ema_crossover': {'fast_period': 20, 'slow_period': 50},
            'volatility_breakout': {'period': 20, 'atr_period': 14, 'breakout_threshold': 1.5},
            'channel_reversal': {'period': 20, 'std_dev': 2.0, 'entry_threshold': 0.8},
            'lokz_reversal': {'asia_session': '00:00-08:00', 'lokz_session': '08:00-10:00', 'timezone': 'UTC'}
        }
        return defaults.get(strategy_name, {})

    def get_available_strategies(self):
        available = {}
        print('🔍 Detectando estrategias optimizadas...')
        if not self.optimization_dir.exists():
            print(f"❌ Directorio no existe: {self.optimization_dir}")
            return available

        for symbol in SYMBOLS:
            available[symbol] = []
            for strategy in AVAILABLE_STRATEGIES:
                study_name = f'{symbol}_{strategy}'
                db_file = self.optimization_dir / f'{study_name}.db'
                is_valid, msg = self.verify_db_file(db_file)
                if is_valid:
                    try:
                        storage_url = f"sqlite:///{db_file}"
                        study = optuna.load_study(study_name=study_name, storage=storage_url)
                        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
                        if completed:
                            available[symbol].append(strategy)
                            best_val = max(completed, key=lambda x: x.value).value
                            print(f"   ✅ {study_name}: {len(completed)} trials, best={best_val:.4f}")
                        else:
                            print(f"   ⚠️ {study_name}: sin trials válidos")
                    except Exception as e:
                        print(f"   ❌ {study_name}: error cargando - {e}")
                else:
                    print(f"   ❌ {study_name}: {msg}")

        total_av = sum(len(v) for v in available.values())
        total_pos = len(SYMBOLS) * len(AVAILABLE_STRATEGIES)
        print('\n📊 RESUMEN:')
        for sym, sts in available.items():
            print(f'   {sym}: {len(sts)}/{len(AVAILABLE_STRATEGIES)} estrategias')
        print(f'   TOTAL: {total_av}/{total_pos} estrategias disponibles\n')
        return available

    # -------------------- BACKTEST --------------------
    def run_backtest_safely(self, symbol, strategy_name, params=None):
        if not BACKTESTER_AVAILABLE:
            print('❌ Backtester no disponible')
            return None
        if strategy_name not in STRATEGY_IMPORTS:
            print(f'❌ Estrategia {strategy_name} no disponible')
            return None
        try:
            if params is None:
                params = self.load_optimized_params_from_db(symbol, strategy_name)
            strategy_class = STRATEGY_IMPORTS[strategy_name]
            strategy_instance = strategy_class(**params)
            backtester = Backtester(symbol=symbol, strategy=strategy_instance)
            report, data_with_indicators = backtester.run(return_data=True)
            trade_log = backtester.get_trade_log()
            return {
                'report': report,
                'trade_log': trade_log,
                'data': data_with_indicators,
                'params': params,
                'strategy_name': strategy_name,
                'symbol': symbol,
                'success': True
            }
        except Exception as e:
            print(f'❌ Error en backtest {symbol}_{strategy_name}: {e}')
            return {'success': False, 'error': str(e), 'symbol': symbol, 'strategy_name': strategy_name}

    # -------------------- EXPORT CAUSAL INPUTS --------------------
    def export_tradelevel_for_causality(self, backtest_results):
        """
        Exporta CSVs para análisis causal:
        - trades.csv: 1 fila por trade con snapshot de variables X en la vela de entrada
        - bars.csv: dataset de barras/indicadores (opcional, auditoría)
        """
        try:
            symbol = backtest_results['symbol']
            strategy = backtest_results['strategy_name']
            trade_log = backtest_results.get('trade_log')
            data = backtest_results.get('data')

            if trade_log is None or trade_log.empty:
                print(f"⚠️ Trade log vacío: {symbol}_{strategy}")
                return None

            time_candidates = ['entry_time', 'Entry time', 'timestamp', 'time']
            pnl_candidates  = ['pnl', 'profit', 'return', 'P&L', 'PnL', 'net_profit', 'profit_loss']

            tcol = next((c for c in time_candidates if c in trade_log.columns), None)
            if tcol is None:
                raise ValueError('No encuentro columna de tiempo de entrada en trade_log.')

            pnl_col = next((c for c in pnl_candidates if c in trade_log.columns), None)
            if pnl_col is None:
                raise ValueError('No encuentro columna de PnL en trade_log.')

            trades = trade_log.copy()
            trades[tcol] = pd.to_datetime(trades[tcol], errors='coerce')
            if 'exit_time' in trades.columns:
                trades['exit_time'] = pd.to_datetime(trades['exit_time'], errors='coerce')

            trades['result'] = (trades[pnl_col].astype(float) > 0).astype(int)

            # Snapshot de X desde data (indicadores al momento de entrada)
            if data is not None and not data.empty:
                # Canonizar tiempo de trades y barras
                trades, tcol_norm = _canonicalize_time_column(trades, prefer_col=tcol, fallback_name="entry_time_norm")
                d, dtime = _canonicalize_time_column(data, prefer_col=tcol, fallback_name="bar_time")

                if d is None or tcol_norm is None:
                    print("⚠️ No se pudo normalizar columnas de tiempo para merge (trades/data).")
                else:
                    # Redondeo/floor para evitar desalineaciones por segundos/milisegundos
                    trades[tcol_norm] = pd.to_datetime(trades[tcol_norm], utc=True).dt.floor('min')
                    d[dtime] = pd.to_datetime(d[dtime], utc=True).dt.floor('min')

                    # Hints para columnas X (numéricas/categóricas)
                    def is_hint(col, hints):
                        cl = str(col).lower()
                        return any(h in cl for h in hints)

                    X_HINTS_NUM = [
                        'spread','atr','atr_','atr14','volatility','volatility_proxy',
                        'ema_','sma_','rsi','bb_','macd','adx','trend_slope','distance_to_sr',
                        'volume','duration'
                    ]
                    X_HINTS_CAT = ['session','session_london','session_ny','session_asia','side']

                    x_num = [c for c in d.columns if is_hint(c, X_HINTS_NUM)]
                    x_cat = [c for c in d.columns if is_hint(c, X_HINTS_CAT)]

                    snap_cols = list(dict.fromkeys([dtime] + x_num + x_cat))
                    valid_snap_cols = [c for c in snap_cols if c in d.columns]

                    if len(valid_snap_cols) > 1:
                        d_sorted = d[valid_snap_cols].drop_duplicates(subset=[dtime]).sort_values(dtime)
                        trades_sorted = trades.sort_values(tcol_norm)

                        # Tolerancia más amplia para M5 y posibles desfasajes
                        merged = pd.merge_asof(
                            trades_sorted,
                            d_sorted,
                            left_on=tcol_norm, right_on=dtime,
                            direction='backward',
                            tolerance=pd.Timedelta('10min')
                        )
                        trades = merged
                        print(f"✅ Merge asof exitoso ({len(valid_snap_cols)} columnas de indicadores).")
                    else:
                        print(f"⚠️ Insuficientes columnas válidas para merge: {valid_snap_cols}")
            else:
                print("⚠️ Data de barras/indicadores vacío: no se agregan snapshots X.")# FIX: Usar la columna de tiempo correcta después del merge
            # Buscar qué columna de tiempo tenemos disponible en el DataFrame resultante
            time_col_to_use = None
            for potential_col in [tcol, 'entry_time', 'timestamp', 'time']:
                if potential_col in trades.columns:
                    time_col_to_use = potential_col
                    break
            
            if time_col_to_use is not None:
                et = pd.to_datetime(trades[time_col_to_use], errors='coerce')
                trades['hour'] = et.dt.hour
                trades['weekday'] = et.dt.weekday
                trades['month'] = et.dt.month
            else:
                print(f"⚠️ No se pudo encontrar columna de tiempo válida para extraer hour/weekday/month")
                print(f"   Columnas disponibles: {list(trades.columns)}")
                # Agregar columnas por defecto para evitar errores
                trades['hour'] = 0
                trades['weekday'] = 0
                trades['month'] = 1

            if 'symbol' not in trades.columns: trades['symbol'] = symbol
            if 'strategy' not in trades.columns: trades['strategy'] = strategy

            causal_dir = self.export_dir / 'causal_inputs'
            causal_dir.mkdir(exist_ok=True)

            trades_path = causal_dir / f'{symbol}_{strategy}_trades.csv'
            bars_path   = causal_dir / f'{symbol}_{strategy}_bars.csv'

            trades.to_csv(trades_path, index=False)
            if data is not None and not data.empty:
                data.to_csv(bars_path, index=False)

            print(f'🧾 Export causal trades: {trades_path}')
            if data is not None and not data.empty:
                print(f'🧾 Export bars/indicators: {bars_path}')
            return str(trades_path)
        except Exception as e:
            print(f'❌ Error exportando causal inputs: {e}')
            traceback.print_exc()  # Agregar traceback para debug
            return None

    # -------------------- ANÁLISIS DE PATRONES --------------------
    def safe_calculate_profit_factor(self, winning_pnl, losing_pnl):
        try:
            if len(winning_pnl) == 0:
                return 0.0
            gross_profit = winning_pnl.sum()
            if len(losing_pnl) == 0 or losing_pnl.sum() >= 0:
                return 999.9 if gross_profit > 0 else 0.0
            gross_loss = abs(losing_pnl.sum())
            if gross_loss == 0:
                return 999.9 if gross_profit > 0 else 0.0
            pf = gross_profit / gross_loss
            return min(pf, 999.9)
        except Exception as e:
            print(f'⚠️ Error calculando Profit Factor: {e}')
            return 0.0

    def analyze_trade_patterns_safely(self, backtest_results):
        if not backtest_results.get('success', False):
            return None
        trade_log = backtest_results.get('trade_log')
        if trade_log is None or trade_log.empty:
            print('⚠️ Trade log vacío o no disponible')
            return None
        symbol = backtest_results['symbol']
        strategy_name = backtest_results['strategy_name']
        print(f'🔍 Analizando {len(trade_log)} trades para {symbol}_{strategy_name}')

        pnl_columns = ['pnl','profit','return','P&L','PnL','net_profit','profit_loss']
        pnl_column = next((c for c in pnl_columns if c in trade_log.columns), None)
        if pnl_column is None:
            print(f'❌ No se encontró columna de PnL. Columnas: {list(trade_log.columns)}')
            return None

        try:
            winning_trades = trade_log[trade_log[pnl_column] > 0].copy()
            losing_trades  = trade_log[trade_log[pnl_column] <= 0].copy()
            total_trades = len(trade_log)
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
            avg_win = winning_trades[pnl_column].mean() if not winning_trades.empty else 0
            avg_loss = losing_trades[pnl_column].mean() if not losing_trades.empty else 0
            largest_win = winning_trades[pnl_column].max() if not winning_trades.empty else 0
            largest_loss = losing_trades[pnl_column].min() if not losing_trades.empty else 0
            total_pnl = trade_log[pnl_column].sum()
            profit_factor = self.safe_calculate_profit_factor(
                winning_trades[pnl_column] if not winning_trades.empty else pd.Series([], dtype=float),
                losing_trades[pnl_column] if not losing_trades.empty else pd.Series([], dtype=float)
            )
            analysis = {
                'symbol': symbol, 'strategy': strategy_name,
                'total_trades': total_trades, 'winning_trades': win_count, 'losing_trades': loss_count,
                'win_rate': round(win_rate, 2), 'avg_win': round(avg_win, 4), 'avg_loss': round(avg_loss, 4),
                'largest_win': round(largest_win, 4), 'largest_loss': round(largest_loss, 4),
                'total_pnl': round(total_pnl, 4), 'profit_factor': round(profit_factor, 3),
                'pnl_column_used': pnl_column, 'trade_log_available': True
            }
            print(f"✅ Análisis completado: PF={profit_factor:.3f}, WR={win_rate:.1f}%")
            return analysis
        except Exception as e:
            print(f'❌ Error en análisis de patrones: {e}')
            traceback.print_exc()
            return None

    # -------------------- ML (opcional demo) --------------------
    def safe_ml_analysis(self, analysis):
        if not ML_AVAILABLE:
            return None
        if not analysis or not analysis.get('trade_log_available', False):
            return None
        try:
            print(f"🤖 Iniciando análisis ML para {analysis['symbol']}_{analysis['strategy']}")
            sample_size = max(50, analysis['total_trades'])
            np.random.seed(42)
            features = pd.DataFrame({
                'rsi_14': np.random.uniform(20, 80, sample_size),
                'atr_14': np.random.uniform(0.0001, 0.0050, sample_size),
                'ema_fast': np.random.uniform(1.1000, 1.3000, sample_size),
                'hour': np.random.randint(0, 24, sample_size),
                'day_of_week': np.random.randint(0, 7, sample_size),
                'volatility': np.random.uniform(0.5, 2.0, sample_size)
            })
            win_prob = analysis['win_rate'] / 100
            targets = np.random.choice([0,1], size=sample_size, p=[1-win_prob, win_prob])

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=42)
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_split=5)
            model.fit(X_train, y_train)
            from sklearn.metrics import accuracy_score
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            fi = pd.DataFrame({'feature': features.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
            print(f'   ✅ ML completado - Accuracy: {accuracy:.3f}')
            return {
                'model_type': 'RandomForest', 'accuracy': round(accuracy, 3),
                'feature_importance': fi.to_dict('records'), 'top_features': fi.head(3)['feature'].tolist(),
                'sample_size': sample_size, 'note': 'Análisis basado en datos sintéticos (demo)'
            }
        except Exception as e:
            print(f'❌ Error en análisis ML: {e}')
            return None

    # -------------------- VISUALIZACIONES --------------------
    def create_safe_visualizations(self, analysis):
        try:
            symbol = analysis['symbol']; strategy = analysis['strategy']
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Análisis: {symbol} - {strategy}', fontsize=16, fontweight='bold')

            # 1. Métricas
            ax1 = axes[0,0]
            metrics = ['Profit Factor','Win Rate %','Total Trades']
            values = [analysis['profit_factor'], analysis['win_rate'], analysis['total_trades']]
            bars = ax1.bar(metrics, values, alpha=0.7)
            ax1.set_title('Métricas Clave'); ax1.set_ylabel('Valor')
            for bar, value in zip(bars, values):
                ax1.text(bar.get_x()+bar.get_width()/2., bar.get_height()*1.02, f'{value:.2f}', ha='center')

            # 2. Win/Loss
            ax2 = axes[0,1]
            win_loss = [analysis['winning_trades'], analysis['losing_trades']]
            ax2.pie(win_loss, labels=['Wins','Losses'], autopct='%1.1f%%', startangle=90)
            ax2.set_title(f'Distribución de Trades (Total: {analysis["total_trades"]})')

            # 3. PnL
            ax3 = axes[1,0]
            pnl_metrics = ['Avg Win','Avg Loss','Total PnL']
            pnl_vals = [analysis['avg_win'], abs(analysis['avg_loss']), analysis['total_pnl']]
            bars = ax3.bar(pnl_metrics, pnl_vals, alpha=0.7)
            ax3.set_title('Análisis PnL'); ax3.set_ylabel('Valor ($)')
            for bar, value in zip(bars, pnl_vals):
                ax3.text(bar.get_x()+bar.get_width()/2., bar.get_height()*1.02, f'${value:.4f}', ha='center')

            # 4. Resumen
            ax4 = axes[1,1]; ax4.axis('off')
            summary = f"""
RESUMEN DE PERFORMANCE

Symbol: {symbol}
Strategy: {strategy}

📊 Métricas:
• Profit Factor: {analysis['profit_factor']:.3f}
• Win Rate: {analysis['win_rate']:.1f}%
• Total Trades: {analysis['total_trades']}

💰 PnL:
• Total PnL: ${analysis['total_pnl']:.4f}
• Avg Win: ${analysis['avg_win']:.4f}
• Avg Loss: ${analysis['avg_loss']:.4f}
"""
            ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10, va='top', family='monospace')
            plt.tight_layout()
            chart_path = self.export_dir / f'{symbol}_{strategy}_analysis.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight'); plt.close()
            print(f'📊 Gráfico guardado: {chart_path}')
            return str(chart_path)
        except Exception as e:
            print(f'❌ Error creando visualizaciones: {e}')
            return None

    # -------------------- REPORTES Y CÓDIGO --------------------
    def export_comprehensive_report(self, analysis, ml_results=None):
        try:
            symbol = analysis['symbol']; strategy = analysis['strategy']
            report = {
                'metadata': {'timestamp': datetime.now().isoformat(), 'version': 'v2.1_causal_export', 'symbol': symbol, 'strategy': strategy},
                'performance': analysis,
                'ml_analysis': ml_results if ml_results else {'available': False},
                'recommendations': self.generate_recommendations(analysis),
                'risk_assessment': self.assess_risk(analysis)
            }
            report_path = self.export_dir / f'{symbol}_{strategy}_report.json'
            with open(report_path, 'w') as f: json.dump(report, f, indent=2, default=str)
            print(f'📋 Reporte guardado: {report_path}')
            return str(report_path)
        except Exception as e:
            print(f'❌ Error exportando reporte: {e}')
            return None

    def generate_recommendations(self, analysis):
        recs = []
        pf = analysis['profit_factor']; wr = analysis['win_rate']; n = analysis['total_trades']
        if pf < 1.0:
            recs.append({'type':'CRITICAL','message':'Estrategia no rentable - Revisar completamente','priority':'HIGH'})
        elif pf < 1.2:
            recs.append({'type':'WARNING','message':'Profit Factor bajo - Optimizar parámetros','priority':'MEDIUM'})
        if wr < 30:
            recs.append({'type':'WARNING','message':'Win Rate muy bajo - Revisar señales de entrada','priority':'HIGH'})
        elif wr > 80:
            recs.append({'type':'INFO','message':'Win Rate alto - Verificar overfitting','priority':'MEDIUM'})
        if n < 30:
            recs.append({'type':'INFO','message':'Pocos trades - Aumentar período de prueba','priority':'LOW'})
        return recs

    def assess_risk(self, analysis):
        pf = analysis['profit_factor']; wr = analysis['win_rate']
        avg_win = analysis['avg_win']; avg_loss = abs(analysis['avg_loss'])
        rr = avg_win / avg_loss if avg_loss > 0 else 0
        if pf < 1.0: lvl='VERY_HIGH'
        elif pf < 1.2: lvl='HIGH'
        elif pf < 1.5: lvl='MEDIUM'
        else: lvl='LOW'
        return {'risk_level': lvl, 'risk_reward_ratio': round(rr,3), 'drawdown_potential': 'HIGH' if wr<40 else 'MEDIUM' if wr<60 else 'LOW',
                'consistency_score': min(100, (pf*wr)/100*100)}

    # -------------------- ORQUESTACIÓN --------------------
    def analyze_all_available_strategies(self):
        print('🚀 INICIANDO ANÁLISIS ULTIMATE V2 (con export causal)')
        print('='*80)
        if not BACKTESTER_AVAILABLE:
            print('❌ Backtester no disponible'); return
        if not AVAILABLE_STRATEGIES:
            print('❌ No hay estrategias disponibles'); return

        available = self.get_available_strategies()
        if not any(available.values()):
            print('❌ No se encontraron estrategias optimizadas'); return

        all_results=[]; stats={'analyzed':0,'successful':0,'failed':0,'ml_completed':0}

        for symbol in SYMBOLS:
            for strategy_name in available.get(symbol, []):
                print(f'\n🔬 ANALIZANDO: {symbol} - {strategy_name}')
                print('-'*50); stats['analyzed']+=1
                try:
                    backtest_results = self.run_backtest_safely(symbol, strategy_name)
                    if not backtest_results or not backtest_results.get('success', False):
                        print(f'❌ Backtest falló para {symbol}_{strategy_name}'); stats['failed']+=1; continue

                    # >>> NUEVO: exportar insumos causales
                    self.export_tradelevel_for_causality(backtest_results)

                    analysis = self.analyze_trade_patterns_safely(backtest_results)
                    if not analysis:
                        print(f'❌ Análisis de patrones falló para {symbol}_{strategy_name}'); stats['failed']+=1; continue

                    ml_results = self.safe_ml_analysis(analysis)
                    if ml_results: stats['ml_completed']+=1

                    chart_path = self.create_safe_visualizations(analysis)
                    report_path = self.export_comprehensive_report(analysis, ml_results)
                    code_path = None
                    if self.generate_code:
                        code_path = self.generate_strategy_code_safely(symbol, strategy_name, backtest_results.get('params', {}), analysis)

                    stats['successful']+=1
                    result={'symbol':symbol,'strategy':strategy_name,'analysis':analysis,'ml_results':ml_results,
                            'chart_path':chart_path,'report_path':report_path,'code_path':code_path,'success':True}
                    all_results.append(result)
                    print(f"✅ COMPLETADO: {symbol}_{strategy_name} | PF: {analysis['profit_factor']:.3f}, WR: {analysis['win_rate']:.1f}%")

                except Exception as e:
                    print(f'❌ Error procesando {symbol}_{strategy_name}: {e}'); stats['failed']+=1; continue

        self.generate_final_summary(all_results, stats)

    def generate_strategy_code_safely(self, symbol, strategy_name, params, analysis):
        try:
            class_name = ''.join(w.capitalize() for w in strategy_name.split('_'))
            code = f"""# AUTO-GENERATED OPTIMIZED STRATEGY
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Symbol: {symbol} | Strategy: {strategy_name}
# Performance: PF={analysis["profit_factor"]:.3f}, WR={analysis["win_rate"]:.1f}%

class Optimized{class_name}{symbol}:
    def __init__(self):
        self.symbol = "{symbol}"
        self.strategy_name = "{strategy_name}"
        self.params = {params}
        self.historical_performance = {{
            "profit_factor": {analysis["profit_factor"]},
            "win_rate": {analysis["win_rate"]},
            "total_trades": {analysis["total_trades"]},
            "total_pnl": {analysis["total_pnl"]},
            "avg_win": {analysis["avg_win"]},
            "avg_loss": {analysis["avg_loss"]}
        }}
    def get_params(self): return self.params.copy()
    def get_performance(self): return self.historical_performance.copy()
    def get_signal(self, data): return 0  # TODO
    def __str__(self): return f"Optimized{class_name}{symbol} (PF: {analysis['profit_factor']:.3f})"

def create_strategy():
    return Optimized{class_name}{symbol}()
"""
            filename = self.auto_optimization_dir / f'{symbol}_{strategy_name}_optimized.py'
            with open(filename, 'w', encoding='utf-8') as f: f.write(code)
            print(f'🔧 Código generado: {filename}')
            return str(filename)
        except Exception as e:
            print(f'❌ Error generando código: {e}')
            return None

    def generate_final_summary(self, results, stats):
        print('\n'+'='*80)
        print('📊 RESUMEN FINAL DEL ANÁLISIS')
        print('='*80)
        print('📈 Estadísticas:')
        print(f"   • Total analizadas: {stats['analyzed']}")
        print(f"   • Exitosas: {stats['successful']}")
        print(f"   • Fallidas: {stats['failed']}")
        print(f"   • Con análisis ML: {stats['ml_completed']}")
        if not results:
            print('❌ No se generaron resultados válidos'); return
        try:
            sorted_results = sorted(results, key=lambda x: x['analysis']['profit_factor'], reverse=True)
            print('\n🏆 TOP ESTRATEGIAS POR PROFIT FACTOR:')
            for i, r in enumerate(sorted_results[:10], 1):
                a=r['analysis']
                print(f"   {i:2d}. {r['symbol']:>7}_{r['strategy']:<20} | PF:{a['profit_factor']:6.3f} | WR:{a['win_rate']:5.1f}% | Trades:{a['total_trades']:4d}")
            all_pf=[r['analysis']['profit_factor'] for r in results]
            all_wr=[r['analysis']['win_rate'] for r in results]
            all_tr=[r['analysis']['total_trades'] for r in results]
            print('\n📊 ESTADÍSTICAS GENERALES:')
            print(f'   • Profit Factor promedio: {np.mean(all_pf):.3f}')
            print(f'   • Win Rate promedio: {np.mean(all_wr):.1f}%')
            print(f'   • Trades promedio: {np.mean(all_tr):.0f}')
            print(f'   • Estrategias rentables: {len([pf for pf in all_pf if pf>1.0])}/{len(all_pf)}')
            self.export_summary_report(results, stats, sorted_results)
        except Exception as e:
            print(f'❌ Error generando resumen: {e}')
        print('\n🎉 ANÁLISIS COMPLETADO!')
        print('📁 Archivos en:')
        print(f'   📊 Reportes: {self.export_dir}')
        print(f'   🔧 Código: {self.auto_optimization_dir}')

    def export_summary_report(self, results, stats, sorted_results):
        try:
            summary = {
                'metadata': {'timestamp': datetime.now().isoformat(), 'version': 'v2.1_causal_export', 'total_strategies': len(results)},
                'execution_stats': stats,
                'top_strategies': [
                    {'rank': i+1, 'name': f"{r['symbol']}_{r['strategy']}", 'profit_factor': r['analysis']['profit_factor'],
                     'win_rate': r['analysis']['win_rate'], 'total_trades': r['analysis']['total_trades'],
                     'total_pnl': r['analysis']['total_pnl']}
                    for i, r in enumerate(sorted_results[:10])
                ],
                'overall_statistics': {
                    'avg_profit_factor': float(np.mean([r['analysis']['profit_factor'] for r in results])),
                    'avg_win_rate': float(np.mean([r['analysis']['win_rate'] for r in results])),
                    'profitable_strategies': int(len([r for r in results if r['analysis']['profit_factor']>1.0])),
                    'total_strategies': int(len(results))
                }
            }
            summary_path = self.export_dir / 'analysis_summary.json'
            with open(summary_path, 'w') as f: json.dump(summary, f, indent=2, default=str)
            print(f'📋 Resumen general guardado: {summary_path}')
        except Exception as e:
            print(f'❌ Error exportando resumen: {e}')

def main():
    print('🚀 ULTIMATE TRADE ANALYZER V2 (con export causal)')
    print('='*80)
    try:
        optimization_dir = Path('optimization_studies')
        if not optimization_dir.exists():
            print(f'❌ DIRECTORIO NO ENCONTRADO: {optimization_dir}'); return
        db_files = list(optimization_dir.glob('*.db'))
        if not db_files:
            print(f'❌ NO SE ENCONTRARON ARCHIVOS .db en {optimization_dir}'); return
        print(f'✅ Encontrados {len(db_files)} archivos de optimización')
        analyzer = UltimateTradeAnalyzerV2()
        analyzer.analyze_all_available_strategies()
        print('\n'+'='*80)
        print('✅ EJECUCIÓN COMPLETADA')
    except KeyboardInterrupt:
        print('\n⚠️ Interrumpido por el usuario')
    except Exception as e:
        print('\n❌ ERROR CRÍTICO DURANTE EJECUCIÓN:')
        print(f'   {str(e)}')
        traceback.print_exc()

if __name__ == '__main__':
    main()
            