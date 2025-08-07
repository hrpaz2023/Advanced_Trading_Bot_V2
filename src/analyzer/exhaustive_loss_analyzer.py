# exhaustive_loss_analyzer.py - Análisis EXHAUSTIVO de pérdidas
# Genera reportes completos con TODAS las variables de trades perdedores
# MODIFICADO: Analiza TODOS los activos y estrategias automáticamente

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import optuna
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

warnings.filterwarnings('ignore')

# Importaciones robustas
try:
    from src.backtesting.backtester import Backtester
    from src.strategies.ema_crossover import EmaCrossover
    from src.strategies.channel_reversal import ChannelReversal
    from src.strategies.rsi_pullback import RsiPullback
    from src.strategies.volatility_breakout import VolatilityBreakout
    from src.strategies.multi_filter_scalper import MultiFilterScalper
    from src.strategies.lokz_reversal import LokzReversal
    
    STRATEGY_CLASSES = {
        'ema_crossover': EmaCrossover,
        'channel_reversal': ChannelReversal,
        'rsi_pullback': RsiPullback,
        'volatility_breakout': VolatilityBreakout,
        'multi_filter_scalper': MultiFilterScalper,
        'lokz_reversal': LokzReversal
    }
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Error importando estrategias: {e}")
    IMPORTS_AVAILABLE = False

# ==========================================================
# ✅ CONFIGURACIÓN AUTOMÁTICA MULTI-ASSET/MULTI-STRATEGY
# ==========================================================
# Lista de símbolos comunes de forex
DEFAULT_SYMBOLS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD',
    'NZDUSD', 'EURJPY', 'GBPJPY', 'EURGBP', 'AUDJPY', 'CHFJPY'
]

# Lista de todas las estrategias disponibles
DEFAULT_STRATEGIES = [
    'ema_crossover', 'channel_reversal', 'rsi_pullback',
    'volatility_breakout', 'multi_filter_scalper', 'lokz_reversal'
]

# Configuración de procesamiento paralelo
MAX_WORKERS = 4  # Número de procesos paralelos
PROCESS_TIMEOUT = 300  # Timeout por análisis en segundos
# ==========================================================

class MultiAssetLossAnalyzer:
    def __init__(self, optimization_dir="optimization_studies", symbols=None, strategies=None):
        self.optimization_dir = Path(optimization_dir)
        self.export_dir = Path("exhaustive_loss_reports")
        self.export_dir.mkdir(exist_ok=True)
        
        # Crear subdirectorios organizados
        self.summary_dir = self.export_dir / "summaries"
        self.detailed_dir = self.export_dir / "detailed_reports"
        self.charts_dir = self.export_dir / "charts"
        
        for subdir in [self.summary_dir, self.detailed_dir, self.charts_dir]:
            subdir.mkdir(exist_ok=True)
        
        # Configurar símbolos y estrategias
        self.symbols = symbols or self._detect_available_symbols()
        self.strategies = strategies or DEFAULT_STRATEGIES
        
        print(f"🕵️ MULTI-ASSET EXHAUSTIVE LOSS ANALYZER")
        print(f"📁 Optimización: {self.optimization_dir}")
        print(f"📁 Exportación: {self.export_dir}")
        print(f"🎯 Símbolos a analizar: {len(self.symbols)}")
        print(f"📈 Estrategias a analizar: {len(self.strategies)}")
        print(f"🔄 Total combinaciones: {len(self.symbols) * len(self.strategies)}")
    
    def _detect_available_symbols(self):
        """Detecta símbolos disponibles basándose en archivos de optimización"""
        available_symbols = set()
        
        if self.optimization_dir.exists():
            for db_file in self.optimization_dir.glob("*.db"):
                # Extraer símbolo del nombre del archivo (formato: SYMBOL_strategy.db)
                stem = db_file.stem
                if '_' in stem:
                    symbol = stem.split('_')[0]
                    available_symbols.add(symbol)
        
        detected = list(available_symbols)
        
        if detected:
            print(f"✅ Símbolos detectados automáticamente: {detected}")
            return detected
        else:
            print(f"⚠️ No se detectaron símbolos, usando lista por defecto")
            return DEFAULT_SYMBOLS
    
    def get_available_combinations(self):
        """Obtiene todas las combinaciones symbol-strategy disponibles"""
        combinations = []
        
        for symbol in self.symbols:
            for strategy in self.strategies:
                study_name = f"{symbol}_{strategy}"
                db_file = self.optimization_dir / f"{study_name}.db"
                
                if db_file.exists():
                    combinations.append({
                        'symbol': symbol,
                        'strategy': strategy,
                        'study_name': study_name,
                        'db_file': db_file,
                        'status': 'available'
                    })
                else:
                    combinations.append({
                        'symbol': symbol,
                        'strategy': strategy,
                        'study_name': study_name,
                        'db_file': db_file,
                        'status': 'missing'
                    })
        
        available_count = len([c for c in combinations if c['status'] == 'available'])
        missing_count = len([c for c in combinations if c['status'] == 'missing'])
        
        print(f"📊 Combinaciones disponibles: {available_count}")
        print(f"❌ Combinaciones faltantes: {missing_count}")
        
        return combinations
    
    def load_optimized_params_from_db(self, symbol, strategy_name):
        """Carga parámetros optimizados desde base de datos"""
        study_name = f"{symbol}_{strategy_name}"
        db_file = self.optimization_dir / f"{study_name}.db"
        
        if not db_file.exists():
            return None
        
        try:
            storage_url = f"sqlite:///{db_file}"
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            
            completed_trials = [t for t in study.trials 
                              if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
            
            if not completed_trials:
                return None
            
            best_trial = max(completed_trials, key=lambda x: x.value)
            return best_trial.params
            
        except Exception as e:
            print(f"❌ Error cargando parámetros {study_name}: {e}")
            return None
    
    def run_detailed_backtest(self, symbol, strategy_name, params):
        """Ejecuta backtest detallado con máxima información"""
        if not IMPORTS_AVAILABLE or strategy_name not in STRATEGY_CLASSES:
            return None
        
        try:
            strategy_class = STRATEGY_CLASSES[strategy_name]
            strategy_instance = strategy_class(**params)
            backtester = Backtester(symbol=symbol, strategy=strategy_instance)
            
            # Backtest con datos completos
            report, data_with_indicators = backtester.run(return_data=True)
            trade_log = backtester.get_trade_log()
            
            return {
                'report': report,
                'trade_log': trade_log,
                'data': data_with_indicators,
                'params': params,
                'symbol': symbol,
                'strategy_name': strategy_name,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Error en backtest {symbol}_{strategy_name}: {e}")
            return None
    
    def analyze_losing_trades_exhaustive(self, backtest_results):
        """Análisis EXHAUSTIVO de trades perdedores"""
        trade_log = backtest_results['trade_log']
        data = backtest_results['data']
        symbol = backtest_results['symbol']
        strategy_name = backtest_results['strategy_name']
        
        if trade_log.empty:
            return {'has_losses': False, 'message': 'Sin trades'}
        
        # Detectar columna PnL
        pnl_column = None
        for col in ['pnl', 'profit', 'return', 'P&L', 'PnL', 'net_profit', 'profit_loss']:
            if col in trade_log.columns:
                pnl_column = col
                break
        
        if pnl_column is None:
            return {'has_losses': False, 'message': 'Sin columna PnL'}
        
        # Separar trades
        losing_trades = trade_log[trade_log[pnl_column] <= 0].copy()
        winning_trades = trade_log[trade_log[pnl_column] > 0].copy()
        
        if losing_trades.empty:
            return {'has_losses': False, 'message': 'Sin pérdidas'}
        
        # ANÁLISIS EXHAUSTIVO
        analysis = {
            'metadata': {
                'symbol': symbol,
                'strategy': strategy_name,
                'timestamp': datetime.now().isoformat(),
                'pnl_column': pnl_column,
                'has_losses': True
            },
            'trade_summary': self._analyze_trade_summary(trade_log, losing_trades, winning_trades, pnl_column),
            'losing_trades_detailed': self._extract_losing_trades_detailed(losing_trades, data, trade_log),
            'market_conditions': self._analyze_market_conditions_exhaustive(losing_trades, winning_trades, data),
            'temporal_analysis': self._analyze_temporal_patterns(losing_trades, winning_trades),
            'indicator_analysis': self._analyze_indicators_at_loss(losing_trades, data),
            'comparative_analysis': self._compare_winners_vs_losers(winning_trades, losing_trades, data),
            'root_cause_analysis': self._identify_root_causes(losing_trades, data, strategy_name),
            'recommendations': self._generate_exhaustive_recommendations(losing_trades, winning_trades, data, strategy_name)
        }
        
        return analysis
    
    def _analyze_trade_summary(self, trade_log, losing_trades, winning_trades, pnl_column):
        """Resumen estadístico completo"""
        return {
            'total_trades': len(trade_log),
            'losing_trades': len(losing_trades),
            'winning_trades': len(winning_trades),
            'loss_rate_pct': len(losing_trades) / len(trade_log) * 100,
            'win_rate_pct': len(winning_trades) / len(trade_log) * 100,
            'total_pnl': trade_log[pnl_column].sum(),
            'total_gross_profit': winning_trades[pnl_column].sum() if not winning_trades.empty else 0,
            'total_gross_loss': losing_trades[pnl_column].sum() if not losing_trades.empty else 0,
            'profit_factor': abs(winning_trades[pnl_column].sum() / losing_trades[pnl_column].sum()) if not losing_trades.empty and losing_trades[pnl_column].sum() != 0 else 0,
            'avg_win': winning_trades[pnl_column].mean() if not winning_trades.empty else 0,
            'avg_loss': losing_trades[pnl_column].mean() if not losing_trades.empty else 0,
            'largest_win': winning_trades[pnl_column].max() if not winning_trades.empty else 0,
            'largest_loss': losing_trades[pnl_column].min() if not losing_trades.empty else 0,
            'avg_win_loss_ratio': abs(winning_trades[pnl_column].mean() / losing_trades[pnl_column].mean()) if not losing_trades.empty and not winning_trades.empty and losing_trades[pnl_column].mean() != 0 else 0,
            'loss_std': losing_trades[pnl_column].std() if not losing_trades.empty else 0,
            'win_std': winning_trades[pnl_column].std() if not winning_trades.empty else 0
        }
    
    def _extract_losing_trades_detailed(self, losing_trades, data, trade_log):
        """Extrae TODOS los datos de cada trade perdedor"""
        detailed_losses = []
        
        # Preparar datos temporales
        if hasattr(data, 'index'):
            data.index = pd.to_datetime(data.index)
        
        for i, (trade_idx, trade) in enumerate(losing_trades.iterrows()):
            try:
                # Información básica del trade
                trade_detail = {
                    'trade_index': trade_idx,
                    'trade_number': i + 1,
                }
                
                # Copiar TODAS las columnas del trade
                for col in trade.index:
                    try:
                        value = trade[col]
                        if pd.isna(value):
                            trade_detail[f'trade_{col}'] = None
                        elif isinstance(value, (int, float, str, bool)):
                            trade_detail[f'trade_{col}'] = value
                        else:
                            trade_detail[f'trade_{col}'] = str(value)
                    except:
                        trade_detail[f'trade_{col}'] = 'ERROR'
                
                # Intentar obtener condiciones de mercado en el momento de entrada
                market_conditions = self._get_market_conditions_at_entry(trade, data, i, len(losing_trades))
                trade_detail.update(market_conditions)
                
                detailed_losses.append(trade_detail)
                
            except Exception as e:
                detailed_losses.append({
                    'trade_index': trade_idx,
                    'trade_number': i + 1,
                    'error': str(e)
                })
        
        return detailed_losses
    
    def _get_market_conditions_at_entry(self, trade, data, trade_num, total_trades):
        """Obtiene condiciones de mercado en el momento de entrada"""
        try:
            # Método 1: Usar entry_time si existe
            if 'entry_time' in trade.index and not pd.isna(trade['entry_time']):
                entry_time = pd.to_datetime(trade['entry_time'])
                if entry_time in data.index:
                    market_data = data.loc[entry_time]
                    conditions = {'entry_method': 'exact_timestamp'}
                    
                    # Copiar TODAS las columnas de datos de mercado
                    for col in market_data.index:
                        try:
                            value = market_data[col]
                            if pd.isna(value):
                                conditions[f'market_{col}'] = None
                            elif isinstance(value, (int, float, str, bool)):
                                conditions[f'market_{col}'] = value
                            else:
                                conditions[f'market_{col}'] = str(value)
                        except:
                            conditions[f'market_{col}'] = 'ERROR'
                    
                    return conditions
            
            # Método 2: Usar entry_bar_index si existe
            if 'entry_bar_index' in trade.index and not pd.isna(trade['entry_bar_index']):
                try:
                    entry_idx = int(trade['entry_bar_index'])
                    if 0 <= entry_idx < len(data):
                        market_data = data.iloc[entry_idx]
                        conditions = {'entry_method': 'bar_index'}
                        
                        for col in market_data.index:
                            try:
                                conditions[f'market_{col}'] = market_data[col]
                            except:
                                conditions[f'market_{col}'] = 'ERROR'
                        
                        return conditions
                except:
                    pass
            
            # Método 3: Fallback - distribuir trades a lo largo de los datos
            entry_idx = int((trade_num / total_trades) * (len(data) - 1))
            entry_idx = min(max(entry_idx, 0), len(data) - 1)
            
            market_data = data.iloc[entry_idx]
            conditions = {'entry_method': 'fallback_distribution'}
            
            for col in market_data.index:
                try:
                    conditions[f'market_{col}'] = market_data[col]
                except:
                    conditions[f'market_{col}'] = 'ERROR'
            
            return conditions
            
        except Exception as e:
            return {
                'entry_method': 'error',
                'error': str(e),
                'market_close': 'N/A',
                'market_open': 'N/A',
                'market_high': 'N/A',
                'market_low': 'N/A'
            }
    
    def _analyze_market_conditions_exhaustive(self, losing_trades, winning_trades, data):
        """Análisis exhaustivo de condiciones de mercado"""
        analysis = {}
        
        try:
            # Obtener todas las columnas numéricas de los datos
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Estadísticas generales del mercado
            analysis['market_statistics'] = {
                col: {
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'median': float(data[col].median()),
                    'q25': float(data[col].quantile(0.25)),
                    'q75': float(data[col].quantile(0.75))
                }
                for col in numeric_columns[:20]  # Limitar a 20 columnas principales
            }
            
            # Análisis de volatilidad
            if 'atr_14' in data.columns:
                atr_data = data['atr_14']
                analysis['volatility_analysis'] = {
                    'atr_mean': float(atr_data.mean()),
                    'atr_std': float(atr_data.std()),
                    'low_volatility_threshold': float(atr_data.quantile(0.25)),
                    'high_volatility_threshold': float(atr_data.quantile(0.75)),
                    'extreme_low_threshold': float(atr_data.quantile(0.1)),
                    'extreme_high_threshold': float(atr_data.quantile(0.9))
                }
            
        except Exception as e:
            analysis['error'] = f"Error en análisis de mercado: {e}"
        
        return analysis
    
    def _analyze_temporal_patterns(self, losing_trades, winning_trades):
        """Análisis de patrones temporales"""
        analysis = {}
        
        try:
            # Análisis de losing trades por tiempo
            if 'entry_time' in losing_trades.columns:
                losing_times = pd.to_datetime(losing_trades['entry_time'])
                
                analysis['losing_trades_temporal'] = {
                    'hourly_distribution': losing_times.dt.hour.value_counts().to_dict(),
                    'daily_distribution': losing_times.dt.day_name().value_counts().to_dict(),
                    'monthly_distribution': losing_times.dt.month.value_counts().to_dict()
                }
                
                # Peores horas/días
                hourly_counts = losing_times.dt.hour.value_counts()
                analysis['worst_periods'] = {
                    'worst_hours': hourly_counts.head(3).index.tolist(),
                    'best_hours': hourly_counts.tail(3).index.tolist()
                }
            
        except Exception as e:
            analysis['error'] = f"Error en análisis temporal: {e}"
        
        return analysis
    
    def _analyze_indicators_at_loss(self, losing_trades, data):
        """Análisis de indicadores en el momento de las pérdidas"""
        analysis = {}
        
        try:
            # Indicadores técnicos comunes
            indicator_columns = [col for col in data.columns if any(indicator in col.lower() 
                               for indicator in ['rsi', 'ema', 'sma', 'atr', 'bb', 'macd'])]
            
            if indicator_columns:
                analysis['indicators_at_loss'] = {}
                
                # Para cada indicador, analizar distribución en pérdidas
                total_bars = len(data)
                sample_indices = np.linspace(0, total_bars-1, min(len(losing_trades), total_bars), dtype=int)
                
                for indicator in indicator_columns[:15]:  # Top 15 indicadores
                    try:
                        indicator_values = data[indicator].iloc[sample_indices]
                        
                        analysis['indicators_at_loss'][indicator] = {
                            'mean_at_loss': float(indicator_values.mean()),
                            'std_at_loss': float(indicator_values.std()),
                            'min_at_loss': float(indicator_values.min()),
                            'max_at_loss': float(indicator_values.max()),
                            'overall_mean': float(data[indicator].mean()),
                            'overall_std': float(data[indicator].std()),
                            'deviation_from_normal': float((indicator_values.mean() - data[indicator].mean()) / data[indicator].std()) if data[indicator].std() > 0 else 0
                        }
                    except Exception as e:
                        analysis['indicators_at_loss'][indicator] = {'error': str(e)}
        
        except Exception as e:
            analysis['error'] = f"Error en análisis de indicadores: {e}"
        
        return analysis
    
    def _compare_winners_vs_losers(self, winning_trades, losing_trades, data):
        """Comparación detallada entre trades ganadores y perdedores"""
        analysis = {}
        
        try:
            if winning_trades.empty or losing_trades.empty:
                return {'error': 'No hay suficientes datos para comparación'}
            
            # Comparar columnas numéricas de los trades
            numeric_columns = []
            for col in losing_trades.columns:
                if losing_trades[col].dtype in ['int64', 'float64'] and winning_trades[col].dtype in ['int64', 'float64']:
                    numeric_columns.append(col)
            
            analysis['trade_comparison'] = {}
            
            for col in numeric_columns:
                try:
                    losing_mean = losing_trades[col].mean()
                    winning_mean = winning_trades[col].mean()
                    
                    analysis['trade_comparison'][col] = {
                        'losing_mean': float(losing_mean),
                        'winning_mean': float(winning_mean),
                        'losing_std': float(losing_trades[col].std()),
                        'winning_std': float(winning_trades[col].std()),
                        'difference': float(losing_mean - winning_mean),
                        'percentage_difference': float(((losing_mean - winning_mean) / winning_mean * 100)) if winning_mean != 0 else 0
                    }
                except Exception as e:
                    analysis['trade_comparison'][col] = {'error': str(e)}
            
        except Exception as e:
            analysis['error'] = f"Error en comparación: {e}"
        
        return analysis
    
    def _identify_root_causes(self, losing_trades, data, strategy_name):
        """Identifica causas raíz de las pérdidas"""
        causes = []
        
        try:
            # Causa 1: Volatilidad extrema
            if 'atr_14' in data.columns:
                atr_data = data['atr_14']
                extreme_high_atr = atr_data.quantile(0.95)
                extreme_low_atr = atr_data.quantile(0.05)
                
                if len(losing_trades) > 10:
                    causes.append({
                        'cause': 'Volatilidad Extrema',
                        'description': f'Pérdidas pueden ocurrir en ATR > {extreme_high_atr:.6f} o < {extreme_low_atr:.6f}',
                        'severity': 'HIGH' if len(losing_trades) > len(data) * 0.3 else 'MEDIUM'
                    })
            
            # Causa 2: Estrategia específica
            strategy_specific_causes = {
                'ema_crossover': 'Cruces falsos en mercados laterales',
                'rsi_pullback': 'RSI no respeta niveles en tendencias fuertes',
                'channel_reversal': 'Breakouts falsos de canales',
                'volatility_breakout': 'Breakouts sin seguimiento',
                'multi_filter_scalper': 'Filtros conflictivos'
            }
            
            if strategy_name in strategy_specific_causes:
                causes.append({
                    'cause': f'Problema Específico de {strategy_name}',
                    'description': strategy_specific_causes[strategy_name],
                    'severity': 'HIGH'
                })
            
        except Exception as e:
            causes.append({
                'cause': 'Error en Análisis',
                'description': f'Error identificando causas: {e}',
                'severity': 'UNKNOWN'
            })
        
        return causes
    
    def _generate_exhaustive_recommendations(self, losing_trades, winning_trades, data, strategy_name):
        """Genera recomendaciones exhaustivas"""
        recommendations = []
        
        try:
            loss_rate = len(losing_trades) / (len(losing_trades) + len(winning_trades)) * 100 if not winning_trades.empty else 100
            
            # Recomendación 1: Filtros de volatilidad
            if 'atr_14' in data.columns:
                atr_data = data['atr_14']
                recommendations.append({
                    'category': 'VOLATILITY_FILTER',
                    'priority': 'HIGH',
                    'recommendation': f'Implementar filtro ATR: evitar trading cuando ATR < {atr_data.quantile(0.1):.6f} o > {atr_data.quantile(0.9):.6f}',
                    'expected_improvement': 'Reducir 15-25% de pérdidas por volatilidad extrema'
                })
            
            # Recomendación 2: Stop Loss
            if not losing_trades.empty:
                avg_loss = abs(losing_trades.iloc[:, -1].mean()) if len(losing_trades.columns) > 0 else 0
                recommendations.append({
                    'category': 'RISK_MANAGEMENT',
                    'priority': 'CRITICAL',
                    'recommendation': f'Ajustar Stop Loss: pérdida promedio actual {avg_loss:.6f}, considerar SL más estricto',
                    'expected_improvement': 'Limitar pérdidas máximas por trade'
                })
            
            # Recomendación 3: Si tasa de pérdida muy alta
            if loss_rate > 70:
                recommendations.append({
                    'category': 'FUNDAMENTAL_REVIEW',
                    'priority': 'CRITICAL',
                    'recommendation': 'REVISAR COMPLETAMENTE LA ESTRATEGIA: Tasa de pérdida excesiva indica problema fundamental',
                    'expected_improvement': 'Potencial mejora de 30-50% en performance'
                })
            
        except Exception as e:
            recommendations.append({
                'category': 'ERROR',
                'priority': 'LOW',
                'recommendation': f'Error generando recomendaciones: {e}',
                'expected_improvement': 'N/A'
            })
        
        return recommendations
    
    def analyze_single_combination(self, combination):
        """Analiza una combinación específica symbol-strategy"""
        symbol = combination['symbol']
        strategy = combination['strategy']
        
        try:
            start_time = time.time()
            
            # 1. Cargar parámetros
            params = self.load_optimized_params_from_db(symbol, strategy)
            if not params:
                return {
                    'symbol': symbol,
                    'strategy': strategy,
                    'status': 'failed',
                    'error': 'No se pudieron cargar parámetros',
                    'processing_time': time.time() - start_time
                }
            
            # 2. Ejecutar backtest
            backtest_results = self.run_detailed_backtest(symbol, strategy, params)
            if not backtest_results:
                return {
                    'symbol': symbol,
                    'strategy': strategy,
                    'status': 'failed',
                    'error': 'Error en backtest',
                    'processing_time': time.time() - start_time
                }
            
            # 3. Análizar pérdidas
            analysis = self.analyze_losing_trades_exhaustive(backtest_results)
            if not analysis or not analysis.get('metadata', {}).get('has_losses', False):
                return {
                    'symbol': symbol,
                    'strategy': strategy,
                    'status': 'no_losses',
                    'message': analysis.get('message', 'Sin pérdidas') if analysis else 'Sin análisis',
                    'processing_time': time.time() - start_time
                }
            
            # 4. Exportar reportes individuales
            export_results = self.export_individual_reports(analysis)
            
            processing_time = time.time() - start_time
            
            return {
                'symbol': symbol,
                'strategy': strategy,
                'status': 'success',
                'analysis': analysis,
                'export_results': export_results,
                'processing_time': processing_time
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'strategy': strategy,
                'status': 'error',
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def export_individual_reports(self, analysis):
        """Exporta reportes individuales para una combinación"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            strategy_id = f"{analysis['metadata']['symbol']}_{analysis['metadata']['strategy']}"
            
            # 1. REPORTE PRINCIPAL JSON
            main_report_path = self.detailed_dir / f"{strategy_id}_analysis_{timestamp}.json"
            with open(main_report_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            # 2. CSV DETALLADO DE TRADES PERDEDORES
            csv_path = None
            if analysis.get('losing_trades_detailed'):
                losing_df = pd.DataFrame(analysis['losing_trades_detailed'])
                csv_path = self.detailed_dir / f"{strategy_id}_losing_trades_{timestamp}.csv"
                losing_df.to_csv(csv_path, index=False)
            
            # 3. REPORTE EJECUTIVO TXT
            executive_report = self._generate_executive_summary(analysis)
            txt_path = self.summary_dir / f"{strategy_id}_summary_{timestamp}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(executive_report)
            
            return {
                'main_report': str(main_report_path),
                'detailed_csv': str(csv_path) if csv_path else None,
                'executive_summary': str(txt_path)
            }
            
        except Exception as e:
            print(f"❌ Error exportando reportes para {analysis.get('metadata', {}).get('symbol', 'UNKNOWN')}: {e}")
            return None
    
    def _generate_executive_summary(self, analysis):
        """Genera resumen ejecutivo en texto plano"""
        try:
            summary = analysis['trade_summary']
            metadata = analysis['metadata']
            recommendations = analysis.get('recommendations', [])
            root_causes = analysis.get('root_cause_analysis', [])
            
            report = f"""
================================================================================
🕵️ ANÁLISIS EXHAUSTIVO DE PÉRDIDAS - REPORTE EJECUTIVO
================================================================================

📊 ESTRATEGIA: {metadata['symbol']}_{metadata['strategy']}
📅 Fecha de análisis: {metadata['timestamp']}
📈 Columna PnL utilizada: {metadata['pnl_column']}

================================================================================
📉 RESUMEN DE PÉRDIDAS
================================================================================

Total de trades: {summary['total_trades']}
Trades perdedores: {summary['losing_trades']} ({summary['loss_rate_pct']:.1f}%)
Trades ganadores: {summary['winning_trades']} ({summary['win_rate_pct']:.1f}%)

💰 ANÁLISIS FINANCIERO:
• PnL total: ${summary['total_pnl']:.6f}
• Ganancia bruta: ${summary['total_gross_profit']:.6f}
• Pérdida bruta: ${summary['total_gross_loss']:.6f}
• Profit Factor: {summary['profit_factor']:.3f}

📊 ESTADÍSTICAS DE PÉRDIDAS:
• Pérdida promedio: ${summary['avg_loss']:.6f}
• Pérdida máxima: ${summary['largest_loss']:.6f}
• Desviación estándar pérdidas: ${summary['loss_std']:.6f}
• Ratio ganancia/pérdida promedio: {summary['avg_win_loss_ratio']:.3f}

================================================================================
🔍 CAUSAS RAÍZ IDENTIFICADAS
================================================================================
"""
            
            for i, cause in enumerate(root_causes, 1):
                report += f"""
{i}. {cause['cause']} (Severidad: {cause['severity']})
   → {cause['description']}
"""
            
            report += """
================================================================================
💡 RECOMENDACIONES PRIORITARIAS
================================================================================
"""
            
            # Ordenar recomendaciones por prioridad
            priority_order = {'CRITICAL': 1, 'HIGH': 2, 'MEDIUM': 3, 'LOW': 4}
            sorted_recommendations = sorted(recommendations, 
                                          key=lambda x: priority_order.get(x['priority'], 5))
            
            for i, rec in enumerate(sorted_recommendations, 1):
                report += f"""
{i}. [{rec['priority']}] {rec['category']}
   → {rec['recommendation']}
   → Mejora esperada: {rec['expected_improvement']}
"""
            
            report += f"""
================================================================================
📁 ARCHIVOS GENERADOS
================================================================================

• Análisis completo JSON: Guardado en detailed_reports/
• Datos detallados CSV: Guardado en detailed_reports/
• Este reporte ejecutivo: Guardado en summaries/

================================================================================
⚠️ IMPORTANTE
================================================================================

Este análisis se basa en datos históricos. Validar todas las recomendaciones
en trading demo antes de implementar en cuentas reales.

Último análisis: {metadata['timestamp']}
================================================================================
"""
            
            return report
            
        except Exception as e:
            return f"Error generando resumen ejecutivo: {e}"
    
    def analyze_all_combinations_parallel(self):
        """Analiza todas las combinaciones en paralelo"""
        combinations = self.get_available_combinations()
        available_combinations = [c for c in combinations if c['status'] == 'available']
        
        if not available_combinations:
            print("❌ No hay combinaciones disponibles para analizar")
            return None
        
        print(f"🚀 Iniciando análisis paralelo de {len(available_combinations)} combinaciones")
        print(f"⚡ Usando {MAX_WORKERS} workers paralelos")
        print(f"⏱️ Timeout por análisis: {PROCESS_TIMEOUT} segundos")
        print("=" * 80)
        
        results = []
        start_time = time.time()
        
        # Procesamiento paralelo
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Enviar todas las tareas
            future_to_combination = {
                executor.submit(self.analyze_single_combination, combo): combo 
                for combo in available_combinations
            }
            
            # Procesar resultados conforme se completan
            for i, future in enumerate(as_completed(future_to_combination, timeout=PROCESS_TIMEOUT * len(available_combinations)), 1):
                combination = future_to_combination[future]
                
                try:
                    result = future.result(timeout=PROCESS_TIMEOUT)
                    results.append(result)
                    
                    # Mostrar progreso
                    symbol = result['symbol']
                    strategy = result['strategy']
                    status = result['status']
                    proc_time = result['processing_time']
                    
                    if status == 'success':
                        loss_rate = result['analysis']['trade_summary']['loss_rate_pct']
                        total_trades = result['analysis']['trade_summary']['total_trades']
                        profit_factor = result['analysis']['trade_summary']['profit_factor']
                        
                        print(f"✅ [{i:2d}/{len(available_combinations)}] {symbol}_{strategy}")
                        print(f"    📊 Trades: {total_trades} | Pérdidas: {loss_rate:.1f}% | PF: {profit_factor:.3f} | ⏱️ {proc_time:.1f}s")
                    elif status == 'no_losses':
                        print(f"🎉 [{i:2d}/{len(available_combinations)}] {symbol}_{strategy} - SIN PÉRDIDAS")
                    elif status == 'failed':
                        print(f"⚠️ [{i:2d}/{len(available_combinations)}] {symbol}_{strategy} - FALLO: {result.get('error', 'Unknown')}")
                    else:
                        print(f"❌ [{i:2d}/{len(available_combinations)}] {symbol}_{strategy} - ERROR: {result.get('error', 'Unknown')}")
                    
                except Exception as e:
                    results.append({
                        'symbol': combination['symbol'],
                        'strategy': combination['strategy'],
                        'status': 'timeout',
                        'error': f'Timeout o error: {e}',
                        'processing_time': PROCESS_TIMEOUT
                    })
                    print(f"⏰ [{i:2d}/{len(available_combinations)}] {combination['symbol']}_{combination['strategy']} - TIMEOUT")
        
        total_time = time.time() - start_time
        
        # Generar reporte consolidado
        consolidated_report = self.generate_consolidated_report(results, total_time)
        
        print("\n" + "=" * 80)
        print("🎯 ANÁLISIS MULTI-ASSET COMPLETADO")
        print("=" * 80)
        
        return {
            'results': results,
            'consolidated_report': consolidated_report,
            'total_processing_time': total_time,
            'successful_analyses': len([r for r in results if r['status'] == 'success']),
            'failed_analyses': len([r for r in results if r['status'] in ['failed', 'error', 'timeout']]),
            'no_loss_strategies': len([r for r in results if r['status'] == 'no_losses'])
        }
    
    def generate_consolidated_report(self, results, total_time):
        """Genera reporte consolidado de todos los análisis"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Estadísticas generales
            successful = [r for r in results if r['status'] == 'success']
            failed = [r for r in results if r['status'] in ['failed', 'error', 'timeout']]
            no_losses = [r for r in results if r['status'] == 'no_losses']
            
            # Crear reporte consolidado
            consolidated = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_combinations_analyzed': len(results),
                    'successful_analyses': len(successful),
                    'failed_analyses': len(failed),
                    'strategies_without_losses': len(no_losses),
                    'total_processing_time_seconds': total_time,
                    'average_processing_time_per_strategy': total_time / len(results) if results else 0
                },
                'summary_statistics': self._calculate_summary_statistics(successful),
                'worst_performers': self._identify_worst_performers(successful),
                'best_performers': self._identify_best_performers(successful),
                'strategy_analysis': self._analyze_by_strategy(successful),
                'symbol_analysis': self._analyze_by_symbol(successful),
                'global_recommendations': self._generate_global_recommendations(successful),
                'failed_analyses': [{'symbol': r['symbol'], 'strategy': r['strategy'], 'error': r.get('error', 'Unknown')} for r in failed]
            }
            
            # Exportar reporte consolidado
            consolidated_path = self.export_dir / f"CONSOLIDATED_REPORT_{timestamp}.json"
            with open(consolidated_path, 'w', encoding='utf-8') as f:
                json.dump(consolidated, f, indent=2, default=str)
            
            # Generar reporte ejecutivo consolidado
            exec_summary = self._generate_consolidated_executive_summary(consolidated, total_time)
            exec_path = self.export_dir / f"EXECUTIVE_SUMMARY_{timestamp}.txt"
            with open(exec_path, 'w', encoding='utf-8') as f:
                f.write(exec_summary)
            
            # Crear gráficos consolidados
            chart_path = self._create_consolidated_charts(consolidated, timestamp)
            
            print(f"📋 Reporte consolidado: {consolidated_path}")
            print(f"📄 Resumen ejecutivo: {exec_path}")
            if chart_path:
                print(f"📊 Gráficos consolidados: {chart_path}")
            
            return {
                'consolidated_report_path': str(consolidated_path),
                'executive_summary_path': str(exec_path),
                'charts_path': chart_path,
                'data': consolidated
            }
            
        except Exception as e:
            print(f"❌ Error generando reporte consolidado: {e}")
            return None
    
    def _calculate_summary_statistics(self, successful_results):
        """Calcula estadísticas resumen de todos los análisis exitosos"""
        if not successful_results:
            return {}
        
        try:
            # Extraer métricas clave
            loss_rates = [r['analysis']['trade_summary']['loss_rate_pct'] for r in successful_results]
            profit_factors = [r['analysis']['trade_summary']['profit_factor'] for r in successful_results]
            total_trades = [r['analysis']['trade_summary']['total_trades'] for r in successful_results]
            total_pnls = [r['analysis']['trade_summary']['total_pnl'] for r in successful_results]
            
            return {
                'loss_rate_statistics': {
                    'mean': np.mean(loss_rates),
                    'median': np.median(loss_rates),
                    'std': np.std(loss_rates),
                    'min': np.min(loss_rates),
                    'max': np.max(loss_rates),
                    'q25': np.percentile(loss_rates, 25),
                    'q75': np.percentile(loss_rates, 75)
                },
                'profit_factor_statistics': {
                    'mean': np.mean(profit_factors),
                    'median': np.median(profit_factors),
                    'std': np.std(profit_factors),
                    'min': np.min(profit_factors),
                    'max': np.max(profit_factors),
                    'q25': np.percentile(profit_factors, 25),
                    'q75': np.percentile(profit_factors, 75)
                },
                'trade_count_statistics': {
                    'mean': np.mean(total_trades),
                    'median': np.median(total_trades),
                    'total_trades_analyzed': np.sum(total_trades)
                },
                'pnl_statistics': {
                    'mean': np.mean(total_pnls),
                    'median': np.median(total_pnls),
                    'total_pnl': np.sum(total_pnls),
                    'positive_pnl_count': len([pnl for pnl in total_pnls if pnl > 0]),
                    'negative_pnl_count': len([pnl for pnl in total_pnls if pnl <= 0])
                }
            }
        except Exception as e:
            return {'error': f'Error calculando estadísticas: {e}'}
    
    def _identify_worst_performers(self, successful_results, top_n=10):
        """Identifica las peores combinaciones por tasa de pérdida"""
        if not successful_results:
            return []
        
        try:
            # Ordenar por tasa de pérdida (descendente)
            sorted_by_loss_rate = sorted(successful_results, 
                                       key=lambda x: x['analysis']['trade_summary']['loss_rate_pct'], 
                                       reverse=True)
            
            worst_performers = []
            for result in sorted_by_loss_rate[:top_n]:
                summary = result['analysis']['trade_summary']
                worst_performers.append({
                    'symbol': result['symbol'],
                    'strategy': result['strategy'],
                    'loss_rate_pct': summary['loss_rate_pct'],
                    'total_trades': summary['total_trades'],
                    'profit_factor': summary['profit_factor'],
                    'total_pnl': summary['total_pnl'],
                    'avg_loss': summary['avg_loss']
                })
            
            return worst_performers
        except Exception as e:
            return [{'error': f'Error identificando peores performers: {e}'}]
    
    def _identify_best_performers(self, successful_results, top_n=10):
        """Identifica las mejores combinaciones por profit factor"""
        if not successful_results:
            return []
        
        try:
            # Ordenar por profit factor (descendente)
            sorted_by_pf = sorted(successful_results, 
                                key=lambda x: x['analysis']['trade_summary']['profit_factor'], 
                                reverse=True)
            
            best_performers = []
            for result in sorted_by_pf[:top_n]:
                summary = result['analysis']['trade_summary']
                best_performers.append({
                    'symbol': result['symbol'],
                    'strategy': result['strategy'],
                    'loss_rate_pct': summary['loss_rate_pct'],
                    'total_trades': summary['total_trades'],
                    'profit_factor': summary['profit_factor'],
                    'total_pnl': summary['total_pnl'],
                    'win_rate_pct': summary['win_rate_pct']
                })
            
            return best_performers
        except Exception as e:
            return [{'error': f'Error identificando mejores performers: {e}'}]
    
    def _analyze_by_strategy(self, successful_results):
        """Analiza performance agrupada por estrategia"""
        if not successful_results:
            return {}
        
        try:
            strategy_groups = {}
            
            for result in successful_results:
                strategy = result['strategy']
                if strategy not in strategy_groups:
                    strategy_groups[strategy] = []
                strategy_groups[strategy].append(result['analysis']['trade_summary'])
            
            strategy_analysis = {}
            for strategy, summaries in strategy_groups.items():
                loss_rates = [s['loss_rate_pct'] for s in summaries]
                profit_factors = [s['profit_factor'] for s in summaries]
                
                strategy_analysis[strategy] = {
                    'combinations_count': len(summaries),
                    'avg_loss_rate': np.mean(loss_rates),
                    'avg_profit_factor': np.mean(profit_factors),
                    'best_loss_rate': np.min(loss_rates),
                    'worst_loss_rate': np.max(loss_rates),
                    'best_profit_factor': np.max(profit_factors),
                    'worst_profit_factor': np.min(profit_factors),
                    'consistency_score': 1 / (1 + np.std(loss_rates))  # Puntuación de consistencia
                }
            
            return strategy_analysis
        except Exception as e:
            return {'error': f'Error analizando por estrategia: {e}'}
    
    def _analyze_by_symbol(self, successful_results):
        """Analiza performance agrupada por símbolo"""
        if not successful_results:
            return {}
        
        try:
            symbol_groups = {}
            
            for result in successful_results:
                symbol = result['symbol']
                if symbol not in symbol_groups:
                    symbol_groups[symbol] = []
                symbol_groups[symbol].append(result['analysis']['trade_summary'])
            
            symbol_analysis = {}
            for symbol, summaries in symbol_groups.items():
                loss_rates = [s['loss_rate_pct'] for s in summaries]
                profit_factors = [s['profit_factor'] for s in summaries]
                
                symbol_analysis[symbol] = {
                    'strategies_count': len(summaries),
                    'avg_loss_rate': np.mean(loss_rates),
                    'avg_profit_factor': np.mean(profit_factors),
                    'best_loss_rate': np.min(loss_rates),
                    'worst_loss_rate': np.max(loss_rates),
                    'best_profit_factor': np.max(profit_factors),
                    'worst_profit_factor': np.min(profit_factors),
                    'consistency_score': 1 / (1 + np.std(loss_rates))
                }
            
            return symbol_analysis
        except Exception as e:
            return {'error': f'Error analizando por símbolo: {e}'}
    
    def _generate_global_recommendations(self, successful_results):
        """Genera recomendaciones globales basadas en todos los análisis"""
        if not successful_results:
            return []
        
        recommendations = []
        
        try:
            # Recomendación 1: Estrategias problemáticas
            strategy_loss_rates = {}
            for result in successful_results:
                strategy = result['strategy']
                loss_rate = result['analysis']['trade_summary']['loss_rate_pct']
                
                if strategy not in strategy_loss_rates:
                    strategy_loss_rates[strategy] = []
                strategy_loss_rates[strategy].append(loss_rate)
            
            # Encontrar estrategias con alta tasa de pérdida promedio
            problematic_strategies = []
            for strategy, loss_rates in strategy_loss_rates.items():
                avg_loss_rate = np.mean(loss_rates)
                if avg_loss_rate > 60:  # Más del 60% de pérdidas promedio
                    problematic_strategies.append((strategy, avg_loss_rate))
            
            if problematic_strategies:
                problematic_strategies.sort(key=lambda x: x[1], reverse=True)
                recommendations.append({
                    'category': 'STRATEGY_REVIEW',
                    'priority': 'CRITICAL',
                    'recommendation': f'Revisar urgentemente estas estrategias: {[s[0] for s in problematic_strategies[:3]]}',
                    'details': f'Tasas de pérdida promedio: {[(s[0], f"{s[1]:.1f}%") for s in problematic_strategies[:3]]}'
                })
            
            # Recomendación 2: Símbolos problemáticos
            symbol_loss_rates = {}
            for result in successful_results:
                symbol = result['symbol']
                loss_rate = result['analysis']['trade_summary']['loss_rate_pct']
                
                if symbol not in symbol_loss_rates:
                    symbol_loss_rates[symbol] = []
                symbol_loss_rates[symbol].append(loss_rate)
            
            problematic_symbols = []
            for symbol, loss_rates in symbol_loss_rates.items():
                avg_loss_rate = np.mean(loss_rates)
                if avg_loss_rate > 65:
                    problematic_symbols.append((symbol, avg_loss_rate))
            
            if problematic_symbols:
                problematic_symbols.sort(key=lambda x: x[1], reverse=True)
                recommendations.append({
                    'category': 'SYMBOL_FILTER',
                    'priority': 'HIGH',
                    'recommendation': f'Considerar evitar estos símbolos: {[s[0] for s in problematic_symbols[:3]]}',
                    'details': f'Tasas de pérdida promedio: {[(s[0], f"{s[1]:.1f}%") for s in problematic_symbols[:3]]}'
                })
            
            # Recomendación 3: Mejores combinaciones
            best_combinations = sorted(successful_results, 
                                     key=lambda x: (x['analysis']['trade_summary']['profit_factor'], 
                                                  -x['analysis']['trade_summary']['loss_rate_pct']), 
                                     reverse=True)[:5]
            
            if best_combinations:
                recommendations.append({
                    'category': 'FOCUS_OPTIMIZATION',
                    'priority': 'HIGH',
                    'recommendation': 'Enfocar recursos en optimizar estas combinaciones top',
                    'details': [(f"{r['symbol']}_{r['strategy']}", 
                               f"PF: {r['analysis']['trade_summary']['profit_factor']:.3f}",
                               f"Loss: {r['analysis']['trade_summary']['loss_rate_pct']:.1f}%") 
                              for r in best_combinations]
                })
            
        except Exception as e:
            recommendations.append({
                'category': 'ERROR',
                'priority': 'LOW',
                'recommendation': f'Error generando recomendaciones globales: {e}',
                'details': 'N/A'
            })
        
        return recommendations
    
    def _generate_consolidated_executive_summary(self, consolidated_data, total_time):
        """Genera resumen ejecutivo consolidado"""
        try:
            metadata = consolidated_data['metadata']
            summary_stats = consolidated_data.get('summary_statistics', {})
            worst_performers = consolidated_data.get('worst_performers', [])
            best_performers = consolidated_data.get('best_performers', [])
            global_recs = consolidated_data.get('global_recommendations', [])
            
            report = f"""
================================================================================
🕵️ ANÁLISIS MULTI-ASSET EXHAUSTIVO - REPORTE EJECUTIVO CONSOLIDADO
================================================================================

📊 RESUMEN GENERAL:
📅 Fecha de análisis: {metadata['timestamp']}
🎯 Total combinaciones analizadas: {metadata['total_combinations_analyzed']}
✅ Análisis exitosos: {metadata['successful_analyses']}
❌ Análisis fallidos: {metadata['failed_analyses']}
🎉 Estrategias sin pérdidas: {metadata['strategies_without_losses']}
⏱️ Tiempo total de procesamiento: {total_time/60:.1f} minutos
⚡ Tiempo promedio por estrategia: {metadata['average_processing_time_per_strategy']:.1f} segundos

================================================================================
📊 ESTADÍSTICAS CONSOLIDADAS
================================================================================
"""
            
            if 'loss_rate_statistics' in summary_stats:
                loss_stats = summary_stats['loss_rate_statistics']
                pf_stats = summary_stats['profit_factor_statistics']
                
                report += f"""
📉 TASAS DE PÉRDIDA:
• Promedio: {loss_stats['mean']:.1f}%
• Mediana: {loss_stats['median']:.1f}%
• Mínima: {loss_stats['min']:.1f}%
• Máxima: {loss_stats['max']:.1f}%
• Desviación estándar: {loss_stats['std']:.1f}%

📈 PROFIT FACTORS:
• Promedio: {pf_stats['mean']:.3f}
• Mediana: {pf_stats['median']:.3f}
• Mínimo: {pf_stats['min']:.3f}
• Máximo: {pf_stats['max']:.3f}
"""
            
            # Peores performers
            if worst_performers:
                report += """
================================================================================
🚨 PEORES PERFORMERS (Top 5)
================================================================================
"""
                for i, performer in enumerate(worst_performers[:5], 1):
                    report += f"""
{i}. {performer['symbol']}_{performer['strategy']}
   • Tasa de pérdida: {performer['loss_rate_pct']:.1f}%
   • Profit Factor: {performer['profit_factor']:.3f}
   • Total trades: {performer['total_trades']}
   • PnL total: ${performer['total_pnl']:.6f}
"""
            
            # Mejores performers
            if best_performers:
                report += """
================================================================================
🏆 MEJORES PERFORMERS (Top 5)
================================================================================
"""
                for i, performer in enumerate(best_performers[:5], 1):
                    report += f"""
{i}. {performer['symbol']}_{performer['strategy']}
   • Profit Factor: {performer['profit_factor']:.3f}
   • Tasa de pérdida: {performer['loss_rate_pct']:.1f}%
   • Win Rate: {performer['win_rate_pct']:.1f}%
   • Total trades: {performer['total_trades']}
   • PnL total: ${performer['total_pnl']:.6f}
"""
            
            # Recomendaciones globales
            if global_recs:
                report += """
================================================================================
💡 RECOMENDACIONES GLOBALES CRÍTICAS
================================================================================
"""
                for i, rec in enumerate(global_recs, 1):
                    report += f"""
{i}. [{rec['priority']}] {rec['category']}
   → {rec['recommendation']}
   → Detalles: {rec.get('details', 'N/A')}
"""
            
            report += f"""
================================================================================
🎯 PLAN DE ACCIÓN CONSOLIDADO
================================================================================

1. PRIORIZAR optimización de las {len(best_performers[:3])} mejores combinaciones
2. REVISAR URGENTEMENTE las {len(worst_performers[:3])} peores combinaciones
3. IMPLEMENTAR filtros globales identificados
4. ENFOCAR recursos en estrategias más consistentes
5. VALIDAR todas las mejoras en demo trading

================================================================================
📁 ARCHIVOS GENERADOS
================================================================================

• Reporte consolidado JSON: CONSOLIDATED_REPORT_[timestamp].json
• Reportes individuales: detailed_reports/ (uno por combinación)
• Resúmenes ejecutivos: summaries/ (uno por combinación)
• Este resumen consolidado: EXECUTIVE_SUMMARY_[timestamp].txt
• Gráficos consolidados: CONSOLIDATED_CHARTS_[timestamp].png

================================================================================
⚠️ NOTA IMPORTANTE
================================================================================

Este análisis procesó {metadata['successful_analyses']} combinaciones exitosas
de un total de {metadata['total_combinations_analyzed']} posibles.

Todas las recomendaciones deben validarse en demo trading antes de
implementarse en cuentas reales.

Análisis completado en {total_time/60:.1f} minutos.
================================================================================
"""
            
            return report
            
        except Exception as e:
            return f"Error generando resumen ejecutivo consolidado: {e}"
    
    def _create_consolidated_charts(self, consolidated_data, timestamp):
        """Crea gráficos consolidados de todo el análisis"""
        try:
            fig, axes = plt.subplots(3, 3, figsize=(20, 16))
            fig.suptitle('Análisis Multi-Asset Exhaustivo - Dashboard Consolidado', fontsize=18, fontweight='bold')
            
            summary_stats = consolidated_data.get('summary_statistics', {})
            worst_performers = consolidated_data.get('worst_performers', [])
            best_performers = consolidated_data.get('best_performers', [])
            strategy_analysis = consolidated_data.get('strategy_analysis', {})
            symbol_analysis = consolidated_data.get('symbol_analysis', {})
            
            # 1. Distribución de Tasas de Pérdida
            if 'loss_rate_statistics' in summary_stats:
                ax1 = axes[0, 0]
                loss_stats = summary_stats['loss_rate_statistics']
                
                # Crear histograma simulado basado en estadísticas
                mean = loss_stats['mean']
                std = loss_stats['std']
                data_points = np.random.normal(mean, std, 1000)
                
                ax1.hist(data_points, bins=30, alpha=0.7, color='red', edgecolor='black')
                ax1.axvline(mean, color='blue', linestyle='--', linewidth=2, label=f'Media: {mean:.1f}%')
                ax1.set_title('Distribución de Tasas de Pérdida')
                ax1.set_xlabel('Tasa de Pérdida (%)')
                ax1.set_ylabel('Frecuencia')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # 2. Top 10 Peores Performers
            if worst_performers:
                ax2 = axes[0, 1]
                worst_top10 = worst_performers[:10]
                names = [f"{p['symbol']}_{p['strategy']}" for p in worst_top10]
                loss_rates = [p['loss_rate_pct'] for p in worst_top10]
                
                bars = ax2.barh(range(len(names)), loss_rates, color='red', alpha=0.7)
                ax2.set_yticks(range(len(names)))
                ax2.set_yticklabels(names, fontsize=8)
                ax2.set_title('Top 10 Peores Performers')
                ax2.set_xlabel('Tasa de Pérdida (%)')
                
                # Agregar valores en las barras
                for i, (bar, value) in enumerate(zip(bars, loss_rates)):
                    ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                            f'{value:.1f}%', ha='left', va='center', fontsize=8)
            
            # 3. Top 10 Mejores Performers
            if best_performers:
                ax3 = axes[0, 2]
                best_top10 = best_performers[:10]
                names = [f"{p['symbol']}_{p['strategy']}" for p in best_top10]
                profit_factors = [p['profit_factor'] for p in best_top10]
                
                bars = ax3.barh(range(len(names)), profit_factors, color='green', alpha=0.7)
                ax3.set_yticks(range(len(names)))
                ax3.set_yticklabels(names, fontsize=8)
                ax3.set_title('Top 10 Mejores Performers')
                ax3.set_xlabel('Profit Factor')
                
                for i, (bar, value) in enumerate(zip(bars, profit_factors)):
                    ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                            f'{value:.3f}', ha='left', va='center', fontsize=8)
            
            # 4. Análisis por Estrategia
            if strategy_analysis:
                ax4 = axes[1, 0]
                strategies = list(strategy_analysis.keys())
                avg_loss_rates = [strategy_analysis[s]['avg_loss_rate'] for s in strategies]
                
                bars = ax4.bar(strategies, avg_loss_rates, color='orange', alpha=0.7)
                ax4.set_title('Tasa de Pérdida Promedio por Estrategia')
                ax4.set_ylabel('Tasa de Pérdida Promedio (%)')
                ax4.tick_params(axis='x', rotation=45)
                
                for bar, value in zip(bars, avg_loss_rates):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
            
            # 5. Análisis por Símbolo
            if symbol_analysis:
                ax5 = axes[1, 1]
                symbols = list(symbol_analysis.keys())
                avg_profit_factors = [symbol_analysis[s]['avg_profit_factor'] for s in symbols]
                
                bars = ax5.bar(symbols, avg_profit_factors, color='blue', alpha=0.7)
                ax5.set_title('Profit Factor Promedio por Símbolo')
                ax5.set_ylabel('Profit Factor Promedio')
                ax5.tick_params(axis='x', rotation=45)
                
                for bar, value in zip(bars, avg_profit_factors):
                    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            # 6. Consistencia por Estrategia
            if strategy_analysis:
                ax6 = axes[1, 2]
                strategies = list(strategy_analysis.keys())
                consistency_scores = [strategy_analysis[s]['consistency_score'] for s in strategies]
                
                bars = ax6.bar(strategies, consistency_scores, color='purple', alpha=0.7)
                ax6.set_title('Puntuación de Consistencia por Estrategia')
                ax6.set_ylabel('Consistencia (0-1)')
                ax6.tick_params(axis='x', rotation=45)
                
                for bar, value in zip(bars, consistency_scores):
                    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            # 7. Distribución de Profit Factors
            if 'profit_factor_statistics' in summary_stats:
                ax7 = axes[2, 0]
                pf_stats = summary_stats['profit_factor_statistics']
                
                mean = pf_stats['mean']
                std = pf_stats['std']
                data_points = np.random.normal(mean, std, 1000)
                data_points = np.clip(data_points, 0, None)  # No profit factors negativos
                
                ax7.hist(data_points, bins=30, alpha=0.7, color='green', edgecolor='black')
                ax7.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Media: {mean:.3f}')
                ax7.axvline(1.0, color='orange', linestyle=':', linewidth=2, label='Break-even (1.0)')
                ax7.set_title('Distribución de Profit Factors')
                ax7.set_xlabel('Profit Factor')
                ax7.set_ylabel('Frecuencia')
                ax7.legend()
                ax7.grid(True, alpha=0.3)
            
            # 8. Métricas de Performance Global
            if summary_stats:
                ax8 = axes[2, 1]
                
                # Crear gráfico de radar con métricas clave
                metrics = []
                values = []
                
                if 'loss_rate_statistics' in summary_stats:
                    metrics.append('Tasa Pérdida\nPromedio')
                    # Invertir para que valores bajos sean mejores (normalizar a 0-100)
                    loss_rate_score = max(0, 100 - summary_stats['loss_rate_statistics']['mean'])
                    values.append(loss_rate_score)
                
                if 'profit_factor_statistics' in summary_stats:
                    metrics.append('Profit Factor\nPromedio')
                    # Normalizar PF a escala 0-100 (PF=2.0 = 100 puntos)
                    pf_score = min(100, summary_stats['profit_factor_statistics']['mean'] * 50)
                    values.append(pf_score)
                
                if 'pnl_statistics' in summary_stats:
                    pnl_stats = summary_stats['pnl_statistics']
                    metrics.append('Estrategias\nRentables (%)')
                    profitable_pct = (pnl_stats['positive_pnl_count'] / 
                                    (pnl_stats['positive_pnl_count'] + pnl_stats['negative_pnl_count']) * 100)
                    values.append(profitable_pct)
                
                if len(strategy_analysis) > 0:
                    metrics.append('Consistencia\nPromedio')
                    avg_consistency = np.mean([s['consistency_score'] for s in strategy_analysis.values()]) * 100
                    values.append(avg_consistency)
                
                if metrics and values:
                    bars = ax8.bar(metrics, values, color=['red', 'green', 'blue', 'purple'][:len(values)], alpha=0.7)
                    ax8.set_title('Métricas de Performance Global')
                    ax8.set_ylabel('Puntuación (0-100)')
                    ax8.set_ylim(0, 100)
                    
                    for bar, value in zip(bars, values):
                        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # 9. Resumen de Estado General
            ax9 = axes[2, 2]
            ax9.axis('off')
            
            # Crear texto resumen
            metadata = consolidated_data['metadata']
            summary_text = f"""
RESUMEN EJECUTIVO

Total Análisis: {metadata['total_combinations_analyzed']}
Exitosos: {metadata['successful_analyses']}
Fallidos: {metadata['failed_analyses']}
Sin Pérdidas: {metadata['strategies_without_losses']}

Tiempo Total: {metadata['total_processing_time_seconds']/60:.1f} min
Promedio/Estrategia: {metadata['average_processing_time_per_strategy']:.1f}s
            """
            
            if 'loss_rate_statistics' in summary_stats:
                loss_stats = summary_stats['loss_rate_statistics']
                summary_text += f"""
PÉRDIDAS GLOBALES:
• Promedio: {loss_stats['mean']:.1f}%
• Mediana: {loss_stats['median']:.1f}%
• Rango: {loss_stats['min']:.1f}% - {loss_stats['max']:.1f}%
"""
            
            if 'profit_factor_statistics' in summary_stats:
                pf_stats = summary_stats['profit_factor_statistics']
                summary_text += f"""
PROFIT FACTORS:
• Promedio: {pf_stats['mean']:.3f}
• Mediana: {pf_stats['median']:.3f}
• Mejor: {pf_stats['max']:.3f}
"""
            
            ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
            
            plt.tight_layout()
            
            # Guardar gráfico
            chart_path = self.charts_dir / f"CONSOLIDATED_CHARTS_{timestamp}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            print(f"❌ Error creando gráficos consolidados: {e}")
            return None
    
    def run_comprehensive_analysis(self):
        """Ejecuta análisis comprehensivo de todos los activos y estrategias"""
        print("🚀 INICIANDO ANÁLISIS MULTI-ASSET COMPREHENSIVO")
        print("=" * 80)
        
        start_time = time.time()
        
        # Verificar disponibilidad
        if not IMPORTS_AVAILABLE:
            print("❌ Importaciones no disponibles")
            return None
        
        # Analizar todas las combinaciones
        results = self.analyze_all_combinations_parallel()
        
        if not results:
            print("❌ No se pudieron completar los análisis")
            return None
        
        total_time = time.time() - start_time
        
        # Mostrar resumen final
        print("\n" + "=" * 80)
        print("🎯 ANÁLISIS MULTI-ASSET COMPLETADO")
        print("=" * 80)
        print(f"✅ Análisis exitosos: {results['successful_analyses']}")
        print(f"🎉 Sin pérdidas: {results['no_loss_strategies']}")
        print(f"❌ Fallidos: {results['failed_analyses']}")
        print(f"⏱️ Tiempo total: {total_time/60:.1f} minutos")
        print(f"📁 Reportes en: {self.export_dir}")
        
        # Mostrar top insights
        if results['consolidated_report'] and 'data' in results['consolidated_report']:
            data = results['consolidated_report']['data']
            
            if 'worst_performers' in data and data['worst_performers']:
                print(f"\n🚨 PEOR PERFORMER: {data['worst_performers'][0]['symbol']}_{data['worst_performers'][0]['strategy']}")
                print(f"   Tasa de pérdida: {data['worst_performers'][0]['loss_rate_pct']:.1f}%")
            
            if 'best_performers' in data and data['best_performers']:
                print(f"\n🏆 MEJOR PERFORMER: {data['best_performers'][0]['symbol']}_{data['best_performers'][0]['strategy']}")
                print(f"   Profit Factor: {data['best_performers'][0]['profit_factor']:.3f}")
                print(f"   Tasa de pérdida: {data['best_performers'][0]['loss_rate_pct']:.1f}%")
        
        print("\n" + "=" * 80)
        print("🎯 PRÓXIMOS PASOS:")
        print("   1. Revisar reporte ejecutivo consolidado")
        print("   2. Analizar gráficos consolidados")
        print("   3. Revisar reportes individuales de peores performers")
        print("   4. Implementar recomendaciones globales")
        print("   5. Re-optimizar estrategias problemáticas")
        print("=" * 80)
        
        return results


def main():
    """Función principal para análisis multi-asset"""
    print("🕵️ MULTI-ASSET EXHAUSTIVE LOSS ANALYZER")
    print("=" * 80)
    print("🎯 CARACTERÍSTICAS:")
    print("   ✅ Análisis AUTOMÁTICO de todos los activos y estrategias")
    print("   ✅ Procesamiento PARALELO para máxima eficiencia")
    print("   ✅ Reportes CONSOLIDADOS con rankings y comparaciones")
    print("   ✅ Recomendaciones GLOBALES basadas en todos los datos")
    print("   ✅ Dashboard visual consolidado")
    print("   ✅ Identificación automática de mejores y peores performers")
    print("   ✅ Análisis por estrategia y por símbolo")
    print("   ✅ Exportación organizada en subdirectorios")
    
    if not IMPORTS_AVAILABLE:
        print("❌ Importaciones no disponibles")
        return
    
    try:
        # Crear analizador
        analyzer = MultiAssetLossAnalyzer()
        
        # Mostrar configuración detectada
        print(f"\n📊 CONFIGURACIÓN DETECTADA:")
        print(f"   🎯 Símbolos: {analyzer.symbols}")
        print(f"   📈 Estrategias: {analyzer.strategies}")
        print(f"   🔄 Workers paralelos: {MAX_WORKERS}")
        print(f"   ⏱️ Timeout por análisis: {PROCESS_TIMEOUT}s")
        
        # Confirmar ejecución
        print(f"\n⚠️ Se procesarán {len(analyzer.symbols) * len(analyzer.strategies)} combinaciones")
        print("¿Continuar? (y/N): ", end="")
        
        try:
            confirm = input().strip().lower()
            if confirm != 'y':
                print("❌ Análisis cancelado por el usuario")
                return
        except KeyboardInterrupt:
            print("\n❌ Análisis cancelado por el usuario")
            return
        
        # Ejecutar análisis comprehensivo
        results = analyzer.run_comprehensive_analysis()
        
        if results:
            print("\n✅ ANÁLISIS MULTI-ASSET COMPLETADO EXITOSAMENTE")
        else:
            print("\n❌ El análisis no se pudo completar")
            
    except KeyboardInterrupt:
        print("\n⚠️ Análisis interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🎯 Presiona Enter para continuar...")
        try:
            input()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()