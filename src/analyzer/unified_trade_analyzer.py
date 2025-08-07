# -*- coding: utf-8 -*-
"""
unified_trade_analyzer.py - GENERADOR UNIFICADO DE DATOS PARA ANÁLISIS CAUSAL
-----------------------------------------------------------------------------
- Procesa TODOS los activos y estrategias automáticamente
- Genera formato NORMALIZADO compatible con causal_trading_analyzer_v3_fixed.py
- Unifica análisis de ganancias y pérdidas en una sola pasada
- Exporta datos listos para análisis causal con todas las columnas necesarias
- Evita duplicación y asegura consistencia de datos

SALIDA: Archivos CSV en formato normalizado para análisis causal directo
"""
import os
import json
import pandas as pd
import numpy as np
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
# CONFIGURACIÓN UNIFICADA
# ==========================================================
DEFAULT_SYMBOLS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD',
    'NZDUSD', 'EURJPY', 'GBPJPY', 'EURGBP', 'AUDJPY', 'CHFJPY'
]

DEFAULT_STRATEGIES = [
    'ema_crossover', 'channel_reversal', 'rsi_pullback',
    'volatility_breakout', 'multi_filter_scalper', 'lokz_reversal'
]

# Configuración de procesamiento
MAX_WORKERS = 4
PROCESS_TIMEOUT = 300

class UnifiedTradeAnalyzer:
    def __init__(self, optimization_dir="optimization_studies", output_dir=None):
        self.optimization_dir = Path(optimization_dir)
        
        # Directorio de salida compatible con el analizador causal
        if output_dir is None:
            self.output_dir = Path("E:/Cryptos/Advanced_Trading_Bot_V2/reports/causal_reports/causal_inputs")
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear subdirectorios para organización
        self.metadata_dir = self.output_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        print(f"🔄 UNIFIED TRADE ANALYZER - GENERADOR DE DATOS CAUSALES")
        print(f"📁 Optimización: {self.optimization_dir}")
        print(f"📁 Salida normalizada: {self.output_dir}")
        print(f"🎯 Compatible con: causal_trading_analyzer_v3_fixed.py")
    
    def detect_available_combinations(self):
        """Detecta todas las combinaciones disponibles automáticamente"""
        combinations = []
        
        if not self.optimization_dir.exists():
            print(f"❌ Directorio de optimización no existe: {self.optimization_dir}")
            return combinations
        
        # Buscar archivos .db de optimización
        for db_file in self.optimization_dir.glob("*.db"):
            try:
                stem = db_file.stem
                if '_' in stem:
                    parts = stem.split('_', 1)
                    symbol = parts[0].upper()
                    strategy = parts[1].lower()
                    
                    if strategy in STRATEGY_CLASSES:
                        combinations.append({
                            'symbol': symbol,
                            'strategy': strategy,
                            'db_file': db_file,
                            'study_name': stem
                        })
                        print(f"✅ Detectado: {symbol} + {strategy}")
            except Exception as e:
                print(f"⚠️ Error procesando {db_file.name}: {e}")
        
        print(f"🎯 Total combinaciones detectadas: {len(combinations)}")
        return combinations
    
    def load_optimized_params(self, combination):
        """Carga parámetros optimizados desde base de datos"""
        try:
            study_name = combination['study_name']
            db_file = combination['db_file']
            
            storage_url = f"sqlite:///{db_file}"
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            
            completed_trials = [t for t in study.trials 
                              if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
            
            if not completed_trials:
                return None
            
            best_trial = max(completed_trials, key=lambda x: x.value)
            return best_trial.params
            
        except Exception as e:
            print(f"❌ Error cargando parámetros para {combination['symbol']}_{combination['strategy']}: {e}")
            return None
    
    def run_comprehensive_backtest(self, combination, params):
        """Ejecuta backtest completo con máxima información"""
        symbol = combination['symbol']
        strategy_name = combination['strategy']
        
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
                'symbol': symbol,
                'strategy': strategy_name,
                'trade_log': trade_log,
                'market_data': data_with_indicators,
                'report': report,
                'params': params,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Error en backtest {symbol}_{strategy_name}: {e}")
            return None
    
    def normalize_trade_data(self, backtest_result):
        """Normaliza datos de trades al formato esperado por el analizador causal"""
        try:
            trade_log = backtest_result['trade_log']
            market_data = backtest_result['market_data']
            symbol = backtest_result['symbol']
            strategy = backtest_result['strategy']
            
            if trade_log.empty:
                print(f"⚠️ {symbol}_{strategy}: Sin trades")
                return None
            
            # Crear DataFrame normalizado
            normalized_trades = []
            
            for idx, trade in trade_log.iterrows():
                # Estructura base del trade normalizado
                normalized_trade = {
                    'trade_index': idx,
                    'trade_number': idx + 1,
                    'asset': symbol,
                    'strategy': strategy,
                }
                
                # Mapear columnas estándar de trades
                trade_mapping = {
                    'trade_entry_time': ['entry_time', 'open_time', 'start_time'],
                    'trade_exit_time': ['exit_time', 'close_time', 'end_time'],
                    'trade_side': ['side', 'direction', 'type'],
                    'trade_entry_price': ['entry_price', 'open_price'],
                    'trade_exit_price': ['exit_price', 'close_price'],
                    'trade_pnl': ['pnl', 'profit', 'return', 'P&L', 'PnL', 'net_profit', 'profit_loss']
                }
                
                # Aplicar mapeo
                for target_col, source_cols in trade_mapping.items():
                    value = None
                    for source_col in source_cols:
                        if source_col in trade.index and pd.notna(trade[source_col]):
                            value = trade[source_col]
                            break
                    normalized_trade[target_col] = value
                
                # Asegurar que tenemos PnL
                if normalized_trade['trade_pnl'] is None:
                    print(f"⚠️ Trade {idx}: Sin PnL válido")
                    continue
                
                # Obtener condiciones de mercado en el momento de entrada
                market_conditions = self.get_market_conditions_at_entry(
                    trade, market_data, idx, len(trade_log)
                )
                
                # Combinar todo
                normalized_trade.update(market_conditions)
                
                # Agregar metadatos adicionales
                normalized_trade.update({
                    'entry_method': market_conditions.get('entry_method', 'index_based'),
                    'result': 1 if float(normalized_trade['trade_pnl']) > 0 else 0,
                })
                
                normalized_trades.append(normalized_trade)
            
            if not normalized_trades:
                print(f"⚠️ {symbol}_{strategy}: Sin trades válidos después de normalización")
                return None
            
            df_normalized = pd.DataFrame(normalized_trades)
            
            # Validar columnas críticas
            required_cols = ['trade_pnl', 'asset', 'strategy']
            missing_cols = [col for col in required_cols if col not in df_normalized.columns]
            if missing_cols:
                print(f"❌ {symbol}_{strategy}: Columnas faltantes: {missing_cols}")
                return None
            
            print(f"✅ {symbol}_{strategy}: {len(df_normalized)} trades normalizados")
            return df_normalized
            
        except Exception as e:
            print(f"❌ Error normalizando {symbol}_{strategy}: {e}")
            return None
    
    def get_market_conditions_at_entry(self, trade, market_data, trade_idx, total_trades):
        """Obtiene condiciones de mercado completas en el momento de entrada"""
        try:
            market_conditions = {}
            
            # Método 1: Usar entry_time si está disponible
            if 'entry_time' in trade.index and pd.notna(trade['entry_time']):
                try:
                    entry_time = pd.to_datetime(trade['entry_time'])
                    if hasattr(market_data, 'index') and isinstance(market_data.index, pd.DatetimeIndex):
                        if entry_time in market_data.index:
                            market_row = market_data.loc[entry_time]
                            market_conditions['entry_method'] = 'exact_timestamp'
                            for col in market_data.columns:
                                market_conditions[f'market_{col}'] = market_row[col]
                            return market_conditions
                except:
                    pass
            
            # Método 2: Usar entry_bar_index si está disponible
            if 'entry_bar_index' in trade.index and pd.notna(trade['entry_bar_index']):
                try:
                    entry_idx = int(trade['entry_bar_index'])
                    if 0 <= entry_idx < len(market_data):
                        market_row = market_data.iloc[entry_idx]
                        market_conditions['entry_method'] = 'bar_index'
                        for col in market_data.columns:
                            market_conditions[f'market_{col}'] = market_row[col]
                        return market_conditions
                except:
                    pass
            
            # Método 3: Distribución uniforme (fallback)
            entry_idx = int((trade_idx / max(total_trades - 1, 1)) * max(len(market_data) - 1, 0))
            entry_idx = min(max(entry_idx, 0), len(market_data) - 1)
            
            market_row = market_data.iloc[entry_idx]
            market_conditions['entry_method'] = 'fallback_distribution'
            
            # Copiar TODAS las columnas de market data con prefijo
            for col in market_data.columns:
                try:
                    value = market_row[col]
                    if pd.isna(value):
                        market_conditions[f'market_{col}'] = None
                    else:
                        market_conditions[f'market_{col}'] = float(value) if isinstance(value, (int, float)) else value
                except:
                    market_conditions[f'market_{col}'] = None
            
            # Asegurar columnas básicas OHLC
            basic_mapping = {
                'market_Open': 'market_open',
                'market_High': 'market_high', 
                'market_Low': 'market_low',
                'market_Close': 'market_close',
                'market_Volume': 'market_volume'
            }
            
            for old_key, new_key in basic_mapping.items():
                if old_key in market_conditions and new_key not in market_conditions:
                    market_conditions[new_key] = market_conditions[old_key]
            
            return market_conditions
            
        except Exception as e:
            print(f"⚠️ Error obteniendo condiciones de mercado para trade {trade_idx}: {e}")
            return {
                'entry_method': 'error',
                'market_close': None,
                'market_open': None,
                'market_high': None,
                'market_low': None,
                'error': str(e)
            }
    
    def export_normalized_data(self, df_normalized, symbol, strategy):
        """Exporta datos normalizados en formato compatible con analizador causal"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{strategy}_trades.csv"
            filepath = self.output_dir / filename
            
            # Exportar CSV principal
            df_normalized.to_csv(filepath, index=False)
            
            # Exportar metadatos
            metadata = {
                'symbol': symbol,
                'strategy': strategy,
                'export_timestamp': datetime.now().isoformat(),
                'total_trades': len(df_normalized),
                'winning_trades': len(df_normalized[df_normalized['trade_pnl'] > 0]),
                'losing_trades': len(df_normalized[df_normalized['trade_pnl'] <= 0]),
                'total_pnl': float(df_normalized['trade_pnl'].sum()),
                'columns': list(df_normalized.columns),
                'data_quality': {
                    'has_entry_times': df_normalized['trade_entry_time'].notna().sum(),
                    'has_exit_times': df_normalized['trade_exit_time'].notna().sum(),
                    'has_market_data': len([col for col in df_normalized.columns if col.startswith('market_')]),
                    'null_pnl_count': df_normalized['trade_pnl'].isna().sum()
                }
            }
            
            metadata_file = self.metadata_dir / f"{symbol}_{strategy}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"📁 Exportado: {filepath}")
            print(f"📋 Metadatos: {metadata_file}")
            
            return {
                'csv_file': str(filepath),
                'metadata_file': str(metadata_file),
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"❌ Error exportando {symbol}_{strategy}: {e}")
            return None
    
    def process_single_combination(self, combination):
        """Procesa una combinación completa: backtest + normalización + export"""
        symbol = combination['symbol']
        strategy = combination['strategy']
        
        try:
            start_time = time.time()
            print(f"\n🔄 Procesando {symbol}_{strategy}...")
            
            # 1. Cargar parámetros optimizados
            params = self.load_optimized_params(combination)
            if not params:
                return {
                    'symbol': symbol,
                    'strategy': strategy,
                    'status': 'failed',
                    'error': 'No se pudieron cargar parámetros',
                    'processing_time': time.time() - start_time
                }
            
            # 2. Ejecutar backtest completo
            backtest_result = self.run_comprehensive_backtest(combination, params)
            if not backtest_result or not backtest_result.get('success'):
                return {
                    'symbol': symbol,
                    'strategy': strategy,
                    'status': 'failed',
                    'error': 'Error en backtest',
                    'processing_time': time.time() - start_time
                }
            
            # 3. Normalizar datos
            normalized_df = self.normalize_trade_data(backtest_result)
            if normalized_df is None or normalized_df.empty:
                return {
                    'symbol': symbol,
                    'strategy': strategy,
                    'status': 'failed',
                    'error': 'Error en normalización o sin datos válidos',
                    'processing_time': time.time() - start_time
                }
            
            # 4. Exportar datos normalizados
            export_result = self.export_normalized_data(normalized_df, symbol, strategy)
            if not export_result:
                return {
                    'symbol': symbol,
                    'strategy': strategy,
                    'status': 'failed',
                    'error': 'Error en exportación',
                    'processing_time': time.time() - start_time
                }
            
            processing_time = time.time() - start_time
            
            return {
                'symbol': symbol,
                'strategy': strategy,
                'status': 'success',
                'export_result': export_result,
                'processing_time': processing_time,
                'summary': {
                    'total_trades': len(normalized_df),
                    'winning_trades': len(normalized_df[normalized_df['trade_pnl'] > 0]),
                    'win_rate': len(normalized_df[normalized_df['trade_pnl'] > 0]) / len(normalized_df) * 100,
                    'total_pnl': float(normalized_df['trade_pnl'].sum()),
                    'profit_factor': self.calculate_profit_factor(normalized_df)
                }
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'strategy': strategy,
                'status': 'error',
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def calculate_profit_factor(self, df):
        """Calcula profit factor de forma segura"""
        try:
            gross_profit = df[df['trade_pnl'] > 0]['trade_pnl'].sum()
            gross_loss = abs(df[df['trade_pnl'] < 0]['trade_pnl'].sum())
            
            if gross_loss == 0:
                return float('inf') if gross_profit > 0 else 0
            
            return gross_profit / gross_loss
        except:
            return 0
    
    def process_all_combinations_parallel(self):
        """Procesa todas las combinaciones en paralelo"""
        combinations = self.detect_available_combinations()
        
        if not combinations:
            print("❌ No hay combinaciones disponibles para procesar")
            return None
        
        print(f"\n🚀 Procesando {len(combinations)} combinaciones en paralelo")
        print(f"⚡ Workers: {MAX_WORKERS} | Timeout: {PROCESS_TIMEOUT}s")
        print("=" * 80)
        
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Enviar todas las tareas
            future_to_combination = {
                executor.submit(self.process_single_combination, combo): combo 
                for combo in combinations
            }
            
            # Procesar resultados
            for i, future in enumerate(as_completed(future_to_combination, timeout=PROCESS_TIMEOUT * len(combinations)), 1):
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
                        summary = result['summary']
                        print(f"✅ [{i:2d}/{len(combinations)}] {symbol}_{strategy}")
                        print(f"    📊 Trades: {summary['total_trades']} | Win Rate: {summary['win_rate']:.1f}% | PF: {summary['profit_factor']:.3f} | ⏱️ {proc_time:.1f}s")
                    else:
                        print(f"❌ [{i:2d}/{len(combinations)}] {symbol}_{strategy} - {result.get('error', 'Unknown')}")
                    
                except Exception as e:
                    results.append({
                        'symbol': combination['symbol'],
                        'strategy': combination['strategy'],
                        'status': 'timeout',
                        'error': f'Timeout: {e}',
                        'processing_time': PROCESS_TIMEOUT
                    })
                    print(f"⏰ [{i:2d}/{len(combinations)}] {combination['symbol']}_{combination['strategy']} - TIMEOUT")
        
        total_time = time.time() - start_time
        
        # Generar reporte de procesamiento
        processing_report = self.generate_processing_report(results, total_time)
        
        print("\n" + "=" * 80)
        print("🎯 PROCESAMIENTO UNIFICADO COMPLETADO")
        print("=" * 80)
        
        return {
            'results': results,
            'processing_report': processing_report,
            'total_time': total_time,
            'successful_exports': len([r for r in results if r['status'] == 'success']),
            'failed_exports': len([r for r in results if r['status'] in ['failed', 'error', 'timeout']])
        }
    
    def generate_processing_report(self, results, total_time):
        """Genera reporte del procesamiento unificado"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            successful = [r for r in results if r['status'] == 'success']
            failed = [r for r in results if r['status'] != 'success']
            
            report = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_combinations_processed': len(results),
                    'successful_exports': len(successful),
                    'failed_exports': len(failed),
                    'success_rate': len(successful) / len(results) * 100 if results else 0,
                    'total_processing_time': total_time,
                    'average_time_per_combination': total_time / len(results) if results else 0,
                    'output_directory': str(self.output_dir)
                },
                'successful_exports': [
                    {
                        'symbol': r['symbol'],
                        'strategy': r['strategy'],
                        'csv_file': r['export_result']['csv_file'],
                        'total_trades': r['summary']['total_trades'],
                        'win_rate': r['summary']['win_rate'],
                        'profit_factor': r['summary']['profit_factor']
                    } for r in successful
                ],
                'failed_exports': [
                    {
                        'symbol': r['symbol'],
                        'strategy': r['strategy'],
                        'error': r.get('error', 'Unknown')
                    } for r in failed
                ],
                'statistics': {
                    'total_trades_exported': sum(r['summary']['total_trades'] for r in successful),
                    'total_combinations_with_data': len(successful),
                    'average_trades_per_combination': np.mean([r['summary']['total_trades'] for r in successful]) if successful else 0,
                    'best_performer': max(successful, key=lambda x: x['summary']['profit_factor']) if successful else None
                }
            }
            
            # Exportar reporte
            report_file = self.metadata_dir / f"processing_report_{timestamp}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Resumen ejecutivo
            summary_text = f"""
================================================================================
🔄 REPORTE DE PROCESAMIENTO UNIFICADO
================================================================================

📅 Timestamp: {report['metadata']['timestamp']}
📊 Combinaciones procesadas: {report['metadata']['total_combinations_processed']}
✅ Exportaciones exitosas: {report['metadata']['successful_exports']}
❌ Exportaciones fallidas: {report['metadata']['failed_exports']}
📈 Tasa de éxito: {report['metadata']['success_rate']:.1f}%
⏱️ Tiempo total: {total_time/60:.1f} minutos

================================================================================
📁 ARCHIVOS GENERADOS
================================================================================

Directorio de salida: {self.output_dir}
Archivos CSV generados: {len(successful)}
Total trades exportados: {report['statistics']['total_trades_exported']}

Los archivos están listos para ser procesados por:
causal_trading_analyzer_v3_fixed.py

================================================================================
🎯 PRÓXIMOS PASOS
================================================================================

1. Ejecutar análisis causal:
   python causal_trading_analyzer_v3_fixed.py --export-dir "{self.output_dir}"

2. Los datos están normalizados y contienen:
   - Información completa de trades
   - Condiciones de mercado en momento de entrada
   - Metadatos de calidad de datos

================================================================================
"""
            
            summary_file = self.output_dir / f"PROCESSING_SUMMARY_{timestamp}.txt"
            with open(summary_file, 'w') as f:
                f.write(summary_text)
            
            print(f"📋 Reporte completo: {report_file}")
            print(f"📄 Resumen ejecutivo: {summary_file}")
            
            return report
            
        except Exception as e:
            print(f"❌ Error generando reporte: {e}")
            return None
    
    def run_unified_analysis(self):
        """Ejecuta análisis unificado completo"""
        print("🚀 INICIANDO ANÁLISIS UNIFICADO DE DATOS")
        print("=" * 80)
        print("🎯 OBJETIVOS:")
        print("   ✅ Detectar automáticamente todas las combinaciones disponibles")
        print("   ✅ Ejecutar backtests completos con datos de mercado")
        print("   ✅ Normalizar formato para compatibilidad con análisis causal")
        print("   ✅ Exportar datos listos para causal_trading_analyzer_v3_fixed.py")
        print("   ✅ Evitar duplicación y asegurar consistencia")
        
        if not IMPORTS_AVAILABLE:
            print("❌ Importaciones de estrategias no disponibles")
            return None
        
        try:
            # Procesar todas las combinaciones
            result = self.process_all_combinations_parallel()
            
            if not result:
                print("❌ No se pudieron procesar las combinaciones")
                return None
            
            print(f"\n✅ PROCESAMIENTO COMPLETADO:")
            print(f"   📊 Exportaciones exitosas: {result['successful_exports']}")
            print(f"   ❌ Exportaciones fallidas: {result['failed_exports']}")
            print(f"   ⏱️ Tiempo total: {result['total_time']/60:.1f} minutos")
            print(f"   📁 Archivos en: {self.output_dir}")
            
            if result['successful_exports'] > 0:
                print(f"\n🎯 SIGUIENTE PASO:")
                print(f"   Ejecutar análisis causal con:")
                print(f"   python causal_trading_analyzer_v3_fixed.py --export-dir \"{self.output_dir}\"")
            
            return result
            
        except Exception as e:
            print(f"❌ Error en análisis unificado: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Función principal"""
    print("🔄 UNIFIED TRADE ANALYZER")
    print("=" * 80)
    print("🎯 VENTAJAS DEL GENERADOR UNIFICADO:")
    print("   ✅ Elimina duplicación de código y datos")
    print("   ✅ Formato NORMALIZADO compatible con análisis causal")
    print("   ✅ Procesamiento PARALELO para máxima eficiencia")
    print("   ✅ Detección AUTOMÁTICA de combinaciones disponibles")
    print("   ✅ Metadatos completos de calidad de datos")
    print("   ✅ Exportación directa para causal_trading_analyzer_v3_fixed.py")
    
    if not IMPORTS_AVAILABLE:
        print("\n❌ IMPORTS NO DISPONIBLES")
        print("   Asegúrate de que las estrategias estén disponibles")
        return
    
    try:
        # Configurar directorio de salida
        output_dir = input("\n📁 Directorio de salida (Enter para default): ").strip()
        if not output_dir:
            output_dir = "E:/Cryptos/Advanced_Trading_Bot_V2/reports/ultimate_analysis_reports_v2/causal_inputs"
        
        # Crear analizador
        analyzer = UnifiedTradeAnalyzer(output_dir=output_dir)
        
        # Mostrar combinaciones detectadas
        combinations = analyzer.detect_available_combinations()
        
        if not combinations:
            print("\n❌ No se detectaron combinaciones disponibles")
            print("   Verifica que existan archivos .db en optimization_studies/")
            return
        
        print(f"\n📊 COMBINACIONES DETECTADAS: {len(combinations)}")
        for combo in combinations[:5]:  # Mostrar primeras 5
            print(f"   • {combo['symbol']} + {combo['strategy']}")
        if len(combinations) > 5:
            print(f"   ... y {len(combinations) - 5} más")
        
        # Confirmar procesamiento
        print(f"\n⚠️ Se procesarán {len(combinations)} combinaciones")
        print("   Esto puede tomar varios minutos dependiendo de la cantidad de datos")
        confirm = input("¿Continuar? (y/N): ").strip().lower()
        
        if confirm != 'y':
            print("❌ Procesamiento cancelado")
            return
        
        # Ejecutar análisis unificado
        print("\n🚀 INICIANDO PROCESAMIENTO UNIFICADO...")
        result = analyzer.run_unified_analysis()
        
        if result and result['successful_exports'] > 0:
            print("\n🎉 ¡PROCESAMIENTO COMPLETADO EXITOSAMENTE!")
            print(f"   📁 Archivos generados: {result['successful_exports']}")
            print(f"   📊 Listos para análisis causal en: {analyzer.output_dir}")
            
            # Sugerir siguiente paso
            print(f"\n🎯 COMANDO PARA ANÁLISIS CAUSAL:")
            print(f"   python causal_trading_analyzer_v3_fixed.py --export-dir \"{analyzer.output_dir}\" --verbose")
        else:
            print("\n❌ No se pudieron generar archivos válidos")
            
    except KeyboardInterrupt:
        print("\n⚠️ Procesamiento interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n📋 Presiona Enter para continuar...")
        try:
            input()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
    
    