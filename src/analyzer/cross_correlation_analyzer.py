# cross_correlation_analyzer.py - Análisis EXHAUSTIVO de correlaciones cruzadas
# Combina análisis de pérdidas y ganancias para detectar patrones cross-asset y cross-strategy

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
from scipy import stats
from scipy.stats import pearsonr, spearmanr, chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import itertools

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
# ✅ CONFIGURACIÓN DE ANÁLISIS CRUZADO
# ==========================================================
ASSETS_TO_ANALYZE = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
STRATEGIES_TO_ANALYZE = ['ema_crossover', 'channel_reversal', 'rsi_pullback', 
                        'volatility_breakout', 'multi_filter_scalper', 'lokz_reversal']
# ==========================================================

class CrossCorrelationAnalyzer:
    def __init__(self, optimization_dir="optimization_studies"):
        self.optimization_dir = Path(optimization_dir)
        self.export_dir = Path("cross_correlation_reports")
        self.export_dir.mkdir(exist_ok=True)
        
        self.assets = ASSETS_TO_ANALYZE
        self.strategies = STRATEGIES_TO_ANALYZE
        self.all_data = {}  # Almacena todos los datos de backtests
        self.correlation_matrices = {}
        self.cross_asset_patterns = {}
        self.cross_strategy_patterns = {}
        
        print(f"🔄 CROSS-CORRELATION ANALYZER")
        print(f"📊 Assets: {self.assets}")
        print(f"🎯 Strategies: {self.strategies}")
        print(f"📁 Export: {self.export_dir}")
    
    def run_comprehensive_analysis(self):
        """Ejecuta análisis comprehensivo de correlaciones cruzadas"""
        print("🚀 INICIANDO ANÁLISIS COMPREHENSIVO DE CORRELACIONES")
        print("=" * 80)
        
        # Paso 1: Recopilar todos los datos
        print("📥 Paso 1: Recopilando datos de todos los asset-strategy pairs...")
        self._collect_all_backtest_data()
        
        # Paso 2: Análisis de correlaciones entre assets
        print("\n🔄 Paso 2: Analizando correlaciones entre assets...")
        self._analyze_cross_asset_correlations()
        
        # Paso 3: Análisis de correlaciones entre estrategias
        print("\n🎯 Paso 3: Analizando correlaciones entre estrategias...")
        self._analyze_cross_strategy_correlations()
        
        # Paso 4: Análisis de patrones temporales cruzados
        print("\n⏰ Paso 4: Analizando patrones temporales cruzados...")
        self._analyze_temporal_cross_patterns()
        
        # Paso 5: Clustering y segmentación
        print("\n🎪 Paso 5: Clustering de asset-strategy combinations...")
        self._perform_clustering_analysis()
        
        # Paso 6: Generación de reglas para bots
        print("\n🤖 Paso 6: Generando reglas para bots algorítmicos...")
        bot_rules = self._generate_algorithmic_rules()
        
        # Paso 7: Crear reporte final
        print("\n📋 Paso 7: Generando reporte final...")
        final_report = self._create_comprehensive_report(bot_rules)
        
        # Paso 8: Exportar todo
        print("\n📁 Paso 8: Exportando análisis...")
        export_results = self._export_comprehensive_analysis(final_report)
        
        self._display_executive_summary(final_report)
        
        return {
            'comprehensive_analysis': final_report,
            'bot_rules': bot_rules,
            'export_results': export_results,
            'raw_data': self.all_data
        }
    
    def _collect_all_backtest_data(self):
        """Recopila datos de backtest para todos los asset-strategy pairs"""
        total_combinations = len(self.assets) * len(self.strategies)
        current_combination = 0
        
        for asset in self.assets:
            self.all_data[asset] = {}
            
            for strategy in self.strategies:
                current_combination += 1
                print(f"   📊 [{current_combination}/{total_combinations}] {asset}_{strategy}")
                
                try:
                    # Cargar parámetros optimizados
                    params = self._load_optimized_params(asset, strategy)
                    
                    if params:
                        # Ejecutar backtest
                        backtest_result = self._run_backtest(asset, strategy, params)
                        
                        if backtest_result and backtest_result.get('success'):
                            self.all_data[asset][strategy] = backtest_result
                            print(f"      ✅ Datos cargados: {len(backtest_result.get('trade_log', []))} trades")
                        else:
                            print(f"      ❌ Error en backtest")
                            self.all_data[asset][strategy] = None
                    else:
                        print(f"      ⚠️ Sin parámetros optimizados")
                        self.all_data[asset][strategy] = None
                        
                except Exception as e:
                    print(f"      ❌ Error: {e}")
                    self.all_data[asset][strategy] = None
        
        # Resumen de datos recopilados
        successful_combinations = 0
        for asset in self.all_data:
            for strategy in self.all_data[asset]:
                if self.all_data[asset][strategy] is not None:
                    successful_combinations += 1
        
        print(f"\n✅ Datos recopilados: {successful_combinations}/{total_combinations} combinaciones exitosas")
    
    def _load_optimized_params(self, symbol, strategy_name):
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
            return None
    
    def _run_backtest(self, symbol, strategy_name, params):
        """Ejecuta backtest individual"""
        if not IMPORTS_AVAILABLE or strategy_name not in STRATEGY_CLASSES:
            return None
        
        try:
            strategy_class = STRATEGY_CLASSES[strategy_name]
            strategy_instance = strategy_class(**params)
            backtester = Backtester(symbol=symbol, strategy=strategy_instance)
            
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
            return None
    
    def _analyze_cross_asset_correlations(self):
        """Analiza correlaciones entre diferentes assets"""
        print("   🔄 Calculando matrices de correlación entre assets...")
        
        # Detectar columna PnL universal
        pnl_columns = ['pnl', 'profit', 'return', 'P&L', 'PnL', 'net_profit', 'profit_loss']
        
        # Crear matrices de resultados por strategy
        for strategy in self.strategies:
            strategy_results = {}
            
            for asset in self.assets:
                if (asset in self.all_data and 
                    strategy in self.all_data[asset] and 
                    self.all_data[asset][strategy] is not None):
                    
                    trade_log = self.all_data[asset][strategy]['trade_log']
                    
                    if not trade_log.empty:
                        # Encontrar columna PnL
                        pnl_col = None
                        for col in pnl_columns:
                            if col in trade_log.columns:
                                pnl_col = col
                                break
                        
                        if pnl_col:
                            # Crear serie temporal de resultados
                            strategy_results[asset] = self._create_temporal_series(trade_log, pnl_col)
            
            if len(strategy_results) >= 2:
                # Calcular correlaciones para esta estrategia
                self.correlation_matrices[strategy] = self._calculate_asset_correlations(strategy_results)
        
        # Análisis de correlaciones globales (todas las estrategias)
        global_results = {}
        for asset in self.assets:
            asset_performance = []
            
            for strategy in self.strategies:
                if (asset in self.all_data and 
                    strategy in self.all_data[asset] and 
                    self.all_data[asset][strategy] is not None):
                    
                    trade_log = self.all_data[asset][strategy]['trade_log']
                    
                    if not trade_log.empty:
                        pnl_col = None
                        for col in pnl_columns:
                            if col in trade_log.columns:
                                pnl_col = col
                                break
                        
                        if pnl_col:
                            total_return = trade_log[pnl_col].sum()
                            win_rate = (trade_log[pnl_col] > 0).mean() * 100
                            avg_trade = trade_log[pnl_col].mean()
                            
                            asset_performance.extend([total_return, win_rate, avg_trade])
            
            if asset_performance:
                global_results[asset] = asset_performance
        
        # Correlación global entre assets
        if len(global_results) >= 2:
            self.cross_asset_patterns['global_correlation'] = self._calculate_asset_correlations(global_results)
    
    def _create_temporal_series(self, trade_log, pnl_col):
        """Crea serie temporal de resultados"""
        try:
            if 'entry_time' in trade_log.columns:
                trade_log['entry_time'] = pd.to_datetime(trade_log['entry_time'])
                trade_log = trade_log.sort_values('entry_time')
                
                # Crear serie temporal acumulativa
                cumulative_pnl = trade_log[pnl_col].cumsum()
                return cumulative_pnl.values
            else:
                # Usar índice secuencial
                return trade_log[pnl_col].cumsum().values
                
        except Exception as e:
            return trade_log[pnl_col].values
    
    def _calculate_asset_correlations(self, asset_results):
        """Calcula matriz de correlaciones entre assets"""
        correlations = {}
        
        try:
            # Convertir a DataFrame para análisis
            max_length = max(len(values) for values in asset_results.values()) if asset_results else 0
            
            if max_length == 0:
                return correlations
            
            # Normalizar longitudes (padding o truncate)
            normalized_data = {}
            for asset, values in asset_results.items():
                if len(values) > max_length:
                    normalized_data[asset] = values[:max_length]
                else:
                    # Pad con último valor
                    padded = list(values) + [values[-1]] * (max_length - len(values))
                    normalized_data[asset] = padded
            
            df = pd.DataFrame(normalized_data)
            
            # Matriz de correlación de Pearson
            correlations['pearson'] = df.corr(method='pearson').to_dict()
            
            # Matriz de correlación de Spearman
            correlations['spearman'] = df.corr(method='spearman').to_dict()
            
            # Análisis de significancia estadística
            correlations['significance'] = {}
            asset_pairs = list(itertools.combinations(df.columns, 2))
            
            for asset1, asset2 in asset_pairs:
                # Test de Pearson
                pearson_corr, pearson_p = pearsonr(df[asset1], df[asset2])
                
                # Test de Spearman
                spearman_corr, spearman_p = spearmanr(df[asset1], df[asset2])
                
                pair_key = f"{asset1}_vs_{asset2}"
                correlations['significance'][pair_key] = {
                    'pearson_correlation': float(pearson_corr),
                    'pearson_p_value': float(pearson_p),
                    'pearson_significant': pearson_p < 0.05,
                    'spearman_correlation': float(spearman_corr),
                    'spearman_p_value': float(spearman_p),
                    'spearman_significant': spearman_p < 0.05,
                    'correlation_strength': self._classify_correlation_strength(abs(pearson_corr)),
                    'correlation_direction': 'Positive' if pearson_corr > 0 else 'Negative',
                    'trading_implication': self._generate_trading_implication(pearson_corr, pearson_p < 0.05)
                }
            
        except Exception as e:
            correlations['error'] = str(e)
        
        return correlations
    
    def _classify_correlation_strength(self, abs_correlation):
        """Clasifica la fuerza de la correlación"""
        if abs_correlation >= 0.8:
            return 'Very Strong'
        elif abs_correlation >= 0.6:
            return 'Strong'
        elif abs_correlation >= 0.4:
            return 'Moderate'
        elif abs_correlation >= 0.2:
            return 'Weak'
        else:
            return 'Very Weak'
    
    def _generate_trading_implication(self, correlation, is_significant):
        """Genera implicaciones para trading basadas en correlación"""
        if not is_significant:
            return "No significant correlation - trade independently"
        
        if correlation > 0.6:
            return "Strong positive correlation - avoid duplicating exposure"
        elif correlation < -0.6:
            return "Strong negative correlation - consider hedging opportunities"
        elif 0.3 < correlation <= 0.6:
            return "Moderate positive correlation - diversify carefully"
        elif -0.6 <= correlation < -0.3:
            return "Moderate negative correlation - potential portfolio balance"
        else:
            return "Weak correlation - limited impact on position sizing"
    
    def _analyze_cross_strategy_correlations(self):
        """Analiza correlaciones entre diferentes estrategias"""
        print("   🎯 Calculando correlaciones entre estrategias...")
        
        # Por cada asset, analizar correlaciones entre estrategias
        for asset in self.assets:
            if asset not in self.all_data:
                continue
            
            strategy_results = {}
            
            for strategy in self.strategies:
                if (strategy in self.all_data[asset] and 
                    self.all_data[asset][strategy] is not None):
                    
                    trade_log = self.all_data[asset][strategy]['trade_log']
                    
                    if not trade_log.empty:
                        # Encontrar columna PnL
                        pnl_col = None
                        for col in ['pnl', 'profit', 'return', 'P&L', 'PnL']:
                            if col in trade_log.columns:
                                pnl_col = col
                                break
                        
                        if pnl_col:
                            # Métricas de performance de la estrategia
                            total_return = trade_log[pnl_col].sum()
                            win_rate = (trade_log[pnl_col] > 0).mean()
                            avg_trade = trade_log[pnl_col].mean()
                            volatility = trade_log[pnl_col].std()
                            max_drawdown = self._calculate_max_drawdown(trade_log[pnl_col])
                            
                            strategy_results[strategy] = [
                                total_return, win_rate, avg_trade, volatility, max_drawdown
                            ]
            
            if len(strategy_results) >= 2:
                self.cross_strategy_patterns[asset] = self._calculate_strategy_correlations(strategy_results)
        
        # Análisis global de estrategias (todas las assets)
        global_strategy_performance = {}
        
        for strategy in self.strategies:
            strategy_metrics = []
            
            for asset in self.assets:
                if (asset in self.all_data and 
                    strategy in self.all_data[asset] and 
                    self.all_data[asset][strategy] is not None):
                    
                    trade_log = self.all_data[asset][strategy]['trade_log']
                    
                    if not trade_log.empty:
                        pnl_col = None
                        for col in ['pnl', 'profit', 'return', 'P&L', 'PnL']:
                            if col in trade_log.columns:
                                pnl_col = col
                                break
                        
                        if pnl_col:
                            total_return = trade_log[pnl_col].sum()
                            win_rate = (trade_log[pnl_col] > 0).mean()
                            strategy_metrics.extend([total_return, win_rate])
            
            if strategy_metrics:
                global_strategy_performance[strategy] = strategy_metrics
        
        if len(global_strategy_performance) >= 2:
            self.cross_strategy_patterns['global'] = self._calculate_strategy_correlations(global_strategy_performance)
    
    def _calculate_max_drawdown(self, pnl_series):
        """Calcula máximo drawdown"""
        try:
            cumulative = pnl_series.cumsum()
            running_max = cumulative.expanding().max()
            drawdown = cumulative - running_max
            return drawdown.min()
        except:
            return 0
    
    def _calculate_strategy_correlations(self, strategy_results):
        """Calcula correlaciones entre estrategias"""
        correlations = {}
        
        try:
            # Crear DataFrame con métricas de estrategias
            df = pd.DataFrame(strategy_results).T
            
            if df.empty or len(df.columns) < 2:
                return correlations
            
            # Correlaciones
            correlations['pearson'] = df.corr(method='pearson').to_dict()
            correlations['spearman'] = df.corr(method='spearman').to_dict()
            
            # Análisis de pares de estrategias
            correlations['strategy_pairs'] = {}
            strategy_pairs = list(itertools.combinations(df.index, 2))
            
            for strategy1, strategy2 in strategy_pairs:
                try:
                    data1 = df.loc[strategy1].values
                    data2 = df.loc[strategy2].values
                    
                    # Correlación
                    pearson_corr, pearson_p = pearsonr(data1, data2)
                    
                    pair_key = f"{strategy1}_vs_{strategy2}"
                    correlations['strategy_pairs'][pair_key] = {
                        'correlation': float(pearson_corr),
                        'p_value': float(pearson_p),
                        'significant': pearson_p < 0.05,
                        'strength': self._classify_correlation_strength(abs(pearson_corr)),
                        'complementarity': self._assess_strategy_complementarity(pearson_corr, pearson_p < 0.05),
                        'portfolio_recommendation': self._generate_portfolio_recommendation(strategy1, strategy2, pearson_corr, pearson_p < 0.05)
                    }
                    
                except Exception as e:
                    correlations['strategy_pairs'][f"{strategy1}_vs_{strategy2}"] = {'error': str(e)}
        
        except Exception as e:
            correlations['error'] = str(e)
        
        return correlations
    
    def _assess_strategy_complementarity(self, correlation, is_significant):
        """Evalúa complementariedad entre estrategias"""
        if not is_significant:
            return "Independent - can be combined freely"
        
        if correlation > 0.7:
            return "Highly redundant - avoid combining"
        elif correlation > 0.4:
            return "Somewhat redundant - limit combined exposure"
        elif -0.4 <= correlation <= 0.4:
            return "Complementary - good for diversification"
        elif correlation < -0.4:
            return "Highly complementary - excellent for hedging"
        else:
            return "Mixed signals - analyze further"
    
    def _generate_portfolio_recommendation(self, strategy1, strategy2, correlation, is_significant):
        """Genera recomendación de portfolio para par de estrategias"""
        if not is_significant:
            return f"Combine {strategy1} and {strategy2} with equal weights"
        
        if correlation > 0.6:
            return f"Use either {strategy1} OR {strategy2}, not both simultaneously"
        elif correlation < -0.6:
            return f"Perfect hedge: use {strategy1} and {strategy2} with opposite position sizing"
        elif 0.3 < correlation <= 0.6:
            return f"Use {strategy1} and {strategy2} with 70/30 or 60/40 weight distribution"
        elif -0.6 <= correlation < -0.3:
            return f"Balance portfolio: equal weights for {strategy1} and {strategy2}"
        else:
            return f"Standard diversification: combine {strategy1} and {strategy2} normally"
    
    def _analyze_temporal_cross_patterns(self):
        """Analiza patrones temporales cruzados"""
        print("   ⏰ Analizando patrones temporales cruzados...")
        
        temporal_patterns = {}
        
        # Análizar por horas del día
        hourly_performance = self._analyze_hourly_cross_patterns()
        temporal_patterns['hourly'] = hourly_performance
        
        # Análizar por días de la semana
        daily_performance = self._analyze_daily_cross_patterns()
        temporal_patterns['daily'] = daily_performance
        
        # Análizar por sesiones de trading
        session_performance = self._analyze_session_cross_patterns()
        temporal_patterns['sessions'] = session_performance
        
        self.cross_asset_patterns['temporal'] = temporal_patterns
    
    def _analyze_hourly_cross_patterns(self):
        """Analiza patrones por hora del día"""
        hourly_data = {}
        
        for asset in self.assets:
            if asset not in self.all_data:
                continue
                
            asset_hourly = {}
            
            for strategy in self.strategies:
                if (strategy in self.all_data[asset] and 
                    self.all_data[asset][strategy] is not None):
                    
                    trade_log = self.all_data[asset][strategy]['trade_log']
                    
                    if not trade_log.empty and 'entry_time' in trade_log.columns:
                        try:
                            trade_log['entry_time'] = pd.to_datetime(trade_log['entry_time'])
                            trade_log['hour'] = trade_log['entry_time'].dt.hour
                            
                            # PnL por hora
                            pnl_col = None
                            for col in ['pnl', 'profit', 'return']:
                                if col in trade_log.columns:
                                    pnl_col = col
                                    break
                            
                            if pnl_col:
                                hourly_pnl = trade_log.groupby('hour')[pnl_col].agg(['sum', 'mean', 'count']).to_dict('index')
                                asset_hourly[strategy] = hourly_pnl
                                
                        except Exception as e:
                            continue
            
            if asset_hourly:
                hourly_data[asset] = asset_hourly
        
        return hourly_data
    
    def _analyze_daily_cross_patterns(self):
        """Analiza patrones por día de la semana"""
        daily_data = {}
        
        for asset in self.assets:
            if asset not in self.all_data:
                continue
                
            asset_daily = {}
            
            for strategy in self.strategies:
                if (strategy in self.all_data[asset] and 
                    self.all_data[asset][strategy] is not None):
                    
                    trade_log = self.all_data[asset][strategy]['trade_log']
                    
                    if not trade_log.empty and 'entry_time' in trade_log.columns:
                        try:
                            trade_log['entry_time'] = pd.to_datetime(trade_log['entry_time'])
                            trade_log['day_name'] = trade_log['entry_time'].dt.day_name()
                            
                            # PnL por día
                            pnl_col = None
                            for col in ['pnl', 'profit', 'return']:
                                if col in trade_log.columns:
                                    pnl_col = col
                                    break
                            
                            if pnl_col:
                                daily_pnl = trade_log.groupby('day_name')[pnl_col].agg(['sum', 'mean', 'count']).to_dict('index')
                                asset_daily[strategy] = daily_pnl
                                
                        except Exception as e:
                            continue
            
            if asset_daily:
                daily_data[asset] = asset_daily
        
        return daily_data
    
    def _analyze_session_cross_patterns(self):
        """Analiza patrones por sesiones de trading"""
        session_data = {}
        
        # Definir sesiones de trading
        sessions = {
            'Asian': (0, 8),
            'European': (8, 16), 
            'American': (16, 24)
        }
        
        for asset in self.assets:
            if asset not in self.all_data:
                continue
                
            asset_sessions = {}
            
            for strategy in self.strategies:
                if (strategy in self.all_data[asset] and 
                    self.all_data[asset][strategy] is not None):
                    
                    trade_log = self.all_data[asset][strategy]['trade_log']
                    
                    if not trade_log.empty and 'entry_time' in trade_log.columns:
                        try:
                            trade_log['entry_time'] = pd.to_datetime(trade_log['entry_time'])
                            trade_log['hour'] = trade_log['entry_time'].dt.hour
                            
                            # Clasificar por sesión
                            def classify_session(hour):
                                for session_name, (start, end) in sessions.items():
                                    if start <= hour < end:
                                        return session_name
                                return 'Unknown'
                            
                            trade_log['session'] = trade_log['hour'].apply(classify_session)
                            
                            # PnL por sesión
                            pnl_col = None
                            for col in ['pnl', 'profit', 'return']:
                                if col in trade_log.columns:
                                    pnl_col = col
                                    break
                            
                            if pnl_col:
                                session_pnl = trade_log.groupby('session')[pnl_col].agg(['sum', 'mean', 'count']).to_dict('index')
                                asset_sessions[strategy] = session_pnl
                                
                        except Exception as e:
                            continue
            
            if asset_sessions:
                session_data[asset] = asset_sessions
        
        return session_data
    
    def _perform_clustering_analysis(self):
        """Realiza clustering de combinaciones asset-strategy"""
        print("   🎪 Realizando clustering de asset-strategy combinations...")
        
        # Preparar datos para clustering
        clustering_data = []
        labels = []
        
        for asset in self.assets:
            if asset not in self.all_data:
                continue
                
            for strategy in self.strategies:
                if (strategy in self.all_data[asset] and 
                    self.all_data[asset][strategy] is not None):
                    
                    trade_log = self.all_data[asset][strategy]['trade_log']
                    
                    if not trade_log.empty:
                        # Extraer características
                        features = self._extract_clustering_features(trade_log, asset, strategy)
                        
                        if features:
                            clustering_data.append(features)
                            labels.append(f"{asset}_{strategy}")
        
        if len(clustering_data) >= 3:  # Mínimo para clustering
            clustering_results = self._perform_kmeans_clustering(clustering_data, labels)
            self.cross_asset_patterns['clustering'] = clustering_results
    
    def _extract_clustering_features(self, trade_log, asset, strategy):
        """Extrae características para clustering"""
        try:
            # Encontrar columna PnL
            pnl_col = None
            for col in ['pnl', 'profit', 'return', 'P&L', 'PnL']:
                if col in trade_log.columns:
                    pnl_col = col
                    break
            
            if not pnl_col:
                return None
            
            # Características básicas
            total_return = trade_log[pnl_col].sum()
            win_rate = (trade_log[pnl_col] > 0).mean()
            avg_win = trade_log[trade_log[pnl_col] > 0][pnl_col].mean() if (trade_log[pnl_col] > 0).any() else 0
            avg_loss = trade_log[trade_log[pnl_col] <= 0][pnl_col].mean() if (trade_log[pnl_col] <= 0).any() else 0
            volatility = trade_log[pnl_col].std()
            max_drawdown = self._calculate_max_drawdown(trade_log[pnl_col])
            trade_count = len(trade_log)
            
            # Características de distribución
            skewness = trade_log[pnl_col].skew()
            kurtosis = trade_log[pnl_col].kurtosis()
            
            # Ratio riesgo/beneficio
            risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            # Profit factor
            gross_profit = trade_log[trade_log[pnl_col] > 0][pnl_col].sum()
            gross_loss = abs(trade_log[trade_log[pnl_col] <= 0][pnl_col].sum())
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0
            
            # Características temporales (si disponibles)
            time_features = [0, 0, 0]  # [avg_duration, peak_hour, peak_day]
            if 'entry_time' in trade_log.columns and 'exit_time' in trade_log.columns:
                try:
                    trade_log['entry_time'] = pd.to_datetime(trade_log['entry_time'])
                    trade_log['exit_time'] = pd.to_datetime(trade_log['exit_time'])
                    trade_log['duration'] = (trade_log['exit_time'] - trade_log['entry_time']).dt.total_seconds() / 3600
                    
                    avg_duration = trade_log['duration'].mean()
                    peak_hour = trade_log['entry_time'].dt.hour.mode()[0] if not trade_log['entry_time'].dt.hour.mode().empty else 12
                    peak_day = trade_log['entry_time'].dt.dayofweek.mode()[0] if not trade_log['entry_time'].dt.dayofweek.mode().empty else 2
                    
                    time_features = [avg_duration, peak_hour, peak_day]
                except:
                    pass
            
            features = [
                total_return,
                win_rate,
                avg_win,
                abs(avg_loss),
                volatility,
                abs(max_drawdown),
                trade_count,
                skewness,
                kurtosis,
                risk_reward_ratio,
                profit_factor
            ] + time_features
            
            # Filtrar valores infinitos o NaN
            features = [f if np.isfinite(f) else 0 for f in features]
            
            return features
            
        except Exception as e:
            return None
    
    def _perform_kmeans_clustering(self, clustering_data, labels):
        """Realiza K-means clustering"""
        try:
            # Normalizar datos
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(clustering_data)
            
            # Determinar número óptimo de clusters (método del codo)
            max_k = min(8, len(clustering_data) - 1)
            inertias = []
            k_range = range(2, max_k + 1)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(normalized_data)
                inertias.append(kmeans.inertia_)
            
            # Elegir k óptimo (simplificado)
            optimal_k = 3 if len(clustering_data) >= 6 else 2
            
            # Clustering final
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(normalized_data)
            
            # Análisis de clusters
            clustering_results = {
                'optimal_k': optimal_k,
                'cluster_assignments': {},
                'cluster_characteristics': {},
                'trading_implications': {}
            }
            
            # Asignar labels a clusters
            for i, label in enumerate(labels):
                clustering_results['cluster_assignments'][label] = int(cluster_labels[i])
            
            # Analizar características de cada cluster
            df_clustering = pd.DataFrame(clustering_data, index=labels)
            df_clustering['cluster'] = cluster_labels
            
            feature_names = [
                'total_return', 'win_rate', 'avg_win', 'avg_loss', 'volatility',
                'max_drawdown', 'trade_count', 'skewness', 'kurtosis', 
                'risk_reward_ratio', 'profit_factor', 'avg_duration', 'peak_hour', 'peak_day'
            ]
            
            df_clustering.columns = feature_names + ['cluster']
            
            for cluster_id in range(optimal_k):
                cluster_data = df_clustering[df_clustering['cluster'] == cluster_id]
                
                if not cluster_data.empty:
                    characteristics = {}
                    
                    for feature in feature_names:
                        if feature in cluster_data.columns:
                            characteristics[feature] = {
                                'mean': float(cluster_data[feature].mean()),
                                'std': float(cluster_data[feature].std()),
                                'min': float(cluster_data[feature].min()),
                                'max': float(cluster_data[feature].max())
                            }
                    
                    clustering_results['cluster_characteristics'][f'cluster_{cluster_id}'] = {
                        'members': cluster_data.index.tolist(),
                        'size': len(cluster_data),
                        'characteristics': characteristics,
                        'cluster_type': self._classify_cluster_type(characteristics),
                        'performance_tier': self._classify_performance_tier(characteristics)
                    }
                    
                    # Implicaciones de trading
                    clustering_results['trading_implications'][f'cluster_{cluster_id}'] = self._generate_cluster_trading_implications(characteristics, cluster_data.index.tolist())
            
            # Análisis PCA para visualización
            if len(clustering_data[0]) > 2:
                pca = PCA(n_components=2)
                pca_data = pca.fit_transform(normalized_data)
                
                clustering_results['pca_analysis'] = {
                    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                    'pca_coordinates': {labels[i]: pca_data[i].tolist() for i in range(len(labels))}
                }
            
            return clustering_results
            
        except Exception as e:
            return {'error': str(e)}
    
    def _classify_cluster_type(self, characteristics):
        """Clasifica el tipo de cluster basado en características"""
        try:
            win_rate = characteristics.get('win_rate', {}).get('mean', 0.5)
            profit_factor = characteristics.get('profit_factor', {}).get('mean', 1)
            volatility = characteristics.get('volatility', {}).get('mean', 0)
            
            if win_rate > 0.6 and profit_factor > 1.5:
                return "HIGH_PERFORMANCE"
            elif win_rate < 0.4 or profit_factor < 0.8:
                return "LOW_PERFORMANCE"
            elif volatility > characteristics.get('volatility', {}).get('std', 0) * 2:
                return "HIGH_VOLATILITY"
            else:
                return "MODERATE_PERFORMANCE"
        except:
            return "UNKNOWN"
    
    def _classify_performance_tier(self, characteristics):
        """Clasifica el tier de performance"""
        try:
            total_return = characteristics.get('total_return', {}).get('mean', 0)
            win_rate = characteristics.get('win_rate', {}).get('mean', 0.5)
            profit_factor = characteristics.get('profit_factor', {}).get('mean', 1)
            
            score = total_return * 0.4 + win_rate * 0.3 + profit_factor * 0.3
            
            if score > 1.5:
                return "TIER_1_ELITE"
            elif score > 1.0:
                return "TIER_2_STRONG"
            elif score > 0.5:
                return "TIER_3_MODERATE"
            else:
                return "TIER_4_WEAK"
        except:
            return "UNCLASSIFIED"
    
    def _generate_cluster_trading_implications(self, characteristics, members):
        """Genera implicaciones de trading para cada cluster"""
        implications = []
        
        try:
            win_rate = characteristics.get('win_rate', {}).get('mean', 0.5)
            profit_factor = characteristics.get('profit_factor', {}).get('mean', 1)
            volatility = characteristics.get('volatility', {}).get('mean', 0)
            
            # Implicaciones basadas en win rate
            if win_rate > 0.65:
                implications.append("HIGH_WIN_RATE: Increase position sizing for cluster members")
            elif win_rate < 0.4:
                implications.append("LOW_WIN_RATE: Reduce position sizing or avoid cluster members")
            
            # Implicaciones basadas en profit factor
            if profit_factor > 2.0:
                implications.append("EXCELLENT_PROFIT_FACTOR: Prioritize cluster members in portfolio")
            elif profit_factor < 1.0:
                implications.append("POOR_PROFIT_FACTOR: Review or eliminate cluster members")
            
            # Implicaciones basadas en volatilidad
            if volatility > 0.01:  # Alto para Forex
                implications.append("HIGH_VOLATILITY: Use tighter risk management for cluster members")
            
            # Recomendaciones específicas por miembros
            asset_strategies = [member.split('_') for member in members]
            assets_in_cluster = list(set([item[0] for item in asset_strategies]))
            strategies_in_cluster = list(set([item[1] for item in asset_strategies]))
            
            if len(assets_in_cluster) == 1:
                implications.append(f"ASSET_SPECIFIC: All members use {assets_in_cluster[0]} - consider asset-specific optimization")
            
            if len(strategies_in_cluster) == 1:
                implications.append(f"STRATEGY_SPECIFIC: All members use {strategies_in_cluster[0]} - consider strategy-specific optimization")
            
            if len(assets_in_cluster) > 2 and len(strategies_in_cluster) > 2:
                implications.append("DIVERSIFIED_CLUSTER: Good for portfolio diversification")
            
        except Exception as e:
            implications.append(f"ERROR_GENERATING_IMPLICATIONS: {e}")
        
        return implications
    
    def _generate_algorithmic_rules(self):
        """Genera reglas específicas para bots algorítmicos"""
        print("   🤖 Generando reglas para bots algorítmicos...")
        
        bot_rules = {
            'correlation_rules': [],
            'position_sizing_rules': [],
            'timing_rules': [],
            'portfolio_rules': [],
            'risk_management_rules': [],
            'implementation_priority': []
        }
        
        # Reglas basadas en correlaciones entre assets
        self._generate_asset_correlation_rules(bot_rules)
        
        # Reglas basadas en correlaciones entre estrategias  
        self._generate_strategy_correlation_rules(bot_rules)
        
        # Reglas temporales
        self._generate_temporal_rules(bot_rules)
        
        # Reglas de clustering
        self._generate_clustering_rules(bot_rules)
        
        # Priorizar reglas por impacto esperado
        self._prioritize_bot_rules(bot_rules)
        
        return bot_rules
    
    def _generate_asset_correlation_rules(self, bot_rules):
        """Genera reglas basadas en correlaciones entre assets"""
        try:
            if 'global_correlation' in self.cross_asset_patterns:
                correlations = self.cross_asset_patterns['global_correlation']
                
                if 'significance' in correlations:
                    for pair, data in correlations['significance'].items():
                        correlation = data['pearson_correlation']
                        is_significant = data['pearson_significant']
                        
                        if is_significant:
                            assets = pair.split('_vs_')
                            
                            if correlation > 0.7:
                                rule = {
                                    'rule_type': 'HIGH_POSITIVE_CORRELATION',
                                    'assets': assets,
                                    'correlation': correlation,
                                    'rule': f"Avoid simultaneous long positions in {assets[0]} and {assets[1]} (correlation: {correlation:.3f})",
                                    'implementation': f"IF position_{assets[0]} > 0 AND signal_{assets[1]} == 'BUY' THEN reduce_position_size_{assets[1]} *= 0.5",
                                    'expected_impact': 'Reduce portfolio risk by 15-25%'
                                }
                                bot_rules['correlation_rules'].append(rule)
                                
                            elif correlation < -0.7:
                                rule = {
                                    'rule_type': 'HIGH_NEGATIVE_CORRELATION',
                                    'assets': assets,
                                    'correlation': correlation,
                                    'rule': f"Use {assets[0]} and {assets[1]} as natural hedges (correlation: {correlation:.3f})",
                                    'implementation': f"IF position_{assets[0]} > 0 THEN allow_opposite_position_{assets[1]} = True",
                                    'expected_impact': 'Natural portfolio hedging, reduce drawdown 20-30%'
                                }
                                bot_rules['correlation_rules'].append(rule)
                                
                            elif 0.4 < correlation <= 0.7:
                                rule = {
                                    'rule_type': 'MODERATE_POSITIVE_CORRELATION',
                                    'assets': assets,
                                    'correlation': correlation,
                                    'rule': f"Limit combined exposure to {assets[0]} and {assets[1]} (correlation: {correlation:.3f})",
                                    'implementation': f"total_exposure_{assets[0]}_and_{assets[1]} <= max_combined_exposure * 0.75",
                                    'expected_impact': 'Better risk distribution across portfolio'
                                }
                                bot_rules['position_sizing_rules'].append(rule)
        
        except Exception as e:
            bot_rules['correlation_rules'].append({'error': f'Error generating asset correlation rules: {e}'})
    
    def _generate_strategy_correlation_rules(self, bot_rules):
        """Genera reglas basadas en correlaciones entre estrategias"""
        try:
            for asset, strategy_correlations in self.cross_strategy_patterns.items():
                if 'strategy_pairs' in strategy_correlations:
                    for pair, data in strategy_correlations['strategy_pairs'].items():
                        if 'correlation' in data and 'significant' in data:
                            correlation = data['correlation']
                            is_significant = data['significant']
                            
                            if is_significant:
                                strategies = pair.split('_vs_')
                                
                                if correlation > 0.8:
                                    rule = {
                                        'rule_type': 'REDUNDANT_STRATEGIES',
                                        'asset': asset,
                                        'strategies': strategies,
                                        'correlation': correlation,
                                        'rule': f"For {asset}: Use either {strategies[0]} OR {strategies[1]}, not both (highly correlated: {correlation:.3f})",
                                        'implementation': f"IF {asset}_{strategies[0]}_active == True THEN {asset}_{strategies[1]}_active = False",
                                        'expected_impact': 'Eliminate redundancy, improve resource allocation'
                                    }
                                    bot_rules['portfolio_rules'].append(rule)
                                    
                                elif correlation < -0.6:
                                    rule = {
                                        'rule_type': 'COMPLEMENTARY_STRATEGIES',
                                        'asset': asset,
                                        'strategies': strategies,
                                        'correlation': correlation,
                                        'rule': f"For {asset}: Combine {strategies[0]} and {strategies[1]} for natural hedging (correlation: {correlation:.3f})",
                                        'implementation': f"IF {asset}_{strategies[0]}_signal == 'BUY' AND {asset}_{strategies[1]}_signal == 'SELL' THEN increase_confidence = True",
                                        'expected_impact': 'Better risk-adjusted returns through strategy diversification'
                                    }
                                    bot_rules['portfolio_rules'].append(rule)
        
        except Exception as e:
            bot_rules['portfolio_rules'].append({'error': f'Error generating strategy correlation rules: {e}'})
    
    def _generate_temporal_rules(self, bot_rules):
        """Genera reglas basadas en patrones temporales"""
        try:
            if 'temporal' in self.cross_asset_patterns:
                temporal_data = self.cross_asset_patterns['temporal']
                
                # Reglas de sesiones
                if 'sessions' in temporal_data:
                    session_performance = {}
                    
                    for asset, asset_sessions in temporal_data['sessions'].items():
                        for strategy, strategy_sessions in asset_sessions.items():
                            for session, session_data in strategy_sessions.items():
                                if session != 'Unknown':
                                    key = f"{asset}_{strategy}_{session}"
                                    avg_pnl = session_data.get('mean', 0)
                                    trade_count = session_data.get('count', 0)
                                    
                                    if trade_count >= 5:  # Mínimo de trades para ser significativo
                                        session_performance[key] = avg_pnl
                    
                    # Identificar mejores sesiones
                    if session_performance:
                        sorted_sessions = sorted(session_performance.items(), key=lambda x: x[1], reverse=True)
                        top_sessions = sorted_sessions[:5]  # Top 5 sesiones
                        
                        for session_key, avg_pnl in top_sessions:
                            if avg_pnl > 0:
                                parts = session_key.split('_')
                                asset, strategy, session = parts[0], parts[1], parts[2]
                                
                                rule = {
                                    'rule_type': 'OPTIMAL_TRADING_SESSION',
                                    'asset': asset,
                                    'strategy': strategy,
                                    'session': session,
                                    'avg_pnl': avg_pnl,
                                    'rule': f"Prioritize {strategy} on {asset} during {session} session (avg PnL: {avg_pnl:.6f})",
                                    'implementation': f"IF current_session == '{session}' THEN increase_signal_confidence_{asset}_{strategy} *= 1.2",
                                    'expected_impact': 'Focus trading on most profitable time windows'
                                }
                                bot_rules['timing_rules'].append(rule)
                
                # Reglas horarias
                if 'hourly' in temporal_data:
                    best_hours = {}
                    
                    for asset, asset_hourly in temporal_data['hourly'].items():
                        for strategy, strategy_hourly in asset_hourly.items():
                            for hour, hour_data in strategy_hourly.items():
                                avg_pnl = hour_data.get('mean', 0)
                                trade_count = hour_data.get('count', 0)
                                
                                if trade_count >= 3 and avg_pnl > 0:
                                    key = f"{asset}_{strategy}"
                                    if key not in best_hours:
                                        best_hours[key] = []
                                    best_hours[key].append((hour, avg_pnl))
                    
                    # Generar reglas para mejores horas
                    for key, hours_data in best_hours.items():
                        if len(hours_data) >= 2:
                            # Ordenar por PnL promedio
                            hours_data.sort(key=lambda x: x[1], reverse=True)
                            top_hours = [str(hour) for hour, _ in hours_data[:3]]
                            
                            asset, strategy = key.split('_', 1)
                            
                            rule = {
                                'rule_type': 'GOLDEN_HOURS',
                                'asset': asset,
                                'strategy': strategy,
                                'hours': top_hours,
                                'rule': f"Trade {strategy} on {asset} preferably during hours: {', '.join(top_hours)}",
                                'implementation': f"IF current_hour IN {top_hours} THEN enable_{asset}_{strategy} = True",
                                'expected_impact': 'Concentrate trading during most profitable hours'
                            }
                            bot_rules['timing_rules'].append(rule)
        
        except Exception as e:
            bot_rules['timing_rules'].append({'error': f'Error generating temporal rules: {e}'})
    
    def _generate_clustering_rules(self, bot_rules):
        """Genera reglas basadas en análisis de clustering"""
        try:
            if 'clustering' in self.cross_asset_patterns:
                clustering_data = self.cross_asset_patterns['clustering']
                
                if 'cluster_characteristics' in clustering_data:
                    for cluster_id, cluster_info in clustering_data['cluster_characteristics'].items():
                        performance_tier = cluster_info.get('performance_tier', 'UNCLASSIFIED')
                        members = cluster_info.get('members', [])
                        characteristics = cluster_info.get('characteristics', {})
                        
                        if performance_tier == 'TIER_1_ELITE' and len(members) > 1:
                            rule = {
                                'rule_type': 'ELITE_CLUSTER_PRIORITIZATION',
                                'cluster': cluster_id,
                                'members': members,
                                'performance_tier': performance_tier,
                                'rule': f"Prioritize {cluster_id} members: {', '.join(members)} (Elite performance)",
                                'implementation': f"FOR member IN {members}: increase_allocation_multiplier = 1.5",
                                'expected_impact': 'Focus capital on highest performing combinations'
                            }
                            bot_rules['position_sizing_rules'].append(rule)
                            
                        elif performance_tier == 'TIER_4_WEAK' and len(members) > 1:
                            rule = {
                                'rule_type': 'WEAK_CLUSTER_LIMITATION',
                                'cluster': cluster_id,
                                'members': members,
                                'performance_tier': performance_tier,
                                'rule': f"Limit exposure to {cluster_id} members: {', '.join(members)} (Weak performance)",
                                'implementation': f"FOR member IN {members}: max_allocation_multiplier = 0.5",
                                'expected_impact': 'Reduce capital at risk in underperforming combinations'
                            }
                            bot_rules['risk_management_rules'].append(rule)
                        
                        # Reglas específicas basadas en características del cluster
                        if 'win_rate' in characteristics:
                            win_rate = characteristics['win_rate'].get('mean', 0.5)
                            
                            if win_rate > 0.65:
                                rule = {
                                    'rule_type': 'HIGH_WIN_RATE_CLUSTER',
                                    'cluster': cluster_id,
                                    'members': members,
                                    'win_rate': win_rate,
                                    'rule': f"Increase position sizing for high win rate cluster ({win_rate:.1%}): {', '.join(members)}",
                                    'implementation': f"FOR member IN {members}: IF win_rate > 0.65 THEN position_multiplier = 1.3",
                                    'expected_impact': 'Capitalize on consistently winning combinations'
                                }
                                bot_rules['position_sizing_rules'].append(rule)
                            
                            elif win_rate < 0.4:
                                rule = {
                                    'rule_type': 'LOW_WIN_RATE_CLUSTER',
                                    'cluster': cluster_id,
                                    'members': members,
                                    'win_rate': win_rate,
                                    'rule': f"Reduce or avoid low win rate cluster ({win_rate:.1%}): {', '.join(members)}",
                                    'implementation': f"FOR member IN {members}: IF win_rate < 0.4 THEN position_multiplier = 0.3",
                                    'expected_impact': 'Minimize exposure to unreliable combinations'
                                }
                                bot_rules['risk_management_rules'].append(rule)
        
        except Exception as e:
            bot_rules['risk_management_rules'].append({'error': f'Error generating clustering rules: {e}'})
    
    def _prioritize_bot_rules(self, bot_rules):
        """Prioriza reglas por impacto esperado y facilidad de implementación"""
        priority_scores = {}
        
        all_rules = []
        for category, rules in bot_rules.items():
            if category != 'implementation_priority':
                for rule in rules:
                    if 'error' not in rule:
                        all_rules.append((category, rule))
        
        # Calcular score de prioridad para cada regla
        for category, rule in all_rules:
            rule_id = f"{category}_{rule.get('rule_type', 'UNKNOWN')}"
            
            # Score basado en tipo de regla y impacto esperado
            impact_score = self._calculate_impact_score(rule, category)
            implementation_ease = self._calculate_implementation_ease(rule, category)
            
            priority_score = impact_score * 0.7 + implementation_ease * 0.3
            priority_scores[rule_id] = {
                'score': priority_score,
                'category': category,
                'rule': rule,
                'priority_level': self._get_priority_level(priority_score)
            }
        
        # Ordenar por prioridad
        sorted_priorities = sorted(priority_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        bot_rules['implementation_priority'] = [
            {
                'rank': i + 1,
                'rule_id': rule_id,
                'priority_level': data['priority_level'],
                'score': data['score'],
                'category': data['category'],
                'rule_summary': data['rule'].get('rule', 'No description'),
                'implementation': data['rule'].get('implementation', 'No implementation details'),
                'expected_impact': data['rule'].get('expected_impact', 'Unknown impact')
            }
            for i, (rule_id, data) in enumerate(sorted_priorities[:15])  # Top 15 reglas
        ]
    
    def _calculate_impact_score(self, rule, category):
        """Calcula score de impacto esperado"""
        base_scores = {
            'correlation_rules': 8,
            'position_sizing_rules': 9,
            'timing_rules': 6,
            'portfolio_rules': 7,
            'risk_management_rules': 10
        }
        
        base_score = base_scores.get(category, 5)
        
        # Ajustes por tipo específico de regla
        rule_type = rule.get('rule_type', '')
        
        if 'ELITE' in rule_type or 'HIGH_PERFORMANCE' in rule_type:
            base_score += 2
        elif 'WEAK' in rule_type or 'LOW_PERFORMANCE' in rule_type:
            base_score += 1
        
        if 'CORRELATION' in rule_type:
            correlation = abs(rule.get('correlation', 0))
            if correlation > 0.8:
                base_score += 2
            elif correlation > 0.6:
                base_score += 1
        
        return min(base_score, 10)  # Max score 10
    
    def _calculate_implementation_ease(self, rule, category):
        """Calcula facilidad de implementación"""
        # Facilidad basada en complejidad de la implementación
        implementation = rule.get('implementation', '')
        
        if 'IF' in implementation and 'THEN' in implementation:
            return 8  # Lógica simple
        elif 'FOR' in implementation:
            return 6  # Requiere loops
        elif len(implementation.split()) > 20:
            return 4  # Implementación compleja
        else:
            return 7  # Medianamente simple
    
    def _get_priority_level(self, score):
        """Asigna nivel de prioridad basado en score"""
        if score >= 8.5:
            return 'CRITICAL'
        elif score >= 7.0:
            return 'HIGH'
        elif score >= 5.5:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _create_comprehensive_report(self, bot_rules):
        """Crea reporte comprehensivo final"""
        timestamp = datetime.now().isoformat()
        
        report = {
            'metadata': {
                'analysis_type': 'CROSS_CORRELATION_COMPREHENSIVE',
                'timestamp': timestamp,
                'assets_analyzed': self.assets,
                'strategies_analyzed': self.strategies,
                'total_combinations': len(self.assets) * len(self.strategies)
            },
            'executive_summary': self._generate_executive_summary(),
            'cross_asset_analysis': self.cross_asset_patterns,
            'cross_strategy_analysis': self.cross_strategy_patterns,
            'correlation_matrices': self.correlation_matrices,
            'algorithmic_rules': bot_rules,
            'key_findings': self._extract_key_findings(),
            'risk_insights': self._generate_risk_insights(),
            'optimization_opportunities': self._identify_optimization_opportunities(),
            'implementation_roadmap': self._create_implementation_roadmap(bot_rules)
        }
        
        return report
    
    def _generate_executive_summary(self):
        """Genera resumen ejecutivo del análisis"""
        summary = {
            'overview': f'Análisis comprehensivo de correlaciones cruzadas entre {len(self.assets)} assets y {len(self.strategies)} estrategias',
            'data_coverage': {},
            'key_correlations': [],
            'performance_insights': [],
            'bot_recommendations': []
        }
        
        # Cobertura de datos
        successful_combinations = 0
        total_combinations = len(self.assets) * len(self.strategies)
        
        for asset in self.all_data:
            for strategy in self.all_data[asset]:
                if self.all_data[asset][strategy] is not None:
                    successful_combinations += 1
        
        summary['data_coverage'] = {
            'successful_combinations': successful_combinations,
            'total_combinations': total_combinations,
            'coverage_percentage': (successful_combinations / total_combinations) * 100,
            'data_quality': 'EXCELLENT' if successful_combinations / total_combinations > 0.8 else 'GOOD' if successful_combinations / total_combinations > 0.6 else 'MODERATE'
        }
        
        # Correlaciones clave
        try:
            if 'global_correlation' in self.cross_asset_patterns and 'significance' in self.cross_asset_patterns['global_correlation']:
                significant_correlations = []
                for pair, data in self.cross_asset_patterns['global_correlation']['significance'].items():
                    if data.get('pearson_significant', False):
                        significant_correlations.append({
                            'pair': pair,
                            'correlation': data['pearson_correlation'],
                            'strength': data['correlation_strength'],
                            'implication': data['trading_implication']
                        })
                
                # Top 3 correlaciones más fuertes
                significant_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
                summary['key_correlations'] = significant_correlations[:3]
        except:
            pass
        
        # Insights de performance
        if 'clustering' in self.cross_asset_patterns and 'cluster_characteristics' in self.cross_asset_patterns['clustering']:
            elite_clusters = []
            weak_clusters = []
            
            for cluster_id, cluster_info in self.cross_asset_patterns['clustering']['cluster_characteristics'].items():
                performance_tier = cluster_info.get('performance_tier', 'UNCLASSIFIED')
                members = cluster_info.get('members', [])
                
                if performance_tier == 'TIER_1_ELITE':
                    elite_clusters.extend(members)
                elif performance_tier == 'TIER_4_WEAK':
                    weak_clusters.extend(members)
            
            summary['performance_insights'] = {
                'elite_combinations': elite_clusters,
                'weak_combinations': weak_clusters,
                'elite_count': len(elite_clusters),
                'weak_count': len(weak_clusters),
                'performance_distribution': f'{len(elite_clusters)} elite, {len(weak_clusters)} weak combinations'
            }
        
        return summary
    
    def _extract_key_findings(self):
        """Extrae hallazgos clave del análisis"""
        findings = []
        
        try:
            # Finding 1: Correlaciones más fuertes entre assets
            if 'global_correlation' in self.cross_asset_patterns:
                strongest_correlation = 0
                strongest_pair = ""
                
                if 'significance' in self.cross_asset_patterns['global_correlation']:
                    for pair, data in self.cross_asset_patterns['global_correlation']['significance'].items():
                        correlation = abs(data.get('pearson_correlation', 0))
                        if correlation > strongest_correlation and data.get('pearson_significant', False):
                            strongest_correlation = correlation
                            strongest_pair = pair
                
                if strongest_pair:
                    findings.append({
                        'finding_type': 'STRONGEST_ASSET_CORRELATION',
                        'description': f'Correlación más fuerte encontrada entre {strongest_pair.replace("_vs_", " y ")}: {strongest_correlation:.3f}',
                        'implication': 'Considerar estos assets como substitutos o hedge natural',
                        'action': 'Implementar reglas de position sizing para evitar sobre-exposición'
                    })
            
            # Finding 2: Estrategias más complementarias
            complementary_strategies = []
            for asset, strategy_data in self.cross_strategy_patterns.items():
                if 'strategy_pairs' in strategy_data:
                    for pair, data in strategy_data['strategy_pairs'].items():
                        correlation = data.get('correlation', 0)
                        if data.get('significant', False) and correlation < -0.5:
                            complementary_strategies.append({
                                'asset': asset,
                                'strategies': pair.split('_vs_'),
                                'correlation': correlation
                            })
            
            if complementary_strategies:
                best_complement = min(complementary_strategies, key=lambda x: x['correlation'])
                findings.append({
                    'finding_type': 'BEST_STRATEGY_COMPLEMENT',
                    'description': f'Estrategias más complementarias en {best_complement["asset"]}: {" y ".join(best_complement["strategies"])} (correlación: {best_complement["correlation"]:.3f})',
                    'implication': 'Usar estas estrategias juntas para hedging natural',
                    'action': 'Implementar portfolio balanceado con estas estrategias'
                })
            
            # Finding 3: Patrones temporales
            if 'temporal' in self.cross_asset_patterns and 'sessions' in self.cross_asset_patterns['temporal']:
                best_sessions = {}
                for asset, asset_sessions in self.cross_asset_patterns['temporal']['sessions'].items():
                    for strategy, strategy_sessions in asset_sessions.items():
                        for session, session_data in strategy_sessions.items():
                            if session != 'Unknown':
                                avg_pnl = session_data.get('mean', 0)
                                if avg_pnl > 0:
                                    key = f"{asset}_{strategy}"
                                    if key not in best_sessions or avg_pnl > best_sessions[key]['pnl']:
                                        best_sessions[key] = {'session': session, 'pnl': avg_pnl}
                
                if best_sessions:
                    findings.append({
                        'finding_type': 'OPTIMAL_TRADING_SESSIONS',
                        'description': f'Identificadas {len(best_sessions)} combinaciones con sesiones óptimas específicas',
                        'implication': 'Concentrar trading en sesiones de mayor rentabilidad',
                        'action': 'Implementar filtros temporales en el bot de trading'
                    })
            
            # Finding 4: Clustering insights
            if 'clustering' in self.cross_asset_patterns:
                clustering_data = self.cross_asset_patterns['clustering']
                if 'cluster_characteristics' in clustering_data:
                    cluster_count = len(clustering_data['cluster_characteristics'])
                    performance_distribution = {}
                    
                    for cluster_info in clustering_data['cluster_characteristics'].values():
                        tier = cluster_info.get('performance_tier', 'UNCLASSIFIED')
                        performance_distribution[tier] = performance_distribution.get(tier, 0) + 1
                    
                    findings.append({
                        'finding_type': 'CLUSTERING_INSIGHTS',
                        'description': f'Identificados {cluster_count} clusters distintos de performance: {dict(performance_distribution)}',
                        'implication': 'Diferentes combinaciones requieren tratamiento diferenciado',
                        'action': 'Implementar position sizing basado en cluster de performance'
                    })
        
        except Exception as e:
            findings.append({
                'finding_type': 'ANALYSIS_ERROR',
                'description': f'Error extrayendo hallazgos: {e}',
                'implication': 'Revisar análisis manualmente',
                'action': 'Debug y re-ejecutar análisis'
            })
        
        return findings
    
    def _generate_risk_insights(self):
        """Genera insights específicos de riesgo"""
        risk_insights = {
            'correlation_risks': [],
            'concentration_risks': [],
            'temporal_risks': [],
            'portfolio_risks': [],
            'mitigation_strategies': []
        }
        
        try:
            # Riesgos de correlación
            if 'global_correlation' in self.cross_asset_patterns:
                high_correlations = []
                if 'significance' in self.cross_asset_patterns['global_correlation']:
                    for pair, data in self.cross_asset_patterns['global_correlation']['significance'].items():
                        correlation = data.get('pearson_correlation', 0)
                        if data.get('pearson_significant', False) and abs(correlation) > 0.6:
                            high_correlations.append({
                                'pair': pair,
                                'correlation': correlation,
                                'risk_level': 'HIGH' if abs(correlation) > 0.8 else 'MEDIUM'
                            })
                
                risk_insights['correlation_risks'] = high_correlations
            
            # Riesgos de concentración
            if 'clustering' in self.cross_asset_patterns:
                clustering_data = self.cross_asset_patterns['clustering']
                if 'cluster_characteristics' in clustering_data:
                    large_clusters = []
                    for cluster_id, cluster_info in clustering_data['cluster_characteristics'].items():
                        members = cluster_info.get('members', [])
                        if len(members) > 3:  # Clusters grandes
                            large_clusters.append({
                                'cluster': cluster_id,
                                'size': len(members),
                                'members': members,
                                'risk': 'Concentración excesiva en características similares'
                            })
                    
                    risk_insights['concentration_risks'] = large_clusters
            
            # Estrategias de mitigación
            mitigation_strategies = [
                {
                    'risk_type': 'HIGH_CORRELATION',
                    'strategy': 'Implementar límites de exposición combinada para assets altamente correlacionados',
                    'implementation': 'max_combined_exposure = total_capital * 0.15 para correlaciones > 0.7'
                },
                {
                    'risk_type': 'STRATEGY_REDUNDANCY',
                    'strategy': 'Evitar ejecutar estrategias altamente correlacionadas simultáneamente',
                    'implementation': 'IF strategy_A_active AND correlation(A,B) > 0.8 THEN disable_strategy_B'
                },
                {
                    'risk_type': 'CONCENTRATION',
                    'strategy': 'Diversificar entre clusters de performance diferentes',
                    'implementation': 'Allocate maximum 40% to any single performance cluster'
                },
                {
                    'risk_type': 'TEMPORAL',
                    'strategy': 'Implementar stop-trading durante períodos de alta volatilidad correlacionada',
                    'implementation': 'IF market_stress_indicator > threshold THEN reduce_all_positions *= 0.5'
                }
            ]
            
            risk_insights['mitigation_strategies'] = mitigation_strategies
        
        except Exception as e:
            risk_insights['error'] = str(e)
        
        return risk_insights
    
    def _identify_optimization_opportunities(self):
        """Identifica oportunidades de optimización"""
        opportunities = []
        
        try:
            # Oportunidad 1: Combinar estrategias complementarias
            complementary_pairs = []
            for asset, strategy_data in self.cross_strategy_patterns.items():
                if 'strategy_pairs' in strategy_data:
                    for pair, data in strategy_data['strategy_pairs'].items():
                        correlation = data.get('correlation', 0)
                        if data.get('significant', False) and -0.7 < correlation < -0.3:
                            complementary_pairs.append({
                                'asset': asset,
                                'strategies': pair.split('_vs_'),
                                'correlation': correlation
                            })
            
            if complementary_pairs:
                opportunities.append({
                    'opportunity_type': 'STRATEGY_COMBINATION',
                    'description': f'Encontradas {len(complementary_pairs)} parejas de estrategias complementarias',
                    'potential_benefit': 'Reducir drawdown 15-25% manteniendo returns',
                    'implementation_complexity': 'MEDIUM',
                    'expected_timeframe': '2-4 semanas',
                    'details': complementary_pairs[:3]  # Top 3
                })
            
            # Oportunidad 2: Optimización temporal
            if 'temporal' in self.cross_asset_patterns:
                temporal_opportunities = 0
                for asset_data in self.cross_asset_patterns['temporal'].get('sessions', {}).values():
                    for strategy_data in asset_data.values():
                        profitable_sessions = sum(1 for session_data in strategy_data.values() 
                                                if session_data.get('mean', 0) > 0)
                        if profitable_sessions >= 2:
                            temporal_opportunities += 1
                
                if temporal_opportunities > 0:
                    opportunities.append({
                        'opportunity_type': 'TEMPORAL_OPTIMIZATION',
                        'description': f'{temporal_opportunities} combinaciones con patrones temporales claros',
                        'potential_benefit': 'Aumentar profit factor 20-30% concentrando en mejores horarios',
                        'implementation_complexity': 'LOW',
                        'expected_timeframe': '1-2 semanas',
                        'details': 'Implementar filtros de trading por sesiones y horas'
                    })
            
            # Oportunidad 3: Rebalanceo basado en clustering
            if 'clustering' in self.cross_asset_patterns:
                clustering_data = self.cross_asset_patterns['clustering']
                if 'cluster_characteristics' in clustering_data:
                    elite_members = []
                    weak_members = []
                    
                    for cluster_info in clustering_data['cluster_characteristics'].values():
                        tier = cluster_info.get('performance_tier', 'UNCLASSIFIED')
                        members = cluster_info.get('members', [])
                        
                        if tier == 'TIER_1_ELITE':
                            elite_members.extend(members)
                        elif tier == 'TIER_4_WEAK':
                            weak_members.extend(members)
                    
                    if elite_members and weak_members:
                        opportunities.append({
                            'opportunity_type': 'PORTFOLIO_REBALANCING',
                            'description': f'Rebalancear: aumentar {len(elite_members)} elite, reducir {len(weak_members)} weak',
                            'potential_benefit': 'Mejorar Sharpe ratio 25-40% optimizando allocation',
                            'implementation_complexity': 'HIGH',
                            'expected_timeframe': '3-6 semanas',
                            'details': {
                                'increase_allocation': elite_members,
                                'decrease_allocation': weak_members
                            }
                        })
            
            # Oportunidad 4: Hedging natural
            natural_hedges = []
            if 'global_correlation' in self.cross_asset_patterns:
                if 'significance' in self.cross_asset_patterns['global_correlation']:
                    for pair, data in self.cross_asset_patterns['global_correlation']['significance'].items():
                        correlation = data.get('pearson_correlation', 0)
                        if data.get('pearson_significant', False) and correlation < -0.6:
                            natural_hedges.append({
                                'pair': pair,
                                'correlation': correlation
                            })
            
            if natural_hedges:
                opportunities.append({
                    'opportunity_type': 'NATURAL_HEDGING',
                    'description': f'Implementar {len(natural_hedges)} hedges naturales identificados',
                    'potential_benefit': 'Reducir portfolio volatility 30-50% sin sacrificar returns',
                    'implementation_complexity': 'MEDIUM',
                    'expected_timeframe': '2-3 semanas',
                    'details': natural_hedges
                })
            
        except Exception as e:
            opportunities.append({
                'opportunity_type': 'ERROR',
                'description': f'Error identificando oportunidades: {e}',
                'potential_benefit': 'Unknown',
                'implementation_complexity': 'UNKNOWN',
                'expected_timeframe': 'Unknown'
            })
        
        return opportunities
    
    def _create_implementation_roadmap(self, bot_rules):
        """Crea roadmap de implementación"""
        roadmap = {
            'phase_1_immediate': [],
            'phase_2_short_term': [],
            'phase_3_medium_term': [],
            'phase_4_long_term': [],
            'implementation_order': [],
            'resource_requirements': {},
            'success_metrics': []
        }
        
        try:
            # Distribuir reglas por fases basado en prioridad y complejidad
            if 'implementation_priority' in bot_rules:
                for rule_info in bot_rules['implementation_priority']:
                    priority = rule_info.get('priority_level', 'LOW')
                    category = rule_info.get('category', 'unknown')
                    
                    if priority == 'CRITICAL':
                        roadmap['phase_1_immediate'].append({
                            'rule': rule_info['rule_summary'],
                            'implementation': rule_info['implementation'],
                            'category': category,
                            'expected_impact': rule_info['expected_impact']
                        })
                    elif priority == 'HIGH':
                        roadmap['phase_2_short_term'].append({
                            'rule': rule_info['rule_summary'],
                            'implementation': rule_info['implementation'],
                            'category': category,
                            'expected_impact': rule_info['expected_impact']
                        })
                    elif priority == 'MEDIUM':
                        roadmap['phase_3_medium_term'].append({
                            'rule': rule_info['rule_summary'],
                            'implementation': rule_info['implementation'],
                            'category': category,
                            'expected_impact': rule_info['expected_impact']
                        })
                    else:
                        roadmap['phase_4_long_term'].append({
                            'rule': rule_info['rule_summary'],
                            'implementation': rule_info['implementation'],
                            'category': category,
                            'expected_impact': rule_info['expected_impact']
                        })
            
            # Orden de implementación
            implementation_order = [
                {
                    'step': 1,
                    'phase': 'IMMEDIATE (1-2 semanas)',
                    'focus': 'Implementar reglas críticas de correlación y risk management',
                    'rules_count': len(roadmap['phase_1_immediate']),
                    'success_criteria': 'Reducir riesgo de correlación > 15%'
                },
                {
                    'step': 2,
                    'phase': 'SHORT-TERM (2-4 semanas)',
                    'focus': 'Position sizing optimization y temporal filters',
                    'rules_count': len(roadmap['phase_2_short_term']),
                    'success_criteria': 'Mejorar Sharpe ratio > 10%'
                },
                {
                    'step': 3,
                    'phase': 'MEDIUM-TERM (1-3 meses)',
                    'focus': 'Portfolio balancing y strategy combinations',
                    'rules_count': len(roadmap['phase_3_medium_term']),
                    'success_criteria': 'Reducir drawdown > 20%'
                },
                {
                    'step': 4,
                    'phase': 'LONG-TERM (3-6 meses)',
                    'focus': 'Advanced optimization y machine learning integration',
                    'rules_count': len(roadmap['phase_4_long_term']),
                    'success_criteria': 'Automatización completa del sistema'
                }
            ]
            
            roadmap['implementation_order'] = implementation_order
            
            # Requerimientos de recursos
            roadmap['resource_requirements'] = {
                'development_time': '3-6 meses total',
                'technical_complexity': 'MEDIUM-HIGH',
                'team_requirements': [
                    'Quantitative analyst (1 FTE)',
                    'Trading systems developer (0.5 FTE)',
                    'Risk management specialist (0.25 FTE)'
                ],
                'infrastructure_needs': [
                    'Real-time data feeds',
                    'Enhanced backtesting framework',
                    'Portfolio monitoring dashboard',
                    'Automated alert system'
                ],
                'estimated_cost': 'Medium investment for infrastructure upgrades'
            }
            
            # Métricas de éxito
            roadmap['success_metrics'] = [
                {
                    'metric': 'Portfolio Correlation Reduction',
                    'target': '> 20% reduction in unintended correlations',
                    'measurement': 'Weekly correlation matrix analysis'
                },
                {
                    'metric': 'Risk-Adjusted Returns',
                    'target': '> 15% improvement in Sharpe ratio',
                    'measurement': 'Monthly Sharpe ratio calculation'
                },
                {
                    'metric': 'Maximum Drawdown',
                    'target': '< 25% reduction in max drawdown',
                    'measurement': 'Continuous drawdown monitoring'
                },
                {
                    'metric': 'Strategy Diversification',
                    'target': 'Reduce strategy correlation < 0.3',
                    'measurement': 'Quarterly strategy correlation analysis'
                },
                {
                    'metric': 'Implementation Compliance',
                    'target': '> 95% rule adherence rate',
                    'measurement': 'Daily compliance monitoring'
                }
            ]
        
        except Exception as e:
            roadmap['error'] = str(e)
        
        return roadmap
    
    def _export_comprehensive_analysis(self, final_report):
        """Exporta análisis comprehensivo en múltiples formatos"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 1. REPORTE PRINCIPAL JSON
            main_report_path = self.export_dir / f"cross_correlation_analysis_{timestamp}.json"
            with open(main_report_path, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, indent=2, default=str)
            
            print(f"📋 Reporte principal: {main_report_path}")
            
            # 2. REGLAS DE BOT IMPLEMENTABLES
            bot_rules_path = self.export_dir / f"algorithmic_bot_rules_{timestamp}.py"
            with open(bot_rules_path, 'w', encoding='utf-8') as f:
                f.write(self._generate_bot_rules_code(final_report['algorithmic_rules'], timestamp))
            print(f"🤖 Reglas de bot: {bot_rules_path}")
            
            # 3. REPORTE EJECUTIVO
            executive_path = self.export_dir / f"executive_summary_{timestamp}.txt"
            with open(executive_path, 'w', encoding='utf-8') as f:
                f.write(self._generate_executive_report_text(final_report))
            print(f"📄 Reporte ejecutivo: {executive_path}")
            
            # 4. MATRIZ DE CORRELACIONES CSV
            correlation_path = self._export_correlation_matrices(timestamp)
            
            # 5. GRÁFICOS DE ANÁLISIS
            chart_paths = self._create_comprehensive_charts(final_report, timestamp)
            
            return {
                'main_report': str(main_report_path),
                'bot_rules': str(bot_rules_path),
                'executive_summary': str(executive_path),
                'correlation_matrices': correlation_path,
                'charts': chart_paths,
                'export_directory': str(self.export_dir)
            }
            
        except Exception as e:
            print(f"❌ Error exportando análisis: {e}")
            return None
    
    def _generate_bot_rules_code(self, bot_rules, timestamp):
        """Genera código Python implementable para las reglas del bot"""
        # ⚠️ Precalcular números para no meter expresiones largas en el f-string
        corr_cnt = len(bot_rules.get("correlation_rules", []))
        port_cnt = len(bot_rules.get("portfolio_rules", []))

        code = f''''# algorithmic_bot_rules_{timestamp}.py
# Reglas algorítmicas generadas por Cross-Correlation Analyzer
# Implementa correlaciones cruzadas y optimizaciones identificadas

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class CrossCorrelationBotRules:
    """
    Implementa reglas de correlación cruzada para bot de trading algorítmico
    Basado en análisis exhaustivo de {corr_cnt} correlaciones y {port_cnt} reglas de portfolio
    """
    
    def __init__(self):
        self.name = "Cross-Correlation Trading Rules"
        self.generated_on = "{timestamp}"
        self.assets = {{self.assets}}
        self.strategies = {{self.strategies}}
        
        # Matrices de correlación (simplificadas)
        self.asset_correlations = self._initialize_correlations()
        
        # Configuración de trading
        self.max_correlation_threshold = 0.7
        self.hedge_correlation_threshold = -0.6
        self.position_size_base = 0.02  # 2% per trade base
        
    def _initialize_correlations(self):
        """Inicializa matrices de correlación basadas en análisis"""
        # Placeholder - reemplazar con correlaciones reales del análisis
        correlations = {{}}
        return correlations
    
    def check_correlation_constraints(self, current_positions: Dict, new_signal: Dict) -> Dict:
        """
        Verifica constraints de correlación antes de abrir nueva posición
        
        Args:
            current_positions: {{{{'asset_strategy': position_size, ...}}}}
            new_signal: {{{{'asset': str, 'strategy': str, 'direction': str, 'confidence': float}}}}
        
        Returns:
            {{{{'allowed': bool, 'adjusted_size': float, 'reason': str}}}}
        """
        try:
            asset = new_signal['asset']
            strategy = new_signal['strategy']
            direction = new_signal['direction']
            
            result = {{{{
                'allowed': True,
                'adjusted_size': self.position_size_base,
                'reason': 'No correlation conflicts',
                'applied_rules': []
            }}}}
            
            # Regla 1: Verificar correlaciones altas entre assets
            for existing_pos, pos_size in current_positions.items():
                if pos_size > 0:  # Solo posiciones activas
                    existing_asset = existing_pos.split('_')[0]
                    
                    # Simular verificación de correlación
                    correlation = self._get_asset_correlation(asset, existing_asset)
                    
                    if abs(correlation) > self.max_correlation_threshold:
                        if correlation > 0 and direction == self._get_position_direction(existing_pos):
                            # Alta correlación positiva en misma dirección
                            result['adjusted_size'] *= 0.5
                            result['applied_rules'].append(f"Reduced size due to high positive correlation with {{{{existing_asset}}}}")
                        elif correlation < self.hedge_correlation_threshold:
                            # Correlación negativa fuerte - permitir hedge
                            result['adjusted_size'] *= 1.2
                            result['applied_rules'].append(f"Increased size for natural hedge with {{{{existing_asset}}}}")
            
            # Regla 2: Verificar redundancia de estrategias
            strategy_correlation = self._get_strategy_correlation(asset, strategy, current_positions)
            if strategy_correlation > 0.8:
                result['adjusted_size'] *= 0.3
                result['applied_rules'].append("Reduced size due to strategy redundancy")
            
            # Regla 3: Límites de exposición total
            total_exposure = sum(current_positions.values()) + result['adjusted_size']
            if total_exposure > 0.15:  # Máximo 15% de exposición total
                result['adjusted_size'] = max(0, 0.15 - sum(current_positions.values()))
                result['applied_rules'].append("Adjusted to respect maximum exposure limit")
            
            if result['adjusted_size'] <= 0:
                result['allowed'] = False
                result['reason'] = "Position size reduced to zero due to risk constraints"
            
            return result
            
        except Exception as e:
            return {{{{
                'allowed': False,
                'adjusted_size': 0,
                'reason': f"Error in correlation check: {{{{e}}}}",
                'applied_rules': []
            }}}}
    
    def _get_asset_correlation(self, asset1: str, asset2: str) -> float:
        """Obtiene correlación entre dos assets"""
        # Implementar basado en análisis real
        correlations_map = {{{{
            # Ejemplo basado en correlaciones típicas de Forex
            ('EURUSD', 'GBPUSD'): 0.75,
            ('USDJPY', 'AUDUSD'): -0.45,
            ('EURUSD', 'USDJPY'): -0.65,
            # Agregar más basado en análisis real
        }}}}
        
        key1 = (asset1, asset2)
        key2 = (asset2, asset1)
        
        return correlations_map.get(key1, correlations_map.get(key2, 0.0))
    
    def _get_strategy_correlation(self, asset: str, strategy: str, current_positions: Dict) -> float:
        """Calcula correlación de estrategia con posiciones existentes"""
        # Simplificado - implementar basado en análisis real
        correlations = 0.0
        count = 0
        
        for pos_key in current_positions:
            if pos_key.startswith(asset):
                existing_strategy = pos_key.split('_', 1)[1]
                # Obtener correlación entre estrategias para este asset
                corr = self._get_strategies_correlation(strategy, existing_strategy)
                correlations += corr
                count += 1
        
        return correlations / count if count > 0 else 0.0
    
    def _get_strategies_correlation(self, strategy1: str, strategy2: str) -> float:
        """Obtiene correlación entre dos estrategias"""
        # Implementar basado en análisis cross-strategy
        strategy_correlations = {{{{
            # Ejemplo de correlaciones entre estrategias
            ('ema_crossover', 'volatility_breakout'): 0.65,
            ('rsi_pullback', 'channel_reversal'): -0.35,
            ('multi_filter_scalper', 'lokz_reversal'): 0.45,
            # Agregar más basado en análisis real
        }}
        
        key1 = (strategy1, strategy2)
        key2 = (strategy2, strategy1)
        
        return strategy_correlations.get(key1, strategy_correlations.get(key2, 0.0))
    
    def _get_position_direction(self, position_key: str) -> str:
        """Determina dirección de posición existente"""
        # Simplificado - en implementación real obtener de sistema de trading
        return 'long'  # Placeholder
    
    def get_optimal_trading_session(self, asset: str, strategy: str, current_time: datetime) -> Dict:
        """
        Determina si es sesión óptima para trading específico
        
        Returns:
            {{'is_optimal': bool, 'session': str, 'confidence_multiplier': float}}
        """
        hour = current_time.hour
        
        # Sesiones optimizadas basadas en análisis temporal
        optimal_sessions = {{
            # Formato: (asset, strategy): [(hour_start, hour_end, confidence_multiplier), ...]
            ('EURUSD', 'ema_crossover'): [(8, 12, 1.3), (14, 18, 1.2)],
            ('GBPUSD', 'channel_reversal'): [(7, 11, 1.4), (20, 23, 1.1)],
            ('USDJPY', 'rsi_pullback'): [(0, 4, 1.2), (21, 24, 1.3)],
            ('AUDUSD', 'volatility_breakout'): [(22, 2, 1.5), (8, 12, 1.2)],
            # Agregar más basado en análisis temporal real
        }}
        
        key = (asset, strategy)
        if key in optimal_sessions:
            for start_hour, end_hour, multiplier in optimal_sessions[key]:
                if start_hour <= hour < end_hour or (start_hour > end_hour and (hour >= start_hour or hour < end_hour)):
                    return {{
                        'is_optimal': True,
                        'session': f"{{start_hour:02d}}-{{end_hour:02d}}",
                        'confidence_multiplier': multiplier
                    }}
        
        return {{
            'is_optimal': False,
            'session': 'off_hours',
            'confidence_multiplier': 0.7
        }}
    
    def calculate_portfolio_balance_score(self, current_positions: Dict) -> Dict:
        """
        Calcula score de balance del portfolio basado en correlaciones
        
        Returns:
            {{'balance_score': float, 'recommendations': List[str], 'risk_level': str}}
        """
        try:
            if not current_positions:
                return {{
                    'balance_score': 1.0,
                    'recommendations': ['Portfolio empty - can add any positions'],
                    'risk_level': 'NONE'
                }}
            
            # Calcular correlaciones ponderadas
            total_correlation_risk = 0.0
            total_weight = 0.0
            recommendations = []
            
            positions_list = list(current_positions.items())
            
            for i, (pos1, size1) in enumerate(positions_list):
                for pos2, size2 in positions_list[i+1:]:
                    asset1, strategy1 = pos1.split('_', 1)
                    asset2, strategy2 = pos2.split('_', 1)
                    
                    # Correlación entre assets
                    asset_corr = abs(self._get_asset_correlation(asset1, asset2))
                    
                    # Correlación entre estrategias (si mismo asset)
                    strategy_corr = 0.0
                    if asset1 == asset2:
                        strategy_corr = abs(self._get_strategies_correlation(strategy1, strategy2))
                    
                    # Peso combinado
                    combined_weight = size1 * size2
                    correlation_risk = max(asset_corr, strategy_corr) * combined_weight
                    
                    total_correlation_risk += correlation_risk
                    total_weight += combined_weight
                    
                    # Generar recomendaciones
                    if asset_corr > 0.7:
                        recommendations.append(f"High asset correlation between {{asset1}} and {{asset2}}: {{asset_corr:.2f}}")
                    if strategy_corr > 0.8:
                        recommendations.append(f"Redundant strategies {{strategy1}} and {{strategy2}} on {{asset1}}")
            
            # Calcular score de balance (0-1, donde 1 es perfecto balance)
            if total_weight > 0:
                avg_correlation_risk = total_correlation_risk / total_weight
                balance_score = max(0, 1 - avg_correlation_risk)
            else:
                balance_score = 1.0
            
            # Determinar nivel de riesgo
            if balance_score >= 0.8:
                risk_level = 'LOW'
            elif balance_score >= 0.6:
                risk_level = 'MEDIUM'
            elif balance_score >= 0.4:
                risk_level = 'HIGH'
            else:
                risk_level = 'CRITICAL'
            
            # Agregar recomendaciones generales
            if balance_score < 0.6:
                recommendations.append("Consider reducing correlated positions")
            if len(current_positions) < 3:
                recommendations.append("Portfolio may benefit from more diversification")
            
            return {{
                'balance_score': balance_score,
                'recommendations': recommendations,
                'risk_level': risk_level
            }}
            
        except Exception as e:
            return {{
                'balance_score': 0.0,
                'recommendations': [f"Error calculating balance: {{e}}"],
                'risk_level': 'UNKNOWN'
            }}
    
    def get_hedge_recommendations(self, current_positions: Dict) -> List[Dict]:
        """
        Sugiere posiciones de hedge basadas en correlaciones negativas
        """
        hedge_recommendations = []
        
        try:
            for pos_key, pos_size in current_positions.items():
                if pos_size > 0:
                    asset, strategy = pos_key.split('_', 1)
                    
                    # Buscar assets con correlación negativa fuerte
                    for potential_hedge_asset in self.assets:
                        if potential_hedge_asset != asset:
                            correlation = self._get_asset_correlation(asset, potential_hedge_asset)
                            
                            if correlation < -0.6:  # Correlación negativa fuerte
                                # Buscar mejor estrategia para el hedge
                                best_hedge_strategy = self._find_best_hedge_strategy(potential_hedge_asset, strategy)
                                
                                hedge_recommendations.append({{
                                    'original_position': pos_key,
                                    'hedge_asset': potential_hedge_asset,
                                    'hedge_strategy': best_hedge_strategy,
                                    'correlation': correlation,
                                    'recommended_size': pos_size * 0.8,  # 80% hedge ratio
                                    'hedge_effectiveness': abs(correlation),
                                    'rationale': f"Natural hedge due to {{correlation:.2f}} correlation"
                                }})
            
            # Ordenar por efectividad
            hedge_recommendations.sort(key=lambda x: x['hedge_effectiveness'], reverse=True)
            
        except Exception as e:
            hedge_recommendations.append({{
                'error': f"Error generating hedge recommendations: {{e}}"
            }})
        
        return hedge_recommendations[:3]  # Top 3 hedge recommendations
    
    def _find_best_hedge_strategy(self, asset: str, original_strategy: str) -> str:
        """Encuentra la mejor estrategia para hedge en un asset específico"""
        # Buscar estrategia con menor correlación
        min_correlation = float('inf')
        best_strategy = self.strategies[0]  # Default
        
        for strategy in self.strategies:
            correlation = abs(self._get_strategies_correlation(original_strategy, strategy))
            if correlation < min_correlation:
                min_correlation = correlation
                best_strategy = strategy
        
        return best_strategy
    
    def apply_all_rules(self, current_positions: Dict, new_signals: List[Dict], current_time: datetime) -> Dict:
        """
        Aplica todas las reglas de correlación cruzada
        
        Args:
            current_positions: Posiciones actuales
            new_signals: Lista de nuevas señales de trading
            current_time: Timestamp actual
        
        Returns:
            Análisis completo con recomendaciones
        """
        results = {{
            'timestamp': current_time.isoformat(),
            'portfolio_analysis': {{}},
            'signal_analysis': [],
            'hedge_recommendations': [],
            'overall_recommendations': []
        }}
        
        try:
            # 1. Análisis del portfolio actual
            results['portfolio_analysis'] = self.calculate_portfolio_balance_score(current_positions)
            
            # 2. Análisis de cada nueva señal
            for signal in new_signals:
                signal_analysis = self.check_correlation_constraints(current_positions, signal)
                session_analysis = self.get_optimal_trading_session(signal['asset'], signal['strategy'], current_time)
                
                combined_analysis = {{
                    'signal': signal,
                    'correlation_check': signal_analysis,
                    'timing_analysis': session_analysis,
                    'final_recommendation': {{
                        'execute': signal_analysis['allowed'] and session_analysis['is_optimal'],
                        'position_size': signal_analysis['adjusted_size'] * session_analysis['confidence_multiplier'],
                        'confidence': signal.get('confidence', 1.0) * session_analysis['confidence_multiplier']
                    }}
                }}
                
                results['signal_analysis'].append(combined_analysis)
            
            # 3. Recomendaciones de hedge
            results['hedge_recommendations'] = self.get_hedge_recommendations(current_positions)
            
            # 4. Recomendaciones generales
            portfolio_score = results['portfolio_analysis']['balance_score']
            
            if portfolio_score < 0.5:
                results['overall_recommendations'].append("URGENT: Portfolio has high correlation risk - consider rebalancing")
            
            if len(current_positions) > 10:
                results['overall_recommendations'].append("Portfolio may be over-diversified - consider consolidation")
            elif len(current_positions) < 3:
                results['overall_recommendations'].append("Portfolio may benefit from more diversification")
            
            # Contar señales ejecutables
            executable_signals = sum(1 for analysis in results['signal_analysis'] 
                                   if analysis['final_recommendation']['execute'])
            
            if executable_signals == 0 and new_signals:
                results['overall_recommendations'].append("No signals meet correlation and timing criteria - wait for better opportunities")
            
        except Exception as e:
            results['error'] = str(e)
        
        return results

# Funciones de utilidad para integración

def create_correlation_bot_instance():
    """Factory function para crear instancia del bot"""
    return CrossCorrelationBotRules()

def quick_correlation_check(asset1: str, asset2: str) -> float:
    """Verificación rápida de correlación entre dos assets"""
    bot = CrossCorrelationBotRules()
    return bot._get_asset_correlation(asset1, asset2)

def validate_portfolio_balance(positions: Dict) -> str:
    """Validación rápida del balance del portfolio"""
    bot = CrossCorrelationBotRules()
    analysis = bot.calculate_portfolio_balance_score(positions)
    return analysis['risk_level']

# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar bot de reglas
    bot = CrossCorrelationBotRules()
    
    # Ejemplo de posiciones actuales
    current_positions = {{
        'EURUSD_ema_crossover': 0.03,
        'GBPUSD_channel_reversal': 0.025,
        'USDJPY_rsi_pullback': 0.02
    }}
    
    # Ejemplo de nuevas señales
    new_signals = [
        {{
            'asset': 'AUDUSD',
            'strategy': 'volatility_breakout',
            'direction': 'long',
            'confidence': 0.8
        }},
        {{
            'asset': 'EURUSD',
            'strategy': 'volatility_breakout',
            'direction': 'long',
            'confidence': 0.75
        }}
    ]
    
    # Aplicar todas las reglas
    analysis = bot.apply_all_rules(current_positions, new_signals, datetime.now())
    
    # Mostrar resultados
    print("=== ANÁLISIS DE CORRELACIONES CRUZADAS ===")
    print(f"Portfolio Balance Score: {{analysis['portfolio_analysis']['balance_score']:.2f}}")
    print(f"Risk Level: {{analysis['portfolio_analysis']['risk_level']}}")
    
    print("\\nSignal Analysis:")
    for i, signal_analysis in enumerate(analysis['signal_analysis']):
        signal = signal_analysis['signal']
        recommendation = signal_analysis['final_recommendation']
        print(f"  {{i+1}}. {{signal['asset']}}_{{signal['strategy']}}: {{'EXECUTE' if recommendation['execute'] else 'SKIP'}}")
        print(f"     Size: {{recommendation['position_size']:.3f}}, Confidence: {{recommendation['confidence']:.2f}}")
    
    if analysis['hedge_recommendations']:
        print("\\nHedge Recommendations:")
        for hedge in analysis['hedge_recommendations']:
            if 'error' not in hedge:
                print(f"  - Hedge {{hedge['original_position']}} with {{hedge['hedge_asset']}}_{{hedge['hedge_strategy']}}")
                print(f"    Correlation: {{hedge['correlation']:.2f}}, Size: {{hedge['recommended_size']:.3f}}")
    
    print("\\nOverall Recommendations:")
    for rec in analysis['overall_recommendations']:
        print(f"  • {{rec}}")
'''
        
        return code
    
    def _generate_executive_report_text(self, final_report):
        """Genera reporte ejecutivo en texto plano"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""
================================================================================
🔄 ANÁLISIS EXHAUSTIVO DE CORRELACIONES CRUZADAS - REPORTE EJECUTIVO
================================================================================

📅 Fecha de análisis: {timestamp}
🎯 Scope: {len(self.assets)} assets × {len(self.strategies)} estrategias = {len(self.assets) * len(self.strategies)} combinaciones
📊 Assets analizados: {', '.join(self.assets)}
🎪 Estrategias analizadas: {', '.join(self.strategies)}

================================================================================
📊 RESUMEN EJECUTIVO
================================================================================
"""
        
        # Agregar resumen de cobertura de datos
        if 'executive_summary' in final_report and 'data_coverage' in final_report['executive_summary']:
            coverage = final_report['executive_summary']['data_coverage']
            report += f"""
📈 COBERTURA DE DATOS:
• Combinaciones exitosas: {coverage.get('successful_combinations', 0)}/{coverage.get('total_combinations', 0)}
• Porcentaje de cobertura: {coverage.get('coverage_percentage', 0):.1f}%
• Calidad de datos: {coverage.get('data_quality', 'UNKNOWN')}
"""
        
        # Agregar correlaciones clave
        if 'executive_summary' in final_report and 'key_correlations' in final_report['executive_summary']:
            correlations = final_report['executive_summary']['key_correlations']
            if correlations:
                report += """
🔄 CORRELACIONES MÁS SIGNIFICATIVAS:
"""
                for i, corr in enumerate(correlations[:3], 1):
                    pair = corr['pair'].replace('_vs_', ' ↔ ')
                    report += f"""
{i}. {pair}
   • Correlación: {corr['correlation']:.3f} ({corr['strength']})
   • Implicación: {corr['implication']}
"""
        
        # Agregar hallazgos clave
        if 'key_findings' in final_report:
            report += """
================================================================================
🔍 HALLAZGOS CLAVE
================================================================================
"""
            for i, finding in enumerate(final_report['key_findings'][:5], 1):
                report += f"""
{i}. {finding.get('finding_type', 'UNKNOWN').replace('_', ' ')}
   📝 {finding.get('description', 'N/A')}
   💡 Implicación: {finding.get('implication', 'N/A')}
   🎯 Acción: {finding.get('action', 'N/A')}
"""
        
        # Agregar reglas críticas para bots
        if 'algorithmic_rules' in final_report and 'implementation_priority' in final_report['algorithmic_rules']:
            priority_rules = final_report['algorithmic_rules']['implementation_priority']
            critical_rules = [rule for rule in priority_rules if rule.get('priority_level') == 'CRITICAL']
            
            if critical_rules:
                report += f"""
================================================================================
🤖 REGLAS CRÍTICAS PARA BOTS ALGORÍTMICOS
================================================================================

Se identificaron {len(critical_rules)} reglas CRÍTICAS que deben implementarse inmediatamente:
"""
                for i, rule in enumerate(critical_rules[:5], 1):
                    report += f"""
{i}. {rule.get('rule_summary', 'N/A')}
   🔧 Implementación: {rule.get('implementation', 'N/A')}
   📈 Impacto esperado: {rule.get('expected_impact', 'N/A')}
"""
        
        # Agregar oportunidades de optimización
        if 'optimization_opportunities' in final_report:
            opportunities = final_report['optimization_opportunities']
            if opportunities:
                report += """
================================================================================
🚀 OPORTUNIDADES DE OPTIMIZACIÓN IDENTIFICADAS
================================================================================
"""
                for i, opp in enumerate(opportunities[:4], 1):
                    report += f"""
{i}. {opp.get('opportunity_type', 'UNKNOWN').replace('_', ' ')}
   📝 {opp.get('description', 'N/A')}
   💰 Beneficio potencial: {opp.get('potential_benefit', 'N/A')}
   🔧 Complejidad: {opp.get('implementation_complexity', 'UNKNOWN')}
   ⏱️ Timeframe: {opp.get('expected_timeframe', 'Unknown')}
"""
        
        # Agregar roadmap de implementación
        if 'implementation_roadmap' in final_report:
            roadmap = final_report['implementation_roadmap']
            if 'implementation_order' in roadmap:
                report += """
================================================================================
📋 ROADMAP DE IMPLEMENTACIÓN
================================================================================
"""
                for phase in roadmap['implementation_order']:
                    report += f"""
FASE {phase.get('step', 0)}: {phase.get('phase', 'UNKNOWN')}
• Enfoque: {phase.get('focus', 'N/A')}
• Reglas a implementar: {phase.get('rules_count', 0)}
• Criterio de éxito: {phase.get('success_criteria', 'N/A')}
"""
        
        # Agregar insights de riesgo
        if 'risk_insights' in final_report:
            risk_insights = final_report['risk_insights']
            report += f"""
================================================================================
⚠️ ANÁLISIS DE RIESGOS
================================================================================

🔄 RIESGOS DE CORRELACIÓN:
• Identificados {len(risk_insights.get('correlation_risks', []))} pares de assets con correlación alta
• Riesgo de concentración en {len(risk_insights.get('concentration_risks', []))} clusters grandes
"""
            
            # Estrategias de mitigación
            if 'mitigation_strategies' in risk_insights:
                report += """
🛡️ ESTRATEGIAS DE MITIGACIÓN RECOMENDADAS:
"""
                for i, strategy in enumerate(risk_insights['mitigation_strategies'][:4], 1):
                    report += f"""
{i}. {strategy.get('risk_type', 'UNKNOWN').replace('_', ' ')}
   → {strategy.get('strategy', 'N/A')}
   → Implementación: {strategy.get('implementation', 'N/A')}
"""
        
        # Conclusiones y próximos pasos
        report += f"""
================================================================================
🎯 CONCLUSIONES Y PRÓXIMOS PASOS
================================================================================

CONCLUSIONES PRINCIPALES:
1. Identificadas correlaciones significativas que requieren gestión activa
2. Oportunidades claras de optimización mediante reglas algorítmicas
3. Necesidad de implementar controles de riesgo basados en correlaciones
4. Potencial de mejora significativa en risk-adjusted returns

PRÓXIMOS PASOS INMEDIATOS:
1. ✅ IMPLEMENTAR reglas críticas identificadas (1-2 semanas)
2. 🔧 DESARROLLAR sistema de monitoreo de correlaciones en tiempo real
3. 📊 ESTABLECER dashboards de control de riesgo de portfolio
4. 🤖 INTEGRAR reglas en bot de trading algorítmico
5. 📈 MONITOREAR performance y ajustar continuamente

MÉTRICAS DE ÉXITO:
• Reducción > 20% en correlaciones no deseadas
• Mejora > 15% en Sharpe ratio
• Reducción > 25% en maximum drawdown
• Aumento en diversificación efectiva del portfolio

================================================================================
📁 ARCHIVOS GENERADOS
================================================================================

📋 Reporte completo JSON: cross_correlation_analysis_[timestamp].json
🤖 Reglas implementables: algorithmic_bot_rules_[timestamp].py
📊 Matrices de correlación: correlation_matrices_[timestamp].csv
📈 Gráficos de análisis: comprehensive_charts_[timestamp].png
📄 Este reporte ejecutivo: executive_summary_[timestamp].txt

================================================================================
⚠️ DISCLAIMER
================================================================================

Este análisis se basa en datos históricos y correlaciones identificadas.
Las correlaciones pueden cambiar con las condiciones de mercado.
Validar todas las recomendaciones en entorno de demo antes de implementar.
Monitorear continuamente la efectividad de las reglas implementadas.

Último análisis: {timestamp}
Próxima revisión recomendada: En 30 días o después de cambios significativos de mercado
================================================================================
"""
        
        return report
    
    def _export_correlation_matrices(self, timestamp):
        """Exporta matrices de correlación en formato CSV"""
        try:
            correlation_files = []
            
            # Matriz global de correlaciones entre assets
            if 'global_correlation' in self.cross_asset_patterns:
                global_corr = self.cross_asset_patterns['global_correlation']
                
                if 'pearson' in global_corr:
                    df_global = pd.DataFrame(global_corr['pearson'])
                    global_path = self.export_dir / f"asset_correlations_global_{timestamp}.csv"
                    df_global.to_csv(global_path)
                    correlation_files.append(str(global_path))
                    print(f"📊 Correlaciones globales: {global_path}")
            
            # Matrices por estrategia
            for strategy, strategy_corr in self.correlation_matrices.items():
                if 'pearson' in strategy_corr:
                    df_strategy = pd.DataFrame(strategy_corr['pearson'])
                    strategy_path = self.export_dir / f"asset_correlations_{strategy}_{timestamp}.csv"
                    df_strategy.to_csv(strategy_path)
                    correlation_files.append(str(strategy_path))
                    print(f"📊 Correlaciones {strategy}: {strategy_path}")
            
            # Matriz de correlaciones entre estrategias
            strategy_correlation_summary = []
            for asset, strategy_data in self.cross_strategy_patterns.items():
                if 'strategy_pairs' in strategy_data:
                    for pair, data in strategy_data['strategy_pairs'].items():
                        if 'correlation' in data:
                            strategy_correlation_summary.append({
                                'asset': asset,
                                'strategy_pair': pair,
                                'correlation': data['correlation'],
                                'p_value': data.get('p_value', 0),
                                'significant': data.get('significant', False)
                            })
            
            if strategy_correlation_summary:
                df_strategies = pd.DataFrame(strategy_correlation_summary)
                strategies_path = self.export_dir / f"strategy_correlations_{timestamp}.csv"
                df_strategies.to_csv(strategies_path, index=False)
                correlation_files.append(str(strategies_path))
                print(f"📊 Correlaciones entre estrategias: {strategies_path}")
            
            return correlation_files
            
        except Exception as e:
            print(f"❌ Error exportando matrices de correlación: {e}")
            return []
    
    def _create_comprehensive_charts(self, final_report, timestamp):
        """Crea gráficos comprehensivos del análisis"""
        chart_paths = []
        
        try:
            # Configurar estilo
            plt.style.use('default')
            
            # Gráfico 1: Heatmap de correlaciones entre assets
            if self._create_correlation_heatmap(timestamp):
                chart_paths.append(f"correlation_heatmap_{timestamp}.png")
            
            # Gráfico 2: Análisis de clustering
            if self._create_clustering_visualization(timestamp):
                chart_paths.append(f"clustering_analysis_{timestamp}.png")
            
            # Gráfico 3: Performance por asset-strategy combinations
            if self._create_performance_matrix(timestamp):
                chart_paths.append(f"performance_matrix_{timestamp}.png")
            
            # Gráfico 4: Análisis temporal
            if self._create_temporal_analysis_chart(timestamp):
                chart_paths.append(f"temporal_analysis_{timestamp}.png")
            
            print(f"📈 Gráficos creados: {len(chart_paths)} archivos")
            
        except Exception as e:
            print(f"❌ Error creando gráficos: {e}")
        
        return chart_paths
    
    def _create_correlation_heatmap(self, timestamp):
        """Crea heatmap de correlaciones"""
        try:
            if 'global_correlation' in self.cross_asset_patterns and 'pearson' in self.cross_asset_patterns['global_correlation']:
                correlation_data = self.cross_asset_patterns['global_correlation']['pearson']
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Convertir a DataFrame para heatmap
                df_corr = pd.DataFrame(correlation_data)
                
                # Crear heatmap
                im = ax.imshow(df_corr.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                
                # Configurar ejes
                ax.set_xticks(range(len(df_corr.columns)))
                ax.set_yticks(range(len(df_corr.index)))
                ax.set_xticklabels(df_corr.columns, rotation=45)
                ax.set_yticklabels(df_corr.index)
                
                # Agregar valores en las celdas
                for i in range(len(df_corr.index)):
                    for j in range(len(df_corr.columns)):
                        value = df_corr.iloc[i, j]
                        color = 'white' if abs(value) > 0.5 else 'black'
                        ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=color)
                
                # Colorbar
                cbar = plt.colorbar(im)
                cbar.set_label('Correlación', rotation=270, labelpad=20)
                
                plt.title('Matriz de Correlaciones entre Assets\n(Basada en Performance Agregada)', fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                chart_path = self.export_dir / f"correlation_heatmap_{timestamp}.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                return True
        except Exception as e:
            print(f"Error creando heatmap: {e}")
            return False
    
    def _create_clustering_visualization(self, timestamp):
        """Crea visualización de clustering"""
        try:
            if 'clustering' in self.cross_asset_patterns and 'pca_analysis' in self.cross_asset_patterns['clustering']:
                pca_data = self.cross_asset_patterns['clustering']['pca_analysis']
                cluster_assignments = self.cross_asset_patterns['clustering']['cluster_assignments']
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Preparar datos para scatter plot
                x_coords = []
                y_coords = []
                colors = []
                labels = []
                
                color_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange', 4: 'purple', 5: 'brown'}
                
                for label, coords in pca_data['pca_coordinates'].items():
                    x_coords.append(coords[0])
                    y_coords.append(coords[1])
                    cluster_id = cluster_assignments.get(label, 0)
                    colors.append(color_map.get(cluster_id, 'gray'))
                    labels.append(label)
                
                # Scatter plot
                scatter = ax.scatter(x_coords, y_coords, c=colors, s=100, alpha=0.7)
                
                # Agregar labels
                for i, label in enumerate(labels):
                    ax.annotate(label, (x_coords[i], y_coords[i]), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8)
                
                # Configurar gráfico
                ax.set_xlabel(f'PC1 ({pca_data["explained_variance_ratio"][0]:.1%} varianza)')
                ax.set_ylabel(f'PC2 ({pca_data["explained_variance_ratio"][1]:.1%} varianza)')
                ax.set_title('Clustering de Combinaciones Asset-Strategy\n(Análisis de Componentes Principales)', 
                           fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Leyenda de clusters
                cluster_info = self.cross_asset_patterns['clustering']['cluster_characteristics']
                legend_elements = []
                for cluster_id, color in color_map.items():
                    if f'cluster_{cluster_id}' in cluster_info:
                        cluster_data = cluster_info[f'cluster_{cluster_id}']
                        performance_tier = cluster_data.get('performance_tier', 'UNKNOWN')
                        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                        markerfacecolor=color, markersize=10,
                                                        label=f'Cluster {cluster_id} ({performance_tier})'))
                
                if legend_elements:
                    ax.legend(handles=legend_elements, loc='upper right')
                
                plt.tight_layout()
                
                chart_path = self.export_dir / f"clustering_analysis_{timestamp}.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                return True
        except Exception as e:
            print(f"Error creando clustering visualization: {e}")
            return False
    
    def _create_performance_matrix(self, timestamp):
        """Crea matriz de performance por asset-strategy"""
        try:
            # Preparar datos de performance
            performance_matrix = []
            asset_labels = []
            strategy_labels = []
            
            for asset in self.assets:
                if asset in self.all_data:
                    asset_performance = []
                    
                    for strategy in self.strategies:
                        if (strategy in self.all_data[asset] and 
                            self.all_data[asset][strategy] is not None):
                            
                            trade_log = self.all_data[asset][strategy]['trade_log']
                            
                            if not trade_log.empty:
                                # Encontrar columna PnL
                                pnl_col = None
                                for col in ['pnl', 'profit', 'return']:
                                    if col in trade_log.columns:
                                        pnl_col = col
                                        break
                                
                                if pnl_col:
                                    total_return = trade_log[pnl_col].sum()
                                    win_rate = (trade_log[pnl_col] > 0).mean()
                                    # Score combinado
                                    performance_score = total_return * win_rate
                                    asset_performance.append(performance_score)
                                else:
                                    asset_performance.append(0)
                            else:
                                asset_performance.append(0)
                        else:
                            asset_performance.append(0)
                    
                    if any(p != 0 for p in asset_performance):
                        performance_matrix.append(asset_performance)
                        asset_labels.append(asset)
            
            if not performance_matrix:
                return False
            
            strategy_labels = self.strategies[:len(performance_matrix[0])]
            
            # Crear heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            performance_array = np.array(performance_matrix)
            
            # Normalizar para mejor visualización
            vmax = np.percentile(performance_array[performance_array > 0], 95) if np.any(performance_array > 0) else 1
            vmin = np.percentile(performance_array[performance_array < 0], 5) if np.any(performance_array < 0) else -1
            
            im = ax.imshow(performance_array, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)
            
            # Configurar ejes
            ax.set_xticks(range(len(strategy_labels)))
            ax.set_yticks(range(len(asset_labels)))
            ax.set_xticklabels(strategy_labels, rotation=45, ha='right')
            ax.set_yticklabels(asset_labels)
            
            # Agregar valores en las celdas
            for i in range(len(asset_labels)):
                for j in range(len(strategy_labels)):
                    value = performance_array[i, j]
                    color = 'white' if abs(value) > abs(vmax) * 0.5 else 'black'
                    ax.text(j, i, f'{value:.3f}', ha='center', va='center', 
                           color=color, fontsize=8)
            
            # Colorbar
            cbar = plt.colorbar(im)
            cbar.set_label('Performance Score (Return × Win Rate)', rotation=270, labelpad=20)
            
            plt.title('Matriz de Performance: Asset × Strategy\n(Score = Total Return × Win Rate)', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            chart_path = self.export_dir / f"performance_matrix_{timestamp}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"Error creando performance matrix: {e}")
            return False
    
    def _create_temporal_analysis_chart(self, timestamp):
        """Crea gráfico de análisis temporal"""
        try:
            if 'temporal' in self.cross_asset_patterns and 'sessions' in self.cross_asset_patterns['temporal']:
                sessions_data = self.cross_asset_patterns['temporal']['sessions']
                
                # Preparar datos para gráfico de barras por sesiones
                session_performance = {'Asian': [], 'European': [], 'American': []}
                combination_labels = []
                
                for asset, asset_sessions in sessions_data.items():
                    for strategy, strategy_sessions in asset_sessions.items():
                        combination = f"{asset}_{strategy}"
                        combination_labels.append(combination)
                        
                        for session in ['Asian', 'European', 'American']:
                            if session in strategy_sessions:
                                avg_pnl = strategy_sessions[session].get('mean', 0)
                                session_performance[session].append(avg_pnl)
                            else:
                                session_performance[session].append(0)
                
                if not combination_labels:
                    return False
                
                # Crear gráfico de barras agrupadas
                fig, ax = plt.subplots(figsize=(15, 8))
                
                x = np.arange(len(combination_labels))
                width = 0.25
                
                bars1 = ax.bar(x - width, session_performance['Asian'], width, 
                              label='Asian Session', color='orange', alpha=0.8)
                bars2 = ax.bar(x, session_performance['European'], width, 
                              label='European Session', color='blue', alpha=0.8)
                bars3 = ax.bar(x + width, session_performance['American'], width, 
                              label='American Session', color='green', alpha=0.8)
                
                # Configurar gráfico
                ax.set_xlabel('Asset-Strategy Combinations')
                ax.set_ylabel('Average PnL per Trade')
                ax.set_title('Performance por Sesiones de Trading\n(PnL Promedio por Trade)', 
                           fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(combination_labels, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                # Línea de referencia en 0
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                
                chart_path = self.export_dir / f"temporal_analysis_{timestamp}.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                return True
                
        except Exception as e:
            print(f"Error creando temporal analysis: {e}")
            return False
    
    def _display_executive_summary(self, final_report):
        """Muestra resumen ejecutivo en consola"""
        print("\n" + "=" * 80)
        print("🔄 RESUMEN EJECUTIVO - ANÁLISIS DE CORRELACIONES CRUZADAS")
        print("=" * 80)
        
        # Metadata
        metadata = final_report.get('metadata', {})
        print(f"📊 Assets analizados: {len(metadata.get('assets_analyzed', []))}")
        print(f"🎯 Estrategias analizadas: {len(metadata.get('strategies_analyzed', []))}")
        print(f"🔄 Total combinaciones: {metadata.get('total_combinations', 0)}")
        
        # Cobertura de datos
        if 'executive_summary' in final_report and 'data_coverage' in final_report['executive_summary']:
            coverage = final_report['executive_summary']['data_coverage']
            print(f"📈 Cobertura de datos: {coverage.get('coverage_percentage', 0):.1f}% ({coverage.get('data_quality', 'UNKNOWN')})")
        
        # Correlaciones clave
        if 'executive_summary' in final_report and 'key_correlations' in final_report['executive_summary']:
            correlations = final_report['executive_summary']['key_correlations']
            if correlations:
                print(f"\n🔄 CORRELACIONES MÁS FUERTES:")
                for i, corr in enumerate(correlations[:3], 1):
                    pair = corr['pair'].replace('_vs_', ' ↔ ')
                    print(f"   {i}. {pair}: {corr['correlation']:+.3f} ({corr['strength']})")
        
        # Performance insights
        if 'executive_summary' in final_report and 'performance_insights' in final_report['executive_summary']:
            insights = final_report['executive_summary']['performance_insights']
            if insights:
                print(f"\n🏆 PERFORMANCE INSIGHTS:")
                print(f"   • Combinaciones elite: {insights.get('elite_count', 0)}")
                print(f"   • Combinaciones débiles: {insights.get('weak_count', 0)}")
        
        # Reglas críticas
        if 'algorithmic_rules' in final_report and 'implementation_priority' in final_report['algorithmic_rules']:
            priority_rules = final_report['algorithmic_rules']['implementation_priority']
            critical_rules = [rule for rule in priority_rules if rule.get('priority_level') == 'CRITICAL']
            
            if critical_rules:
                print(f"\n🚨 REGLAS CRÍTICAS IDENTIFICADAS: {len(critical_rules)}")
                for i, rule in enumerate(critical_rules[:3], 1):
                    print(f"   {i}. {rule.get('rule_summary', 'N/A')[:60]}...")
        
        # Oportunidades principales
        if 'optimization_opportunities' in final_report:
            opportunities = final_report['optimization_opportunities']
            if opportunities:
                print(f"\n🚀 OPORTUNIDADES PRINCIPALES:")
                for i, opp in enumerate(opportunities[:3], 1):
                    opp_type = opp.get('opportunity_type', 'UNKNOWN').replace('_', ' ')
                    print(f"   {i}. {opp_type}: {opp.get('potential_benefit', 'N/A')}")
        
        # Próximos pasos
        print(f"\n📋 PRÓXIMOS PASOS:")
        print(f"   1. Revisar reporte completo JSON")
        print(f"   2. Implementar reglas críticas en bot")
        print(f"   3. Establecer monitoreo de correlaciones")
        print(f"   4. Validar en demo trading")
        print(f"   5. Ejecutar roadmap de implementación")
        
        print(f"\n📁 Archivos exportados en: {self.export_dir}")
        print("=" * 80)


def main():
    """Función principal"""
    print("🔄 CROSS-CORRELATION ANALYZER")
    print("=" * 80)
    print("🎯 CARACTERÍSTICAS:")
    print("   ✅ Análisis exhaustivo de correlaciones cross-asset")
    print("   ✅ Correlaciones cross-strategy por asset")
    print("   ✅ Patrones temporales cruzados")
    print("   ✅ Clustering de combinaciones asset-strategy")
    print("   ✅ Generación de reglas algorítmicas para bots")
    print("   ✅ Roadmap de implementación prioritizado")
    print("   ✅ Análisis de riesgos y oportunidades")
    print("   ✅ Exportación en múltiples formatos")
    print(f"\n🎯 Configurado para: {ASSETS_TO_ANALYZE} × {STRATEGIES_TO_ANALYZE}")
    
    if not IMPORTS_AVAILABLE:
        print("❌ Importaciones no disponibles")
        return
    
    try:
        analyzer = CrossCorrelationAnalyzer()
        result = analyzer.run_comprehensive_analysis()
        
        if result:
            print("\n" + "=" * 80)
            print("✅ ANÁLISIS COMPREHENSIVO COMPLETADO")
            print("🎯 DELIVERABLES GENERADOS:")
            print("   📋 Reporte exhaustivo JSON con todos los datos")
            print("   🤖 Código Python implementable para bot algorítmico")
            print("   📄 Reporte ejecutivo en texto plano")
            print("   📊 Matrices de correlación en CSV")
            print("   📈 Gráficos comprehensivos de análisis")
            print("   📋 Roadmap de implementación prioritizado")
            print("\n🚀 PRÓXIMOS PASOS:")
            print("   1. Revisar reporte ejecutivo para insights clave")
            print("   2. Implementar reglas críticas en bot de trading")
            print("   3. Establecer monitoreo de correlaciones en tiempo real")
            print("   4. Validar todas las reglas en demo trading")
            print("   5. Ejecutar roadmap de implementación por fases")
            print("   6. Re-analizar mensualmente para adaptación continua")
        else:
            print("\n❌ No se pudo completar el análisis comprehensivo")
            
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🎯 Presiona Enter para continuar...")
        input()


if __name__ == "__main__":
    main()