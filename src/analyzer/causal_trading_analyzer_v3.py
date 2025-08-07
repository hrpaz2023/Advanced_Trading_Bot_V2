# -*- coding: utf-8 -*-
"""
causal_trading_analyzer_integrated.py - ANALIZADOR CAUSAL INTEGRADO
-------------------------------------------------------------------
- Versión optimizada para trabajar con datos del unified_trade_analyzer.py
- Manejo mejorado de datos normalizados
- Análisis causal robusto con mejor interpretación
- Exporta resultados listos para consolidate_global_insights.py

Requiere: econml, scikit-learn, pandas, numpy, pyyaml
"""
import os, json, yaml, warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# EconML
ECONML_AVAILABLE = True
try:
    from econml.dml import LinearDML, CausalForestDML
    from econml.metalearners import SLearner, TLearner
    from econml.policy import PolicyTree
except Exception:
    ECONML_AVAILABLE = False

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

def _safe_numeric(a: np.ndarray) -> np.ndarray:
    """Convierte a numérico de forma segura"""
    a = np.asarray(a, dtype=float)
    a[~np.isfinite(a)] = np.nan
    return a

def _clean_XYT(X, Y, T):
    """Limpia datos X, Y, T removiendo valores no finitos"""
    X = _safe_numeric(X)
    Y = _safe_numeric(Y).reshape(-1)
    T = _safe_numeric(T).reshape(-1)
    
    # Máscara para filas finitas
    finite_mask = np.isfinite(Y) & np.isfinite(T)
    if X.ndim == 1: 
        X = X.reshape(-1,1)
    finite_mask &= np.all(np.isfinite(X), axis=1)
    
    Xc, Yc, Tc = X[finite_mask], Y[finite_mask], T[finite_mask]
    return Xc, Yc, Tc, finite_mask

def _ensure_datetime_column(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Asegura que la columna de tiempo sea datetime"""
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    return df

@dataclass
class IntegratedCausalAnalyzer:
    input_dir: Path = Path("reports/advanced_reports")  # Tu directorio por defecto
    output_dir: Path = Path("reports")
    skip_forest: bool = False
    min_trades_per_analysis: int = 20

    def __post_init__(self):
        (self.output_dir / "causal_analysis_reports_v3").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "causal_reports").mkdir(parents=True, exist_ok=True)

    def load_unified_trade_data(self) -> pd.DataFrame:
        """Carga datos unificados del generador"""
        print(f"📂 Cargando datos desde: {self.input_dir}")
        
        trade_files = list(self.input_dir.glob("*_trades.csv"))
        if not trade_files:
            print(f"❌ No se encontraron archivos *_trades.csv en {self.input_dir}")
            return pd.DataFrame()
        
        all_trades = []
        loaded_files = 0
        
        for file_path in trade_files:
            try:
                print(f"📖 Cargando: {file_path.name}")
                df = pd.read_csv(file_path)
                
                # Validar columnas esenciales
                required_cols = ['trade_pnl', 'asset', 'strategy']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    print(f"⚠️ {file_path.name}: Faltan columnas {missing_cols}, saltando...")
                    continue
                
                # Limpiar datos
                df = df.dropna(subset=['trade_pnl'])
                df['trade_pnl'] = pd.to_numeric(df['trade_pnl'], errors='coerce')
                df = df.dropna(subset=['trade_pnl'])
                
                if len(df) < self.min_trades_per_analysis:
                    print(f"⚠️ {file_path.name}: Solo {len(df)} trades, mínimo {self.min_trades_per_analysis}")
                    continue
                
                # Asegurar columnas de tiempo
                df = _ensure_datetime_column(df, 'trade_entry_time')
                df = _ensure_datetime_column(df, 'trade_exit_time')
                
                all_trades.append(df)
                loaded_files += 1
                print(f"✅ {file_path.name}: {len(df)} trades cargados")
                
            except Exception as e:
                print(f"❌ Error cargando {file_path.name}: {e}")
                continue
        
        if not all_trades:
            print("❌ No se pudieron cargar datos válidos")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_trades, ignore_index=True)
        print(f"📊 Total combinado: {len(combined_df)} trades de {loaded_files} archivos")
        
        return combined_df

    def build_causal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construye features para análisis causal"""
        print("🔧 Construyendo features causales...")
        d = df.copy()
        
        # === TREATMENTS (Variables de intervención) ===
        
        # Treatment 1: Tipo de estrategia
        d["treatment_strategy_type"] = d["strategy"].astype(str).fillna("UNKNOWN")
        
        # Treatment 2: Régimen de volatilidad
        vol_col = self._find_volatility_column(d)
        if vol_col:
            vol_values = pd.to_numeric(d[vol_col], errors='coerce').fillna(d[vol_col].median())
            q33, q67 = vol_values.quantile([0.33, 0.67])
            d["treatment_volatility_regime"] = pd.cut(
                vol_values, 
                bins=[-np.inf, q33, q67, np.inf], 
                labels=["low_vol", "medium_vol", "high_vol"]
            ).astype(str).fillna("medium_vol")
        else:
            d["treatment_volatility_regime"] = "medium_vol"
        
        # Treatment 3: Timing de sesión
        d["treatment_session_timing"] = self._extract_session_timing(d)
        
        # Treatment 4: Dirección del trade
        if 'trade_side' in d.columns:
            d["treatment_trade_direction"] = d['trade_side'].astype(str).fillna("UNKNOWN")
        else:
            # Inferir dirección desde PnL y precios
            d["treatment_trade_direction"] = self._infer_trade_direction(d)
        
        # Treatment 5: Estrés de mercado
        d["treatment_market_stress"] = self._calculate_market_stress(d)
        
        # === OUTCOMES (Variables de resultado) ===
        
        # Outcome 1: PnL del trade
        d["outcome_trade_pnl"] = pd.to_numeric(d["trade_pnl"], errors='coerce').fillna(0)
        
        # Outcome 2: Probabilidad de ganancia (binario)
        d["outcome_win_probability"] = (d["outcome_trade_pnl"] > 0).astype(int)
        
        # Outcome 3: Magnitud de ganancia (solo trades ganadores)
        winning_mask = d["outcome_trade_pnl"] > 0
        d["outcome_win_magnitude"] = 0.0
        if winning_mask.any():
            d.loc[winning_mask, "outcome_win_magnitude"] = d.loc[winning_mask, "outcome_trade_pnl"]
        
        # === CONTROLS (Variables de control) ===
        
        # Control 1: Volatilidad normalizada
        if vol_col:
            vol_values = pd.to_numeric(d[vol_col], errors='coerce').fillna(d[vol_col].median())
            d["control_volatility_normalized"] = (vol_values - vol_values.mean()) / (vol_values.std() + 1e-8)
        else:
            d["control_volatility_normalized"] = 0
        
        # Control 2: Precio de entrada normalizado
        if 'trade_entry_price' in d.columns:
            entry_prices = pd.to_numeric(d['trade_entry_price'], errors='coerce')
            if entry_prices.notna().any():
                d["control_entry_price_normalized"] = (entry_prices - entry_prices.mean()) / (entry_prices.std() + 1e-8)
            else:
                d["control_entry_price_normalized"] = 0
        else:
            d["control_entry_price_normalized"] = 0
        
        # Control 3: Tiempo (hora del día)
        d["control_hour_of_day"] = self._extract_hour_from_time(d)
        
        # Control 4: Día de la semana
        d["control_day_of_week"] = self._extract_day_of_week(d)
        
        # Control 5: Asset encoding
        le_asset = LabelEncoder()
        d["control_asset_encoded"] = le_asset.fit_transform(d["asset"].astype(str))
        
        # Control 6: Indicadores técnicos (si están disponibles)
        technical_controls = self._extract_technical_controls(d)
        d.update(technical_controls)
        
        print(f"✅ Features construidos: {len(d)} observaciones")
        return d

    def _find_volatility_column(self, df):
        """Encuentra columna de volatilidad disponible"""
        vol_candidates = [
            'market_atr', 'market_atr_14', 'market_atr_20', 'market_true_range',
            'atr', 'atr_14', 'volatility', 'market_volatility'
        ]
        for col in vol_candidates:
            if col in df.columns and df[col].notna().any():
                return col
        return None

    def _extract_session_timing(self, df):
        """Extrae timing de sesión de trading"""
        # Método 1: Usar flags de sesión si están disponibles
        session_flags = ['market_session_london', 'market_session_ny', 'market_session_overlap']
        if all(col in df.columns for col in session_flags):
            def map_session_from_flags(row):
                if pd.to_numeric(row.get('market_session_overlap', 0), errors='coerce') > 0.5:
                    return "OVERLAP_LONDON_NY"
                elif pd.to_numeric(row.get('market_session_london', 0), errors='coerce') > 0.5:
                    return "LONDON"
                elif pd.to_numeric(row.get('market_session_ny', 0), errors='coerce') > 0.5:
                    return "NEW_YORK"
                else:
                    return "ASIAN"
            return df.apply(map_session_from_flags, axis=1)
        
        # Método 2: Usar hora si está disponible
        hour_col = self._find_hour_column(df)
        if hour_col:
            hours = pd.to_numeric(df[hour_col], errors='coerce').fillna(12)
            return hours.apply(self._hour_to_session)
        
        # Método 3: Extraer de timestamp
        if 'trade_entry_time' in df.columns:
            times = pd.to_datetime(df['trade_entry_time'], errors='coerce')
            hours = times.dt.hour.fillna(12)
            return hours.apply(self._hour_to_session)
        
        # Fallback
        return "LONDON"

    def _find_hour_column(self, df):
        """Encuentra columna de hora"""
        hour_candidates = ['market_hour', 'hour', 'entry_hour']
        for col in hour_candidates:
            if col in df.columns and df[col].notna().any():
                return col
        return None

    def _hour_to_session(self, hour):
        """Convierte hora UTC a sesión de trading"""
        try:
            h = int(hour) % 24
            if 0 <= h < 8: return "ASIAN"
            elif 8 <= h < 13: return "LONDON"
            elif 13 <= h < 16: return "OVERLAP_LONDON_NY"
            elif 16 <= h < 21: return "NEW_YORK"
            else: return "INACTIVE"
        except:
            return "LONDON"

    def _infer_trade_direction(self, df):
        """Infiere dirección del trade desde precios"""
        if 'trade_entry_price' in df.columns and 'trade_exit_price' in df.columns:
            entry = pd.to_numeric(df['trade_entry_price'], errors='coerce')
            exit_price = pd.to_numeric(df['trade_exit_price'], errors='coerce')
            pnl = df['outcome_trade_pnl']
            
            # Si exit > entry y pnl > 0 => LONG
            # Si exit < entry y pnl > 0 => SHORT
            conditions = [
                ((exit_price > entry) & (pnl > 0)) | ((exit_price < entry) & (pnl < 0)),
                ((exit_price < entry) & (pnl > 0)) | ((exit_price > entry) & (pnl < 0))
            ]
            choices = ['LONG', 'SHORT']
            return np.select(conditions, choices, default='UNKNOWN')
        else:
            return "UNKNOWN"

    def _calculate_market_stress(self, df):
        """Calcula indicador de estrés de mercado"""
        stress_score = 0
        
        # Factor 1: Volatilidad extrema
        vol_col = self._find_volatility_column(df)
        if vol_col:
            vol_values = pd.to_numeric(df[vol_col], errors='coerce').fillna(0)
            high_vol_threshold = vol_values.quantile(0.8)
            stress_score += (vol_values > high_vol_threshold).astype(int)
        
        # Factor 2: RSI extremo
        rsi_cols = [col for col in df.columns if 'rsi' in col.lower()]
        if rsi_cols:
            rsi_col = rsi_cols[0]
            rsi_values = pd.to_numeric(df[rsi_col], errors='coerce').fillna(50)
            extreme_rsi = ((rsi_values > 80) | (rsi_values < 20)).astype(int)
            stress_score += extreme_rsi
        
        # Factor 3: Spreads amplios
        if 'market_Spread' in df.columns:
            spreads = pd.to_numeric(df['market_Spread'], errors='coerce').fillna(0)
            high_spread_threshold = spreads.quantile(0.8)
            stress_score += (spreads > high_spread_threshold).astype(int)
        
        return (stress_score >= 2).astype(int)  # Stress si 2+ factores

    def _extract_hour_from_time(self, df):
        """Extrae hora del día"""
        hour_col = self._find_hour_column(df)
        if hour_col:
            return pd.to_numeric(df[hour_col], errors='coerce').fillna(12)
        
        if 'trade_entry_time' in df.columns:
            times = pd.to_datetime(df['trade_entry_time'], errors='coerce')
            return times.dt.hour.fillna(12)
        
        return 12  # Fallback

    def _extract_day_of_week(self, df):
        """Extrae día de la semana"""
        if 'trade_entry_time' in df.columns:
            times = pd.to_datetime(df['trade_entry_time'], errors='coerce')
            return times.dt.dayofweek.fillna(2)  # Default: martes
        
        return 2  # Fallback

    def _extract_technical_controls(self, df):
        """Extrae controles de indicadores técnicos"""
        controls = {}
        
        # RSI
        rsi_cols = [col for col in df.columns if 'rsi' in col.lower()]
        if rsi_cols:
            rsi_values = pd.to_numeric(df[rsi_cols[0]], errors='coerce').fillna(50)
            controls["control_rsi_level"] = rsi_values
        else:
            controls["control_rsi_level"] = 50
        
        # MACD
        macd_cols = [col for col in df.columns if 'macd' in col.lower() and 'signal' not in col.lower()]
        if macd_cols:
            macd_values = pd.to_numeric(df[macd_cols[0]], errors='coerce').fillna(0)
            controls["control_macd_level"] = macd_values
        else:
            controls["control_macd_level"] = 0
        
        # Bollinger Bands position
        bb_cols = [col for col in df.columns if 'bb_' in col.lower()]
        if len(bb_cols) >= 3:
            try:
                bb_upper = pd.to_numeric(df[[col for col in bb_cols if 'upper' in col][0]], errors='coerce')
                bb_lower = pd.to_numeric(df[[col for col in bb_cols if 'lower' in col][0]], errors='coerce')
                bb_middle = pd.to_numeric(df[[col for col in bb_cols if 'middle' in col][0]], errors='coerce')
                
                # Position within bands (0 = lower, 0.5 = middle, 1 = upper)
                if 'market_Close' in df.columns:
                    close_price = pd.to_numeric(df['market_Close'], errors='coerce')
                    bb_position = (close_price - bb_lower) / (bb_upper - bb_lower + 1e-8)
                    controls["control_bb_position"] = bb_position.fillna(0.5)
                else:
                    controls["control_bb_position"] = 0.5
            except:
                controls["control_bb_position"] = 0.5
        else:
            controls["control_bb_position"] = 0.5
        
        return controls

    def estimate_causal_effects(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Estima efectos causales usando EconML"""
        print("🧮 Estimando efectos causales...")
        
        if not ECONML_AVAILABLE:
            return {"error": "econml_not_available"}
        
        results = {}
        
        # Preparar variables de control
        control_cols = [col for col in df.columns if col.startswith("control_")]
        if not control_cols:
            X = np.ones((len(df), 1))  # Intercepto solo
        else:
            X = df[control_cols].values
            # Limpiar X
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # === ANÁLISIS 1: Estrategia → PnL ===
        try:
            Y = df["outcome_trade_pnl"].values
            T_raw = df["treatment_strategy_type"].astype(str).values
            
            le = LabelEncoder()
            T = le.fit_transform(T_raw)
            
            Xc, Yc, Tc, _ = _clean_XYT(X, Y, T)
            
            if len(Xc) >= 30 and len(np.unique(Tc)) >= 2:
                dml = LinearDML(
                    model_y=GradientBoostingRegressor(n_estimators=50, random_state=42),
                    model_t=GradientBoostingRegressor(n_estimators=50, random_state=42),
                    random_state=42
                )
                dml.fit(Yc, Tc, X=Xc)
                effects = dml.effect(Xc)
                
                strategy_effects = {}
                for i, strategy_name in enumerate(le.classes_):
                    mask = (Tc == i)
                    if mask.sum() > 0:
                        strategy_effects[strategy_name] = {
                            "average_treatment_effect": float(np.nanmean(effects[mask])),
                            "observations": int(mask.sum()),
                            "confidence_interval": [
                                float(np.nanpercentile(effects[mask], 5)),
                                float(np.nanpercentile(effects[mask], 95))
                            ]
                        }
                
                results["strategy_effects"] = {
                    "method": "LinearDML",
                    "overall_ate": float(np.nanmean(effects)),
                    "effect_heterogeneity": float(np.nanstd(effects)),
                    "strategy_effects": strategy_effects
                }
                print("✅ Efectos de estrategia estimados")
            else:
                results["strategy_effects"] = {"error": "insufficient_data"}
                
        except Exception as e:
            results["strategy_effects"] = {"error": str(e)}
        
        # === ANÁLISIS 2: Volatilidad → Win Probability ===
        try:
            Y = df["outcome_win_probability"].astype(float).values
            T_raw = df["treatment_volatility_regime"].astype(str).values
            
            le = LabelEncoder()
            T = le.fit_transform(T_raw)
            
            Xc, Yc, Tc, _ = _clean_XYT(X, Y, T)
            
            if len(Xc) >= 30 and len(np.unique(Tc)) >= 2:
                slearner = SLearner(
                    overall_model=RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
                )
                slearner.fit(Yc, Tc, X=Xc)
                
                # Calcular efectos por régimen
                vol_effects = {}
                for i, regime_name in enumerate(le.classes_):
                    mask = (Tc == i)
                    if mask.sum() > 0:
                        vol_effects[regime_name] = {
                            "average_win_rate": float(np.nanmean(Yc[mask])),
                            "observations": int(mask.sum()),
                            "relative_effect": float(np.nanmean(Yc[mask]) - np.nanmean(Yc))
                        }
                
                results["volatility_effects"] = {
                    "method": "SLearner",
                    "volatility_effects": vol_effects
                }
                print("✅ Efectos de volatilidad estimados")
            else:
                results["volatility_effects"] = {"error": "insufficient_data"}
                
        except Exception as e:
            results["volatility_effects"] = {"error": str(e)}
        
        # === ANÁLISIS 3: Session Timing → PnL ===
        try:
            Y = df["outcome_trade_pnl"].values
            # Convertir session timing a binario (LONDON vs otros)
            T_bin = (df["treatment_session_timing"].astype(str) == "LONDON").astype(int).values
            
            Xc, Yc, Tc, _ = _clean_XYT(X, Y, T_bin)
            
            if len(Xc) >= 30 and len(np.unique(Tc)) >= 2:
                tlearner = TLearner(models=[
                    RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1),
                    RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
                ])
                tlearner.fit(Yc, Tc, X=Xc)
                effects = tlearner.effect(Xc)
                
                # Análisis por sesión
                session_effects = {}
                for session in df["treatment_session_timing"].unique():
                    session_mask = (df["treatment_session_timing"] == session)
                    if session_mask.sum() > 0:
                        session_pnl = df.loc[session_mask, "outcome_trade_pnl"]
                        session_effects[session] = {
                            "average_pnl": float(session_pnl.mean()),
                            "observations": int(session_mask.sum()),
                            "win_rate": float((session_pnl > 0).mean())
                        }
                
                results["timing_effects"] = {
                    "method": "TLearner",
                    "london_vs_others_effect": float(np.nanmean(effects)),
                    "session_effects": session_effects
                }
                print("✅ Efectos de timing estimados")
            else:
                results["timing_effects"] = {"error": "insufficient_data"}
                
        except Exception as e:
            results["timing_effects"] = {"error": str(e)}
        
        # === ANÁLISIS 4: Market Stress → PnL (opcional con CausalForest) ===
        if not self.skip_forest:
            try:
                Y = df["outcome_trade_pnl"].values
                T = df["treatment_market_stress"].astype(int).values
                
                Xc, Yc, Tc, _ = _clean_XYT(X, Y, T)
                
                if len(Xc) >= 50 and len(np.unique(Tc)) >= 2:
                    cf = CausalForestDML(
                        model_y=RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=1),
                        model_t=RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=1),
                        n_estimators=50,
                        min_samples_leaf=5,
                        random_state=42
                    )
                    cf.fit(Yc, Tc, X=Xc)
                    effects = cf.effect(Xc)
                    
                    results["stress_effects"] = {
                        "method": "CausalForestDML",
                        "average_treatment_effect": float(np.nanmean(effects)),
                        "heterogeneity_score": float(np.nanstd(effects)),
                        "stress_impact": {
                            "high_stress_trades": int((df["treatment_market_stress"] == 1).sum()),
                            "low_stress_trades": int((df["treatment_market_stress"] == 0).sum()),
                            "stress_penalty": float(np.nanmean(effects))
                        }
                    }
                    print("✅ Efectos de estrés estimados")
                else:
                    results["stress_effects"] = {"error": "insufficient_data"}
                    
            except Exception as e:
                results["stress_effects"] = {"error": str(e)}
        
        return results

    def analyze_heterogeneity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza heterogeneidad de efectos por asset y strategy"""
        print("📊 Analizando heterogeneidad...")
        
        heterogeneity = {
            "by_asset": {},
            "by_strategy": {},
            "by_asset_strategy": {},
            "summary_statistics": {}
        }
        
        try:
            # Por asset
            for asset in df["asset"].unique():
                asset_data = df[df["asset"] == asset]
                if len(asset_data) >= 10:
                    heterogeneity["by_asset"][asset] = {
                        "total_trades": len(asset_data),
                        "average_pnl": float(asset_data["outcome_trade_pnl"].mean()),
                        "win_rate": float((asset_data["outcome_trade_pnl"] > 0).mean()),
                        "profit_factor": self._calculate_profit_factor(asset_data),
                        "volatility": float(asset_data["outcome_trade_pnl"].std())
                    }
            
            # Por strategy
            for strategy in df["strategy"].unique():
                strategy_data = df[df["strategy"] == strategy]
                if len(strategy_data) >= 10:
                    heterogeneity["by_strategy"][strategy] = {
                        "total_trades": len(strategy_data),
                        "average_pnl": float(strategy_data["outcome_trade_pnl"].mean()),
                        "win_rate": float((strategy_data["outcome_trade_pnl"] > 0).mean()),
                        "profit_factor": self._calculate_profit_factor(strategy_data),
                        "assets_covered": len(strategy_data["asset"].unique())
                    }
            
            # Por combinación asset-strategy
            for (asset, strategy), group in df.groupby(["asset", "strategy"]):
                if len(group) >= 5:
                    heterogeneity["by_asset_strategy"][f"{asset}_{strategy}"] = {
                        "total_trades": len(group),
                        "average_pnl": float(group["outcome_trade_pnl"].mean()),
                        "win_rate": float((group["outcome_trade_pnl"] > 0).mean()),
                        "profit_factor": self._calculate_profit_factor(group),
                        "total_pnl": float(group["outcome_trade_pnl"].sum())
                    }
            
            # Estadísticas resumen
            heterogeneity["summary_statistics"] = {
                "total_assets": len(df["asset"].unique()),
                "total_strategies": len(df["strategy"].unique()),
                "total_combinations": len(df.groupby(["asset", "strategy"])),
                "best_asset": max(heterogeneity["by_asset"].items(), key=lambda x: x[1]["average_pnl"])[0] if heterogeneity["by_asset"] else None,
                "best_strategy": max(heterogeneity["by_strategy"].items(), key=lambda x: x[1]["average_pnl"])[0] if heterogeneity["by_strategy"] else None
            }
            
        except Exception as e:
            heterogeneity["error"] = str(e)
        
        return heterogeneity

    def _calculate_profit_factor(self, data):
        """Calcula profit factor de forma segura"""
        try:
            gross_profit = data[data["outcome_trade_pnl"] > 0]["outcome_trade_pnl"].sum()
            gross_loss = abs(data[data["outcome_trade_pnl"] < 0]["outcome_trade_pnl"].sum())
            
            if gross_loss == 0:
                return float('inf') if gross_profit > 0 else 0
            
            return gross_profit / gross_loss
        except:
            return 0

    def export_causal_insights(self, df: pd.DataFrame, causal_effects: Dict[str, Any], heterogeneity: Dict[str, Any]) -> str:
        """Exporta insights causales en formato para consolidate_global_insights.py"""
        print("📤 Exportando insights causales...")
        
        insights = []
        
        # Extraer mapas de efectos
        strategy_effects = causal_effects.get("strategy_effects", {}).get("strategy_effects", {})
        volatility_effects = causal_effects.get("volatility_effects", {}).get("volatility_effects", {})
        timing_effects = causal_effects.get("timing_effects", {}).get("session_effects", {})
        
        # Encontrar mejores condiciones
        best_volatility_regime = None
        if volatility_effects:
            best_volatility_regime = max(volatility_effects.items(), 
                                       key=lambda x: x[1].get("average_win_rate", 0))[0]
        
        best_session = None
        if timing_effects:
            best_session = max(timing_effects.items(), 
                             key=lambda x: x[1].get("average_pnl", 0))[0]
        
        # Policy value proxy
        policy_value = causal_effects.get("timing_effects", {}).get("london_vs_others_effect", 0.0)
        
        # Generar registro por combinación asset-strategy
        for (asset, strategy), group in df.groupby(["asset", "strategy"]):
            if len(group) < 5:
                continue
            
            insight = {
                "asset": asset,
                "strategy": strategy,
                "n_trades": len(group),
                "profit_factor": self._calculate_profit_factor(group),
                "win_rate": float((group["outcome_trade_pnl"] > 0).mean()),
                "total_pnl": float(group["outcome_trade_pnl"].sum()),
                "average_pnl": float(group["outcome_trade_pnl"].mean()),
                
                # Efectos causales
                "ATE": float(strategy_effects.get(strategy, {}).get("average_treatment_effect", 0.0)),
                "policy_value": float(policy_value),
                "best_session_causal": best_session,
                "best_volatility_regime_causal": best_volatility_regime,
                
                # Metadatos adicionales
                "causal_analysis_timestamp": pd.Timestamp.now().isoformat(),
                "data_quality_score": self._calculate_data_quality_score(group),
                "statistical_significance": self._assess_statistical_significance(group)
            }
            
            insights.append(insight)
        
        # Exportar archivo principal
        output_path = self.output_dir / "causal_reports" / "causal_insights.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(insights, f, indent=2, ensure_ascii=False)
        
        print(f"📝 Exportados {len(insights)} insights causales a: {output_path}")
        return str(output_path)

    def _calculate_data_quality_score(self, group):
        """Calcula score de calidad de datos (0-1)"""
        score = 0.0
        
        # Factor 1: Completitud de datos (30%)
        completeness = 1 - (group.isnull().sum().sum() / (len(group) * len(group.columns)))
        score += completeness * 0.3
        
        # Factor 2: Cantidad de trades (30%)
        trade_score = min(len(group) / 100, 1.0)  # Normalizado a 100 trades máximo
        score += trade_score * 0.3
        
        # Factor 3: Diversidad temporal (20%)
        if 'trade_entry_time' in group.columns:
            entry_times = pd.to_datetime(group['trade_entry_time'], errors='coerce')
            unique_days = entry_times.dt.date.nunique()
            temporal_score = min(unique_days / 30, 1.0)  # Normalizado a 30 días
        else:
            temporal_score = 0.5
        score += temporal_score * 0.2
        
        # Factor 4: Balance win/loss (20%)
        win_rate = (group["outcome_trade_pnl"] > 0).mean()
        balance_score = 1 - abs(win_rate - 0.5) * 2  # Penaliza extremos
        score += balance_score * 0.2
        
        return round(score, 3)

    def _assess_statistical_significance(self, group):
        """Evalúa significancia estadística de los resultados"""
        try:
            from scipy.stats import ttest_1samp
            
            # Test t de una muestra vs 0 (H0: PnL promedio = 0)
            pnl_values = group["outcome_trade_pnl"].dropna()
            
            if len(pnl_values) < 10:
                return {"significance": "insufficient_data", "p_value": None}
            
            t_stat, p_value = ttest_1samp(pnl_values, 0)
            
            if p_value < 0.01:
                significance = "highly_significant"
            elif p_value < 0.05:
                significance = "significant"
            elif p_value < 0.1:
                significance = "marginally_significant"
            else:
                significance = "not_significant"
            
            return {
                "significance": significance,
                "p_value": float(p_value),
                "t_statistic": float(t_stat)
            }
            
        except:
            return {"significance": "test_failed", "p_value": None}

    def export_detailed_results(self, df: pd.DataFrame, causal_effects: Dict[str, Any], heterogeneity: Dict[str, Any]) -> Dict[str, str]:
        """Exporta resultados detallados"""
        print("📋 Exportando resultados detallados...")
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        results_dir = self.output_dir / "causal_analysis_reports_v3"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # 1. Efectos causales completos
        effects_file = results_dir / f"causal_effects_{timestamp}.json"
        with open(effects_file, 'w', encoding='utf-8') as f:
            json.dump(causal_effects, f, indent=2, ensure_ascii=False, default=str)
        exported_files["causal_effects"] = str(effects_file)
        
        # 2. Análisis de heterogeneidad
        heterogeneity_file = results_dir / f"heterogeneity_analysis_{timestamp}.json"
        with open(heterogeneity_file, 'w', encoding='utf-8') as f:
            json.dump(heterogeneity, f, indent=2, ensure_ascii=False, default=str)
        exported_files["heterogeneity"] = str(heterogeneity_file)
        
        # 3. Resumen estadístico
        summary_stats = {
            "dataset_summary": {
                "total_observations": len(df),
                "unique_assets": df["asset"].nunique(),
                "unique_strategies": df["strategy"].nunique(),
                "date_range": {
                    "start": df['trade_entry_time'].min() if 'trade_entry_time' in df.columns else None,
                    "end": df['trade_entry_time'].max() if 'trade_entry_time' in df.columns else None
                },
                "pnl_statistics": {
                    "total_pnl": float(df["outcome_trade_pnl"].sum()),
                    "mean_pnl": float(df["outcome_trade_pnl"].mean()),
                    "std_pnl": float(df["outcome_trade_pnl"].std()),
                    "min_pnl": float(df["outcome_trade_pnl"].min()),
                    "max_pnl": float(df["outcome_trade_pnl"].max()),
                    "overall_win_rate": float((df["outcome_trade_pnl"] > 0).mean()),
                    "overall_profit_factor": self._calculate_profit_factor(df)
                }
            },
            "treatment_distributions": {
                "strategy_distribution": df["treatment_strategy_type"].value_counts().to_dict(),
                "volatility_distribution": df["treatment_volatility_regime"].value_counts().to_dict(),
                "session_distribution": df["treatment_session_timing"].value_counts().to_dict(),
                "stress_distribution": df["treatment_market_stress"].value_counts().to_dict()
            }
        }
        
        summary_file = results_dir / f"summary_statistics_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False, default=str)
        exported_files["summary_statistics"] = str(summary_file)
        
        return exported_files

    def run_integrated_analysis(self) -> Dict[str, Any]:
        """Ejecuta análisis causal integrado completo"""
        print("🚀 INICIANDO ANÁLISIS CAUSAL INTEGRADO")
        print("=" * 80)
        
        try:
            # 1. Cargar datos unificados
            print("📂 Cargando datos unificados...")
            df = self.load_unified_trade_data()
            
            if df.empty:
                return {"error": "no_data_loaded"}
            
            # 2. Construir features causales
            print("🔧 Construyendo features causales...")
            df_features = self.build_causal_features(df)
            
            # 3. Estimar efectos causales
            print("🧮 Estimando efectos causales...")
            causal_effects = self.estimate_causal_effects(df_features)
            
            # 4. Analizar heterogeneidad
            print("📊 Analizando heterogeneidad...")
            heterogeneity = self.analyze_heterogeneity(df_features)
            
            # 5. Exportar insights para orquestador
            print("📤 Exportando insights para orquestador...")
            insights_path = self.export_causal_insights(df_features, causal_effects, heterogeneity)
            
            # 6. Exportar resultados detallados
            print("📋 Exportando resultados detallados...")
            detailed_files = self.export_detailed_results(df_features, causal_effects, heterogeneity)
            
            print("\n✅ ANÁLISIS CAUSAL INTEGRADO COMPLETADO")
            print("=" * 80)
            
            return {
                "status": "success",
                "insights_path": insights_path,
                "detailed_files": detailed_files,
                "causal_effects": causal_effects,
                "heterogeneity": heterogeneity,
                "dataset_summary": {
                    "total_observations": len(df_features),
                    "unique_assets": df_features["asset"].nunique(),
                    "unique_strategies": df_features["strategy"].nunique(),
                    "total_pnl": float(df_features["outcome_trade_pnl"].sum()),
                    "overall_win_rate": float((df_features["outcome_trade_pnl"] > 0).mean())
                }
            }
            
        except Exception as e:
            print(f"❌ Error en análisis causal integrado: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}

# Función CLI
def main():
    """Función principal para CLI - Compatible con tu sistema existente"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Causal Trading Analyzer V3 - Versión Mejorada")
    # Mantener compatibilidad con tu CLI existente
    parser.add_argument("--input_dir", 
                       default="reports/advanced_reports",
                       help="Directorio con reportes de entrada")
    parser.add_argument("--out", default="reports", help="Directorio de salida")
    parser.add_argument("--no-forest", action="store_true", help="Desactivar CausalForest")
    parser.add_argument("--fast", action="store_true", help="Modo rápido (sin CausalForest)")
    parser.add_argument("--min-trades", type=int, default=20, help="Mínimo trades por análisis")
    parser.add_argument("--verbose", "-v", action="store_true", help="Salida detallada")
    
    args = parser.parse_args()
    
    # Determinar directorio de entrada
    if args.input_dir == "reports/advanced_reports":
        # Tu formato existente - buscar en advanced_reports
        input_dir = Path(args.input_dir)
    else:
        # Formato unificado si se especifica otro directorio
        input_dir = Path(args.input_dir)
    
    analyzer = IntegratedCausalAnalyzer(
        input_dir=input_dir,
        output_dir=Path(args.out),
        skip_forest=args.no_forest or args.fast,
        min_trades_per_analysis=args.min_trades
    )
    
    try:
        result = analyzer.run_integrated_analysis()
        
        if result.get("status") == "success":
            # Output compatible con tu sistema existente
            output_json = {
                "causal_orchestrator_json": result['insights_path'],
                **result['detailed_files']
            }
            
            if args.verbose:
                print(f"\n🎯 RESULTADOS:")
                print(f"   📁 Insights causales: {result['insights_path']}")
                print(f"   📊 Observaciones procesadas: {result['dataset_summary']['total_observations']}")
                print(f"   💰 PnL total: ${result['dataset_summary']['total_pnl']:.4f}")
                print(f"   📈 Win rate general: {result['dataset_summary']['overall_win_rate']:.1%}")
                
                print(f"\n📋 ARCHIVOS DETALLADOS:")
                for key, path in result['detailed_files'].items():
                    print(f"   • {key}: {path}")
            else:
                # Output simple compatible con tu script actual
                print(json.dumps(output_json, indent=2, ensure_ascii=False))
            
            print("✅ Listo")
            return 0
        else:
            print(f"\n❌ Error: {result.get('error', 'Unknown')}")
            return 2
            
    except Exception as e:
        print(f"\n💥 Error fatal: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    raise SystemExit(main())