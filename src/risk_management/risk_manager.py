"""
Risk Manager mejorado para position sizing y stop loss precision
Versión optimizada para trading bot con FTMO compliance
"""

class RiskManager:
    """
    Risk Manager mejorado para cálculos de position sizing y trade params
    """
    
    def __init__(self, risk_per_trade=1.0):
        self.risk_per_trade = risk_per_trade
        self.name = "Enhanced Risk Manager"
        
        # ✅ VALORES DE PIP POR SÍMBOLO
        self.pip_values = {
            'EURUSD': 10.0, 'GBPUSD': 10.0, 'AUDUSD': 10.0, 'NZDUSD': 10.0,
            'USDJPY': 9.09, 'USDCAD': 7.69, 'USDCHF': 10.0,
            'EURGBP': 12.82, 'EURJPY': 9.09, 'GBPJPY': 9.09,
            'XAUUSD': 1.0, 'US30': 1.0, 'NAS100': 1.0, 'GER40': 1.0
        }
        
        # ✅ TAMAÑOS DE PIP POR SÍMBOLO
        self.pip_sizes = {
            'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'AUDUSD': 0.0001, 'NZDUSD': 0.0001,
            'USDJPY': 0.01, 'USDCAD': 0.0001, 'USDCHF': 0.0001,
            'EURGBP': 0.0001, 'EURJPY': 0.01, 'GBPJPY': 0.01,
            'XAUUSD': 0.01, 'US30': 1.0, 'NAS100': 1.0, 'GER40': 1.0
        }
    
    def get_pip_value(self, symbol):
        """Obtiene valor de pip para símbolo específico"""
        return self.pip_values.get(symbol, 10.0)  # Default para majors
    
    def get_pip_size(self, symbol):
        """Obtiene tamaño de pip para símbolo específico"""
        return self.pip_sizes.get(symbol, 0.0001)  # Default para majors
    
    def calculate_position_size(self, account_equity, stop_loss_pips, symbol="EURUSD", risk_pct=None):
        """
        Calcula tamaño de posición en lotes.
        Reglas:
        - Si hay override fijo en configs/risk_config.json -> usarlo (también por símbolo).
        - Si no hay override, usar cálculo por % de riesgo (comportamiento actual).
        Siempre normaliza al step/min/max de MT5 si disponible.
        """
        import json, os
        try:
            # --- helpers locales ---
            def _load_fixed_override(sym: str):
                """Lee configs/risk_config.json (opcional) y devuelve lots fijos si corresponden."""
                paths = ["configs/risk_config.json", "risk_config.json"]
                for p in paths:
                    if os.path.exists(p):
                        try:
                            with open(p, "r", encoding="utf-8") as f:
                                cfg = json.load(f)
                            ps = cfg.get("position_sizing", cfg)
                            mode = str(ps.get("mode", "")).lower()
                            # override por símbolo
                            sym_over = ps.get("symbol_overrides", {}) or {}
                            if sym in sym_over:
                                return float(sym_over[sym])
                            # modo fijo global
                            if mode == "fixed":
                                fl = ps.get("fixed_lots", ps.get("base_lot"))
                                if fl is not None:
                                    return float(fl)
                        except Exception:
                            pass
                return None

            def _mt5_round_volume(sym: str, lots: float) -> float:
                """Redondeo/clamp a parámetros del símbolo en MT5. Fallback step=0.01, [0.01, 100]."""
                try:
                    import MetaTrader5 as mt5
                    info = mt5.symbol_info(sym)
                    if info is None:
                        step = 0.01; vmin = 0.01; vmax = 100.0
                    else:
                        step = (getattr(info, "volume_step", 0.01) or 0.01)
                        vmin = (getattr(info, "volume_min", 0.01) or 0.01)
                        vmax = (getattr(info, "volume_max", 100.0) or 100.0)
                    rounded = round(float(lots) / step) * step
                    return max(vmin, min(rounded, vmax))
                except Exception:
                    return max(0.01, min(round(float(lots), 2), 100.0))
            # --- fin helpers ---

            # 0) Override fijo desde config (prioridad absoluta)
            fixed = _load_fixed_override(symbol)
            if fixed is not None:
                return _mt5_round_volume(symbol, float(fixed))

            # 1) VALIDACIONES BÁSICAS (mismo comportamiento que tenías)
            if account_equity <= 0:
                print(f"⚠️ Account equity inválido: {account_equity}")
                return 0.01
            if stop_loss_pips <= 0:
                print(f"⚠️ Stop loss pips inválido: {stop_loss_pips}, usando 20 por defecto")
                stop_loss_pips = 20

            # 2) CÁLCULO POR % RIESGO (como antes)
            risk_percentage = (risk_pct or self.risk_per_trade)
            risk_amount = account_equity * (risk_percentage / 100.0)

            pip_value = self.get_pip_value(symbol)
            position_size = risk_amount / (stop_loss_pips * pip_value)

            # 3) NORMALIZACIÓN: primero bounding genérico y luego step de MT5
            normalized_size = round(position_size, 2)
            bounded = max(0.01, min(10.0, normalized_size))

            # 4) Aplicar step/min/max reales del símbolo en MT5
            final_size = _mt5_round_volume(symbol, bounded)
            return final_size

        except Exception as e:
            print(f"❌ Error calculando position size: {e}")
            return 0.01


    
    def get_trade_params(self, entry_price, atr, side, symbol="EURUSD"):
        """
        Calcula stop loss y take profit basado en ATR
        
        Args:
            entry_price: Precio de entrada
            atr: Average True Range
            side: 1 para BUY, -1 para SELL
            symbol: Símbolo del instrumento
            
        Returns:
            tuple: (stop_loss, take_profit)
        """
        try:
            # ✅ VALIDAR Y AJUSTAR ATR
            if atr <= 0 or atr > entry_price * 0.1:  # ATR no puede ser > 10% del precio
                # Calcular ATR como porcentaje del precio (más realista)
                atr = entry_price * 0.002  # 0.2% del precio como fallback
                print(f"⚠️ ATR ajustado a {atr:.5f} para {symbol}")
            
            # ✅ MULTIPLICADORES ADAPTATIVOS SEGÚN SÍMBOLO
            if 'JPY' in symbol:
                atr_multiplier = 1.2  # JPY pairs más volátiles
                rr_ratio = 1.8
            elif symbol in ['XAUUSD', 'US30', 'NAS100', 'GER40']:
                atr_multiplier = 2.5  # Índices/commodities más espacio
                rr_ratio = 2.2
            else:
                atr_multiplier = 1.5  # Majors estándar
                rr_ratio = 2.0
            
            stop_distance = atr * atr_multiplier
            tp_distance = stop_distance * rr_ratio
            
            if side == 1:  # BUY
                sl = entry_price - stop_distance
                tp = entry_price + tp_distance
            else:  # SELL
                sl = entry_price + stop_distance
                tp = entry_price - tp_distance
            
            # ✅ VALIDAR QUE LOS PRECIOS SEAN RAZONABLES
            if sl <= 0 or tp <= 0:
                raise ValueError(f"Precios SL/TP inválidos: SL={sl}, TP={tp}")
            
            return sl, tp
            
        except Exception as e:
            print(f"❌ Error calculando trade params para {symbol}: {e}")
            # ✅ FALLBACK MÁS ROBUSTO
            fallback_distance = entry_price * 0.001  # 0.1% del precio
            
            if side == 1:
                return entry_price - fallback_distance, entry_price + (fallback_distance * 2)
            else:
                return entry_price + fallback_distance, entry_price - (fallback_distance * 2)
    
    def validate_trade(self, symbol, entry_price, stop_loss, take_profit, side):
        """
        Valida parámetros de trade de forma exhaustiva
        
        Args:
            symbol: Símbolo del instrumento
            entry_price: Precio de entrada
            stop_loss: Precio de stop loss
            take_profit: Precio de take profit
            side: 1 para BUY, -1 para SELL
            
        Returns:
            tuple: (bool, str) - (es_válido, razón)
        """
        try:
            # ✅ VALIDACIONES BÁSICAS
            if entry_price <= 0:
                return False, f"Entry price inválido: {entry_price}"
            
            if stop_loss <= 0:
                return False, f"Stop loss inválido: {stop_loss}"
            
            if take_profit <= 0:
                return False, f"Take profit inválido: {take_profit}"
            
            # ✅ VALIDAR DIRECCIONES SEGÚN SIDE
            if side == 1:  # BUY
                if stop_loss >= entry_price:
                    return False, f"SL debe ser menor que entry para BUY: SL={stop_loss}, Entry={entry_price}"
                
                if take_profit <= entry_price:
                    return False, f"TP debe ser mayor que entry para BUY: TP={take_profit}, Entry={entry_price}"
            
            elif side == -1:  # SELL
                if stop_loss <= entry_price:
                    return False, f"SL debe ser mayor que entry para SELL: SL={stop_loss}, Entry={entry_price}"
                
                if take_profit >= entry_price:
                    return False, f"TP debe ser menor que entry para SELL: TP={take_profit}, Entry={entry_price}"
            
            else:
                return False, f"Side inválido: {side} (debe ser 1 o -1)"
            
            # ✅ VALIDAR DISTANCIAS MÍNIMAS
            pip_size = self.get_pip_size(symbol)
            min_distance = pip_size * 10  # Mínimo 10 pips
            
            sl_distance = abs(entry_price - stop_loss)
            tp_distance = abs(entry_price - take_profit)
            
            if sl_distance < min_distance:
                return False, f"Distancia SL muy pequeña: {sl_distance/pip_size:.1f} pips (min 10)"
            
            if tp_distance < min_distance:
                return False, f"Distancia TP muy pequeña: {tp_distance/pip_size:.1f} pips (min 10)"
            
            # ✅ VALIDAR RISK/REWARD MÍNIMO
            risk_reward = tp_distance / sl_distance
            if risk_reward < 1.0:
                return False, f"Risk/Reward muy bajo: {risk_reward:.2f} (min 1.0)"
            
            return True, "Trade válido"
            
        except Exception as e:
            return False, f"Error validando trade: {e}"
    
    def get_stop_loss(self, entry_price, atr, side, symbol="EURUSD"):
        """Calcula solo el stop loss"""
        sl, _ = self.get_trade_params(entry_price, atr, side, symbol)
        return sl
    
    def get_take_profit(self, entry_price, atr, side, symbol="EURUSD"):
        """Calcula solo el take profit"""
        _, tp = self.get_trade_params(entry_price, atr, side, symbol)
        return tp
    
    def calculate_risk_reward(self, entry_price, stop_loss, take_profit, side):
        """
        Calcula ratio risk/reward
        
        Returns:
            float: Ratio R/R (ej: 2.0 = 1:2)
        """
        try:
            if side == 1:  # BUY
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
            else:  # SELL
                risk = stop_loss - entry_price
                reward = entry_price - take_profit
            
            if risk <= 0:
                return 0.0
            
            return reward / risk
            
        except Exception as e:
            print(f"❌ Error calculando R/R: {e}")
            return 0.0
    
    def get_position_info(self, symbol, entry_price, stop_loss, take_profit, lot_size, side):
        """
        Retorna información completa de la posición
        
        Returns:
            dict: Información detallada de la posición
        """
        try:
            pip_size = self.get_pip_size(symbol)
            pip_value = self.get_pip_value(symbol)
            
            # Calcular distancias en pips
            sl_pips = abs(entry_price - stop_loss) / pip_size
            tp_pips = abs(entry_price - take_profit) / pip_size
            
            # Calcular valores monetarios
            risk_amount = sl_pips * pip_value * lot_size
            reward_amount = tp_pips * pip_value * lot_size
            
            # Risk/Reward ratio
            rr_ratio = self.calculate_risk_reward(entry_price, stop_loss, take_profit, side)
            
            return {
                'symbol': symbol,
                'side': 'BUY' if side == 1 else 'SELL',
                'lot_size': lot_size,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'sl_pips': round(sl_pips, 1),
                'tp_pips': round(tp_pips, 1),
                'risk_amount': round(risk_amount, 2),
                'reward_amount': round(reward_amount, 2),
                'risk_reward_ratio': round(rr_ratio, 2)
            }
            
        except Exception as e:
            print(f"❌ Error obteniendo info de posición: {e}")
            return {}

# ✅ FUNCIÓN DE UTILIDAD PARA TESTING
def test_risk_manager():
    """Función para probar el RiskManager"""
    
    print("🧪 TESTING RISK MANAGER")
    print("=" * 40)
    
    rm = RiskManager(risk_per_trade=1.0)
    
    # Test 1: Position sizing
    print("\n1️⃣ Test Position Sizing:")
    equity = 10000
    sl_pips = 20
    symbol = "EURUSD"
    
    position_size = rm.calculate_position_size(equity, sl_pips, symbol)
    print(f"   Equity: ${equity}, SL: {sl_pips} pips")
    print(f"   Position Size: {position_size} lotes")
    
    # Test 2: Trade params
    print("\n2️⃣ Test Trade Params:")
    entry_price = 1.1000
    atr = 0.0015
    side = 1
    
    sl, tp = rm.get_trade_params(entry_price, atr, side, symbol)
    print(f"   Entry: {entry_price}, ATR: {atr}")
    print(f"   SL: {sl:.5f}, TP: {tp:.5f}")
    
    # Test 3: Validation
    print("\n3️⃣ Test Validation:")
    is_valid, reason = rm.validate_trade(symbol, entry_price, sl, tp, side)
    print(f"   Valid: {is_valid}, Reason: {reason}")
    
    # Test 4: Position info
    print("\n4️⃣ Test Position Info:")
    info = rm.get_position_info(symbol, entry_price, sl, tp, position_size, side)
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Testing completado")

if __name__ == "__main__":
    test_risk_manager()