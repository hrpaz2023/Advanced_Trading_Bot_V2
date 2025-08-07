# src/trading/trading_client.py
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import time
import pytz
from typing import Dict, Any, Optional

class TradingClient:
    def __init__(self, login, password, server, magic_number):
        self.login = login
        self.password = password  
        self.server = server
        self.magic_number = magic_number
        self.connected = False
        
        print(f"🔧 TradingClient inicializado:")
        print(f"   Login: {login}")
        print(f"   Server: {server}")
        print(f"   Magic: {magic_number}")
    
    def initialize(self):
        """Inicializar MT5 con debugging detallado"""
        try:
            print("🔌 Intentando inicializar MT5...")
            
            # Verificar si MT5 está instalado
            if not hasattr(mt5, 'initialize'):
                print("❌ Error: MetaTrader5 no está instalado correctamente")
                print("💡 Instala con: pip install MetaTrader5")
                return False
            
            # Inicializar MT5
            if not mt5.initialize():
                error_code = mt5.last_error()
                print(f"❌ MT5 initialize() falló")
                print(f"   Código de error: {error_code}")
                print(f"💡 Verifica:")
                print(f"   • MT5 Terminal está instalado")
                print(f"   • MT5 Terminal está cerrado (debe estar cerrado para la conexión)")
                return False
            
            print("✅ MT5 inicializado correctamente")
            
            # Intentar login
            print(f"🔑 Intentando login...")
            print(f"   Login: {self.login}")
            print(f"   Server: {self.server}")
            
            # Verificar que el password no esté vacío
            if not self.password or self.password == "xxxx":
                print("❌ Error: Password no configurado en global_config.json")
                print("💡 Actualiza el password en configs/global_config.json")
                mt5.shutdown()
                return False
            
            login_result = mt5.login(
                login=int(self.login), 
                password=str(self.password), 
                server=str(self.server)
            )
            
            if not login_result:
                error_code = mt5.last_error()
                print(f"❌ MT5 login falló")
                print(f"   Código de error: {error_code}")
                print(f"💡 Posibles causas:")
                print(f"   • Credenciales incorrectas")
                print(f"   • Servidor no disponible")
                print(f"   • Cuenta bloqueada/suspendida")
                print(f"   • Conexión a internet")
                mt5.shutdown()
                return False
            
            print("✅ Login exitoso")
            
            # Verificar información de cuenta
            account_info = mt5.account_info()
            if account_info is None:
                print("⚠️ No se pudo obtener información de cuenta")
            else:
                print(f"✅ Cuenta conectada:")
                print(f"   Nombre: {account_info.name}")
                print(f"   Balance: ${account_info.balance:,.2f}")
                print(f"   Equity: ${account_info.equity:,.2f}")
                print(f"   Compañía: {account_info.company}")
                print(f"   Moneda: {account_info.currency}")
                
                # Verificar trading permitido
                if hasattr(account_info, 'trade_allowed'):
                    if account_info.trade_allowed:
                        print("✅ Trading automático permitido")
                    else:
                        print("⚠️ Trading automático NO permitido")
                        print("💡 Habilita trading automático en MT5")
            
            self.connected = True
            return True
            
        except Exception as e:
            print(f"❌ Error inesperado en initialize(): {e}")
            print(f"💡 Verifica que MT5 esté instalado correctamente")
            try:
                mt5.shutdown()
            except:
                pass
            return False
    
    def get_account_info(self):
        """Obtener info de cuenta"""
        if not self.connected:
            return {}
            
        try:
            info = mt5.account_info()
            return info._asdict() if info else {}
        except Exception as e:
            print(f"❌ Error obteniendo info de cuenta: {e}")
            return {}
    
    def get_account_balance(self):
        """Obtener balance"""
        if not self.connected:
            return 0
            
        try:
            info = mt5.account_info()
            return info.balance if info else 0
        except Exception as e:
            print(f"❌ Error obteniendo balance: {e}")
            return 0

    def get_candles(self, symbol, timeframe, count):
        """Obtener datos OHLC corrigiendo zona horaria UTC+3"""
        try:
            # Mapear timeframes
            timeframe_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            
            mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_M5)
            
            # Obtener datos de MT5
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
            
            if rates is None or len(rates) == 0:
                print(f"❌ No se pudieron obtener datos para {symbol}")
                return None
            
            # Convertir a DataFrame
            df = pd.DataFrame(rates)
            
            # Corrección crítica: FTMO servidor usa UTC+3
            df['time'] = pd.to_datetime(df['time'], unit='s') - timedelta(hours=3)
            
            # Renombrar columnas
            df.rename(columns={
                'time': 'Time', 'open': 'Open', 'high': 'High', 
                'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume',
                'spread':'Spread', 'real_volume': 'RealVolume'
            }, inplace=True)
            
            # Asegurar columnas requeridas
            required_cols = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Spread']
            for col in required_cols:
                if col not in df.columns:
                     df[col] = 0

            df = df[required_cols]
            df.set_index('Time', inplace=True)
            
            print(f"✅ Datos obtenidos para {symbol}: {len(df)} velas")
            return df
            
        except Exception as e:
            print(f"❌ Error obteniendo datos de {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def verify_timezone_correction(self):
        """Verifica que la corrección de zona horaria funciona"""
        try:
            print(f"🔍 Verificando corrección de zona horaria...")
            
            df = self.get_candles('EURUSD', 'M5', 3)
            if df is not None and len(df) > 0:
                latest_time = df.index[-1]
                current_time = datetime.now(pytz.UTC).replace(tzinfo=None)
                age_minutes = (current_time - latest_time.to_pydatetime()).total_seconds() / 60
                
                print(f"📊 Test de corrección:")
                print(f"    Última vela: {latest_time}")
                print(f"    Hora actual: {current_time}")
                print(f"    Diferencia: {age_minutes:.1f} minutos")
                
                if 0 <= age_minutes <= 15:
                    print(f"✅ Corrección UTC+3 funcionando correctamente")
                    return True
                else:
                    print(f"❌ Corrección no funciona - diferencia: {age_minutes:.1f} min")
                    return False
            else:
                print(f"❌ No se pudieron obtener datos para verificación")
                return False
                
        except Exception as e:
            print(f"❌ Error verificando corrección: {e}")
            return False

    # =========================================================================
    # 🚀 MÉTODOS DE EJECUCIÓN DE ÓRDENES (NUEVOS)
    # =========================================================================
    
    def _validate_symbol(self, symbol: str) -> bool:
        """Validar y preparar símbolo para trading"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"❌ Símbolo {symbol} no encontrado")
                return False
            
            # Activar símbolo si no está visible
            if not symbol_info.visible:
                print(f"⚠️ Activando símbolo {symbol}...")
                if not mt5.symbol_select(symbol, True):
                    print(f"❌ No se pudo activar {symbol}")
                    return False
            
            # Verificar si está disponible para trading
            if not symbol_info.trade_mode & mt5.SYMBOL_TRADE_MODE_FULL:
                print(f"⚠️ {symbol} no disponible para trading completo")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error validando símbolo {symbol}: {e}")
            return False
    
    def _get_current_price(self, symbol: str, side: str) -> Optional[float]:
        """Obtener precio actual para la operación"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                print(f"❌ No se pudo obtener precio para {symbol}")
                return None
            
            # BUY usa Ask, SELL usa Bid
            price = tick.ask if side.upper() == "BUY" else tick.bid
            print(f"💰 Precio actual {symbol} {side}: {price}")
            return price
            
        except Exception as e:
            print(f"❌ Error obteniendo precio para {symbol}: {e}")
            return None
    
    def market_order(self, symbol: str, side: str, lots: float, price: Optional[float] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        🎯 MÉTODO PRINCIPAL: Ejecutar orden de mercado
        Este es el método que ExecutionController intentará llamar primero
        """
        if not self.connected:
            return {"error": "MT5 no conectado"}
        
        try:
            print(f"🎯 Ejecutando orden de mercado:")
            print(f"   Símbolo: {symbol}")
            print(f"   Lado: {side}")
            print(f"   Lotes: {lots}")
            print(f"   Magic: {self.magic_number}")
            
            # Validar símbolo
            if not self._validate_symbol(symbol):
                return {"error": f"Símbolo {symbol} no válido para trading"}
            
            # Obtener precio actual
            current_price = self._get_current_price(symbol, side)
            if current_price is None:
                return {"error": f"No se pudo obtener precio para {symbol}"}
            
            # Usar precio actual si no se especifica
            if price is None:
                price = current_price
                
            # Determinar tipo de orden
            order_type = mt5.ORDER_TYPE_BUY if side.upper() == "BUY" else mt5.ORDER_TYPE_SELL
            
            # Preparar comentario
            comment = f"Bot_{self.magic_number}"
            if metadata:
                strategy = metadata.get("strategy", "")
                timestamp = metadata.get("timestamp", "")
                if strategy:
                    comment += f"_{strategy}"
                if timestamp:
                    comment += f"_{timestamp[:10]}"  # Solo fecha
            
            # Construir request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(lots),
                "type": order_type,
                "price": float(price),
                "deviation": 20,  # Desviación en puntos
                "magic": int(self.magic_number),
                "comment": comment[:31],  # MT5 limita a 31 caracteres
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            print(f"📤 Enviando orden:")
            print(f"   Request: {request}")
            
            # Enviar orden
            result = mt5.order_send(request)
            
            if result is None:
                return {"error": "order_send devolvió None"}
            
            print(f"📥 Resultado MT5:")
            print(f"   Retcode: {result.retcode}")
            print(f"   Comment: {result.comment if hasattr(result, 'comment') else 'N/A'}")
            
            # Verificar resultado
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"MT5 Error {result.retcode}"
                if hasattr(result, 'comment'):
                    error_msg += f": {result.comment}"
                
                # Mapear códigos de error comunes
                error_codes = {
                    10004: "Requote - precio cambió",
                    10006: "Request rejected - solicitud rechazada",
                    10007: "Request canceled - solicitud cancelada", 
                    10008: "Order placed - orden colocada (pendiente)",
                    10009: "Request completed - completado",
                    10010: "Request partial - parcialmente completado",
                    10011: "Request processing error - error de procesamiento",
                    10012: "Request canceled by timeout - cancelado por timeout",
                    10013: "Invalid request - request inválido",
                    10014: "Invalid volume - volumen inválido",
                    10015: "Invalid price - precio inválido",
                    10016: "Invalid stops - stops inválidos",
                    10017: "Trade disabled - trading deshabilitado",
                    10018: "Market closed - mercado cerrado",
                    10019: "Not enough money - fondos insuficientes",
                    10020: "Price changed - precio cambió",
                    10021: "Off quotes - sin cotizaciones",
                    10022: "Invalid expiration - expiración inválida",
                    10023: "Order state changed - estado de orden cambió",
                    10024: "Too frequent requests - requests muy frecuentes",
                    10025: "No changes - sin cambios",
                    10026: "Auto trading disabled - auto trading deshabilitado",
                    10027: "Auto trading disabled by server - auto trading deshabilitado por servidor",
                    10028: "Auto trading disabled by client - auto trading deshabilitado por cliente",
                    10029: "Request locked - request bloqueado",
                    10030: "Order or position frozen - orden/posición congelada",
                    10031: "Invalid symbol type - tipo de símbolo inválido",
                }
                
                detailed_error = error_codes.get(result.retcode, "Error desconocido")
                print(f"❌ {error_msg} - {detailed_error}")
                
                return {"error": f"{error_msg} - {detailed_error}"}
            
            # Éxito
            ticket = result.order if hasattr(result, 'order') else result.deal
            executed_price = result.price if hasattr(result, 'price') else price
            executed_volume = result.volume if hasattr(result, 'volume') else lots
            
            success_result = {
                "ticket": ticket,
                "price": executed_price,
                "volume": executed_volume,
                "retcode": result.retcode
            }
            
            print(f"✅ Orden ejecutada exitosamente:")
            print(f"   Ticket: {ticket}")
            print(f"   Precio: {executed_price}")
            print(f"   Volumen: {executed_volume}")
            
            return success_result
            
        except Exception as e:
            error_msg = f"Error inesperado en market_order: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return {"error": error_msg}
    
    def send_order(self, symbol: str, side: str, lots: float, price: Optional[float] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Alias para market_order - ExecutionController también busca este método"""
        return self.market_order(symbol, side, lots, price, metadata)
    
    def place_market_order(self, symbol: str, side: str, lots: float) -> Dict[str, Any]:
        """Versión simplificada sin metadata - ExecutionController también busca este método"""
        return self.market_order(symbol, side, lots)
    
    def order_send(self, symbol: str, side: str, lots: float, price: float) -> Dict[str, Any]:
        """Versión con precio fijo - ExecutionController también busca este método"""
        return self.market_order(symbol, side, lots, price)

    # =========================================================================
    # MÉTODOS AUXILIARES Y LEGACY
    # =========================================================================
    
    def place_order(self, symbol, action, lots, price, sl, tp, comment):
        """Método legacy - mantener por compatibilidad"""
        if not self.connected:
            print(f"❌ No conectado a MT5")
            return None
            
        try:
            # Ahora usa el nuevo sistema
            result = self.market_order(
                symbol=symbol,
                side=action,
                lots=lots,
                price=price,
                metadata={"comment": comment}
            )
            
            if result.get("ticket"):
                return result["ticket"]
            else:
                print(f"❌ Error en place_order: {result.get('error')}")
                return None
                
        except Exception as e:
            print(f"❌ Error ejecutando orden legacy: {e}")
            return None
    
    def get_positions(self):
        """Obtener posiciones abiertas"""
        if not self.connected:
            return []
            
        try:
            positions = mt5.positions_get()
            return list(positions) if positions else []
        except Exception as e:
            print(f"❌ Error obteniendo posiciones: {e}")
            return []
    
    def close_position(self, ticket):
        """Cerrar posición por ticket"""
        if not self.connected:
            return False
            
        try:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                print(f"❌ Posición {ticket} no encontrada")
                return False
                
            position = positions[0]
            
            # Determinar acción contraria
            close_action = "SELL" if position.type == mt5.POSITION_TYPE_BUY else "BUY"
            
            # Cerrar usando market_order
            result = self.market_order(
                symbol=position.symbol,
                side=close_action,
                lots=position.volume,
                metadata={"comment": f"Close_{ticket}"}
            )
            
            return result.get("ticket") is not None
            
        except Exception as e:
            print(f"❌ Error cerrando posición {ticket}: {e}")
            return False
    
    def shutdown(self):
        """Cerrar MT5"""
        try:
            if self.connected:
                mt5.shutdown()
                print("🛑 MT5 desconectado")
                self.connected = False
        except Exception as e:
            print(f"⚠️ Error cerrando MT5: {e}")

# =========================================================================
# FUNCIÓN DE PRUEBA MEJORADA
# =========================================================================

def test_connection():
    """Prueba completa de conexión y ejecución"""
    print("🧪 PRUEBA COMPLETA DE TRADING CLIENT")
    print("=" * 50)
    
    # Credenciales - ACTUALIZA ESTAS
    client = TradingClient(
        login=1511119883,  # Tu login real
        password="tu_password_real",  # ← CAMBIAR AQUÍ  
        server="FTMO-Demo",
        magic_number=202401
    )
    
    try:
        # Test 1: Conexión
        print("\n🔌 Test 1: Conexión")
        if not client.initialize():
            print("❌ Conexión falló - no se puede continuar")
            return
        
        # Test 2: Datos
        print("\n📊 Test 2: Obtener datos")
        data = client.get_candles("EURUSD", "M5", 10)
        if data is not None:
            print(f"✅ Datos obtenidos: {len(data)} velas")
            print(f"   Última vela: {data.index[-1]}")
        else:
            print("❌ No se pudieron obtener datos")
        
        # Test 3: Validación de símbolo
        print("\n🎯 Test 3: Validación de símbolo")
        if client._validate_symbol("EURUSD"):
            print("✅ EURUSD validado para trading")
        else:
            print("❌ EURUSD no válido")
        
        # Test 4: Precio actual
        print("\n💰 Test 4: Obtener precios")
        buy_price = client._get_current_price("EURUSD", "BUY")
        sell_price = client._get_current_price("EURUSD", "SELL")
        if buy_price and sell_price:
            print(f"✅ EURUSD - BUY: {buy_price}, SELL: {sell_price}")
            print(f"   Spread: {(buy_price - sell_price) * 10000:.1f} pips")
        
        # Test 5: Orden simulada (comentar si no quieres orden real)
        print("\n🚨 Test 5: Orden de prueba (SIMULACIÓN)")
        print("⚠️ Para orden real, descomenta las líneas siguientes")
        
        # DESCOMENTA ESTAS LÍNEAS SOLO SI QUIERES HACER UNA ORDEN REAL:
        # result = client.market_order(
        #     symbol="EURUSD",
        #     side="BUY", 
        #     lots=0.01,  # Micro lote
        #     metadata={"strategy": "test", "timestamp": "2024-01-01"}
        # )
        # 
        # if result.get("ticket"):
        #     print(f"✅ Orden test ejecutada: {result}")
        # else:
        #     print(f"❌ Orden test falló: {result.get('error')}")
        
        print("✅ Todos los tests completados")
        
    finally:
        client.shutdown()

if __name__ == "__main__":
    test_connection()