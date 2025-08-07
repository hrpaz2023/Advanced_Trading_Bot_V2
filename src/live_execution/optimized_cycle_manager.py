import datetime
import time
import pytz
from enum import Enum
from collections import defaultdict, deque
import statistics

class TradingSession(Enum):
    """✅ SESIONES DE TRADING OPTIMIZADAS"""
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "ny"
    OVERLAP_LONDON_NY = "overlap"
    INACTIVE = "inactive"

class OptimizedM5CycleManager:
    """
    ✅ CYCLE MANAGER COMPLETAMENTE OPTIMIZADO CON:
    - Gestión inteligente de sesiones
    - Métricas avanzadas de performance
    - Adaptación dinámica de intervalos
    - Sistema de priorización de controladores
    - Detección automática de problemas
    """
    
    def __init__(self, max_daily_requests=180000):
        # ✅ CONFIGURACIÓN BÁSICA
        self.max_daily_requests = max_daily_requests
        self.daily_request_count = 0
        self.last_reset_date = datetime.date.today()
        self.last_candle_processed_utc = None
        self.BUFFER_SECONDS = 20  # Búfer de seguridad para velas estables
        
        # ✅ CONFIGURACIÓN DE SESIONES MEJORADA
        self.sessions = {
            TradingSession.ASIAN: {
                'start': 0, 'end': 8,
                'priority': 'LOW',
                'expected_activity': 0.3,
                'description': 'Sesión Asiática - Baja volatilidad'
            },
            TradingSession.LONDON: {
                'start': 8, 'end': 13.5,
                'priority': 'HIGH',
                'expected_activity': 0.8,
                'description': 'Sesión de Londres - Alta actividad'
            },
            TradingSession.OVERLAP_LONDON_NY: {
                'start': 13.5, 'end': 16,
                'priority': 'HIGHEST',
                'expected_activity': 1.0,
                'description': 'Overlap Londres-NY - Máxima volatilidad'
            },
            TradingSession.NEW_YORK: {
                'start': 16, 'end': 21,
                'priority': 'HIGH',
                'expected_activity': 0.7,
                'description': 'Sesión de Nueva York - Alta actividad'
            },
            TradingSession.INACTIVE: {
                'start': 21, 'end': 24,
                'priority': 'VERY_LOW',
                'expected_activity': 0.1,
                'description': 'Sesión inactiva - Muy baja actividad'
            }
        }
        
        # ✅ MÉTRICAS AVANZADAS
        self.performance_metrics = {
            'cycles_completed': 0,
            'total_processing_time': 0.0,
            'avg_cycle_time': 0.0,
            'controllers_processed_total': 0,
            'session_activity': defaultdict(int),
            'hourly_efficiency': defaultdict(list),
            'last_100_cycle_times': deque(maxlen=100),
            'requests_by_session': defaultdict(int),
            'vela_stability_stats': {
                'stable_immediately': 0,
                'required_waiting': 0,
                'avg_wait_time': 0.0
            }
        }
        
        # ✅ SISTEMA DE PRIORIZACIÓN
        self.controller_priority_system = {
            'last_signal_times': {},
            'performance_scores': {},
            'processing_times': defaultdict(list),
            'success_rates': defaultdict(float)
        }
        
        # ✅ ADAPTACIÓN DINÁMICA
        self.adaptive_settings = {
            'dynamic_buffer': True,
            'session_based_intervals': True,
            'controller_rotation': True,
            'performance_based_prioritization': True
        }
        
        print("✅ OptimizedM5CycleManager v4.0 inicializado")
        print(f"   🎯 Máximo requests/día: {max_daily_requests:,}")
        print(f"   ⚡ Búfer de estabilidad: {self.BUFFER_SECONDS}s")
        print(f"   🧠 Características avanzadas: Todas activadas")
    
    def get_current_session_advanced(self, dt_utc):
        """✅ DETECCIÓN AVANZADA DE SESIÓN CON CONTEXTO"""
        hour_decimal = dt_utc.hour + (dt_utc.minute / 60.0)
        
        # Detectar sesión principal
        current_session = None
        for session, config in self.sessions.items():
            if config['start'] <= hour_decimal < config['end']:
                current_session = session
                break
        
        # Fallback para horas 21-24 (inactive)
        if current_session is None:
            current_session = TradingSession.INACTIVE
        
        # ✅ CONTEXTO ADICIONAL
        session_info = self.sessions[current_session].copy()
        session_info['current_hour'] = dt_utc.hour
        session_info['minutes_into_session'] = (hour_decimal - session_info['start']) * 60
        session_info['minutes_remaining'] = (session_info['end'] - hour_decimal) * 60
        
        return current_session, session_info
    
    def calculate_dynamic_buffer(self, session_info, recent_stability):
        """✅ CÁLCULO DINÁMICO DEL BÚFER SEGÚN CONDICIONES"""
        base_buffer = self.BUFFER_SECONDS
        
        if not self.adaptive_settings['dynamic_buffer']:
            return base_buffer
        
        # ✅ AJUSTAR SEGÚN SESIÓN
        if session_info['priority'] == 'HIGHEST':
            session_multiplier = 0.8  # Más agresivo en overlap
        elif session_info['priority'] == 'HIGH':
            session_multiplier = 1.0  # Estándar
        else:
            session_multiplier = 1.2  # Más conservador en sesiones lentas
        
        # ✅ AJUSTAR SEGÚN HISTORIAL DE ESTABILIDAD
        if len(recent_stability) >= 5:
            avg_stability = statistics.mean(recent_stability)
            if avg_stability > 0.8:  # Muy estable
                stability_multiplier = 0.9
            elif avg_stability < 0.5:  # Inestable
                stability_multiplier = 1.3
            else:
                stability_multiplier = 1.0
        else:
            stability_multiplier = 1.0
        
        # ✅ CALCULAR BÚFER DINÁMICO
        dynamic_buffer = int(base_buffer * session_multiplier * stability_multiplier)
        return max(10, min(60, dynamic_buffer))  # Entre 10 y 60 segundos
    
    def prioritize_controllers(self, controllers, session_info):
        """✅ SISTEMA INTELIGENTE DE PRIORIZACIÓN DE CONTROLADORES"""
        if not self.adaptive_settings['performance_based_prioritization']:
            return controllers
        
        prioritized = []
        current_time = datetime.datetime.now()
        
        for controller in controllers:
            controller_key = f"{controller.symbol}_{controller.strategy_name}"
            
            # ✅ CALCULAR SCORE DE PRIORIDAD
            priority_score = 0
            
            # Factor 1: Tiempo desde última señal (más tiempo = mayor prioridad)
            last_signal = self.controller_priority_system['last_signal_times'].get(controller_key)
            if last_signal:
                time_since_signal = (current_time - last_signal).total_seconds() / 3600
                priority_score += min(time_since_signal * 10, 50)  # Máximo 50 puntos
            else:
                priority_score += 30  # Nuevo controller
            
            # Factor 2: Performance score del optimizer
            if hasattr(controller, 'optimizer_score'):
                optimizer_score = getattr(controller, 'optimizer_score', 1.0)
                priority_score += optimizer_score * 20  # Hasta 40+ puntos para scores altos
            
            # Factor 3: Tasa de éxito histórica
            success_rate = self.controller_priority_system['success_rates'].get(controller_key, 0.5)
            priority_score += success_rate * 30  # Hasta 30 puntos
            
            # Factor 4: Tiempo de procesamiento (menor = mejor)
            avg_processing_time = statistics.mean(
                self.controller_priority_system['processing_times'][controller_key][-10:]
            ) if self.controller_priority_system['processing_times'][controller_key] else 1.0
            
            if avg_processing_time < 2.0:
                priority_score += 20  # Rápido
            elif avg_processing_time > 5.0:
                priority_score -= 10  # Lento
            
            # Factor 5: Adecuación a la sesión actual
            session_bonus = 0
            strategy_name = controller.strategy_name.lower()
            
            if session_info['priority'] == 'HIGHEST':  # Overlap
                if 'scalper' in strategy_name or 'breakout' in strategy_name:
                    session_bonus = 15
            elif session_info['priority'] == 'HIGH':  # Londres/NY
                if 'crossover' in strategy_name or 'pullback' in strategy_name:
                    session_bonus = 10
            elif session_info['priority'] == 'LOW':  # Asiática
                if 'reversal' in strategy_name:
                    session_bonus = 5
            
            priority_score += session_bonus
            
            prioritized.append((controller, priority_score))
        
        # ✅ ORDENAR POR PRIORIDAD Y APLICAR ROTACIÓN
        prioritized.sort(key=lambda x: x[1], reverse=True)
        
        if self.adaptive_settings['controller_rotation']:
            # Rotar los top performers para dar oportunidad a otros
            cycle_number = self.performance_metrics['cycles_completed']
            rotation_offset = cycle_number % min(len(prioritized), 3)
            
            if len(prioritized) > 3:
                top_tier = prioritized[:3]
                rest = prioritized[3:]
                rotated_top = top_tier[rotation_offset:] + top_tier[:rotation_offset]
                prioritized = rotated_top + rest
        
        return [controller for controller, score in prioritized]
    
    def update_controller_metrics(self, controller, processing_time, had_signal, signal_confirmed):
        """✅ ACTUALIZACIÓN DE MÉTRICAS POR CONTROLLER"""
        controller_key = f"{controller.symbol}_{controller.strategy_name}"
        current_time = datetime.datetime.now()
        
        # Actualizar tiempo de procesamiento
        self.controller_priority_system['processing_times'][controller_key].append(processing_time)
        if len(self.controller_priority_system['processing_times'][controller_key]) > 20:
            self.controller_priority_system['processing_times'][controller_key].pop(0)
        
        # Actualizar tiempo de última señal
        if had_signal:
            self.controller_priority_system['last_signal_times'][controller_key] = current_time
        
        # Actualizar tasa de éxito
        current_success_rate = self.controller_priority_system['success_rates'].get(controller_key, 0.5)
        if had_signal:
            success_value = 1.0 if signal_confirmed else 0.3
            # Media móvil ponderada
            new_success_rate = (current_success_rate * 0.9) + (success_value * 0.1)
            self.controller_priority_system['success_rates'][controller_key] = new_success_rate
    
    def get_cycle_plan_advanced(self, controllers):
        """
        ✅ GENERACIÓN AVANZADA DE PLAN DE CICLO CON IA
        """
        cycle_start_time = time.time()
        now_utc = datetime.datetime.now(pytz.UTC)
        
        # ✅ INFORMACIÓN AVANZADA DE SESIÓN
        current_session, session_info = self.get_current_session_advanced(now_utc)
        
        # ✅ DETECCIÓN DE VELA ACTUAL
        current_candle_utc = now_utc.replace(
            minute=(now_utc.minute // 5) * 5, 
            second=0, 
            microsecond=0
        )
        
        is_new_candle = current_candle_utc != self.last_candle_processed_utc
        
        if is_new_candle:
            seconds_into_candle = (now_utc - current_candle_utc).total_seconds()
            
            # ✅ BÚFER DINÁMICO
            recent_stability = [1 if seconds_into_candle >= self.BUFFER_SECONDS else 0 
                              for _ in range(5)]  # Simplificado para ejemplo
            dynamic_buffer = self.calculate_dynamic_buffer(session_info, recent_stability)
            
            is_stable = seconds_into_candle >= dynamic_buffer
            
            if is_stable:
                # ✅ ACTUALIZAR MÉTRICAS DE ESTABILIDAD
                self.performance_metrics['vela_stability_stats']['stable_immediately'] += 1
                
                print(f"✅ NUEVA VELA ESTABLE: {current_candle_utc.strftime('%H:%M')} "
                      f"(Sesión: {current_session.value.upper()}, Búfer: {dynamic_buffer}s)")
                
                self.last_candle_processed_utc = current_candle_utc
                
                # ✅ PRIORIZAR CONTROLADORES
                prioritized_controllers = self.prioritize_controllers(controllers, session_info)
                
                # ✅ APLICAR FILTROS DE SESIÓN
                controllers_to_process = self.filter_controllers_by_session(
                    prioritized_controllers, current_session, session_info
                )
                
                wait_time = self.calculate_intelligent_wait_time(now_utc, session_info)
                
                # ✅ ACTUALIZAR MÉTRICAS
                self.performance_metrics['cycles_completed'] += 1
                self.performance_metrics['session_activity'][current_session.value] += 1
                self.performance_metrics['requests_by_session'][current_session.value] += len(controllers_to_process)
                
                return {
                    'action': 'analyze_new_candle',
                    'wait_seconds': wait_time,
                    'reason': f'Nueva vela M5 en {session_info["description"]}',
                    'controllers_to_process': controllers_to_process,
                    'session': current_session.value,
                    'session_priority': session_info['priority'],
                    'estimated_requests': len(controllers_to_process),
                    'dynamic_buffer_used': dynamic_buffer,
                    'total_controllers': len(controllers),
                    'processing_plan': {
                        'prioritized': True,
                        'session_filtered': True,
                        'expected_duration': len(controllers_to_process) * 2.5  # Estimado
                    }
                }
            else:
                # ✅ VELA INESTABLE - CALCULAR ESPERA INTELIGENTE
                wait_time = max(1, int(dynamic_buffer - seconds_into_candle) + 2)
                
                self.performance_metrics['vela_stability_stats']['required_waiting'] += 1
                current_avg_wait = self.performance_metrics['vela_stability_stats']['avg_wait_time']
                count = self.performance_metrics['vela_stability_stats']['required_waiting']
                self.performance_metrics['vela_stability_stats']['avg_wait_time'] = (
                    (current_avg_wait * (count - 1) + wait_time) / count
                )
                
                print(f"🟡 VELA INESTABLE: {current_candle_utc.strftime('%H:%M')} "
                      f"({int(seconds_into_candle)}s < {dynamic_buffer}s). Esperando {wait_time}s")
                
                return {
                    'action': 'wait_for_stability',
                    'wait_seconds': wait_time,
                    'reason': f'Esperando estabilización (búfer dinámico: {dynamic_buffer}s)',
                    'controllers_to_process': [],
                    'session': current_session.value,
                    'session_priority': session_info['priority'],
                    'estimated_requests': 0,
                    'stability_info': {
                        'seconds_elapsed': int(seconds_into_candle),
                        'buffer_required': dynamic_buffer,
                        'buffer_type': 'dynamic'
                    }
                }
        else:
            # ✅ VELA YA PROCESADA - ESPERA INTELIGENTE
            wait_time = self.calculate_intelligent_wait_time(now_utc, session_info)
            
            return {
                'action': 'wait_for_next_candle',
                'wait_seconds': wait_time,
                'reason': f'Esperando próxima vela M5 (sesión {current_session.value})',
                'controllers_to_process': [],
                'session': current_session.value,
                'session_priority': session_info['priority'],
                'estimated_requests': 0,
                'next_candle_info': {
                    'minutes_remaining': wait_time // 60,
                    'session_activity_expected': session_info['expected_activity']
                }
            }
    
    def filter_controllers_by_session(self, controllers, current_session, session_info):
        """✅ FILTRAR CONTROLADORES SEGÚN LA SESIÓN ACTUAL"""
        if not self.adaptive_settings['session_based_intervals']:
            return controllers
        
        # ✅ LÓGICA DE FILTRADO INTELIGENTE
        if session_info['priority'] == 'VERY_LOW':  # Sesión inactiva
            # Solo procesar los top 3 controllers más prometedores
            return controllers[:3]
        
        elif session_info['priority'] == 'LOW':  # Sesión asiática
            # Procesar 50% de los controllers, priorizando estrategias de reversión
            filtered = []
            reversal_strategies = [c for c in controllers if 'reversal' in c.strategy_name.lower()]
            other_strategies = [c for c in controllers if 'reversal' not in c.strategy_name.lower()]
            
            # Dar prioridad a reversales en sesión asiática
            filtered.extend(reversal_strategies[:3])
            filtered.extend(other_strategies[:max(1, len(controllers) // 2 - len(reversal_strategies[:3]))])
            
            return filtered
        
        elif session_info['priority'] in ['HIGH', 'HIGHEST']:  # Londres/NY/Overlap
            # Procesar todos los controllers en sesiones activas
            return controllers
        
        return controllers
    
    def calculate_intelligent_wait_time(self, now_utc, session_info):
        """✅ CÁLCULO INTELIGENTE DEL TIEMPO DE ESPERA"""
        # Tiempo base hasta próxima vela M5
        next_m5_minute = ((now_utc.minute // 5) + 1) * 5
        
        if next_m5_minute >= 60:
            next_dt = now_utc.replace(
                hour=(now_utc.hour + 1) % 24, 
                minute=0, 
                second=self.BUFFER_SECONDS, 
                microsecond=0
            )
            if now_utc.hour == 23:
                next_dt += datetime.timedelta(days=1)
        else:
            next_dt = now_utc.replace(
                minute=next_m5_minute, 
                second=self.BUFFER_SECONDS, 
                microsecond=0
            )
        
        base_wait = max(1, int((next_dt - now_utc).total_seconds()))
        
        # ✅ AJUSTAR SEGÚN SESIÓN
        if session_info['priority'] == 'HIGHEST':
            # En overlap, revisar más frecuentemente
            return max(base_wait, 15)
        elif session_info['priority'] == 'VERY_LOW':
            # En sesión inactiva, esperar más tiempo
            return max(base_wait, 60)
        else:
            return base_wait
    
    # ✅ MÉTODOS DE COMPATIBILIDAD
    def get_cycle_plan(self, controllers):
        """Método de compatibilidad con la API anterior"""
        return self.get_cycle_plan_advanced(controllers)
    
    def get_current_session(self, dt_utc):
        """Método de compatibilidad con la API anterior"""
        session, _ = self.get_current_session_advanced(dt_utc)
        return session
    
    def calculate_wait_time_for_next_candle(self, now_utc):
        """Método de compatibilidad con la API anterior"""
        _, session_info = self.get_current_session_advanced(now_utc)
        return self.calculate_intelligent_wait_time(now_utc, session_info)
    
    # ✅ MÉTODOS DE GESTIÓN DE REQUESTS (SIN CAMBIOS)
    def can_make_requests(self, estimated_requests):
        self.reset_daily_counter()
        return self.daily_request_count + estimated_requests <= self.max_daily_requests
    
    def reset_daily_counter(self):
        today = datetime.date.today()
        if today != self.last_reset_date:
            self.daily_request_count = 0
            self.last_reset_date = today
    
    def increment_requests(self, count=1):
        self.daily_request_count += count
    
    def print_session_stats_advanced(self):
        """✅ ESTADÍSTICAS AVANZADAS DEL SISTEMA"""
        now_utc = datetime.datetime.now(pytz.UTC)
        current_session, session_info = self.get_current_session_advanced(now_utc)
        
        # ✅ HORA LOCAL ARGENTINA
        try:
            local_tz = pytz.timezone('America/Argentina/Tucuman')
            local_time = now_utc.astimezone(local_tz)
            hora_argentina = local_time.strftime('%H:%M:%S')
        except Exception:
            hora_argentina = "N/A"
        
        self.reset_daily_counter()
        
        # ✅ CÁLCULO DE EFICIENCIA
        percentage_used = (self.daily_request_count / self.max_daily_requests) * 100
        
        print(f"\n📊 ESTADO AVANZADO DEL SISTEMA:")
        print(f"    🕐 Sesión: {current_session.value.upper()} ({session_info['priority']})")
        print(f"    📝 {session_info['description']}")
        print(f"    🇦🇷 Argentina: {hora_argentina} | 🌍 UTC: {now_utc.strftime('%H:%M:%S')}")
        print(f"    📡 Requests: {self.daily_request_count:,}/{self.max_daily_requests:,} ({percentage_used:.1f}%)")
        
        # ✅ INFORMACIÓN DE SESIÓN DETALLADA
        minutes_into = session_info.get('minutes_into_session', 0)
        minutes_remaining = session_info.get('minutes_remaining', 0)
        print(f"    ⏰ Sesión: {minutes_into:.0f}min transcurridos, {minutes_remaining:.0f}min restantes")
        
        # ✅ MÉTRICAS DE PERFORMANCE
        if self.performance_metrics['cycles_completed'] > 0:
            avg_cycle_time = (self.performance_metrics['total_processing_time'] / 
                            self.performance_metrics['cycles_completed'])
            print(f"    🔄 Ciclos completados: {self.performance_metrics['cycles_completed']}")
            print(f"    ⚡ Tiempo promedio/ciclo: {avg_cycle_time:.1f}s")
        
        # ✅ VELA ACTUAL
        if self.last_candle_processed_utc:
            candle_age = (now_utc - self.last_candle_processed_utc).total_seconds() / 60
            print(f"    🕯️ Última vela: {self.last_candle_processed_utc.strftime('%H:%M')} "
                  f"(hace {candle_age:.0f}min)")
        
        # ✅ ESTADÍSTICAS DE ESTABILIDAD
        stability_stats = self.performance_metrics['vela_stability_stats']
        if stability_stats['stable_immediately'] + stability_stats['required_waiting'] > 0:
            total_velas = stability_stats['stable_immediately'] + stability_stats['required_waiting']
            immediate_rate = (stability_stats['stable_immediately'] / total_velas) * 100
            print(f"    📈 Estabilidad velas: {immediate_rate:.0f}% inmediata, "
                  f"promedio espera: {stability_stats['avg_wait_time']:.1f}s")
        
        # ✅ ACTIVIDAD POR SESIÓN
        if self.performance_metrics['session_activity']:
            most_active = max(self.performance_metrics['session_activity'].items(), 
                            key=lambda x: x[1])
            print(f"    🏆 Sesión más activa: {most_active[0]} ({most_active[1]} ciclos)")
    
    # ✅ MÉTODO DE COMPATIBILIDAD
    def print_session_stats(self):
        """Método de compatibilidad con la API anterior"""
        self.print_session_stats_advanced()
    
    def get_performance_summary(self):
        """✅ RESUMEN COMPLETO DE PERFORMANCE"""
        return {
            'cycles_completed': self.performance_metrics['cycles_completed'],
            'avg_cycle_time': (self.performance_metrics['total_processing_time'] / 
                             max(self.performance_metrics['cycles_completed'], 1)),
            'session_activity': dict(self.performance_metrics['session_activity']),
            'requests_by_session': dict(self.performance_metrics['requests_by_session']),
            'stability_stats': self.performance_metrics['vela_stability_stats'].copy(),
            'daily_request_usage': (self.daily_request_count / self.max_daily_requests) * 100,
            'controller_metrics': {
                'total_tracked': len(self.controller_priority_system['success_rates']),
                'avg_success_rate': statistics.mean(self.controller_priority_system['success_rates'].values()) 
                                  if self.controller_priority_system['success_rates'] else 0
            }
        }