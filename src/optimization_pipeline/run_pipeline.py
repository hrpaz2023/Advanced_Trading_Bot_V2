# run_pipeline.py (Versión Optimizada con Integración IA)
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# ✅ NUEVAS IMPORTACIONES PARA INTEGRACIÓN
try:
    from src.model_training.train_model import train_model
    from src.data_preparation.populate_chroma import populate_database
    ML_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Módulos ML no disponibles: {e}")
    ML_MODULES_AVAILABLE = False

# ✅ INTEGRACIÓN CON ULTIMATE ANALYZER
try:
    from ultimate_trade_analyzer import UltimateTradeAnalyzer
    ULTIMATE_ANALYZER_AVAILABLE = True
except ImportError:
    print("⚠️ Ultimate Analyzer no disponible - modo básico")
    ULTIMATE_ANALYZER_AVAILABLE = False

class OptimizedPipeline:
    def __init__(self):
        self.config = None
        self.valid_combinations = []
        self.results = {
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'total_processing_time': 0,
            'combinations_processed': []
        }
        self.start_time = None
        
    def validate_prerequisites(self):
        """✅ VALIDACIÓN MEJORADA con verificaciones adicionales"""
        print("🔍 Validando prerequisitos del pipeline optimizado...")
        
        issues = []
        warnings = []
        
        # 1. Archivo de configuración principal
        config_path = 'configs/optimized_parameters.json'
        if not os.path.exists(config_path):
            issues.append("❌ Falta 'configs/optimized_parameters.json'")
            issues.append("   👉 Ejecuta: python run_optimization.py -> python direct_config_generator.py")
        else:
            # Verificar estructura del archivo
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Contar estrategias válidas
                strategy_count = 0
                for symbol, strategies in config.items():
                    if symbol == 'generation_info':
                        continue
                    if isinstance(strategies, dict):
                        strategy_count += len(strategies)
                
                if strategy_count == 0:
                    issues.append("❌ No se encontraron estrategias válidas en el archivo de configuración")
                else:
                    print(f"✅ Configuración válida: {strategy_count} estrategias encontradas")
            except json.JSONDecodeError:
                issues.append("❌ Archivo de configuración corrupto")
        
        # 2. Módulos ML
        if not ML_MODULES_AVAILABLE:
            warnings.append("⚠️ Módulos ML no disponibles - funcionalidad limitada")
        else:
            print("✅ Módulos ML disponibles")
        
        # 3. Ultimate Analyzer
        if not ULTIMATE_ANALYZER_AVAILABLE:
            warnings.append("⚠️ Ultimate Analyzer no disponible - análisis básico")
        else:
            print("✅ Ultimate Analyzer disponible para análisis avanzado")
        
        # 4. Directorios necesarios
        required_dirs = ['src', 'configs', 'data/features']
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                issues.append(f"❌ Falta directorio: {dir_path}")
        
        # 5. Verificar datos de features
        features_dir = Path('data/features')
        if features_dir.exists():
            feature_files = list(features_dir.glob('*_features.parquet'))
            if len(feature_files) == 0:
                issues.append("❌ No se encontraron archivos de features")
                issues.append("   👉 Ejecuta el generador de features primero")
            else:
                print(f"✅ {len(feature_files)} archivos de features encontrados")
        
        # Mostrar warnings
        if warnings:
            print("\n⚠️ ADVERTENCIAS:")
            for warning in warnings:
                print(f"  {warning}")
        
        # Mostrar errores críticos
        if issues:
            print("\n🚨 PROBLEMAS CRÍTICOS:")
            for issue in issues:
                print(f"  {issue}")
            return False
        
        print("✅ Todos los prerequisitos están listos")
        return True
    
    def load_configuration(self):
        """✅ CARGA DE CONFIGURACIÓN OPTIMIZADA"""
        try:
            with open('configs/optimized_parameters.json', 'r') as f:
                self.config = json.load(f)
            
            # Filtrar y validar combinaciones
            for symbol, strategies in self.config.items():
                if symbol == 'generation_info':
                    continue
                
                if not isinstance(strategies, dict):
                    continue
                
                for strategy_name, data in strategies.items():
                    if not isinstance(data, dict) or 'params' not in data:
                        print(f"⚠️ Datos incompletos para {symbol}_{strategy_name}")
                        continue
                    
                    # ✅ PRIORIZACIÓN por score
                    score = data.get('optimizer_score', 0)
                    
                    self.valid_combinations.append({
                        'symbol': symbol,
                        'strategy': strategy_name,
                        'data': data,
                        'score': score,
                        'priority': 'HIGH' if score >= 2.0 else 'MEDIUM' if score >= 1.5 else 'LOW'
                    })
            
            # ✅ ORDENAR por prioridad y score
            self.valid_combinations.sort(key=lambda x: (-x['score'], x['priority']))
            
            print(f"📊 Configuración cargada: {len(self.valid_combinations)} combinaciones válidas")
            
            # Mostrar distribución por prioridad
            priority_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            for combo in self.valid_combinations:
                priority_counts[combo['priority']] += 1
            
            print(f"   🔥 Alta prioridad: {priority_counts['HIGH']}")
            print(f"   ⚡ Media prioridad: {priority_counts['MEDIUM']}")
            print(f"   📈 Baja prioridad: {priority_counts['LOW']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando configuración: {e}")
            return False
    
    def process_combination_optimized(self, combination):
        """✅ PROCESAMIENTO OPTIMIZADO DE COMBINACIÓN INDIVIDUAL"""
        symbol = combination['symbol']
        strategy = combination['strategy']
        data = combination['data']
        score = combination['score']
        priority = combination['priority']
        
        start_time = time.time()
        
        print(f"\n🔧 [{priority}] Procesando: {symbol}_{strategy} (Score: {score:.3f})")
        
        results = {
            'symbol': symbol,
            'strategy': strategy,
            'score': score,
            'priority': priority,
            'tasks_completed': [],
            'tasks_failed': [],
            'processing_time': 0,
            'status': 'PROCESSING'
        }
        
        # ✅ TAREA 1: Entrenamiento ML (si está disponible)
        if ML_MODULES_AVAILABLE:
            try:
                print(f"    🤖 Entrenando modelo ML...")
                train_model(symbol, strategy)
                results['tasks_completed'].append('ML_TRAINING')
                print(f"    ✅ Modelo ML entrenado")
            except Exception as e:
                print(f"    ❌ Error en ML training: {e}")
                results['tasks_failed'].append(f'ML_TRAINING: {str(e)}')
        
        # ✅ TAREA 2: Población de ChromaDB (si está disponible)
        if ML_MODULES_AVAILABLE:
            try:
                print(f"    🗄️ Poblando ChromaDB...")
                populate_database(symbol, strategy)
                results['tasks_completed'].append('CHROMA_DB')
                print(f"    ✅ ChromaDB poblado")
            except Exception as e:
                print(f"    ❌ Error en ChromaDB: {e}")
                results['tasks_failed'].append(f'CHROMA_DB: {str(e)}')
        
        # ✅ TAREA 3: Validación con Ultimate Analyzer (si está disponible)
        if ULTIMATE_ANALYZER_AVAILABLE and len(results['tasks_completed']) > 0:
            try:
                print(f"    🧠 Ejecutando análisis avanzado...")
                analyzer = UltimateTradeAnalyzer()
                
                # Ejecutar análisis específico para esta combinación
                backtest_results = analyzer.run_backtest_with_details(symbol, strategy)
                if backtest_results:
                    analysis = analyzer.analyze_trade_patterns(backtest_results)
                    if analysis and analysis.get('total_trades', 0) > 0:
                        results['tasks_completed'].append('ULTIMATE_ANALYSIS')
                        results['analysis_summary'] = {
                            'total_trades': analysis['total_trades'],
                            'win_rate': analysis['win_rate'],
                            'profit_factor': analysis['profit_factor']
                        }
                        print(f"    ✅ Análisis completado: {analysis['total_trades']} trades, {analysis['win_rate']:.1f}% WR")
                    else:
                        results['tasks_failed'].append('ULTIMATE_ANALYSIS: No trades generated')
                else:
                    results['tasks_failed'].append('ULTIMATE_ANALYSIS: Backtest failed')
            except Exception as e:
                print(f"    ❌ Error en Ultimate Analysis: {e}")
                results['tasks_failed'].append(f'ULTIMATE_ANALYSIS: {str(e)}')
        
        # Calcular resultado final
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        
        total_tasks = len(results['tasks_completed']) + len(results['tasks_failed'])
        success_rate = len(results['tasks_completed']) / max(total_tasks, 1) * 100
        
        if len(results['tasks_completed']) == 0:
            results['status'] = 'FAILED'
            status_icon = "❌"
        elif len(results['tasks_failed']) == 0:
            results['status'] = 'SUCCESS'
            status_icon = "✅"
        else:
            results['status'] = 'PARTIAL'
            status_icon = "⚠️"
        
        print(f"    {status_icon} Completado en {processing_time:.1f}s - Éxito: {success_rate:.0f}%")
        return results
    
    def run_parallel_processing(self, max_workers=3):
        """✅ PROCESAMIENTO PARALELO OPTIMIZADO"""
        if not self.valid_combinations:
            print("❌ No hay combinaciones para procesar")
            return False
        
        print(f"\n🚀 Iniciando procesamiento paralelo con {max_workers} workers")
        print(f"📊 Total combinaciones: {len(self.valid_combinations)}")
        
        self.start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Enviar tareas
            future_to_combination = {
                executor.submit(self.process_combination_optimized, combo): combo
                for combo in self.valid_combinations
            }
            
            # Procesar resultados
            for future in as_completed(future_to_combination):
                combination = future_to_combination[future]
                
                try:
                    result = future.result(timeout=300)  # 5 minutos timeout
                    
                    self.results['combinations_processed'].append(result)
                    
                    if result['status'] == 'SUCCESS':
                        self.results['successful'] += 1
                    elif result['status'] == 'FAILED':
                        self.results['failed'] += 1
                    else:
                        self.results['skipped'] += 1
                    
                    # Mostrar progreso
                    completed = len(self.results['combinations_processed'])
                    total = len(self.valid_combinations)
                    progress = completed / total * 100
                    
                    print(f"📊 Progreso: {completed}/{total} ({progress:.1f}%) - {result['status']}")
                    
                except Exception as e:
                    print(f"❌ Error procesando {combination['symbol']}_{combination['strategy']}: {e}")
                    self.results['failed'] += 1
        
        self.results['total_processing_time'] = time.time() - self.start_time
        return True
    
    def generate_comprehensive_report(self):
        """✅ REPORTE COMPREHENSIVO"""
        print(f"\n" + "=" * 70)
        print("📊 REPORTE FINAL DEL PIPELINE OPTIMIZADO")
        print("=" * 70)
        
        # Estadísticas generales
        total = len(self.results['combinations_processed'])
        successful = self.results['successful']
        failed = self.results['failed']
        partial = self.results['skipped']
        
        print(f"⏱️ Tiempo total: {self.results['total_processing_time']:.1f} segundos")
        print(f"✅ Exitosos: {successful}")
        print(f"⚠️ Parciales: {partial}")
        print(f"❌ Fallidos: {failed}")
        print(f"📈 Tasa de éxito: {successful/max(total,1)*100:.1f}%")
        
        # Análisis por prioridad
        priority_stats = {'HIGH': {'success': 0, 'total': 0}, 'MEDIUM': {'success': 0, 'total': 0}, 'LOW': {'success': 0, 'total': 0}}
        
        for result in self.results['combinations_processed']:
            priority = result['priority']
            priority_stats[priority]['total'] += 1
            if result['status'] == 'SUCCESS':
                priority_stats[priority]['success'] += 1
        
        print(f"\n📊 ANÁLISIS POR PRIORIDAD:")
        for priority, stats in priority_stats.items():
            if stats['total'] > 0:
                success_rate = stats['success'] / stats['total'] * 100
                print(f"   {priority}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
        
        # Top performers
        successful_combinations = [r for r in self.results['combinations_processed'] if r['status'] == 'SUCCESS']
        if successful_combinations:
            print(f"\n🏆 TOP PERFORMERS:")
            sorted_successful = sorted(successful_combinations, key=lambda x: x['score'], reverse=True)
            
            for i, result in enumerate(sorted_successful[:5], 1):
                symbol = result['symbol']
                strategy = result['strategy']
                score = result['score']
                tasks = len(result['tasks_completed'])
                
                analysis_info = ""
                if 'analysis_summary' in result:
                    summary = result['analysis_summary']
                    analysis_info = f" | {summary['total_trades']} trades, {summary['win_rate']:.1f}% WR"
                
                print(f"   {i}. {symbol}_{strategy}: Score {score:.3f} | {tasks} tareas{analysis_info}")
        
        # Tareas más problemáticas
        all_failed_tasks = []
        for result in self.results['combinations_processed']:
            all_failed_tasks.extend(result['tasks_failed'])
        
        if all_failed_tasks:
            from collections import Counter
            task_failures = Counter([task.split(':')[0] for task in all_failed_tasks])
            
            print(f"\n❌ TAREAS MÁS PROBLEMÁTICAS:")
            for task, count in task_failures.most_common(3):
                print(f"   {task}: {count} fallos")
        
        # Guardar reporte detallado
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'pipeline_version': 'optimized_v1.0',
            'summary': {
                'total_combinations': total,
                'successful': successful,
                'partial': partial,
                'failed': failed,
                'success_rate': successful/max(total,1)*100,
                'total_processing_time': self.results['total_processing_time']
            },
            'priority_breakdown': priority_stats,
            'detailed_results': self.results['combinations_processed']
        }
        
        os.makedirs('reports', exist_ok=True)
        report_path = f"reports/pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Reporte detallado guardado: {report_path}")
        
        # Recomendaciones
        print(f"\n💡 RECOMENDACIONES:")
        if successful >= total * 0.8:
            print("   🎉 Excelente! La mayoría de combinaciones fueron exitosas")
        elif successful >= total * 0.5:
            print("   👍 Buen rendimiento, considera revisar las fallas")
        else:
            print("   ⚠️ Muchas fallas detectadas, revisa la configuración")
        
        if priority_stats['HIGH']['total'] > 0:
            high_success_rate = priority_stats['HIGH']['success'] / priority_stats['HIGH']['total']
            if high_success_rate < 0.8:
                print("   🔥 Prioriza arreglar las estrategias de ALTA prioridad")
        
        return report_path

def main():
    """✅ FUNCIÓN PRINCIPAL OPTIMIZADA"""
    print("🚀 PIPELINE OPTIMIZADO DE CONSTRUCCIÓN ML")
    print("Incluye: Entrenamiento ML + ChromaDB + Análisis IA")
    print("=" * 60)
    
    pipeline = OptimizedPipeline()
    
    # Validación
    if not pipeline.validate_prerequisites():
        print("\n🚨 Corrige los problemas antes de continuar.")
        sys.exit(1)
    
    # Cargar configuración
    if not pipeline.load_configuration():
        print("\n❌ Error cargando configuración.")
        sys.exit(1)
    
    # Mostrar plan de ejecución
    print(f"\n📋 PLAN DE EJECUCIÓN:")
    print(f"   🔧 Combinaciones a procesar: {len(pipeline.valid_combinations)}")
    print(f"   🤖 ML Training: {'✅' if ML_MODULES_AVAILABLE else '❌'}")
    print(f"   🗄️ ChromaDB: {'✅' if ML_MODULES_AVAILABLE else '❌'}")
    print(f"   🧠 Ultimate Analysis: {'✅' if ULTIMATE_ANALYZER_AVAILABLE else '❌'}")
    
    # Confirmar ejecución
    if len(pipeline.valid_combinations) > 10:
        response = input(f"\n⚠️ Se procesarán {len(pipeline.valid_combinations)} combinaciones. ¿Continuar? (y/N): ")
        if response.lower() != 'y':
            print("❌ Operación cancelada")
            sys.exit(0)
    
    # Ejecutar pipeline
    print(f"\n🚀 Iniciando pipeline optimizado...")
    
    success = pipeline.run_parallel_processing(max_workers=3)
    
    if success:
        report_path = pipeline.generate_comprehensive_report()
        
        if pipeline.results['successful'] > 0:
            print(f"\n🎉 Pipeline completado exitosamente!")
            print(f"📊 {pipeline.results['successful']} combinaciones procesadas correctamente")
            print(f"📁 Reporte: {report_path}")
        else:
            print(f"\n❌ Pipeline falló - ninguna combinación exitosa")
            sys.exit(1)
    else:
        print(f"\n❌ Error ejecutando pipeline")
        sys.exit(1)

if __name__ == '__main__':
    main()