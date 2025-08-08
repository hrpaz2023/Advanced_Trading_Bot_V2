# test_pmi_logging.py - VERSIÓN CORREGIDA
import sys
import os
sys.path.insert(0, os.getcwd())

try:
    from pmi.logger import log_pmi_decision
    from pmi.decision import PMIDecision, DecisionAction
    
    print("✅ Imports PMI exitosos")
    
    # Verificar la estructura de PMIDecision
    print("🔍 Inspeccionando PMIDecision...")
    import inspect
    sig = inspect.signature(PMIDecision.__init__)
    print(f"Parámetros de PMIDecision.__init__: {sig}")
    
    # Crear decisión de prueba con TODOS los parámetros requeridos
    test_decision = PMIDecision(
        ticket=123456,
        action=DecisionAction.HOLD,
        confidence=0.85,  # ✅ AGREGADO: parámetro faltante
        reason="test_manual",
        close_score=0.3,
        telemetry={"manual_test": True}
    )
    
    print("✅ Decisión de prueba creada exitosamente")
    print(f"Decisión: ticket={test_decision.ticket}, action={test_decision.action}, confidence={test_decision.confidence}")
    
    # Intentar guardar
    print("🔄 Intentando guardar con log_pmi_decision...")
    result = log_pmi_decision(test_decision)
    print(f"✅ log_pmi_decision retornó: {result}")
    
    # Verificar archivo
    log_file = "logs/pmi_decisions.jsonl"
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"✅ Archivo {log_file} tiene {len(lines)} líneas")
        if lines:
            print(f"📄 Última línea: {lines[-1].strip()}")
    else:
        print(f"❌ Archivo {log_file} NO existe")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# EXTRA: Verificar estructura de DecisionAction
try:
    print("\n🔍 Valores disponibles en DecisionAction:")
    for attr in dir(DecisionAction):
        if not attr.startswith('_'):
            print(f"  - {attr} = {getattr(DecisionAction, attr)}")
except Exception as e:
    print(f"❌ Error inspeccionando DecisionAction: {e}")

# EXTRA: Verificar permisos de escritura
print("\n🔍 Verificando permisos de escritura...")
try:
    os.makedirs("logs", exist_ok=True)
    test_file = "logs/test_write.tmp"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("test de escritura")
    os.remove(test_file)
    print("✅ Permisos de escritura OK")
except Exception as e:
    print(f"❌ Error de permisos: {e}")