# check_pmi_activity.py - Verificar actividad PMI reciente
import json
import os
from datetime import datetime, timezone, timedelta

def analyze_pmi_log():
    log_file = "logs/pmi_decisions.jsonl"
    
    if not os.path.exists(log_file):
        print("❌ No existe logs/pmi_decisions.jsonl")
        return
    
    print(f"📊 Analizando {log_file}...")
    
    # Leer todas las líneas
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    print(f"📄 Total de entradas: {len(lines)}")
    
    # Analizar últimas 20 entradas
    recent_entries = []
    now = datetime.now(timezone.utc)
    one_hour_ago = now - timedelta(hours=1)
    
    for line in lines[-50:]:  # Últimas 50 líneas
        try:
            entry = json.loads(line.strip())
            
            # Convertir timestamp a datetime si existe
            ts_str = entry.get("_ts_utc", "")
            if ts_str:
                # Manejar diferentes formatos de timestamp
                try:
                    if ts_str.endswith("+00:00"):
                        ts = datetime.fromisoformat(ts_str)
                    else:
                        ts = datetime.fromisoformat(ts_str + "+00:00")
                    entry["_parsed_ts"] = ts
                except:
                    entry["_parsed_ts"] = None
            
            recent_entries.append(entry)
        except json.JSONDecodeError:
            continue
    
    print(f"\n📋 Últimas 10 entradas PMI:")
    print("-" * 80)
    
    for i, entry in enumerate(recent_entries[-10:], 1):
        ticket = entry.get("ticket", "?")
        action = entry.get("action", "?")
        reason = entry.get("reason", "?")
        confidence = entry.get("confidence", "?")
        close_score = entry.get("close_score", "?")
        symbol = entry.get("symbol", "?")
        ts = entry.get("_ts_utc", "?")
        
        # Tiempo relativo
        parsed_ts = entry.get("_parsed_ts")
        time_ago = ""
        if parsed_ts:
            delta = now - parsed_ts
            if delta.total_seconds() < 3600:  # Menos de 1 hora
                mins = int(delta.total_seconds() / 60)
                time_ago = f"(hace {mins}min)"
            elif delta.total_seconds() < 86400:  # Menos de 1 día
                hours = int(delta.total_seconds() / 3600)
                time_ago = f"(hace {hours}h)"
            else:
                days = int(delta.total_seconds() / 86400)
                time_ago = f"(hace {days}d)"
        
        print(f"{i:2d}. Ticket:{ticket} | {action} | {symbol} | conf:{confidence} | {reason} | {time_ago}")
    
    # Estadísticas por acción
    print(f"\n📈 Estadísticas de acciones (últimas 50 entradas):")
    actions = {}
    reasons = {}
    
    for entry in recent_entries:
        action = entry.get("action", "UNKNOWN")
        reason = entry.get("reason", "unknown")
        
        actions[action] = actions.get(action, 0) + 1
        reasons[reason] = reasons.get(reason, 0) + 1
    
    for action, count in sorted(actions.items()):
        print(f"  • {action}: {count}")
    
    print(f"\n📝 Razones más comunes:")
    for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  • {reason}: {count}")
    
    # Verificar entradas recientes (última hora)
    recent_count = 0
    for entry in recent_entries:
        parsed_ts = entry.get("_parsed_ts")
        if parsed_ts and parsed_ts > one_hour_ago:
            recent_count += 1
    
    print(f"\n⏰ Entradas en la última hora: {recent_count}")
    
    if recent_count == 0:
        print("⚠️  No hay entradas PMI en la última hora. Posibles causas:")
        print("   • Bot no está ejecutándose")
        print("   • No hay posiciones abiertas")
        print("   • PMI está en cooldown")
        print("   • Error en _pmi_integration_step")
    else:
        print("✅ PMI está activo y generando decisiones")

if __name__ == "__main__":
    analyze_pmi_log()
