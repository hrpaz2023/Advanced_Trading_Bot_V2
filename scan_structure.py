#!/usr/bin/env python3
"""
Script para Escanear y Documentar la Estructura Completa del Proyecto
Genera un reporte detallado de todos los archivos y directorios
"""

import os
import json
from pathlib import Path
from datetime import datetime

def scan_project_structure(root_path="."):
    """
    Escanea la estructura completa del proyecto
    """
    
    structure = {
        "scan_info": {
            "timestamp": datetime.now().isoformat(),
            "root_path": os.path.abspath(root_path),
            "total_files": 0,
            "total_dirs": 0
        },
        "directories": {},
        "python_files": [],
        "config_files": [],
        "important_files": [],
        "file_locations": {}
    }
    
    # Archivos importantes a buscar específicamente
    important_files = [
        "main_bot.py",
        "execution_controller.py", 
        "trading_client.py",
        "daily_risk_manager.py",
        "ftmo_bot_config.py",
        "notifier.py",
        "risk_manager.py",
        "analyze_results_ftmo.py",
        "run_optimization.py",
        "run_pipeline.py"
    ]
    
    # Extensiones de archivos de configuración
    config_extensions = [".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"]
    
    print("🔍 Escaneando estructura del proyecto...")
    print("=" * 60)
    
    for root, dirs, files in os.walk(root_path):
        # Ignorar directorios comunes que no son relevantes
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]
        
        rel_path = os.path.relpath(root, root_path)
        if rel_path == ".":
            rel_path = "ROOT"
        
        structure["directories"][rel_path] = {
            "files": [],
            "subdirs": dirs.copy(),
            "file_count": len(files)
        }
        
        structure["scan_info"]["total_dirs"] += 1
        
        for file in files:
            if file.startswith('.'):
                continue
                
            file_path = os.path.join(root, file)
            rel_file_path = os.path.relpath(file_path, root_path)
            
            structure["scan_info"]["total_files"] += 1
            structure["directories"][rel_path]["files"].append(file)
            
            # Clasificar archivos
            if file.endswith('.py'):
                structure["python_files"].append(rel_file_path)
            
            # Buscar archivos de configuración
            if any(file.endswith(ext) for ext in config_extensions):
                structure["config_files"].append(rel_file_path)
            
            # Buscar archivos importantes
            if file in important_files:
                structure["important_files"].append(rel_file_path)
                structure["file_locations"][file] = rel_file_path
    
    return structure

def print_structure_tree(structure, max_depth=4):
    """
    Imprime estructura en formato árbol
    """
    print("\n🌳 ESTRUCTURA DEL PROYECTO (ÁRBOL)")
    print("=" * 60)
    
    def print_directory(path, level=0, max_level=max_depth):
        if level > max_level:
            return
            
        indent = "│   " * level
        
        if path == "ROOT":
            print("📁 Advanced_Trading_Bot/")
            dir_info = structure["directories"]["ROOT"]
        else:
            dir_name = os.path.basename(path)
            print(f"{indent}├── 📁 {dir_name}/")
            dir_info = structure["directories"].get(path, {})
        
        # Mostrar subdirectorios
        subdirs = dir_info.get("subdirs", [])
        for subdir in sorted(subdirs):
            if path == "ROOT":
                subdir_path = subdir
            else:
                subdir_path = os.path.join(path, subdir).replace("\\", "/")
            print_directory(subdir_path, level + 1, max_level)
        
        # Mostrar archivos importantes
        files = dir_info.get("files", [])
        important_in_dir = [f for f in files if f in [
            "main_bot.py", "execution_controller.py", "trading_client.py",
            "daily_risk_manager.py", "ftmo_bot_config.py", "notifier.py",
            "analyze_results_ftmo.py", "run_optimization.py"
        ]]
        
        for file in sorted(important_in_dir):
            file_indent = "│   " * (level + 1)
            print(f"{file_indent}├── 📄 {file} ⭐")
        
        # Mostrar algunos archivos Python adicionales
        python_files = [f for f in files if f.endswith('.py') and f not in important_in_dir]
        if python_files and level < 2:  # Solo mostrar en niveles superiores
            for file in sorted(python_files[:3]):  # Máximo 3 archivos
                file_indent = "│   " * (level + 1)
                print(f"{file_indent}├── 📄 {file}")
            if len(python_files) > 3:
                file_indent = "│   " * (level + 1)
                print(f"{file_indent}├── 📄 ... (+{len(python_files)-3} más)")
    
    print_directory("ROOT")

def print_important_files(structure):
    """
    Imprime ubicaciones de archivos importantes
    """
    print("\n⭐ ARCHIVOS IMPORTANTES Y SUS UBICACIONES")
    print("=" * 60)
    
    file_locations = structure["file_locations"]
    
    if not file_locations:
        print("❌ No se encontraron archivos importantes")
        return
    
    for file, location in sorted(file_locations.items()):
        print(f"📄 {file:<25} → {location}")
    
    print(f"\n✅ Total archivos importantes encontrados: {len(file_locations)}")

def print_imports_analysis(structure):
    """
    Analiza y sugiere importaciones basadas en la estructura
    """
    print("\n🔧 ANÁLISIS DE IMPORTACIONES SUGERIDAS")
    print("=" * 60)
    
    file_locations = structure["file_locations"]
    
    main_bot_location = file_locations.get("main_bot.py")
    if not main_bot_location:
        print("❌ main_bot.py no encontrado")
        return
    
    main_bot_dir = os.path.dirname(main_bot_location)
    print(f"📍 main_bot.py ubicado en: {main_bot_location}")
    print(f"📁 Directorio: {main_bot_dir}")
    
    print(f"\n🔗 IMPORTACIONES SUGERIDAS PARA main_bot.py:")
    print("-" * 40)
    
    # Analizar ubicaciones relativas
    for file, location in file_locations.items():
        if file == "main_bot.py":
            continue
            
        file_dir = os.path.dirname(location)
        
        # Calcular ruta relativa
        if main_bot_dir == file_dir:
            # Mismo directorio
            import_path = f"from .{file[:-3]} import ..."
        elif main_bot_dir.startswith("src/") and file_dir.startswith("src/"):
            # Ambos en src/, calcular ruta relativa
            main_parts = main_bot_dir.split("/")
            file_parts = file_dir.split("/")
            
            # Subir niveles necesarios
            up_levels = len(main_parts) - 1  # -1 porque src/ es el nivel base
            relative_parts = file_parts[1:]  # Omitir 'src'
            
            dots = ".." + "." * (up_levels - 1) if up_levels > 1 else ".."
            import_path = f"from {dots}.{'.'.join(relative_parts)}.{file[:-3]} import ..."
        else:
            # Usar importación absoluta
            import_path = f"from {file_dir.replace('/', '.')}.{file[:-3]} import ..."
        
        print(f"📄 {file:<25} → {import_path}")

def generate_summary(structure):
    """
    Genera resumen ejecutivo
    """
    print("\n📊 RESUMEN EJECUTIVO")
    print("=" * 60)
    
    scan_info = structure["scan_info"]
    
    print(f"📁 Directorios totales: {scan_info['total_dirs']}")
    print(f"📄 Archivos totales: {scan_info['total_files']}")
    print(f"🐍 Archivos Python: {len(structure['python_files'])}")
    print(f"⚙️ Archivos config: {len(structure['config_files'])}")
    print(f"⭐ Archivos importantes: {len(structure['important_files'])}")
    
    print(f"\n📍 Ubicación raíz: {scan_info['root_path']}")
    print(f"🕐 Escaneado: {scan_info['timestamp']}")

def main():
    """
    Función principal
    """
    print("🚀 ESCÁNER DE ESTRUCTURA DE PROYECTO")
    print("=" * 60)
    print("Analizando Advanced_Trading_Bot...")
    
    # Escanear estructura
    structure = scan_project_structure()
    
    # Mostrar resultados
    print_structure_tree(structure)
    print_important_files(structure)
    print_imports_analysis(structure)
    generate_summary(structure)
    
    # Guardar reporte en archivo
    output_file = "project_structure_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(structure, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Reporte guardado en: {output_file}")
    print("\n✅ Escaneo completado!")
    
    # Instrucciones finales
    print("\n" + "=" * 60)
    print("📋 PRÓXIMOS PASOS:")
    print("1. Revisar las ubicaciones de archivos importantes")
    print("2. Usar las importaciones sugeridas en main_bot.py")
    print("3. Mover ftmo_bot_config.py según análisis")
    print("4. Compartir este reporte para correcciones exactas")

if __name__ == "__main__":
    main()
