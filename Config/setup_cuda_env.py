"""
setup_cuda_env.py
Script para configurar CUDA_HOME y preparar el ambiente para GPU.
"""
import os
import sys
import subprocess

print("\n" + "="*70)
print("CONFIGURACIÓN DE CUDA ENVIRONMENT")
print("="*70)

# =====================================
# 1. CONFIGURAR CUDA_HOME
# =====================================
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"

print("\n1. CONFIGURANDO CUDA_HOME")
print("-" * 70)

if os.path.exists(cuda_path):
    print(f"✓ Ruta CUDA encontrada: {cuda_path}")
    
    # Configurar en la sesión actual
    os.environ['CUDA_HOME'] = cuda_path
    print(f"✓ CUDA_HOME configurado en la sesión actual")
    
    # Configurar en el sistema (requiere privilegios de admin)
    try:
        result = subprocess.run(
            f'setx CUDA_HOME "{cuda_path}"',
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✓ CUDA_HOME configurado PERMANENTEMENTE en el sistema")
        else:
            print(f"⚠ Se necesitan permisos de administrador para configurar permanentemente")
            print(f"  Ejecuta PowerShell como administrador y corre:")
            print(f'  [Environment]::SetEnvironmentVariable("CUDA_HOME", "{cuda_path}", "Machine")')
    except Exception as e:
        print(f"Nota: {e}")
else:
    print(f"✗ No se encontró CUDA en: {cuda_path}")
    sys.exit(1)

# =====================================
# 2. VERIFICAR NVIDIA-SMI
# =====================================
print("\n2. VERIFICANDO GPU")
print("-" * 70)

try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'], 
                          capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        gpu_info = result.stdout.strip()
        print(f"✓ GPU detectada: {gpu_info}")
    else:
        print(f"✗ Error al consultar GPU: {result.stderr}")
except Exception as e:
    print(f"✗ Error: {e}")

# =====================================
# 3. INSTALAR TENSORFLOW 2.10
# =====================================
print("\n3. PREPARANDO INSTALACIÓN DE TENSORFLOW 2.10")
print("-" * 70)

print("\nEjecutarás los siguientes comandos:")
print("\n1. Desinstalar versiones actuales:")
print("   pip uninstall tensorflow tensorflow-gpu -y")
print("\n2. Instalar TensorFlow 2.10 GPU:")
print("   pip install tensorflow==2.10.0")
print("\n3. Verificar instalación:")
print("   python gpu_diagnostico.py")

print("\n" + "="*70)
print("CONFIGURACIÓN COMPLETADA")
print("="*70)
print("\nPróximo paso: Abre una NUEVA ventana de PowerShell/CMD y ejecuta:")
print("   pip uninstall tensorflow tensorflow-gpu -y")
print("   pip install tensorflow==2.10.0")
print("\n(Necesitas una ventana nueva para que se cargue CUDA_HOME)")
print("="*70 + "\n")
