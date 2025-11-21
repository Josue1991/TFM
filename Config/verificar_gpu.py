"""
verificar_gpu.py
Script para verificar si Windows detecta la GPU NVIDIA 780 y si los drivers funcionan.
"""
import subprocess
import sys
import os

print("\n" + "="*70)
print("VERIFICACIÓN DE GPU NVIDIA 780")
print("="*70)

# =====================================
# 1. VERIFICAR nvidia-smi
# =====================================
print("\n1. BUSCANDO nvidia-smi...")
print("-" * 70)

try:
    # Intentar ejecutar nvidia-smi
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
    
    if result.returncode == 0:
        print("✓ nvidia-smi encontrado. Salida:\n")
        print(result.stdout)
        print("✓ DRIVERS NVIDIA FUNCIONANDO CORRECTAMENTE")
    else:
        print("✗ nvidia-smi retornó error:")
        print(result.stderr)
except FileNotFoundError:
    print("✗ nvidia-smi NO encontrado en el PATH")
    print("\nPosibles causas:")
    print("  - Los drivers NVIDIA no están instalados")
    print("  - nvidia-smi no está en el PATH del sistema")
    print("\nSolución:")
    print("  1. Descarga drivers desde: https://www.nvidia.com/Download/driverDetails.aspx")
    print("  2. Busca por GPU 'GeForce GTX 780' y SO 'Windows'")
    print("  3. Instala los drivers y reinicia")
except subprocess.TimeoutExpired:
    print("✗ nvidia-smi tomó demasiado tiempo")
except Exception as e:
    print(f"✗ Error ejecutando nvidia-smi: {e}")

# =====================================
# 2. VERIFICAR VARIABLES DE ENTORNO
# =====================================
print("\n2. VERIFICANDO VARIABLES DE ENTORNO")
print("-" * 70)

env_vars = {
    'CUDA_HOME': 'Ruta de CUDA',
    'CUDA_PATH': 'Ruta de CUDA (alternativa)',
    'CUDNN_PATH': 'Ruta de cuDNN',
    'PATH': 'Variable PATH (buscar CUDA)',
}

for var, desc in env_vars.items():
    value = os.environ.get(var)
    if value:
        print(f"✓ {var}: {value[:80]}...")
    else:
        if var != 'CUDNN_PATH':
            print(f"✗ {var}: NO DEFINIDO ({desc})")

# =====================================
# 3. VERIFICAR CARPETAS DE CUDA
# =====================================
print("\n3. BUSCANDO CARPETAS DE CUDA")
print("-" * 70)

cuda_paths = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
    r"C:\Program Files\NVIDIA\CUDA",
    r"C:\CUDA",
]

found_cuda = False
for path in cuda_paths:
    if os.path.exists(path):
        print(f"✓ Encontrada carpeta CUDA: {path}")
        # Listar versiones
        if os.path.isdir(path):
            versions = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            if versions:
                print(f"  Versiones disponibles: {versions}")
            found_cuda = True

if not found_cuda:
    print("✗ No se encontraron carpetas de CUDA")
    print("  CUDA no está instalado. Se recomienda:")
    print("  1. Descargar CUDA 11.8 desde: https://developer.nvidia.com/cuda-11-8-0-download-archive")
    print("  2. Seleccionar Windows, x86_64, Windows 11, installer local")
    print("  3. Instalar y reiniciar la computadora")

# =====================================
# 4. RESUMEN Y RECOMENDACIONES
# =====================================
print("\n4. RESUMEN Y PRÓXIMOS PASOS")
print("-" * 70)

print("""
CHECKLIST:
  [ ] ¿nvidia-smi se ejecutó correctamente?
  [ ] ¿Muestra tu GPU GTX 780?
  [ ] ¿CUDA_HOME está definido?
  
SI TODAS LAS RESPUESTAS SON 'SÍ':
  → La GPU está detectada correctamente
  → Instala: pip install --upgrade tensorflow==2.10.0
  
SI NVIDIA-SMI NO SE ENCONTRÓ:
  → Instala drivers NVIDIA
  → Descarga desde: https://www.nvidia.com/Download/driverDetails.aspx
  → Reinicia la computadora después de instalar
  
SI CUDA_HOME NO ESTÁ DEFINIDO:
  → Instala CUDA 11.8
  → Descarga desde: https://developer.nvidia.com/cuda-11-8-0-download-archive
  → Reinicia después de instalar
  
ORDEN RECOMENDADO:
  1. Actualiza drivers NVIDIA (si es necesario)
  2. Instala CUDA 11.8
  3. Instala cuDNN 8.x para CUDA 11.x
  4. Reinstala TensorFlow: pip install --upgrade tensorflow==2.10.0
  5. Prueba nuevamente este script
""")

print("\n" + "="*70)
print("FIN DE LA VERIFICACIÓN")
print("="*70 + "\n")
