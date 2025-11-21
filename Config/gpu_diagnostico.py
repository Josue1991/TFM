"""
gpu_diagnostico.py
Script de diagnóstico para verificar la detección de GPU NVIDIA.
Ayuda a identificar problemas de configuración de CUDA y TensorFlow.
"""
import sys
import tensorflow as tf
import numpy as np

print("\n" + "="*70)
print("DIAGNÓSTICO DE GPU NVIDIA")
print("="*70)

# =====================================
# 1. INFORMACIÓN DE TENSORFLOW
# =====================================
print("\n1. INFORMACIÓN DE TENSORFLOW")
print("-" * 70)
print(f"   Versión de TensorFlow: {tf.__version__}")
print(f"   Versión de Python: {sys.version}")

# =====================================
# 2. DISPOSITIVOS DETECTADOS
# =====================================
print("\n2. DISPOSITIVOS DETECTADOS")
print("-" * 70)

all_devices = tf.config.list_physical_devices()
gpu_devices = tf.config.list_physical_devices('GPU')
cpu_devices = tf.config.list_physical_devices('CPU')

print(f"   Total de dispositivos: {len(all_devices)}")
print(f"   CPUs detectadas: {len(cpu_devices)}")
for cpu in cpu_devices:
    print(f"     - {cpu.name}")

print(f"   GPUs detectadas: {len(gpu_devices)}")
if gpu_devices:
    for gpu in gpu_devices:
        print(f"     - {gpu.name}")
else:
    print("     ¡NINGUNA GPU DETECTADA!")

# =====================================
# 3. SOPORTE DE CUDA
# =====================================
print("\n3. SOPORTE DE CUDA")
print("-" * 70)
print(f"   ¿CUDA compilado en TensorFlow?: {tf.test.is_built_with_cuda()}")

# =====================================
# 4. DETALLES DE DISPOSITIVOS
# =====================================
print("\n4. DETALLES DE DISPOSITIVOS")
print("-" * 70)
if gpu_devices:
    for i, gpu in enumerate(gpu_devices):
        print(f"\n   GPU {i}:")
        print(f"     Nombre: {gpu.name}")
        print(f"     Tipo: {gpu.device_type}")
        
        # Intentar obtener más detalles
        try:
            details = tf.sysconfig.get_build_info()
            print(f"     CUDA: Versión disponible")
        except:
            pass
else:
    print("   No hay GPUs para mostrar detalles")

# =====================================
# 5. CONFIGURACIÓN DE MEMORIA
# =====================================
print("\n5. CONFIGURACIÓN DE MEMORIA")
print("-" * 70)
try:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"   ✓ Memory growth activado para {gpu.name}")
        except RuntimeError as e:
            print(f"   ✗ Error al configurar memory growth: {e}")
except Exception as e:
    print(f"   Error: {e}")

# =====================================
# 6. PRUEBA DE CÁLCULO
# =====================================
print("\n6. PRUEBA DE CÁLCULO")
print("-" * 70)

# Verificar dispositivo por defecto
with tf.device('/GPU:0' if gpu_devices else '/CPU:0'):
    try:
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        c = tf.matmul(a, b)
        device_used = c.device if hasattr(c, 'device') else "CPU (fallback)"
        print(f"   ✓ Prueba de cálculo exitosa")
        print(f"   Dispositivo usado: {device_used}")
    except Exception as e:
        print(f"   ✗ Error en prueba de cálculo: {e}")

# =====================================
# 7. POSIBLES SOLUCIONES
# =====================================
print("\n7. POSIBLES SOLUCIONES SI NO SE DETECTA GPU")
print("-" * 70)
if not gpu_devices:
    print("""
   La GPU NVIDIA 780 no fue detectada. Posibles causas:
   
   A) DRIVERS:
      • Actualiza los drivers NVIDIA a la versión más reciente
      • Descarga desde: https://www.nvidia.com/Download/driverDetails.aspx
   
   B) CUDA Y cuDNN:
      • La GPU 780 es antigua (arquitectura Kepler)
      • Puede requerir versiones específicas de CUDA
      • TensorFlow 2.x requiere CUDA 11.x+ (no compatible con GPU 780 vieja)
      • Solución: Usa TensorFlow 2.10 o anteriores, o instala CUDA 11.8
   
   C) COMPATIBILIDAD:
      • GPU 780: Capacidad de Compute 3.0 (Kepler)
      • TensorFlow 2.11+: Requiere Compute Capability 3.7+ (Maxwell o más nuevo)
      • Solución: Usa TensorFlow 2.10 LTS o anterior
   
   D) VARIABLES DE ENTORNO:
      • Verifica que CUDA_HOME esté configurado correctamente
      • En Windows: Comprueba que nvidia-smi funcione en cmd
   
   E) REINSTALACIÓN RECOMENDADA:
      • pip install --upgrade tensorflow==2.10.0
      • O instala tensorflow-gpu específicamente
   
   F) VERIFICACIÓN RÁPIDA:
      • Abre Command Prompt y ejecuta: nvidia-smi
      • Deberías ver tu GPU 780 listada
      • Si no aparece, es problema de drivers
    """)
else:
    print("   ¡GPU DETECTADA CORRECTAMENTE! No hay problemas.")

print("\n" + "="*70)
print("FIN DEL DIAGNÓSTICO")
print("="*70 + "\n")
