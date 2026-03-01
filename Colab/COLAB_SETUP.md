# 🚀 Usar TFM con Google Colab GPU

## 📋 Notebooks Disponibles

### ⭐ **RECOMENDADO: TFM_Colab_Completo_Benchmarking.ipynb**
- ✅ **Fase 1 (CNN):** Fashion MNIST + CIFAR-10  
- ✅ **Fase 2 (LSTM):** ECG5000 + UCI HAR  
- ✅ **Benchmarking:** 26 métricas detalladas  
- ✅ **CSV completo** para comparativas CPU/GPU/Multi-GPU  
- ⏱️ **Tiempo:** 25-45 minutos

### Otros notebooks disponibles:
- **TFM_Colab_GPU.ipynb** - Versión optimizada sin benchmarking
- **TFM_Colab_Simple.ipynb** - Versión básica educativa

📚 **Ver comparativa completa:** [README_NOTEBOOKS.md](README_NOTEBOOKS.md)

---

## ¿Por qué Google Colab?

- ✅ GPU **GRATIS** (Tesla K80, T4, A100)
- ✅ No requiere instalación local
- ✅ 12GB RAM + GPU VRAM
- ✅ Pre-instalado TensorFlow con CUDA
- ✅ Ideal para entrenamientos de ML

## Pasos Rápidos

### 1. Abrir Google Colab

```
https://colab.research.google.com/
```

### 2. Crear Notebook Nuevo
- Menú: File → New Notebook
- O importar desde GitHub/Drive

### 3. Habilitar GPU ⚙️

**IMPORTANTE**: Hacer esto PRIMERO antes de ejecutar código

```
Menú: Runtime → Change Runtime Type
├─ Hardware accelerator: [Dropdown]
└─ Seleccionar: GPU (T4, L4 o A100)
```

Debe ver: `⚡ GPU enabled` en verde

### 4. Copiar Código del Notebook

**Opción A (Recomendada):** Upload del notebook completo
- **File → Upload Notebook**
- Seleccionar: `TFM_Colab_Completo_Benchmarking.ipynb`

**Opción B:** Otros notebooks
- `TFM_Colab_GPU.ipynb` - Sin benchmarking
- `TFM_Colab_Simple.ipynb` - Versión básica

### 5. Ejecutar

```
Shift + Enter   : Ejecutar celda actual
Ctrl + F9       : Ejecutar todas las celdas
```

### 6. Descargar Resultados

Los archivos se descargan automáticamente:
- `resultados_gpu_colab.csv`
- `comparacion_cpu_gpu.csv`
- `grafica_entrenamiento_gpu.png`
- `comparacion_cpu_gpu.png`

---

## 📊 Qué Hace el Notebook

### Paso 1: Verificar GPU
```
✓ GPUs detectadas: 1
  GPU 0: /job:localhost/replica:0/task:0/device:GPU:0
✓ TensorFlow compilado con CUDA: True
✓ GPU disponible: True
```

### Paso 2-4: Entrenar Modelos
```
ECG5000 (50 épocas):
  - Entrenamiento en GPU
  - Resultados guardados

UCI HAR (30 épocas):
  - Descarga dataset real
  - Entrenamiento en GPU
  - Métricas calculadas
```

### Paso 5: Comparación CPU vs GPU

| Dataset | CPU Local | GPU Colab | Speedup |
|---------|-----------|-----------|---------|
| ECG5000 | 55.5s | ~15-20s | **2.8-3.7x** |
| UCI HAR | 543.7s | ~150-200s | **2.7-3.6x** |

### Paso 6-7: Gráficas y Descargas
- Gráficas Accuracy/Loss
- CSV con resultados
- Archivos descargados automáticamente

---

## 🖥️ Tiempos Estimados

- **Setup inicial**: 2-3 min
- **ECG5000 GPU**: 10-15 min
- **UCI HAR GPU**: 5-10 min
- **Gráficas**: 1-2 min
- **TOTAL**: 20-30 min

---

## ⚙️ Variables de Entorno (Opcional)

Si Colab detecta problema con GPU, intentar:

```python
# En primera celda
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
print(tf.test.is_built_with_cuda())  # Debe ser True
```

---

## 🔍 Troubleshooting

### ❌ No detecta GPU
**Solución:**
1. Runtime → Change Runtime Type
2. Verificar "GPU" está seleccionado
3. Hacer click "Save"
4. Ejecutar primera celda de nuevo

### ❌ "CUDA not found"
**Solución:**
- Colab automáticamente instala CUDA
- Si error persiste, reiniciar runtime:
  - Menú: Runtime → Restart Runtime

### ❌ Descargas no aparecen
**Solución:**
- Las descargas están en carpeta `Downloads`
- Si no aparecen, usar: `files.download('archivo.csv')`

---

## 📥 Importar Código Local

Si quieres usar tu código local en Colab:

```python
# En Colab
from google.colab import drive
drive.mount('/content/drive')

# Ahora accedes a Drive:
# /content/drive/My Drive/TFM_Proyecto/
```

---

## 💾 Guardar Resultados en Drive

```python
# Guardar en Drive (si deseas persistencia)
import shutil
shutil.copy('resultados_gpu_colab.csv', 
            '/content/drive/My Drive/TFM_Proyecto/resultados_gpu.csv')
```

---

## 📈 Benchmarks Esperados

### ECG5000 (GPU Colab - T4)
```
Época 1:  Accuracy: 0.25 | Loss: 1.45
Época 10: Accuracy: 0.35 | Loss: 1.25
Época 50: Accuracy: 0.40-0.50 | Loss: 0.90-1.10
```

### UCI HAR (GPU Colab - T4)
```
Época 1:  Accuracy: 0.45 | Loss: 1.80
Época 10: Accuracy: 0.75 | Loss: 0.70
Época 30: Accuracy: 0.88-0.92 | Loss: 0.30-0.40
```

---

## 🎯 Siguientes Pasos

1. ✅ Ejecutar notebook en Colab
2. ✅ Anotar tiempos GPU y compararlos con CPU local
3. ✅ Descargar gráficas y resultados
4. ✅ Documentar speedup conseguido
5. ⏳ Aumentar épocas si tiempo lo permite

---

## 📞 Errores Comunes

| Error | Causa | Solución |
|-------|-------|----------|
| `ModuleNotFoundError: numpy` | Pip no instaló | `!pip install numpy` |
| `CUDA not found` | GPU no habilitada | Cambiar Runtime a GPU |
| `Out of memory` | Batch size muy grande | Reducir de 32 a 16 |
| `IncompleteRead` | Conexión internet débil | Reintentar celda |

---

## 🚀 Comando Rápido

Copiar TODO el código en una celda Colab:

```python
# 1. GPU Check
import tensorflow as tf
print(f"GPU: {tf.test.is_built_with_cuda()}")

# 2. Instalar dependencias
!pip install -q scikit-learn pandas matplotlib

# 3. Definir LSTM (ver celdas 3-4)
# 4. Generar ECG dataset (ver celda 5)
# 5. Entrenar ECG (ver celda 6)
# 6. Descargar UCI HAR (ver celda 7)
# 7. Entrenar UCI HAR (ver celda 7)
# 8. Gráficas y comparación (ver celdas 8-10)
```

---

**Última actualización:** 20 Nov 2025
**Estado:** ✅ Listo para usar
**Soporte:** TensorFlow 2.13+, Python 3.7+
