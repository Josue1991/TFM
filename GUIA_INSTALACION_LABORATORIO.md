# 🔬 GUÍA DE INSTALACIÓN - LABORATORIO

## 📋 Para el Profesor

Esta guía te ayudará a instalar el proyecto usando **Conda** en una PC de laboratorio.

---

## ✅ Requisitos Previos

- **Anaconda** o **Miniconda** instalado
- **Python 3.11+** (se instalará con conda)
- **GPU NVIDIA** (opcional, para entrenamiento rápido)
- **10 GB espacio libre** en disco

---

## 🚀 Opción 1: Instalación con Conda (RECOMENDADA)

### Paso 1: Crear entorno desde archivo YML

```powershell
# Navegar a la carpeta del proyecto
cd c:\Proyectos\TFM

# Crear entorno desde environment.yml
conda env create -f environment.yml

# Activar entorno
conda activate tfm_proyecto
```

### Paso 2: Verificar instalación

```powershell
# Verificar Python
python --version
# Debe mostrar: Python 3.11.x

# Verificar TensorFlow
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
# Debe mostrar: TensorFlow: 2.13.0

# Verificar GPU (si está disponible)
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

---

## 🔧 Opción 2: Instalación Manual con Conda

Si `environment.yml` falla, instalar manualmente:

```powershell
# Crear entorno con Python 3.11
conda create -n tfm_proyecto python=3.11 -y

# Activar entorno
conda activate tfm_proyecto

# Instalar dependencias desde requirements.txt
pip install -r CODE/requirements.txt
```

---

## 📦 Librerías Instaladas

| Librería | Versión | Propósito |
|----------|---------|-----------|
| TensorFlow | 2.13.0 | Framework de Deep Learning |
| NumPy | 1.24.3 | Arrays y operaciones numéricas |
| Pandas | 2.0.3 | Manejo de datos tabulares |
| Scikit-learn | 1.3.0 | Preprocesamiento y métricas |
| Matplotlib | 3.7.2 | Visualización de resultados |
| Pillow | 10.0.0 | Procesamiento de imágenes |

---

## ✅ Verificación Completa

```powershell
# Activar entorno
conda activate tfm_proyecto

# Test rápido (5-10 segundos)
python -c "
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
print('✓ Todas las librerías importadas correctamente')
print(f'✓ TensorFlow {tf.__version__}')
print(f'✓ Dispositivos disponibles: {len(tf.config.list_physical_devices())}')
gpu = tf.config.list_physical_devices('GPU')
print(f'✓ GPU: {"Sí (" + gpu[0].name + ")" if gpu else "No (usará CPU)"}')
"
```

**Salida esperada:**
```
✓ Todas las librerías importadas correctamente
✓ TensorFlow 2.13.0
✓ Dispositivos disponibles: 1 o 2
✓ GPU: Sí (GPU:0) o No (usará CPU)
```

---

## 🧪 Prueba Rápida del Proyecto

### Test 1: CNN (Fase 1) - 3 minutos

```powershell
cd TFM_Fase1
python cnn_experimento.py
```

**Resultado esperado:**
- ✅ Se ejecuta sin errores
- ✅ Genera `csv_data/resultados_fase1.csv`
- ✅ Genera gráficos en `results/`
- ✅ Accuracy > 70% (Fashion MNIST)

### Test 2: LSTM (Fase 2) - 3 minutos

```powershell
cd ..\TFM_Fase2
python fase2_completo.py
```

**Resultado esperado:**
- ✅ Se ejecuta sin errores
- ✅ Genera `csv_data/ecg_results.csv` y `csv_data/har_results.csv`
- ✅ Genera gráficos en `results/`
- ✅ Accuracy > 80% (ambos datasets)

### Test Completo: Todas las Fases - 10 minutos

```powershell
cd c:\Proyectos\TFM

# Opción A: Ejecutar con script PowerShell
.\EJECUTAR_FASES_MEJORADAS.ps1

# Opción B: Ejecutar manualmente
cd TFM_Fase1
python cnn_experimento.py
cd ..\TFM_Fase2
python fase2_completo.py
```

---

## 🔍 Solución de Problemas

### Problema 1: "conda no reconocido"

**Solución:**
```powershell
# Reiniciar terminal o ejecutar
C:\ProgramData\Anaconda3\Scripts\activate.bat
# o
C:\Users\[Usuario]\Anaconda3\Scripts\activate.bat
```

### Problema 2: Error con TensorFlow

**Solución:**
```powershell
# Desinstalar y reinstalar
pip uninstall tensorflow -y
pip install tensorflow==2.13.0
```

### Problema 3: GPU no detectada

**Verificar:**
```powershell
# Ver si CUDA está instalado
nvidia-smi

# Ver versión de CUDA requerida para TensorFlow 2.13
# Necesita: CUDA 11.8 + cuDNN 8.6
```

**Si no hay GPU:** No hay problema, el código usará CPU automáticamente.

### Problema 4: Espacio en disco insuficiente

**Requerimientos:**
- Entorno conda: ~2 GB
- Datasets: ~500 MB
- Resultados: ~100 MB
- **Total: ~3 GB mínimo**

---

## 📊 Tiempos de Ejecución Estimados

### Con GPU (NVIDIA GTX 1060 o superior)
- **CNN (Fase 1):** 5-10 minutos
- **LSTM (Fase 2):** 5-10 minutos
- **Total:** ~15-20 minutos

### Con CPU (i5/i7 moderno)
- **CNN (Fase 1):** 20-40 minutos
- **LSTM (Fase 2):** 20-40 minutos
- **Total:** ~40-80 minutos

---

## 📁 Archivos Centralizados de Dependencias

| Archivo | Uso | Ubicación |
|---------|-----|-----------|
| `environment.yml` | Conda (RECOMENDADO) | Raíz del proyecto |
| `CODE/requirements.txt` | pip (alternativa) | `CODE/` |

**✅ Las dependencias están centralizadas en estos 2 archivos.**

Los demás `requirements.txt` en subcarpetas son antiguos y pueden ignorarse.

---

## 🎯 Checklist de Instalación

```
Para el profesor, verificar:

□ Anaconda/Miniconda instalado
□ Entorno creado: conda env create -f environment.yml
□ Entorno activado: conda activate tfm_proyecto
□ TensorFlow funciona: python -c "import tensorflow"
□ Test rápido pasado (ver sección Verificación Completa)
□ Fase 1 ejecutada: python TFM_Fase1/cnn_experimento.py
□ Fase 2 ejecutada: python TFM_Fase2/fase2_completo.py
□ Resultados generados en csv_data/ y results/
```

---

## 💾 Exportar Entorno (Para compartir)

Si el profesor quiere guardar el entorno exacto:

```powershell
# Exportar environment.yml
conda env export > environment_exacto.yml

# O exportar requirements.txt
pip freeze > requirements_exacto.txt
```

---

## 🔄 Desinstalar (Después de evaluar)

```powershell
# Desactivar entorno
conda deactivate

# Eliminar entorno
conda env remove -n tfm_proyecto

# Verificar eliminación
conda env list
```

---

## 📞 Contacto

Si hay problemas durante la instalación:

1. **Verificar errores:** Revisar mensaje completo
2. **Buscar en internet:** Copiar mensaje de error
3. **Alternativa:** Usar Google Colab (ver `Colab/COLAB_SETUP.md`)

---

## ✨ Resumen para el Profesor

**Comando único de instalación:**
```powershell
cd c:\Proyectos\TFM
conda env create -f environment.yml
conda activate tfm_proyecto
```

**Comando único de prueba:**
```powershell
cd TFM_Fase1 && python cnn_experimento.py
```

**Tiempo total:** 10-15 minutos instalación + 15-80 minutos ejecución

---

**Listo para evaluar el proyecto. 🚀**
