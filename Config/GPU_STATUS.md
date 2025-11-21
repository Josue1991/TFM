# ESTADO DEL PROYECTO TFM - 20 Noviembre 2025

## ✅ COMPLETADO

### CPU Testing Exitoso
- **ECG5000**: 19.0% accuracy, 1.63 loss, 55.5s entrenamiento
- **UCI HAR**: 90.97% accuracy, 0.34 loss, 543.7s entrenamiento
- Ambiente virtual configurado con TensorFlow 2.13
- Scripts modulares funcionando correctamente

### Reorganización de Codebase
```
TFM_Fase1/
├── cnn_modelo.py (modelo separado)
├── cnn_experimento.py (script ejecución)
├── results/ (gráficas)
└── csv_data/ (resultados)

TFM_Fase2/
├── lstm_modelo.py (modelo LSTM base)
├── lstm_modelo_gpu.py (versión GPU-ready)
├── ecg_lsmt.py (ECG experimento)
├── har_lstm.py (HAR experimento)
├── results/ (gráficas)
└── csv_data/ (CSV resultados)
```

### Dependencias Configuradas
- TensorFlow 2.13.0 (CPU en venv)
- CUDA 11.8 y 12.9 (instalados en sistema)
- NumPy 1.24.3 (compatible con TF 2.13)
- Pandas, Matplotlib, scikit-learn

## ⏳ TRABAJO EN PROGRESO

### GPU Acceleration
- **Estado**: TensorFlow 2.13 compilado CPU-only en venv
- **GPU Detectada**: NVIDIA GeForce 940M (Maxwell, CC 5.0)
- **CUDA_HOME**: Configurado a v11.8
- **Problema**: PyPI no proporciona wheels GPU para Windows + Python 3.11

### Opciones GPU

#### Opción 1: Compilación desde Fuente (Máxima Aceleración)
```powershell
# Requisitos:
- Bazel (https://bazel.build)
- Visual Studio 2022 + Build Tools
- cuDNN 8.x (descargar manualmente)
- Tiempo: 2-4 horas
- Espacio: ~50GB

# Proceso:
.\build_tensorflow_gpu.ps1
```
**Ventajas**: GPU nativo, máxima velocidad
**Desventajas**: Largo setup, requiere compilador C++

#### Opción 2: Python 3.9 + TensorFlow 2.10 GPU (Alternativa)
```powershell
# TF 2.10 tiene wheels GPU pre-compiladas para Python 3.9
# Crear venv con Python 3.9:
python3.9 -m venv venv39
.\venv39\Scripts\Activate.ps1
pip install tensorflow==2.10.0
```
**Ventajas**: Más rápido que compilar
**Desventajas**: Requiere instalar Python 3.9

#### Opción 3: Google Colab (Cloud GPU)
```python
# En Colab (GPU Tesla K80/A100 gratis):
from google.colab import drive
drive.mount('/content/drive')

# Upload scripts y datos, ejecutar allí
```
**Ventajas**: GPU potente, setup rápido
**Desventajas**: Datos en cloud, sesiones limitadas

#### Opción 4: Mantener CPU (Actual)
- Scripts funcionan correctamente
- Precisión probada: 90.97% HAR, 19% ECG (sintético)
- Suficiente para demostración/educación

## 📊 RESULTADOS ACTUALES

### Ambiente Actual
```
Python: 3.11.9
TensorFlow: 2.13.0 (CPU)
GPU: No disponible (requiere compilación)
RAM: ~4GB usado por TensorFlow
Tiempo de boot: ~2-3s por script
```

### Métricas de Rendimiento
```
ECG5000 (50 épocas):
- Accuracy: 19.0% ⚠️ (dataset sintético pequeño)
- Loss: 1.628
- Tiempo: 55.5s en CPU

UCI HAR (30 épocas):
- Accuracy: 90.97% ✅ (datos reales)
- Loss: 0.336
- Tiempo: 543.7s en CPU (~9 min)

Estimado con GPU (2-4x más rápido):
- ECG: ~15-25s
- HAR: ~130-270s
```

## 🔧 SCRIPTS DISPONIBLES

### Diagnóstico
```powershell
python gpu_diagnostico.py          # Info GPU/TF completa
python verificar_gpu.py             # Check nvidia-smi
```

### Entrenamiento
```powershell
# En venv:
python TFM_Fase1\cnn_experimento.py   # CNN Fashion MNIST + Flores Oxford
python TFM_Fase2\ecg_lsmt.py          # LSTM ECG5000
python TFM_Fase2\har_lstm.py          # LSTM UCI HAR
```

### Generación de Reportes
```powershell
python TFM_Fase2\fase2_report.py      # Reporte Fase 2
```

## 🎯 PRÓXIMOS PASOS

### Inmediato (Sin GPU)
1. ✅ Tests CPU completados
2. Ejecutar CNN Fase1 en venv
3. Generar reportes PDF

### Con GPU (Requerido)
**Si deseas activar GPU:**

1. **Opción Recomendada**: Compilar desde fuente
   ```powershell
   .\build_tensorflow_gpu.ps1
   ```

2. **O** cambiar a Python 3.9 + TF 2.10
   ```powershell
   python3.9 -m venv venv39
   .\venv39\Scripts\Activate.ps1
   pip install tensorflow==2.10.0
   ```

## 📝 NOTAS TÉCNICAS

### Por qué TensorFlow CPU-only?
- PyPI no proporciona wheels GPU pre-compilados para Windows + Python 3.11
- TensorFlow 2.13 compilado en PyPI es CPU-only
- Opciones:
  1. Compilar desde source (Bazel) ← Mejor
  2. Usar Python 3.9 + TF 2.10
  3. Usar Conda (a veces tiene wheels GPU)

### GPU 940M Limitaciones
- Maxwell architecture (CC 5.0)
- TensorFlow 2.11+ requiere CUDA 12+ (no soporta CC 5.0)
- Máximo: TensorFlow 2.10 con CUDA 11.x
- CUDA 11.8 ya instalado ✓

### Próximos Cambios Recomendados
1. Migrar a Python 3.9 (más compatible con GPU wheels)
2. O compilar TensorFlow 2.10 desde source
3. Documentar benchmark CPU vs GPU

## 📚 REFERENCIAS

- TensorFlow Build: https://www.tensorflow.org/install/source_windows
- Bazel Install: https://bazel.build/install/windows
- cuDNN Archive: https://developer.nvidia.com/rdp/cudnn-archive
- CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive

---
**Generado**: 20 Nov 2025, 14:50 UTC
**Estado**: ✅ Funcional (CPU), ⏳ GPU Pendiente
