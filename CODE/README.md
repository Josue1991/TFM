# TFM - Código Fuente

Estructura del proyecto optimizado para repositorio Git.

## 📁 Estructura

```
CODE/
├── TFM_Fase1/           # CNN con 2 datasets
│   ├── cnn_experimento.py
│   ├── cnn_modelo.py
│   └── requirements.txt
├── TFM_Fase2/           # LSTM con 2 datasets
│   ├── fase2_completo.py
│   ├── lstm_modelo.py
│   ├── ecg_lstm.py
│   ├── har_lstm.py
│   └── requerimientos.txt
├── utils/
│   └── MEJORAS_APRENDIZAJE.py
├── config/
│   └── [configuración adicional]
├── requirements.txt     # Dependencias principales
└── README.md           # Este archivo
```

## 📋 Requisitos

```bash
pip install -r requirements.txt
```

O por fase:

**Fase 1 (CNN):**
```bash
cd TFM_Fase1
pip install -r requirements.txt
```

**Fase 2 (LSTM):**
```bash
cd TFM_Fase2
pip install -r requerimientos.txt
```

## 🚀 Ejecución Rápida

### Fase 1: CNN (Fashion MNIST + CIFAR-10)

```bash
cd TFM_Fase1
python cnn_experimento.py
```

**Tiempo:**
- CPU: 15-20 minutos (epochs=10)
- GPU: 5-10 minutos (epochs=10)

**Configuración:** Editar líneas 25-26 en `cnn_experimento.py`

```python
EPOCHS_Fashion = 10    # Cambiar aquí
EPOCHS_CIFAR10 = 10    # Cambiar aquí
```

### Fase 2: LSTM (ECG5000 + UCI HAR)

```bash
cd TFM_Fase2
python fase2_completo.py
```

**Tiempo:**
- CPU: 30-40 minutos
- GPU: 10-15 minutos

**Configuración:** Editar líneas 28-30 en `fase2_completo.py`

```python
EPOCHS_ECG = 50      # Cambiar aquí
EPOCHS_HAR = 30      # Cambiar aquí
BATCH_SIZE = 32
```

## ⚙️ Configuración por Necesidad

### Testing Rápido (5-10 minutos)
```python
# Fase 1
EPOCHS_Fashion = 2
EPOCHS_CIFAR10 = 2

# Fase 2
EPOCHS_ECG = 2
EPOCHS_HAR = 2
```

### Resultados Buenos (30-60 minutos)
```python
# Fase 1
EPOCHS_Fashion = 10
EPOCHS_CIFAR10 = 10

# Fase 2
EPOCHS_ECG = 20
EPOCHS_HAR = 15
```

### Resultados Óptimos (2-3+ horas)
```python
# Fase 1
EPOCHS_Fashion = 20
EPOCHS_CIFAR10 = 20

# Fase 2
EPOCHS_ECG = 100
EPOCHS_HAR = 80
```

## 📊 Salidas

Los scripts generan:

**Archivos CSV:**
- `csv_data/resultados_fase1.csv`
- `csv_data/fase2_completo.csv`

**Gráficos:**
- `results/grafico_accuracy.png`
- `results/grafico_loss.png`
- `results/grafico_tiempo.png`
- `results/fase2_lstm_training.png`

## 🎯 Optimizaciones Implementadas

### Early Stopping
- **Parámetro:** patience=15
- **Monitor:** val_loss
- **Efecto:** Evita overfitting automáticamente

### Learning Rate Scheduler
- **Tipo:** ReduceLROnPlateau
- **Factor:** 0.5
- **Patience:** 5
- **Efecto:** Ajusta learning rate cuando val_loss se estanca

### Regularización
- **Batch Normalization:** En cada capa convolucional/LSTM
- **Dropout:** 0.3-0.5 según la capa
- **L2 Regularization:** 0.001 en todas las capas

### Arquitectura
- **CNN:** Conv2D → BatchNorm → Dropout → MaxPool (3 bloques)
- **LSTM:** Bidirectional LSTM con BatchNorm y Dropout

### Optimizador
- **Adam:** learning_rate=0.001, beta_1=0.9, beta_2=0.999

## 🔧 Troubleshooting

### Memoria insuficiente
```python
# En cnn_experimento.py o fase2_completo.py:
BATCH_SIZE = 16  # Reducir de 32
```

### Quiero solo CPU
```python
DEVICE = 'CPU'  # No usar GPU
```

### GPU no detectada
```bash
# Verificar:
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 📝 Estructura de Código

### TFM_Fase1/cnn_experimento.py
- Carga datasets (Fashion MNIST + CIFAR-10)
- Preprocesa imágenes (normalización, resizing, conversión RGB)
- Entrena modelos CNN
- Genera resultados y gráficos

### TFM_Fase1/cnn_modelo.py
- `build_cnn_model()`: Construye arquitectura CNN
- `train_and_measure()`: Entrena con optimizaciones
- Early Stopping + ReduceLROnPlateau

### TFM_Fase2/fase2_completo.py
- Carga datasets (ECG5000 + UCI HAR)
- Construye modelos LSTM bidireccionales
- Entrena con validación separada
- Genera gráficos de training history

### utils/MEJORAS_APRENDIZAJE.py
- 7 técnicas de optimización documentadas
- Ejemplos de implementación
- Comparación antes/después

## 🚀 Git

Para subir a repositorio:

```bash
cd CODE
git init
git add .
git commit -m "TFM - Código fuente con optimizaciones"
git remote add origin [tu-repo]
git push -u origin main
```

## 📚 Documentación

Para guías de uso, configuración e instalación:
- Ve a la carpeta **DOCS/**
- O lee **DOCS/README.md**

## ✅ Verificación

Para verificar que todo funciona:

```bash
# Test Fase 1 (2-3 minutos)
cd TFM_Fase1
# Edita cnn_experimento.py: EPOCHS_Fashion = 2, EPOCHS_CIFAR10 = 2
python cnn_experimento.py

# Test Fase 2 (3-5 minutos)
cd ../TFM_Fase2
# Edita fase2_completo.py: EPOCHS_ECG = 2, EPOCHS_HAR = 2
python fase2_completo.py
```

## 💡 Tips

1. **Usa GPU en Google Colab** para entrenamientos rápidos (25-30 min)
2. **Epochs bajos (2-5)** para testing
3. **Epochs normales (10-30)** para resultados
4. **Cambia BATCH_SIZE** si hay memory issues
5. **Early Stopping detiene automáticamente** el entrenamiento

## 📞 Soporte

Para preguntas o problemas:
1. Revisa la documentación en **DOCS/**
2. Verifica los requisitos en **requirements.txt**
3. Intenta con epochs bajos para diagnosticar

---

**Última actualización:** Noviembre 2025
**Versión:** 1.0
