"""
cnn_experimento.py
Script principal que ejecuta experimentos con CNN en diferentes datasets.
Carga datos, entrena modelos y genera gráficos de resultados.

Keras es una API de alto nivel para construir y entrenar redes 
neuronales fácilmente, integrada dentro de TensorFlow y compatible con CPU y GPU.
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import numpy as np
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cnn_modelo import train_and_measure

# Importar utilidades de benchmarking (si están disponibles)
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CODE', 'utils'))
    from benchmark_utils import get_device_info, enrich_results, print_device_summary, save_enriched_results
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False
    print("[INFO] benchmark_utils no disponible. Solo se guardarán resultados básicos.")

# =====================================
# CONFIGURACIÓN GLOBAL - CAMBIAR AQUÍ
# =====================================
# Para testing rápido: epochs=2 (3-5 minutos)
# Para resultados: epochs=10 (15-20 minutos)
# Para óptimos: epochs=20+ (30+ minutos)
EPOCHS_Fashion = 20   # Epochs para Fashion MNIST (coincide con Colab)
EPOCHS_CIFAR10 = 30   # Epochs para CIFAR-10 (coincide con Colab)
BATCH_SIZE = 32        # Tamaño de lote
DEVICE = 'auto'        # 'GPU', 'CPU', o 'auto'

# =====================================
# CREAR CARPETAS PARA RESULTADOS
# =====================================
RESULTS_DIR = "results"
CSV_DIR = "csv_data"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
if not os.path.exists(CSV_DIR):
    os.makedirs(CSV_DIR)

# =====================================
# DETECTAR CPU O GPU Y CAPTURAR INFO
# =====================================
device_name = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
print(f"Entrenando usando: {device_name}")

# Capturar información detallada del sistema
if BENCHMARK_AVAILABLE:
    device_info = get_device_info()
    print_device_summary(device_info)
else:
    device_info = None


def load_and_preprocess_dataset(dataset_name, img_size):
    """
    Carga dataset de Keras y prepara para entrenamiento.
    
    MEJORAS IMPLEMENTADAS:
    - Normalización de píxeles
    - Validación separada
    - Conversión a RGB para grayscale datasets
    
    Args:
        dataset_name (str): 'fashion_mnist' o 'cifar10'
        img_size (tuple): Tamaño de redimensionamiento (alto, ancho)
        
    Returns:
        tuple: (ds_train, ds_val, ds_test, num_classes)
    """
    print(f"Cargando dataset: {dataset_name}...")
    
    if dataset_name == 'fashion_mnist':
        # Cargar desde Keras
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        num_classes = 10
        grayscale = True
    elif dataset_name == 'cifar10':
        # Cargar CIFAR-10 desde Keras
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        num_classes = 10
        grayscale = False
    else:
        raise ValueError(f"Dataset desconocido: {dataset_name}. Use 'fashion_mnist' o 'cifar10'")
    
    # Normalizar a [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Si es grayscale (Fashion MNIST), expandir dimensión
    if grayscale:
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
    
    # Redimensionar usando tf.image.resize (optimizado para memoria)
    def resize_batch(x, target_size):
        """Redimensionar batch de imágenes usando float32 para ahorrar memoria."""
        # Usar float32 en lugar de float64 para reducir uso de memoria a la mitad
        x_resized = np.zeros((x.shape[0], target_size[0], target_size[1], x.shape[-1]), dtype=np.float32)
        # Procesar en batches más pequeños para evitar picos de memoria
        batch_size = 1000
        for i in range(0, len(x), batch_size):
            end_idx = min(i + batch_size, len(x))
            x_resized[i:end_idx] = tf.image.resize(x[i:end_idx], target_size).numpy().astype(np.float32)
        return x_resized
    
    x_train = resize_batch(x_train, img_size)
    x_test = resize_batch(x_test, img_size)
    
    # Convertir grayscale a RGB (duplicar canales)
    if grayscale:
        x_train = np.repeat(x_train, 3, axis=-1)  # (N, H, W, 3)
        x_test = np.repeat(x_test, 3, axis=-1)
    
    x_train_rgb = x_train
    x_test_rgb = x_test
    
    # Dividir en train (80%) y val (20%)
    split_idx = int(len(x_train_rgb) * 0.8)
    x_val = x_train_rgb[split_idx:]
    y_val = y_train[split_idx:]
    x_train_final = x_train_rgb[:split_idx]
    y_train_final = y_train[:split_idx]
    
    # Crear datasets
    ds_train = tf.data.Dataset.from_tensor_slices((x_train_final, y_train_final))
    ds_train = ds_train.shuffle(buffer_size=5000)
    ds_train = ds_train.batch(32).prefetch(tf.data.AUTOTUNE)
    
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    ds_val = ds_val.batch(32).prefetch(tf.data.AUTOTUNE)
    
    ds_test = tf.data.Dataset.from_tensor_slices((x_test_rgb, y_test))
    ds_test = ds_test.batch(32).prefetch(tf.data.AUTOTUNE)
    
    return ds_train, ds_val, ds_test, num_classes


def generate_result_graphs(df):
    """
    Genera y guarda gráficos de los resultados de los experimentos.
    
    Args:
        df (pd.DataFrame): DataFrame con columnas: dataset, accuracy, loss, training_time
    """
    # ==============================
    # GRÁFICO 1: Accuracy por Dataset
    # ==============================
    plt.figure(figsize=(6, 4))
    plt.bar(df["dataset"], df["accuracy"], color='steelblue')
    plt.title("Accuracy por Dataset", fontsize=14, fontweight='bold')
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "grafico_accuracy.png"), dpi=150)
    plt.close()
    print(f"✓ Gráfico guardado: {RESULTS_DIR}/grafico_accuracy.png")

    # ==============================
    # GRÁFICO 2: Loss por Dataset
    # ==============================
    plt.figure(figsize=(6, 4))
    plt.bar(df["dataset"], df["loss"], color='coral')
    plt.title("Loss por Dataset", fontsize=14, fontweight='bold')
    plt.xlabel("Dataset")
    plt.ylabel("Loss")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "grafico_loss.png"), dpi=150)
    plt.close()
    print(f"✓ Gráfico guardado: {RESULTS_DIR}/grafico_loss.png")

    # ==============================
    # GRÁFICO 3: Tiempo de Entrenamiento
    # ==============================
    plt.figure(figsize=(6, 4))
    plt.bar(df["dataset"], df["training_time"], color='mediumseagreen')
    plt.title("Tiempo de Entrenamiento por Dataset", fontsize=14, fontweight='bold')
    plt.xlabel("Dataset")
    plt.ylabel("Tiempo (segundos)")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "grafico_tiempo.png"), dpi=150)
    plt.close()
    print(f"✓ Gráfico guardado: {RESULTS_DIR}/grafico_tiempo.png")


# =====================================
# EJECUTAR EXPERIMENTOS
# =====================================
print("\n" + "="*60)
print("INICIANDO EXPERIMENTOS CNN")
print("="*60)

results = []
enriched_results = []

# Dataset 1 — Fashion MNIST (28x28 → escalado a 64x64)
# COMENTADO TEMPORALMENTE - Ya completado exitosamente (82.4% accuracy)
# Se ejecutará solo CIFAR-10 para corregir error de memoria
"""
print("\n--- Experimento 1: Fashion MNIST ---")
ds_train_1, ds_val_1, ds_test_1, num_classes_1 = load_and_preprocess_dataset(
    "fashion_mnist", (64, 64)
)
r1, h1, model1 = train_and_measure(
    "Fashion MNIST", (64, 64), ds_train_1, ds_val_1, ds_test_1, num_classes_1, EPOCHS_Fashion
)
results.append(r1)

# Enriquecer resultados con información detallada
if BENCHMARK_AVAILABLE and device_info:
    enriched_r1 = enrich_results(
        base_results=r1,
        history=h1,
        model=model1,
        device_info=device_info,
        dataset_size_train=48000,  # Fashion MNIST train size
        dataset_size_test=10000
    )
    enriched_results.append(enriched_r1)
"""

# Dataset 2 — CIFAR-10 (escalado a 64x64)
print("\n--- Experimento 2: CIFAR-10 ---")
ds_train_2, ds_val_2, ds_test_2, num_classes_2 = load_and_preprocess_dataset(
    "cifar10", (64, 64)
)
r2, h2, model2 = train_and_measure(
    "CIFAR-10", (64, 64), ds_train_2, ds_val_2, ds_test_2, num_classes_2, EPOCHS_CIFAR10
)
results.append(r2)

# Enriquecer resultados con información detallada
if BENCHMARK_AVAILABLE and device_info:
    # No usar baseline_time cruzado entre datasets diferentes
    # El speedup solo tiene sentido comparando el MISMO dataset en CPU vs GPU
    enriched_r2 = enrich_results(
        base_results=r2,
        history=h2,
        model=model2,
        device_info=device_info,
        baseline_time=None,  # Speedup solo para comparar mismo dataset CPU vs GPU
        dataset_size_train=40000,  # CIFAR-10 train size (80% of 50000)
        dataset_size_test=10000
    )
    enriched_results.append(enriched_r2)

# =====================================
# GUARDAR Y PROCESAR RESULTADOS
# =====================================
print("\n" + "="*60)
print("GUARDANDO RESULTADOS")
print("="*60)

df = pd.DataFrame(results)
csv_path = os.path.join(CSV_DIR, "resultados_fase1.csv")
df.to_csv(csv_path, index=False)
print(f"\n✓ Resultados básicos guardados en: {csv_path}")

# Guardar resultados enriquecidos (si está disponible)
if BENCHMARK_AVAILABLE and enriched_results:
    csv_enriched_path = os.path.join(CSV_DIR, "resultados_fase1_detallado.csv")
    df_enriched = save_enriched_results(enriched_results, csv_enriched_path)
    print(f"✓ Resultados detallados guardados en: {csv_enriched_path}")
    print("\nTabla de Resultados Detallados:")
    # Mostrar solo columnas clave
    display_cols = ['dataset', 'device_type', 'accuracy', 'training_time', 
                    'time_per_epoch', 'samples_per_second', 'total_parameters']
    display_cols = [c for c in display_cols if c in df_enriched.columns]
    print(df_enriched[display_cols].to_string(index=False))
else:
    print("\nTabla de Resultados:")
    print(df.to_string(index=False))

# =====================================
# GENERAR GRÁFICOS
# =====================================
print("\n" + "="*60)
print("GENERANDO GRÁFICOS")
print("="*60 + "\n")

generate_result_graphs(df)

print("\n" + "="*60)
print("✓ EXPERIMENTO COMPLETADO")
print("="*60)
print("\nArchivos generados:")
print(f"  - CSV: {csv_path}")
print(f"  - Gráficos: {RESULTS_DIR}/")