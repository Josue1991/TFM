#!/usr/bin/env python3
"""
fase2_completo.py - VERSIÓN MEJORADA Y CONFIGURABLE

Entrena LSTM en 2 datasets: ECG5000 + UCI HAR
Con Early Stopping, Learning Rate Scheduling y soporte GPU/CPU

MEJORAS:
✅ Epochs configurables
✅ Early Stopping automático
✅ ReduceLROnPlateau
✅ GPU/CPU automático
✅ Validación separada
✅ Mejor monitoreo
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.regularizers import l2

# Importar utilidades de benchmarking (si están disponibles)
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CODE', 'utils'))
    from benchmark_utils import get_device_info, enrich_results, print_device_summary, save_enriched_results
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False
    print("[INFO] benchmark_utils no disponible. Solo se guardarán resultados básicos.")

# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================

# CAMBIAR ESTOS VALORES SEGÚN TUS NECESIDADES
EPOCHS_ECG = 30     # Epochs para ECG5000 (coincide con Colab)
EPOCHS_HAR = 30     # Epochs para UCI HAR (coincide con Colab)
BATCH_SIZE = 32      # Tamaño de lote
DEVICE = 'auto'      # 'GPU', 'CPU', o 'auto'

# ============================================================
# CONFIGURAR DISPOSITIVO
# ============================================================
def setup_device():
    """Configurar GPU o CPU automáticamente."""
    print("\n" + "="*70)
    print("CONFIGURACIÓN DE DISPOSITIVO")
    print("="*70)
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus and DEVICE != 'CPU':
        print(f"✓ {len(gpus)} GPU(s) detectada(s)")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  ✓ {gpu.name} - Memory growth habilitado")
            except:
                pass
        device_type = 'GPU'
    else:
        print(f"✓ Usando CPU")
        device_type = 'CPU'
    
    print(f"→ Dispositivo: {device_type}")
    return device_type

device_type = setup_device()

# Capturar información detallada del sistema
if BENCHMARK_AVAILABLE:
    device_info = get_device_info()
    print_device_summary(device_info)
else:
    device_info = None

# ============================================================
# CONSTRUCCIÓN DE MODELO LSTM
# ============================================================
def build_lstm_model(input_shape, num_classes, model_name='LSTM'):
    """
    Construir modelo LSTM bidireccional con regularizaciones.
    
    MEJORAS IMPLEMENTADAS:
    ✅ Bidirectional LSTM
    ✅ Batch Normalization
    ✅ Dropout estratégico (0.3-0.4)
    ✅ L2 Regularization (0.001)
    ✅ GlobalAveragePooling
    ✅ Adam optimizador
    
    Args:
        input_shape: Tupla (timesteps, features)
        num_classes: Número de clases
        model_name: Nombre del modelo
    
    Returns:
        Modelo compilado
    """
    
    print(f"\n[MODEL] Construyendo {model_name}...")
    print(f"  Input shape: {input_shape}")
    print(f"  Clases: {num_classes}")
    
    model = models.Sequential([
        # Input
        layers.Input(shape=input_shape),
        
        # Normalización
        layers.BatchNormalization(),
        
        # LSTM Bidireccional 1
        layers.Bidirectional(
            layers.LSTM(
                128,
                return_sequences=True,
                activation='relu',
                kernel_regularizer=l2(0.001)
            )
        ),
        layers.Dropout(0.4),
        layers.BatchNormalization(),
        
        # LSTM Bidireccional 2
        layers.Bidirectional(
            layers.LSTM(
                64,
                activation='relu',
                kernel_regularizer=l2(0.001)
            )
        ),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        
        # Capas densas
        layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        layers.Dropout(0.3),
        
        layers.Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        
        # Output
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compilar con Adam optimizador
    optimizer = optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"  ✓ Modelo compilado")
    return model

# ============================================================
# ENTRENAMIENTO CON OPTIMIZACIONES
# ============================================================
def train_lstm_optimized(model, X_train, y_train, X_val, y_val, X_test, y_test,
                        dataset_name, epochs=50, batch_size=32):
    """
    Entrenar LSTM con Early Stopping y Learning Rate Scheduling.
    
    CALLBACKS IMPLEMENTADOS:
    ✅ Early Stopping (patience=15)
    ✅ ReduceLROnPlateau (factor=0.5, patience=5)
    ✅ ModelCheckpoint (mejor modelo)
    
    Args:
        model: Modelo compilado
        X_train, y_train: Datos entrenamiento
        X_val, y_val: Datos validación
        X_test, y_test: Datos prueba
        dataset_name: Nombre del dataset
        epochs: Número de épocas
        batch_size: Tamaño de lote
    
    Returns:
        Dict con resultados
    """
    
    print(f"\n{'='*70}")
    print(f"ENTRENAMIENTO: {dataset_name}")
    print(f"{'='*70}")
    print(f"Datos: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test")
    print(f"Epochs máximo: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Dispositivo: {device_type}")
    
    # Configurar callbacks
    print(f"\n[CALLBACKS] Configurando...")
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    print(f"  ✓ Early Stopping: patience=15, monitor='val_loss'")
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    print(f"  ✓ ReduceLROnPlateau: factor=0.5, patience=5")
    
    # Entrenar
    print(f"\n[TRAINING] Iniciando entrenamiento...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
        shuffle=True
    )
    
    training_time = time.time() - start_time
    
    # Evaluación
    print(f"\n[EVALUATION] Evaluando en conjunto de prueba...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n{'='*70}")
    print(f"RESULTADOS: {dataset_name}")
    print(f"{'='*70}")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Loss: {test_loss:.4f}")
    print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")
    print(f"Epochs ejecutados: {len(history.history['loss'])}")
    print(f"{'='*70}\n")
    
    results = {
        'dataset': dataset_name,
        'accuracy': float(test_accuracy),
        'loss': float(test_loss),
        'training_time': training_time,
        'epochs_executed': len(history.history['loss'])
    }
    
    return results, history, model

# ============================================================
# CARGAR Y PREPARAR DATASETS
# ============================================================
def load_ecg5000():
    """Cargar ECG5000 desde archivos locales."""
    print("\n[LOAD] Cargando ECG5000 desde archivos locales...")
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data_ecg')
    train_path = os.path.join(data_dir, 'ECG5000_TRAIN.txt')
    test_path = os.path.join(data_dir, 'ECG5000_TEST.txt')
    
    # Cargar datos (formato UCR: primera columna = label, resto = datos)
    train_data = np.loadtxt(train_path)
    test_data = np.loadtxt(test_path)
    
    X_train = train_data[:, 1:]  # Todas las columnas excepto la primera
    y_train = train_data[:, 0].astype(int) - 1  # Primera columna (labels empiezan en 1)
    X_test = test_data[:, 1:]
    y_test = test_data[:, 0].astype(int) - 1
    
    # Normalizar
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reshape para LSTM: (samples, timesteps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    print(f"  ✓ Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  ✓ Clases: {len(np.unique(y_train))}")
    
    # Crear split de validación desde entrenamiento
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, X_val, y_val

def load_uci_har():
    """Cargar UCI HAR desde señales inerciales (9 sensores x 128 timesteps)."""
    print("\n[LOAD] Cargando UCI HAR desde señales inerciales...")
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data_har')
    
    # 9 señales de sensores
    signals = [
        "total_acc_x", "total_acc_y", "total_acc_z",
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z"
    ]
    
    def load_signals(split):
        data = []
        for sig in signals:
            path = os.path.join(data_dir, split, "Inertial Signals", f"{sig}_{split}.txt")
            data.append(np.loadtxt(path))
        # Transponer: (samples, timesteps, features)
        return np.transpose(np.array(data), (1, 2, 0))
    
    # Cargar train y test
    X_train = load_signals("train")
    X_test = load_signals("test")
    
    # Cargar labels (1-6, convertir a 0-5)
    y_train = np.loadtxt(os.path.join(data_dir, "train", "y_train.txt")).astype(int) - 1
    y_test = np.loadtxt(os.path.join(data_dir, "test", "y_test.txt")).astype(int) - 1
    
    # Normalizar
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    X_train_reshaped = scaler.fit_transform(X_train_reshaped)
    X_test_reshaped = scaler.transform(X_test_reshaped)
    
    X_train = X_train_reshaped.reshape(X_train.shape)
    X_test = X_test_reshaped.reshape(X_test.shape)
    
    print(f"  ✓ Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  ✓ Clases: {len(np.unique(y_train))}")
    
    # Crear split de validación desde entrenamiento
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, X_val, y_val

# ============================================================
# MAIN
# ============================================================
def main():
    """Ejecutar entrenamiento de ambas fases."""
    
    print("\n" + "="*70)
    print("FASE 2: LSTM CON 2 DATASETS")
    print("="*70)
    print(f"ECG5000 epochs: {EPOCHS_ECG}")
    print(f"UCI HAR epochs: {EPOCHS_HAR}")
    print(f"Batch size: {BATCH_SIZE}")
    print("="*70)
    
    results_list = []
    enriched_results = []
    histories = {}
    
    # ========== EXPERIMENTO 1: ECG5000 ==========
    print("\n" + "█"*70)
    print("EXPERIMENTO 1: ECG5000")
    print("█"*70)
    
    X_train_ecg, X_test_ecg, y_train_ecg, y_test_ecg, X_val_ecg, y_val_ecg = load_ecg5000()
    
    model_ecg = build_lstm_model(
        input_shape=(X_train_ecg.shape[1], X_train_ecg.shape[2]),
        num_classes=5,
        model_name='LSTM-ECG5000'
    )
    
    results_ecg, history_ecg, model_ecg = train_lstm_optimized(
        model_ecg,
        X_train_ecg, y_train_ecg,
        X_val_ecg, y_val_ecg,
        X_test_ecg, y_test_ecg,
        dataset_name='ECG5000',
        epochs=EPOCHS_ECG,  # ← USA VARIABLE CONFIGURABLE
        batch_size=BATCH_SIZE
    )
    
    results_list.append(results_ecg)
    histories['ECG5000'] = history_ecg
    
    # Enriquecer resultados con información detallada
    if BENCHMARK_AVAILABLE and device_info:
        enriched_ecg = enrich_results(
            base_results=results_ecg,
            history=history_ecg,
            model=model_ecg,
            device_info=device_info,
            dataset_size_train=len(X_train_ecg),
            dataset_size_test=len(X_test_ecg)
        )
        enriched_results.append(enriched_ecg)
    
    # ========== EXPERIMENTO 2: UCI HAR ==========
    print("\n" + "█"*70)
    print("EXPERIMENTO 2: UCI HAR")
    print("█"*70)
    
    X_train_har, X_test_har, y_train_har, y_test_har, X_val_har, y_val_har = load_uci_har()
    
    model_har = build_lstm_model(
        input_shape=(X_train_har.shape[1], X_train_har.shape[2]),
        num_classes=6,
        model_name='LSTM-UCI HAR'
    )
    
    results_har, history_har, model_har = train_lstm_optimized(
        model_har,
        X_train_har, y_train_har,
        X_val_har, y_val_har,
        X_test_har, y_test_har,
        dataset_name='UCI HAR',
        epochs=EPOCHS_HAR,  # ← USA VARIABLE CONFIGURABLE
        batch_size=BATCH_SIZE
    )
    
    results_list.append(results_har)
    histories['UCI HAR'] = history_har
    
    # Enriquecer resultados con información detallada
    if BENCHMARK_AVAILABLE and device_info:
        # No usar baseline_time cruzado entre datasets diferentes
        # El speedup solo tiene sentido comparando el MISMO dataset en CPU vs GPU
        enriched_har = enrich_results(
            base_results=results_har,
            history=history_har,
            model=model_har,
            device_info=device_info,
            baseline_time=None,  # Speedup solo para comparar mismo dataset CPU vs GPU
            dataset_size_train=len(X_train_har),
            dataset_size_test=len(X_test_har)
        )
        enriched_results.append(enriched_har)
    
    # ========== GUARDAR RESULTADOS ==========
    print("\n" + "="*70)
    print("GUARDANDO RESULTADOS")
    print("="*70)
    
    # Crear DataFrame
    df_results = pd.DataFrame(results_list)
    
    # Crear directorio si no existe
    os.makedirs('csv_data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Guardar CSV
    csv_path = 'csv_data/fase2_completo.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\n✓ Resultados básicos guardados en: {csv_path}")
    
    # Guardar resultados enriquecidos (si está disponible)
    if BENCHMARK_AVAILABLE and enriched_results:
        csv_enriched_path = 'csv_data/fase2_completo_detallado.csv'
        df_enriched = save_enriched_results(enriched_results, csv_enriched_path)
        print(f"✓ Resultados detallados guardados en: {csv_enriched_path}")
        print("\nTabla de Resultados Detallados:")
        # Mostrar solo columnas clave
        display_cols = ['dataset', 'device_type', 'accuracy', 'training_time', 
                        'time_per_epoch', 'samples_per_second', 'total_parameters']
        display_cols = [c for c in display_cols if c in df_enriched.columns]
        print(df_enriched[display_cols].to_string(index=False))
    else:
        print(df_results)
    
    # Graficar historiales
    print(f"\n✓ Generando gráficas...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FASE 2: LSTM Training History', fontsize=16, fontweight='bold')
    
    # ECG5000 Accuracy
    axes[0, 0].plot(history_ecg.history['accuracy'], label='Train', linewidth=2)
    axes[0, 0].plot(history_ecg.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 0].set_title('ECG5000 - Accuracy', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # ECG5000 Loss
    axes[0, 1].plot(history_ecg.history['loss'], label='Train', linewidth=2)
    axes[0, 1].plot(history_ecg.history['val_loss'], label='Validation', linewidth=2)
    axes[0, 1].set_title('ECG5000 - Loss', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # UCI HAR Accuracy
    axes[1, 0].plot(history_har.history['accuracy'], label='Train', linewidth=2)
    axes[1, 0].plot(history_har.history['val_accuracy'], label='Validation', linewidth=2)
    axes[1, 0].set_title('UCI HAR - Accuracy', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # UCI HAR Loss
    axes[1, 1].plot(history_har.history['loss'], label='Train', linewidth=2)
    axes[1, 1].plot(history_har.history['val_loss'], label='Validation', linewidth=2)
    axes[1, 1].set_title('UCI HAR - Loss', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    png_path = 'results/fase2_lstm_training.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica guardada en: {png_path}")
    plt.close()
    
    print("\n" + "="*70)
    print("✓ FASE 2 COMPLETADA")
    print("="*70)

if __name__ == '__main__':
    main()
