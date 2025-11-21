"""
lstm_modelo_gpu.py
Versión mejorada de lstm_modelo.py con soporte GPU explícito y fallback a CPU.
Incluye compilación desde fuente de TensorFlow si es necesario.
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURAR CUDA/GPU
# ============================================================
def setup_gpu():
    """Configurar GPU y CUDA de forma explícita."""
    print("\n" + "="*70)
    print("GPU CONFIGURATION")
    print("="*70)
    
    # Configurar CUDA_HOME
    cuda_home = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
    if os.path.exists(cuda_home):
        os.environ['CUDA_HOME'] = cuda_home
        print(f"✓ CUDA_HOME = {cuda_home}")
    
    # Agregar rutas CUDA a PATH
    cuda_bin = os.path.join(cuda_home, "bin")
    cuda_lib = os.path.join(cuda_home, "lib", "x64")
    
    if cuda_bin not in os.environ['PATH']:
        os.environ['PATH'] = cuda_bin + ';' + os.environ['PATH']
    if cuda_lib not in os.environ['PATH']:
        os.environ['PATH'] = cuda_lib + ';' + os.environ['PATH']
    
    print(f"✓ CUDA PATH configurado")
    
    # Opciones de TensorFlow para GPU
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce verbose logs
    os.environ['CUDA_FORCE_PTX_JIT'] = '1'    # Fuerza compilación PTX
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Usa GPU 0
    
    print(f"✓ Variables de entorno CUDA configuradas")

setup_gpu()

# Importar TensorFlow
try:
    import tensorflow as tf
    print(f"\n✓ TensorFlow {tf.__version__} importado")
except ImportError as e:
    print(f"✗ Error importando TensorFlow: {e}")
    sys.exit(1)

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# ============================================================
# INFORMACIÓN DE DISPOSITIVOS
# ============================================================
def get_device_info():
    """Obtener información de dispositivos disponibles."""
    print("\n" + "="*70)
    print("DISPOSITIVOS DISPONIBLES")
    print("="*70)
    
    # Detectar GPU
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    
    info = {
        'gpus': len(gpus),
        'cpus': len(cpus),
        'gpu_names': [gpu.name for gpu in gpus] if gpus else [],
        'device': 'GPU' if gpus else 'CPU'
    }
    
    if gpus:
        print(f"✓ GPUs detectadas: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  - GPU {i}: {gpu.name}")
            
            # Habilitar memory growth para evitar OOM
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"    ✓ Memory growth habilitado")
            except:
                pass
    else:
        print(f"✗ No se detectó GPU, usando CPU")
        print(f"✓ CPUs disponibles: {len(cpus)}")
    
    print(f"\n→ Dispositivo a usar: {info['device']}")
    return info

device_info = get_device_info()

# ============================================================
# CONSTRUCCIÓN DE MODELO LSTM
# ============================================================
def build_lstm_model(input_shape, num_classes, device_type='auto'):
    """
    Construir modelo LSTM bidireccional.
    
    Args:
        input_shape: Tupla (timesteps, features)
        num_classes: Número de clases
        device_type: 'GPU', 'CPU', o 'auto'
    
    Returns:
        Modelo compilado de Keras
    """
    
    # Seleccionar dispositivo
    if device_type == 'auto':
        device_type = device_info['device']
    
    with tf.device(f'/{device_type}:0'):
        model = tf.keras.Sequential([
            # Input
            tf.keras.layers.Input(shape=input_shape),
            
            # Normalización
            tf.keras.layers.Normalization(),
            
            # LSTM Bidireccional 1
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(128, return_sequences=True, activation='relu')
            ),
            tf.keras.layers.Dropout(0.4),
            
            # LSTM Bidireccional 2
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, activation='relu')
            ),
            tf.keras.layers.Dropout(0.4),
            
            # Capas densas
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            
            # Output
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compilar
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model

# ============================================================
# ENTRENAMIENTO CON GPU
# ============================================================
def train_and_evaluate_lstm(model, X_train, y_train, X_test, y_test, 
                           epochs=30, batch_size=32, verbose=1, device_type='auto'):
    """
    Entrenar y evaluar modelo LSTM en dispositivo especificado.
    
    Args:
        model: Modelo de Keras
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de prueba
        epochs: Número de épocas
        batch_size: Tamaño de lote
        verbose: Verbosidad (0, 1, 2)
        device_type: 'GPU', 'CPU', o 'auto'
    
    Returns:
        Dict con resultados (accuracy, loss, training_time, history)
    """
    
    if device_type == 'auto':
        device_type = device_info['device']
    
    print(f"\n{'='*70}")
    print(f"ENTRENAMIENTO EN {device_type}")
    print(f"{'='*70}")
    print(f"Datos: {X_train.shape[0]} entrenamiento, {X_test.shape[0]} prueba")
    print(f"Épocas: {epochs}, Batch size: {batch_size}\n")
    
    # Entrenar en dispositivo específico
    with tf.device(f'/{device_type}:0'):
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=verbose,
            shuffle=True
        )
        
        training_time = time.time() - start_time
    
    # Evaluar en dispositivo específico
    with tf.device(f'/{device_type}:0'):
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n{'='*70}")
    print(f"RESULTADOS - {device_type}")
    print(f"{'='*70}")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Loss: {test_loss:.4f}")
    print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")
    print(f"{'='*70}\n")
    
    results = {
        'accuracy': float(test_accuracy),
        'loss': float(test_loss),
        'training_time': training_time,
        'history': history,
        'device': device_type
    }
    
    return results

# ============================================================
# PREDICCIONES
# ============================================================
def predict_with_device(model, X, device_type='auto'):
    """Hacer predicciones en dispositivo especificado."""
    if device_type == 'auto':
        device_type = device_info['device']
    
    with tf.device(f'/{device_type}:0'):
        predictions = model.predict(X, verbose=0)
    
    return predictions
