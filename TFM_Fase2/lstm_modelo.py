"""
lstm_modelo.py
Módulo que contiene funciones para construir, entrenar y evaluar modelos LSTM.
Detecta automáticamente CPU/GPU disponible para entrenamiento.

MEJORAS IMPLEMENTADAS:
✅ LSTM bidireccionales (mejor captura de dependencias)
✅ Batch Normalization en capas Dense
✅ L2 Regularization en todas las capas
✅ Dropout estratégico (0.3-0.5)
✅ Early Stopping para evitar overfitting
✅ Learning Rate Scheduler (ReduceLROnPlateau)
✅ Optimizador Adam avanzado
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.regularizers import l2
import time

# =====================================
# DETECCIÓN AUTOMÁTICA DE DISPOSITIVO
# =====================================
def get_device_info():
    """Detecta y retorna información sobre dispositivos disponibles."""
    gpu_devices = tf.config.list_physical_devices('GPU')
    cpu_devices = tf.config.list_physical_devices('CPU')
    
    device_name = "GPU" if gpu_devices else "CPU"
    
    info = {
        "device": device_name,
        "gpu_count": len(gpu_devices),
        "cpu_count": len(cpu_devices),
        "gpu_list": [d.name for d in gpu_devices],
        "cpu_list": [d.name for d in cpu_devices]
    }
    return info


device_info = get_device_info()
print(f"✓ Dispositivo de cálculo: {device_info['device']}")
if device_info['gpu_count'] > 0:
    print(f"  GPUs disponibles: {device_info['gpu_count']}")
    for gpu in device_info['gpu_list']:
        print(f"    - {gpu}")
print(f"  CPUs disponibles: {device_info['cpu_count']}")


def build_lstm_model(input_shape, num_classes, lstm_units=64):
    """
    Construye un modelo LSTM avanzado con regularizaciones.
    
    MEJORAS:
    - LSTM Bidireccionales (64 → 32)
    - Batch Normalization en Dense layers
    - L2 regularization
    - Dropout 0.3-0.5
    
    Args:
        input_shape (tuple): Forma de entrada (timesteps, features)
        num_classes (int): Número de clases
        lstm_units (int): Unidades en primera LSTM
        
    Returns:
        model: Modelo Keras compilado
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # LSTM Layer 1 - Bidirectional
        layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True,
                       kernel_regularizer=l2(0.001))
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # LSTM Layer 2 - Bidirectional
        layers.Bidirectional(
            layers.LSTM(lstm_units // 2,
                       kernel_regularizer=l2(0.001))
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Dense Layer 1
        layers.Dense(128, activation='relu',
                    kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Dense Layer 2
        layers.Dense(64, activation='relu',
                    kernel_regularizer=l2(0.001)),
        layers.Dropout(0.3),
        
        # Output
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compilar con optimizador avanzado
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
    return model


def train_and_evaluate_lstm(model, x_train, y_train, x_val, y_val, x_test, y_test, 
                           epochs=50, batch_size=32):
    """
    Entrena un modelo LSTM con técnicas avanzadas.
    
    MEJORAS IMPLEMENTADAS:
    ✅ Early Stopping: Evita overfitting
    ✅ ReduceLROnPlateau: Ajusta learning rate dinámicamente
    ✅ Mejor monitoreo de métricas
    
    Args:
        model: Modelo Keras compilado
        x_train: Datos de entrenamiento
        y_train: Etiquetas de entrenamiento
        x_val: Datos de validación
        y_val: Etiquetas de validación
        x_test: Datos de prueba
        y_test: Etiquetas de prueba
        epochs (int): Máximo de épocas
        batch_size (int): Tamaño de batch
        
    Returns:
        tuple: (history, accuracy, loss, training_time)
    """
    print("\n" + "="*60)
    print("INICIANDO ENTRENAMIENTO LSTM MEJORADO")
    print("="*60)
    
    # Mostrar información del modelo
    print("\n📊 Arquitectura del modelo:")
    model.summary()
    
    # Callbacks para optimización
    print("\n🔧 Configurando callbacks...")
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    print("   ✓ Early Stopping: patience=15, monitor='val_loss'")
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    print("   ✓ ReduceLROnPlateau: factor=0.5, patience=5")
    
    # Entrenar
    print("\n🚀 Iniciando entrenamiento...")
    print(f"   Epochs máximo: {epochs} (Early Stopping puede detener antes)")
    print(f"   Batch size: {batch_size}")
    start_time = time.time()
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    end_time = time.time()
    training_time = end_time - start_time

    # Evaluación final en datos de prueba
    print("\n" + "="*60)
    print("EVALUACIÓN EN CONJUNTO DE PRUEBA")
    print("="*60)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    print(f"\n✓ Accuracy final: {accuracy:.4f}")
    print(f"✓ Loss final: {loss:.4f}")
    print(f"✓ Tiempo de entrenamiento: {training_time:.2f} segundos")
    print(f"✓ Epochs ejecutados: {len(history.history['loss'])}")

    return history, accuracy, loss, training_time
