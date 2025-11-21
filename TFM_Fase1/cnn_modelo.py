"""
cnn_modelo.py
Módulo que contiene las funciones para construir, entrenar y medir el rendimiento de modelos CNN.

MEJORAS IMPLEMENTADAS:
✅ Batch Normalization en cada capa
✅ Dropout para regularización (0.3-0.5)
✅ L2 Regularization
✅ GlobalAveragePooling en vez de Flatten
✅ Early Stopping para evitar overfitting
✅ Learning Rate Scheduler (ReduceLROnPlateau)
✅ Optimizador Adam avanzado
✅ Validación separada durante entrenamiento
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.regularizers import l2
import time
import numpy as np


def build_cnn_model(num_classes):
    """
    Construye una CNN mejorada con regularizaciones.
    
    MEJORAS:
    - Batch Normalization después de cada Conv2D
    - Dropout 0.3 en convolucionales, 0.4-0.5 en Dense
    - L2 regularization en todas las capas
    - GlobalAveragePooling en lugar de Flatten (mejor generalización)
    - Más capas densas para mayor capacidad
    
    Args:
        num_classes (int): Número de clases para la capa de salida
        
    Returns:
        model: Modelo Keras compilado y listo para entrenar
    """
    model = models.Sequential([
        # Input (sin Rescaling, ya normalizamos en augment_and_resize)
        layers.Input(shape=(None, None, 3)),
        
        # Bloque 1: 32 filtros
        layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.MaxPooling2D((2, 2)),
        
        # Bloque 2: 64 filtros
        layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.MaxPooling2D((2, 2)),
        
        # Bloque 3: 128 filtros
        layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.MaxPooling2D((2, 2)),
        
        # Global Average Pooling (mejor que Flatten)
        layers.GlobalAveragePooling2D(),
        
        # Capas Densas
        layers.Dense(256, activation='relu',
                    kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu',
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
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


def train_and_measure(dataset_name, img_size, ds_train, ds_val, ds_test, num_classes, epochs=10):
    """
    Entrena un modelo CNN con técnicas avanzadas y mide rendimiento.
    
    MEJORAS IMPLEMENTADAS:
    ✅ Early Stopping: Evita overfitting
    ✅ ReduceLROnPlateau: Ajusta learning rate dinámicamente
    ✅ Validación durante entrenamiento
    ✅ Mejor monitoreo de métricas
    
    Args:
        dataset_name (str): Nombre del dataset
        img_size (tuple): Tamaño de las imágenes
        ds_train: Dataset de entrenamiento
        ds_val: Dataset de validación
        ds_test: Dataset de prueba
        num_classes (int): Número de clases
        
    Returns:
        tuple: (resultados_dict, history)
    """
    print(f"\n{'='*60}")
    print(f"Entrenando con {dataset_name}")
    print(f"{'='*60}")
    print(f"Tamaño de imágenes: {img_size}")
    print(f"Número de clases: {num_classes}")
    print(f"Epochs configurados: {epochs}")  # Mostrar epochs

    # Crear modelo
    model = build_cnn_model(num_classes)
    print("\n[MODEL] Modelo creado:")
    model.summary()

    # Callbacks para optimización
    print("\n[CONFIG] Configurando callbacks...")
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    print("   [OK] Early Stopping: patience=15, monitor='val_loss'")
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    print("   [OK] ReduceLROnPlateau: factor=0.5, patience=5")
    
    # Entrenar
    print("\n[START] Iniciando entrenamiento...")
    print("   Epochs máximo: 100 (Early Stopping puede detener antes)")
    start_time = time.time()
    # Modifica el numero de Epochs para pruebas con cpu
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    end_time = time.time()
    training_time = end_time - start_time

    # Evaluación final
    print("\n[EVAL] Evaluando en conjunto de prueba...")
    loss, accuracy = model.evaluate(ds_test, verbose=0)
    print(f"   Accuracy final: {accuracy:.4f}")
    print(f"   Loss final: {loss:.4f}")
    print(f"   Tiempo de entrenamiento: {training_time:.2f} segundos")
    print(f"   Epochs ejecutados: {len(history.history['loss'])}")

    # Guardar resultados
    results = {
        "dataset": dataset_name,
        "accuracy": accuracy,
        "loss": loss,
        "training_time": training_time
    }

    return results, history
