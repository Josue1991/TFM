"""
benchmark_utils.py
Utilidades para capturar métricas detalladas de rendimiento en experimentos de Deep Learning.
Diseñado para comparativas entre CPU, GPU Single, Multi-GPU y Colab.

COMPATIBILIDAD: No modifica resultados existentes, solo agrega información adicional.
"""

import tensorflow as tf
import platform
import os
from datetime import datetime
import numpy as np

# Importar psutil solo si está disponible
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def get_device_info():
    """
    Captura información detallada del dispositivo de ejecución.
    
    Returns:
        dict: Información completa del dispositivo
    """
    info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'platform': platform.system(),
        'cpu_model': platform.processor() or 'Unknown',
        'cpu_cores': psutil.cpu_count(logical=False) if PSUTIL_AVAILABLE else os.cpu_count() or 0,
        'cpu_threads': psutil.cpu_count(logical=True) if PSUTIL_AVAILABLE else os.cpu_count() or 0,
        'ram_total_gb': round(psutil.virtual_memory().total / (1024**3), 2) if PSUTIL_AVAILABLE else 0,
        'tensorflow_version': tf.__version__,
        'device_type': 'CPU',
        'device_name': 'CPU',
        'gpu_count': 0,
        'gpu_names': [],
        'gpu_memory_total_mb': 0,
        'cuda_version': 'N/A',
        'cudnn_version': 'N/A'
    }
    
    # Detectar GPUs
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        info['gpu_count'] = len(gpus)
        info['device_type'] = 'GPU_Multi' if len(gpus) > 1 else 'GPU_Single'
        
        gpu_names = []
        for gpu in gpus:
            gpu_names.append(gpu.name)
        info['gpu_names'] = gpu_names
        
        # Obtener nombre de la primera GPU
        try:
            gpu_devices = tf.config.experimental.list_physical_devices('GPU')
            if gpu_devices:
                info['device_name'] = gpu_devices[0].name
        except:
            info['device_name'] = 'GPU'
        
        # Intentar obtener información de CUDA
        try:
            from tensorflow.python.platform import build_info
            info['cuda_version'] = build_info.build_info.get('cuda_version', 'N/A')
            info['cudnn_version'] = build_info.build_info.get('cudnn_version', 'N/A')
        except:
            pass
        
        # Intentar obtener memoria GPU
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                memories = [int(x.strip()) for x in result.stdout.strip().split('\n')]
                info['gpu_memory_total_mb'] = sum(memories)
        except:
            pass
    
    # Detectar si estamos en Colab
    try:
        import google.colab
        info['device_type'] = 'Colab_GPU' if gpus else 'Colab_CPU'
        info['platform'] = 'Colab'
    except:
        pass
    
    return info


def get_gpu_memory_usage():
    """
    Obtiene el uso actual de memoria GPU.
    
    Returns:
        int: Memoria GPU usada en MB, o 0 si no hay GPU
    """
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            memories = [int(x.strip()) for x in result.stdout.strip().split('\n')]
            return sum(memories)
    except:
        pass
    return 0


def calculate_model_metrics(model):
    """
    Calcula métricas detalladas del modelo.
    
    Args:
        model: Modelo Keras/TensorFlow
        
    Returns:
        dict: Métricas del modelo
    """
    total_params = model.count_params()
    
    # Contar parámetros entrenables
    trainable_params = sum([tf.reduce_prod(var.shape).numpy() 
                           for var in model.trainable_variables])
    
    metrics = {
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'non_trainable_parameters': int(total_params - trainable_params),
        'layers_count': len(model.layers)
    }
    
    return metrics


def enrich_results(base_results, history, model, device_info, baseline_time=None, 
                   dataset_size_train=0, dataset_size_test=0):
    """
    Enriquece los resultados básicos con información detallada.
    
    IMPORTANTE: No modifica base_results, retorna un nuevo diccionario extendido.
    
    Args:
        base_results (dict): Resultados básicos (accuracy, loss, training_time)
        history: History object de Keras
        model: Modelo entrenado
        device_info (dict): Información del dispositivo
        baseline_time (float): Tiempo de referencia CPU para calcular speedup
        dataset_size_train (int): Tamaño del dataset de entrenamiento
        dataset_size_test (int): Tamaño del dataset de test
        
    Returns:
        dict: Resultados enriquecidos con toda la información adicional
    """
    # Copiar resultados base
    enriched = base_results.copy()
    
    # Agregar información de dispositivo
    enriched.update({
        'timestamp': device_info['timestamp'],
        'device_type': device_info['device_type'],
        'device_name': device_info['device_name'],
        'gpu_count': device_info['gpu_count'],
        'cpu_cores': device_info['cpu_cores'],
        'tensorflow_version': device_info['tensorflow_version'],
    })
    
    # Métricas del modelo
    model_metrics = calculate_model_metrics(model)
    enriched.update(model_metrics)
    
    # Métricas de entrenamiento del history
    if history and hasattr(history, 'history'):
        epochs_executed = len(history.history.get('loss', []))
        enriched['epochs_executed'] = epochs_executed
        
        # Mejor epoch
        if 'val_accuracy' in history.history:
            best_epoch = np.argmax(history.history['val_accuracy']) + 1
            enriched['best_epoch'] = int(best_epoch)
            enriched['best_val_accuracy'] = float(max(history.history['val_accuracy']))
            enriched['best_val_loss'] = float(min(history.history['val_loss']))
        
        # Learning rate final (si está disponible)
        if 'lr' in history.history:
            enriched['final_learning_rate'] = float(history.history['lr'][-1])
    
    # Métricas temporales
    training_time = base_results.get('training_time', 0)
    if training_time > 0 and epochs_executed > 0:
        enriched['time_per_epoch'] = round(training_time / epochs_executed, 2)
    
    # Dataset sizes
    if dataset_size_train > 0:
        enriched['dataset_size_train'] = dataset_size_train
        enriched['samples_per_second'] = round(dataset_size_train / training_time, 2)
    
    if dataset_size_test > 0:
        enriched['dataset_size_test'] = dataset_size_test
    
    # Speedup vs CPU
    if baseline_time and baseline_time > 0 and training_time > 0:
        enriched['speedup_vs_cpu'] = round(baseline_time / training_time, 2)
    
    # Memoria GPU
    gpu_memory_used = get_gpu_memory_usage()
    if gpu_memory_used > 0:
        enriched['gpu_memory_used_mb'] = gpu_memory_used
        enriched['gpu_memory_total_mb'] = device_info['gpu_memory_total_mb']
    
    # Métricas de eficiencia calculadas
    if 'gpu_count' in enriched and enriched['gpu_count'] > 0:
        if 'samples_per_second' in enriched:
            enriched['samples_per_second_per_gpu'] = round(
                enriched['samples_per_second'] / enriched['gpu_count'], 2
            )
    
    # Cost-benefit: accuracy por segundo
    if training_time > 0:
        enriched['accuracy_per_second'] = round(
            base_results.get('accuracy', 0) / training_time, 6
        )
    
    return enriched


def print_device_summary(device_info):
    """
    Imprime un resumen formateado de la información del dispositivo.
    
    Args:
        device_info (dict): Información del dispositivo
    """
    print("\n" + "="*70)
    print("INFORMACIÓN DEL SISTEMA")
    print("="*70)
    print(f"Timestamp: {device_info['timestamp']}")
    print(f"Platform: {device_info['platform']}")
    print(f"CPU: {device_info['cpu_model']}")
    print(f"CPU Cores: {device_info['cpu_cores']} ({device_info['cpu_threads']} threads)")
    print(f"RAM Total: {device_info['ram_total_gb']} GB")
    print(f"TensorFlow: {device_info['tensorflow_version']}")
    print(f"\nDispositivo de Cómputo: {device_info['device_type']}")
    
    if device_info['gpu_count'] > 0:
        print(f"GPUs Detectadas: {device_info['gpu_count']}")
        for i, gpu_name in enumerate(device_info['gpu_names'], 1):
            print(f"  GPU {i}: {gpu_name}")
        if device_info['gpu_memory_total_mb'] > 0:
            print(f"Memoria GPU Total: {device_info['gpu_memory_total_mb']} MB")
        if device_info['cuda_version'] != 'N/A':
            print(f"CUDA Version: {device_info['cuda_version']}")
            print(f"cuDNN Version: {device_info['cudnn_version']}")
    else:
        print("Modo: CPU")
    
    print("="*70)


def save_enriched_results(results_list, output_path):
    """
    Guarda resultados enriquecidos en CSV con todas las columnas ordenadas.
    
    Args:
        results_list (list): Lista de diccionarios con resultados
        output_path (str): Path del archivo CSV
    """
    import pandas as pd
    
    # Orden preferido de columnas para facilitar lectura
    preferred_columns = [
        'timestamp', 'dataset', 'device_type', 'device_name', 'gpu_count',
        'accuracy', 'loss', 'training_time', 'time_per_epoch', 'epochs_executed',
        'best_epoch', 'best_val_accuracy', 'best_val_loss',
        'samples_per_second', 'speedup_vs_cpu', 'accuracy_per_second',
        'total_parameters', 'trainable_parameters', 'layers_count',
        'dataset_size_train', 'dataset_size_test',
        'gpu_memory_used_mb', 'gpu_memory_total_mb',
        'cpu_cores', 'tensorflow_version'
    ]
    
    df = pd.DataFrame(results_list)
    
    # Reordenar columnas (mantener todas, incluso las no listadas)
    existing_preferred = [col for col in preferred_columns if col in df.columns]
    other_columns = [col for col in df.columns if col not in preferred_columns]
    ordered_columns = existing_preferred + other_columns
    
    df = df[ordered_columns]
    df.to_csv(output_path, index=False)
    
    return df
