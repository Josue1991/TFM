"""
har_lstm.py
UCI HAR (Human Activity Recognition) classification with LSTM.
Descarga automáticamente el dataset si no existe.
Genera: csv_data/har_results.csv, results/har_accuracy.png, results/har_loss.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from urllib.request import urlretrieve
import zipfile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lstm_modelo import build_lstm_model, train_and_evaluate_lstm, device_info

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
# CONFIGURAR DATASET UCI HAR
# =====================================
DATA_DIR = "data_har"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
ZIP_PATH = "UCI_HAR.zip"


def download_har_dataset():
    """Descarga el dataset UCI HAR si no existe."""
    if os.path.exists(DATA_DIR):
        print(f"✓ Dataset UCI HAR encontrado en {DATA_DIR}")
        return
    
    print("Descargando UCI HAR dataset...")
    try:
        urlretrieve(URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            z.extractall(".")
        os.rename("UCI HAR Dataset", DATA_DIR)
        os.remove(ZIP_PATH)
        print("✓ Dataset UCI HAR descargado y extraído.")
    except Exception as e:
        print(f"Error descargando dataset: {e}")
        raise


def load_har_processed(base_path):
    """Carga los datos procesados de UCI HAR (X_train.txt, etc.)."""
    x_train = np.loadtxt(os.path.join(base_path, "train", "X_train.txt"))
    y_train = np.loadtxt(os.path.join(base_path, "train", "y_train.txt")).astype(int) - 1
    x_test = np.loadtxt(os.path.join(base_path, "test", "X_test.txt"))
    y_test = np.loadtxt(os.path.join(base_path, "test", "y_test.txt")).astype(int) - 1
    return x_train, y_train, x_test, y_test


def load_inertial_signals(base_path):
    """Carga las señales inerciales raw (9 sensores x 128 timesteps)."""
    signals = ["total_acc_x", "total_acc_y", "total_acc_z",
               "body_acc_x", "body_acc_y", "body_acc_z",
               "body_gyro_x", "body_gyro_y", "body_gyro_z"]
    
    def load_signals_split(split):
        data = []
        for sig in signals:
            path = os.path.join(base_path, split, "Inertial Signals", f"{sig}_{split}.txt")
            data.append(np.loadtxt(path))
        arr = np.transpose(np.array(data), (1, 2, 0))
        return arr
    
    x_train_seq = load_signals_split("train")
    x_test_seq = load_signals_split("test")
    return x_train_seq, x_test_seq


def reshape_features_to_sequences(x, timesteps=128):
    """Convierte features planas a formato secuencial (fallback)."""
    n, features = x.shape
    n_timesteps = timesteps
    n_features = int(np.ceil(features / n_timesteps))
    padded = np.zeros((n, n_timesteps * n_features))
    padded[:, :features] = x
    return padded.reshape(n, n_timesteps, n_features)


def generate_har_graphs(history, results_dir=RESULTS_DIR):
    """Genera y guarda gráficos de entrenamiento."""
    # Gráfico de Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='train', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='val', linewidth=2)
    plt.title("UCI HAR: Accuracy por época", fontsize=14, fontweight='bold')
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "har_accuracy.png"), dpi=150)
    plt.close()
    print(f"✓ Gráfico guardado: {results_dir}/har_accuracy.png")

    # Gráfico de Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='train', linewidth=2)
    plt.plot(history.history['val_loss'], label='val', linewidth=2)
    plt.title("UCI HAR: Loss por época", fontsize=14, fontweight='bold')
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "har_loss.png"), dpi=150)
    plt.close()
    print(f"✓ Gráfico guardado: {results_dir}/har_loss.png")


# =====================================
# DESCARGAR DATASET
# =====================================
print("\n" + "="*60)
print("DATASET UCI HAR (Human Activity Recognition)")
print("="*60)
print(f"Dispositivo: {device_info['device']}")
download_har_dataset()

# =====================================
# CARGAR DATOS
# =====================================
print("\n" + "="*60)
print("CARGANDO DATOS")
print("="*60)

x_train, y_train, x_test, y_test = load_har_processed(DATA_DIR)
print(f"Datos procesados cargados - Train: {x_train.shape}, Test: {x_test.shape}")

# Intentar cargar señales inerciales (raw)
inertial_folder = os.path.join(DATA_DIR, "train", "Inertial Signals")
if os.path.exists(inertial_folder):
    print("Cargando señales inerciales raw...")
    x_train_seq, x_test_seq = load_inertial_signals(DATA_DIR)
    print(f"Señales inerciales cargadas - Train: {x_train_seq.shape}, Test: {x_test_seq.shape}")
    x_combined = np.concatenate([x_train_seq, x_test_seq], axis=0)
    y_combined = np.concatenate([y_train, y_test], axis=0)
else:
    print("No se encontraron señales inerciales; usando reshape fallback...")
    x_combined = np.vstack([x_train, x_test])
    y_combined = np.hstack([y_train, y_test])
    x_combined = reshape_features_to_sequences(x_combined)

# =====================================
# PREPROCESAMIENTO
# =====================================
print("\n" + "="*60)
print("PREPROCESAMIENTO DE DATOS")
print("="*60)

n_samples, timesteps, n_features = x_combined.shape
print(f"Forma antes de normalización: {x_combined.shape}")

# Normalizar
x_flat = x_combined.reshape(-1, n_features)
scaler = StandardScaler()
x_flat = scaler.fit_transform(x_flat)
x_combined = x_flat.reshape(n_samples, timesteps, n_features)

# Split train/val/test
n_train_len = len(x_train)
x_train_final = x_combined[:n_train_len]
y_train_final = y_combined[:n_train_len]
x_test_final = x_combined[n_train_len:]
y_test_final = y_combined[n_train_len:]

x_train_final, x_val, y_train_final, y_val = train_test_split(
    x_train_final, y_train_final, test_size=0.1, random_state=42, stratify=y_train_final
)

num_classes = len(np.unique(y_combined))
print(f"Número de clases (actividades): {num_classes}")
print(f"Forma de entrada: {(timesteps, n_features)}")

# =====================================
# CONSTRUIR Y ENTRENAR MODELO
# =====================================
print("\n" + "="*60)
print("MODELO LSTM BIDIRECCIONAL PARA UCI HAR")
print("="*60)

model = build_lstm_model((timesteps, n_features), num_classes, lstm_units=64)
history, accuracy, loss, training_time = train_and_evaluate_lstm(
    model, x_train_final, y_train_final, x_val, y_val, x_test_final, y_test_final, epochs=30
)

# =====================================
# GUARDAR RESULTADOS
# =====================================
print("\n" + "="*60)
print("GUARDANDO RESULTADOS")
print("="*60)

results = {
    "dataset": "UCI_HAR",
    "accuracy": float(accuracy),
    "loss": float(loss),
    "training_time": float(training_time)
}

csv_path = os.path.join(CSV_DIR, "har_results.csv")
pd.DataFrame([results]).to_csv(csv_path, index=False)
print(f"✓ Resultados guardados en: {csv_path}")
print(f"\n{pd.DataFrame([results]).to_string(index=False)}")

# =====================================
# GENERAR GRÁFICOS
# =====================================
print("\n" + "="*60)
print("GENERANDO GRÁFICOS")
print("="*60 + "\n")

generate_har_graphs(history)

print("\n" + "="*60)
print("✓ EXPERIMENTO UCI HAR COMPLETADO")
print("="*60)
