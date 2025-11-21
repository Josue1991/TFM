import os
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

BASE_DIR = os.path.dirname(__file__)
GRAPH_DIR = os.path.join(BASE_DIR, "graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)

def load_results():
    har_path = os.path.join(BASE_DIR, "har_results.csv")
    ecg_path = os.path.join(BASE_DIR, "ecg_results.csv")

    if not os.path.exists(har_path) or not os.path.exists(ecg_path):
        raise FileNotFoundError("har_results.csv o ecg_results.csv no existen. Ejecuta los scripts primero.")

    har = pd.read_csv(har_path)
    ecg = pd.read_csv(ecg_path)

    return har, ecg

def generate_graphics(har, ecg):
    # Accuracy
    plt.figure()
    plt.bar(["HAR", "ECG5000"], [har["accuracy"][0], ecg["accuracy"][0]])
    plt.title("Comparación de Accuracy (HAR vs ECG5000)")
    plt.ylabel("Accuracy")
    acc_path = os.path.join(GRAPH_DIR, "fase2_accuracy.png")
    plt.savefig(acc_path)
    plt.close()

    # Loss
    plt.figure()
    plt.bar(["HAR", "ECG5000"], [har["loss"][0], ecg["loss"][0]])
    plt.title("Comparación de Loss (HAR vs ECG5000)")
    plt.ylabel("Loss")
    loss_path = os.path.join(GRAPH_DIR, "fase2_loss.png")
    plt.savefig(loss_path)
    plt.close()

    # Tiempo
    plt.figure()
    plt.bar(["HAR", "ECG5000"], [har["training_time"][0], ecg["training_time"][0]])
    plt.title("Tiempo de Entrenamiento (HAR vs ECG5000)")
    plt.ylabel("Segundos")
    time_path = os.path.join(GRAPH_DIR, "fase2_tiempo.png")
    plt.savefig(time_path)
    plt.close()

    return acc_path, loss_path, time_path


def generate_pdf(har, ecg, graphics):
    pdf_path = os.path.join(BASE_DIR, "fase2_informe.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)

    styles = getSampleStyleSheet()
    story = []

    title = Paragraph("<b>Informe Fase 2 – Modelos LSTM sobre Time Series</b>", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 20))

    story.append(Paragraph("<b>1. Resultados Obtenidos</b>", styles["Heading2"]))

    data = [
        ["Dataset", "Accuracy", "Loss", "Training Time (s)"],
        ["UCI HAR", har["]()]()
