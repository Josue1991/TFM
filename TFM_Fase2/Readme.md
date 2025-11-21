Es te proyecto contiene la experimentos para la FASE 2 (LSTM sobre UCI HAR y ECG5000). Está preparado para ejecutarse en entornos con CPU, 1 GPU o múltiples GPUs (laboratorio) usando `tf.distribute.MirroredStrategy`.


## Requisitos
- Python 3.9+ (recomendado)
- Crear entorno virtual (venv / conda)
- Instalar dependencias: `pip install -r requirements.txt`


## Estructura
Ver la raíz del repo. Cada fase tiene sus scripts y un sub-directorio `graphs/` donde se guardan las imágenes resultantes.


## Cómo ejecutar
1. Activar entorno virtual


```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

--Ejecucion
# Comandos para ejecutar 

python har_lstm.py # UCI HAR
python ecg_lstm.py # ECG5000
python fase2_report.py # (opcional) combinar resultados en PDF