"""
build_tensorflow_gpu.ps1 (PowerShell script)

Script para compilar TensorFlow desde fuente con soporte GPU para CUDA 11.8.
Requiere: Bazel, MSVC, Python 3.11

Este script es para referencia educativa. En caso de compilar desde fuente:
- Requiere ~50GB de espacio
- Tarda 2-4 horas
- Necesita herramientas de desarrollo de Visual Studio
"""

Write-Host "============================================================"
Write-Host "COMPILACIÓN DE TENSORFLOW 2.13 CON GPU CUDA 11.8"
Write-Host "============================================================"

# Verificar si Bazel está instalado
Write-Host "`nVerificando Bazel..."
if (-not (Get-Command bazel -ErrorAction SilentlyContinue)) {
    Write-Host "✗ Bazel no instalado"
    Write-Host "Instala Bazel desde: https://bazel.build/install/windows"
    exit 1
}
Write-Host "✓ Bazel encontrado"

# Verificar CUDA
$cudaHome = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
if (-not (Test-Path $cudaHome)) {
    Write-Host "✗ CUDA 11.8 no encontrado en $cudaHome"
    exit 1
}
Write-Host "✓ CUDA 11.8 encontrado"

# Verificar cuDNN
$cudnnPath = "C:\tools\cuda"  # Cambiar según tu instalación
if (-not (Test-Path $cudnnPath)) {
    Write-Host "⚠ cuDNN no encontrado en $cudnnPath"
    Write-Host "Descargar desde: https://developer.nvidia.com/rdp/cudnn-archive"
}

# Configurar variables de entorno
Write-Host "`nConfigurando variables de entorno..."
$env:CUDA_HOME = $cudaHome
$env:PATH = "$cudaHome\bin;$env:PATH"
$env:BAZEL_VC = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC"

Write-Host "✓ Variables configuradas"

# Descargar TensorFlow 2.13
Write-Host "`nDescargando TensorFlow 2.13..."
if (-not (Test-Path "tensorflow")) {
    git clone --branch v2.13.0 https://github.com/tensorflow/tensorflow.git
}
cd tensorflow

# Configurar build
Write-Host "`nConfigurando build..."
python configure.py
# Responder a las preguntas interactivas:
# - GPU: Y
# - CUDA compute capability: 5.0 (para GeForce 940M)
# - CUDA version: 11.8
# - cuDNN version: 8.x

# Compilar
Write-Host "`nCompilando TensorFlow (esto puede tomar 2-4 horas)..."
bazel build //tensorflow/tools/pip_package:build_pip_package `
    --config=opt `
    --config=cuda `
    -c opt

# Crear wheel
Write-Host "`nCreando wheel..."
bazel-bin\tensorflow\tools\pip_package\build_pip_package C:\tmp\tensorflow_pkg

# Instalar
Write-Host "`nInstalando TensorFlow compilado..."
pip install C:\tmp\tensorflow_pkg\tensorflow-2.13.0-cp311-cp311-win_amd64.whl

Write-Host "`n============================================================"
Write-Host "Compilación completada"
Write-Host "============================================================"
