# ✅ Proyecto Organizado para Git

## 📁 Estructura Final

```
TFM_Proyecto/
│
├── CODE/                          ← SUBIR A REPOSITORIO GIT
│   ├── TFM_Fase1/
│   │   ├── cnn_experimento.py
│   │   ├── cnn_modelo.py
│   │   └── requirements.txt
│   │
│   ├── TFM_Fase2/
│   │   ├── fase2_completo.py
│   │   ├── lstm_modelo.py
│   │   ├── ecg_lstm.py
│   │   ├── har_lstm.py
│   │   ├── fase2_report.py
│   │   └── requerimientos.txt
│   │
│   ├── utils/
│   │   └── MEJORAS_APRENDIZAJE.py
│   │
│   ├── config/
│   │   └── (configuración adicional)
│   │
│   ├── requirements.txt
│   ├── README.md
│   ├── .gitignore
│   └── [archivos Python]
│
├── DOCS/                          ← REFERENCIA LOCAL
│   ├── 20 archivos .md
│   ├── 4 archivos .txt
│   ├── 4 scripts .ps1
│   ├── 1 script .bat
│   ├── 2 scripts .sh
│   └── README.md
│
├── TFM_Fase1/                     ← ORIGINAL (mantener como backup)
├── TFM_Fase2/                     ← ORIGINAL (mantener como backup)
└── [otros archivos...]
```

---

## 🚀 Próximos Pasos

### 1. Verifica que CODE/ esté listo:

```bash
cd CODE
ls -la                          # Ver todos los archivos
ls -la TFM_Fase1/              # Ver Fase 1
ls -la TFM_Fase2/              # Ver Fase 2
ls -la utils/                   # Ver utils
```

### 2. Inicializa Git:

```bash
cd CODE
git init
git config user.email "tu@email.com"
git config user.name "Tu Nombre"
```

### 3. Agrega y confirma:

```bash
git add .
git status                      # Verificar qué se subirá

git commit -m "TFM: Código fuente optimizado

- Fase 1: CNN con Fashion MNIST + CIFAR-10
- Fase 2: LSTM con ECG5000 + UCI HAR
- 7 técnicas de optimización implementadas:
  * Early Stopping (patience=15)
  * Learning Rate Scheduler (ReduceLROnPlateau)
  * Batch Normalization en todas las capas
  * Dropout estratégico (0.3-0.5)
  * L2 Regularization (0.001)
  * Validación separada (80/20)
  * Adam Optimizer avanzado
- Epochs configurables
- GPU/CPU automático
"
```

### 4. Agrega repositorio remoto:

```bash
# En GitHub/GitLab/Bitbucket:
# 1. Crear repositorio nuevo (dejar VACÍO)
# 2. Copiar URL

# Luego:
git remote add origin https://github.com/usuario/tfm.git
git branch -M main
git push -u origin main
```

### 5. Verifica que se subió:

```bash
git log --oneline
git remote -v
```

---

## 📋 Checklist de Organización

- ✅ CODE/ creada con estructura correcta
- ✅ Código copiado a CODE/
- ✅ DOCS/ creada con documentación
- ✅ README.md en CODE/
- ✅ README.md en DOCS/
- ✅ .gitignore creado
- ✅ Estructura lista para Git

---

## 📊 Qué se Sube a Git (CODE/)

```
✅ cnn_experimento.py
✅ cnn_modelo.py
✅ fase2_completo.py
✅ lstm_modelo.py
✅ MEJORAS_APRENDIZAJE.py
✅ requirements.txt
✅ requerimientos.txt
✅ README.md
✅ .gitignore
```

## 🚫 Qué NO se Sube (DOCS/)

```
❌ CAMBIOS_FASE1_CIFAR10.md
❌ COLAB_*.md
❌ OPTIMIZACION_COMPLETA_APRENDIZAJE.md
❌ *.ps1 (scripts)
❌ *.bat (scripts)
❌ *.sh (scripts)
❌ RESUMEN_MEJORAS_IMPLEMENTADAS.md
❌ [otros .md .txt]
```

---

## 🔧 Línea de Comando para Copiar

Si necesitas copiar desde terminal:

```bash
# Desde PowerShell:
cd C:\Proyectos\TFM_Proyecto\CODE
git init
git add .
git commit -m "Initial commit: TFM code"

# O si prefieres agregar manualmente:
git add TFM_Fase1/
git add TFM_Fase2/
git add utils/
git add requirements.txt
git add .gitignore
git add README.md
git commit -m "Initial commit"
```

---

## 📚 Documentación

Toda la documentación está en `DOCS/`:
- Lee primero: `DOCS/INICIO_AQUI.txt`
- Para ejecutar: `DOCS/GUIA_RAPIDA_EJECUCION.md`
- Técnico: `DOCS/OPTIMIZACION_COMPLETA_APRENDIZAJE.md`
- Colab: `DOCS/COLAB_FASE1_FASE2_COMPLETO.md`

---

## ✅ Verificación Final

```bash
# En carpeta CODE/:
pwd                            # Debe mostrar: .../CODE
ls -la                         # Debe mostrar: TFM_Fase1/ TFM_Fase2/ utils/ README.md .gitignore
git status                     # Debe mostrar archivos sin stagear (antes de git add)
git log                        # Después de git commit
```

---

## 🎯 Resumen

| Elemento | Ubicación | Acción |
|----------|-----------|--------|
| Código Python | CODE/ | ✅ Subir a Git |
| Documentación | DOCS/ | 📖 Referencia local |
| Datos originales | TFM_Fase1/, TFM_Fase2/ | 💾 Mantener como backup |
| Este archivo | . | 📝 Guía |

---

## 💡 Tips

1. **Ejecutar desde CODE/**:
   ```bash
   cd CODE/TFM_Fase1
   python cnn_experimento.py
   ```

2. **Si hay cambios**:
   ```bash
   cd CODE
   git add .
   git commit -m "Descripción del cambio"
   git push
   ```

3. **Para colaboradores**:
   ```bash
   git clone https://github.com/usuario/tfm.git
   cd tfm
   pip install -r requirements.txt
   ```

---

**¡Listo para subir a Git!** 🚀
