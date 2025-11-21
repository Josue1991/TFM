# 🚀 Instrucciones para Subir a Git

## Paso 1: Preparar (1 minuto)

```bash
cd c:\Proyectos\TFM_Proyecto\CODE
```

## Paso 2: Inicializar Git (1 minuto)

```bash
# Iniciar repositorio
git init

# Configurar usuario (solo la primera vez)
git config user.email "tu.email@example.com"
git config user.name "Tu Nombre"

# Verificar
git config --list
```

## Paso 3: Agregar Archivos (1 minuto)

```bash
# Agregar TODOS los archivos
git add .

# Verificar qué se agregó
git status
```

**Debe mostrar:**
- `TFM_Fase1/`
- `TFM_Fase2/`
- `utils/`
- `config/`
- `requirements.txt`
- `README.md`
- `.gitignore`

## Paso 4: Primer Commit (1 minuto)

```bash
git commit -m "TFM: Código fuente con optimizaciones

- Fase 1: CNN con Fashion MNIST + CIFAR-10
- Fase 2: LSTM con ECG5000 + UCI HAR

Optimizaciones implementadas:
* Early Stopping (patience=15)
* Learning Rate Scheduler (ReduceLROnPlateau)
* Batch Normalization en todas las capas
* Dropout estratégico (0.3-0.5)
* L2 Regularization (0.001)
* Validación separada (80/20)
* Adam Optimizer avanzado

Características:
* Epochs configurables
* Soporte GPU/CPU automático
* Gráficos automáticos
* Resultados en CSV
"
```

## Paso 5: Crear Repositorio Remoto (2 minutos)

### En GitHub:

1. Ve a https://github.com/new
2. Nombre: `tfm` (o lo que quieras)
3. Descripción: `TFM - Optimizaciones de Aprendizaje Profundo`
4. **IMPORTANTE:** Dejar VACÍO (no agregar README)
5. Click: "Create repository"
6. Copiar URL (HTTPS o SSH)

### En GitLab:

1. Ve a https://gitlab.com/projects/new
2. Proyecto name: `tfm`
3. Visibility: Public o Private
4. **IMPORTANTE:** Dejar VACÍO
5. Click: "Create project"
6. Copiar URL

### En Bitbucket:

1. Ve a https://bitbucket.org/repo/create
2. Repository name: `tfm`
3. **IMPORTANTE:** Dejar VACÍO
4. Click: "Create"
5. Copiar URL

## Paso 6: Conectar a Remoto (1 minuto)

```bash
# Reemplazar <URL> con la URL de tu repositorio
git remote add origin <URL>

# Ejemplo (GitHub HTTPS):
git remote add origin https://github.com/usuario/tfm.git

# Ejemplo (GitHub SSH):
git remote add origin git@github.com:usuario/tfm.git

# Verificar
git remote -v
```

## Paso 7: Subir a Remoto (1-5 minutos según conexión)

```bash
# Renombrar rama a "main" (estándar)
git branch -M main

# Subir
git push -u origin main
```

**Primer push:** Puede pedir credenciales
- GitHub: Token de acceso personal
- GitLab: Token o contraseña
- Bitbucket: Contraseña de aplicación

## ✅ Verificación

### En terminal:
```bash
git log --oneline           # Ver commits
git remote -v               # Ver conexión remota
git status                  # Debe decir "up to date"
```

### En el sitio web:
- Ve a tu repositorio
- Debe mostrar los archivos de CODE/
- Debe mostrar el commit que hiciste

---

## 🔄 Próximas Actualizaciones

Si realizas cambios en los archivos de `CODE/`:

```bash
# Ver qué cambió
git status

# Agregar cambios
git add .

# Confirmar cambio
git commit -m "Descripción del cambio"

# Subir
git push
```

---

## 🆘 Troubleshooting

### Error: "fatal: No commits yet"
```bash
# Asegúrate de haber hecho git commit primero
git commit -m "Initial commit"
```

### Error: "fatal: repository not found"
```bash
# Verifica que la URL sea correcta
git remote -v
git remote remove origin
git remote add origin <URL-CORRECTA>
```

### Error: "Authentication failed"
```bash
# Crea un token de acceso personal:
# GitHub: https://github.com/settings/tokens
# GitLab: https://gitlab.com/-/profile/personal_access_tokens

# Luego:
git push  # Te pedirá el token
```

### No se suben todos los archivos
```bash
# Asegúrate de:
git add .           # Punto importante (significa TODO)
git status          # Verifica qué se va a subir
git commit -m "..."
git push
```

---

## 📋 Comandos Útiles

```bash
# Ver estado actual
git status

# Ver historial
git log
git log --oneline
git log --graph --oneline --all

# Ver cambios desde último commit
git diff

# Deshacer cambios no confirmados
git checkout .

# Deshacer último commit (CUIDADO)
git reset --soft HEAD~1

# Ver rama actual
git branch

# Cambiar rama
git checkout -b feature/nueva-rama
```

---

## 🎯 Checklist Final

- ✅ Estoy en carpeta CODE/
- ✅ git init ejecutado
- ✅ git add . ejecutado
- ✅ git commit ejecutado
- ✅ Repositorio remoto creado (vacío)
- ✅ git remote add origin ejecutado
- ✅ git push origin main ejecutado
- ✅ Veo los archivos en GitHub/GitLab/Bitbucket

---

## 📞 Si Algo Falla

1. Verifica la carpeta actual: `pwd` (debe ser CODE/)
2. Verifica Git: `git --version`
3. Verifica remoto: `git remote -v`
4. Verifica status: `git status`
5. Lee el mensaje de error completo

---

**¡Éxito! 🚀**
