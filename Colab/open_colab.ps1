"""
open_colab.ps1 (PowerShell Script)
Abre Google Colab e imprime instrucciones
"""

Write-Host "`n" + "="*70 -ForegroundColor Cyan
Write-Host "🚀 ABRIR TFM EN GOOGLE COLAB" -ForegroundColor Green
Write-Host "="*70 -ForegroundColor Cyan

Write-Host "`n📋 INSTRUCCIONES RÁPIDAS:`n" -ForegroundColor Yellow

Write-Host "1. Se abrirá Google Colab en tu navegador" -ForegroundColor White
Write-Host "2. Ve a: File → Upload Notebook" -ForegroundColor White
Write-Host "3. Selecciona: c:\Proyectos\TFM_Proyecto\TFM_Colab_GPU.ipynb" -ForegroundColor White
Write-Host "4. Espera a que cargue" -ForegroundColor White
Write-Host "5. Runtime → Change Runtime Type" -ForegroundColor White
Write-Host "6. Hardware accelerator → GPU" -ForegroundColor White
Write-Host "7. Save" -ForegroundColor White
Write-Host "8. Runtime → Run All" -ForegroundColor White

Write-Host "`n⏱️ Tiempo estimado: 5-10 minutos`n" -ForegroundColor Cyan

Write-Host "="*70 -ForegroundColor Cyan

# Abrir Colab
Write-Host "`n🔗 Abriendo Google Colab..." -ForegroundColor Green
Start-Process "https://colab.research.google.com/"

Write-Host "`n✅ Colab abierto en navegador" -ForegroundColor Green
Write-Host "📁 Ruta del notebook: c:\Proyectos\TFM_Proyecto\TFM_Colab_GPU.ipynb" -ForegroundColor White

Write-Host "`n" + "="*70 -ForegroundColor Cyan
