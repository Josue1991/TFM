"""
colab_upload_helper.py
Helper para subir notebook a Google Colab automáticamente
"""

import webbrowser
import os

def get_colab_url():
    """Generar URL para Colab"""
    
    print("\n" + "="*70)
    print("HELPER: SUBIR A GOOGLE COLAB")
    print("="*70)
    
    print("\n📋 OPCIÓN 1: Crear notebook nuevo en Colab")
    print("   1. Ve a: https://colab.research.google.com/")
    print("   2. Nuevo notebook")
    print("   3. Copiar código desde: TFM_Colab_GPU.ipynb")
    print("   4. Runtime → Change Runtime Type → GPU")
    print("   5. Run All")
    
    print("\n📋 OPCIÓN 2: Importar desde GitHub")
    github_url = (
        "https://colab.research.google.com/github/"
        "[USERNAME]/[REPO]/blob/main/TFM_Colab_GPU.ipynb"
    )
    print(f"   URL: {github_url}")
    
    print("\n📋 OPCIÓN 3: Importar desde Google Drive")
    print("   1. Subir archivo a Google Drive")
    print("   2. Clic derecho → Open with → Google Colaboratory")
    
    print("\n" + "="*70)
    print("PASOS CRÍTICOS:")
    print("="*70)
    print("\n✅ Antes de ejecutar código:")
    print("   1. Runtime → Change Runtime Type")
    print("   2. Hardware accelerator: GPU (T4 o superior)")
    print("   3. Save")
    print("\n✅ Luego ejecutar celdas en orden")
    print("✅ Descargar resultados al final")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    get_colab_url()
    
    print("\n🔗 Abriendo Google Colab...")
    webbrowser.open("https://colab.research.google.com/")
