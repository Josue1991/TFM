"""
comparador_dispositivos.py
Script para generar gráficos comparativos entre ejecuciones en CPU, GPU y Colab.

USO:
1. Ejecuta los experimentos en diferentes dispositivos
2. Asegúrate de tener los archivos CSV detallados
3. Ejecuta este script para generar comparativas visuales

EJEMPLO:
    python comparador_dispositivos.py --fase1 --fase2
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse
from pathlib import Path

# Configurar estilo matplotlib (seaborn deshabilitado por problemas de instalación)
plt.style.use('default')
HAS_SEABORN = False

class DispositivoComparador:
    """Clase para comparar resultados entre diferentes dispositivos."""
    
    def __init__(self, output_dir='comparativas'):
        """
        Inicializar comparador.
        
        Args:
            output_dir (str): Directorio para guardar gráficos
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.data = []
        
    def cargar_csv(self, csv_path, etiqueta=None):
        """
        Cargar archivo CSV de resultados.
        
        Args:
            csv_path (str): Path al CSV
            etiqueta (str): Etiqueta opcional para identificar el experimento
        """
        if not os.path.exists(csv_path):
            print(f"⚠️  Advertencia: {csv_path} no encontrado")
            return False
        
        df = pd.read_csv(csv_path)
        if etiqueta:
            df['experiment_label'] = etiqueta
        
        self.data.append(df)
        print(f"✓ Cargado: {csv_path} ({len(df)} filas)")
        return True
    
    def consolidar_datos(self):
        """Consolidar todos los DataFrames cargados."""
        if not self.data:
            raise ValueError("No hay datos cargados. Usa cargar_csv() primero.")
        
        self.df_consolidado = pd.concat(self.data, ignore_index=True)
        print(f"\n✓ Consolidado: {len(self.df_consolidado)} registros totales")
        return self.df_consolidado
    
    def generar_comparativa_tiempos(self, filename='comparativa_tiempos.png'):
        """Gráfico comparativo de tiempos de entrenamiento."""
        df = self.df_consolidado
        
        if 'training_time' not in df.columns or 'device_type' not in df.columns:
            print("⚠️  Columnas necesarias no encontradas para comparativa de tiempos")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Por dataset y dispositivo
        pivot = df.pivot_table(values='training_time', 
                              index='dataset', 
                              columns='device_type', 
                              aggfunc='mean')
        
        pivot.plot(kind='bar', ax=axes[0], width=0.8)
        axes[0].set_title('Tiempo de Entrenamiento por Dataset y Dispositivo', 
                         fontweight='bold', fontsize=12)
        axes[0].set_xlabel('Dataset')
        axes[0].set_ylabel('Tiempo (segundos)')
        axes[0].legend(title='Dispositivo')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Speedup vs CPU
        if 'speedup_vs_cpu' in df.columns:
            df_speedup = df[df['speedup_vs_cpu'].notna()]
            if not df_speedup.empty:
                pivot_speedup = df_speedup.pivot_table(values='speedup_vs_cpu', 
                                                       index='dataset', 
                                                       columns='device_type')
                pivot_speedup.plot(kind='bar', ax=axes[1], width=0.8, color=['green', 'orange'])
                axes[1].set_title('Speedup vs CPU', fontweight='bold', fontsize=12)
                axes[1].set_xlabel('Dataset')
                axes[1].set_ylabel('Speedup (veces más rápido)')
                axes[1].axhline(y=1, color='red', linestyle='--', label='CPU baseline')
                axes[1].legend(title='Dispositivo')
                axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {output_path}")
        plt.close()
    
    def generar_comparativa_accuracy(self, filename='comparativa_accuracy.png'):
        """Gráfico comparativo de accuracy."""
        df = self.df_consolidado
        
        if 'accuracy' not in df.columns:
            print("⚠️  Columna 'accuracy' no encontrada")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        pivot = df.pivot_table(values='accuracy', 
                              index='dataset', 
                              columns='device_type', 
                              aggfunc='mean')
        
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Accuracy por Dataset y Dispositivo', 
                    fontweight='bold', fontsize=14)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.legend(title='Dispositivo')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {output_path}")
        plt.close()
    
    def generar_comparativa_eficiencia(self, filename='comparativa_eficiencia.png'):
        """Gráfico de eficiencia (samples/segundo)."""
        df = self.df_consolidado
        
        if 'samples_per_second' not in df.columns:
            print("⚠️  Columna 'samples_per_second' no encontrada")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        pivot = df.pivot_table(values='samples_per_second', 
                              index='dataset', 
                              columns='device_type', 
                              aggfunc='mean')
        
        pivot.plot(kind='bar', ax=ax, width=0.8, color=['steelblue', 'orange', 'green'])
        ax.set_title('Throughput: Muestras Procesadas por Segundo', 
                    fontweight='bold', fontsize=14)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Muestras / segundo')
        ax.legend(title='Dispositivo')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {output_path}")
        plt.close()
    
    def generar_tabla_resumen(self, filename='resumen_comparativo.csv'):
        """Genera tabla resumen con estadísticas clave."""
        df = self.df_consolidado
        
        # Columnas de interés
        cols = ['dataset', 'device_type', 'accuracy', 'training_time', 
                'time_per_epoch', 'samples_per_second', 'speedup_vs_cpu']
        cols = [c for c in cols if c in df.columns]
        
        # Agrupar por dataset y dispositivo
        resumen = df[cols].groupby(['dataset', 'device_type']).agg({
            'accuracy': 'mean',
            'training_time': 'mean',
            **{c: 'mean' for c in cols if c not in ['dataset', 'device_type', 'accuracy', 'training_time']}
        }).reset_index()
        
        # Redondear
        for col in resumen.columns:
            if resumen[col].dtype in ['float64', 'float32']:
                resumen[col] = resumen[col].round(4)
        
        output_path = os.path.join(self.output_dir, filename)
        resumen.to_csv(output_path, index=False)
        print(f"✓ Guardado: {output_path}")
        
        return resumen
    
    def generar_grafico_completo(self, filename='comparativa_completa.png'):
        """Gráfico consolidado con múltiples métricas."""
        df = self.df_consolidado
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comparativa Completa: CPU vs GPU vs Multi-GPU vs Colab', 
                    fontsize=16, fontweight='bold')
        
        # 1. Tiempo de entrenamiento
        if 'training_time' in df.columns:
            pivot = df.pivot_table(values='training_time', 
                                  index='dataset', 
                                  columns='device_type')
            pivot.plot(kind='bar', ax=axes[0, 0], width=0.8)
            axes[0, 0].set_title('Tiempo de Entrenamiento', fontweight='bold')
            axes[0, 0].set_ylabel('Segundos')
            axes[0, 0].legend(title='Dispositivo', fontsize=9)
            axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Accuracy
        if 'accuracy' in df.columns:
            pivot = df.pivot_table(values='accuracy', 
                                  index='dataset', 
                                  columns='device_type')
            pivot.plot(kind='bar', ax=axes[0, 1], width=0.8)
            axes[0, 1].set_title('Accuracy', fontweight='bold')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].legend(title='Dispositivo', fontsize=9)
            axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Samples por segundo
        if 'samples_per_second' in df.columns:
            pivot = df.pivot_table(values='samples_per_second', 
                                  index='dataset', 
                                  columns='device_type')
            pivot.plot(kind='bar', ax=axes[1, 0], width=0.8)
            axes[1, 0].set_title('Throughput (samples/sec)', fontweight='bold')
            axes[1, 0].set_ylabel('Muestras / segundo')
            axes[1, 0].legend(title='Dispositivo', fontsize=9)
            axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Speedup
        if 'speedup_vs_cpu' in df.columns:
            df_speedup = df[df['speedup_vs_cpu'].notna()]
            if not df_speedup.empty:
                pivot = df_speedup.pivot_table(values='speedup_vs_cpu', 
                                              index='dataset', 
                                              columns='device_type')
                pivot.plot(kind='bar', ax=axes[1, 1], width=0.8)
                axes[1, 1].set_title('Speedup vs CPU', fontweight='bold')
                axes[1, 1].set_ylabel('Factor de aceleración')
                axes[1, 1].axhline(y=1, color='red', linestyle='--', linewidth=1.5)
                axes[1, 1].legend(title='Dispositivo', fontsize=9)
                axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {output_path}")
        plt.close()


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Comparar resultados entre dispositivos')
    parser.add_argument('--fase1-dir', type=str, default='../TFM_Fase1/csv_data',
                       help='Directorio con CSVs de Fase 1')
    parser.add_argument('--fase2-dir', type=str, default='../TFM_Fase2/csv_data',
                       help='Directorio con CSVs de Fase 2')
    parser.add_argument('--output', type=str, default='comparativas',
                       help='Directorio de salida')
    parser.add_argument('--fase1', action='store_true', help='Procesar Fase 1')
    parser.add_argument('--fase2', action='store_true', help='Procesar Fase 2')
    
    args = parser.parse_args()
    
    print("="*70)
    print("COMPARADOR DE DISPOSITIVOS")
    print("="*70)
    
    comparador = DispositivoComparador(output_dir=args.output)
    
    # Cargar datos
    archivos_cargados = 0
    
    if args.fase1 or (not args.fase1 and not args.fase2):
        fase1_csv = os.path.join(args.fase1_dir, 'resultados_fase1_detallado.csv')
        if comparador.cargar_csv(fase1_csv, 'Fase1'):
            archivos_cargados += 1
    
    if args.fase2 or (not args.fase1 and not args.fase2):
        fase2_csv = os.path.join(args.fase2_dir, 'fase2_completo_detallado.csv')
        if comparador.cargar_csv(fase2_csv, 'Fase2'):
            archivos_cargados += 1
    
    if archivos_cargados == 0:
        print("\n❌ No se cargaron archivos. Verifica que existan:")
        print(f"   - {os.path.join(args.fase1_dir, 'resultados_fase1_detallado.csv')}")
        print(f"   - {os.path.join(args.fase2_dir, 'fase2_completo_detallado.csv')}")
        print("\n💡 Tip: Ejecuta primero los experimentos para generar los CSVs detallados.")
        return
    
    # Consolidar
    df = comparador.consolidar_datos()
    
    # Generar gráficos
    print("\n" + "="*70)
    print("GENERANDO GRÁFICOS COMPARATIVOS")
    print("="*70)
    
    comparador.generar_grafico_completo()
    comparador.generar_comparativa_tiempos()
    comparador.generar_comparativa_accuracy()
    comparador.generar_comparativa_eficiencia()
    
    # Tabla resumen
    print("\n" + "="*70)
    print("GENERANDO TABLA RESUMEN")
    print("="*70)
    resumen = comparador.generar_tabla_resumen()
    print("\n" + resumen.to_string(index=False))
    
    print("\n" + "="*70)
    print("✓ COMPARATIVA COMPLETADA")
    print("="*70)
    print(f"Todos los archivos guardados en: {args.output}/")


if __name__ == '__main__':
    main()
