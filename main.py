import subprocess
import sys

"""
main.py - script de orquestación para ejecutar los scripts de entrenamiento y evaluación de MPE y MDE.
│
├── Ejecuta MPE/train.py
│   └── Debe entrenar, buscar/escoger mejor modelo y guardar resultados
│
├── Ejecuta MPE/tst.py
│   └── Debe cargar el modelo, evaluar y generar entregables de MPE
│
├── Ejecuta MDE/train.py
│   └── Debe entrenar, buscar/escoger mejor modelo y guardar resultados
│
├── Ejecuta MDE/tst.py
│   └── Debe cargar el modelo, evaluar y generar entregables de MDE
│
└── Muestra mensaje final de éxito
"""

def ejecutar_script(ruta):
    """Ejecuta un script de Python y muestra su salida en la terminal."""

    print(f"\n{'='*60}\n>>> Ejecutando: {ruta}\n{'='*60}")
    
    proceso = subprocess.run([sys.executable, ruta])

    if proceso.returncode != 0:
        print(f"\n[ERROR] Falló la ejecución de {ruta}. Deteniendo el pipeline.")
        sys.exit(1)


if __name__ == "__main__":
    print("INICIANDO GENERACIÓN DE ENTREGABLES - TAREA #1 (solo carga 0 HP)")

    # 1. Entropía Multi-escala de Permutación (MPE)
    ejecutar_script("MPE/train.py")   # entrena, elige el mejor modelo, guarda .npz, .json y CSV
    ejecutar_script("MPE/tst.py")     # evalúa train/test, guarda CSV y genera el PDF

    # 2. Entropía Multi-escala de Dispersión (MDE)
    ejecutar_script("MDE/train.py")
    ejecutar_script("MDE/tst.py")

    print("\n" + "=" * 60)
    print("¡PIPELINE COMPLETADO! Entregables en MPE/ y MDE/ (PDF, CSV, JSON, NPZ).")
    print("=" * 60)
