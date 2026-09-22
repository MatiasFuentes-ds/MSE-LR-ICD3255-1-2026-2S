# ============================================================
# tst.py (MDE) -> evalúa el modelo MDE seleccionado,
# guarda matrices de confusión y F-scores en CSV,
# y genera el PDF de entregables.
# ============================================================

import os
import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import utility as ut
import plot
from train import predict


OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(OUT_DIR, "models")
METRICS_DIR = os.path.join(OUT_DIR, "metrics")
FIGURES_DIR = os.path.join(OUT_DIR, "figures")
REPORTS_DIR = os.path.join(OUT_DIR, "reports")

for carpeta in (MODELS_DIR, METRICS_DIR, FIGURES_DIR, REPORTS_DIR):
    os.makedirs(carpeta, exist_ok=True)

# Clase 0: señal normal, archivo 97.mat.
# Clase 1: falla en pista interior, archivo 105.mat.
CLASES = ["Normal", "Falla en pista interior"]


"""
MDE/tst.py

├── Carga los parámetros del modelo seleccionado
│   └── Lee parametros.json
│
├── Carga el modelo entrenado
│   └── Lee mejor_modelo.npz
│
├── Reconstruye el dataset usando la misma configuración
│   ├── Train: primer 50 % de cada señal
│   ├── Validation: siguiente 20 % de cada señal
│   └── Test: último 30 % de cada señal
│
├── Aplica la normalización aprendida durante train.py
│   ├── Usa la misma media guardada
│   └── Usa la misma desviación estándar guardada
│
├── Genera predicciones
│   ├── Predicciones para train
│   └── Predicciones para test
│
├── Calcula métricas para train y test
│   ├── Matriz de confusión
│   ├── Precision
│   ├── Recall
│   ├── F1 por clase
│   └── F1 macro
│
├── Guarda resultados numéricos como CSV
│   ├── matriz_confusion_train.csv
│   ├── matriz_confusion_test.csv
│   ├── fscores_train.csv
│   └── fscores_test.csv
│
└── Genera el PDF final mediante plot.py
    └── graficas_mejor_modelo_MDE.pdf
"""


def tabla_fscores(y_real, y_predicho):
    """
    Construye una tabla con precision, recall y F1-score.

    Incluye una fila para cada clase y una fila macro, que representa
    el promedio simple entre ambas clases.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_real,
        y_predicho,
        labels=[0, 1],
        zero_division=0,
    )

    tabla = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "F1": f1,
        },
        index=CLASES,
    )

    tabla.loc["macro"] = tabla.mean()
    return tabla


if __name__ == "__main__":
    print("--- Iniciando Evaluación MDE ---")

    try:
        with open(
            os.path.join(MODELS_DIR, "parametros.json"),
            encoding="utf-8",
        ) as archivo:
            params = json.load(archivo)

        modelo = np.load(
            os.path.join(MODELS_DIR, "mejor_modelo.npz")
        )

    except FileNotFoundError:
        raise SystemExit(
            "Falta parametros.json o mejor_modelo.npz. "
            "Ejecuta MDE/train.py primero."
        )

    # Este control entrega un error claro si se intenta usar
    # un parametros.json antiguo, sin train / validation / test.
    if "validation_size" not in params:
        raise SystemExit(
            "parametros.json no contiene 'validation_size'. "
            "Ejecuta nuevamente MDE/train.py antes de ejecutar tst.py."
        )

    # Reconstruye train, validation y test usando los mismos
    # parámetros utilizados durante el entrenamiento.
    Xtr, Xval, Xte, ytr, yval, yte = ut.build_dataset(
        params["window_size"],
        params["overlap"],
        params["validation_size"],
        params["test_size"],
    )

    # Se usa la normalización aprendida durante el entrenamiento.
    # La media y desviación estándar vienen desde mejor_modelo.npz.
    Xtr, Xval, _, _ = ut.standardize(
        Xtr,
        Xval,
        modelo["mean"],
        modelo["std"],
    )

    # Test se transforma con la misma media y desviación estándar
    # calculadas desde train.
    Xte = (Xte - modelo["mean"]) / modelo["std"]

    # Validation ya fue usada por train.py para seleccionar
    # W y lambda. Por eso los entregables finales reportan
    # train y test, como solicita la tarea.
    yp_train = predict(
        Xtr,
        modelo["weights"],
        float(modelo["bias"]),
    )

    yp_test = predict(
        Xte,
        modelo["weights"],
        float(modelo["bias"]),
    )

    resultados = [
        ("train", ytr, yp_train),
        ("test", yte, yp_test),
    ]

    for nombre, y_real, y_predicho in resultados:
        matriz = confusion_matrix(
            y_real,
            y_predicho,
            labels=[0, 1],
        )

        pd.DataFrame(
            matriz,
            index=[f"real_{clase}" for clase in CLASES],
            columns=[f"pred_{clase}" for clase in CLASES],
        ).to_csv(
            os.path.join(
                METRICS_DIR,
                f"matriz_confusion_{nombre}.csv",
            )
        )

        fscores = tabla_fscores(y_real, y_predicho)

        fscores.to_csv(
            os.path.join(
                METRICS_DIR,
                f"fscores_{nombre}.csv",
            )
        )

        print(f"\n--- {nombre.upper()} ---")
        print(matriz)
        print(fscores.round(4))

    # plot.py genera el PDF con parámetros, curva de convergencia,
    # matrices de confusión y F-scores.
    plot.generar_pdf(
        ytr,
        yp_train,
        yte,
        yp_test,
        MODELS_DIR,
        METRICS_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
    )

    print("\n¡Evaluación completada!")
    print("CSV de métricas guardados en:", METRICS_DIR)
    print("Figuras PNG guardadas en:", FIGURES_DIR)
    print("PDF final guardado en:", REPORTS_DIR)