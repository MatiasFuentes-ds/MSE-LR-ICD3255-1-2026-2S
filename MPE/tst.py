import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import utility as ut
import plot
from train import predict

"""
MPE/tst.py

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
    └── graficas_mejor_modelo_MPE.pdf
"""

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(OUT_DIR, "models")
METRICS_DIR = os.path.join(OUT_DIR, "metrics")
FIGURES_DIR = os.path.join(OUT_DIR, "figures")
REPORTS_DIR = os.path.join(OUT_DIR, "reports")

for carpeta in (MODELS_DIR, METRICS_DIR, FIGURES_DIR, REPORTS_DIR):
    os.makedirs(carpeta, exist_ok=True)

# Clase 0: señal normal, proveniente de 97.mat.
# Clase 1: falla en pista interior, proveniente de 105.mat.
CLASES = ["Normal", "Falla en pista interior"]

def tabla_fscores(y_real, y_predicho):
    """
    Construye una tabla con precision, recall y F1-score.

    Incluye una fila para cada clase y una fila macro, que representa
    el promedio simple de ambas clases.
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
    print("--- Iniciando Evaluación MPE ---")

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
            "Ejecuta train.py primero."
        )

    # Este control entrega un mensaje claro si train.py aún no guardó
    # la nueva configuración train / validation / test.
    if "validation_size" not in params:
        raise SystemExit(
            "parametros.json no contiene 'validation_size'. "
            "Ejecuta nuevamente MPE/train.py antes de ejecutar tst.py."
        )

    # Reconstruye los tres conjuntos con la misma configuración
    # usada durante el entrenamiento.
    Xtr, Xval, Xte, ytr, yval, yte = ut.build_dataset(
        params["window_size"],
        params["overlap"],
        params["validation_size"],
        params["test_size"],
    )

    # Se aplica la misma normalización guardada durante train.py.
    # Train y validation se transforman con la media y desviación
    # estándar calculadas originalmente solo con train.
    Xtr, Xval, _, _ = ut.standardize(
        Xtr,
        Xval,
        modelo["mean"],
        modelo["std"],
    )

    # Test usa exactamente la misma transformación que train.
    Xte = (Xte - modelo["mean"]) / modelo["std"]

    # Validation ya fue utilizada por train.py para elegir W y lambda.
    # Por eso, en este archivo se reportan métricas finales solo para
    # train y test, tal como solicita la tarea.
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

    # plot.py crea el PDF con parámetros, curva de convergencia,
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

    print(f"\n¡Evaluación completada! PDF generado en {REPORTS_DIR}")