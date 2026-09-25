import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import utility as ut

"""
MPE/train.py
│
├── Define regresión logística binaria
│   ├── Sigmoide
│   ├── Entropía cruzada ponderada
│   ├── Penalización L2 opcional
│   ├── Gradiente
│   └── Descenso del gradiente con momentum
│
├── Prueba tres tamaños de ventana
│   ├── W = 600
│   ├── W = 1200
│   └── W = 2400
│
├── Para cada tamaño de ventana, prueba cuatro penalizaciones
│   ├── λ = 0.0
│   ├── λ = 0.001
│   ├── λ = 0.01
│   └── λ = 0.1
│
├── Calcula F1 macro en train y test
│
├── Escoge un “mejor modelo”
│
└── Guarda modelo, parámetros, convergencia, coeficientes y resumen
"""

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(OUT_DIR, "models")
METRICS_DIR = os.path.join(OUT_DIR, "metrics")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

def sigmoid(z):
    """
    Calcula la función sigmoide para obtener probabilidades
    en un modelo de regresión logística.

    Args:
        z: Salida lineal del modelo, z = Xβ.

    Returns:
        Valores entre 0 y 1 correspondientes a las probabilidades
        estimadas de la clase positiva.

    Notes:
        Se utiliza np.clip para limitar z y evitar overflow numérico
        al calcular la exponencial.
    """
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))



def compute_loss(y, y_hat, weights, lambda_pen=0.0, sample_weights=None):
    """
    Calcula la función de pérdida de entropía cruzada con ponderación por muestra
    y penalización L2 sobre los pesos del modelo.

    Args:
        y: Valores reales de las clases.
        y_hat: Probabilidades predichas por el modelo.
        weights: Pesos del modelo utilizados en la penalización L2.
        lambda_pen: Coeficiente de regularización L2.
                    Si es 0.0, no se aplica regularización.
        sample_weights: Pesos asignados a cada muestra.
                        Si es None, todas las muestras tienen el mismo peso.

    Returns:
        Valor de la función de pérdida, compuesto por la entropía cruzada
        ponderada y la penalización L2.
    """
    m = len(y)
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
    
    if sample_weights is None:
        sample_weights = np.ones(m)

    loss = -(1 / m) * np.sum(sample_weights * (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))
    return loss + 0.5 * lambda_pen * np.sum(weights ** 2)       # El sesgo no se penaliza



def train_logistic_regression_mGD(X, y, epochs=1500, lr=0.5, beta=0.9, lambda_pen=0.0):
    """
    Entrena un modelo de regresión logística mediante descenso del gradiente
    con momentum (mGD) y ponderación de clases.

    Args:
        X: Matriz de características de entrada.
        y: Vector con las etiquetas de clase.
        epochs: Número de iteraciones de entrenamiento.
        lr: Tasa de aprendizaje utilizada para actualizar los parámetros.
        beta: Coeficiente de momentum utilizado para suavizar las actualizaciones.
        lambda_pen: Coeficiente de regularización L2 aplicado a los pesos.

    Returns:
        weights: Pesos aprendidos por el modelo.
        bias: Sesgo aprendido por el modelo.
        hist: Historial de la función de pérdida durante el entrenamiento.
    """
    m, n = X.shape
    weights, bias = np.zeros(n), 0.0
    v_w, v_b = np.zeros(n), 0.0

    # Pesos inversamente proporcionales a la cantidad de ventanas de cada clase.
    # La clase minoritaria recibe mayor peso para equilibrar su contribución
    #  a la función de pérdida durante el entrenamiento.
    
    w0 = m / (2.0 * np.sum(y == 0)); w1 = m / (2.0 * np.sum(y == 1))
    sw = np.where(y == 0, w0, w1)
    hist = []

    for _ in range(epochs):
        y_hat = sigmoid(X @ weights + bias)
        hist.append(compute_loss(y, y_hat, weights, lambda_pen, sw))
        error = sw * (y_hat - y)
        dw = (1 / m) * (X.T @ error) + lambda_pen * weights
        db = (1 / m) * np.sum(error)
        v_w = beta * v_w + (1 - beta) * dw          # momentum
        v_b = beta * v_b + (1 - beta) * db
        weights -= lr * v_w
        bias -= lr * v_b
    return weights, bias, np.array(hist)



def predict(X, weights, bias, threshold=0.5):
    """
    Genera predicciones binarias a partir de las probabilidades estimadas
    por un modelo de regresión logística.

    Args:
        X: Matriz de características de entrada.
        weights: Pesos aprendidos por el modelo.
        bias: Sesgo aprendido por el modelo.
        threshold: Umbral de decisión utilizado para convertir la probabilidad
            estimada en una clase binaria.

    Returns:
        Vector de predicciones binarias, donde 1 corresponde a una probabilidad
        mayor o igual al umbral y 0 a una probabilidad inferior.
    """
    return (sigmoid(X @ weights + bias) >= threshold).astype(int)



if __name__ == "__main__":
    print("--- Iniciando Entrenamiento MPE (solo 0 HP) ---")
    
    # ---------------- Parámetros a probar ----------------
    VENTANAS = [600, 1200, 2400]        # tamaños de ventana W
    LAMBDAS = [0.0, 1e-3, 1e-2, 1e-1]   # 0.0 = entropía cruzada normal; >0 = penalizada
    EPOCHS, LR, BETA = 1500, 0.5, 0.9
    
    SOLAPE = 0.5
    VALIDATION_SIZE = 0.2
    TEST_SIZE = 0.3

    modelos = []
    
    for W in VENTANAS:
        print(f"\nExtrayendo MPE con W={W} ...")
        Xtr, Xval, Xte, ytr, yval, yte = ut.build_dataset(
            W,
            SOLAPE,
            VALIDATION_SIZE,
            TEST_SIZE,
        )

        # La normalización se aprende solamente desde train.
        Xtr, Xval, mean, std = ut.standardize(Xtr, Xval)

        # Test usa exactamente la media y desviación estándar aprendidas desde train.
        Xte = (Xte - mean) / std

        print(
            f" ventanas train: {len(ytr)} "
            f"(Normal={np.sum(ytr == 0)}, Falla={np.sum(ytr == 1)}) | "
            f"validation: {len(yval)} "
            f"(Normal={np.sum(yval == 0)}, Falla={np.sum(yval == 1)}) | "
            f"test: {len(yte)} "
            f"(Normal={np.sum(yte == 0)}, Falla={np.sum(yte == 1)})"
        )
            
        for lam in LAMBDAS:
            w, b, hist = train_logistic_regression_mGD(
                Xtr,
                ytr,
                EPOCHS,
                LR,
                BETA,
                lam,
            )

            f1_tr = f1_score(
                ytr,
                predict(Xtr, w, b),
                average="macro",
            )

            f1_val = f1_score(
                yval,
                predict(Xval, w, b),
                average="macro",
            )

            modelos.append(
                dict(
                    W=W,
                    lam=lam,
                    w=w,
                    b=b,
                    hist=hist,
                    mean=mean,
                    std=std,
                    f1_tr=f1_tr,
                    f1_val=f1_val,
                )
            )

            print(
                f" lambda={lam:<6} "
                f"F1 macro train={f1_tr:.4f} | "
                f"validation={f1_val:.4f} | "
                f"pérdida final={hist[-1]:.4f}"
            )

    pd.DataFrame([
        {k: m[k] for k in ("W", "lam", "f1_tr", "f1_val")}
        for m in modelos
    ]).rename(
        columns={
            "W": "window_size",
            "lam": "lambda",
            "f1_tr": "F1_macro_train",
            "f1_val": "F1_macro_validation",
        }
    ).to_csv(
        os.path.join(METRICS_DIR, "resumen_modelos.csv"),
        index=False,
    )

    # ---------------- Mejor configuración (mayor F1 macro en validation) ----------------
    mejor = max(modelos, key=lambda m: m["f1_val"])
    # modelo "del otro tipo" con la misma ventana, para comparar las curvas de convergencia
    otros = [m for m in modelos if m["W"] == mejor["W"] and ((m["lam"] > 0) != (mejor["lam"] > 0))]
    otro = max(otros, key=lambda m: m["f1_val"])
    normal, penal = (mejor, otro) if mejor["lam"] == 0 else (otro, mejor)

    print(
        f"\nMEJOR CONFIGURACIÓN SEGÚN VALIDATION: "
        f"W={mejor['W']}, "
        f"lambda={mejor['lam']}, "
        f"F1 macro validation={mejor['f1_val']:.4f}"
    )

    params = {
        "metodo": "MPE",
        "window_size": mejor["W"],
        "overlap": SOLAPE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "momentum_beta": BETA,
        "lambda_penalty": mejor["lam"],
        "tipo_perdida": "entropia cruzada normal" if mejor["lam"] == 0 else "entropia cruzada penalizada (L2)",
        "optimizador": "mGD",
        "max_scale": ut.MAX_SCALE,
        "embedding_dim": ut.EMBED_DIM,
        "delay": ut.DELAY,
        "test_size": TEST_SIZE,
        "carga_motor_hp": 0,
        "fs_hz": ut.FS_OBJETIVO,
        "validation_size": VALIDATION_SIZE,
        "division_train_validation_test": (
                    "50% train / 20% validation / 30% test"
        ),
        "criterio_seleccion_modelo": "Mayor F1 macro en validation",
    }
    
    with open(os.path.join(MODELS_DIR, "parametros.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4, ensure_ascii=False)

    np.savez(
        os.path.join(MODELS_DIR, "mejor_modelo.npz"), 
        weights=mejor["w"], 
        bias=mejor["b"],
        mean=mejor["mean"], 
        std=mejor["std"]
    )

    pref = "mpe"
    
    pd.DataFrame({
        "Epoch": np.arange(1, EPOCHS + 1), 
        "Loss_Normal": normal["hist"],
        "Loss_Penalizada": penal["hist"]
    }).to_csv(
        os.path.join(METRICS_DIR, f"{pref}_convergencia_mGD.csv"), 
        index=False
    )
    
    pd.DataFrame({
        "Coeficiente": ["Bias"] + [f"Escala_{i+1}" for i in range(len(mejor["w"]))],
        "Normal": [normal["b"]] + list(normal["w"]),
        "Penalizada": [penal["b"]] + list(penal["w"]),
        "Mejor_modelo": [mejor["b"]] + list(mejor["w"])
    }).to_csv(
        os.path.join(METRICS_DIR, f"{pref}_coeficientes_regresion.csv"), 
        index=False
    )
    
    print("\n¡Entrenamiento listo!")
    print("Modelo y parámetros:", MODELS_DIR)
    print("Métricas de entrenamiento:", METRICS_DIR)
