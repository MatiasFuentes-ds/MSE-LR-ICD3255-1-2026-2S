# ============================================================
# plot.py (MPE) -> genera el PDF con las gráficas del mejor modelo
# ============================================================
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from matplotlib.backends.backend_pdf import PdfPages

CLASES = ["Normal", "Falla en pista interior"]
DESCRIPCIONES = {
    "metodo": "Tipo de entropía multi-escala usada como característica",
    "window_size": "Tamaño de la ventana (nº de muestras por segmento de señal)",
    "overlap": "Traslape entre ventanas consecutivas (0.5 = 50%)",
    "epochs": "Número de iteraciones (épocas) del mGD",
    "learning_rate": "Tasa de aprendizaje del mGD",
    "momentum_beta": "Factor de momentum del mGD",
    "lambda_penalty": "Coeficiente de penalización L2 (0 = entropía cruzada normal)",
    "tipo_perdida": "Función de pérdida del mejor modelo",
    "optimizador": "Algoritmo de optimización (descenso del gradiente con momentum)",
    "max_scale": "Número de escalas de la entropía multi-escala (1..max_scale)",
    "embedding_dim": "Dimensión de embedding m (largo de cada patrón de orden)",
    "delay": "Retardo entre puntos del patrón",
    "test_size": "Proporción de la señal usada como test",
    "validation_size": (
        "Proporción de la señal usada para seleccionar los hiperparámetros"
    ),
    "division_train_validation_test": (
        "Separación temporal de cada señal en train, validation y test"
    ),
    "criterio_seleccion_modelo": (
        "Regla usada para escoger la mejor configuración del modelo"
    ),
    "carga_motor_hp": "Carga del motor usada (HP)",
    "fs_hz": "Frecuencia de muestreo de trabajo (Hz)",
}

def generar_pdf(
    y_train,
    yp_train,
    y_test,
    yp_test,
    models_dir,
    metrics_dir,
    figures_dir,
    reports_dir,
):
    with open(os.path.join(models_dir, "parametros.json"), encoding="utf-8") as f:
        params = json.load(f)
    pref = "mpe"
    pdf_path = os.path.join(reports_dir, "graficas_mejor_modelo_MPE.pdf")
    with PdfPages(pdf_path) as pdf:
        # Página 1: lista de parámetros (nombre, valor y descripción)
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle("Multiscale Permutation Entropy: parámetros del mejor modelo", fontsize=13)
        y0 = 0.93
        for k, v in params.items():
            fig.text(0.06, y0, f"{k} = {v}", fontsize=10, weight="bold")
            fig.text(0.06, y0 - 0.022, DESCRIPCIONES.get(k, ""), fontsize=8.5, color="dimgray")
            y0 -= 0.048
        fig.tight_layout()
        pdf.savefig(fig); fig.savefig(os.path.join(figures_dir, "pagina1.png"), dpi=90); plt.close(fig)

        # Página 2: curva de convergencia
        df = pd.read_csv(os.path.join(metrics_dir, f"{pref}_convergencia_mGD.csv"))
        fig = plt.figure(figsize=(8, 6))
        plt.plot(df["Epoch"], df["Loss_Normal"], label="Entropía cruzada normal")
        plt.plot(df["Epoch"], df["Loss_Penalizada"], label="Entropía cruzada penalizada", linestyle="--")
        plt.title("Curvas de convergencia MPE: pérdida normal y penalizada (mGD)"); plt.xlabel("Épocas"); plt.ylabel("Pérdida")
        plt.legend(); plt.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); fig.savefig(os.path.join(figures_dir, "pagina2.png"), dpi=90); plt.close(fig)

        # Páginas 3 y 4: matrices de confusión
        for i, (nombre, y, yp) in enumerate([("Entrenamiento", y_train, yp_train), ("Prueba", y_test, yp_test)]):
            fig = plt.figure(figsize=(6, 5))
            sns.heatmap(confusion_matrix(y, yp, labels=[0, 1]), annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=CLASES, yticklabels=CLASES)
            plt.xlabel("Predicho"); plt.ylabel("Real"); plt.title(f"Matriz de confusión ({nombre})")
            fig.tight_layout()
            pdf.savefig(fig); fig.savefig(os.path.join(figures_dir, f"pagina{3+i}.png"), dpi=90); plt.close(fig)

        # Página 5: F-scores (F1 por clase y macro, train y test)
        fig, ax = plt.subplots(figsize=(8, 5))
        etiquetas = CLASES + ["macro"]
        valores = {}
        for nombre, y, yp in [("Train", y_train, yp_train), ("Test", y_test, yp_test)]:
            f = precision_recall_fscore_support(y, yp, labels=[0, 1], zero_division=0)[2]
            valores[nombre] = list(f) + [float(np.mean(f))]
        x = np.arange(len(etiquetas))
        for j, (nombre, vals) in enumerate(valores.items()):
            barras = ax.bar(x + (j - 0.5) * 0.35, vals, 0.35, label=nombre)
            for b in barras:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(etiquetas, rotation=12)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("F1-score"); ax.set_title("F-scores MPE (train / test)"); ax.legend()
        fig.tight_layout()
        pdf.savefig(fig); fig.savefig(os.path.join(figures_dir, "pagina5.png"), dpi=90); plt.close(fig)
