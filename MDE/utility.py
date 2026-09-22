# ============================================================
# utility.py (MDE) -> lectura de datos, ventanas y entropía multi-escala
# ============================================================
import os
import math
import numpy as np
import scipy.io as sio
from functools import lru_cache
from scipy.signal import decimate
from scipy.stats import norm

# ---------------- Rutas de datos ----------------
CARPETA_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(CARPETA_ACTUAL, ".."))

# Experimento binario CWRU, ambos archivos bajo carga de 0 HP:
# Clase 0: condición normal, archivo 97.mat.
# Clase 1: falla en pista interior, archivo 105.mat.
CARPETA_NORMAL = os.path.join(
    RAIZ,
    "data",
    "raw",
    "cwru_12k_drive_end",
    "normal",
)

CARPETA_FALLO = os.path.join(
    RAIZ,
    "data",
    "raw",
    "cwru_12k_drive_end",
    "inner_race_fault",
)

# Frecuencia objetivo requerida para el análisis.
FS_OBJETIVO = 12000

# Los archivos normales CWRU 97, 98, 99 y 100 se tratan como
# señales de 48 kHz; se reducen a 12 kHz para compararlas con fallas 12 kHz.
ARCHIVOS_48K = {97, 98, 99, 100}

# ---------------- Parámetros de la entropía ----------------
MAX_SCALE = 10   # escalas 1..10
EMBED_DIM = 2    # dimensión de embedding m
CLASSES = 6      # nº de clases de amplitud c
DELAY = 1        # retardo

@lru_cache(maxsize=None)
def get_de_time(file_path):
    """Lee el .mat, extrae DE_time y lo deja a 12 kHz (los Normal se reducen de 48 kHz a 12 kHz)."""
    mat = sio.loadmat(file_path)
    senal = None
    for key in mat.keys():
        if 'DE_time' in key:
            senal = mat[key].flatten().astype(float)
    if senal is None:
        return None
    numero = int(os.path.splitext(os.path.basename(file_path))[0])
    if numero in ARCHIVOS_48K:
        senal = decimate(senal, 4)          # 48 kHz / 4 = 12 kHz
    return senal

def coarse_graining(signal, scale):
    """Promedia bloques de 'scale' muestras (cambio de escala)."""
    n = len(signal)
    max_len = n - (n % scale)
    return np.mean(signal[:max_len].reshape(-1, scale), axis=1)

def _entropia_desde_codigos(codigos, n_posibles):
    _, cuentas = np.unique(codigos, return_counts=True)
    p = cuentas / cuentas.sum()
    return -np.sum(p * np.log2(p)) / np.log2(n_posibles)     # Shannon normalizada (0 a 1)

def dispersion_entropy(signal, classes=6, m=2, delay=1):
    """Entropía de dispersión: se mapea con la normal a 'classes' niveles y se miden los patrones."""
    sigma = np.std(signal)
    N = len(signal) - (m - 1) * delay
    if sigma == 0 or N < 2:
        return 0.0
    # 1. Se mapea la amplitud de la señal al intervalo [0, 1]
    #    mediante la función de distribución acumulada normal.
    y = norm.cdf(
        signal,
        loc=np.mean(signal),
        scale=sigma,
    )

    # 2. Se asigna cada valor a una de las clases 0, ..., classes - 1.
    #    Cada clase representa un intervalo de amplitud de ancho 1/classes.
    z = np.floor(classes * y).astype(int)
    z = np.clip(z, 0, classes - 1)

    idx = np.arange(m) * delay + np.arange(N)[:, None]
    codigos = (z[idx] * (classes ** np.arange(m))).sum(axis=1)      # 3) patrones de dispersión
    return _entropia_desde_codigos(codigos, classes ** m)

def multiscale_entropy(signal):
    """Vector de entropías, una por escala 1..MAX_SCALE."""
    return np.array([dispersion_entropy(coarse_graining(signal, s), CLASSES, EMBED_DIM, DELAY) for s in range(1, MAX_SCALE + 1)])

def ventanas(senal, W, solape):
    """Corta la señal en ventanas de W muestras con traslape 'solape' (0.5 = 50%)."""
    paso = max(1, int(W * (1 - solape)))
    return [senal[i:i + W] for i in range(0, len(senal) - W + 1, paso)]

def _features_de_tramo(carpeta, a, b, W, solape):
    """Entropías de todas las ventanas del tramo [a,b) (fracción de la señal) de cada archivo de la carpeta."""
    X = []
    for filename in sorted(os.listdir(carpeta)):
        if not filename.endswith('.mat'):
            continue
        senal = get_de_time(os.path.join(carpeta, filename))
        if senal is None:
            continue
        tramo = senal[int(len(senal) * a): int(len(senal) * b)]
        for w in ventanas(tramo, W, solape):
            X.append(multiscale_entropy(w))
    return X

def build_dataset(W, solape=0.5, validation_size=0.2, test_size=0.3):
    """
    Construye los conjuntos train, validation y test sin mezclar
    temporalmente las ventanas.

    Para cada archivo:
    - Primer 50 %: entrenamiento.
    - Siguiente 20 %: validación.
    - Último 30 %: prueba.

    Etiquetas:
    - 0: Normal, archivo 97.mat.
    - 1: Falla en pista interior, archivo 105.mat.
    """

    if validation_size <= 0 or test_size <= 0:
        raise ValueError(
            "validation_size y test_size deben ser mayores que 0."
        )

    if validation_size + test_size >= 1:
        raise ValueError(
            "validation_size + test_size debe ser menor que 1."
        )

    train_end = 1 - validation_size - test_size
    validation_end = 1 - test_size

    # Clase normal: etiqueta 0.
    Xn_train = _features_de_tramo(
        CARPETA_NORMAL,
        0,
        train_end,
        W,
        solape,
    )

    Xn_validation = _features_de_tramo(
        CARPETA_NORMAL,
        train_end,
        validation_end,
        W,
        solape,
    )

    Xn_test = _features_de_tramo(
        CARPETA_NORMAL,
        validation_end,
        1,
        W,
        solape,
    )

    # Clase de falla en pista interior: etiqueta 1.
    Xf_train = _features_de_tramo(
        CARPETA_FALLO,
        0,
        train_end,
        W,
        solape,
    )

    Xf_validation = _features_de_tramo(
        CARPETA_FALLO,
        train_end,
        validation_end,
        W,
        solape,
    )

    Xf_test = _features_de_tramo(
        CARPETA_FALLO,
        validation_end,
        1,
        W,
        solape,
    )

    X_train = np.array(Xn_train + Xf_train)
    y_train = np.array(
        [0] * len(Xn_train) + [1] * len(Xf_train)
    )

    X_validation = np.array(Xn_validation + Xf_validation)
    y_validation = np.array(
        [0] * len(Xn_validation) + [1] * len(Xf_validation)
    )

    X_test = np.array(Xn_test + Xf_test)
    y_test = np.array(
        [0] * len(Xn_test) + [1] * len(Xf_test)
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )

def standardize(X_train, X_test, mean=None, std=None):
    """Z-score. Si no se entregan mean/std se calculan SOLO con train."""
    if mean is None:
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std, mean, std
