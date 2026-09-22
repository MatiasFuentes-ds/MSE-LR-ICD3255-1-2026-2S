from pathlib import Path
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent
FILES = {
    "Normal (clase 0)": ROOT / "data" / "raw" / "cwru_12k_drive_end" / "normal" / "97.mat",
    "Falla pista interior (clase 1)": ROOT / "data" / "raw" / "cwru_12k_drive_end" / "inner_race_fault" / "105.mat",
}


def summarize_array(name, value):
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        print(f"  - {name}: tipo no numérico ({array.dtype})")
        return

    flat = array.ravel()
    print(f"  - {name}")
    print(f"      forma: {array.shape} | muestras: {flat.size} | tipo: {array.dtype}")
    print(f"      mínimo: {flat.min():.6g} | máximo: {flat.max():.6g}")
    print(f"      media: {flat.mean():.6g} | desviación estándar: {flat.std():.6g}")
    print(f"      primeras 5 muestras: {flat[:5]}")


for label, path in FILES.items():
    print("=" * 72)
    print(f"{label}: {path}")

    if not path.is_file():
        print("ERROR: no se encontró el archivo. Revisa que la estructura de carpetas sea correcta.")
        continue

    mat = loadmat(path)
    variables = {key: value for key, value in mat.items() if not key.startswith("__")}
    print("Variables encontradas:", ", ".join(variables))

    de_keys = [key for key in variables if "DE_time" in key]
    if not de_keys:
        print("ADVERTENCIA: no se encontró una variable que contenga 'DE_time'.")

    for key, value in variables.items():
        if key in de_keys:
            print("Señal Drive End detectada:")
            summarize_array(key, value)
        elif np.asarray(value).size <= 10:
            print("Variable auxiliar:")
            summarize_array(key, value)

print("=" * 72)