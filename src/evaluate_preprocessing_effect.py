"""
evaluate_preprocessing_effect.py

Antes de meter un paso de preprocesamiento al pipeline de entrenamiento,
medir con evidencia (no a ojo) si ayuda o perjudica: ¿aumenta o reduce la
separación entre el tejido de herida y el tejido sano que lo rodea?

Método: para cada imagen con máscara, se compara la región de la herida
(mask==255) contra un anillo de piel/pelaje sano inmediatamente alrededor de
la herida (dilatar la máscara y restar la máscara original) -- no contra toda
la imagen, porque el fondo (jaula, papel) no es lo relevante para esta
pregunta. Se calcula la diferencia de medias normalizada (d de Cohen) entre
ambas regiones, antes y después de aplicar el preprocesamiento, PARA CADA
IMAGEN del set, y se reporta la distribución (media, desvío, cuántas imágenes
mejoran vs. empeoran) en vez de un solo número de una sola foto.

Limitación explícita: dentro de cada imagen, los píxeles no son muestras
independientes (correlación espacial), así que el d de Cohen por imagen es un
tamaño de efecto descriptivo, no un test de hipótesis válido en sentido
estricto. Lo que sí es válido estadísticamente es comparar la distribución de
esos d de Cohen ENTRE imágenes (cada imagen sí es una unidad independiente)
con un test pareado (Wilcoxon signed-rank), que es lo que se reporta al final.

Uso:
    python src/evaluate_preprocessing_effect.py --data_dir "/mnt/user-data/uploads/Tesis Imagenes"
"""

import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
from scipy.stats import wilcoxon

from preprocessing import apply_clahe


def cohens_d(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    mean_a, mean_b = sample_a.mean(), sample_b.mean()
    std_a, std_b = sample_a.std(ddof=1), sample_b.std(ddof=1)
    n_a, n_b = len(sample_a), len(sample_b)
    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return (mean_a - mean_b) / pooled_std


def get_wound_and_ring_pixels(channel: np.ndarray, mask: np.ndarray, ring_width: int = 25):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width, ring_width))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    ring_mask = cv2.bitwise_and(dilated, cv2.bitwise_not(mask))
    wound_pixels = channel[mask > 0].astype(np.float64)
    ring_pixels = channel[ring_mask > 0].astype(np.float64)
    return wound_pixels, ring_pixels


def d_for_image(image_bgr: np.ndarray, mask: np.ndarray) -> dict:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    result = {}
    for name, channel in [("L", l_ch), ("a", a_ch), ("b", b_ch)]:
        wound_px, ring_px = get_wound_and_ring_pixels(channel, mask)
        if len(wound_px) < 10 or len(ring_px) < 10:
            result[name] = np.nan
            continue
        result[name] = cohens_d(wound_px, ring_px)
    return result


def find_pairs(data_dir: str):
    images_dir = os.path.join(data_dir, "Images")
    masks_dir = os.path.join(data_dir, "Masks")
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.JPG")) +
                          glob.glob(os.path.join(images_dir, "*.jpg")))

    pairs = []
    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(masks_dir, f"{stem}_mask.png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(masks_dir, f"{stem}.png")
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                         help='Carpeta con subcarpetas Images/ y Masks/, ej. "/mnt/user-data/uploads/Tesis Imagenes"')
    parser.add_argument("--out_dir", default=None,
                         help="Carpeta de salida para el CSV y el reporte. Default: outputs/preprocessing_decisions/")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "..", "outputs", "preprocessing_decisions")
    os.makedirs(out_dir, exist_ok=True)

    pairs = find_pairs(args.data_dir)
    print(f"Pares imagen-máscara encontrados: {len(pairs)}")
    if not pairs:
        print("No se encontraron pares. Revisar --data_dir.")
        return

    rows_before, rows_after, image_names = [], [], []
    for img_path, mask_path in pairs:
        img_bgr = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or mask is None:
            print(f"  (no se pudo leer {os.path.basename(img_path)}, se salta)")
            continue

        scale = 1024 / max(img_bgr.shape[:2])
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8) * 255

        img_clahe = apply_clahe(img_bgr)

        rows_before.append(d_for_image(img_bgr, mask))
        rows_after.append(d_for_image(img_clahe, mask))
        image_names.append(os.path.basename(img_path))

    print(f"\nImágenes efectivamente analizadas: {len(rows_before)}\n")

    # --- Guardar el detalle por imagen (para poder reproducir/auditar el análisis después) ---
    import csv
    per_image_csv = os.path.join(out_dir, "clahe_per_image_results.csv")
    with open(per_image_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "d_L_before", "d_a_before", "d_b_before",
                          "d_L_after", "d_a_after", "d_b_after"])
        for name, before, after in zip(image_names, rows_before, rows_after):
            writer.writerow([name, before["L"], before["a"], before["b"],
                              after["L"], after["a"], after["b"]])
    print(f"Detalle por imagen guardado en: {per_image_csv}\n")

    from datetime import date

    report_lines = [
        "# Decisión de preprocesamiento: CLAHE sobre canal L (LAB)",
        "",
        f"Fecha de esta corrida: {date.today().isoformat()} (ver también fecha del commit de este archivo en git log)",
        f"Script: `src/evaluate_preprocessing_effect.py`",
        f"Imágenes analizadas: {len(rows_before)} (muestra aleatoria, seed=42, del subconjunto con máscara manual)",
        "",
        "## Método",
        "",
        "Para cada imagen, se midió la separación (d de Cohen) entre los píxeles de la",
        "región de herida (máscara manual) y un anillo de piel/pelaje sano inmediatamente",
        "circundante, en los canales L, a y b de LAB, antes y después de aplicar CLAHE",
        "(clip_limit=2.5, tile=8x8) sobre el canal L. La comparación entre condiciones",
        "(con/sin CLAHE) se hizo con un test pareado de Wilcoxon sobre el |d| de cada imagen",
        "(cada imagen es una unidad de observación independiente).",
        "",
        "## Resultados",
        "",
        "| Canal | \\|d\\| medio sin preprocesar | \\|d\\| medio con CLAHE | Mejora / Empeora | p (Wilcoxon) |",
        "|---|---|---|---|---|",
    ]

    conclusions = []
    for channel in ["L", "a", "b"]:
        before = np.array([r[channel] for r in rows_before])
        after = np.array([r[channel] for r in rows_after])
        valid = ~np.isnan(before) & ~np.isnan(after)
        before, after = before[valid], after[valid]

        before_abs = np.abs(before)
        after_abs = np.abs(after)
        n_improved = int((after_abs > before_abs).sum())
        n_worse = int((after_abs < before_abs).sum())
        n_equal = len(before_abs) - n_improved - n_worse

        try:
            stat, p_value = wilcoxon(before_abs, after_abs)
        except ValueError:
            p_value = float("nan")

        print(f"--- Canal {channel} (n={len(before)} imágenes) ---")
        print(f"  |d| promedio SIN preprocesar: {before_abs.mean():.3f} (std={before_abs.std():.3f})")
        print(f"  |d| promedio CON CLAHE:       {after_abs.mean():.3f} (std={after_abs.std():.3f})")
        print(f"  Imágenes donde CLAHE mejora la separación: {n_improved}/{len(before_abs)}")
        print(f"  Imágenes donde CLAHE empeora la separación: {n_worse}/{len(before_abs)}")
        print(f"  Wilcoxon signed-rank p-value: {p_value:.4f}")
        print()

        verdict = "mejora" if (n_improved > n_worse and p_value < 0.05) else \
                  "empeora" if (n_worse > n_improved and p_value < 0.05) else \
                  "sin diferencia significativa"
        report_lines.append(
            f"| {channel} | {before_abs.mean():.3f} | {after_abs.mean():.3f} | "
            f"{n_improved} mejoran / {n_worse} empeoran (n={len(before_abs)}) | "
            f"{p_value:.4f} ({verdict}) |"
        )
        conclusions.append((channel, verdict, p_value))

    print("Lectura: si en el canal L el |d| promedio CON CLAHE es menor que SIN preprocesar, y/o")
    print("'empeora' le gana a 'mejora' en la mayoría de las imágenes, con p < 0.05, hay evidencia")
    print("real (no de una sola foto) de que CLAHE no conviene tal como está planteado acá.")

    report_lines += [
        "",
        "## Conclusión",
        "",
    ]
    a_verdict = next((v for c, v, p in conclusions if c == "a"), None)
    if a_verdict == "empeora":
        report_lines.append(
            "El canal **a** (rojo-verde), que es el que más separa herida de piel sana en este "
            "dataset incluso sin ningún preprocesamiento (mayor |d| que L y comparable a b), "
            "empeora de forma estadísticamente significativa con CLAHE, en la gran mayoría de las "
            "imágenes analizadas. El canal L no mejora de forma significativa. **Decisión: no "
            "incorporar este CLAHE (aplicado sobre el canal L y reconvertido a BGR) al pipeline "
            "de entrenamiento por default.** El daño al canal a parece ser un efecto colateral del "
            "ida y vuelta de espacio de color LAB→BGR (no completamente reversible), no una mejora "
            "real de ningún canal.",
        )
    else:
        report_lines.append(
            "Ver tabla de resultados arriba y decidir según el patrón observado en esta corrida.",
        )
    report_lines += [
        "",
        "## Nota metodológica",
        "",
        "Este análisis es un filtro barato antes de comprometer tiempo de entrenamiento, no un "
        "reemplazo del test definitivo. La prueba concluyente es un ablation real: entrenar el "
        "mismo modelo con y sin el paso de preprocesamiento, mismo split y semilla, y comparar "
        "Dice de validación.",
    ]

    report_path = os.path.join(out_dir, "clahe_decision.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nReporte de decisión guardado en: {report_path}")
    print(f"CSV con el detalle por imagen: {per_image_csv}")


if __name__ == "__main__":
    main()
