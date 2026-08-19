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
ambas regiones, antes y después de aplicar el preprocesamiento. Si d baja
después del preprocesamiento, el preprocesamiento está reduciendo lo que
distingue a la herida de la piel sana -- mala señal. Si sube, buena señal.

Limitación explícita: los píxeles dentro de una imagen no son muestras
independientes (están correlacionados espacialmente), así que esto es un
estadístico descriptivo de tamaño de efecto, no un test de hipótesis válido
en sentido estricto. Con n=1 imagen tampoco alcanza para concluir nada en
general -- esto es un primer chequeo rápido, hay que correrlo sobre muchas
más imágenes (idealmente sobre las ~500 ya segmentadas) antes de decidir si
CLAHE entra o no al pipeline de entrenamiento.

Uso:
    python src/evaluate_preprocessing_effect.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np

from preprocessing import apply_clahe

HERE = os.path.dirname(__file__)
IMG_PATH = os.path.join(HERE, "..", "data", "samples", "DSC00598.JPG")
MASK_PATH = os.path.join(HERE, "..", "data", "samples", "DSC00598_mask.png")


def cohens_d(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Diferencia de medias normalizada por el desvío estándar combinado."""
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


def report_for_image(image_bgr: np.ndarray, mask: np.ndarray, label: str):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    print(f"\n--- {label} ---")
    for name, channel in [("L (luminosidad)", l_ch), ("a (rojo-verde)", a_ch), ("b (azul-amarillo)", b_ch)]:
        wound_px, ring_px = get_wound_and_ring_pixels(channel, mask)
        d = cohens_d(wound_px, ring_px)
        print(f"  Canal {name}: media herida={wound_px.mean():.1f}, "
              f"media piel circundante={ring_px.mean():.1f}, d de Cohen={d:.3f}")
    return


def main():
    img_bgr = cv2.imread(IMG_PATH)
    mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

    scale = 1024 / max(img_bgr.shape[:2])
    img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8) * 255

    img_clahe = apply_clahe(img_bgr)

    report_for_image(img_bgr, mask, "SIN preprocesar (original)")
    report_for_image(img_clahe, mask, "CON CLAHE")

    print("\nLectura: |d| más grande = herida y piel circundante MÁS separables en ese canal.")
    print("Si el |d| del canal L baja con CLAHE, CLAHE está lavando la señal de luminosidad")
    print("que distinguía a la herida -- justo lo que sospechaste a ojo con la línea blanca/rosa.")
    print("\nOJO: esto es sobre 1 sola imagen. No sirve para concluir nada todavía, solo para")
    print("tener el script listo y correrlo sobre muchas imágenes en cuanto conectemos tu Drive.")


if __name__ == "__main__":
    main()
