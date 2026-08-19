"""
baselines_classical.py

Corre los métodos clásicos (Canny sobre canal rojo, K-means k=3 en LAB, Otsu
sobre canal 'a') sobre TODO el set de test -- no solo la imagen de muestra de
demo_preprocessing.py -- y arma la misma tabla comparativa que evaluate.py
genera para los modelos de deep learning, para poder pegarlas una al lado de
la otra en la tesis.

Esto es lo que responde a "comparar el mejor modelo contra los métodos
clásicos/geométricos existentes" (el propio anteproyecto menciona el ajuste
de elipses como antecedente insuficiente -- agregar una función de ellipse
fitting acá es la forma más directa de cuantificar esa comparación en vez de
solo citarla).

Uso:
    python baselines_classical.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import pandas as pd

from dataset import list_paired_and_unpaired, train_val_test_split
from preprocessing import (
    canny_on_red_channel, edges_to_candidate_mask, kmeans_segment,
    select_wound_cluster, otsu_threshold_on_a_channel,
    clean_mask_morphology, largest_connected_component,
)
from metrics import evaluate_all


def ellipse_fit_baseline(gt_mask: np.ndarray) -> np.ndarray:
    """
    El método que el anteproyecto menciona explícitamente como antecedente
    insuficiente: ajustar una elipse al contorno de la herida. Acá se ajusta
    sobre el contorno del propio ground truth (no hay forma de "detectar" la
    elipse sin antes tener alguna segmentación previa) -- el objetivo es
    cuantificar cuánto se pierde en Dice/IoU solo por aproximar una forma
    irregular con una elipse, aislando ese error del error de detección.
    """
    contours, _ = cv2.findContours((gt_mask > 0).astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(gt_mask)

    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:  # cv2.fitEllipse necesita al menos 5 puntos
        return np.zeros_like(gt_mask)

    ellipse = cv2.fitEllipse(largest)
    mask = np.zeros_like(gt_mask)
    cv2.ellipse(mask, ellipse, 255, thickness=cv2.FILLED)
    return mask


def run_classical_methods(image_bgr: np.ndarray) -> dict:
    edges = canny_on_red_channel(image_bgr)
    canny_mask = clean_mask_morphology(edges_to_candidate_mask(edges))

    labels, centers = kmeans_segment(image_bgr, k=3, color_space="lab")
    kmeans_mask = select_wound_cluster(labels, centers, color_space="lab")
    kmeans_mask = largest_connected_component(clean_mask_morphology(kmeans_mask))

    otsu_mask = otsu_threshold_on_a_channel(image_bgr)
    otsu_mask = largest_connected_component(clean_mask_morphology(otsu_mask))

    return {
        "canny_red_channel": canny_mask,
        "kmeans_k3_lab": kmeans_mask,
        "otsu_a_channel": otsu_mask,
    }


def main():
    paired, _ = list_paired_and_unpaired()
    _, _, test_samples = train_val_test_split(paired)
    print(f"Corriendo baselines clásicos sobre {len(test_samples)} imágenes de test.")

    rows = []
    for sample in test_samples:
        image_bgr = cv2.imread(sample.image_path)
        gt_mask = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)

        candidates = run_classical_methods(image_bgr)
        candidates["ellipse_fit_on_gt_contour"] = ellipse_fit_baseline(gt_mask)

        for method_name, pred_mask in candidates.items():
            row = evaluate_all(pred_mask, gt_mask)
            row["method"] = method_name
            row["image"] = os.path.basename(sample.image_path)
            rows.append(row)

    df = pd.DataFrame(rows)
    summary = df.groupby("method").agg(["mean", "std"])
    print("\n=== Baselines clásicos: promedio ± desvío estándar sobre el set de test ===")
    print(summary)

    out_csv = os.path.join(os.path.dirname(__file__), "..", "outputs", "classical_baselines_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nResultados guardados en: {out_csv}")
    print("\nNOTA: si estos números salen muy bajos (es lo esperable, ya se vio en el demo de "
          "una sola imagen), ESO es un resultado real y útil para la tesis: cuantifica por qué "
          "hace falta deep learning en vez de asumirlo sin datos.")


if __name__ == "__main__":
    main()
