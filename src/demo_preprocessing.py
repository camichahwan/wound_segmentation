"""
demo_preprocessing.py

Corre las funciones de preprocessing.py sobre UNA imagen de muestra (la que
ya está en data/samples/) y genera una figura comparativa contra la máscara
manual real, para poder ver de entrada si estos métodos clásicos tienen
sentido en este dataset antes de invertir más tiempo en ellos.

Uso:
    python src/demo_preprocessing.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import (
    to_lab, canny_on_red_channel, edges_to_candidate_mask,
    kmeans_segment, select_wound_cluster, otsu_threshold_on_a_channel,
    clean_mask_morphology, largest_connected_component, estimate_foreground_mask,
)

HERE = os.path.dirname(__file__)
IMG_PATH = os.path.join(HERE, "..", "data", "samples", "DSC00598.JPG")
MASK_PATH = os.path.join(HERE, "..", "data", "samples", "DSC00598_mask.png")
OUT_PATH = os.path.join(HERE, "..", "outputs", "figures", "preprocessing_demo.png")


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return 2.0 * intersection / denom


def main():
    img_bgr = cv2.imread(IMG_PATH)
    gt_mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

    # Downscale para que el demo corra rápido (las imágenes originales son ~4300x3200)
    scale = 1024 / max(img_bgr.shape[:2])
    new_size = (int(img_bgr.shape[1] * scale), int(img_bgr.shape[0] * scale))
    img_bgr = cv2.resize(img_bgr, new_size, interpolation=cv2.INTER_AREA)
    gt_mask = cv2.resize(gt_mask, new_size, interpolation=cv2.INTER_NEAREST)

    # --- LAB ---
    lab = to_lab(img_bgr)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # --- Paso 0: intento de aislar al animal del fondo (jaula/papel/sombra) ---
    # NOTA: en las pruebas sobre esta imagen esta heurística resultó inestable
    # (a veces selecciona la jaula, a veces un borde del papel, no el animal).
    # Se deja la función en preprocessing.py documentada como experimental, y
    # el demo corre los métodos clásicos SIN ROI para reportar un número
    # honesto: así de mal les va a estos métodos sin más trabajo de por medio.
    fg_mask = estimate_foreground_mask(img_bgr)  # se guarda para inspección, no se usa abajo

    # --- Canny sobre canal rojo ---
    edges = canny_on_red_channel(img_bgr)
    canny_mask = edges_to_candidate_mask(edges)
    canny_mask = clean_mask_morphology(canny_mask)

    # --- K-means k=3 en LAB sobre la imagen completa, cluster más "rojo" ---
    labels, centers = kmeans_segment(img_bgr, k=3, color_space="lab")
    kmeans_mask = select_wound_cluster(labels, centers, color_space="lab")
    kmeans_mask = clean_mask_morphology(kmeans_mask)
    kmeans_mask = largest_connected_component(kmeans_mask)

    # --- Otsu sobre canal 'a' de toda la imagen ---
    otsu_mask = otsu_threshold_on_a_channel(img_bgr)
    otsu_mask = clean_mask_morphology(otsu_mask)
    otsu_mask = largest_connected_component(otsu_mask)

    # --- Métricas contra el ground truth manual ---
    results = {
        "Canny (canal rojo)": dice_score(canny_mask, gt_mask),
        "K-means k=3 (LAB, cluster rojo)": dice_score(kmeans_mask, gt_mask),
        "Otsu sobre canal 'a'": dice_score(otsu_mask, gt_mask),
    }

    print("Dice de cada método clásico contra la máscara manual (esta imagen):")
    for name, dice in results.items():
        print(f"  {name}: {dice:.3f}")

    # --- Figura comparativa ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    axes[0, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Imagen original")

    axes[0, 1].imshow(l_ch, cmap="gray")
    axes[0, 1].set_title("Canal L (LAB)")

    axes[0, 2].imshow(a_ch, cmap="gray")
    axes[0, 2].set_title("Canal a (LAB) — rojo/verde")

    axes[0, 3].imshow(gt_mask, cmap="gray")
    axes[0, 3].set_title("Máscara manual (ground truth)")

    axes[1, 0].imshow(fg_mask, cmap="gray")
    axes[1, 0].set_title("ROI del animal\n(aislado del fondo)")

    axes[1, 1].imshow(canny_mask, cmap="gray")
    axes[1, 1].set_title(f"Máscara desde Canny\nDice={results['Canny (canal rojo)']:.3f}")

    axes[1, 2].imshow(kmeans_mask, cmap="gray")
    axes[1, 2].set_title(f"K-means k=3 (cluster rojo)\nDice={results['K-means k=3 (LAB, cluster rojo)']:.3f}")

    otsu_dice = results["Otsu sobre canal 'a'"]
    axes[1, 3].imshow(otsu_mask, cmap="gray")
    axes[1, 3].set_title(f"Otsu sobre canal 'a'\nDice={otsu_dice:.3f}")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nFigura guardada en: {OUT_PATH}")


if __name__ == "__main__":
    main()
