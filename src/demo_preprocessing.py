"""
demo_preprocessing.py

Muestra el efecto de apply_clahe() sobre la imagen de muestra: antes/después,
más la diferencia de histograma del canal L, para juzgar a ojo si mejora algo
antes de decidir si lo sumamos al pipeline de entrenamiento.

Uso:
    python src/demo_preprocessing.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import matplotlib.pyplot as plt

from preprocessing import apply_clahe

HERE = os.path.dirname(__file__)
IMG_PATH = os.path.join(HERE, "..", "data", "samples", "DSC00598.JPG")
OUT_PATH = os.path.join(HERE, "..", "outputs", "figures", "clahe_demo.png")


def main():
    img_bgr = cv2.imread(IMG_PATH)
    scale = 1024 / max(img_bgr.shape[:2])
    img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    img_eq = apply_clahe(img_bgr)

    l_before = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    l_after = cv2.cvtColor(img_eq, cv2.COLOR_BGR2LAB)[:, :, 0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(img_eq, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Con CLAHE (canal L)")
    axes[0, 1].axis("off")

    axes[1, 0].hist(l_before.ravel(), bins=50, color="gray")
    axes[1, 0].set_title("Histograma canal L - original")

    axes[1, 1].hist(l_after.ravel(), bins=50, color="gray")
    axes[1, 1].set_title("Histograma canal L - con CLAHE")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Figura guardada en: {OUT_PATH}")


if __name__ == "__main__":
    main()
