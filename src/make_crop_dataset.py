"""
make_crop_dataset.py

Genera el dataset de RECORTES para entrenar la etapa 2 del enfoque en dos
etapas (ver outputs/error_analysis/heridas_chicas_resize.md): a partir de
cada imagen+máscara real, recorta la región alrededor de la herida (con
margen y algo de aleatoriedad, ver roi_utils.py) y guarda el recorte
redimensionado a --crop_size, en una carpeta hermana de "Tesis Imagenes" en
Drive (get_crops_data_root() en config.py) -- así queda persistente, no en
disco temporal.

Usa la máscara REAL (ground truth) para el recorte, no una predicción de la
etapa 1 -- es así a propósito: la etapa 2 aprende a segmentar bien DENTRO de
un recorte razonable, entrenamiento desacoplado de qué tan buena sea la
etapa 1. Los recortes de train llevan aleatoriedad (jitter) para que la
etapa 2 tolere que en producción la etapa 1 no va a ser perfecta; los de
val/test NO llevan jitter (deterministicos, reproducibles).

OJO -- esto genera DOS "test sets" distintos, no confundir:
  1. El test set de ESTE dataset de recortes (recortes de la máscara real)
     -- solo sirve para elegir el mejor checkpoint de la etapa 2 durante SU
     entrenamiento, no mide el sistema completo.
  2. La evaluación de punta a punta real (evaluate_two_stage.py), que corre
     la etapa 1 sobre las imágenes de test ORIGINALES completas, recorta
     con lo que la etapa 1 predijo (no con la máscara real), y recién ahí
     corre la etapa 2 -- esa es la comparación válida contra la línea de
     base (unet_pretrained solo).

Como los nombres de archivo no cambian, train_val_test_split(seed=42) sobre
este dataset nuevo da EXACTAMENTE la misma asignación train/val/test que
sobre el dataset original (mismo split, misma seed, mismos nombres).

Uso:
    python make_crop_dataset.py --crop_size 512
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2

from config import get_crops_data_root
from dataset import list_paired_and_unpaired, train_val_test_split
from roi_utils import compute_crop_region


def process_sample(sample, is_train: bool, crop_size: int, rng, out_images_dir, out_masks_dir):
    img = cv2.imread(sample.image_path)
    mask = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        print(f"  (no se pudo leer {sample.image_path}, se salta)")
        return False

    region = compute_crop_region(mask, img.shape, jitter=is_train, rng=rng)
    if region is None:
        print(f"  ({os.path.basename(sample.image_path)}: mascara vacia, se salta)")
        return False
    y0, y1, x0, x1 = region

    img_crop = img[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1]
    img_resized = cv2.resize(img_crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask_crop, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)

    stem = os.path.splitext(os.path.basename(sample.image_path))[0]
    ext = os.path.splitext(sample.image_path)[1] or ".jpg"
    out_img_path = os.path.join(out_images_dir, f"{stem}{ext}")
    out_mask_path = os.path.join(out_masks_dir, f"{stem}_mask.png")
    cv2.imwrite(out_img_path, img_resized)
    cv2.imwrite(out_mask_path, mask_resized)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paired, _ = list_paired_and_unpaired()
    train_samples, val_samples, test_samples = train_val_test_split(paired, seed=args.seed)
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    crops_root = get_crops_data_root()
    out_images_dir = os.path.join(crops_root, "Images")
    out_masks_dir = os.path.join(crops_root, "Masks")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)
    print(f"Generando recortes en: {crops_root}")

    rng = np.random.RandomState(args.seed)
    n_ok, n_skipped = 0, 0
    for split_name, samples, is_train in [("train", train_samples, True),
                                           ("val", val_samples, False),
                                           ("test", test_samples, False)]:
        split_ok = 0
        for sample in samples:
            if process_sample(sample, is_train, args.crop_size, rng, out_images_dir, out_masks_dir):
                n_ok += 1
                split_ok += 1
            else:
                n_skipped += 1
        print(f"  {split_name}: {split_ok}/{len(samples)} recortes generados")

    print(f"\nTotal recortes generados: {n_ok} | saltados (mascara vacia o error de lectura): {n_skipped}")
    print(f"Dataset de recortes listo en: {crops_root}")
    print("Para entrenar la etapa 2 sobre esto, correr train.py con la variable de entorno "
          "WOUND_DATA_ROOT apuntando a esa carpeta (ver notebook, celda de fine-tuning etapa 2).")


if __name__ == "__main__":
    main()
