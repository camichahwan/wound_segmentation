"""
sam_baseline.py

Comparación contra Segment Anything (SAM) en modo zero-shot -- la
"herramienta actual" más relevante para comparar hoy en día, más allá de los
métodos clásicos de baselines_classical.py. SAM no se entrena con tus datos:
se le da un prompt (un punto o una caja) y devuelve una máscara candidata.

Esto queda en un archivo separado y NO se instala por default (ver
requirements-optional.txt) porque:
  - El checkpoint de SAM pesa varios cientos de MB y hay que descargarlo aparte.
  - No hace falta para arrancar con el objetivo de mínima (U-Net preentrenada
    vs. desde cero); es la comparación de la Idea 4 del documento de ideas de
    tesis (benchmark de foundation models), para cuando llegues a esa parte.

Setup (correr una sola vez):
    pip install segment-anything
    # Descargar un checkpoint, por ejemplo el más chico (ViT-B, ~375MB):
    # https://github.com/facebookresearch/segment-anything#model-checkpoints
    # y guardarlo en outputs/checkpoints/sam_vit_b.pth

Uso del prompt: como SAM necesita un punto o caja de referencia y acá no hay
un detector previo de "dónde está la herida", el punto de prompt que se usa
por default es el CENTRO DE MASA de la máscara ground truth -- es decir, este
script mide "qué tan buena es la máscara de SAM si alguien le señala
manualmente el centro de la herida", no una segmentación 100% automática.
Eso es exactamente lo que hay que aclarar en la tesis al reportar estos
números: es una comparación de calidad de máscara dado un prompt correcto,
no una comparación de detección automática end-to-end.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import pandas as pd

from dataset import list_paired_and_unpaired, train_val_test_split
from metrics import evaluate_all


def mask_centroid(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def run_sam_on_sample(predictor, image_bgr: np.ndarray, prompt_point) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    point_coords = np.array([prompt_point])
    point_labels = np.array([1])  # 1 = punto positivo (pertenece al objeto)

    masks, scores, _ = predictor.predict(
        point_coords=point_coords, point_labels=point_labels, multimask_output=True
    )
    best_mask = masks[int(np.argmax(scores))]
    return (best_mask.astype(np.uint8)) * 255


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam_checkpoint", required=True,
                         help="Ruta al .pth descargado, ej. outputs/checkpoints/sam_vit_b.pth")
    parser.add_argument("--model_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from segment_anything import sam_model_registry, SamPredictor

    device = "cuda" if _cuda_available() else "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    paired, _ = list_paired_and_unpaired()
    _, _, test_samples = train_val_test_split(paired, seed=args.seed)
    print(f"Evaluando SAM ({args.model_type}, zero-shot, prompt=centroide del GT) "
          f"sobre {len(test_samples)} imágenes de test.")

    rows = []
    for sample in test_samples:
        image_bgr = cv2.imread(sample.image_path)
        gt_mask = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)

        centroid = mask_centroid(gt_mask)
        if centroid is None:
            continue

        pred_mask = run_sam_on_sample(predictor, image_bgr, centroid)
        row = evaluate_all(pred_mask, gt_mask)
        row["model"] = f"sam_{args.model_type}_zeroshot"
        row["image"] = os.path.basename(sample.image_path)
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n=== SAM zero-shot: promedio ± desvío estándar ===")
    print(df.drop(columns=["model", "image"]).agg(["mean", "std"]))

    out_csv = os.path.join(os.path.dirname(__file__), "..", "outputs", "sam_baseline_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nResultados guardados en: {out_csv}")


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
