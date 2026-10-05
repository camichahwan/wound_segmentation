"""
evaluate_stage1_localization.py

Mide SOLO la etapa 1 (localizador), sin la etapa 2, porque el diagnostico en
validacion (ver `diagnostico_validacion_fallas_heridas_chicas.md`) mostro que
en las heridas chicas el pipeline en dos etapas funciona muy bien cuando la
etapa 1 localiza bien (Dice 0.90 vs 0.88 del pretrained solo en 24/33 casos) y
mal cuando la etapa 1 falla (9/33 casos, Dice ~0.4 en ambos). El cuello de
botella es la etapa 1.

Para cada checkpoint de etapa 1 (y opcionalmente con TTA de flips) calcula, por
imagen de validacion, si la localizacion es "buena":
    - la etapa 1 (filtrada con la componente mas grande) NO esta vacia,
    - el recorte contiene entero el bounding box real,
    - el area predicha esta entre 0.5x y 2x el area real (no sobre/sub-detecta).
y lo resume en todo el split y por tamaño de herida (chica: < 0.2% de la imagen).

Es solo inferencia. No modifica ningun script ni modelo existente.

Uso (Colab):
    python evaluate_stage1_localization.py --checkpoints unet_localizer_1024.pt unet_localizer_1024_os.pt --split val
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
import cv2

from config import get_checkpoints_dir
from dataset import list_paired_and_unpaired, train_val_test_split
from roi_utils import compute_crop_region, mask_bbox
from postprocess_and_ensemble import load_model, predict_prob
from evaluate_two_stage_s1filter import keep_largest_component

TINY_FRAC = 0.002  # herida chica: menos de 0.2% del area de la imagen


def predict_prob_tta(model, image_bgr, size, device):
    """Promedio de la imagen original + flip horizontal + flip vertical (invertidos de vuelta)."""
    p = predict_prob(model, image_bgr, size, device)
    p_h = predict_prob(model, image_bgr[:, ::-1].copy(), size, device)[:, ::-1]
    p_v = predict_prob(model, image_bgr[::-1].copy(), size, device)[::-1]
    return (p + p_h + p_v) / 3.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                         help="Nombres (dentro de la carpeta de checkpoints) o rutas de las etapas 1 a comparar")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--tta", action="store_true", help="Evaluar tambien cada checkpoint con TTA de flips")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = get_checkpoints_dir()
    out_dir = args.out_dir or ckpt_dir

    paired, _ = list_paired_and_unpaired()
    _, val_samples, test_samples = train_val_test_split(paired, seed=args.seed)
    samples = val_samples if args.split == "val" else test_samples

    configs = []
    for c in args.checkpoints:
        path = c if os.path.isabs(c) else os.path.join(ckpt_dir, c)
        model, margs = load_model(path, device)
        configs.append((os.path.basename(c), False, model, margs))
        if args.tta:
            configs.append((os.path.basename(c) + " +TTA", True, model, margs))

    rows = []
    for i, sample in enumerate(samples):
        image_bgr = cv2.imread(sample.image_path)
        gt_mask = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)
        name = os.path.basename(sample.image_path)
        h, w = gt_mask.shape[:2]
        gt_area = int((gt_mask > 127).sum())
        gt_bbox = mask_bbox(gt_mask)

        for label, use_tta, model, margs in configs:
            fn = predict_prob_tta if use_tta else predict_prob
            prob = fn(model, image_bgr, margs["image_size"], device)
            mask = keep_largest_component((prob > 0.5).astype(np.uint8) * 255)
            area = int((mask > 127).sum())
            region = compute_crop_region(mask, image_bgr.shape, jitter=False)
            empty = region is None
            contains = False
            if not empty and gt_bbox is not None:
                y0, y1, x0, x1 = region
                gy0, gy1, gx0, gx1 = gt_bbox
                contains = bool(gy0 >= y0 and gy1 <= y1 and gx0 >= x0 and gx1 <= x1)
            ratio = area / gt_area if gt_area else np.nan
            rows.append({"checkpoint": label, "image": name, "gt_frac": gt_area / (h * w),
                         "area_ratio": ratio, "empty": empty, "crop_contains_gt": contains,
                         "ok": (not empty) and contains and 0.5 <= ratio <= 2.0})
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(samples)} imagenes procesadas")

    df = pd.DataFrame(rows)
    df["tiny"] = df["gt_frac"] < TINY_FRAC
    df.to_csv(os.path.join(out_dir, f"stage1_localization_{args.split}.csv"), index=False)

    lines = [f"# Localizacion de la etapa 1 (split: {args.split}, {len(samples)} imagenes)", "",
             "Localizacion buena = etapa 1 no vacia + recorte contiene la herida real + area predicha entre 0.5x y 2x la real. "
             f"Herida chica = menos de {TINY_FRAC*100:.1f}% de la imagen.", "",
             "| Etapa 1 | Buenas (todas) | Buenas (chicas) | Vacias | Recorte no contiene herida | Ratio mediano |",
             "|---|---|---|---|---|---|"]
    print()
    for label, g in df.groupby("checkpoint", sort=False):
        t = g[g.tiny]
        line = (f"| {label} | {int(g.ok.sum())}/{len(g)} | {int(t.ok.sum())}/{len(t)} | {int(g.empty.sum())} | "
                f"{int((~g.crop_contains_gt & ~g.empty).sum())} | {g.area_ratio.median():.2f} |")
        lines.append(line)
        print(line)
    report = os.path.join(out_dir, f"stage1_localization_{args.split}.md")
    with open(report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReporte guardado en: {report}")


if __name__ == "__main__":
    main()
