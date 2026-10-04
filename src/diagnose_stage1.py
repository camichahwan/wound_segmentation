"""
diagnose_stage1.py

Diagnóstico de la etapa 1 (localizador) del enfoque en dos etapas, para
entender POR QUÉ aparecieron las 2 fallas catastróficas nuevas (DSC01378.JPG,
DSC01037.JPG) antes de decidir cómo corregirlas -- ver
`decision_postproc_sobre_dos_etapas.md` en el proyecto.

La hipótesis (todavía sin confirmar visualmente) es que en esas imágenes la
etapa 1 sobre-detecta la herida, el recorte sale demasiado grande/mal puesto,
y la etapa 2 hereda el error. Este script NO cambia ningún modelo ni código
existente: solo corre la etapa 1 sobre el set de test y mide/guarda:

  - Por imagen: área predicha por la etapa 1 vs. área real (ratio), qué
    fracción de la imagen ocupa el recorte que se calcularía, y si ese recorte
    contiene o no la herida real (cobertura del bounding box real).
  - Imágenes superpuestas (overlay) de los peores casos + las 2 fallas ya
    conocidas: contorno VERDE = herida real, ROJO = predicción de la etapa 1,
    AZUL = recorte que recibiría la etapa 2.

El objetivo es elegir el umbral de la "validación de sensatez" a partir de los
datos (en qué ratio / fracción de imagen empiezan los problemas de verdad),
no a ojo.

Uso (Colab, no reentrena nada):
    python diagnose_stage1.py --stage1_checkpoint <ruta a unet_localizer_1024.pt>
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
from postprocess_and_ensemble import load_model, predict_prob
from roi_utils import compute_crop_region, mask_bbox

KNOWN_FAILURES = ["DSC01378.JPG", "DSC01037.JPG"]


def draw_overlay(image_bgr, gt_mask, s1_mask, region, out_path, max_side=1400):
    vis = image_bgr.copy()
    for mask, color in [(gt_mask, (0, 200, 0)), (s1_mask, (0, 0, 255))]:
        contours, _ = cv2.findContours((mask > 127).astype(np.uint8),
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, thickness=max(3, vis.shape[1] // 400))
    if region is not None:
        y0, y1, x0, x1 = region
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 80, 0), thickness=max(3, vis.shape[1] // 400))
    scale = max_side / max(vis.shape[:2])
    if scale < 1:
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, vis)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_worst", type=int, default=8,
                         help="Cuantos overlays de los peores casos guardar (ademas de las 2 fallas conocidas)")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or get_checkpoints_dir()
    overlay_dir = os.path.join(out_dir, "stage1_diagnostics_overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    model, model_args = load_model(args.stage1_checkpoint, device)

    paired, _ = list_paired_and_unpaired()
    _, _, test_samples = train_val_test_split(paired, seed=args.seed)
    print(f"Diagnosticando etapa 1 sobre {len(test_samples)} imagenes de test.")

    rows, cache = [], {}
    for i, sample in enumerate(test_samples):
        image_bgr = cv2.imread(sample.image_path)
        gt_mask = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)
        name = os.path.basename(sample.image_path)
        h, w = gt_mask.shape[:2]

        prob = predict_prob(model, image_bgr, model_args["image_size"], device)
        s1_mask = (prob > 0.5).astype(np.uint8) * 255
        region = compute_crop_region(s1_mask, image_bgr.shape, jitter=False)

        gt_area, s1_area = int((gt_mask > 127).sum()), int((s1_mask > 127).sum())
        gt_bbox = mask_bbox(gt_mask)
        crop_frac = crop_contains_gt = np.nan
        if region is not None:
            y0, y1, x0, x1 = region
            crop_frac = ((y1 - y0) * (x1 - x0)) / (h * w)
            if gt_bbox is not None:
                gy0, gy1, gx0, gx1 = gt_bbox
                crop_contains_gt = float(gy0 >= y0 and gy1 <= y1 and gx0 >= x0 and gx1 <= x1)

        rows.append({
            "image": name,
            "gt_area_frac_img": gt_area / (h * w),
            "s1_area_frac_img": s1_area / (h * w),
            "area_ratio_pred_over_gt": s1_area / gt_area if gt_area else np.nan,
            "crop_frac_img": crop_frac,
            "crop_contains_gt_bbox": crop_contains_gt,
            "s1_empty": s1_area == 0,
        })
        cache[name] = (image_bgr, gt_mask, s1_mask, region)

        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(test_samples)} imagenes procesadas")

    df = pd.DataFrame(rows).set_index("image")
    csv_path = os.path.join(out_dir, "stage1_diagnostics.csv")
    df.to_csv(csv_path)
    print(f"\nDetalle por imagen guardado en: {csv_path}")

    print("\n=== Distribucion: area predicha por etapa 1 / area real ===")
    print(df["area_ratio_pred_over_gt"].describe(percentiles=[.05, .25, .5, .75, .9, .95]).round(3))
    print("\n=== Distribucion: fraccion de la imagen que ocupa el recorte ===")
    print(df["crop_frac_img"].describe(percentiles=[.5, .75, .9, .95]).round(3))
    n_miss = int((df["crop_contains_gt_bbox"] == 0).sum())
    print(f"\nRecortes que NO contienen entera la herida real: {n_miss}/{len(df)}")

    print("\n=== Las 2 fallas conocidas ===")
    for name in KNOWN_FAILURES:
        if name in df.index:
            print(name, df.loc[name].round(3).to_dict())

    worst = (df["area_ratio_pred_over_gt"] - 1).abs().sort_values(ascending=False).head(args.n_worst).index.tolist()
    print("\n=== Peores casos por desvio del ratio de area (ratio 1.0 = perfecto) ===")
    for name in worst:
        print(f"{name}: ratio={df.loc[name, 'area_ratio_pred_over_gt']:.2f} | "
              f"crop={df.loc[name, 'crop_frac_img']:.2f} de la imagen | "
              f"contiene herida real={df.loc[name, 'crop_contains_gt_bbox']}")

    for name in dict.fromkeys(KNOWN_FAILURES + worst):
        if name in cache:
            image_bgr, gt_mask, s1_mask, region = cache[name]
            draw_overlay(image_bgr, gt_mask, s1_mask, region,
                         os.path.join(overlay_dir, f"{os.path.splitext(name)[0]}_stage1.jpg"))
    print(f"\nOverlays guardados en: {overlay_dir}")
    print("Leyenda: verde = herida real | rojo = prediccion etapa 1 | azul = recorte para etapa 2")


if __name__ == "__main__":
    main()
