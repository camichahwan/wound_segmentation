"""
evaluate_two_stage.py

Evaluación de punta a punta del enfoque en dos etapas (ver
outputs/error_analysis/heridas_chicas_resize.md), sobre las imágenes de test
ORIGINALES (completas, no recortadas):

    Etapa 1 (localizador, ej. unet_localizer_1024.pt): corre sobre la imagen
    completa, da una máscara aproximada. Se calcula su bounding box + margen
    (roi_utils.py) y se recorta la imagen ORIGINAL en alta resolución con
    esa región -- SIN usar la máscara real, para que esta evaluación sea
    honesta (en producción no hay máscara real disponible).

    Etapa 2 (segmentador fino, ej. unet_stage2_finetuned.pt): corre sobre
    ese recorte, y el resultado se pega de vuelta en un lienzo del tamaño de
    la imagen original, en la posición del recorte.

Si la etapa 1 no detecta nada (máscara vacía), se usa la imagen completa
como "recorte" -- degrada de forma prolija al comportamiento de un solo
modelo en vez de romper.

Compara contra la línea de base (unet_pretrained.pt solo, recalculado acá
mismo para garantizar que es una comparación pareada sobre las MISMAS
imágenes) con test de Wilcoxon, misma metodología que el resto del repo.

Uso (correr en Colab, necesita los 3 checkpoints: unet_pretrained.pt,
el localizador de etapa 1, y el segmentador fino de etapa 2):
    python evaluate_two_stage.py --stage1_checkpoint ../outputs/../unet_localizer_1024.pt \
                                  --stage2_checkpoint ../outputs/../unet_stage2_finetuned.pt
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import wilcoxon

from config import get_checkpoints_dir
from dataset import list_paired_and_unpaired, train_val_test_split
from metrics import evaluate_all
from roi_utils import compute_crop_region
from postprocess_and_ensemble import load_model, predict_prob


def run_two_stage(stage1_model, stage1_args, stage2_model, stage2_args, image_bgr, device):
    h, w = image_bgr.shape[:2]

    stage1_prob = predict_prob(stage1_model, image_bgr, stage1_args["image_size"], device)
    stage1_mask = (stage1_prob > 0.5).astype(np.uint8) * 255

    region = compute_crop_region(stage1_mask, image_bgr.shape, jitter=False)
    if region is None:
        # La etapa 1 no detecto nada -- degradar a usar la imagen completa,
        # no romper ni devolver una mascara vacia sin mas.
        crop = image_bgr
        y0, y1, x0, x1 = 0, h, 0, w
    else:
        y0, y1, x0, x1 = region
        crop = image_bgr[y0:y1, x0:x1]

    stage2_prob_crop = predict_prob(stage2_model, crop, stage2_args["image_size"], device)

    full_prob = np.zeros((h, w), dtype=np.float32)
    full_prob[y0:y1, x0:x1] = stage2_prob_crop
    final_mask = (full_prob > 0.5).astype(np.uint8) * 255
    return final_mask, stage1_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--stage2_checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = get_checkpoints_dir()
    out_dir = args.out_dir or ckpt_dir
    os.makedirs(out_dir, exist_ok=True)

    print("Cargando modelos...")
    baseline_model, baseline_args = load_model(os.path.join(ckpt_dir, "unet_pretrained.pt"), device)
    stage1_model, stage1_args = load_model(args.stage1_checkpoint, device)
    stage2_model, stage2_args = load_model(args.stage2_checkpoint, device)

    paired, _ = list_paired_and_unpaired()
    _, _, test_samples = train_val_test_split(paired, seed=args.seed)
    print(f"Evaluando dos etapas sobre {len(test_samples)} imagenes de test (imagenes completas originales).")

    rows = []
    n_stage1_empty = 0
    for i, sample in enumerate(test_samples):
        image_bgr = cv2.imread(sample.image_path)
        gt_mask = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)
        image_name = os.path.basename(sample.image_path)

        baseline_prob = predict_prob(baseline_model, image_bgr, baseline_args["image_size"], device)
        baseline_mask = (baseline_prob > 0.5).astype(np.uint8) * 255

        final_mask, stage1_mask = run_two_stage(
            stage1_model, stage1_args, stage2_model, stage2_args, image_bgr, device)
        if not np.any(stage1_mask):
            n_stage1_empty += 1

        row_baseline = evaluate_all(baseline_mask, gt_mask)
        row_baseline["variant"] = "pretrained_baseline"
        row_baseline["image"] = image_name
        rows.append(row_baseline)

        row_two_stage = evaluate_all(final_mask, gt_mask)
        row_two_stage["variant"] = "two_stage"
        row_two_stage["image"] = image_name
        rows.append(row_two_stage)

        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(test_samples)} imagenes procesadas")

    print(f"\nImagenes donde la etapa 1 no detecto nada (se uso la imagen completa): {n_stage1_empty}/{len(test_samples)}")

    df = pd.DataFrame(rows)
    per_image_csv = os.path.join(out_dir, "two_stage_results.csv")
    df.to_csv(per_image_csv, index=False)
    print(f"Detalle por imagen guardado en: {per_image_csv}")

    metric_cols = [c for c in df.columns if c not in ("variant", "image")]
    summary = df.groupby("variant")[metric_cols].agg(["mean", "std"])
    print("\n=== Resumen: pretrained solo vs. dos etapas ===")
    print(summary)

    baseline = df[df.variant == "pretrained_baseline"].set_index("image").sort_index()
    two_stage = df[df.variant == "two_stage"].set_index("image").sort_index()
    assert (baseline.index == two_stage.index).all()

    report_lines = [
        "# Enfoque en dos etapas: evaluacion de punta a punta",
        "",
        f"Comparacion pareada (Wilcoxon, por imagen) contra `unet_pretrained` solo, "
        f"sobre {len(test_samples)} imagenes de test ORIGINALES completas (no recortes). "
        f"Etapa 1: `{os.path.basename(args.stage1_checkpoint)}`. "
        f"Etapa 2: `{os.path.basename(args.stage2_checkpoint)}`. "
        f"Etapa 1 no detecto nada en {n_stage1_empty}/{len(test_samples)} imagenes "
        "(se uso la imagen completa como fallback en esos casos).",
        "",
        "| Métrica | Pretrained solo | Dos etapas | p (Wilcoxon) |",
        "|---|---|---|---|",
    ]
    print("\n=== Test pareado Wilcoxon: dos etapas vs. pretrained solo ===")
    for m in ["dice", "iou", "hausdorff_px", "rel_error_pct"]:
        a, b = baseline[m].values, two_stage[m].values
        _, p = wilcoxon(a, b)
        veredicto = "sin diferencia significativa"
        if p < 0.05:
            veredicto = "dos etapas mejora" if (
                (m != "hausdorff_px" and m != "rel_error_pct" and b.mean() > a.mean()) or
                (m in ("hausdorff_px", "rel_error_pct") and b.mean() < a.mean())
            ) else "dos etapas empeora"
        print(f"{m:15s} pretrained={a.mean():.4f} dos_etapas={b.mean():.4f} | p={p:.4f} | {veredicto}")
        report_lines.append(f"| {m} | {a.mean():.4f} | {b.mean():.4f} | {p:.4f} ({veredicto}) |")

    # Foco especial: como le fue especificamente en las 4 imagenes chicas ya documentadas
    small_wound_images = ["DSC01428.JPG", "DSC01017.JPG", "DSC01995.JPG", "DSC01234.JPG"]
    report_lines += ["", "## Las 4 imagenes de heridas chicas documentadas (heridas_chicas_resize.md)", "",
                      "| Imagen | Dice pretrained solo | Dice dos etapas |", "|---|---|---|"]
    print("\n=== Las 4 imagenes de heridas chicas ===")
    for img_name in small_wound_images:
        if img_name in baseline.index:
            d_base = baseline.loc[img_name, "dice"]
            d_two = two_stage.loc[img_name, "dice"]
            print(f"{img_name}: pretrained={d_base:.4f} -> dos_etapas={d_two:.4f}")
            report_lines.append(f"| {img_name} | {d_base:.4f} | {d_two:.4f} |")

    report_path = os.path.join(out_dir, "two_stage_decision.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nReporte de decision guardado en: {report_path}")


if __name__ == "__main__":
    main()
