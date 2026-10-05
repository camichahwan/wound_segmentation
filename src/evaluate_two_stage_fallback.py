"""
evaluate_two_stage_fallback.py

Prueba la correccion que sale del diagnostico en VALIDACION
(`diagnose_stage1.py --split val`): cuando la etapa 1 NO detecta nada, el
pipeline en dos etapas le pasa la IMAGEN COMPLETA a la etapa 2, que fue
entrenada solo con recortes chicos alrededor de la herida. Resultado: una
sobre-segmentacion gigante (DSC01102: pretrained solo Dice 0.909, dos etapas
~0.00, area predicha ~18x la real).

Correccion probada: si la etapa 1 (ya filtrada con la componente mas grande)
queda vacia, usar la prediccion del modelo de una sola etapa (unet_pretrained)
en lugar de pasar la imagen completa a la etapa 2.

Compara, sobre las mismas imagenes y en la misma corrida:
    - pretrained_baseline: unet_pretrained.pt solo.
    - two_stage_s1filter: dos etapas + filtro de etapa 1 (fallback = imagen completa a etapa 2).
    - two_stage_s1filter_fb: igual, pero con fallback = prediccion del pretrained solo.

NO modifica ningun script existente; reutiliza sus funciones. Solo inferencia.

Uso (Colab):
    python evaluate_two_stage_fallback.py --stage1_checkpoint ... --stage2_checkpoint ... --split val --tag val_ftv_fb
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
from metrics import evaluate_all
from roi_utils import compute_crop_region
from postprocess_and_ensemble import load_model, predict_prob
from evaluate_two_stage_postproc import summarize_metric
from evaluate_two_stage_s1filter import keep_largest_component


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--stage2_checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val",
                         help="Para TOMAR DECISIONES usar val; el test se reserva para el numero final.")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = get_checkpoints_dir()
    out_dir = args.out_dir or ckpt_dir
    os.makedirs(out_dir, exist_ok=True)

    base_model, base_args = load_model(os.path.join(ckpt_dir, "unet_pretrained.pt"), device)
    s1_model, s1_args = load_model(args.stage1_checkpoint, device)
    s2_model, s2_args = load_model(args.stage2_checkpoint, device)

    paired, _ = list_paired_and_unpaired()
    _, val_samples, test_samples = train_val_test_split(paired, seed=args.seed)
    samples = val_samples if args.split == "val" else test_samples
    suffix = f"_{args.tag}" if args.tag else f"_{args.split}"
    print(f"Evaluando {len(samples)} imagenes del split '{args.split}'.")

    rows, fallback_images = [], []
    for i, sample in enumerate(samples):
        image_bgr = cv2.imread(sample.image_path)
        gt_mask = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)
        name = os.path.basename(sample.image_path)
        h, w = gt_mask.shape[:2]

        base_prob = predict_prob(base_model, image_bgr, base_args["image_size"], device)
        base_mask = (base_prob > 0.5).astype(np.uint8) * 255

        s1_prob = predict_prob(s1_model, image_bgr, s1_args["image_size"], device)
        s1_mask = keep_largest_component((s1_prob > 0.5).astype(np.uint8) * 255)
        region = compute_crop_region(s1_mask, image_bgr.shape, jitter=False)

        if region is None:
            fallback_images.append(name)
            # variante actual: imagen completa a la etapa 2
            s2_prob = predict_prob(s2_model, image_bgr, s2_args["image_size"], device)
            cur_mask = (s2_prob > 0.5).astype(np.uint8) * 255
            fb_mask = base_mask
        else:
            y0, y1, x0, x1 = region
            s2_prob = predict_prob(s2_model, image_bgr[y0:y1, x0:x1], s2_args["image_size"], device)
            full_prob = np.zeros((h, w), dtype=np.float32)
            full_prob[y0:y1, x0:x1] = s2_prob
            cur_mask = (full_prob > 0.5).astype(np.uint8) * 255
            fb_mask = cur_mask

        for variant, pred in [("pretrained_baseline", base_mask),
                               ("two_stage_s1filter", cur_mask),
                               ("two_stage_s1filter_fb", fb_mask)]:
            row = evaluate_all(pred, gt_mask)
            row["variant"], row["image"] = variant, name
            rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(samples)} imagenes procesadas")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"two_stage_fallback_results{suffix}.csv"), index=False)
    get = lambda v: df[df.variant == v].set_index("image").sort_index()
    base, cur, fb = get("pretrained_baseline"), get("two_stage_s1filter"), get("two_stage_s1filter_fb")

    print(f"\nImagenes donde la etapa 1 quedo vacia (se activa el fallback): {len(fallback_images)}/{len(samples)}")
    print("  ->", fallback_images)

    lines = [f"# Dos etapas + fallback a pretrained cuando la etapa 1 queda vacia (split: {args.split}, "
             f"etapa 2: {os.path.basename(args.stage2_checkpoint)})", "",
             f"Imagenes con etapa 1 vacia: {len(fallback_images)}/{len(samples)} "
             f"({', '.join(fallback_images) if fallback_images else 'ninguna'}).", ""]
    header = ["| Métrica | Media base | Media var. | Mediana base | Mediana var. | mejoran/empeoran/igual | p (Wilcoxon) | Veredicto |",
              "|---|---|---|---|---|---|---|---|"]
    for title, a, b in [("two_stage_s1filter_fb vs. two_stage_s1filter (¿el fallback ayuda?)", cur, fb),
                        ("two_stage_s1filter_fb vs. pretrained_baseline", base, fb)]:
        print(f"\n=== {title} ===")
        lines += [f"## {title}", ""] + header
        for m, lower in [("dice", False), ("iou", False), ("hausdorff_px", True), ("rel_error_pct", True)]:
            summarize_metric(lines, m, a[m], b[m], lower)
        lines.append("")

    if fallback_images:
        lines += ["## Imagenes donde se activo el fallback", "",
                  "| Imagen | Dice pretrained | Dice two_stage_s1filter | Dice con fallback |", "|---|---|---|---|"]
        print("\n=== Imagenes donde se activo el fallback ===")
        for n in fallback_images:
            print(f"{n}: dice pretrained {base.loc[n,'dice']:.3f} | actual {cur.loc[n,'dice']:.3f} | fallback {fb.loc[n,'dice']:.3f}")
            lines.append(f"| {n} | {base.loc[n,'dice']:.3f} | {cur.loc[n,'dice']:.3f} | {fb.loc[n,'dice']:.3f} |")

    report = os.path.join(out_dir, f"two_stage_fallback_decision{suffix}.md")
    with open(report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReporte guardado en: {report}")


if __name__ == "__main__":
    main()
