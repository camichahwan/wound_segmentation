"""
evaluate_two_stage_s1filter.py

Prueba la corrección que sale del diagnóstico de la etapa 1
(`diagnose_stage1.py`): las 2 fallas catastróficas del enfoque en dos etapas
(DSC01378.JPG, DSC01037.JPG) NO se deben a que la etapa 1 sobre-estime el
tamaño de la herida (sus ratios de área son 1.36 y 1.09, normales), sino a que
la máscara de la etapa 1 trae además un blob falso chico y separado (en
DSC01378 sobre la oreja/cuello, en DSC01037 un puntito sobre el guante) que
infla el bounding box y por lo tanto el recorte que recibe la etapa 2.

Corrección: antes de calcular el recorte, quedarse SOLO con la componente
conexa más grande de la máscara de la etapa 1 (sin tocar la máscara final ni
los modelos). Es seguro para este dataset: las 110 máscaras reales de test
tienen exactamente 1 componente conexa.

Compara, sobre las mismas 110 imágenes de test y en la misma corrida:
    - pretrained_baseline: unet_pretrained.pt solo.
    - two_stage: enfoque en dos etapas actual (sin filtrar la etapa 1).
    - two_stage_s1filter: enfoque en dos etapas filtrando la etapa 1.

NO modifica evaluate_two_stage.py ni ningún otro script existente; reutiliza
sus funciones.

Uso (Colab, no reentrena nada):
    python evaluate_two_stage_s1filter.py --stage1_checkpoint ... --stage2_checkpoint ...
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
from evaluate_two_stage import run_two_stage
from evaluate_two_stage_postproc import summarize_metric


def keep_largest_component(mask_uint8: np.ndarray) -> np.ndarray:
    """Solo la componente conexa mas grande (sin cierre morfologico: aca
    el objetivo es unicamente que un blob falso chico no infle el recorte)."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask_uint8 > 127).astype(np.uint8), connectivity=8)
    if n_labels <= 2:
        return mask_uint8  # 0 o 1 componentes: nada que filtrar
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask_uint8)
    out[labels == largest] = 255
    return out


def run_two_stage_s1filter(stage1_model, stage1_args, stage2_model, stage2_args, image_bgr, device):
    h, w = image_bgr.shape[:2]
    s1_prob = predict_prob(stage1_model, image_bgr, stage1_args["image_size"], device)
    s1_mask = (s1_prob > 0.5).astype(np.uint8) * 255
    s1_filtered = keep_largest_component(s1_mask)

    region = compute_crop_region(s1_filtered, image_bgr.shape, jitter=False)
    if region is None:
        crop, (y0, y1, x0, x1) = image_bgr, (0, h, 0, w)
    else:
        y0, y1, x0, x1 = region
        crop = image_bgr[y0:y1, x0:x1]

    s2_prob = predict_prob(stage2_model, crop, stage2_args["image_size"], device)
    full_prob = np.zeros((h, w), dtype=np.float32)
    full_prob[y0:y1, x0:x1] = s2_prob
    return (full_prob > 0.5).astype(np.uint8) * 255, bool(np.any(s1_mask != s1_filtered))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--stage2_checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test",
                         help="Sobre que split evaluar. Para TOMAR DECISIONES usar val; el test se "
                              "reserva para el numero final de la tesis.")
    parser.add_argument("--tag", default="",
                         help="Sufijo para los archivos de salida (ej. 'val_ftv') para no pisar corridas.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = get_checkpoints_dir()
    out_dir = args.out_dir or ckpt_dir
    os.makedirs(out_dir, exist_ok=True)

    baseline_model, baseline_args = load_model(os.path.join(ckpt_dir, "unet_pretrained.pt"), device)
    stage1_model, stage1_args = load_model(args.stage1_checkpoint, device)
    stage2_model, stage2_args = load_model(args.stage2_checkpoint, device)

    paired, _ = list_paired_and_unpaired()
    _, val_samples, test_samples = train_val_test_split(paired, seed=args.seed)
    if args.split == "val":
        test_samples = val_samples
    suffix = f"_{args.tag}" if args.tag else (f"_{args.split}" if args.split != "test" else "")
    print(f"Evaluando {len(test_samples)} imagenes del split '{args.split}'.")

    rows, changed_images = [], []
    for i, sample in enumerate(test_samples):
        image_bgr = cv2.imread(sample.image_path)
        gt_mask = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)
        name = os.path.basename(sample.image_path)

        base_prob = predict_prob(baseline_model, image_bgr, baseline_args["image_size"], device)
        base_mask = (base_prob > 0.5).astype(np.uint8) * 255
        ts_mask, _ = run_two_stage(stage1_model, stage1_args, stage2_model, stage2_args, image_bgr, device)
        tsf_mask, s1_changed = run_two_stage_s1filter(stage1_model, stage1_args, stage2_model, stage2_args, image_bgr, device)
        if s1_changed:
            changed_images.append(name)

        for variant, pred in [("pretrained_baseline", base_mask), ("two_stage", ts_mask), ("two_stage_s1filter", tsf_mask)]:
            row = evaluate_all(pred, gt_mask)
            row["variant"], row["image"] = variant, name
            rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(test_samples)} imagenes procesadas")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"two_stage_s1filter_results{suffix}.csv"), index=False)

    get = lambda v: df[df.variant == v].set_index("image").sort_index()
    base, ts, tsf = get("pretrained_baseline"), get("two_stage"), get("two_stage_s1filter")

    print(f"\nImagenes donde el filtro cambio la mascara de la etapa 1: {len(changed_images)}/{len(test_samples)}")
    print("  ->", changed_images)

    lines = [f"# Dos etapas + filtro de componente mayor en la etapa 1 (split: {args.split}, etapa 2: {os.path.basename(args.stage2_checkpoint)})", "",
             f"Imagenes donde el filtro cambio la mascara de la etapa 1: {len(changed_images)}/{len(test_samples)} "
             f"({', '.join(changed_images) if changed_images else 'ninguna'}).", ""]
    header = ["| Métrica | Media base | Media var. | Mediana base | Mediana var. | mejoran/empeoran/igual | p (Wilcoxon) | Veredicto |",
              "|---|---|---|---|---|---|---|---|"]
    comparisons = [("two_stage_s1filter vs. two_stage (¿el filtro ayuda?)", ts, tsf),
                   ("two_stage_s1filter vs. pretrained_baseline", base, tsf)]
    for title, a, b in comparisons:
        print(f"\n=== {title} ===")
        lines += [f"## {title}", ""] + header
        for m, lower in [("dice", False), ("iou", False), ("hausdorff_px", True), ("rel_error_pct", True)]:
            summarize_metric(lines, m, a[m], b[m], lower)
        lines.append("")

    lines += ["## Las 2 fallas catastroficas y las 4 heridas chicas", "",
              "| Imagen | Dice pretrained | Dice two_stage | Dice two_stage_s1filter | Hausdorff pretrained | Hausdorff two_stage | Hausdorff two_stage_s1filter |",
              "|---|---|---|---|---|---|---|"]
    print("\n=== Las 2 fallas catastroficas y las 4 heridas chicas ===")
    for n in ["DSC01378.JPG", "DSC01037.JPG", "DSC01428.JPG", "DSC01017.JPG", "DSC01995.JPG", "DSC01234.JPG"]:
        if n in base.index:
            print(f"{n}: dice {base.loc[n,'dice']:.3f} -> {ts.loc[n,'dice']:.3f} -> {tsf.loc[n,'dice']:.3f} | "
                  f"hausdorff {base.loc[n,'hausdorff_px']:.1f} -> {ts.loc[n,'hausdorff_px']:.1f} -> {tsf.loc[n,'hausdorff_px']:.1f}")
            lines.append(f"| {n} | {base.loc[n,'dice']:.3f} | {ts.loc[n,'dice']:.3f} | {tsf.loc[n,'dice']:.3f} | "
                         f"{base.loc[n,'hausdorff_px']:.1f} | {ts.loc[n,'hausdorff_px']:.1f} | {tsf.loc[n,'hausdorff_px']:.1f} |")

    report = os.path.join(out_dir, f"two_stage_s1filter_decision{suffix}.md")
    with open(report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReporte guardado en: {report}")


if __name__ == "__main__":
    main()
