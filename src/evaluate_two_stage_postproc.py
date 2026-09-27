"""
evaluate_two_stage_postproc.py

¿El post-procesamiento morfológico (cierre + quedarse con la componente
conexa más grande, ver `postprocess_and_ensemble.py`) arregla las fallas
catastróficas nuevas que apareció el enfoque en dos etapas en la evaluación
de punta a punta? (ver el hallazgo documentado: 2 imágenes -- DSC01378.JPG,
DSC01037.JPG -- donde la etapa 1 sobre-detectó la región y la etapa 2 terminó
sobre-segmentando dentro de ese recorte más grande de lo necesario).

Compara 3 variantes sobre las mismas 110 imágenes de test, todas evaluadas en
esta misma corrida (no se reusa el CSV de evaluate_two_stage.py, para que la
comparación pareada sea sobre exactamente las mismas máscaras):

    - pretrained_baseline: unet_pretrained.pt solo (línea de base de siempre).
    - two_stage: el enfoque en dos etapas tal cual, sin post-procesar.
    - two_stage_postproc: el resultado de dos etapas + el mismo
      post-procesamiento que ya funcionó bien en el otro experimento.

Lección aprendida de la corrida anterior: el PROMEDIO del Hausdorff puede
quedar totalmente dominado por 1 o 2 outliers catastróficos y dar un
veredicto engañoso. Por eso este script reporta, para cada métrica, la
media, la MEDIANA, y cuántas imágenes mejoran/empeoran/quedan igual -- hay
que mirar los tres juntos, no solo la media.

Uso (necesita los 3 checkpoints ya entrenados, no reentrena nada):
    python evaluate_two_stage_postproc.py \
        --stage1_checkpoint ../outputs/.../unet_localizer_1024.pt \
        --stage2_checkpoint ../outputs/.../unet_stage2_finetuned.pt
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
from postprocess_and_ensemble import load_model, predict_prob, keep_largest_component_and_close
from evaluate_two_stage import run_two_stage


def summarize_metric(report_lines, name, baseline_vals, variant_vals, lower_is_better):
    diff = variant_vals - baseline_vals  # variante - linea de base
    _, p = wilcoxon(baseline_vals, variant_vals)

    n_improve = int((diff < 0).sum()) if lower_is_better else int((diff > 0).sum())
    n_worsen = int((diff > 0).sum()) if lower_is_better else int((diff < 0).sum())
    n_equal = int((diff == 0).sum())

    mean_b, mean_v = baseline_vals.mean(), variant_vals.mean()
    median_b, median_v = baseline_vals.median(), variant_vals.median()

    # Veredicto basado en la MEDIANA (mas robusta a outliers que la media,
    # ver nota del docstring) combinado con el conteo de mejoras/empeoras.
    if p >= 0.05:
        veredicto = "sin diferencia significativa"
    else:
        median_favorece_variante = (median_v < median_b) if lower_is_better else (median_v > median_b)
        veredicto = "mejora (mediana + mayoria)" if (median_favorece_variante and n_improve > n_worsen) else \
                    "mejora en la mayoria pero con outliers en contra" if n_improve > n_worsen else \
                    "empeora"

    print(f"{name:15s} media: {mean_b:.4f} -> {mean_v:.4f} | mediana: {median_b:.4f} -> {median_v:.4f} | "
          f"mejoran={n_improve} empeoran={n_worsen} igual={n_equal} | p={p:.4f} | {veredicto}")
    report_lines.append(
        f"| {name} | {mean_b:.4f} | {mean_v:.4f} | {median_b:.4f} | {median_v:.4f} | "
        f"{n_improve}/{n_worsen}/{n_equal} | {p:.4f} | {veredicto} |"
    )


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
    print(f"Evaluando {len(test_samples)} imagenes de test (imagenes completas originales).")

    rows = []
    for i, sample in enumerate(test_samples):
        image_bgr = cv2.imread(sample.image_path)
        gt_mask = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)
        image_name = os.path.basename(sample.image_path)

        baseline_prob = predict_prob(baseline_model, image_bgr, baseline_args["image_size"], device)
        baseline_mask = (baseline_prob > 0.5).astype(np.uint8) * 255

        two_stage_mask, _ = run_two_stage(
            stage1_model, stage1_args, stage2_model, stage2_args, image_bgr, device)
        two_stage_postproc_mask = keep_largest_component_and_close(two_stage_mask)

        variants = {
            "pretrained_baseline": baseline_mask,
            "two_stage": two_stage_mask,
            "two_stage_postproc": two_stage_postproc_mask,
        }
        for variant_name, pred_mask in variants.items():
            row = evaluate_all(pred_mask, gt_mask)
            row["variant"] = variant_name
            row["image"] = image_name
            rows.append(row)

        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(test_samples)} imagenes procesadas")

    df = pd.DataFrame(rows)
    per_image_csv = os.path.join(out_dir, "two_stage_postproc_results.csv")
    df.to_csv(per_image_csv, index=False)
    print(f"\nDetalle por imagen guardado en: {per_image_csv}")

    baseline = df[df.variant == "pretrained_baseline"].set_index("image").sort_index()
    two_stage = df[df.variant == "two_stage"].set_index("image").sort_index()
    two_stage_pp = df[df.variant == "two_stage_postproc"].set_index("image").sort_index()
    assert (baseline.index == two_stage.index).all() and (baseline.index == two_stage_pp.index).all()

    report_lines = [
        "# Enfoque en dos etapas + post-procesamiento: evaluacion con evidencia",
        "",
        f"Comparacion pareada (Wilcoxon, por imagen) sobre {len(test_samples)} imagenes "
        "de test ORIGINALES completas. Reporta media, mediana y conteo de imagenes que "
        "mejoran/empeoran/quedan igual para cada metrica -- la media sola puede quedar "
        "dominada por 1-2 outliers catastroficos (ver `analisis_resultados_dos_etapas.md` "
        "en el proyecto de Claude).",
        "",
        "## two_stage vs. pretrained_baseline",
        "",
        "| Métrica | Media base | Media var. | Mediana base | Mediana var. | mejoran/empeoran/igual | p (Wilcoxon) | Veredicto |",
        "|---|---|---|---|---|---|---|---|",
    ]
    print("\n=== two_stage vs. pretrained_baseline ===")
    for m, lower_is_better in [("dice", False), ("iou", False), ("hausdorff_px", True), ("rel_error_pct", True)]:
        summarize_metric(report_lines, m, baseline[m], two_stage[m], lower_is_better)

    report_lines += ["", "## two_stage_postproc vs. pretrained_baseline", "",
                      "| Métrica | Media base | Media var. | Mediana base | Mediana var. | mejoran/empeoran/igual | p (Wilcoxon) | Veredicto |",
                      "|---|---|---|---|---|---|---|---|"]
    print("\n=== two_stage_postproc vs. pretrained_baseline ===")
    for m, lower_is_better in [("dice", False), ("iou", False), ("hausdorff_px", True), ("rel_error_pct", True)]:
        summarize_metric(report_lines, m, baseline[m], two_stage_pp[m], lower_is_better)

    report_lines += ["", "## two_stage_postproc vs. two_stage (¿el post-procesamiento ayuda?)", "",
                      "| Métrica | Media base | Media var. | Mediana base | Mediana var. | mejoran/empeoran/igual | p (Wilcoxon) | Veredicto |",
                      "|---|---|---|---|---|---|---|---|"]
    print("\n=== two_stage_postproc vs. two_stage ===")
    for m, lower_is_better in [("dice", False), ("iou", False), ("hausdorff_px", True), ("rel_error_pct", True)]:
        summarize_metric(report_lines, m, two_stage[m], two_stage_pp[m], lower_is_better)

    # Foco especial: los 2 outliers catastroficos nuevos que motivaron esta corrida
    outlier_images = ["DSC01378.JPG", "DSC01037.JPG"]
    report_lines += ["", "## Los 2 outliers catastroficos (DSC01378, DSC01037): ¿se arreglaron?", "",
                      "| Imagen | Hausdorff pretrained | Hausdorff two_stage | Hausdorff two_stage_postproc | "
                      "Dice pretrained | Dice two_stage | Dice two_stage_postproc |",
                      "|---|---|---|---|---|---|---|"]
    print("\n=== Los 2 outliers catastroficos ===")
    for img_name in outlier_images:
        if img_name in baseline.index:
            hd_b, hd_ts, hd_pp = baseline.loc[img_name, "hausdorff_px"], two_stage.loc[img_name, "hausdorff_px"], two_stage_pp.loc[img_name, "hausdorff_px"]
            d_b, d_ts, d_pp = baseline.loc[img_name, "dice"], two_stage.loc[img_name, "dice"], two_stage_pp.loc[img_name, "dice"]
            print(f"{img_name}: hausdorff {hd_b:.1f} -> {hd_ts:.1f} -> {hd_pp:.1f} | dice {d_b:.3f} -> {d_ts:.3f} -> {d_pp:.3f}")
            report_lines.append(f"| {img_name} | {hd_b:.1f} | {hd_ts:.1f} | {hd_pp:.1f} | {d_b:.3f} | {d_ts:.3f} | {d_pp:.3f} |")

    # Las 4 imagenes de heridas chicas ya documentadas, para no perderlas de vista
    small_wound_images = ["DSC01428.JPG", "DSC01017.JPG", "DSC01995.JPG", "DSC01234.JPG"]
    report_lines += ["", "## Las 4 imagenes de heridas chicas documentadas", "",
                      "| Imagen | Dice pretrained | Dice two_stage | Dice two_stage_postproc |", "|---|---|---|---|"]
    print("\n=== Las 4 imagenes de heridas chicas ===")
    for img_name in small_wound_images:
        if img_name in baseline.index:
            d_b, d_ts, d_pp = baseline.loc[img_name, "dice"], two_stage.loc[img_name, "dice"], two_stage_pp.loc[img_name, "dice"]
            print(f"{img_name}: dice {d_b:.3f} -> {d_ts:.3f} -> {d_pp:.3f}")
            report_lines.append(f"| {img_name} | {d_b:.3f} | {d_ts:.3f} | {d_pp:.3f} |")

    report_path = os.path.join(out_dir, "two_stage_postproc_decision.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nReporte de decision guardado en: {report_path}")


if __name__ == "__main__":
    main()
