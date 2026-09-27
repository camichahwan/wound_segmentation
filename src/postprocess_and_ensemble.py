"""
postprocess_and_ensemble.py

Evalua si un post-procesamiento simple (cierre morfologico + quedarse con el
componente conectado mas grande) y/o un ensemble (promediar las
probabilidades de unet_pretrained + unet_scratch) mejoran los resultados
sobre el set de test, comparado contra el mejor modelo solo (unet_pretrained),
con la misma metodologia de evidencia (test pareado de Wilcoxon por imagen)
que el resto de las decisiones de este repo.

Esto NO ataca las fallas de heridas chicas/tempranas (ver
outputs/error_analysis/heridas_chicas_resize.md) -- ese es un problema de
informacion perdida en el resize a 512x512, no algo que postprocesamiento o
ensemble puedan recuperar. Apunta a las OTRAS fallas: bordes ruidosos, blobs
falsos positivos sueltos lejos de la herida real, casos donde un modelo se
equivoca y el otro no.

Uso (correr en Colab, necesita los dos checkpoints entrenados + las imagenes
de test -- no reentrena nada, solo corre inferencia con lo que ya existe):
    python postprocess_and_ensemble.py
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
from models.unet_scratch import UNetFromScratch
from models.unet_pretrained import build_pretrained_unet
from metrics import evaluate_all


def load_model(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt["args"]
    if saved_args["model"] == "scratch":
        model = UNetFromScratch(base_channels=saved_args["base_channels"])
    else:
        model = build_pretrained_unet(encoder_name=saved_args["encoder"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, saved_args


def predict_prob(model, image_bgr: np.ndarray, image_size: int, device) -> np.ndarray:
    """Igual que predict_mask() de evaluate.py, pero devuelve la probabilidad
    cruda (antes del umbral 0.5) -- hace falta para poder promediar dos
    modelos antes de binarizar, no despues."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    original_h, original_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    prob = cv2.resize(prob, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    return prob


def keep_largest_component_and_close(mask_uint8: np.ndarray, close_kernel: int = 15) -> np.ndarray:
    """Post-procesamiento barato: cierre morfologico (rellena huequitos) +
    quedarse solo con el componente conectado mas grande (elimina blobs
    falsos positivos sueltos que no tocan la region real de la herida --
    esto es lo que suele disparar el Hausdorff distance sin cambiar mucho
    el Dice)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n_labels <= 1:
        return closed  # nada detectado, no hay nada que filtrar

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(closed)
    out[labels == largest_label] = 255
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=None,
                         help="Default: misma carpeta de Drive que los checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = get_checkpoints_dir()
    out_dir = args.out_dir or ckpt_dir
    os.makedirs(out_dir, exist_ok=True)

    pretrained_model, pretrained_args = load_model(os.path.join(ckpt_dir, "unet_pretrained.pt"), device)
    scratch_model, scratch_args = load_model(os.path.join(ckpt_dir, "unet_scratch.pt"), device)

    paired, _ = list_paired_and_unpaired()
    _, _, test_samples = train_val_test_split(paired, seed=args.seed)
    print(f"Evaluando variantes sobre {len(test_samples)} imagenes de test.")

    rows = []
    for i, sample in enumerate(test_samples):
        image_bgr = cv2.imread(sample.image_path)
        gt_mask = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)
        image_name = os.path.basename(sample.image_path)

        prob_pre = predict_prob(pretrained_model, image_bgr, pretrained_args["image_size"], device)
        prob_scr = predict_prob(scratch_model, image_bgr, scratch_args["image_size"], device)
        prob_ens = (prob_pre + prob_scr) / 2.0

        mask_pre = (prob_pre > 0.5).astype(np.uint8) * 255
        mask_ens = (prob_ens > 0.5).astype(np.uint8) * 255

        variants = {
            "pretrained": mask_pre,
            "pretrained_postproc": keep_largest_component_and_close(mask_pre),
            "ensemble": mask_ens,
            "ensemble_postproc": keep_largest_component_and_close(mask_ens),
        }

        for variant_name, pred_mask in variants.items():
            row = evaluate_all(pred_mask, gt_mask)
            row["variant"] = variant_name
            row["image"] = image_name
            rows.append(row)

        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(test_samples)} imagenes procesadas")

    df = pd.DataFrame(rows)
    per_image_csv = os.path.join(out_dir, "postprocess_ensemble_results.csv")
    df.to_csv(per_image_csv, index=False)
    print(f"\nDetalle por imagen guardado en: {per_image_csv}")

    metric_cols = [c for c in df.columns if c not in ("variant", "image")]
    summary = df.groupby("variant")[metric_cols].agg(["mean", "std"])
    print("\n=== Resumen por variante ===")
    print(summary)

    baseline = df[df.variant == "pretrained"].set_index("image").sort_index()

    report_lines = [
        "# Post-procesamiento y ensemble: evaluacion con evidencia",
        "",
        f"Comparacion pareada (Wilcoxon, por imagen) contra `unet_pretrained` solo "
        f"(linea de base actual), sobre {len(test_samples)} imagenes de test. No "
        "requiere reentrenar nada, solo reutiliza los dos checkpoints ya entrenados.",
        "",
        "| Variante | Dice | Hausdorff (px) | p Dice | p Hausdorff | Veredicto |",
        "|---|---|---|---|---|---|",
    ]
    print("\n=== Test pareado Wilcoxon vs. pretrained solo (linea de base) ===")
    for variant_name in ["pretrained_postproc", "ensemble", "ensemble_postproc"]:
        v = df[df.variant == variant_name].set_index("image").sort_index()
        assert (v.index == baseline.index).all(), "las imagenes no estan alineadas"

        _, p_dice = wilcoxon(v["dice"], baseline["dice"])
        _, p_hd = wilcoxon(v["hausdorff_px"], baseline["hausdorff_px"])
        dice_mean, hd_mean = v["dice"].mean(), v["hausdorff_px"].mean()

        veredicto = []
        if p_dice < 0.05:
            veredicto.append("Dice mejora" if dice_mean > baseline["dice"].mean() else "Dice empeora")
        if p_hd < 0.05:
            veredicto.append("Hausdorff mejora" if hd_mean < baseline["hausdorff_px"].mean() else "Hausdorff empeora")
        veredicto_str = "; ".join(veredicto) if veredicto else "sin diferencia significativa"

        print(f"{variant_name:20s} dice={dice_mean:.4f} (p={p_dice:.4f}) | "
              f"hausdorff={hd_mean:.1f} (p={p_hd:.4f}) | {veredicto_str}")
        report_lines.append(
            f"| {variant_name} | {dice_mean:.4f} | {hd_mean:.1f} | {p_dice:.4f} | {p_hd:.4f} | {veredicto_str} |"
        )

    report_path = os.path.join(out_dir, "postprocess_ensemble_decision.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nReporte de decision guardado en: {report_path}")


if __name__ == "__main__":
    main()
