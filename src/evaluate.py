"""
evaluate.py

Corre uno o más checkpoints entrenados (train.py) sobre el set de TEST, y
arma la tabla comparativa que finalmente va a la tesis: Dice/IoU/Precision/
Recall/Hausdorff/error de área por modelo, más el promedio y desvío estándar
entre imágenes (no solo un número global).

Uso:
    python evaluate.py --checkpoints outputs/checkpoints/unet_scratch.pt outputs/checkpoints/unet_pretrained.pt
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
import cv2

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


def predict_mask(model, image_bgr: np.ndarray, image_size: int, device) -> np.ndarray:
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

    pred_mask = (prob > 0.5).astype(np.uint8) * 255
    pred_mask = cv2.resize(pred_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    return pred_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paired, _ = list_paired_and_unpaired()
    _, _, test_samples = train_val_test_split(paired, seed=args.seed)
    print(f"Evaluando sobre {len(test_samples)} imágenes de test.")

    all_rows = []
    for ckpt_path in args.checkpoints:
        run_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        model, saved_args = load_model(ckpt_path, device)
        image_size = saved_args["image_size"]

        for sample in test_samples:
            image_bgr = cv2.imread(sample.image_path)
            gt_mask = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)

            pred_mask = predict_mask(model, image_bgr, image_size, device)
            row = evaluate_all(pred_mask, gt_mask)
            row["model"] = run_name
            row["image"] = os.path.basename(sample.image_path)
            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    summary = df.groupby("model").agg(["mean", "std"])
    print("\n=== Resumen por modelo (promedio ± desvío estándar sobre el set de test) ===")
    print(summary)

    out_csv = args.out_csv or os.path.join(
        os.path.dirname(__file__), "..", "outputs", "evaluation_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nResultados por imagen guardados en: {out_csv}")


if __name__ == "__main__":
    main()
