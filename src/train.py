"""
train.py

Loop de entrenamiento genérico: sirve tanto para UNetFromScratch como para el
Unet preentrenado de segmentation_models_pytorch, porque ambos exponen la
misma firma (logits [B,1,H,W] a partir de una imagen [B,3,H,W]).

Uso típico (correr desde la carpeta src/):
    python train.py --model scratch --epochs 60 --image_size 512
    python train.py --model pretrained --encoder resnet34 --epochs 40 --freeze_encoder

Guarda el mejor checkpoint (según Dice de validación) en
outputs/checkpoints/<run_name>.pt y un log de métricas por época en
outputs/checkpoints/<run_name>_history.csv, para poder graficar después las
curvas de ambos modelos una al lado de la otra.
"""

import os
import csv
import argparse
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_checkpoints_dir
from dataset import list_paired_and_unpaired, train_val_test_split, WoundSegmentationDataset
from models.unet_scratch import UNetFromScratch
from models.unet_pretrained import build_pretrained_unet
from metrics import dice_coefficient, iou_score


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    intersection = (probs_flat * targets_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def combined_loss(logits, targets, bce_weight: float = 0.5):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    dsc = dice_loss(logits, targets)
    return bce_weight * bce + (1 - bce_weight) * dsc


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_dice, total_iou, n_batches = 0.0, 0.0, 0.0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = combined_loss(logits, masks)

            if train:
                loss.backward()
                optimizer.step()

            preds = (torch.sigmoid(logits) > 0.5).float()
            batch_dice = dice_coefficient(preds.cpu().numpy(), masks.cpu().numpy())
            batch_iou = iou_score(preds.cpu().numpy(), masks.cpu().numpy())

            total_loss += loss.item()
            total_dice += batch_dice
            total_iou += batch_iou
            n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "dice": total_dice / max(n_batches, 1),
        "iou": total_iou / max(n_batches, 1),
    }


def build_model(args):
    if args.model == "scratch":
        return UNetFromScratch(base_channels=args.base_channels)
    elif args.model == "pretrained":
        return build_pretrained_unet(encoder_name=args.encoder, freeze_encoder=args.freeze_encoder)
    raise ValueError(f"Modelo desconocido: {args.model}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["scratch", "pretrained"], required=True)
    parser.add_argument("--encoder", default="resnet34", help="Solo aplica a --model pretrained")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--base_channels", type=int, default=64, help="Solo aplica a --model scratch")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--init_checkpoint", default=None,
                         help="Ruta a un .pt de una corrida anterior para arrancar desde esos pesos "
                              "en vez de inicializacion aleatoria/ImageNet (fine-tuning). Util para "
                              "la etapa 2 del enfoque en dos etapas: partir de unet_pretrained.pt ya "
                              "entrenado sobre imagenes completas, y afinarlo sobre recortes.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    paired, unpaired = list_paired_and_unpaired()
    print(f"Imágenes etiquetadas: {len(paired)} | sin etiquetar: {len(unpaired)}")
    train_samples, val_samples, test_samples = train_val_test_split(paired, seed=args.seed)
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    train_ds = WoundSegmentationDataset(train_samples, image_size=args.image_size, augment=True)
    val_ds = WoundSegmentationDataset(val_samples, image_size=args.image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = build_model(args).to(device)
    if args.init_checkpoint:
        print(f"Fine-tuning: cargando pesos iniciales desde {args.init_checkpoint}")
        init_ckpt = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(init_ckpt["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    run_name = args.run_name or f"unet_{args.model}"
    ckpt_dir = get_checkpoints_dir()
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Los checkpoints y el historial de esta corrida se guardan en: {ckpt_dir}")
    ckpt_path = os.path.join(ckpt_dir, f"{run_name}.pt")
    history_path = os.path.join(ckpt_dir, f"{run_name}_history.csv")

    best_val_dice = -1.0
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_dice", "train_iou",
                          "val_loss", "val_dice", "val_iou"])

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
            val_metrics = run_epoch(model, val_loader, optimizer, device, train=False)
            scheduler.step(val_metrics["dice"])

            writer.writerow([epoch, train_metrics["loss"], train_metrics["dice"], train_metrics["iou"],
                              val_metrics["loss"], val_metrics["dice"], val_metrics["iou"]])
            f.flush()

            print(f"[{run_name}] Epoch {epoch}/{args.epochs} | "
                  f"train_dice={train_metrics['dice']:.4f} | val_dice={val_metrics['dice']:.4f}")

            if val_metrics["dice"] > best_val_dice:
                best_val_dice = val_metrics["dice"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "val_dice": best_val_dice,
                }, ckpt_path)
                print(f"  -> nuevo mejor checkpoint guardado (val_dice={best_val_dice:.4f})")

    print(f"\nEntrenamiento terminado. Mejor val_dice: {best_val_dice:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Historial: {history_path}")


if __name__ == "__main__":
    main()
