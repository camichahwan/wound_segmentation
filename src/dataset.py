"""
dataset.py

Empareja imágenes con máscaras, separa las que SÍ tienen máscara (para
entrenamiento/validación/test supervisado) de las que todavía NO la tienen
(para usar más adelante en semi-supervised / active learning), y expone un
Dataset de PyTorch listo para usar en train.py.
"""

import os
import glob
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import get_images_dir, get_masks_dir

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG")


@dataclass
class PairedSample:
    image_path: str
    mask_path: str


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _match_mask_filename(image_stem: str, masks_by_stem: dict) -> Optional[str]:
    """
    Soporta dos convenciones de nombre de máscara para el mismo stem de imagen:
        DSC00598.JPG  ->  DSC00598.png          (mismo nombre, otra extensión)
        DSC00598.JPG  ->  DSC00598_mask.png      (sufijo "_mask")
    """
    if image_stem in masks_by_stem:
        return masks_by_stem[image_stem]
    if f"{image_stem}_mask" in masks_by_stem:
        return masks_by_stem[f"{image_stem}_mask"]
    return None


def list_paired_and_unpaired(images_dir: str = None, masks_dir: str = None
                              ) -> Tuple[List[PairedSample], List[str]]:
    """
    Recorre Images/ y Masks/ y devuelve:
      - paired: lista de PairedSample (imagen CON máscara manual) -> para
        entrenamiento/validación/test supervisado.
      - unpaired_images: lista de rutas de imagen SIN máscara todavía -> para
        semi-supervised learning / active learning / pretraining, o
        simplemente porque falta anotarlas.
    """
    images_dir = images_dir or get_images_dir()
    masks_dir = masks_dir or get_masks_dir()

    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
    image_paths = sorted(set(image_paths))

    mask_paths = []
    for ext in IMAGE_EXTENSIONS:
        mask_paths.extend(glob.glob(os.path.join(masks_dir, f"*{ext}")))
    masks_by_stem = {_stem(p): p for p in mask_paths}

    paired, unpaired = [], []
    for img_path in image_paths:
        stem = _stem(img_path)
        mask_path = _match_mask_filename(stem, masks_by_stem)
        if mask_path is not None:
            paired.append(PairedSample(image_path=img_path, mask_path=mask_path))
        else:
            unpaired.append(img_path)

    return paired, unpaired


DEFAULT_SPLIT_MANIFEST = os.path.join(os.path.dirname(__file__), "..", "splits", "split_v1.csv")


def _split_from_manifest(paired: List[PairedSample], manifest_path: str):
    """
    Split CONGELADO (splits/split_v1.csv, versionado en git): cada imagen ya
    anotada cuando se congeló conserva SIEMPRE su asignacion train/val/test,
    aunque despues se sumen mas imagenes anotadas. Las imagenes nuevas (que
    no estan en el manifiesto) van a train -- asi el set de test queda fijo y
    ningun modelo entrenado antes ve en train imagenes que hoy estarian en
    test. Sin esto, cada vez que se agregan mascaras el shuffle cambia y se
    contamina la evaluacion.
    """
    import csv
    with open(manifest_path, newline="") as f:
        assignment = {row["image"]: row["split"] for row in csv.DictReader(f)}
    train, val, test, n_new = [], [], [], 0
    for sample in paired:
        split = assignment.get(os.path.basename(sample.image_path))
        if split is None:
            n_new += 1
            split = "train"
        {"train": train, "val": val, "test": test}[split].append(sample)
    if n_new:
        print(f"[split] {n_new} imagenes anotadas despues de congelar el split -> se usan en TRAIN "
              f"(val/test no cambian).")
    return train, val, test


def train_val_test_split(paired: List[PairedSample], val_frac: float = 0.15,
                          test_frac: float = 0.15, seed: int = 42,
                          manifest_path: Optional[str] = "default"
                          ) -> Tuple[List[PairedSample], List[PairedSample], List[PairedSample]]:
    """
    Si existe el manifiesto congelado (splits/split_v1.csv) se usa ese split
    (ver _split_from_manifest) y val_frac/test_frac/seed se ignoran. Si no
    existe, o se pasa manifest_path=None, cae al split aleatorio por seed.

    OJO: si varias fotos del dataset corresponden al mismo animal en
    distintos días (seguimiento longitudinal), lo correcto es separar por
    ANIMAL, no por imagen suelta, para no filtrar información del mismo
    sujeto entre train y test. Este split por imagen es un punto de partida
    razonable mientras se define/confirma esa estructura; conviene revisarlo
    antes de reportar resultados finales.
    """
    if manifest_path == "default":
        manifest_path = DEFAULT_SPLIT_MANIFEST
    if manifest_path and os.path.exists(manifest_path):
        return _split_from_manifest(paired, manifest_path)

    rng = random.Random(seed)
    shuffled = paired.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)

    val = shuffled[:n_val]
    test = shuffled[n_val:n_val + n_test]
    train = shuffled[n_val + n_test:]
    return train, val, test


try:
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:
    # Permite importar este módulo (y usar list_paired_and_unpaired /
    # train_val_test_split) en un entorno sin PyTorch instalado. Si de verdad
    # se intenta instanciar WoundSegmentationDataset sin torch, va a fallar
    # ahí, con un error mucho más claro que un ImportError al importar el
    # módulo entero.
    _TorchDataset = object


class WoundSegmentationDataset(_TorchDataset):
    """Dataset de PyTorch para segmentación binaria herida/no-herida."""

    def __init__(self, samples: List[PairedSample], image_size: int = 512,
                 augment: bool = False):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment
        self.transform = self._build_transform()

    def _build_transform(self):
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        if self.augment:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
                A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=25, val_shift_limit=15, p=0.4),
                A.GaussNoise(p=0.15),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(sample.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        transformed = self.transform(image=img, mask=mask)
        image_t = transformed["image"]
        mask_t = transformed["mask"].unsqueeze(0).float()
        return image_t, mask_t


if __name__ == "__main__":
    paired, unpaired = list_paired_and_unpaired()
    print(f"Imágenes con máscara (usables para entrenar): {len(paired)}")
    print(f"Imágenes SIN máscara todavía: {len(unpaired)}")
    train, val, test = train_val_test_split(paired)
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
