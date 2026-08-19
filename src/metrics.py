"""
metrics.py

Métricas de segmentación, reutilizables tanto para evaluar los modelos de
deep learning (evaluate.py) como los baselines clásicos (baselines_classical.py).
Trabajan sobre arrays de numpy (0/1 o 0/255), no sobre tensores de torch, para
poder usarlas también fuera de un contexto de entrenamiento.
"""

import numpy as np
from scipy.spatial.distance import directed_hausdorff


def _binarize(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def dice_coefficient(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred, gt = _binarize(pred), _binarize(gt)
    intersection = np.logical_and(pred, gt).sum()
    return float((2.0 * intersection + eps) / (pred.sum() + gt.sum() + eps))


def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred, gt = _binarize(pred), _binarize(gt)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((intersection + eps) / (union + eps))


def precision_recall(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7):
    pred, gt = _binarize(pred), _binarize(gt)
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return float(precision), float(recall)


def hausdorff_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Distancia de Hausdorff simétrica entre los contornos de pred y gt, en
    píxeles. A diferencia de Dice/IoU (que miden solapamiento de área), esta
    métrica es sensible a errores de BORDE puntuales aunque el área general
    coincida bien -- relevante acá porque el borde es justamente donde hay
    más desacuerdo entre anotadores humanos.

    Devuelve infinito si alguna de las dos máscaras está vacía (no hay
    contorno que medir).
    """
    pred, gt = _binarize(pred), _binarize(gt)
    pred_points = np.argwhere(pred > 0)
    gt_points = np.argwhere(gt > 0)

    if len(pred_points) == 0 or len(gt_points) == 0:
        return float("inf")

    d1 = directed_hausdorff(pred_points, gt_points)[0]
    d2 = directed_hausdorff(gt_points, pred_points)[0]
    return float(max(d1, d2))


def area_error(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Error absoluto y relativo de área en píxeles (no cm² -- para eso hace
    falta resolver el problema de escala sin marcador, ver el documento de
    ideas de tesis). Relevante porque el objetivo final del proyecto es
    reportar área de la herida, no solo Dice.
    """
    pred, gt = _binarize(pred), _binarize(gt)
    pred_area = int(pred.sum())
    gt_area = int(gt.sum())
    abs_error = abs(pred_area - gt_area)
    rel_error = abs_error / gt_area if gt_area > 0 else float("inf")
    return {
        "pred_area_px": pred_area,
        "gt_area_px": gt_area,
        "abs_error_px": abs_error,
        "rel_error_pct": rel_error * 100,
    }


def evaluate_all(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Corre todas las métricas de una vez y devuelve un dict prolijo para loggear/tabular."""
    precision, recall = precision_recall(pred, gt)
    result = {
        "dice": dice_coefficient(pred, gt),
        "iou": iou_score(pred, gt),
        "precision": precision,
        "recall": recall,
        "hausdorff_px": hausdorff_distance(pred, gt),
    }
    result.update(area_error(pred, gt))
    return result
