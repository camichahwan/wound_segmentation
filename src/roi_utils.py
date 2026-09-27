"""
roi_utils.py

Lógica COMPARTIDA para el enfoque en dos etapas (ver
outputs/error_analysis/heridas_chicas_resize.md): a partir de una máscara
(binaria, 0/255), calcular la región de recorte (ROI) que se le pasa a la
etapa 2.

Se usa TANTO al generar el dataset de recortes para entrenar la etapa 2
(make_crop_dataset.py, con la máscara real) COMO al hacer inferencia en dos
etapas (evaluate_two_stage.py, con la máscara predicha por la etapa 1) --
que las dos usen la MISMA fórmula es importante: si el recorte de
entrenamiento y el de inferencia tuvieran una proporción de contexto
distinta alrededor de la herida, la etapa 2 vería en producción algo
distinto de lo que aprendió en entrenamiento (distribution shift).
"""

import numpy as np

MARGIN_FRACTION = 0.6  # padding a cada lado, como fraccion del ancho/alto del bbox
MIN_CROP_SIDE = 600     # lado minimo del recorte (px de la imagen ORIGINAL) -- para
                         # que heridas muy chicas no queden con un recorte demasiado
                         # ajustado, sin nada de contexto alrededor


def mask_bbox(mask: np.ndarray):
    """(y0, y1, x0, x1) del bounding box de los pixeles > 127, o None si la
    mascara esta vacia (nada por encima del umbral)."""
    ys, xs = np.where(mask > 127)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def compute_crop_region(mask: np.ndarray, image_shape, jitter: bool = False,
                         rng: "np.random.RandomState | None" = None):
    """
    A partir de una máscara (predicha o real), calcula la región
    (y0, y1, x0, x1) a recortar de la imagen ORIGINAL (mismo tamaño que
    `mask`), con margen y, opcionalmente, aleatoriedad.

    jitter=True se usa SOLO al generar el dataset de entrenamiento de la
    etapa 2 (con la máscara real), para que la etapa 2 tolere que la etapa 1
    no sea perfecta. jitter=False (default) es el comportamiento
    determinístico que se usa en inferencia real y en val/test.

    Devuelve None si la máscara está vacía -- quien llama decide el
    fallback (por ejemplo, usar la imagen completa sin recortar).
    """
    h, w = image_shape[:2]
    bbox = mask_bbox(mask)
    if bbox is None:
        return None
    y0, y1, x0, x1 = bbox

    bbox_h, bbox_w = (y1 - y0 + 1), (x1 - x0 + 1)
    pad_y = max(bbox_h * MARGIN_FRACTION, (MIN_CROP_SIDE - bbox_h) / 2)
    pad_x = max(bbox_w * MARGIN_FRACTION, (MIN_CROP_SIDE - bbox_w) / 2)

    if jitter:
        rng = rng if rng is not None else np.random.RandomState()
        pad_y *= rng.uniform(0.75, 1.25)
        pad_x *= rng.uniform(0.75, 1.25)
        y0 = y0 + rng.uniform(-0.15, 0.15) * bbox_h
        y1 = y1 + rng.uniform(-0.15, 0.15) * bbox_h
        x0 = x0 + rng.uniform(-0.15, 0.15) * bbox_w
        x1 = x1 + rng.uniform(-0.15, 0.15) * bbox_w

    crop_y0 = int(max(0, y0 - pad_y))
    crop_y1 = int(min(h, y1 + pad_y))
    crop_x0 = int(max(0, x0 - pad_x))
    crop_x1 = int(min(w, x1 + pad_x))
    return crop_y0, crop_y1, crop_x0, crop_x1
